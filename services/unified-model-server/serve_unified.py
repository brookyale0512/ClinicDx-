#!/usr/bin/env python3
"""Unified model server: CDS text + Scribe audio through one Gemma3WithAudioModel.

Replaces vLLM with a custom server that loads the full model once:
  - MedASR encoder (105M, frozen)
  - AudioProjector (11.8M, trained)
  - CDS MedGemma (4.3B, frozen)

Endpoints:
  POST /v1/completions        — OpenAI-compatible text completions (CDS)
  POST /v1/audio/extract      — Audio → observations (Scribe)
  GET  /health                — Health check
  GET  /v1/models             — List models (OpenAI compat)
"""

import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling.gemma3_audio import Gemma3WithAudioModel
from transformers import AutoTokenizer, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("unified")

MODEL_DIR = Path(__file__).resolve().parent / "model" / "medgemma_cds_think_v1"
MEDASR_DIR = Path("/var/www/ClinicDx/model/medASR")
PROJECTOR_CKPT = Path(__file__).resolve().parent / "checkpoints_cds" / "projector_final.pt"
SAMPLE_RATE = 16000

model: Optional[Gemma3WithAudioModel] = None
tokenizer: Optional[AutoTokenizer] = None
feature_extractor = None
model_name: str = ""


# ── Pydantic models (OpenAI-compatible) ─────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    stream: bool = False

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]

class AudioExtractRequest(BaseModel):
    manifest: str = ""

class AudioExtractResponse(BaseModel):
    observations: list[dict]
    raw_output: str
    duration_ms: float
    encoder_shape: list[int]
    projected_shape: list[int]


# ── Model loading ───────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer, feature_extractor, model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = str(MODEL_DIR)

    log.info("Loading unified model from %s", MODEL_DIR)
    model = Gemma3WithAudioModel(
        base_model_path=str(MODEL_DIR),
        audio_encoder_path=str(MEDASR_DIR),
        torch_dtype=torch.bfloat16,
    )
    model.load_base_model()
    model.load_audio_encoder()

    llm_device = next(model._base_model.parameters()).device
    model.audio_projector = model.audio_projector.to(llm_device)

    if PROJECTOR_CKPT.exists():
        model.load_projector_checkpoint(str(PROJECTOR_CKPT))
        log.info("AudioProjector loaded from %s", PROJECTOR_CKPT)
    else:
        log.warning("No projector checkpoint at %s — using random weights", PROJECTOR_CKPT)

    model.audio_projector.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    asr_proc = AutoProcessor.from_pretrained(str(MEDASR_DIR), trust_remote_code=True)
    feature_extractor = asr_proc.feature_extractor

    total_params = (
        sum(p.numel() for p in model._base_model.parameters())
        + sum(p.numel() for p in model._audio_encoder.parameters())
        + sum(p.numel() for p in model.audio_projector.parameters())
    )
    log.info("Unified model loaded: %.1fB total params on %s", total_params / 1e9, llm_device)


# ── Audio processing helpers ────────────────────────────────────────────────

def load_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        audio_np, sr = _convert_with_ffmpeg(audio_bytes)

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio_np = resample_poly(audio_np, SAMPLE_RATE, sr).astype(np.float32)
    return audio_np, SAMPLE_RATE


def _convert_with_ffmpeg(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert any audio format (webm, opus, mp3, etc.) to WAV via ffmpeg."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-f", "wav", tmp_out_path],
            capture_output=True, timeout=30, check=True,
        )
        audio_np, sr = sf.read(tmp_out_path, dtype="float32")
        return audio_np, sr
    finally:
        import os
        os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)


def extract_fbank(waveform: np.ndarray, device: str) -> torch.Tensor:
    waveform_t = torch.from_numpy(waveform).to(device)
    feats = feature_extractor._torch_extract_fbank_features(waveform_t, device=device)
    return feats.unsqueeze(0).float()


def parse_observations(text: str) -> list[dict]:
    import re as _re

    # Strip <think>...</think> blocks
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    # Also strip unclosed <think> block (model cut off)
    text = _re.sub(r"<think>.*", "", text, flags=_re.DOTALL)

    SKIP_PREFIXES = (
        "i need", "i will", "let me", "the patient", "observations",
        "concept", "extract", "here", "based on", "from the",
        "audio", "manifest", "note", "output", "result",
    )

    obs = []
    for line in text.strip().splitlines():
        line = line.strip().lstrip("- *")
        if not line or ":" not in line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower().replace(" ", "_"), value.strip()
        if not key or not value:
            continue
        if any(key.startswith(p) for p in SKIP_PREFIXES):
            continue
        if len(key) > 50 or len(key.split("_")) > 5:
            continue
        obs.append({"label": key, "value": value})
    return obs


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Unified CDS+Scribe Model Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok" if model is not None else "loading"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "clinicdx",
            "root": model_name,
            "parent": None,
            "max_model_len": 8192,
            "permission": [{
                "id": f"modelperm-{uuid.uuid4().hex[:16]}",
                "object": "model_permission",
                "created": int(time.time()),
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False,
            }],
        }],
    }


# ── Text completions (OpenAI-compatible, for CDS) ──────────────────────────

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    device = next(model._base_model.parameters()).device
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

    stop_strings = request.stop or []
    stop_token_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])

    gen_kwargs = {
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        gen_kwargs["temperature"] = request.temperature
        gen_kwargs["top_p"] = request.top_p

    if request.stream:
        return _stream_completions(input_ids, gen_kwargs, stop_strings, request)

    with torch.inference_mode():
        output_ids = model._base_model.generate(input_ids=input_ids, **gen_kwargs)

    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    for s in stop_strings:
        idx = generated_text.find(s)
        if idx != -1:
            generated_text = generated_text[:idx]
            break

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[CompletionChoice(text=generated_text, finish_reason="stop")],
    )


def _stream_completions(input_ids, gen_kwargs, stop_strings, request):
    from transformers import TextIteratorStreamer
    from threading import Thread

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["streamer"] = streamer

    thread = Thread(target=model._base_model.generate, kwargs=gen_kwargs)
    thread.start()

    async def event_stream():
        accumulated = ""
        stopped = False
        for text_chunk in streamer:
            if stopped:
                break
            accumulated += text_chunk
            for s in stop_strings:
                idx = accumulated.find(s)
                if idx != -1:
                    remaining = text_chunk[:len(text_chunk) - (len(accumulated) - idx)]
                    if remaining:
                        evt = {
                            "choices": [{"text": remaining, "index": 0, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(evt)}\n\n"
                    stopped = True
                    break
            if not stopped:
                evt = {"choices": [{"text": text_chunk, "index": 0, "finish_reason": None}]}
                yield f"data: {json.dumps(evt)}\n\n"

        final = {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
        thread.join()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Audio extraction (Scribe) ──────────────────────────────────────────────

SCRIBE_SYSTEM = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "Audio embeddings from a clinical recording are provided between "
    "<start_of_audio> and <end_of_audio> markers.\n"
    "Extract structured medical observations from the audio.\n"
    "Return ONLY key: value lines matching concepts from the manifest."
)

@app.post("/v1/audio/extract", response_model=AudioExtractResponse)
async def audio_extract(
    audio: UploadFile = File(...),
    manifest: str = Form(""),
):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    device = next(model._base_model.parameters()).device

    audio_np, sr = load_audio_bytes(audio_bytes)
    input_features = extract_fbank(audio_np, str(device))

    with torch.inference_mode():
        encoder_out = model._audio_encoder(input_features=input_features)
        enc_embs = (
            encoder_out.last_hidden_state
            if hasattr(encoder_out, "last_hidden_state")
            else encoder_out[0]
        )

    with torch.inference_mode():
        projected = model.audio_projector(enc_embs.float().to(device))

    prompt_text = SCRIBE_SYSTEM + "\n" + manifest + "\n\nOUTPUT:\n"
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)

    embed_layer = model._base_model.get_input_embeddings()
    with torch.inference_mode():
        prompt_embeds = embed_layer(prompt_ids)

    projected_cast = projected.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    inputs_embeds = torch.cat([prompt_embeds, projected_cast], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    with torch.inference_mode():
        output_ids = model._base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    observations = parse_observations(generated)
    duration_ms = (time.time() - t0) * 1000

    return AudioExtractResponse(
        observations=observations,
        raw_output=generated,
        duration_ms=round(duration_ms, 1),
        encoder_shape=list(enc_embs.shape),
        projected_shape=list(projected.shape),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--projector", default=str(PROJECTOR_CKPT))
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)
    PROJECTOR_CKPT = Path(args.projector)
    model_name = str(MODEL_DIR)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
