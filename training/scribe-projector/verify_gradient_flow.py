#!/usr/bin/env python3
"""Verify gradient flow before first training run.

Checks:
  1. projector.audio_start.grad is non-None after loss.backward()
  2. projector.audio_end.grad is non-None after loss.backward()
  3. All projector.proj.* params have non-None gradients
  4. ALL MedGemma parameters have grad=None (frozen, no leakage)
  5. ALL MedASR encoder parameters have grad=None (frozen)

Usage:
    python scripts/verify_gradient_flow.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

SYSTEM_PROMPT = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "Audio embeddings from a clinical recording are provided between "
    "<audio_start> and <audio_end> markers.\n"
    "Extract structured medical observations from the audio.\n"
    "Return ONLY key: value lines matching concepts from the manifest.\n\n"
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Load components ───────────────────────────────────────────────────────
    from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading MedASR encoder...")
    encoder_model = AutoModel.from_pretrained(
        "/var/www/ClinicDx/model/medASR", torch_dtype=torch.float32
    ).to(device)
    encoder_model.eval()
    encoder = encoder_model.encoder if hasattr(encoder_model, "encoder") else encoder_model
    for p in encoder.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained("/var/www/ClinicDx/model/medASR")

    logger.info("Loading MedGemma (full precision)...")
    tokenizer = AutoTokenizer.from_pretrained("/var/www/ClinicDx/model/medgemma")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "/var/www/ClinicDx/model/medgemma",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    embed_layer = llm.get_input_embeddings()
    logger.info("Embed layer: %s", type(embed_layer).__name__)

    logger.info("Loading AudioProjector...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from service.projector import AudioProjector
    projector = AudioProjector(encoder_dim=512, llm_dim=2560, stack_factor=4).to(device)
    logger.info("Projector params: %d", projector.param_count())

    # ── Synthetic forward pass ────────────────────────────────────────────────
    print("[1/4] Models loaded. Running synthetic forward pass...", flush=True)

    audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.01
    manifest = "CONCEPTS:\n[dx] malaria\nnausea (present/absent)\n"
    target = "malaria: confirmed"

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        enc_out = encoder(input_features=input_features)
        enc_embs = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]

    projected = projector(enc_embs)
    print(f"[2/4] Projected shape: {projected.shape} (expect [1, T/4+2, 2560])", flush=True)

    prompt_text = SYSTEM_PROMPT + manifest + "\n\nOUTPUT:\n"
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(llm.device)
    target_ids = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False).to(llm.device)
    eos_id = torch.tensor([[tokenizer.eos_token_id]], device=llm.device)
    target_ids = torch.cat([target_ids, eos_id], dim=1)

    with torch.no_grad():
        prompt_embeds = embed_layer(prompt_ids)
        target_embeds = embed_layer(target_ids)

    projected_cast = projected.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    inputs_embeds = torch.cat([prompt_embeds, projected_cast, target_embeds], dim=1)

    prompt_len = prompt_embeds.shape[1]
    audio_len = projected_cast.shape[1]
    target_len = target_ids.shape[1]

    labels = torch.full((1, prompt_len + audio_len + target_len), -100, dtype=torch.long, device=llm.device)
    labels[0, prompt_len + audio_len:] = target_ids[0]

    print(f"[3/4] Sequence: prompt={prompt_len} + audio={audio_len} + target={target_len} = {inputs_embeds.shape[1]} tokens", flush=True)

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

    outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    print(f"[4/4] Loss: {loss.item():.4f} — running backward...", flush=True)

    loss.backward()

    # ── Check gradients ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  GRADIENT FLOW VERIFICATION")
    logger.info("=" * 60)

    checks = []

    # Projector audio_start
    g = projector.audio_start.grad
    checks.append(("projector.audio_start.grad", g is not None, f"norm={g.norm().item():.6f}" if g is not None else "None"))

    # Projector audio_end
    g = projector.audio_end.grad
    checks.append(("projector.audio_end.grad", g is not None, f"norm={g.norm().item():.6f}" if g is not None else "None"))

    # Projector MLP layers
    for name, param in projector.proj.named_parameters():
        g = param.grad
        checks.append((f"projector.proj.{name}.grad", g is not None, f"norm={g.norm().item():.6f}" if g is not None else "None"))

    # MedGemma — ALL should be None
    llm_grad_count = 0
    llm_param_count = 0
    for name, param in llm.named_parameters():
        llm_param_count += 1
        if param.grad is not None:
            llm_grad_count += 1

    checks.append((f"MedGemma params with grad ({llm_param_count} total)", llm_grad_count == 0, f"{llm_grad_count} have grad"))

    # Vision encoder — ALL should be None and frozen
    vision_grad_count = 0
    vision_param_count = 0
    vision_tower = getattr(llm, "vision_tower", None) or getattr(getattr(llm, "model", None), "vision_tower", None)
    if vision_tower is not None:
        for name, param in vision_tower.named_parameters():
            vision_param_count += 1
            if param.requires_grad:
                vision_grad_count += 1
    checks.append((f"Vision encoder frozen ({vision_param_count} params)", vision_grad_count == 0,
                    f"{vision_grad_count} trainable" if vision_grad_count else "all frozen"))

    # MedASR Encoder — ALL should be None
    enc_grad_count = 0
    enc_param_count = 0
    for name, param in encoder.named_parameters():
        enc_param_count += 1
        if param.grad is not None:
            enc_grad_count += 1

    checks.append((f"MedASR encoder params with grad ({enc_param_count} total)", enc_grad_count == 0, f"{enc_grad_count} have grad"))

    # Print results to stdout (not logger) so they're visible through progress bar noise
    all_ok = True
    print("\n" + "=" * 60, flush=True)
    print("  GRADIENT FLOW VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    for desc, ok, detail in checks:
        status = "✓" if ok else "✗"
        if not ok: all_ok = False
        print(f"  {status} {desc} — {detail}", flush=True)

    print(flush=True)
    if all_ok:
        print("  ✓ ALL CHECKS PASS — gradient flow is correct", flush=True)
        print("  Ready for Phase 2 training.", flush=True)
    else:
        print("  ✗ GRADIENT FLOW ISSUES DETECTED", flush=True)
        print("  DO NOT proceed with training until fixed.", flush=True)
    print("=" * 60, flush=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
