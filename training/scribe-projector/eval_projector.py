#!/usr/bin/env python3
"""Evaluate AudioProjector checkpoints on a fixed validation set (CPU-only).

Loads MedASR encoder, AudioProjector, and MedGemma entirely on CPU so it can
run in a separate terminal without disturbing GPU training.

Usage:
    python eval_projector.py \
        --checkpoint /path/to/projector_stepN.pt \
        --eval-set /var/www/ClinicDx/eval/fixed_val_set.jsonl \
        --output /var/www/ClinicDx/eval/results_stepN.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

SYSTEM_PROMPT = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "Audio embeddings from a clinical recording are provided between "
    "<audio_start> and <audio_end> markers.\n"
    "Extract structured medical observations from the audio.\n"
    "Return ONLY key: value lines matching concepts from the manifest.\n\n"
)


def parse_output(text: str) -> set:
    lines = set()
    for line in text.strip().splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().lower()
            if key and value:
                lines.add((key, value))
    return lines


def score(predicted: str, expected: str) -> dict:
    pred_lines = parse_output(predicted)
    exp_lines = parse_output(expected)

    exact_match = 1.0 if pred_lines == exp_lines else 0.0

    pred_keys = {k for k, v in pred_lines}
    exp_keys = {k for k, v in exp_lines}

    if not exp_lines:
        return {"exact_match": exact_match, "line_f1": 0.0,
                "key_precision": 0.0, "key_recall": 0.0}

    tp = len(pred_lines & exp_lines)
    precision = tp / len(pred_lines) if pred_lines else 0.0
    recall = tp / len(exp_lines) if exp_lines else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    key_tp = len(pred_keys & exp_keys)
    key_prec = key_tp / len(pred_keys) if pred_keys else 0.0
    key_rec = key_tp / len(exp_keys) if exp_keys else 0.0

    return {
        "exact_match": exact_match,
        "line_f1": f1,
        "line_precision": precision,
        "line_recall": recall,
        "key_precision": key_prec,
        "key_recall": key_rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioProjector checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to projector .pt file")
    parser.add_argument("--eval-set", required=True, help="Path to fixed_val_set.jsonl")
    parser.add_argument("--encoder-path", default="/var/www/ClinicDx/model/medASR")
    parser.add_argument("--medgemma-path", default="/var/www/ClinicDx/model/medgemma_sft")
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Running evaluation on GPU")
    else:
        device = torch.device("cpu")
        logger.info("Running evaluation on CPU")
    logger.info("Checkpoint: %s", args.checkpoint)

    # ── Load encoder ──────────────────────────────────────────────────────────
    logger.info("Loading MedASR encoder from %s ...", args.encoder_path)
    from transformers import AutoModel, AutoProcessor
    encoder_model = AutoModel.from_pretrained(
        args.encoder_path, torch_dtype=torch.float32,
    ).to(device)
    encoder_model.eval()
    encoder = encoder_model.encoder if hasattr(encoder_model, "encoder") else encoder_model
    processor = AutoProcessor.from_pretrained(args.encoder_path)
    logger.info("Encoder loaded")

    # ── Load MedGemma ─────────────────────────────────────────────────────────
    logger.info("Loading MedGemma from %s (CPU, bfloat16) ...", args.medgemma_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.medgemma_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.medgemma_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else {"": "cpu"},
    )
    llm.eval()
    embed_layer = llm.get_input_embeddings()
    logger.info("MedGemma loaded on CPU")

    # ── Load projector ────────────────────────────────────────────────────────
    logger.info("Loading AudioProjector ...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from service.projector import AudioProjector

    projector = AudioProjector(encoder_dim=512, llm_dim=2560, stack_factor=4).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "projector_state_dict" in state:
        projector.load_state_dict(state["projector_state_dict"])
    else:
        projector.load_state_dict(state)
    projector.eval()
    logger.info("Projector loaded (%d params)", projector.param_count())

    # ── Load eval set ─────────────────────────────────────────────────────────
    eval_clips = []
    with open(args.eval_set) as f:
        for line in f:
            if line.strip():
                eval_clips.append(json.loads(line))
    logger.info("Eval set: %d clips", len(eval_clips))

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []
    cat_metrics = {}
    total_time = 0.0

    for i, clip in enumerate(eval_clips):
        clip_id = clip["clip_id"]
        voice = clip["voice"]
        wav_path = clip["wav"]
        expected = clip["output"]
        manifest = clip["manifest"]
        category = clip["category"]

        logger.info("[%d/%d] %s_%s (cat=%s) ...", i + 1, len(eval_clips), clip_id, voice, category)

        t0 = time.time()

        # 1. Load + encode audio
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)

        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs["input_features"].to(device)

        with torch.no_grad():
            enc_out = encoder(input_features=input_features)
            enc_embs = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]

        # 2. Project
        with torch.no_grad():
            projected = projector(enc_embs)

        # 3. Build prompt
        prompt_text = SYSTEM_PROMPT + manifest + "\n\nOUTPUT:\n"
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)

        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids)

        projected_cast = projected.to(dtype=prompt_embeds.dtype, device=device)
        inputs_embeds = torch.cat([prompt_embeds, projected_cast], dim=1)

        # 4. Generate
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        predicted = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        elapsed = time.time() - t0
        total_time += elapsed

        # 5. Score
        metrics = score(predicted, expected)

        result = {
            "clip_id": clip_id,
            "voice": voice,
            "category": category,
            "phrase_count": clip["phrase_count"],
            "expected": expected,
            "predicted": predicted,
            "elapsed_s": round(elapsed, 1),
            **metrics,
        }
        results.append(result)

        if category not in cat_metrics:
            cat_metrics[category] = []
        cat_metrics[category].append(metrics)

        em_str = "EXACT" if metrics["exact_match"] else "MISS"
        logger.info(
            "  [%s] F1=%.2f | expected='%s' | predicted='%s' | %.1fs",
            em_str, metrics["line_f1"],
            expected[:60], predicted[:60], elapsed,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS: %s", args.checkpoint)
    logger.info("=" * 70)

    for cat in sorted(cat_metrics.keys()):
        ms = cat_metrics[cat]
        n = len(ms)
        em = sum(m["exact_match"] for m in ms) / n
        f1 = sum(m["line_f1"] for m in ms) / n
        kr = sum(m["key_recall"] for m in ms) / n
        logger.info(
            "  %s (%d clips): Exact=%.0f%% | F1=%.0f%% | KeyRecall=%.0f%%",
            cat, n, em * 100, f1 * 100, kr * 100,
        )

    all_ms = [m for ms in cat_metrics.values() for m in ms]
    n = len(all_ms)
    if n > 0:
        em = sum(m["exact_match"] for m in all_ms) / n
        f1 = sum(m["line_f1"] for m in all_ms) / n
        kr = sum(m["key_recall"] for m in all_ms) / n
        logger.info("  OVERALL (%d clips): Exact=%.0f%% | F1=%.0f%% | KeyRecall=%.0f%%", n, em * 100, f1 * 100, kr * 100)

    logger.info("Total inference time: %.1fs (%.1fs/clip avg)", total_time, total_time / max(n, 1))
    logger.info("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output
    if not output_path:
        ckpt_name = Path(args.checkpoint).stem
        output_path = f"/var/www/ClinicDx/eval/results_{ckpt_name}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "num_clips": n,
            "total_time_s": round(total_time, 1),
            "overall": {
                "exact_match": round(em, 4),
                "line_f1": round(f1, 4),
                "key_recall": round(kr, 4),
            },
            "per_category": {
                cat: {
                    "n": len(ms),
                    "exact_match": round(sum(m["exact_match"] for m in ms) / len(ms), 4),
                    "line_f1": round(sum(m["line_f1"] for m in ms) / len(ms), 4),
                    "key_recall": round(sum(m["key_recall"] for m in ms) / len(ms), 4),
                }
                for cat, ms in cat_metrics.items()
            },
            "samples": results,
        }, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
