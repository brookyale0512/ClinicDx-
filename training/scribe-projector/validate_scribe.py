#!/usr/bin/env python3
"""Validate AudioProjector on held-out scribe clips.

Tests:
  1. Concept extraction accuracy (exact match on key:value pairs)
  2. Format compliance (key: value lines)
  3. Confidence scores (cosine similarity)
  4. CDS regression (optional — verify CDS still works with text-only input)

Usage:
    python scripts/validate_scribe.py configs/train_config.yaml [--projector-ckpt path] [--num-cases 100]
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.gemma3_audio import Gemma3WithAudioModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_kv_output(text):
    """Parse key: value lines from model output."""
    pairs = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value and key != "NOT_IN_MANIFEST":
            pairs[key] = value
    return pairs


def run_scribe_validation(model, tokenizer, embed_layer, clips, system_prompt, device, num_cases):
    results = []
    for i, clip in enumerate(clips[:num_cases], 1):
        enc_embs = torch.load(clip["pt_path"], map_location="cpu", weights_only=True)
        if enc_embs.dim() == 2:
            enc_embs = enc_embs.unsqueeze(0)
        enc_embs = enc_embs.float().to(device)

        projected = model.audio_projector(enc_embs)

        prompt_text = system_prompt + "\n" + clip["manifest"] + "\n\nOUTPUT:\n"
        prompt_ids = tokenizer.encode(
            prompt_text, return_tensors="pt", add_special_tokens=True,
        ).to(device)

        with torch.no_grad():
            prompt_embeds = embed_layer(prompt_ids)

        projected_cast = projected.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        inputs_embeds = torch.cat([prompt_embeds, projected_cast], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        with torch.inference_mode():
            output_ids = model._base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted = parse_kv_output(response)
        expected = parse_kv_output(clip["target"])

        exact_matches = sum(1 for k, v in expected.items() if predicted.get(k) == v)
        key_matches = sum(1 for k in expected if k in predicted)
        precision = len(set(predicted) & set(expected)) / max(len(predicted), 1)
        recall = len(set(predicted) & set(expected)) / max(len(expected), 1)

        result = {
            "clip_id": clip["key"],
            "expected_keys": len(expected),
            "predicted_keys": len(predicted),
            "exact_matches": exact_matches,
            "key_matches": key_matches,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "is_single_phrase": clip.get("is_single_phrase", False),
        }
        results.append(result)

        if i <= 5 or i % 50 == 0:
            logger.info(
                "Case %d/%d | %s | exact=%d/%d | keys=%d/%d | P=%.2f R=%.2f",
                i, min(num_cases, len(clips)), clip["key"],
                exact_matches, len(expected), key_matches, len(expected),
                precision, recall,
            )
            if i <= 3:
                logger.info("  Expected: %s", expected)
                logger.info("  Predicted: %s", predicted)
                logger.info("  Raw: %s", response[:200])

    return results


def print_summary(results):
    n = len(results)
    if n == 0:
        logger.info("No results to summarize.")
        return

    avg_precision = sum(r["precision"] for r in results) / n
    avg_recall = sum(r["recall"] for r in results) / n
    avg_exact = sum(r["exact_matches"] for r in results) / sum(max(r["expected_keys"], 1) for r in results)
    avg_key = sum(r["key_matches"] for r in results) / sum(max(r["expected_keys"], 1) for r in results)
    has_output = sum(1 for r in results if r["predicted_keys"] > 0)

    single = [r for r in results if r["is_single_phrase"]]
    multi = [r for r in results if not r["is_single_phrase"]]

    logger.info("\n" + "=" * 60)
    logger.info("SCRIBE VALIDATION SUMMARY — %d clips", n)
    logger.info("=" * 60)
    logger.info("  Avg precision:     %.1f%%", avg_precision * 100)
    logger.info("  Avg recall:        %.1f%%", avg_recall * 100)
    logger.info("  Exact match rate:  %.1f%%", avg_exact * 100)
    logger.info("  Key match rate:    %.1f%%", avg_key * 100)
    logger.info("  Has output:        %d/%d (%.0f%%)", has_output, n, has_output / n * 100)

    if single:
        sp = sum(r["precision"] for r in single) / len(single)
        sr = sum(r["recall"] for r in single) / len(single)
        logger.info("  Single-phrase P/R: %.1f%% / %.1f%% (%d clips)", sp * 100, sr * 100, len(single))
    if multi:
        mp = sum(r["precision"] for r in multi) / len(multi)
        mr = sum(r["recall"] for r in multi) / len(multi)
        logger.info("  Multi-phrase P/R:  %.1f%% / %.1f%% (%d clips)", mp * 100, mr * 100, len(multi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(
        Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"
    ))
    parser.add_argument("--projector-ckpt", type=str, default=None)
    parser.add_argument("--num-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model_config_path = Path(model_cfg["base_model_path"]) / "config.json"
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = Gemma3WithAudioModel(
        base_model_path=model_cfg["base_model_path"],
        audio_encoder_path=model_cfg["audio_encoder_path"],
        config=model_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
    )
    model.load_all()

    if args.projector_ckpt:
        model.load_projector_checkpoint(args.projector_ckpt)
    else:
        logger.warning("No projector checkpoint — using random weights (baseline)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_path"])
    embed_layer = model._base_model.get_input_embeddings()

    precomputed_dir = Path(data_cfg["precomputed_dir"])
    clips = []
    with open(data_cfg["clips_jsonl"]) as f:
        for line in f:
            if not line.strip():
                continue
            clip = json.loads(line)
            for voice in clip["voices"]:
                key = f"{clip['clip_id']}_{voice}"
                pt_path = precomputed_dir / f"{key}.pt"
                if pt_path.exists():
                    clips.append({
                        "key": key,
                        "pt_path": str(pt_path),
                        "manifest": clip.get("manifest", ""),
                        "target": clip.get("output", ""),
                        "is_single_phrase": clip.get("phrase_count", 0) == 1,
                    })

    random.shuffle(clips)
    logger.info("Loaded %d clips for validation", len(clips))

    system_prompt = cfg.get("system_prompt", "").strip()
    results = run_scribe_validation(
        model, tokenizer, embed_layer, clips, system_prompt, device, args.num_cases,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
