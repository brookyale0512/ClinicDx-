#!/usr/bin/env python3
"""
Build text SFT training data for Phase 1 LoRA fine-tuning.

Converts scribe_training.jsonl (phrase pairs) into Gemma 3 chat-formatted
JSONL ready for SFTTrainer. Each sample gets a randomized manifest built
from the SAME concept pool used in audio clip generation (clips.jsonl),
ensuring Phase 1 and Phase 2 manifests are consistent.

Format per sample:
  <bos><start_of_turn>user
  {system_prompt}
  {manifest}

  PHRASE: "{phrase}"

  OUTPUT:<end_of_turn>
  <start_of_turn>model
  {key: value}<end_of_turn>

Usage:
    python scripts/build_text_sft_data.py
    python scripts/build_text_sft_data.py --dry-run-samples 1000
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "A clinical phrase is provided below along with a concept manifest.\n"
    "Extract structured medical observations from the phrase.\n"
    "Return ONLY key: value lines matching concepts from the manifest.\n"
)

BOS = "<bos>"
START_TURN = "<start_of_turn>"
END_TURN = "<end_of_turn>"

MANIFEST_SIZE = (25, 45)

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_manifest_pool(clips_jsonl: str) -> list:
    """Extract the manifest line pool from clips.jsonl (same as audio training)."""
    log(f"Building manifest pool from {clips_jsonl} ...")
    lines = set()
    with open(clips_jsonl) as f:
        for line in f:
            if not line.strip(): continue
            clip = json.loads(line)
            for ml in clip.get("manifest", "").split("\n"):
                ml = ml.strip()
                if ml and ml != "CONCEPTS:":
                    lines.add(ml)
    pool = list(lines)
    log(f"  Pool size: {len(pool):,} unique manifest lines")
    return pool


def format_sample(phrase: str, output: str, manifest_line: str,
                   manifest_pool: list, rng: random.Random) -> str:
    """Format one sample as Gemma 3 chat template."""
    target_lines = {manifest_line} if manifest_line else set()
    filler_count = rng.randint(*MANIFEST_SIZE) - len(target_lines)
    fillers = rng.sample(manifest_pool, k=min(filler_count, len(manifest_pool)))
    fillers = [ml for ml in fillers if ml not in target_lines]

    manifest_list = list(target_lines) + fillers
    rng.shuffle(manifest_list)
    manifest_str = "CONCEPTS:\n" + "\n".join(manifest_list)

    user_content = (
        f"{SYSTEM_PROMPT}\n"
        f"{manifest_str}\n\n"
        f"PHRASE: \"{phrase}\"\n\n"
        f"OUTPUT:"
    )

    text = (
        f"{BOS}{START_TURN}user\n"
        f"{user_content}{END_TURN}\n"
        f"{START_TURN}model\n"
        f"{output}{END_TURN}"
    )
    return text


def main():
    parser = argparse.ArgumentParser(description="Build text SFT data")
    parser.add_argument("--input", type=str,
                        default="/var/www/ClinicDx/dataset/CEIL_speech_dataset/training_pairs/scribe_training.jsonl")
    parser.add_argument("--clips-jsonl", type=str,
                        default="/var/www/ClinicDx/dataset/CEIL_speech_dataset/audio/clips.jsonl",
                        help="clips.jsonl to extract manifest pool (same as audio training)")
    parser.add_argument("--output-dir", type=str,
                        default="/var/www/ClinicDx/training")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--dry-run-samples", type=int, default=0,
                        help="Also output a small dry-run subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  Build Text SFT Data — Phase 1")
    log("=" * 60)

    # Build manifest pool from clips.jsonl (same pool as audio training)
    manifest_pool = build_manifest_pool(args.clips_jsonl)

    # Load training pairs
    log(f"Loading training pairs from {args.input} ...")
    pairs = []
    with open(args.input) as f:
        for line in f:
            if not line.strip(): continue
            pairs.append(json.loads(line))
    log(f"  Loaded {len(pairs):,} pairs")

    # Shuffle before split
    rng.shuffle(pairs)

    # Format all samples
    log("Formatting samples ...")
    formatted = []
    for i, pair in enumerate(pairs):
        text = format_sample(
            phrase=pair["phrase"],
            output=pair.get("expected_output", ""),
            manifest_line=pair.get("manifest_line", ""),
            manifest_pool=manifest_pool,
            rng=rng,
        )
        formatted.append({
            "text": text,
            "id": f"sft_{i:06d}",
            "ciel_id": pair.get("ciel_id", ""),
            "concept_class": pair.get("concept_class", ""),
        })
        if (i + 1) % 100_000 == 0:
            log(f"  formatted {i+1:,} / {len(pairs):,}")

    log(f"  Total formatted: {len(formatted):,}")

    # Split train/val
    split_idx = int(len(formatted) * (1 - args.val_split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # Write train
    train_path = out_dir / "text_sft_train.jsonl"
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"  Train: {len(train_data):,} samples -> {train_path}")

    # Write val
    val_path = out_dir / "text_sft_val.jsonl"
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"  Val:   {len(val_data):,} samples -> {val_path}")

    # Dry-run subset
    if args.dry_run_samples > 0:
        n = min(args.dry_run_samples, len(train_data))
        dry_train = train_data[:int(n * 0.9)]
        dry_val = train_data[int(n * 0.9):n]

        dry_train_path = out_dir / "text_sft_dryrun_train.jsonl"
        with open(dry_train_path, "w") as f:
            for item in dry_train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        dry_val_path = out_dir / "text_sft_dryrun_val.jsonl"
        with open(dry_val_path, "w") as f:
            for item in dry_val:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        log(f"  Dry-run train: {len(dry_train):,} -> {dry_train_path}")
        log(f"  Dry-run val:   {len(dry_val):,} -> {dry_val_path}")

    # Show samples
    log("")
    log("Sample formatted text (first 2):")
    for item in formatted[:2]:
        log("-" * 40)
        log(item["text"][:500])
        log("...")

    # Stats
    log("")
    log("=" * 60)
    lengths = [len(item["text"]) for item in formatted[:1000]]
    import statistics
    log(f"  Char lengths (sampled 1K): mean={statistics.mean(lengths):.0f} "
        f"min={min(lengths)} max={max(lengths)}")
    log("  Done.")


if __name__ == "__main__":
    main()
