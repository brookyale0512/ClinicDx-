#!/usr/bin/env python3
"""
Build production-ready SFT dataset for ClinicDx CDS ante-hoc reasoning.

Sources:
  - CDS enriched:  /var/www/kbToolUseLora/data/cds_enrichment/enriched/*/results.jsonl
  - MCQ enriched:  /var/www/kbToolUseLora/benchmark_dataset/mcq_enriched/*/results.jsonl

Output:
  - sft_train/train.jsonl   — 90% split
  - sft_train/val.jsonl     — 10% split
  - sft_train/manifest.json — stats and provenance

Quality gates applied:
  - Require 'text' field with <think> and </think>
  - Require at least 1 KB query with score > 0
  - Require think tokens >= 100
  - MCQ: require correct=True
  - Deduplicate by id

Usage:
    python3 build_sft_dataset.py
    python3 build_sft_dataset.py --dry-run
"""

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

CDS_DIR   = Path("/var/www/kbToolUseLora/data/cds_enrichment/enriched")
MCQ_DIR   = Path("/var/www/kbToolUseLora/benchmark_dataset/mcq_enriched")
OUT_DIR   = Path("/var/www/ClinicDx/dataset/CDS_dataset/sft_train")
RAW_CDS   = Path("/var/www/ClinicDx/dataset/CDS_dataset/raw/cds_enriched")
RAW_MCQ   = Path("/var/www/ClinicDx/dataset/CDS_dataset/raw/mcq_enriched")

VAL_SPLIT = 0.10
SEED      = 42


def passes_quality_gate(record: dict, source_type: str) -> tuple[bool, str]:
    """Return (passes, reason_if_not)."""
    text = record.get("text", "")
    if not text:
        return False, "empty_text"
    if "<think>" not in text or "</think>" not in text:
        return False, "no_think_tags"

    queries = record.get("kb_queries", [])
    scores  = record.get("kb_scores", [])
    if not queries:
        return False, "no_kb_queries"

    valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > 0]
    if not valid_scores:
        return False, "all_zero_scores"

    # Think depth check
    tk = record.get("think_token_counts", {})
    if isinstance(tk, dict) and tk:
        total_think = sum(v for v in tk.values() if isinstance(v, (int, float)))
        if total_think < 100:
            return False, "thin_think"
    elif not tk:
        # No think token count — check text length as proxy
        think_start = text.find("<think>")
        think_end   = text.find("</think>")
        if think_end - think_start < 200:
            return False, "thin_think_text"

    if source_type == "mcq":
        if not record.get("correct"):
            return False, "mcq_wrong_answer"

    return True, "ok"


def load_source(source_dir: Path, source_type: str) -> tuple[list, dict]:
    """Load, deduplicate, quality-filter records. Returns (records, stats)."""
    all_records = []
    seen_ids    = set()
    stats       = Counter()

    files = sorted(source_dir.rglob("results.jsonl"))
    log(f"  Loading {len(files)} files from {source_dir.name} ...")

    for fpath in files:
        for line in open(fpath):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue

            record_id = r.get("id", "")

            # Dedup
            if record_id and record_id in seen_ids:
                stats["duplicate"] += 1
                continue
            if record_id:
                seen_ids.add(record_id)

            passes, reason = passes_quality_gate(r, source_type)
            if not passes:
                stats[f"filtered_{reason}"] += 1
                continue

            # Keep only fields needed for training
            clean = {
                "id":          r.get("id", ""),
                "source":      r.get("source", source_type),
                "stream":      r.get("stream", source_type),
                "text":        r.get("text", ""),
                "kb_queries":  r.get("kb_queries", []),
                "kb_scores":   r.get("kb_scores", []),
            }
            # Preserve category/subject for CDS
            if "category" in r:  clean["category"] = r["category"]
            if "subject"  in r:  clean["subject"]  = r["subject"]
            if "correct"  in r:  clean["correct"]  = r["correct"]

            all_records.append(clean)
            stats["kept"] += 1

    return all_records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(SEED)

    log("=" * 65)
    log("  ClinicDx SFT Dataset Builder")
    log("=" * 65)

    # ── Load CDS ───────────────────────────────────────────────────
    log("\n[1/5] Loading CDS enriched data ...")
    cds_records, cds_stats = load_source(CDS_DIR, "cds")
    log(f"  CDS kept: {cds_stats['kept']:,}")
    for k, v in cds_stats.items():
        if k != "kept" and v > 0:
            log(f"    filtered/{k}: {v:,}")

    # ── Load MCQ ───────────────────────────────────────────────────
    log("\n[2/5] Loading MCQ enriched data ...")
    mcq_records, mcq_stats = load_source(MCQ_DIR, "mcq")
    log(f"  MCQ kept: {mcq_stats['kept']:,}")
    for k, v in mcq_stats.items():
        if k != "kept" and v > 0:
            log(f"    filtered/{k}: {v:,}")

    # ── Mix and shuffle ────────────────────────────────────────────
    log("\n[3/5] Mixing and shuffling ...")
    all_records = cds_records + mcq_records
    rng.shuffle(all_records)

    total     = len(all_records)
    cds_n     = len(cds_records)
    mcq_n     = len(mcq_records)
    cds_pct   = cds_n / total * 100
    mcq_pct   = mcq_n / total * 100

    log(f"  CDS:   {cds_n:>7,}  ({cds_pct:.1f}%)")
    log(f"  MCQ:   {mcq_n:>7,}  ({mcq_pct:.1f}%)")
    log(f"  TOTAL: {total:>7,}")

    # ── Split ──────────────────────────────────────────────────────
    val_n   = int(total * VAL_SPLIT)
    train_n = total - val_n
    train   = all_records[:train_n]
    val     = all_records[train_n:]
    log(f"\n[4/5] Split: train={train_n:,}  val={val_n:,}")

    if args.dry_run:
        log("\nDRY RUN — no files written.")
        return

    # ── Write ──────────────────────────────────────────────────────
    log("\n[5/5] Writing output files ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CDS.mkdir(parents=True, exist_ok=True)
    RAW_MCQ.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train.jsonl"
    val_path   = OUT_DIR / "val.jsonl"

    with open(train_path, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Manifest ───────────────────────────────────────────────────
    # Token length stats (text field)
    text_lens = [len(r["text"]) for r in all_records]
    import statistics
    manifest = {
        "generated":       datetime.now().isoformat(),
        "seed":            SEED,
        "val_split":       VAL_SPLIT,
        "totals": {
            "cds":   cds_n,
            "mcq":   mcq_n,
            "total": total,
            "train": train_n,
            "val":   val_n,
        },
        "ratio": {
            "cds_pct": round(cds_pct, 2),
            "mcq_pct": round(mcq_pct, 2),
        },
        "quality": {
            "cds_filtered": {k: v for k, v in cds_stats.items() if k != "kept"},
            "mcq_filtered": {k: v for k, v in mcq_stats.items() if k != "kept"},
        },
        "text_length_chars": {
            "min":    min(text_lens),
            "max":    max(text_lens),
            "mean":   round(statistics.mean(text_lens)),
            "median": round(statistics.median(text_lens)),
        },
        "files": {
            "train": str(train_path),
            "val":   str(val_path),
        },
        "sources": {
            "cds":  str(CDS_DIR),
            "mcq":  str(MCQ_DIR),
        },
    }

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    train_size = train_path.stat().st_size / 1e6
    val_size   = val_path.stat().st_size / 1e6

    log(f"  train.jsonl:   {train_n:,} rows  ({train_size:.0f} MB)  → {train_path}")
    log(f"  val.jsonl:     {val_n:,} rows  ({val_size:.0f} MB)   → {val_path}")
    log(f"  manifest.json: {manifest_path}")
    log(f"\nDone. Total: {total:,} records  ({train_size+val_size:.0f} MB)")


if __name__ == "__main__":
    main()
