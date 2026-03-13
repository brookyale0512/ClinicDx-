#!/usr/bin/env python3
"""
CDS KB v2 — Cycle 1 Data Prep
===============================
Reads all enriched shards from production_run_v2/train/,
applies quality filter, shuffles, splits 600 records as val,
and writes train.jsonl + val.jsonl to the A100 training data dir.

Usage:
    python3 prep_cycle1.py [--dry-run]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SHARD_DIR = Path("/var/www/ClinicDx/training/react_v6_local/data/production_run_v2/train")
OUT_DIR   = Path("/mnt/a100/training/lora_cds_kb_v2/data")
VAL_SIZE  = 600
SEED      = 42


def quality_pass(record: dict) -> bool:
    q = record.get("metadata", {}).get("quality", {})
    return (
        q.get("has_exact_six_sections", False)
        and q.get("actions_evidence_mapped", False)
        and not q.get("meta_contamination", False)
    )


def load_shards(shard_dir: Path) -> list[dict]:
    records = []
    seen_ids = set()
    shards = sorted(shard_dir.glob("shard_*.jsonl"))
    for shard in shards:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = r.get("id", "")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                records.append(r)
    return records


def print_stats(label: str, records: list[dict]):
    nq = Counter(r.get("num_queries", 0) for r in records)
    lens = [len(r.get("text", "")) for r in records]
    avg_len = sum(lens) // max(len(lens), 1)
    max_len = max(lens) if lens else 0
    bad = sum(1 for r in records if not r.get("text", "").startswith("<bos>"))
    sources = Counter(r.get("source", "?") for r in records)
    print(f"\n  {label}: {len(records):,} records")
    print(f"    queries dist : {dict(sorted(nq.items()))}")
    print(f"    avg chars    : {avg_len:,}  (~{avg_len//4} tokens)")
    print(f"    max chars    : {max_len:,}  (~{max_len//4} tokens)")
    print(f"    bad format   : {bad}")
    print(f"    sources      : {dict(sources)}")
    ids = [r.get("id","") for r in records]
    dupes = len(ids) - len(set(ids))
    print(f"    duplicate IDs: {dupes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report stats but do not write files")
    args = parser.parse_args()

    print(f"Loading shards from: {SHARD_DIR}")
    all_records = load_shards(SHARD_DIR)
    print(f"Raw records (deduped): {len(all_records):,}")

    filtered = [r for r in all_records if quality_pass(r)]
    rejected = len(all_records) - len(filtered)
    print(f"Quality filter: {len(filtered):,} pass, {rejected} rejected ({rejected/max(len(all_records),1)*100:.1f}%)")

    random.seed(SEED)
    random.shuffle(filtered)

    val_records   = filtered[-VAL_SIZE:]
    train_records = filtered[:-VAL_SIZE]

    print_stats("train", train_records)
    print_stats("val",   val_records)

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train.jsonl"
    val_path   = OUT_DIR / "val.jsonl"

    # Warn if overwriting
    for p in [train_path, val_path]:
        if p.exists():
            existing = sum(1 for _ in p.read_text().splitlines() if _.strip())
            print(f"\n  WARNING: {p} already exists ({existing:,} lines) — overwriting")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Written: {train_path}  ({len(train_records):,} records)")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written: {val_path}  ({len(val_records):,} records)")

    # Verify written counts match
    written_train = sum(1 for l in train_path.read_text().splitlines() if l.strip())
    written_val   = sum(1 for l in val_path.read_text().splitlines() if l.strip())
    assert written_train == len(train_records), f"Train write mismatch: {written_train} != {len(train_records)}"
    assert written_val   == len(val_records),   f"Val write mismatch: {written_val} != {len(val_records)}"
    print(f"\n  Verification passed: train={written_train:,}, val={written_val:,}")
    print(f"\nDone. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
