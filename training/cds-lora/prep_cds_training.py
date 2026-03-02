#!/usr/bin/env python3
"""
CDS KB Tool-Use LoRA — Data Preparation
========================================
Reads the pre-built SFT dataset, applies token-length filtering for max_seq_length=5120,
and writes final train/val splits ready for training.

Sources: /var/www/ClinicDx/dataset/CDS_dataset/sft_train/{train,val}.jsonl
Output:  /var/www/ClinicDx/training/cds_kb_lora/data/{train,val}.jsonl
"""

import json
import re
import sys
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

SRC_DIR = Path("/var/www/ClinicDx/dataset/CDS_dataset/sft_train")
OUT_DIR = Path("/var/www/ClinicDx/training/cds_kb_lora/data")
MAX_TOKENS = 16384
MIN_QUERIES = 2
CHARS_PER_TOKEN = 4.0


def load_and_filter(path: Path) -> list:
    records = []
    stats = Counter()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue

            text = r.get("text", "")
            if not text:
                stats["empty_text"] += 1
                continue

            est_tokens = len(text) / CHARS_PER_TOKEN
            if est_tokens > MAX_TOKENS:
                stats["too_long"] += 1
                continue

            query_count = len(re.findall(r"<KB_QUERY>", text))
            if query_count < MIN_QUERIES:
                stats["too_few_queries"] += 1
                continue

            if "<start_of_turn>model" not in text:
                stats["no_model_turn"] += 1
                continue

            records.append(r)
            stats["accepted"] += 1

    return records, stats


def compute_stats(records: list) -> dict:
    sources = Counter()
    query_counts = []
    token_lens = []
    has_retry = 0

    for r in records:
        src = r.get("source", r.get("stream", "unknown"))
        sources[src] += 1
        text = r.get("text", "")
        qc = len(re.findall(r"<KB_QUERY>", text))
        query_counts.append(qc)
        token_lens.append(len(text) / CHARS_PER_TOKEN)
        if r.get("has_retry_training"):
            has_retry += 1

    return {
        "count": len(records),
        "sources": dict(sorted(sources.items(), key=lambda x: -x[1])),
        "query_distribution": dict(sorted(Counter(query_counts).items())),
        "mean_queries": round(sum(query_counts) / max(len(query_counts), 1), 2),
        "token_estimate": {
            "min": round(min(token_lens)) if token_lens else 0,
            "avg": round(sum(token_lens) / max(len(token_lens), 1)),
            "max": round(max(token_lens)) if token_lens else 0,
        },
        "retry_chains": has_retry,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CDS KB LoRA — Data Preparation")
    print(f"Max tokens: {MAX_TOKENS}, Min queries: {MIN_QUERIES}")
    print("=" * 60)
    print()

    all_stats = {}
    for split in ["train", "val"]:
        src_path = SRC_DIR / f"{split}.jsonl"
        if not src_path.exists():
            print(f"ERROR: {src_path} not found")
            sys.exit(1)

        print(f"Loading {split} from {src_path} ...")
        records, filter_stats = load_and_filter(src_path)

        print(f"  Filter results: {dict(filter_stats)}")
        stats = compute_stats(records)
        all_stats[split] = stats

        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(records)} records to {out_path}")
        print(f"  Sources: {stats['sources']}")
        print(f"  Query dist: {stats['query_distribution']}")
        print(f"  Mean queries: {stats['mean_queries']}")
        print(f"  Token estimate: {stats['token_estimate']}")
        print(f"  Retry chains: {stats['retry_chains']}")
        print()

    manifest = {
        "generated": datetime.now().isoformat(),
        "max_tokens": MAX_TOKENS,
        "min_queries": MIN_QUERIES,
        "train": all_stats["train"],
        "val": all_stats["val"],
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")
    print()
    print(f"Total training samples: {all_stats['train']['count']}")
    print(f"Total validation samples: {all_stats['val']['count']}")
    print("Done.")


if __name__ == "__main__":
    main()
