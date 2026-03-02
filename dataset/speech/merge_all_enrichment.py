#!/usr/bin/env python3
"""
Merge all enrichment outputs into flat training JSONL files.

Reads from:
  /medASR/scripts/scribe_enrichment/enriched/      (main: 38,558 concepts)
  /medASR/scripts/question_enrichment/enriched/    (questions: 27 concepts)
  /medASR/scripts/remaining_enrichment/enriched/   (remaining: 16,845 concepts)

Writes to:
  /medASR/data/scribe_training/
    scribe_phrases.jsonl     — one record per concept (with all phrases)
    scribe_training.jsonl    — one record per training pair (flat)
    manifest.json            — summary stats

Usage:
    python3 merge_all_enrichment.py
    python3 merge_all_enrichment.py --check   # just show counts, don't write
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

SOURCES = [
    Path("/medASR/scripts/scribe_enrichment/enriched"),
    Path("/medASR/scripts/question_enrichment/enriched"),
    Path("/medASR/scripts/remaining_enrichment/enriched"),
    Path("/medASR/scripts/final_enrichment/enriched"),    # missing 5% retry
]
OUT_DIR = Path("/medASR/data/scribe_training")


def iter_results(sources: list[Path]):
    for source_dir in sources:
        if not source_dir.exists():
            print(f"  SKIP {source_dir} — not found")
            continue
        count = 0
        for shard_dir in sorted(source_dir.glob("shard_*")):
            results = shard_dir / "results.jsonl"
            if not results.exists():
                continue
            for line in open(results):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), source_dir.parent.name
                    count += 1
                except Exception:
                    continue
        print(f"  {source_dir.parent.name}: {count:,} concepts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Count only, don't write output")
    args = parser.parse_args()

    print("=" * 65)
    print("Merging all enrichment outputs")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_concepts = 0
    total_phrases  = 0
    total_pairs    = 0
    seen_ciel_ids  = set()
    class_counts   = Counter()
    vtype_counts   = Counter()
    fhir_counts    = Counter()

    if args.check:
        for rec, source in iter_results(SOURCES):
            if rec["ciel_id"] in seen_ciel_ids:
                continue
            seen_ciel_ids.add(rec["ciel_id"])
            total_concepts += 1
            total_phrases  += rec.get("phrase_count", 0)
            total_pairs    += rec.get("pair_count", 0)
            class_counts[rec.get("concept_class", "?")] += 1
            for tp in rec.get("training_pairs", []):
                vtype_counts[tp.get("variation_type", "?")] += 1
                fhir_counts[tp.get("fhir_type", "?")] += 1
    else:
        phrases_path  = OUT_DIR / "scribe_phrases.jsonl"
        training_path = OUT_DIR / "scribe_training.jsonl"

        with open(phrases_path, "w") as pf, open(training_path, "w") as tf:
            for rec, source in iter_results(SOURCES):
                if rec["ciel_id"] in seen_ciel_ids:
                    continue
                seen_ciel_ids.add(rec["ciel_id"])
                total_concepts += 1
                total_phrases  += rec.get("phrase_count", 0)
                total_pairs    += rec.get("pair_count", 0)
                class_counts[rec.get("concept_class", "?")] += 1

                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                for tp in rec.get("training_pairs", []):
                    vtype_counts[tp.get("variation_type", "?")] += 1
                    fhir_counts[tp.get("fhir_type", "?")] += 1
                    flat = {
                        "ciel_id":         rec["ciel_id"],
                        "name":            rec["name"],
                        "concept_class":   rec.get("concept_class", ""),
                        "manifest_line":   rec.get("manifest_line", ""),
                        "phrase":          tp["phrase"],
                        "variation_type":  tp["variation_type"],
                        "expected_output": tp["expected_output"],
                        "fhir_type":       tp["fhir_type"],
                        "fhir_template":   rec.get("fhir_template", {}),
                    }
                    tf.write(json.dumps(flat, ensure_ascii=False) + "\n")

        import os
        p_size = os.path.getsize(phrases_path) / 1024 / 1024
        t_size = os.path.getsize(training_path) / 1024 / 1024
        print(f"\n  scribe_phrases.jsonl:  {p_size:.0f} MB")
        print(f"  scribe_training.jsonl: {t_size:.0f} MB")

        manifest = {
            "generated":       datetime.now().isoformat(),
            "total_concepts":  total_concepts,
            "total_phrases":   total_phrases,
            "total_pairs":     total_pairs,
            "by_class":        dict(class_counts),
            "by_variation":    dict(vtype_counts),
            "by_fhir_type":    dict(fhir_counts),
        }
        with open(OUT_DIR / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"TOTALS")
    print(f"{'=' * 65}")
    print(f"  Unique concepts:  {total_concepts:,}")
    print(f"  Total phrases:    {total_phrases:,}")
    print(f"  Training pairs:   {total_pairs:,}")
    print(f"\n  By concept class:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:35} {cnt:>8,}")
    print(f"\n  By variation type:")
    for vt, cnt in sorted(vtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {vt:35} {cnt:>8,}")
    print(f"\n  By FHIR resource type:")
    for ft, cnt in sorted(fhir_counts.items(), key=lambda x: -x[1]):
        print(f"    {ft:35} {cnt:>8,}")


if __name__ == "__main__":
    main()
# NOTE: final_enrichment added as 4th source
