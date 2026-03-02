#!/usr/bin/env python3
"""
Assemble final training dataset from merged enrichment outputs.

Builds two datasets:

1. scribe_text_train.jsonl  — text-only training pairs (for Phase 1 LoRA SFT)
   Format: { "input": "CONCEPTS:\n...\nPHRASE: ...", "output": "key: value\n..." }

2. scribe_audio_index.jsonl — index for audio training (Phase 2)
   Links each training pair to its audio files after TTS generation.

Multi-phrase combination strategy:
  - Every concept appears in at least one training sample (guaranteed coverage)
  - Combo sizes weighted toward natural doctor dictation length (3-5 phrases)
  - Manifests randomly varied per sample (model learns to select from any subset)
  - African + general variation types both included

Usage:
    python3 assemble_training.py
    python3 assemble_training.py --samples 30000 --seed 42
    python3 assemble_training.py --check   # show stats only
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

MERGED_PHRASES = Path("/medASR/data/scribe_training/scribe_phrases.jsonl")
MERGED_PAIRS   = Path("/medASR/data/scribe_training/scribe_training.jsonl")
OUT_DIR        = Path("/medASR/data/scribe_training")

# Combo size distribution — weighted toward natural doctor dictation
COMBO_WEIGHTS = {1: 0.10, 2: 0.15, 3: 0.25, 4: 0.25, 5: 0.15, 6: 0.07, 7: 0.02, 8: 0.01}

# Manifest size: how many concepts to show in the manifest per sample
MANIFEST_SIZE_RANGE = (20, 50)

# Encounter type groupings for realistic manifest building
ENCOUNTER_CLASSES = {
    "outpatient_general": ["Diagnosis", "Finding", "Symptom", "Symptom/Finding", "Misc Order"],
    "vitals":             ["Finding"],
    "lab_order":          ["Test", "LabSet"],
    "medication":         ["Drug", "MedSet"],
    "procedure":          ["Procedure", "Radiology/Imaging Procedure"],
    "social_history":     ["Question"],
    "full_encounter":     None,  # all classes
}


def load_all_pairs(path: Path) -> list[dict]:
    """Load all training pairs."""
    if not path.exists():
        print(f"ERROR: {path} not found. Run merge_all_enrichment.py first.")
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(pairs):,} training pairs")
    return pairs


def load_all_concepts(path: Path) -> list[dict]:
    """Load concept records (with manifest_line and fhir_template)."""
    if not path.exists():
        return []
    concepts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    concepts.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(concepts):,} concept records")
    return concepts


def build_manifest_string(selected_concepts: list[dict]) -> str:
    """Build the manifest string the model sees."""
    lines = ["CONCEPTS:"]
    for c in selected_concepts:
        lines.append(c["manifest_line"])
    return "\n".join(lines)


def extract_key(manifest_line: str) -> str:
    """Extract the bare key from a manifest line."""
    import re
    key = manifest_line
    key = re.sub(r"^\[dx\]\s*", "", key)
    key = re.sub(r"^\[order\]\s*", "", key)
    key = re.sub(r"^\[drug\]\s*", "", key)
    key = re.sub(r"^\[test\]\s*", "", key)
    key = re.sub(r"^\[value\]\s*", "", key)
    key = re.sub(r"\s*\(.*\)$", "", key).strip()
    return key


def assemble_sample(
    target_pairs: list[dict],
    all_concepts_by_id: dict,
    all_concepts_list: list[dict],
    rng: random.Random,
) -> dict | None:
    """Build one training sample from 1-8 concept phrases."""
    if not target_pairs:
        return None

    # Choose combo size
    sizes = list(COMBO_WEIGHTS.keys())
    weights = list(COMBO_WEIGHTS.values())
    size = rng.choices(sizes, weights=weights, k=1)[0]
    size = min(size, len(target_pairs))

    # Select pairs (prefer different concepts)
    selected_pairs = rng.sample(target_pairs, k=size)

    # Build combined phrase
    phrases = [p["phrase"] for p in selected_pairs]
    combined_phrase = ", ".join(phrases)

    # Build expected output (one line per concept)
    output_lines = [p["expected_output"] for p in selected_pairs]
    expected_output = "\n".join(output_lines)

    # Build manifest: include target concepts + random fillers
    target_concept_ids = {p.get("ciel_id") for p in selected_pairs}
    target_manifest_concepts = [
        all_concepts_by_id[cid] for cid in target_concept_ids
        if cid in all_concepts_by_id
    ]

    # Add filler concepts (random subset from all concepts)
    manifest_size = rng.randint(*MANIFEST_SIZE_RANGE)
    filler_count  = max(0, manifest_size - len(target_manifest_concepts))
    fillers = rng.sample(all_concepts_list, k=min(filler_count, len(all_concepts_list)))

    # Combine and shuffle
    manifest_concepts = target_manifest_concepts + fillers
    rng.shuffle(manifest_concepts)

    manifest_string = build_manifest_string(manifest_concepts)

    return {
        "input":           f"{manifest_string}\n\nPHRASE: \"{combined_phrase}\"",
        "output":          expected_output,
        "phrase_count":    size,
        "ciel_ids":        list(target_concept_ids),
        "variation_types": list({p.get("variation_type") for p in selected_pairs}),
        "fhir_types":      list({p.get("fhir_type") for p in selected_pairs}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Target number of training samples (default: 50K)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--check",   action="store_true",
                        help="Show stats only, don't write")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 65)
    print("Assembling Voice Scribe Training Dataset")
    print("=" * 65)

    # Load data
    all_pairs    = load_all_pairs(MERGED_PAIRS)
    all_concepts = load_all_concepts(MERGED_PHRASES)

    if not all_pairs or not all_concepts:
        return

    # Build lookup structures
    all_concepts_by_id = {c["ciel_id"]: c for c in all_concepts}
    pairs_by_ciel      = defaultdict(list)
    for p in all_pairs:
        pairs_by_ciel[p.get("ciel_id", "")].append(p)

    print(f"\nUnique concepts with phrases: {len(pairs_by_ciel):,}")
    print(f"Target training samples:      {args.samples:,}")

    if args.check:
        # Just show what we'd generate
        class_dist = Counter(c.get("concept_class") for c in all_concepts)
        print("\nConcept class distribution:")
        for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
            print(f"  {cls:35} {cnt:>8,}")
        print(f"\nCombo size distribution (for {args.samples:,} samples):")
        for size, weight in COMBO_WEIGHTS.items():
            n = int(args.samples * weight)
            print(f"  {size} phrase(s): {weight*100:.0f}% = ~{n:,} samples")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scribe_text_train.jsonl"

    # ── GUARANTEED COVERAGE PASS ──────────────────────────────────────────────
    # Every concept must appear in at least one training sample.
    # This ensures no concept is left out regardless of random sampling.
    print("\n[Pass 1] Guaranteed coverage — every concept in at least one sample...")
    coverage_samples = []
    all_ciel_ids = list(pairs_by_ciel.keys())
    rng.shuffle(all_ciel_ids)

    covered = set()
    for ciel_id in all_ciel_ids:
        if ciel_id in covered:
            continue
        pairs = pairs_by_ciel[ciel_id]
        if not pairs:
            continue
        sample = assemble_sample(pairs, all_concepts_by_id, all_concepts, rng)
        if sample:
            coverage_samples.append(sample)
            covered.update(sample["ciel_ids"])

    print(f"  Coverage samples: {len(coverage_samples):,} (covers {len(covered):,} concepts)")

    # ── RANDOM COMBINATION PASS ───────────────────────────────────────────────
    # Fill remaining quota with random multi-concept combinations.
    remaining_quota = max(0, args.samples - len(coverage_samples))
    print(f"\n[Pass 2] Random combinations — {remaining_quota:,} additional samples...")

    combo_samples = []
    all_pairs_list = all_pairs  # flat list for random sampling

    for i in range(remaining_quota):
        # Pick a random anchor concept, then build a combo
        anchor_id = rng.choice(all_ciel_ids)
        anchor_pairs = pairs_by_ciel.get(anchor_id, [])
        if not anchor_pairs:
            continue

        # Choose combo size
        size = rng.choices(list(COMBO_WEIGHTS.keys()), weights=list(COMBO_WEIGHTS.values()), k=1)[0]

        # Select pairs: anchor + random others
        other_pairs = rng.sample(all_pairs_list, k=min(size - 1, len(all_pairs_list)))
        target_pairs = [rng.choice(anchor_pairs)] + other_pairs
        target_pairs = target_pairs[:size]

        sample = assemble_sample(target_pairs, all_concepts_by_id, all_concepts, rng)
        if sample:
            combo_samples.append(sample)

        if (i + 1) % 10_000 == 0:
            print(f"  {i+1:,}/{remaining_quota:,} samples built...")

    print(f"  Combo samples: {len(combo_samples):,}")

    # ── WRITE OUTPUT ──────────────────────────────────────────────────────────
    all_samples = coverage_samples + combo_samples
    rng.shuffle(all_samples)

    print(f"\nWriting {len(all_samples):,} samples to {out_path}...")
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    import os
    size_mb = os.path.getsize(out_path) / 1024 / 1024

    # Stats
    phrase_counts = Counter(s["phrase_count"] for s in all_samples)
    fhir_counts   = Counter(ft for s in all_samples for ft in s["fhir_types"])
    vtype_counts  = Counter(vt for s in all_samples for vt in s["variation_types"])

    print(f"\n{'='*65}")
    print(f"ASSEMBLY COMPLETE")
    print(f"{'='*65}")
    print(f"  Total samples:    {len(all_samples):,}")
    print(f"  Coverage samples: {len(coverage_samples):,} (all {len(covered):,} concepts)")
    print(f"  Combo samples:    {len(combo_samples):,}")
    print(f"  Output size:      {size_mb:.0f} MB")
    print(f"\n  Phrase count distribution:")
    for n in sorted(phrase_counts):
        pct = phrase_counts[n] / len(all_samples) * 100
        print(f"    {n} phrase(s): {phrase_counts[n]:>7,} ({pct:.1f}%)")
    print(f"\n  FHIR resource types:")
    for ft, cnt in sorted(fhir_counts.items(), key=lambda x: -x[1]):
        print(f"    {ft:35} {cnt:>8,}")
    print(f"\n  Variation types:")
    for vt, cnt in sorted(vtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {vt:35} {cnt:>8,}")

    # Write manifest
    manifest = {
        "generated":       datetime.now().isoformat(),
        "total_samples":   len(all_samples),
        "coverage_samples": len(coverage_samples),
        "concepts_covered": len(covered),
        "phrase_distribution": dict(phrase_counts),
        "output_file":     str(out_path),
    }
    with open(OUT_DIR / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest: {OUT_DIR / 'training_manifest.json'}")
    print(f"  Training: {out_path}")


if __name__ == "__main__":
    main()
