#!/usr/bin/env python3
"""
Fix 6 identified data-quality issues in scribe_training.jsonl.

Issues:
  #1  Placeholder phrases ("Phrase 1", "Phrase two", etc.)     — DELETE
  #2  Bracket placeholders ("[5 phrases]", "[GENERAL_…]")      — DELETE
  #3  Quoted meta-text  (""Diagnosis:…" (Pidgin influence)")   — STRIP or DELETE
  #4  Meta instructions ("Using X more commonly?")             — DELETE
  #5  Template output artifacts ({{value}})                    — DELETE
  #6  Cross-concept duplicate phrases                          — KEEP FIRST, DROP REST

Usage:
    python3 scripts/fix_training_data.py
"""

import json, re, sys, time
from collections import Counter
from pathlib import Path

INPUT  = Path("/medASR/data/scribe_training/scribe_training.jsonl")
OUTPUT = Path("/medASR/data/scribe_training/scribe_training_fixed.jsonl")

# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def is_placeholder(p):
    p = p.strip().lower()
    return bool(re.match(r'^phrase\s*(\d+|one|two|three|four|five|six|seven|eight)$', p))

def is_bracket(p):
    p = p.strip()
    if not (p.startswith('[') and ']' in p): return False
    inner = p[1:p.index(']')].lower()
    return bool(re.match(r'^\d+\s+phrases?$', inner) or
                any(k in inner for k in ('phrase','general','confirmed','uncertain',
                                          'african','absent','negation')))

def is_quoted_meta(p):
    p = p.strip()
    return p.startswith('"') and '(' in p and (p.endswith(')') or p.endswith(')"'))

def extract_inner(p):
    """Try to pull the real phrase out of quoted meta-text."""
    m = re.match(r'^"([^"]+)"', p.strip())
    if m:
        inner = re.sub(r'\s*\(.*\)$', '', m.group(1)).strip()
        if len(inner.split()) >= 2 and not inner.startswith('Using '):
            return inner
    return None

def is_meta_instruction(p):
    p = p.strip()
    return (p.startswith('Using ') and '?' in p) or \
           (p.startswith('Or sometimes') and '?' in p) or \
           (p.startswith('In some') and '?' in p)

def has_template_output(o):
    return '{{' in o

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Training Data Fix — 6 issues")
    log("=" * 60)
    log(f"  Input:  {INPUT}")
    log(f"  Output: {OUTPUT}")

    total = 0
    kept = 0
    removed = Counter()
    salvaged = 0
    dedup_dropped = 0
    same_dedup = 0

    phrase_owner = {}          # phrase -> first ciel_id seen
    phrase_concept_seen = set() # (phrase, ciel_id) for same-concept dedup

    t0 = time.time()

    with open(INPUT) as fin, open(OUTPUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                r = json.loads(line)
            except Exception:
                removed["json_error"] += 1
                continue

            phrase = r.get("phrase", "")
            output = r.get("expected_output", "")
            cid    = r.get("ciel_id", "")

            # ── Issue #1: placeholder ──
            if is_placeholder(phrase):
                removed["#1_placeholder"] += 1
                continue

            # ── Issue #2: bracket placeholder ──
            if is_bracket(phrase):
                removed["#2_bracket"] += 1
                continue

            # ── Issue #3: quoted meta-text ──
            if is_quoted_meta(phrase):
                inner = extract_inner(phrase)
                if inner:
                    r["phrase"] = inner
                    phrase = inner
                    salvaged += 1
                else:
                    removed["#3_quoted_meta"] += 1
                    continue

            # ── Issue #4: meta instruction ──
            if is_meta_instruction(phrase):
                removed["#4_meta_instr"] += 1
                continue

            # ── Issue #5: template output ──
            if has_template_output(output):
                removed["#5_template_out"] += 1
                continue

            # ── Issue #6: cross-concept duplicate ──
            pc_key = (phrase, cid)
            if pc_key in phrase_concept_seen:
                same_dedup += 1
                removed["#6a_same_dedup"] += 1
                continue
            phrase_concept_seen.add(pc_key)

            if phrase in phrase_owner:
                if phrase_owner[phrase] != cid:
                    dedup_dropped += 1
                    removed["#6b_cross_dedup"] += 1
                    continue
            else:
                phrase_owner[phrase] = cid

            # ── KEEP ──
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

            if total % 100_000 == 0:
                elapsed = time.time() - t0
                rate = total / max(elapsed, 0.1)
                log(f"  processed {total:>9,} | kept {kept:>9,} | "
                    f"removed {total-kept:>6,} | {rate:,.0f} rows/sec")

    elapsed = time.time() - t0

    log("")
    log("=" * 60)
    log("  COMPLETE")
    log("=" * 60)
    log(f"  Input rows:     {total:>10,}")
    log(f"  Output rows:    {kept:>10,}")
    log(f"  Total removed:  {total - kept:>10,}")
    log(f"  Salvaged (#3):  {salvaged:>10,}")
    log(f"  Elapsed:        {elapsed:>10.1f}s")
    log("")
    log("  Breakdown:")
    for reason, cnt in sorted(removed.items()):
        log(f"    {reason:25s}: {cnt:>7,}")
    log("")

    # Quick validation
    log("  Validating output...")
    out_concepts = set()
    out_phrases = set()
    out_count = 0
    with open(OUTPUT) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            out_concepts.add(r.get("ciel_id",""))
            out_phrases.add(r.get("phrase",""))
            out_count += 1

    log(f"    Rows:     {out_count:,}")
    log(f"    Concepts: {len(out_concepts):,}")
    log(f"    Phrases:  {len(out_phrases):,} unique")
    log(f"    Output:   {OUTPUT}")
    log(f"    Size:     {OUTPUT.stat().st_size / 1024 / 1024:.0f} MB")
    log("")
    log("  Done. Review output, then:")
    log("    mv scribe_training_fixed.jsonl scribe_training.jsonl")


if __name__ == "__main__":
    main()
