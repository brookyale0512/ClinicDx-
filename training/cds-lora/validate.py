#!/usr/bin/env python3
"""
Validate think_v1 checkpoint (medgemma-4b-it + LoRA) with live KB queries.
Uses in-process vLLM — no server needed.
Mirrors training conversation structure: <think>...</think><KB_QUERY>...</KB_QUERY>
"""

import re, json, sys, time, random, urllib.request
from collections import Counter

BASE_MODEL = "/var/www/ClinicDx/model/medgemma-4b-it"
LORA_CKPT = "/var/www/ClinicDx/training/lora_cds_kb_v2/checkpoints"
VAL_PATH = "/var/www/ClinicDx/training/lora_cds_kb_v2/data/val.jsonl"
KB_URL = "http://10.128.0.4:4276"

NUM_CASES = 0  # 0 = all cases
MAX_TURNS = 5
MAX_KB_QUERIES = 3
MAX_TOKENS_PER_TURN = 1500
MAX_TOKENS_FINAL = 4096
SEED = 42

EXPECTED_SECTIONS = [
    "Clinical Assessment",
    "Differential Diagnoses",
    "Recommended Investigations",
    "Treatment Plan",
    "Patient Education & Follow-Up",
]


def query_kb(q):
    try:
        data = json.dumps({
            "query": q, "k": 3,
            "source_mode": "auto", "snippet_chars": 1200,
        }).encode()
        req = urllib.request.Request(
            f"{KB_URL}/search", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()).get("hit")
    except Exception as e:
        print(f"    KB ERROR: {e}", file=sys.stderr)
        return None


def extract_user_prompt(text):
    m = re.search(r"<start_of_turn>user\n(.*?)<end_of_turn>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def count_expected_user_turns(text):
    """Count how many model turns (KB queries) exist in the gold data."""
    return text.count("<start_of_turn>model")


def extract_gold_answer(text):
    """Extract the final model turn from gold data (the visible answer)."""
    model_turns = re.findall(
        r"<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)", text, re.DOTALL)
    if model_turns:
        last = model_turns[-1]
        visible = re.sub(r"<think>.*?</think>", "", last, flags=re.DOTALL).strip()
        visible = re.sub(r"<KB_QUERY>.*?</KB_QUERY>", "", visible).strip()
        return visible
    return ""


def run_case(llm, lora_req, SamplingParams, case, idx, total):
    prompt_text = extract_user_prompt(case["text"])
    case_id = case.get("id", "unknown")
    gold_queries = case.get("num_queries", "?")

    print(f"\n{'='*70}")
    print(f"CASE {idx}/{total} | {case_id} | gold_queries={gold_queries}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt_text[:200]}...")

    conv = f"<bos><start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    all_text = ""
    kb_queries = []
    t0 = time.time()
    forced_answer = False

    for turn in range(MAX_TURNS):
        max_tok = MAX_TOKENS_PER_TURN
        params = SamplingParams(temperature=0.0, max_tokens=max_tok, stop=["<end_of_turn>"])
        out = llm.generate([conv], params, lora_request=lora_req)
        gen = out[0].outputs[0].text
        finish = out[0].outputs[0].finish_reason
        all_text += gen

        has_tc = "</think>" in gen
        kbq = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", gen, re.DOTALL)
        think_m = re.search(r"<think>(.*?)</think>", gen, re.DOTALL)
        think_len = len(think_m.group(1)) if think_m else 0

        print(f"  TURN {turn} | {len(gen)}c | finish={finish} | </think>={has_tc} | think={think_len}c | kb={[q.strip()[:60] for q in kbq]}")

        if kbq and turn < MAX_TURNS - 1:
            q = kbq[-1].strip()
            kb = query_kb(q)
            if kb:
                score_str = f"{kb['score']:.1f}/{kb['source']}"
                kb_content = f'<KB_RESULT source="{kb["source"]}" score="{kb["score"]:.1f}">\n{kb["content"]}\n</KB_RESULT>'
            else:
                score_str = "MISS"
                kb_content = '<KB_RESULT source="none" score="0">\nNo evidence found.\n</KB_RESULT>'
            kb_queries.append({"query": q, "score": score_str})
            print(f"    KB: '{q[:80]}' => {score_str}")

            if len(kb_queries) >= MAX_KB_QUERIES:
                print(f"  >> Max KB queries reached ({MAX_KB_QUERIES}), forcing final answer...")
                conv += gen + "<end_of_turn>\n<start_of_turn>user\n" + kb_content + "<end_of_turn>\n<start_of_turn>model\n"
                params2 = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_FINAL, stop=["<end_of_turn>"])
                out2 = llm.generate([conv], params2, lora_request=lora_req)
                gen2 = out2[0].outputs[0].text
                finish2 = out2[0].outputs[0].finish_reason
                all_text += gen2
                print(f"  FINAL | {len(gen2)}c | finish={finish2}")
                forced_answer = True
                break
            else:
                conv += gen + "<end_of_turn>\n<start_of_turn>user\n" + kb_content + "<end_of_turn>\n<start_of_turn>model\n"
        else:
            break

    elapsed = time.time() - t0

    visible = re.sub(r"<think>.*?</think>", "", all_text, flags=re.DOTALL).strip()
    visible = re.sub(r"<KB_QUERY>.*?</KB_QUERY>", "", visible).strip()
    visible = re.sub(r"<KB_RESULT.*?</KB_RESULT>", "", visible, flags=re.DOTALL).strip()
    sections = re.findall(r"^## (.+)$", visible, re.MULTILINE)
    sections_clean = [s.strip() for s in sections]

    matched_sections = [s for s in EXPECTED_SECTIONS if any(s.lower() in sc.lower() for sc in sections_clean)]
    has_cite = bool(re.search(
        r"case data|wikimed|who guidelines|KB_RESULT|knowledge base|according to|evidence",
        visible, re.IGNORECASE))
    natural_exit = not forced_answer and finish == "stop"

    gold_answer = extract_gold_answer(case["text"])
    gold_sections = re.findall(r"^## (.+)$", gold_answer, re.MULTILINE)

    print(f"\n  SECTIONS found: {sections_clean}")
    print(f"  SECTIONS expected: {[s.strip() for s in gold_sections]}")
    print(f"  Matched expected: {len(matched_sections)}/5")
    print(f"  KB queries: {len(kb_queries)} (gold={gold_queries})")
    print(f"  Natural exit: {natural_exit} | Time: {elapsed:.1f}s")
    print(f"  VISIBLE OUTPUT ({len(visible)} chars):")
    for line in visible[:1200].split("\n"):
        print(f"    {line}")
    if len(visible) > 1200:
        print(f"    ... [{len(visible)-1200} more chars]")

    return {
        "id": case_id,
        "turns": turn + 1 + (1 if forced_answer else 0),
        "kb_queries": kb_queries,
        "num_kb": len(kb_queries),
        "gold_queries": gold_queries,
        "sections": sections_clean,
        "matched_sections": len(matched_sections),
        "think_closed": "</think>" in all_text,
        "has_kb": len(kb_queries) > 0,
        "has_cite": has_cite,
        "natural_exit": natural_exit,
        "output_len": len(visible),
        "time": round(elapsed, 1),
        "forced_answer": forced_answer,
    }


def main():
    random.seed(SEED)
    print(f"Config: {NUM_CASES} cases, seed={SEED}, max_turns={MAX_TURNS}, max_kb={MAX_KB_QUERIES}")
    print(f"Model: {BASE_MODEL}")
    print(f"LoRA:  {LORA_CKPT}")
    print(f"Data:  {VAL_PATH}")
    print(f"KB:    {KB_URL}")

    cases = []
    with open(VAL_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") == "cds":
                cases.append(d)
    print(f"\nLoaded {len(cases)} CDS cases from val set")
    random.shuffle(cases)
    if NUM_CASES > 0:
        cases = cases[:NUM_CASES]
    print(f"Selected {len(cases)} cases for evaluation\n")

    print("Loading vLLM + LoRA...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=64,
    )
    lora_req = LoRARequest("cds_v2", 1, LORA_CKPT)
    print("vLLM ready.\n")

    results = []
    for i, case in enumerate(cases, 1):
        results.append(run_case(llm, lora_req, SamplingParams, case, i, len(cases)))

    # ─── Summary ─────────────────────────────────────────────
    n = len(results)
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY — {n} CDS Cases")
    print(f"{'='*70}")

    tc = sum(1 for r in results if r["think_closed"])
    kb = sum(1 for r in results if r["has_kb"])
    cite = sum(1 for r in results if r["has_cite"])
    nat = sum(1 for r in results if r["natural_exit"])
    has_out = sum(1 for r in results if r["output_len"] > 50)
    avg_turns = sum(r["turns"] for r in results) / n
    avg_kb = sum(r["num_kb"] for r in results) / n
    avg_time = sum(r["time"] for r in results) / n
    avg_len = sum(r["output_len"] for r in results) / n
    avg_matched = sum(r["matched_sections"] for r in results) / n
    query_match = sum(1 for r in results if r["num_kb"] == r["gold_queries"]) / n * 100

    print(f"\n  --- Protocol Compliance ---")
    print(f"  Think closed:        {tc}/{n} ({tc/n*100:.0f}%)")
    print(f"  Has KB query:        {kb}/{n} ({kb/n*100:.0f}%)")
    print(f"  KB citations in ans: {cite}/{n} ({cite/n*100:.0f}%)")
    print(f"  Natural exit:        {nat}/{n} ({nat/n*100:.0f}%)")
    print(f"  Has output (>50c):   {has_out}/{n} ({has_out/n*100:.0f}%)")

    print(f"\n  --- Quality Metrics ---")
    print(f"  Avg sections matched: {avg_matched:.1f}/5")
    print(f"  Avg output length:   {avg_len:.0f} chars")
    print(f"  Avg KB queries:      {avg_kb:.1f}")
    print(f"  KB count match gold: {query_match:.0f}%")
    print(f"  Avg turns:           {avg_turns:.1f}")
    print(f"  Avg time/case:       {avg_time:.1f}s")

    all_sections = Counter()
    for r in results:
        for s in r["sections"]:
            all_sections[s] += 1
    print(f"\n  --- Section Distribution ---")
    for s, c in sorted(all_sections.items(), key=lambda x: -x[1]):
        marker = " ✓" if any(exp.lower() in s.lower() for exp in EXPECTED_SECTIONS) else ""
        print(f"    {c:>2d}/{n}  ## {s}{marker}")

    kb_query_dist = Counter(r["num_kb"] for r in results)
    print(f"\n  --- KB Query Distribution ---")
    for k in sorted(kb_query_dist.keys()):
        print(f"    {k} queries: {kb_query_dist[k]} cases")

    print(f"\n  --- Per-Case Summary ---")
    print(f"  {'ID':>30s}  {'KB':>3s}  {'Gold':>4s}  {'Sects':>5s}  {'Len':>5s}  {'Time':>5s}  {'Exit':>6s}")
    for r in results:
        exit_str = "nat" if r["natural_exit"] else "forced"
        print(f"  {r['id']:>30s}  {r['num_kb']:>3d}  {str(r['gold_queries']):>4s}  {r['matched_sections']:>3d}/5  {r['output_len']:>5d}  {r['time']:>5.1f}s  {exit_str:>6s}")


if __name__ == "__main__":
    main()
