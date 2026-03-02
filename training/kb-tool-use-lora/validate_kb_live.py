#!/usr/bin/env python3
"""
Validate 2-Query CDS Model with Live KB Access
================================================
Loads the freshly trained LoRA adapter via vLLM, runs 10 CDS cases from the
validation set with real KB lookups, and prints every token of model output
so you can follow the full multi-turn interaction live.

Expected behavior per case:
  Turn 1: <think>...DECISION: QUERY_MORE</think><KB_QUERY>term1</KB_QUERY>
  Turn 2: <think>...DECISION: QUERY_MORE</think><KB_QUERY>term2</KB_QUERY>
  Turn 3: <think>...DECISION: WRITE_ANSWER</think> + 5-section clinical response
"""

import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

KB_URL = "http://10.128.0.4:4276"
BASE_MODEL = "/var/www/ClinicDx/model/medgemma"
LORA_PATH = "/var/www/ClinicDx/training/portable_2query/checkpoints"
VAL_FILE = Path(__file__).parent / "val_2query.jsonl"
N_CASES = 10
MAX_TURNS = 5

LOG_FILE = Path(__file__).parent / f"validate_kb_live_{datetime.now():%Y%m%d_%H%M%S}.log"


class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()


def ts():
    return datetime.now().strftime("%H:%M:%S")


def query_kb(query_text):
    """Call the KB search API and return the top hit (or None)."""
    try:
        payload = json.dumps({
            "query": query_text, "k": 3, "source_mode": "auto",
            "snippet_chars": 1200, "threshold": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{KB_URL}/search", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        hit = result.get("hit")
        if hit and hit.get("score", 0) >= 20.0:
            return hit
    except Exception as e:
        print(f"  [KB ERROR] {e}", flush=True)
    return None


def load_cds_cases(path, n):
    """Load the first N CDS cases (with KB_QUERY tags) from the val JSONL."""
    cases = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("source") == "cds" and "<KB_QUERY>" in rec.get("text", ""):
                text = rec["text"]
                first_eot = text.find("<end_of_turn>")
                if first_eot < 0:
                    continue
                first_user_turn = text[text.find("<start_of_turn>user"):first_eot + len("<end_of_turn>")]
                expected_queries = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", text)
                cases.append({
                    "id": rec.get("id", "?"),
                    "first_user_turn": first_user_turn,
                    "full_text": text,
                    "expected_queries": expected_queries,
                })
                if len(cases) >= n:
                    break
    return cases


def extract_case_summary(user_turn):
    """Pull age, gender, and a short snippet from the case XML."""
    age = re.search(r"<age>(.*?)</age>", user_turn)
    gender = re.search(r"<gender>(.*?)</gender>", user_turn)
    chief = re.search(r"<chief_complaint>(.*?)</chief_complaint>", user_turn, re.DOTALL)
    parts = []
    if age:
        parts.append(f"age={age.group(1)}")
    if gender:
        parts.append(f"gender={gender.group(1)}")
    if chief:
        cc = chief.group(1).strip()[:120]
        parts.append(f"CC: {cc}")
    return ", ".join(parts) if parts else "(no summary)"


def print_streaming(text, char_delay=0.003):
    """Print text character-by-character to simulate streaming."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if char_delay > 0:
            time.sleep(char_delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_case(llm, lora_req, case, case_num):
    """Run a single CDS case through the multi-turn KB loop."""
    case_id = case["id"]
    user_turn = case["first_user_turn"]
    summary = extract_case_summary(user_turn)

    print(f"\n{'#' * 100}", flush=True)
    print(f"# CASE {case_num}/{N_CASES}: {case_id}", flush=True)
    print(f"# {summary}", flush=True)
    print(f"# Expected KB queries: {case['expected_queries']}", flush=True)
    print(f"{'#' * 100}\n", flush=True)

    conversation = f"<bos>{user_turn}\n<start_of_turn>model\n"

    actual_queries = []
    final_answer_written = False
    sections_found = []

    for turn in range(MAX_TURNS):
        is_last = turn >= MAX_TURNS - 1
        tok_limit = 2048 if turn < 2 else 4096

        params = SamplingParams(
            temperature=0.0,
            max_tokens=tok_limit,
            stop=["<end_of_turn>"],
        )
        t0 = time.time()
        outputs = llm.generate([conversation], params, lora_request=lora_req)
        gen_time = time.time() - t0
        generated = outputs[0].outputs[0].text
        finish = outputs[0].outputs[0].finish_reason
        n_tokens = len(outputs[0].outputs[0].token_ids)

        print(f"  [{ts()}] ── TURN {turn + 1} ── ({n_tokens} tokens, {gen_time:.1f}s, finish={finish})", flush=True)
        print(f"  {'─' * 90}", flush=True)

        print_streaming(generated, char_delay=0.002)

        print(f"  {'─' * 90}", flush=True)

        kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL)
        has_write_answer = bool(re.search(r"DECISION:\s*WRITE_ANSWER", generated))
        has_query_more = bool(re.search(r"DECISION:\s*QUERY_MORE", generated))

        for section in ["Clinical Assessment", "Evidence-Based Considerations",
                        "Suggested Actions", "Safety Alerts", "Key Points"]:
            if f"## {section}" in generated:
                sections_found.append(section)

        if kb_matches:
            actual_queries.extend(kb_matches)

        print(f"  KB_QUERY found: {kb_matches}", flush=True)
        print(f"  DECISION: {'QUERY_MORE' if has_query_more else 'WRITE_ANSWER' if has_write_answer else 'NONE'}", flush=True)

        if kb_matches and not is_last and not has_write_answer:
            raw_q = kb_matches[-1].strip()
            print(f"\n  [{ts()}] Querying KB: \"{raw_q}\"...", flush=True)
            kb_hit = query_kb(raw_q)

            if kb_hit:
                print(f"  [{ts()}] KB HIT: source={kb_hit['source']}, score={kb_hit['score']:.1f}, "
                      f"title=\"{kb_hit.get('title', '?')[:60]}\"", flush=True)
                snippet = kb_hit["content"][:800]
                print(f"  [{ts()}] KB snippet: {snippet[:200]}...\n", flush=True)
                kb_tag = (f'<KB_RESULT source="{kb_hit["source"]}" score="{kb_hit["score"]:.1f}">\n'
                          f'{kb_hit["content"]}\n</KB_RESULT>')
            else:
                print(f"  [{ts()}] KB MISS for \"{raw_q}\"\n", flush=True)
                kb_tag = f'<KB_RESULT source="none" score="0">\nNo KB evidence found for: {raw_q}\n</KB_RESULT>'

            conversation += generated + "<end_of_turn>\n"
            conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"
        else:
            conversation += generated
            final_answer_written = has_write_answer or len(sections_found) >= 3
            break

    print(f"\n  {'=' * 90}", flush=True)
    print(f"  CASE {case_num} SUMMARY: {case_id}", flush=True)
    print(f"  {'=' * 90}", flush=True)
    print(f"  KB queries made:    {len(actual_queries)}  {actual_queries}", flush=True)
    print(f"  Expected queries:   {len(case['expected_queries'])}  {case['expected_queries']}", flush=True)
    print(f"  Sections found:     {sections_found}", flush=True)
    print(f"  Final answer:       {'YES' if final_answer_written else 'NO'}", flush=True)

    used_2_queries = len(actual_queries) >= 2
    wrote_answer = final_answer_written
    print(f"  2-query pattern:    {'PASS' if used_2_queries else 'FAIL'}", flush=True)
    print(f"  Structured answer:  {'PASS' if wrote_answer else 'FAIL'}", flush=True)

    return {
        "id": case_id,
        "n_queries": len(actual_queries),
        "queries": actual_queries,
        "sections": sections_found,
        "final_answer": final_answer_written,
        "pass_2query": used_2_queries,
        "pass_answer": wrote_answer,
    }


def main():
    sys.stdout = TeeWriter(LOG_FILE)

    print(f"{'=' * 100}", flush=True)
    print(f"  2-Query CDS Validation with Live KB Access", flush=True)
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"  Base model:  {BASE_MODEL}", flush=True)
    print(f"  LoRA:        {LORA_PATH}", flush=True)
    print(f"  KB:          {KB_URL}", flush=True)
    print(f"  Cases:       {N_CASES}", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    print(f"[{ts()}] Loading {N_CASES} CDS cases from {VAL_FILE}...", flush=True)
    cases = load_cds_cases(VAL_FILE, N_CASES)
    print(f"[{ts()}] Loaded {len(cases)} cases\n", flush=True)

    print(f"[{ts()}] Checking KB health...", flush=True)
    kb_test = query_kb("malaria treatment")
    if kb_test:
        print(f"[{ts()}] KB OK: score={kb_test['score']:.1f}, source={kb_test['source']}\n", flush=True)
    else:
        print(f"[{ts()}] WARNING: KB returned no results for test query\n", flush=True)

    print(f"[{ts()}] Loading vLLM with LoRA...", flush=True)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    lora_req = LoRARequest("cds_2query", 1, LORA_PATH)
    print(f"[{ts()}] vLLM + LoRA loaded\n", flush=True)

    results = []
    for i, case in enumerate(cases):
        result = run_case(llm, lora_req, case, i + 1)
        results.append(result)

    print(f"\n\n{'=' * 100}", flush=True)
    print(f"  FINAL RESULTS — {len(results)} cases", flush=True)
    print(f"{'=' * 100}", flush=True)

    pass_2q = sum(1 for r in results if r["pass_2query"])
    pass_ans = sum(1 for r in results if r["pass_answer"])
    pass_both = sum(1 for r in results if r["pass_2query"] and r["pass_answer"])

    print(f"  2-query pattern:    {pass_2q}/{len(results)} passed", flush=True)
    print(f"  Structured answer:  {pass_ans}/{len(results)} passed", flush=True)
    print(f"  Both correct:       {pass_both}/{len(results)} passed", flush=True)
    print(f"", flush=True)

    for r in results:
        status = "PASS" if r["pass_2query"] and r["pass_answer"] else "FAIL"
        print(f"  [{status}] {r['id']:35s}  queries={r['n_queries']}  sections={len(r['sections'])}  "
              f"answer={'Y' if r['final_answer'] else 'N'}", flush=True)

    print(f"\n  Log saved to: {LOG_FILE}", flush=True)

    summary_path = LOG_FILE.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({"results": results, "pass_2query": pass_2q, "pass_answer": pass_ans,
                    "pass_both": pass_both, "total": len(results)}, f, indent=2)
    print(f"  Summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
