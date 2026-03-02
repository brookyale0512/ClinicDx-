#!/usr/bin/env python3
"""
CDS KB Tool-Use LoRA — MedQA Evaluation
=========================================
Full 1273-question MedQA eval with multi-turn KB tool use.

Key fixes from test run:
- NO 2-query nudge (model decides when it has enough evidence)
- MAX_TURNS = 6 (matches training data distribution)
- snippet_chars=1200 (matches training data)
- Full root cause classification

Usage:
    python3 eval_medqa.py --model /path/to/merged_model --max 1273
"""

import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

KB_URL = "http://127.0.0.1:4276"
RESULTS_DIR = Path(__file__).parent / "results"

GARBAGE_WORDS = {
    "to", "on", "the", "a", "an", "and", "or", "but", "in", "of", "for",
    "with", "is", "it", "be", "at", "by", "from", "this", "that", "its",
    "per", "as", "query", "kb", "must", "information", "using", "based",
    "following", "context", "case", "treatment", "management", "guideline",
    "protocol", "clinical", "diagnosis", "therapy", "approach", "patient",
    "question", "asks", "scenario", "check", "most", "dose",
}


def setup_log(run_id):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    lg = logging.getLogger("eval")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    log_path = RESULTS_DIR / f"eval_{run_id}.log"
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    for n in ("vllm", "httpx", "httpcore", "transformers", "datasets"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return lg, log_path


def query_kb(query, threshold=25.0, snippet_chars=1200):
    import urllib.request

    try:
        data = json.dumps({
            "query": query, "k": 3, "source_mode": "auto",
            "snippet_chars": snippet_chars, "threshold": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{KB_URL}/search", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        hit = result.get("hit")
        return hit if hit and hit.get("score", 0) >= threshold else None
    except Exception:
        return None


def assess_query(q):
    words = q.lower().split()
    content_words = [w for w in words if w not in GARBAGE_WORDS and len(w) > 1]
    if len(content_words) < 1 or len(q) > 60:
        return "GARBAGE"
    if len(content_words) < 2:
        return "WEAK"
    if len(words) > 4:
        return "WEAK"
    return "GOOD"


def clean_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def run_mcq(llm, item, logger, q_num, total):
    from vllm import SamplingParams

    q = item["question"]
    opts = item["options"]
    otxt = "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()))
    exp = item["answer_idx"]
    qid = item.get("id", "?")

    logger.info("\n" + "=" * 70)
    logger.info("  Q %d/%d  |  %s  |  Expected: %s", q_num, total, qid, exp)
    logger.info("=" * 70)

    conversation = f"<bos><start_of_turn>user\n{q}\n\n{otxt}<end_of_turn>\n<start_of_turn>model\n"

    MAX_TURNS = 6
    turns = []
    kb_queries = []
    kb_results = []
    used_q = set()
    query_assessments = []

    for turn in range(MAX_TURNS):
        is_last = turn == MAX_TURNS - 1
        params = SamplingParams(temperature=0.0, max_tokens=8192, stop=["<end_of_turn>"])
        out = llm.generate([conversation], params)
        generated = out[0].outputs[0].text
        turns.append(generated)

        think = clean_think(generated)
        if think:
            label = "REASONING" if turn == 0 else f"GATE (turn {turn + 1})"
            logger.info("  [%s] %s", label, think[:300])

        kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL)

        if kb_matches and not is_last:
            raw_q = kb_matches[-1].strip()
            q_quality = assess_query(raw_q)

            if raw_q.lower() in used_q:
                conversation += generated + "<end_of_turn>\n"
                conversation += (
                    "<start_of_turn>user\nYou already queried for that. "
                    "Write your final answer now: ## Assessment, ## Analysis, "
                    "Final Answer: [letter]<end_of_turn>\n<start_of_turn>model\n"
                )
                continue

            used_q.add(raw_q.lower())
            kb_queries.append(raw_q)
            query_assessments.append(q_quality)

            logger.info("  KB_QUERY %d: \"%s\" (%s)", len(kb_queries), raw_q, q_quality)

            kb = query_kb(raw_q)
            if kb:
                kb_results.append(kb)
                logger.info("    Score: %.1f | %s", kb["score"], kb["source"])
                kb_tag = (
                    f'<KB_RESULT source="{kb["source"]}" score="{kb["score"]:.1f}">\n'
                    f'{kb["content"]}\n</KB_RESULT>'
                )
            else:
                words = raw_q.split()
                fb = " ".join(words[:2]) if len(words) >= 2 else raw_q
                kb_fb = query_kb(fb) if fb.lower() != raw_q.lower() else None
                if kb_fb:
                    kb_results.append(kb_fb)
                    logger.info("    MISS -> fallback \"%s\" score=%.1f", fb, kb_fb["score"])
                    kb_tag = (
                        f'<KB_RESULT source="{kb_fb["source"]}" score="{kb_fb["score"]:.1f}">\n'
                        f'{kb_fb["content"]}\n</KB_RESULT>'
                    )
                else:
                    kb_results.append(None)
                    logger.info("    TOTAL MISS")
                    kb_tag = f'<KB_RESULT source="none" score="0">\nNo KB evidence found for: {raw_q}\n</KB_RESULT>'

            # NO NUDGE — let the model decide when it has enough evidence
            conversation += generated + "<end_of_turn>\n"
            conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"
        else:
            conversation += generated
            break

    full = "".join(turns)
    fa_m = re.search(r"Final Answer:\s*([A-E])", full)
    pred = fa_m.group(1) if fa_m else None
    correct = pred == exp

    scores = [kb["score"] if kb else 0 for kb in kb_results]
    all_miss = all(s < 25 for s in scores) if scores else True

    if correct:
        cause = "CORRECT"
    elif pred is None:
        cause = "UNPARSED" + (" + KB_MISS" if all_miss else " + HAD_EVIDENCE")
    elif all_miss:
        cause = "KB_MISS"
    else:
        cause = "REASONING_ERROR"

    symbol = "+" if correct else "-"
    logger.info("  [%s] predicted=%s expected=%s queries=%d cause=%s",
                symbol, pred or "?", exp, len(kb_queries), cause)

    return {
        "id": qid, "question": q[:100], "expected": exp, "predicted": pred,
        "correct": correct, "cause": cause,
        "kb_queries": kb_queries, "kb_scores": scores,
        "query_quality": query_assessments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1273)
    parser.add_argument("--model", default="/var/www/ClinicDx/model/medgemma_cds_kb")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_log(run_id)

    logger.info("CDS KB LoRA — MedQA Evaluation")
    logger.info("Model: %s", args.model)
    logger.info("Max questions: %d", args.max)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, gpu_memory_utilization=0.85)

    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    items = list(ds.select(range(min(args.max, len(ds)))))

    logger.info("Loaded %d MedQA questions", len(items))

    results = []
    correct = 0
    t0 = time.time()

    for i, item in enumerate(items):
        r = run_mcq(llm, item, logger, i + 1, len(items))
        results.append(r)
        if r["correct"]:
            correct += 1

        if (i + 1) % 10 == 0:
            acc = correct / (i + 1) * 100
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(items) - i - 1)
            logger.info("  RUNNING: %d/%d = %.1f%% | ETA: %s",
                        correct, i + 1, acc,
                        str(datetime.utcfromtimestamp(eta).strftime("%H:%M:%S")))

    from collections import Counter
    causes = Counter(r["cause"] for r in results)
    unparsed = sum(1 for r in results if r["predicted"] is None)
    accuracy = correct / len(results) * 100
    parseable = len(results) - unparsed
    acc_ex_unparsed = correct / parseable * 100 if parseable else 0

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS: %d questions", len(results))
    logger.info("=" * 70)
    logger.info("  Accuracy:           %.1f%% (%d/%d)", accuracy, correct, len(results))
    logger.info("  Accuracy (parsed):  %.1f%% (%d/%d)", acc_ex_unparsed, correct, parseable)
    logger.info("  Unparsed:           %d (%.1f%%)", unparsed, unparsed / len(results) * 100)
    logger.info("  Failure breakdown:")
    for c, n in causes.most_common():
        logger.info("    %3d  %s", n, c)

    summary = {
        "model": args.model, "n": len(results),
        "correct": correct, "accuracy": round(accuracy, 1),
        "accuracy_parsed": round(acc_ex_unparsed, 1),
        "unparsed": unparsed, "baseline": 64.2,
        "failure_causes": dict(causes),
        "query_quality": dict(Counter(
            q for r in results for q in r.get("query_quality", [])
        )),
        "results": results,
    }

    out_path = RESULTS_DIR / f"eval_{run_id}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results: %s", out_path)

    errors = [r for r in results if not r["correct"] and r["cause"] != "CORRECT"]
    err_path = RESULTS_DIR / f"eval_{run_id}_errors.json"
    with open(err_path, "w") as f:
        json.dump(errors, f, indent=2)
    logger.info("Errors:  %s", err_path)


if __name__ == "__main__":
    main()
