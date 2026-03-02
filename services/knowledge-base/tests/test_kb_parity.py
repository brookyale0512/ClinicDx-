#!/usr/bin/env python3
"""Parity checks for direct memvid retrieval vs local KB daemon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kb.client import daemon_health, query_kb_http  # noqa: E402
from kb.retrieval_core import KBRetriever  # noqa: E402

QUERIES = [
    "severe malaria artesunate",
    "postpartum haemorrhage oxytocin",
    "convulsions magnesium sulfate",
    "neonatal jaundice phototherapy",
    "pneumonia amoxicillin",
]


def compare_hits(direct: Dict, daemon: Dict) -> Dict:
    return {
        "direct_source": direct.get("source", "none") if direct else "none",
        "daemon_source": daemon.get("source", "none") if daemon else "none",
        "direct_has_content": bool(direct and direct.get("content")),
        "daemon_has_content": bool(daemon and daemon.get("content")),
        "source_match": bool(direct and daemon and direct.get("source") == daemon.get("source")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="KB parity test")
    parser.add_argument("--daemon-url", default="http://127.0.0.1:4276")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--source-mode", choices=["auto", "who", "wiki"], default="auto")
    parser.add_argument("--strict-source-match", action="store_true")
    args = parser.parse_args()

    if not daemon_health(args.daemon_url):
        print(f"FAIL: daemon health check failed at {args.daemon_url}")
        return 2

    retriever = KBRetriever()
    retriever.initialize()

    failures: List[str] = []
    report = []

    for query in QUERIES:
        direct_result = retriever.search(
            query=query,
            k=args.k,
            source_mode=args.source_mode,
            snippet_chars=800,
            threshold=0.0,
            who_first_policy=False,
        )
        daemon_hit = query_kb_http(
            query=query,
            daemon_url=args.daemon_url,
            k=args.k,
            source_mode=args.source_mode,
            snippet_chars=800,
            threshold=0.0,
            who_first_policy=False,
        )
        direct_hit = direct_result.get("hit")

        cmp_out = compare_hits(direct_hit, daemon_hit)
        row = {"query": query, **cmp_out}
        report.append(row)

        if not cmp_out["direct_has_content"] or not cmp_out["daemon_has_content"]:
            failures.append(f"{query}: missing content in direct/daemon result")
        if args.strict_source_match and not cmp_out["source_match"]:
            failures.append(
                f"{query}: source mismatch (direct={cmp_out['direct_source']}, daemon={cmp_out['daemon_source']})"
            )

    print(json.dumps({"report": report, "failures": failures}, indent=2))
    if failures:
        print(f"FAIL: {len(failures)} parity checks failed")
        return 1
    print("PASS: parity checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
