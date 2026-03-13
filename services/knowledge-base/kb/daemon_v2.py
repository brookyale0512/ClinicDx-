#!/usr/bin/env python3
"""
KB retrieval daemon — v2 index variant (port 4278).

Identical to daemon.py but points to who_knowledge_vec_v2.mv2 (83K enriched
v2 chunks, built with the custom _lib / memvid-pyo3-shim 0.1.0).

Launch:
    cd /var/www/kbToolUseLora && python3 -m kb.daemon_v2
    # or with explicit port:
    cd /var/www/kbToolUseLora && python3 -m kb.daemon_v2 4278
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

try:
    from kb.retrieval_core_v2 import KBRetriever
except ImportError:
    from retrieval_core_v2 import KBRetriever  # type: ignore[no-redef]

LOGGER = logging.getLogger("kb-daemon-v2")
RETRIEVER: KBRetriever
CONFIG: Dict[str, Any]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4278
VEC_INDEX_V2 = "/var/www/kbToolUseLora/kb/who_knowledge_vec_v2.mv2"


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class KBHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {"ok": True, "index": "v2"})
            return
        if self.path == "/stats":
            try:
                _json_response(self, 200, {"ok": True, "index": "v2", "stats": RETRIEVER.stats()})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/search":
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, 400, {"ok": False, "error": "invalid_json"})
            return

        query = (body.get("query") or "").strip()
        if not query:
            _json_response(self, 400, {"ok": False, "error": "query_required"})
            return

        k = int(body.get("k", CONFIG["k"]))
        snippet_chars = int(body.get("snippet_chars", CONFIG["snippet_chars"]))
        threshold = float(body.get("threshold", CONFIG["threshold"]))
        search_mode = str(body.get("search_mode", CONFIG["search_mode"]))
        safe_top1_guardrail = bool(body.get("safe_top1_guardrail", CONFIG["safe_top1_guardrail"]))

        result = RETRIEVER.search(
            query=query,
            k=k,
            snippet_chars=snippet_chars,
            threshold=threshold,
            search_mode=search_mode,
            safe_top1_guardrail=safe_top1_guardrail,
        )
        _json_response(self, 200, {"ok": True, **result})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    global RETRIEVER, CONFIG

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    CONFIG = {
        "k": 5,
        "snippet_chars": 15000,
        "threshold": 0.0,
        "search_mode": "rrf",
        "safe_top1_guardrail": False,
    }
    RETRIEVER = KBRetriever(who_index=VEC_INDEX_V2)
    RETRIEVER.initialize(enable_vec=True)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer((DEFAULT_HOST, port), KBHandler)
    LOGGER.info("KB daemon v2 listening on http://%s:%d  (index: %s)", DEFAULT_HOST, port, VEC_INDEX_V2)
    server.serve_forever()


if __name__ == "__main__":
    main()
