#!/usr/bin/env python3
"""
Local KB retrieval daemon (HTTP + JSON).

Endpoints:
  GET  /health
  GET  /stats
  POST /search
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

try:
    from kb.retrieval_core import KBRetriever
except ImportError:
    from retrieval_core import KBRetriever

LOGGER = logging.getLogger("kb-daemon")
RETRIEVER: KBRetriever
CONFIG: Dict[str, Any]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4276


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
            _json_response(self, 200, {"ok": True})
            return
        if self.path == "/stats":
            try:
                _json_response(self, 200, {"ok": True, "stats": RETRIEVER.stats()})
            except Exception as exc:  # pragma: no cover
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
        source_mode = str(body.get("source_mode", CONFIG["source_mode"]))
        threshold = float(body.get("threshold", CONFIG["threshold"]))
        who_first_policy = bool(body.get("who_first_policy", CONFIG["who_first_policy"]))
        who_failover_threshold = float(
            body.get("who_failover_threshold", CONFIG["who_failover_threshold"])
        )

        result = RETRIEVER.search(
            query=query,
            k=k,
            snippet_chars=snippet_chars,
            source_mode=source_mode,
            threshold=threshold,
            who_first_policy=who_first_policy,
            who_failover_threshold=who_failover_threshold,
        )
        _json_response(self, 200, {"ok": True, **result})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    global RETRIEVER, CONFIG

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    CONFIG = {
        "k": 3,
        "snippet_chars": 800,
        "source_mode": "auto",
        "threshold": 0.0,
        "who_first_policy": False,
        "who_failover_threshold": 5.0,
    }
    RETRIEVER = KBRetriever()
    RETRIEVER.initialize()

    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer((DEFAULT_HOST, port), KBHandler)
    LOGGER.info("KB daemon listening on http://%s:%d", DEFAULT_HOST, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
