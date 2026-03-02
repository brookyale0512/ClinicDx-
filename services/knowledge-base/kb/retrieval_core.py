#!/usr/bin/env python3
"""
Shared retrieval core for local KB access.

This module centralizes memvid index loading and query logic so both daemon and
direct callers can share the same behavior.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kb.retrieval_core")

DEFAULT_KB_DIR = "/var/www/kbToolUseLora/kb"
KB_DIR = os.environ.get("KB_INDEX_DIR", DEFAULT_KB_DIR)
WHO_INDEX = os.path.join(KB_DIR, "who_knowledge.mv2")
WIKIMED_INDEX = os.path.join(KB_DIR, "wikimed.mv2")

SOURCE_WHO = "WHO Guidelines"
SOURCE_WIKI = "WikiMed"

MAX_K = 50
MAX_SNIPPET_CHARS = 10_000


def _extract_hits(raw: Any) -> List[Dict[str, Any]]:
    """Normalize mem.find() return shape."""
    if isinstance(raw, dict):
        hits = raw.get("hits", [])
        return hits if isinstance(hits, list) else []
    if isinstance(raw, list):
        return raw
    return []


def _normalize_hit(
    hit: Dict[str, Any], source_name: str, snippet_chars: int
) -> Optional[Dict[str, Any]]:
    """Convert raw memvid hit into a stable response shape."""
    score = hit.get("score", 0)
    content = hit.get("snippet", hit.get("frame", ""))
    if not content:
        return None

    frame_id = hit.get("frame_id")
    return {
        "score": float(score),
        "title": hit.get("title", ""),
        "content": str(content)[:snippet_chars],
        "source": source_name,
        "uri": hit.get("uri", ""),
        "frame_id": str(frame_id) if frame_id is not None else "",
    }


class KBRetriever:
    """Thread-safe retriever with lazy-loaded memvid handles."""

    def __init__(
        self,
        who_index: str = WHO_INDEX,
        wiki_index: str = WIKIMED_INDEX,
    ) -> None:
        self.who_index = who_index
        self.wiki_index = wiki_index
        self._who_mem: Any = None
        self._wiki_mem: Any = None
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        """Load indexes once (double-checked locking)."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            import memvid_sdk

            logger.info("Loading WHO index: %s", self.who_index)
            self._who_mem = memvid_sdk.use(
                "basic",
                self.who_index,
                read_only=True,
                enable_vec=False,
                enable_lex=True,
            )
            logger.info("Loading WikiMed index: %s", self.wiki_index)
            self._wiki_mem = memvid_sdk.use(
                "basic",
                self.wiki_index,
                read_only=True,
                enable_vec=False,
                enable_lex=True,
            )
            self._initialized = True
            logger.info("KB indexes loaded successfully")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._who_mem is not None and self._wiki_mem is not None

    def _search_single(
        self,
        mem: Any,
        source_name: str,
        query: str,
        k: int,
        snippet_chars: int,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        errors: List[str] = []
        best_hit: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        try:
            raw = mem.find(query, k=k, snippet_chars=snippet_chars, mode="lex")
            for item in _extract_hits(raw):
                normalized = _normalize_hit(
                    item, source_name=source_name, snippet_chars=snippet_chars
                )
                if not normalized:
                    continue
                if normalized["score"] > best_score:
                    best_score = normalized["score"]
                    best_hit = normalized
        except Exception as exc:
            logger.warning("Search error on %s: %s", source_name, exc)
            errors.append(f"{source_name}: {exc}")
        return best_hit, errors

    def search(
        self,
        query: str,
        k: int = 3,
        snippet_chars: int = 800,
        source_mode: str = "auto",
        threshold: float = 0.0,
        who_first_policy: bool = False,
        who_failover_threshold: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Query KB and return stable structured response.

        source_mode:
          - auto: WHO + WikiMed (best hit) or WHO-first with failover when enabled
          - who: WHO only
          - wiki: WikiMed only
        """
        k = max(1, min(k, MAX_K))
        snippet_chars = max(1, min(snippet_chars, MAX_SNIPPET_CHARS))

        start = time.time()
        self.initialize()

        mode = (source_mode or "auto").lower()
        errors: List[str] = []
        failover_reason: Optional[str] = None
        hit: Optional[Dict[str, Any]] = None

        if mode == "who":
            hit, errs = self._search_single(
                self._who_mem, SOURCE_WHO, query, k, snippet_chars
            )
            errors.extend(errs)
        elif mode == "wiki":
            hit, errs = self._search_single(
                self._wiki_mem, SOURCE_WIKI, query, k, snippet_chars
            )
            errors.extend(errs)
        else:
            who_hit, who_errs = self._search_single(
                self._who_mem, SOURCE_WHO, query, k, snippet_chars
            )
            errors.extend(who_errs)

            if who_first_policy:
                if who_hit and who_hit["score"] >= who_failover_threshold:
                    hit = who_hit
                else:
                    wiki_hit, wiki_errs = self._search_single(
                        self._wiki_mem, SOURCE_WIKI, query, k, snippet_chars
                    )
                    errors.extend(wiki_errs)
                    if not who_hit:
                        failover_reason = "who_no_hit"
                    elif who_hit["score"] < who_failover_threshold:
                        failover_reason = "who_low_score"
                    hit = wiki_hit if wiki_hit else who_hit
            else:
                wiki_hit, wiki_errs = self._search_single(
                    self._wiki_mem, SOURCE_WIKI, query, k, snippet_chars
                )
                errors.extend(wiki_errs)
                candidates = [h for h in (who_hit, wiki_hit) if h]
                if candidates:
                    hit = max(candidates, key=lambda item: item["score"])

        if hit and hit["score"] < threshold:
            hit = None

        latency_ms = (time.time() - start) * 1000.0
        return {
            "query": query,
            "hit": hit,
            "source_used": hit["source"] if hit else "none",
            "latency_ms": latency_ms,
            "failover_reason": failover_reason,
            "errors": errors,
        }

    def stats(self) -> Dict[str, Any]:
        """Expose KB stats for diagnostics."""
        self.initialize()
        out: Dict[str, Any] = {
            "who_loaded": self._who_mem is not None,
            "wiki_loaded": self._wiki_mem is not None,
        }
        if self._who_mem is not None:
            try:
                out["who_stats"] = self._who_mem.stats()
            except Exception:
                out["who_stats"] = {}
        if self._wiki_mem is not None:
            try:
                out["wiki_stats"] = self._wiki_mem.stats()
            except Exception:
                out["wiki_stats"] = {}
        return out
