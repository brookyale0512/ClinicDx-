#!/usr/bin/env python3
"""HTTP client helpers for local KB daemon."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_DAEMON_URL = "http://127.0.0.1:4276"


def query_kb_http(
    query: str,
    daemon_url: str = DEFAULT_DAEMON_URL,
    k: int = 3,
    source_mode: str = "auto",
    snippet_chars: int = 800,
    threshold: float = 0.0,
    who_first_policy: bool = False,
    who_failover_threshold: float = 5.0,
    timeout: float = 10.0,
    retries: int = 1,
    return_full: bool = False,
) -> Optional[Dict[str, Any]]:
    """Query KB daemon.

    By default returns the best `hit` for backward compatibility.
    Set return_full=True to get the full response (including `hits` list).
    """
    payload = {
        "query": query,
        "k": k,
        "source_mode": source_mode,
        "snippet_chars": snippet_chars,
        "threshold": threshold,
        "who_first_policy": who_first_policy,
        "who_failover_threshold": who_failover_threshold,
    }
    body = json.dumps(payload).encode("utf-8")
    endpoint = daemon_url.rstrip("/") + "/search"

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return data if return_full else data.get("hit")
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))

    if last_exc:
        raise RuntimeError(f"KB daemon query failed: {last_exc}") from last_exc
    return None


def daemon_health(daemon_url: str = DEFAULT_DAEMON_URL, timeout: float = 3.0) -> bool:
    """Health check helper."""
    endpoint = daemon_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("ok"))
    except Exception:
        return False
