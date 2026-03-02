"""CDS Router — multi-turn KB tool-use inference via swappable model server.

Receives a patient case prompt, runs multi-turn generation with live KB queries,
and returns the structured CDS response.

The model server URL is configurable via MODEL_SERVER_URL env var,
allowing hot-swapping between local and remote (A100) inference.
"""

import os
import re
import json
import logging
import urllib.request
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger("cds")

router = APIRouter(prefix="/cds", tags=["cds"])

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://10.128.0.4:8000")
KB_URL = os.environ.get("KB_URL", "http://10.128.0.4:4276")
MODEL_NAME = os.environ.get("MODEL_NAME", "/var/www/ClinicDx/model/medgemma_cds_think_v1")

MAX_TURNS = 4
KB_THRESHOLD = 15.0


class CDSRequest(BaseModel):
    prompt: str
    max_turns: int = MAX_TURNS


class CDSResponse(BaseModel):
    response: str
    raw_output: str
    kb_queries: list[dict]
    turns: int
    model_server: str


def _query_kb(query: str) -> Optional[dict]:
    try:
        data = json.dumps({
            "query": query, "k": 3, "source_mode": "auto",
            "snippet_chars": 1200, "threshold": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{KB_URL}/search", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        hit = result.get("hit")
        return hit if hit and hit.get("score", 0) >= KB_THRESHOLD else None
    except Exception as e:
        log.warning("KB query failed: %s", e)
        return None


def _generate(conversation: str, max_tokens: int = 4096) -> str:
    """Call vLLM OpenAI-compatible API."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": conversation,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<end_of_turn>"],
    }).encode()
    req = urllib.request.Request(
        f"{MODEL_SERVER_URL}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        return result["choices"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        log.error("vLLM error %s: %s  model=%s  prompt_len=%d",
                  e.code, body, MODEL_NAME, len(conversation))
        raise


@router.post("/generate", response_model=CDSResponse)
async def generate_cds(request: CDSRequest):
    """Multi-turn CDS generation with live KB tool-use."""
    conversation = request.prompt
    if not conversation.startswith("<bos>"):
        conversation = f"<bos>{conversation}"
    if not conversation.rstrip().endswith("<start_of_turn>model\n"):
        if "<start_of_turn>model" not in conversation:
            conversation += "<start_of_turn>model\n"

    all_text = ""
    kb_queries = []
    used_q = set()

    for turn in range(request.max_turns):
        is_last = turn == request.max_turns - 1
        tok_limit = 1500

        try:
            generated = _generate(conversation, max_tokens=tok_limit)
        except Exception as e:
            log.error("Model server error: %s", e)
            raise HTTPException(status_code=502, detail=f"Model server error: {e}")

        all_text += generated

        kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL | re.IGNORECASE)
        has_answer = bool(re.search(r"## Clinical Assessment", generated))

        if kb_matches and not has_answer and not is_last:
            raw_q = kb_matches[-1].strip()

            if raw_q.lower() in used_q:
                conversation += generated + "<end_of_turn>\n"
                conversation += (
                    "<start_of_turn>user\nDuplicate query. "
                    "Continue your analysis.<end_of_turn>\n<start_of_turn>model\n"
                )
                continue

            used_q.add(raw_q.lower())
            kb = _query_kb(raw_q)
            kb_entry = {"query": raw_q, "score": kb["score"] if kb else 0,
                        "source": kb["source"] if kb else "none"}
            kb_queries.append(kb_entry)

            if kb:
                kb_tag = (
                    f'<KB_RESULT source="{kb["source"]}" score="{kb["score"]:.1f}">\n'
                    f'{kb["content"]}\n</KB_RESULT>'
                )
            else:
                kb_tag = (
                    f'<KB_RESULT source="none" score="0">\n'
                    f'No KB evidence found for: {raw_q}\n</KB_RESULT>'
                )

            conversation += generated + "<end_of_turn>\n"
            conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"
        else:
            break

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", all_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<KB_QUERY>[\s\S]*?</KB_QUERY>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"QUERY_ESTIMATE:.*", "", cleaned)
    cleaned = re.sub(r"DECISION:\s*\w+", "", cleaned)
    cleaned = re.sub(r"NEXT_QUERY:.*", "", cleaned)
    cleaned = re.sub(r"CASE_COMPLEXITY:.*", "", cleaned)

    sections = list(re.finditer(r"^## .+$", cleaned, re.MULTILINE))
    if sections:
        seen_titles = set()
        deduped_parts = []
        for i, m in enumerate(sections):
            title = m.group(0).strip().lower()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            start = m.start()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(cleaned)
            deduped_parts.append(cleaned[start:end].strip())
        visible = "\n\n".join(deduped_parts).strip()
    else:
        visible = cleaned.strip()

    if not visible:
        think_blocks = re.findall(r"<think>([\s\S]*?)</think>", all_text, flags=re.IGNORECASE)
        if think_blocks:
            last = think_blocks[-1]
            md = re.search(r"(## [\s\S]+)", last)
            visible = md.group(1) if md else last

    visible = (visible or "").strip()

    return CDSResponse(
        response=visible,
        raw_output=all_text,
        kb_queries=kb_queries,
        turns=turn + 1,
        model_server=MODEL_SERVER_URL,
    )


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_generate(conversation: str, max_tokens: int = 2000):
    """Call vLLM with streaming, yield tokens as they arrive."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": conversation,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<end_of_turn>"],
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{MODEL_SERVER_URL}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    accumulated = ""
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                evt = json.loads(body)
            except json.JSONDecodeError:
                continue
            token = evt.get("choices", [{}])[0].get("text", "")
            if token:
                accumulated += token
                yield token
    return accumulated


@router.post("/generate_stream")
async def generate_cds_stream(request: CDSRequest):
    """Multi-turn CDS with real-time token streaming via SSE."""

    async def event_stream() -> AsyncGenerator[str, None]:
        conversation = request.prompt
        if not conversation.startswith("<bos>"):
            conversation = f"<bos>{conversation}"
        if not conversation.rstrip().endswith("<start_of_turn>model\n"):
            if "<start_of_turn>model" not in conversation:
                conversation += "<start_of_turn>model\n"

        kb_queries = []
        used_q = set()

        for turn in range(request.max_turns):
            is_last = turn == request.max_turns - 1

            yield _sse_event({"type": "turn_start", "turn": turn + 1})

            generated = ""
            try:
                for token in _stream_generate(conversation, max_tokens=2000):
                    generated += token
                    yield _sse_event({"type": "token", "text": token})
            except Exception as e:
                yield _sse_event({"type": "error", "message": str(e)})
                return

            kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL | re.IGNORECASE)
            has_answer = bool(re.search(r"## Clinical Assessment", generated))

            if kb_matches and not has_answer and not is_last:
                raw_q = kb_matches[-1].strip()

                if raw_q.lower() in used_q:
                    conversation += generated + "<end_of_turn>\n"
                    conversation += (
                        "<start_of_turn>user\nDuplicate query. "
                        "Continue your analysis.<end_of_turn>\n<start_of_turn>model\n"
                    )
                    yield _sse_event({"type": "kb_duplicate", "query": raw_q})
                    continue

                used_q.add(raw_q.lower())
                yield _sse_event({"type": "kb_query", "query": raw_q})

                kb = _query_kb(raw_q)
                kb_entry = {
                    "query": raw_q,
                    "score": kb["score"] if kb else 0,
                    "source": kb["source"] if kb else "none",
                }
                kb_queries.append(kb_entry)

                yield _sse_event({
                    "type": "kb_result",
                    "query": raw_q,
                    "source": kb["source"] if kb else "none",
                    "score": kb["score"] if kb else 0,
                })

                if kb:
                    kb_tag = (
                        f'<KB_RESULT source="{kb["source"]}" score="{kb["score"]:.1f}">\n'
                        f'{kb["content"]}\n</KB_RESULT>'
                    )
                else:
                    kb_tag = (
                        f'<KB_RESULT source="none" score="0">\n'
                        f'No KB evidence found for: {raw_q}\n</KB_RESULT>'
                    )

                conversation += generated + "<end_of_turn>\n"
                conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"
            else:
                break

        yield _sse_event({
            "type": "done",
            "turns": turn + 1,
            "kb_queries": kb_queries,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def cds_health():
    """Check model server and KB connectivity."""
    model_ok = False
    kb_ok = False

    try:
        req = urllib.request.Request(f"{MODEL_SERVER_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            model_ok = True
    except Exception:
        pass

    try:
        kb = _query_kb("malaria treatment")
        kb_ok = kb is not None
    except Exception:
        pass

    return {
        "status": "ok" if model_ok and kb_ok else "degraded",
        "model_server": {"url": MODEL_SERVER_URL, "ok": model_ok},
        "kb": {"url": KB_URL, "ok": kb_ok},
    }
