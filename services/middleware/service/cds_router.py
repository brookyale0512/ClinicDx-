"""CDS Router — multi-turn KB tool-use inference via swappable model server.

Receives a patient case prompt, runs multi-turn generation with live KB queries,
and returns the structured CDS response.

The model server URL is configurable via MODEL_SERVER_URL env var,
allowing hot-swapping between local and remote (A100) inference.
"""

import asyncio
import concurrent.futures
import os
import re
import json
import logging
import urllib.request
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Thread pool for blocking urllib calls — keeps the async event loop responsive
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="cds-gen")

log = logging.getLogger("cds")

router = APIRouter(prefix="/cds", tags=["cds"])

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8180")
KB_URL = os.environ.get("KB_URL", "http://127.0.0.1:4276")
MODEL_NAME = os.environ.get("MODEL_NAME", "clinicdx-v1-q8.gguf")

MAX_TURNS = 5
MAX_KB_QUERIES = 5
KB_THRESHOLD = 0.0

# ── System prompt rules injected at inference time ───────────────────────────

DEPTH_RULES = """\
RESPONSE DEPTH RULES — apply to every section of the clinical assessment:

## Alert Level
- State the level (NORMAL / MODERATE / HIGH / CRITICAL) and one precise clinical reason using actual patient values.
- Example: "CRITICAL — Hb 4.4 g/dL with BP 95/62 mmHg and obtunded consciousness indicating hemodynamic instability."

## Clinical Assessment
- Open with 1-2 sentences orienting the clinical picture using specific patient data (age, vitals, labs, timeline).
- For EACH piece of KB evidence retrieved: state what the guideline says, then explicitly connect it to THIS patient's values.
  Example: "Per [WHO: HTN p12]: combination therapy required when BP >140/90 after 4 weeks monotherapy. This patient has BP 152/95 mmHg after 3 months on enalapril alone — meets escalation criteria."
- Note any evidence gaps: if KB did not cover a specific aspect, say so and fall back to clinical reasoning.

## Differential Considerations
- List 3 differentials minimum, each as: **Diagnosis name** — one sentence of patient-specific reasoning. Evidence: [source].
- For the top differential: explain why it fits (supporting findings) AND what distinguishes it from the others.
- Do not use generic fillers — every differential must reference a specific finding from this patient's chart.

## Recommended Actions
- Number each action. For every drug or intervention include: name, dose, route, frequency, duration.
  Example: "Ceftriaxone IV 1g once daily for 5 days, then switch to oral amoxicillin 500mg three times daily to complete 7 days total. Per [WHO: CHAPTER 3]."
- Include a monitoring instruction for each action (what to watch, how often, what triggers escalation).
- If more than one problem was identified, give at least one action per problem.

## Safety Alerts
- List at least 2 specific safety concerns relevant to THIS patient (not generic warnings).
- For each: state the risk, the threshold that triggers action, and what to do.
  Example: "Digoxin toxicity: patient on digoxin with renal impairment (dry mucosa, reduced turgor). Stop digoxin if HR <60 bpm or nausea/visual changes appear. Check serum K+ before next dose."

## Key Points
- 3-4 bullet points summarising the most important take-aways for the treating clinician.
- Include: the single most time-critical action, one monitoring/follow-up instruction, and one safety reminder.
- End with a return/escalation criterion: "Return immediately if [specific sign]."
"""

KB_SEARCH_RULES = """\
KB QUERY RULES — follow exactly when issuing <KB_QUERY> tags:

SHAPE: [condition] [drug/procedure] [treatment/dose/protocol]
GOOD: "severe malaria artesunate dose" | "neonatal sepsis ampicillin treatment" | "eclampsia magnesium sulfate protocol"
BAD:  "management" alone | "assessment" alone | "guidelines" alone | full sentences

HOW MANY QUERIES:
- Plan queries BEFORE issuing them: identify each distinct clinical problem in the case.
- Issue ONE query per distinct clinical problem (typical case: 2-3 queries).
- Example for malaria with vomiting: query 1 = "malaria artesunate dose adolescent", query 2 = "malaria severe criteria vomiting treatment" — two different problems.
- Example for DM+HTN: query 1 = "hypertension diabetes treatment", query 2 = "hypertension ACE inhibitor dose", query 3 = "diabetic nephropathy renal monitoring".
- Do NOT issue a second query on a topic you already have a KB hit for.
- Do NOT issue more than one query per problem — synonyms are wasteful. Move on.
- After all problems are covered, write the answer immediately. Do not query further.

QUERY SHAPE RULES:
1. Anchor on a specific condition name (not a symptom like "tachycardia" alone — add the condition it relates to).
2. Include at least one drug name, procedure, or treatment class.
3. Use "treatment", "dose", or "protocol" as action term. Never use "guidelines", "classification", or "assessment" as action term.
4. If a query misses: swap ONE token (add a drug name, age group, or synonym). Do not just append "WHO".
5. Never use "WHO" as a standalone anchor — attach it to a clinical term.
"""

SYSTEM_BLOCK = f"<system>\n{DEPTH_RULES}\n---\n{KB_SEARCH_RULES}</system>\n\n"


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
    """Query the KB server and return {"hit": top_hit, "hits": all_hits} or None."""
    try:
        data = json.dumps({
            "query": query, "k": 3, "source_mode": "auto",
            "snippet_chars": 1200, "threshold": 0.0,
            "search_mode": os.environ.get("KB_SEARCH_MODE", "rrf"),
        }).encode()
        req = urllib.request.Request(
            f"{KB_URL}/search", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
        hit = result.get("hit")
        if not hit:
            return None
        hits = result.get("hits", [hit])
        return {"hit": hit, "hits": hits}
    except Exception as e:
        log.warning("KB query failed: %s", e)
        return None


def _format_hits(kb: dict) -> list:
    """Serialize all KB hits for SSE/JSON transport."""
    return [
        {
            "title": h.get("title", ""),
            "content": h.get("content", ""),
            "score": h.get("score", 0),
            "source": h.get("source", ""),
            "uri": h.get("uri", ""),
        }
        for h in kb.get("hits", [])
    ]


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
    raw_prompt = request.prompt
    # SYSTEM_BLOCK injection disabled — model was not trained with this wrapper.
    # if not raw_prompt.startswith(SYSTEM_BLOCK):
    #     raw_prompt = SYSTEM_BLOCK + raw_prompt

    conversation = raw_prompt
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
        tok_limit = 6144 if is_last else 2048

        try:
            generated = _generate(conversation, max_tokens=tok_limit)
        except Exception as e:
            log.error("Model server error: %s", e)
            raise HTTPException(status_code=502, detail=f"Model server error: {e}")

        all_text += generated

        kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL | re.IGNORECASE)
        has_answer = bool(re.search(r"## Clinical Assessment", generated))
        kb_limit_reached = len(kb_queries) >= MAX_KB_QUERIES

        if kb_matches and not has_answer and not is_last and not kb_limit_reached:
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
            top = kb["hit"] if kb else None
            kb_entry = {
                "query": raw_q,
                "score": top["score"] if top else 0,
                "source": top["source"] if top else "none",
                "hits": _format_hits(kb) if kb else [],
            }
            kb_queries.append(kb_entry)

            if top:
                kb_tag = (
                    f'<KB_RESULT source="{top["source"]}" score="{top["score"]:.1f}">\n'
                    f'{top["content"]}\n</KB_RESULT>'
                )
            else:
                kb_tag = (
                    f'<KB_RESULT source="none" score="0">\n'
                    f'No KB evidence found for: {raw_q}\n</KB_RESULT>'
                )

            # If this was the last allowed KB query, force the final answer immediately
            if len(kb_queries) >= MAX_KB_QUERIES:
                force_suffix = (
                    "\n\nYou have gathered sufficient evidence. "
                    "Now write your complete clinical assessment with all 6 sections: "
                    "Alert Level, Clinical Assessment, Differential Considerations, "
                    "Recommended Actions, Safety Alerts, Key Points."
                )
                conversation += generated + "<end_of_turn>\n"
                conversation += (
                    f"<start_of_turn>user\n{kb_tag}{force_suffix}"
                    f"<end_of_turn>\n<start_of_turn>model\n"
                )
            else:
                conversation += generated + "<end_of_turn>\n"
                conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"
        else:
            break

    # ── Safety net: force assessment if the model never wrote one ──────
    has_any_assessment = bool(re.search(
        r"## (?:Alert Level|Clinical Assessment|Recommended)", all_text))

    if not has_any_assessment:
        log.warning("Non-streaming: no clinical assessment after %d turns — forcing", turn + 1)
        force_msg = (
            "No relevant KB evidence was found for this case. "
            "Using your clinical training knowledge, write the complete "
            "clinical assessment now with all 6 sections: "
            "## Alert Level, ## Clinical Assessment, ## Differential Considerations, "
            "## Recommended Actions, ## Safety Alerts, ## Key Points. "
            "Base your recommendations on standard clinical practice."
        )
        conversation += generated + "<end_of_turn>\n"
        conversation += (
            f"<start_of_turn>user\n{force_msg}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        try:
            forced = _generate(conversation, max_tokens=6144)
            all_text += forced
        except Exception as e:
            log.error("Forced generation failed: %s", e)

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
    """Multi-turn CDS with real-time token streaming via SSE.

    Uses an asyncio queue to bridge the blocking urllib token generator
    (running in a thread pool) with the async SSE event stream, so tokens
    are flushed to the browser as they arrive rather than batched at the end.
    """
    loop = asyncio.get_event_loop()

    async def event_stream() -> AsyncGenerator[str, None]:
        raw_prompt = request.prompt
        # SYSTEM_BLOCK injection disabled — model was not trained with this wrapper.
        # Keeping the block defined above for reference; do not inject it.
        # if not raw_prompt.startswith(SYSTEM_BLOCK):
        #     raw_prompt = SYSTEM_BLOCK + raw_prompt

        conversation = raw_prompt
        if not conversation.startswith("<bos>"):
            conversation = f"<bos>{conversation}"
        if not conversation.rstrip().endswith("<start_of_turn>model\n"):
            if "<start_of_turn>model" not in conversation:
                conversation += "<start_of_turn>model\n"

        kb_queries = []
        used_q = set()
        all_generated_text = []  # tracks ALL generated text across turns

        # Track approximate token count to stay inside context window.
        # Rough estimate: 1 token ≈ 3 chars. Model ctx = 8192. Keep 2048 for output.
        MAX_INPUT_CHARS = (8192 - 2048) * 3

        for turn in range(request.max_turns):
            is_last = turn == request.max_turns - 1
            tok_limit = 6144 if is_last else 1536

            yield _sse_event({"type": "turn_start", "turn": turn + 1})

            # ── Run blocking _stream_generate in thread, feed tokens via queue ──
            queue: asyncio.Queue = asyncio.Queue()
            generated_parts: list[str] = []

            def _run_generate() -> None:
                try:
                    for tok in _stream_generate(conversation, max_tokens=tok_limit):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", tok))
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

            _thread_pool.submit(_run_generate)

            error_msg = None
            while True:
                kind, value = await queue.get()
                if kind == "token":
                    generated_parts.append(value)
                    yield _sse_event({"type": "token", "text": value})
                elif kind == "done":
                    break
                else:
                    error_msg = value
                    break

            if error_msg:
                yield _sse_event({"type": "error", "message": error_msg})
                return

            generated = "".join(generated_parts)
            all_generated_text.append(generated)

            kb_matches = re.findall(r"<KB_QUERY>(.*?)</KB_QUERY>", generated, re.DOTALL | re.IGNORECASE)
            has_answer = bool(re.search(r"## Clinical Assessment", generated))
            kb_limit_reached = len(kb_queries) >= MAX_KB_QUERIES

            if kb_matches and not has_answer and not is_last and not kb_limit_reached:
                raw_q = None
                for m in kb_matches:
                    if m.strip().lower() not in used_q:
                        raw_q = m.strip()
                        break

                if raw_q is None:
                    break

                used_q.add(raw_q.lower())
                yield _sse_event({"type": "kb_query", "query": raw_q})

                kb = await loop.run_in_executor(_thread_pool, _query_kb, raw_q)
                top = kb["hit"] if kb else None
                kb_entry = {
                    "query": raw_q,
                    "score": top["score"] if top else 0,
                    "source": top["source"] if top else "none",
                    "hits": _format_hits(kb) if kb else [],
                }
                kb_queries.append(kb_entry)

                yield _sse_event({
                    "type": "kb_result",
                    "query": raw_q,
                    "source": top["source"] if top else "none",
                    "score": top["score"] if top else 0,
                    "hits": _format_hits(kb) if kb else [],
                })

                if top:
                    kb_tag = (
                        f'<KB_RESULT source="{top["source"]}" score="{top["score"]:.1f}">\n'
                        f'{top["content"]}\n</KB_RESULT>'
                    )
                else:
                    kb_tag = (
                        f'<KB_RESULT source="none" score="0">\n'
                        f'No KB evidence found for: {raw_q}\n</KB_RESULT>'
                    )

                if len(kb_queries) >= MAX_KB_QUERIES:
                    force_suffix = (
                        "\n\nYou have gathered sufficient evidence. "
                        "Now write your complete clinical assessment with all 6 sections: "
                        "Alert Level, Clinical Assessment, Differential Considerations, "
                        "Recommended Actions, Safety Alerts, Key Points."
                    )
                    conversation += generated + "<end_of_turn>\n"
                    conversation += (
                        f"<start_of_turn>user\n{kb_tag}{force_suffix}"
                        f"<end_of_turn>\n<start_of_turn>model\n"
                    )
                else:
                    conversation += generated + "<end_of_turn>\n"
                    conversation += f"<start_of_turn>user\n{kb_tag}<end_of_turn>\n<start_of_turn>model\n"

                # Trim conversation if it's grown too long to fit in context
                if len(conversation) > MAX_INPUT_CHARS:
                    # Keep system block + last 60% of conversation
                    keep = int(MAX_INPUT_CHARS * 0.6)
                    system_end = conversation.find("<start_of_turn>user\n") + len("<start_of_turn>user\n")
                    system_part = conversation[:system_end]
                    tail = conversation[-keep:]
                    conversation = system_part + "[...context trimmed...]\n" + tail
                    log.warning("Conversation trimmed to fit context window")
            else:
                break

        # ── Safety net: force assessment if the model never wrote one ──────
        # When KB coverage is poor the model can exhaust all turns generating
        # only <think> blocks and <KB_QUERY> tags with no actual clinical
        # content. The frontend strips those tags and the user sees a blank.
        full_output = "".join(all_generated_text)
        has_any_assessment = bool(re.search(
            r"## (?:Alert Level|Clinical Assessment|Recommended)", full_output))

        if not has_any_assessment:
            log.warning("No clinical assessment after %d turns — forcing final generation", turn + 1)
            force_msg = (
                "No relevant KB evidence was found for this case. "
                "Using your clinical training knowledge, write the complete "
                "clinical assessment now with all 6 sections: "
                "## Alert Level, ## Clinical Assessment, ## Differential Considerations, "
                "## Recommended Actions, ## Safety Alerts, ## Key Points. "
                "Base your recommendations on standard clinical practice."
            )
            conversation += generated + "<end_of_turn>\n"
            conversation += (
                f"<start_of_turn>user\n{force_msg}"
                f"<end_of_turn>\n<start_of_turn>model\n"
            )

            if len(conversation) > MAX_INPUT_CHARS:
                keep = int(MAX_INPUT_CHARS * 0.6)
                system_end = conversation.find("<start_of_turn>user\n") + len("<start_of_turn>user\n")
                system_part = conversation[:system_end]
                tail = conversation[-keep:]
                conversation = system_part + "[...context trimmed...]\n" + tail

            yield _sse_event({"type": "turn_start", "turn": turn + 2})

            queue_final: asyncio.Queue = asyncio.Queue()
            final_parts: list[str] = []

            def _run_final() -> None:
                try:
                    for tok in _stream_generate(conversation, max_tokens=6144):
                        loop.call_soon_threadsafe(queue_final.put_nowait, ("token", tok))
                    loop.call_soon_threadsafe(queue_final.put_nowait, ("done", None))
                except Exception as exc:
                    loop.call_soon_threadsafe(queue_final.put_nowait, ("error", str(exc)))

            _thread_pool.submit(_run_final)

            while True:
                kind, value = await queue_final.get()
                if kind == "token":
                    final_parts.append(value)
                    yield _sse_event({"type": "token", "text": value})
                elif kind == "done":
                    break
                else:
                    yield _sse_event({"type": "error", "message": value})
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
