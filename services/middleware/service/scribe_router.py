"""Voice Scribe API router.

Three endpoints:
  GET  /scribe/manifest?encounter_uuid=...
       Returns manifest string + lookup table for the encounter.

  POST /scribe/process
       Takes manifest + transcription text, calls model (stub for now),
       returns per-item human-readable labels + pre-built FHIR payloads.

  POST /scribe/confirm
       Takes a list of confirmed FHIR payloads, POSTs each to OpenMRS,
       returns per-item success/failure.
"""

import logging
import os
import re
from typing import Any, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .fhir_builder import build_fhir_payload, human_readable
from .manifest import get_builder

logger = logging.getLogger(__name__)

OPENMRS_BASE = os.environ.get("OPENMRS_URL", "")
OPENMRS_USER = os.environ.get("OPENMRS_USER", "admin")
OPENMRS_PASS = os.environ.get("OPENMRS_PASSWORD", "Admin123")


def _require_openmrs() -> None:
    """Raise 501 if OPENMRS_URL is not configured (engine-only mode)."""
    if not OPENMRS_BASE:
        raise HTTPException(
            status_code=501,
            detail=(
                "This endpoint requires OPENMRS_URL to be configured. "
                "Set the OPENMRS_URL environment variable to your OpenMRS instance."
            ),
        )

router = APIRouter(prefix="/scribe", tags=["Voice Scribe"])


# ── Request / Response models ──────────────────────────────────────────────────

class ManifestResponse(BaseModel):
    encounter_uuid: str
    patient_uuid: Optional[str]
    encounter_type_name: Optional[str]
    location_name: Optional[str]
    manifest_string: str
    lookup: dict[str, Any]


class ProcessRequest(BaseModel):
    encounter_uuid: str
    transcription: str
    manifest_string: str
    lookup: dict[str, Any]
    patient_uuid: str


class ExtractedItem(BaseModel):
    id: str
    label: str
    value: str
    human_readable: str
    fhir_type: str
    fhir_payload: dict[str, Any]
    status: str = "pending"  # pending | confirmed | rejected
    confidence: float = 1.0  # 0.0-1.0, from audio-concept cosine similarity
    not_in_manifest: bool = False  # concept mentioned but outside encounter manifest


class ProcessResponse(BaseModel):
    transcription: str
    items: list[ExtractedItem]
    raw_model_output: str


class ConfirmRequest(BaseModel):
    encounter_uuid: str
    patient_uuid: str
    items: list[dict[str, Any]]  # confirmed ExtractedItem dicts


class ConfirmResultItem(BaseModel):
    id: str
    label: str
    status: str  # posted | failed | skipped
    fhir_id: Optional[str] = None
    error: Optional[str] = None


class ConfirmResponse(BaseModel):
    posted: int
    failed: int
    results: list[ConfirmResultItem]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_model_output(raw: str, lookup: dict) -> list[dict]:
    """Parse 'label: value' lines from model output into structured items."""
    items = []
    for i, line in enumerate(raw.strip().splitlines()):
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, value = line.partition(":")
        label = label.strip().lower().replace(" ", "_")
        value = value.strip()

        if not label or not value:
            continue

        items.append({
            "id": f"item_{i}",
            "label": label,
            "value": value,
        })
    return items


MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://10.128.0.4:8000")
SCRIBE_MODEL = os.environ.get("MODEL_NAME", "/var/www/ClinicDx/scribe_version/model/medgemma_cds_think_v1")

SCRIBE_SYSTEM = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "A clinical phrase is provided below along with a concept manifest.\n"
    "Extract structured medical observations from the phrase.\n"
    "Return ONLY key: value lines matching concepts from the manifest."
)

SKIP_LABELS = {"query_1", "query_2", "query_3", "query_4", "query_5",
               "query_estimate", "key", "query", "decision", "case_complexity",
               "next_query"}


def _call_model(transcription: str, manifest_string: str) -> str:
    """Call medgemma via vLLM and extract key:value from think block."""
    import json as _json
    import urllib.request
    import urllib.error

    prompt = (
        f"<bos><start_of_turn>user\n{SCRIBE_SYSTEM}\n\n"
        f"{manifest_string}\n\n"
        f'PHRASE: "{transcription}"\n\n'
        f"OUTPUT:<end_of_turn>\n<start_of_turn>model\n"
    )

    payload = _json.dumps({
        "model": SCRIBE_MODEL,
        "prompt": prompt,
        "max_tokens": 1500,
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
        with urllib.request.urlopen(req, timeout=60) as r:
            result = _json.loads(r.read())
        text = result["choices"][0]["text"]
        logger.info("Scribe model returned %d chars", len(text))
        return text
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        logger.error("Scribe model HTTP error %s: %s", e.code, body)
        return ""
    except Exception as e:
        logger.error("Scribe model call failed: %s", e)
        return ""


def _extract_from_think(raw: str, manifest_labels: set) -> str:
    """Extract key:value lines from inside <think> block, matching manifest concepts."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    content = think_match.group(1) if think_match else raw

    lines = []
    seen = set()
    for line in content.splitlines():
        line = line.strip().lstrip("- ")
        for ml in manifest_labels:
            pattern = re.escape(ml) + r"\s*[:=]\s*(.+)"
            m = re.search(pattern, line, re.IGNORECASE)
            if m and ml not in seen:
                val = m.group(1).strip().rstrip(",").strip()
                if val:
                    seen.add(ml)
                    lines.append(f"{ml}: {val}")
                break

    return "\n".join(lines)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/manifest", response_model=ManifestResponse)
async def get_manifest(encounter_uuid: str):
    """Build and return the concept manifest for an encounter.

    Called by the ESM workspace when it opens, before recording starts.
    Requires OPENMRS_URL to be configured.
    """
    _require_openmrs()
    builder = get_builder()
    try:
        result = await builder.build_manifest(encounter_uuid)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"OpenMRS error: {e.response.text[:200]}",
        )
    except Exception as e:
        logger.exception("Manifest build failed for %s", encounter_uuid)
        raise HTTPException(500, f"Manifest build failed: {e}")

    ctx = result["context"]
    return ManifestResponse(
        encounter_uuid=encounter_uuid,
        patient_uuid=ctx.get("patient_uuid"),
        encounter_type_name=ctx.get("encounter_type_name"),
        location_name=ctx.get("location_name"),
        manifest_string=result["manifest_string"],
        lookup=result["lookup"],
    )


@router.post("/process", response_model=ProcessResponse)
async def process_transcription(req: ProcessRequest):
    """Process a transcription against the manifest.

    Calls the model (stub until MedGemma is integrated) and returns
    per-item human-readable labels + pre-built FHIR payloads ready for
    the doctor to confirm.
    """
    if not req.transcription.strip():
        raise HTTPException(400, "Empty transcription")

    manifest_labels = set()
    for line in req.manifest_string.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("CONCEPTS"):
            continue
        clean = re.sub(r"^\[.*?\]\s*", "", line).strip()
        label = clean.split("(")[0].strip().lower().replace(" ", "_")
        if label:
            manifest_labels.add(label)

    raw_model = _call_model(req.transcription, req.manifest_string)
    raw_output = _extract_from_think(raw_model, manifest_labels)

    # ── Parse output ──────────────────────────────────────────────────────────
    parsed = _parse_model_output(raw_output, req.lookup)

    items: list[ExtractedItem] = []
    for p in parsed:
        label = p["label"]
        value = p["value"]

        if label.startswith("#"):
            continue

        value = value.strip('"').strip("'")
        if any(neg in value.lower() for neg in ["not present", "not mentioned", "absent", "n/a"]):
            continue
        if value.lower().strip() == "present":
            concept_meta = req.lookup.get(label) if req.lookup else None
            if concept_meta and concept_meta.get("value_type") == "Quantity":
                continue

        if not value:
            continue

        readable_label = label.replace("_", " ").title()
        readable = f"{readable_label}: {value}"

        concept_meta = req.lookup.get(label) if req.lookup else None
        if concept_meta:
            try:
                fhir_payload = build_fhir_payload(
                    label=label, value=value, concept_meta=concept_meta,
                    patient_uuid=req.patient_uuid, encounter_uuid=req.encounter_uuid,
                )
            except Exception:
                fhir_payload = {}
            items.append(ExtractedItem(
                id=p["id"], label=label, value=value,
                human_readable=human_readable(label, value, concept_meta),
                fhir_type=concept_meta.get("fhir_type", "Observation"),
                fhir_payload=fhir_payload or {},
                status="pending", confidence=0.9,
            ))
        else:
            items.append(ExtractedItem(
                id=p["id"], label=label, value=value,
                human_readable=readable,
                fhir_type="Observation",
                fhir_payload={},
                status="pending", confidence=0.85,
            ))

    return ProcessResponse(
        transcription=req.transcription,
        items=items,
        raw_model_output=raw_output,
    )


class AudioProcessRequest(BaseModel):
    encounter_uuid: str
    patient_uuid: str
    manifest_string: str
    lookup: dict[str, Any]


@router.post("/process_audio", response_model=ProcessResponse)
async def process_audio(
    audio: UploadFile = File(...),
    encounter_uuid: str = Form(...),
    patient_uuid: str = Form(...),
    manifest_string: str = Form(""),
    lookup: str = Form("{}"),
):
    """Direct audio-to-observations: audio → unified model → FHIR-ready items.

    Sends audio bytes directly to the unified model server's /v1/audio/extract
    endpoint, which runs MedASR encoder → AudioProjector → LLM in one pass.
    """
    import json as _json
    import urllib.request
    import urllib.error
    from io import BytesIO

    import subprocess as _subprocess
    import struct as _struct

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    # ── STAGE 0: Input diagnostics ──────────────────────────────────────────
    logger.info("[SCRIBE_DIAG] ═══════════════════════════════════════════════")
    logger.info("[SCRIBE_DIAG] STAGE 0 — INPUT")
    logger.info("[SCRIBE_DIAG]   encounter=%s  patient=%s", encounter_uuid, patient_uuid)
    logger.info("[SCRIBE_DIAG]   audio_bytes=%d  content_type=%s  filename=%s",
                len(audio_bytes), audio.content_type, audio.filename)

    # Detect format and transcode non-WAV audio to 16 kHz mono PCM-16 WAV via ffmpeg.
    # Browsers send audio/webm (Chrome/Edge) or audio/ogg (Firefox) — neither is WAV.
    # Also normalise existing WAVs to the model's expected format (16kHz, mono, 16-bit).
    _is_wav = (
        len(audio_bytes) >= 44
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )
    _wav_sr = _wav_ch = _wav_bits = None
    if _is_wav:
        try:
            _wav_ch   = _struct.unpack_from("<H", audio_bytes, 22)[0]
            _wav_sr   = _struct.unpack_from("<I", audio_bytes, 24)[0]
            _wav_bits = _struct.unpack_from("<H", audio_bytes, 34)[0]
        except Exception:
            pass

    _needs_transcode = (
        not _is_wav
        or (_wav_sr and _wav_sr != 16000)
        or (_wav_ch and _wav_ch != 1)
        or (_wav_bits and _wav_bits not in (16, 32))
    )

    if _needs_transcode:
        logger.info("[SCRIBE_DIAG]   transcoding via ffmpeg (is_wav=%s sr=%s ch=%s bits=%s)",
                    _is_wav, _wav_sr, _wav_ch, _wav_bits)
        try:
            # Write to a real temp file (not pipe) so ffmpeg can seek back and
            # fill in the correct WAV data_size header. pipe:1 output leaves
            # data_size=0xFFFFFFFF which causes integer overflow in the C++ parser.
            import tempfile as _tempfile
            with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp_in:
                _tmp_in_path = _tmp_in.name
            _tmp_out_path = _tmp_in_path.replace(".wav", "_out.wav")
            try:
                with open(_tmp_in_path, "wb") as _f:
                    _f.write(audio_bytes)
                proc = _subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", _tmp_in_path,  # read from real file
                        "-ar", "16000",      # resample to 16 kHz
                        "-ac", "1",          # mono
                        "-sample_fmt", "s16",# 16-bit signed PCM
                        "-f", "wav",
                        "-y",                # overwrite output
                        _tmp_out_path,       # write to real file (valid data_size header)
                    ],
                    capture_output=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    err = proc.stderr.decode(errors="replace")[:300]
                    logger.error("[SCRIBE_DIAG]   ffmpeg failed (rc=%d): %s", proc.returncode, err)
                    raise HTTPException(400, f"Audio transcoding failed: {err}")
                with open(_tmp_out_path, "rb") as _f:
                    audio_bytes = _f.read()
            finally:
                for _p in (_tmp_in_path, _tmp_out_path):
                    try:
                        os.unlink(_p)
                    except OSError:
                        pass
            logger.info("[SCRIBE_DIAG]   transcoded → %d bytes WAV (16kHz mono 16-bit)", len(audio_bytes))
            # Re-parse header after transcode
            try:
                _wav_ch   = _struct.unpack_from("<H", audio_bytes, 22)[0]
                _wav_sr   = _struct.unpack_from("<I", audio_bytes, 24)[0]
                _wav_bits = _struct.unpack_from("<H", audio_bytes, 34)[0]
            except Exception:
                pass
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("[SCRIBE_DIAG]   ffmpeg exception: %s", exc)
            raise HTTPException(400, f"Audio transcoding error: {exc}")
    else:
        logger.info("[SCRIBE_DIAG]   audio already valid WAV (sr=%s ch=%s bits=%s)",
                    _wav_sr, _wav_ch, _wav_bits)

    logger.info("[SCRIBE_DIAG]   WAV header: channels=%s sr=%s bits=%s",
                _wav_ch, _wav_sr, _wav_bits)
    if _wav_sr and _wav_sr != 16000:
        logger.warning("[SCRIBE_DIAG]   ⚠ SAMPLE RATE MISMATCH: WAV is %d Hz, model expects 16000 Hz", _wav_sr)
    if _wav_ch and _wav_ch != 1:
        logger.warning("[SCRIBE_DIAG]   ⚠ CHANNEL MISMATCH: WAV has %d channels, model expects mono", _wav_ch)

    # ── STAGE 1: Manifest diagnostics ──────────────────────────────────────
    logger.info("[SCRIBE_DIAG] STAGE 1 — MANIFEST")
    manifest_lines = [l.strip() for l in manifest_string.splitlines() if l.strip()]
    logger.info("[SCRIBE_DIAG]   manifest_string length=%d chars  lines=%d",
                len(manifest_string), len(manifest_lines))
    logger.info("[SCRIBE_DIAG]   manifest first 500 chars:\n%s", manifest_string[:500])

    try:
        lookup_dict = _json.loads(lookup)
    except _json.JSONDecodeError:
        lookup_dict = {}
    logger.info("[SCRIBE_DIAG]   lookup_dict keys (%d): %s", len(lookup_dict), list(lookup_dict.keys())[:20])

    boundary = "----FormBoundary" + os.urandom(8).hex()
    body_parts = []

    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="audio"; filename="recording.wav"\r\n')
    body_parts.append(b"Content-Type: audio/wav\r\n\r\n")
    body_parts.append(audio_bytes)
    body_parts.append(b"\r\n")

    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="manifest"\r\n\r\n')
    body_parts.append(manifest_string.encode())
    body_parts.append(b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    # ── STAGE 2: Model server call ──────────────────────────────────────────
    logger.info("[SCRIBE_DIAG] STAGE 2 — MODEL SERVER REQUEST")
    logger.info("[SCRIBE_DIAG]   url=%s/v1/audio/extract  body_size=%d bytes",
                MODEL_SERVER_URL, len(body))
    req = urllib.request.Request(
        f"{MODEL_SERVER_URL}/v1/audio/extract",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    import time as _time
    _t0 = _time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            result = _json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:300]
        logger.error("[SCRIBE_DIAG]   ✗ HTTP %s: %s", e.code, err)
        raise HTTPException(502, f"Model server error: {err}")
    except Exception as e:
        logger.error("[SCRIBE_DIAG]   ✗ connection error: %s", e)
        raise HTTPException(502, f"Model server unreachable: {e}")
    _elapsed_ms = (_time.monotonic() - _t0) * 1000

    raw_output = result.get("raw_output", "")
    model_obs = result.get("observations", [])
    model_duration_ms = result.get("duration_ms", 0)

    # ── STAGE 3: Raw model output ───────────────────────────────────────────
    logger.info("[SCRIBE_DIAG] STAGE 3 — RAW MODEL OUTPUT")
    logger.info("[SCRIBE_DIAG]   round_trip_ms=%.0f  model_duration_ms=%d",
                _elapsed_ms, model_duration_ms)
    logger.info("[SCRIBE_DIAG]   raw_output_len=%d chars  obs_count=%d",
                len(raw_output), len(model_obs))
    logger.info("[SCRIBE_DIAG]   raw_output:\n%s", raw_output[:1000])
    logger.info("[SCRIBE_DIAG]   raw observations from model:")
    for idx, o in enumerate(model_obs):
        logger.info("[SCRIBE_DIAG]     obs[%d]: key=%r  value=%r", idx, o.get("key", o.get("label")), o.get("value"))

    # ── backward-compat logging ─────────────────────────────────────────────
    logger.info("[process_audio] encounter=%s, patient=%s", encounter_uuid, patient_uuid)
    logger.info("[process_audio] lookup_dict keys: %s", list(lookup_dict.keys()))
    logger.info("[process_audio] model returned %d observations, raw_output=%d chars",
                len(model_obs), len(raw_output))
    logger.info("[process_audio] raw_output first 300 chars: %s", raw_output[:300])
    for idx, o in enumerate(model_obs):
        logger.info("[process_audio] model_obs[%d]: label=%r value=%r", idx, o.get("label"), o.get("value"))

    manifest_labels = set()
    for line in manifest_string.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("CONCEPTS") or line.upper().startswith("AVAILABLE"):
            continue
        clean = re.sub(r"^\[.*?\]\s*", "", line).strip()
        label = clean.split("(")[0].strip().lower().replace(" ", "_")
        if label:
            manifest_labels.add(label)

    logger.info("[process_audio] manifest_labels: %s", manifest_labels)

    # ── STAGE 4: Concept lookup diagnostics ────────────────────────────────
    logger.info("[SCRIBE_DIAG] STAGE 4 — CONCEPT LOOKUP")
    logger.info("[SCRIBE_DIAG]   manifest_labels (%d): %s", len(manifest_labels), sorted(manifest_labels))
    logger.info("[SCRIBE_DIAG]   lookup_dict keys (%d): %s", len(lookup_dict), sorted(lookup_dict.keys())[:30])

    items: list[ExtractedItem] = []
    seen_labels: set = set()  # FIX: deduplicate — keep first occurrence per normalised label
    for i, obs in enumerate(model_obs):
        raw_key = obs.get("key", obs.get("label", ""))
        # C++ parser emits {"key": ..., "value": ...} — support both field names
        label = (obs.get("label") or obs.get("key") or "").strip()
        # Strip leading underscores the model sometimes prepends
        label_after_underscore = label.lstrip("_").strip()
        # Strip bracket-type prefixes the model learned: [test], [drug], [value], etc.
        label = re.sub(r'^\[.*?\]\s*', '', label_after_underscore).strip()
        value = obs.get("value", "").strip()

        # FIX: strip JSON fragments that leak into value (e.g. '80h", "patient_id": "patient_id"')
        # Keep only the content up to the first unquoted JSON structural character
        value = re.split(r'(?<![0-9])["\'{}\[\]]', value)[0].strip().rstrip(',').strip()

        # FIX: strip trailing unit suffixes from Quantity values so '175cm' → '175', '38.5C' → '38.5'
        # Only strip alpha suffix when the value starts with a digit (numeric measurement)
        value_clean = re.sub(r'^([0-9]+(?:\.[0-9]+)?)\s*[a-zA-Z%/]+$', r'\1', value)
        if value_clean != value:
            logger.info("[SCRIBE_DIAG]     unit suffix stripped: %r → %r", value, value_clean)
            value = value_clean

        logger.info("[SCRIBE_DIAG]   obs[%d] raw_key=%r → after_underscore=%r → after_bracket=%r  value=%r",
                    i, raw_key, label_after_underscore, label, value)

        if not label or not value:
            logger.info("[SCRIBE_DIAG]     ✗ SKIP: empty label or value after cleaning")
            logger.info("[process_audio] SKIP obs[%d]: empty label or value", i)
            continue
        if any(neg in value.lower() for neg in ["not present", "not mentioned", "n/a"]):
            logger.info("[SCRIBE_DIAG]     ✗ SKIP: negative value sentinel %r", value)
            logger.info("[process_audio] SKIP obs[%d]: negative value %r", i, value)
            continue

        concept_meta = lookup_dict.get(label)

        # FIX: truncated key recovery — model sometimes emits '_kg: 30' instead of 'weight_kg: 30'
        # If direct lookup misses, check if the cleaned label is a suffix of any manifest key
        if concept_meta is None and label and len(label) >= 2:
            for mk in lookup_dict:
                if mk.endswith("_" + label) or mk == label:
                    logger.info("[SCRIBE_DIAG]     suffix-match: %r → manifest key %r", label, mk)
                    label = mk
                    concept_meta = lookup_dict[mk]
                    break

        logger.info("[SCRIBE_DIAG]     lookup_dict.get(%r) → %s", label,
                    "FOUND uuid=%s" % concept_meta.get("uuid","?") if concept_meta else "NOT_FOUND")

        if concept_meta is None:
            # Show closest keys by prefix for debugging
            close = [k for k in lookup_dict if k.startswith(label[:6])] if len(label) >= 6 else []
            if close:
                logger.info("[SCRIBE_DIAG]     nearest keys with prefix %r: %s", label[:6], close[:5])

        if value.lower().strip() in ("absent", "not present", "negative"):
            logger.info("[SCRIBE_DIAG]     ✗ SKIP: absent/negative value")
            logger.info("[process_audio] SKIP obs[%d] %r: absent", i, label)
            continue

        # "present" without a number means detection only — skip for Quantity concepts
        if value.lower().strip() == "present" and concept_meta and concept_meta.get("value_type") == "Quantity":
            logger.info("[SCRIBE_DIAG]     ✗ SKIP: 'present' for Quantity concept")
            logger.info("[process_audio] SKIP obs[%d] %r: 'present' but Quantity expects a number", i, label)
            continue

        # FIX: deduplicate — if we already accepted this label, skip subsequent occurrences
        if label in seen_labels:
            logger.info("[SCRIBE_DIAG]     ✗ SKIP: duplicate label %r (already accepted)", label)
            logger.info("[process_audio] SKIP obs[%d] %r: duplicate", i, label)
            continue
        seen_labels.add(label)

        readable_label = label.replace("_", " ").title()
        readable = f"{readable_label}: {value}"
        logger.info("[process_audio] obs[%d] label=%r -> concept_meta=%s", i, label,
                    "FOUND" if concept_meta else "NOT_FOUND (no fhir_payload)")
        if concept_meta:
            try:
                fhir_payload = build_fhir_payload(
                    label=label, value=value, concept_meta=concept_meta,
                    patient_uuid=patient_uuid, encounter_uuid=encounter_uuid,
                )
                logger.info("[process_audio] obs[%d] fhir_payload built: %d keys, resourceType=%s",
                            i, len(fhir_payload) if fhir_payload else 0,
                            fhir_payload.get("resourceType") if fhir_payload else "NONE")
            except Exception as exc:
                logger.error("[process_audio] obs[%d] fhir_payload build FAILED: %s", i, exc)
                fhir_payload = {}
            items.append(ExtractedItem(
                id=f"item_{i}", label=label, value=value,
                human_readable=human_readable(label, value, concept_meta),
                fhir_type=concept_meta.get("fhir_type", "Observation"),
                fhir_payload=fhir_payload or {},
                status="pending", confidence=0.95,
            ))
        else:
            items.append(ExtractedItem(
                id=f"item_{i}", label=label, value=value,
                human_readable=readable,
                fhir_type="Observation",
                fhir_payload={},
                status="pending", confidence=0.9,
                not_in_manifest=label.lower().replace(" ", "_") not in manifest_labels,
            ))

    logger.info("[process_audio] RESULT: %d items built, %d with fhir_payload",
                len(items), sum(1 for it in items if it.fhir_payload))

    # ── STAGE 5: Final summary ──────────────────────────────────────────────
    logger.info("[SCRIBE_DIAG] STAGE 5 — FINAL RESULT SUMMARY")
    logger.info("[SCRIBE_DIAG]   total_items=%d  items_with_fhir=%d  items_no_fhir=%d",
                len(items),
                sum(1 for it in items if it.fhir_payload),
                sum(1 for it in items if not it.fhir_payload))
    for it in items:
        logger.info("[SCRIBE_DIAG]   item label=%r value=%r fhir=%s not_in_manifest=%s",
                    it.label, it.value,
                    "YES" if it.fhir_payload else "NO",
                    it.not_in_manifest)
    logger.info("[SCRIBE_DIAG] ═══════════════════════════════════════════════")

    return ProcessResponse(
        transcription="[direct audio — no text intermediate]",
        items=items,
        raw_model_output=raw_output,
    )


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm_items(req: ConfirmRequest):
    """POST confirmed FHIR payloads to OpenMRS. Requires OPENMRS_URL."""
    _require_openmrs()
    import json as _json

    logger.info("[confirm] encounter=%s patient=%s items_count=%d",
                req.encounter_uuid, req.patient_uuid, len(req.items))
    for idx, item in enumerate(req.items):
        logger.info("[confirm] item[%d]: id=%s label=%s fhir_type=%s payload_keys=%s payload_empty=%s",
                    idx, item.get("id"), item.get("label"), item.get("fhir_type"),
                    list(item.get("fhir_payload", {}).keys()) if isinstance(item.get("fhir_payload"), dict) else "NOT_DICT",
                    not item.get("fhir_payload"))

    auth = (OPENMRS_USER, OPENMRS_PASS)
    results: list[ConfirmResultItem] = []
    posted = 0
    failed = 0

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, verify=False) as client:
        for item in req.items:
            item_id = item.get("id", "?")
            label = item.get("label", "?")
            payload = item.get("fhir_payload")
            fhir_type = item.get("fhir_type", "Observation")

            if not payload:
                logger.warning("[confirm] SKIPPED %s/%s: no fhir_payload (value=%r)", fhir_type, label, payload)
                results.append(ConfirmResultItem(
                    id=item_id, label=label, status="skipped",
                    error="No FHIR payload"
                ))
                continue

            url = f"{OPENMRS_BASE}/ws/fhir2/R4/{fhir_type}"
            logger.info("[confirm] POSTing %s/%s to %s payload=%s",
                        fhir_type, label, url, _json.dumps(payload)[:300])
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    auth=auth,
                    headers={"Content-Type": "application/fhir+json"},
                )
                resp.raise_for_status()
                fhir_id = resp.json().get("id")
                results.append(ConfirmResultItem(
                    id=item_id, label=label, status="posted", fhir_id=fhir_id
                ))
                posted += 1
                logger.info("[confirm] POSTED %s/%s -> fhir_id=%s", fhir_type, label, fhir_id)
            except httpx.HTTPStatusError as e:
                error_msg = e.response.text[:500]
                results.append(ConfirmResultItem(
                    id=item_id, label=label, status="failed", error=error_msg[:200]
                ))
                failed += 1
                logger.error("[confirm] FAILED %s/%s: HTTP %s: %s", fhir_type, label, e.response.status_code, error_msg)
            except Exception as e:
                results.append(ConfirmResultItem(
                    id=item_id, label=label, status="failed", error=str(e)
                ))
                failed += 1
                logger.exception("[confirm] EXCEPTION posting %s/%s", fhir_type, label)

    logger.info("[confirm] DONE: posted=%d failed=%d skipped=%d",
                posted, failed, len(req.items) - posted - failed)
    return ConfirmResponse(posted=posted, failed=failed, results=results)
