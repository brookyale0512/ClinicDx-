"""FastAPI service for MedASR Voice Form Filler.

Provides endpoints for:
- POST /api/transcribe: audio -> text
- POST /api/extract: text -> structured CIEL observations
- POST /api/pipeline: audio -> structured CIEL observations (combined)
- POST /api/pipeline_direct: audio -> direct audio-to-concept (bypass text)
- GET /api/health: service health check
- GET /api/concepts: list loaded CIEL concepts
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .concept_extractor import ConceptExtractor
from .scribe_router import router as scribe_router
from .cds_router import router as cds_router
from .transcribe import MedASRTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global model instances
transcriber: Optional[MedASRTranscriber] = None
extractor: Optional[ConceptExtractor] = None
direct_pipeline = None  # Optional[DirectAudioPipeline]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Middleware startup — models live on the unified model server, not here."""
    global transcriber, extractor, direct_pipeline

    logger.info("Middleware starting (models served by unified model server)")

    extractor = ConceptExtractor()
    try:
        extractor._load_ciel_mappings()
        logger.info("CIEL mappings loaded for rule-based fallback")
    except Exception as e:
        logger.warning("Failed to load CIEL mappings: %s", e)

    yield

    logger.info("Shutting down middleware.")


app = FastAPI(
    title="MedASR Voice Form Filler",
    description="Converts spoken clinical phrases into structured OpenMRS observations",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O3 frontend runs on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scribe_router)
app.include_router(cds_router)


# --- Request/Response Models ---

class TranscribeResponse(BaseModel):
    text: str
    duration_ms: float


class ExtractRequest(BaseModel):
    text: str
    form_context: Optional[str] = Field(
        None, description="Active form section: vitals, diagnosis, medications, etc."
    )
    encounter_history: Optional[list[dict]] = Field(
        None, description="Previously extracted observations in this encounter"
    )


class Observation(BaseModel):
    concept_id: int
    concept_uuid: Optional[str] = None
    label: str
    value: object
    datatype: str
    units: Optional[str] = None
    confidence: float = 0.0


class CDSAlert(BaseModel):
    type: str  # warning, info, critical
    message: str


class ExtractResponse(BaseModel):
    observations: list[dict]
    cds_alerts: list[dict] = []
    fallback: Optional[bool] = None
    duration_ms: float = 0.0


class PipelineResponse(BaseModel):
    transcription: str
    observations: list[dict]
    cds_alerts: list[dict] = []
    fallback: Optional[bool] = None
    transcribe_ms: float = 0.0
    extract_ms: float = 0.0
    total_ms: float = 0.0


class DirectPipelineResponse(BaseModel):
    observations: list[dict]
    cds_alerts: list[dict] = []
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    transcriber_loaded: bool
    extractor_mode: str  # "llm", "rule_based", or "unavailable"
    concepts_loaded: dict
    direct_audio_available: bool = False


# --- Endpoints ---

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    if extractor is not None and extractor._model is not None:
        mode = "llm"
    elif extractor is not None and extractor._ciel_data is not None:
        mode = "rule_based"
    else:
        mode = "unavailable"

    return HealthResponse(
        status="ok" if (transcriber and transcriber._pipe) else "loading",
        transcriber_loaded=transcriber is not None and transcriber._pipe is not None,
        extractor_mode=mode,
        concepts_loaded=extractor.get_ciel_concepts_summary() if extractor else {},
        direct_audio_available=direct_pipeline is not None and direct_pipeline.is_loaded,
    )


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio to text using MedASR.

    Accepts audio file (wav, webm, mp3, flac).
    Returns transcribed text.
    """
    if transcriber is None or transcriber._pipe is None:
        raise HTTPException(503, "Transcription model not loaded")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    t0 = time.time()
    result = transcriber.transcribe_bytes(audio_bytes)
    duration_ms = (time.time() - t0) * 1000

    return TranscribeResponse(
        text=result["text"],
        duration_ms=round(duration_ms, 1),
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract structured CIEL observations from transcribed text.

    Uses MedGemma to map clinical phrases to OpenMRS concept codes.
    Falls back to rule-based extraction if LLM parsing fails.
    """
    if extractor is None:
        raise HTTPException(503, "Concept extractor not initialized")

    if not request.text.strip():
        raise HTTPException(400, "Empty text")

    t0 = time.time()
    if extractor._model is not None:
        result = extractor.extract(
            text=request.text,
            form_context=request.form_context,
            encounter_history=request.encounter_history,
        )
    else:
        # Rule-based fallback when MedGemma is not available
        result = extractor._rule_based_fallback(request.text)
    duration_ms = (time.time() - t0) * 1000

    return ExtractResponse(
        observations=result.get("observations", []),
        cds_alerts=result.get("cds_alerts", []),
        fallback=result.get("fallback"),
        duration_ms=round(duration_ms, 1),
    )


@app.post("/api/pipeline", response_model=PipelineResponse)
async def pipeline(
    audio: UploadFile = File(...),
    form_context: Optional[str] = Form(None),
):
    """Full pipeline: audio -> transcription -> concept extraction.

    Combines /transcribe and /extract in a single call for convenience.
    """
    if transcriber is None or transcriber._pipe is None:
        raise HTTPException(503, "Transcription model not loaded")
    if extractor is None:
        raise HTTPException(503, "Concept extractor not initialized")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    # Step 1: Transcribe
    t0 = time.time()
    transcription = transcriber.transcribe_bytes(audio_bytes)
    t1 = time.time()
    transcribe_ms = (t1 - t0) * 1000

    text = transcription["text"]
    if not text.strip():
        return PipelineResponse(
            transcription="",
            observations=[],
            cds_alerts=[],
            transcribe_ms=round(transcribe_ms, 1),
            extract_ms=0.0,
            total_ms=round(transcribe_ms, 1),
        )

    # Step 2: Extract concepts (LLM or rule-based fallback)
    t2 = time.time()
    if extractor._model is not None:
        extraction = extractor.extract(text=text, form_context=form_context)
    else:
        extraction = extractor._rule_based_fallback(text)
    t3 = time.time()
    extract_ms = (t3 - t2) * 1000

    return PipelineResponse(
        transcription=text,
        observations=extraction.get("observations", []),
        cds_alerts=extraction.get("cds_alerts", []),
        fallback=extraction.get("fallback"),
        transcribe_ms=round(transcribe_ms, 1),
        extract_ms=round(extract_ms, 1),
        total_ms=round((t3 - t0) * 1000, 1),
    )


@app.post("/api/pipeline_direct", response_model=DirectPipelineResponse)
async def pipeline_direct(audio: UploadFile = File(...)):
    """Direct audio-to-concept pipeline (bypass text intermediate).

    Uses MedASR encoder -> AudioProjector -> MedGemma in a single pass.
    Requires PROJECTOR_CHECKPOINT env var to be set at startup.
    """
    if direct_pipeline is None or not direct_pipeline.is_loaded:
        raise HTTPException(
            503,
            "Direct audio pipeline not available. "
            "Set PROJECTOR_CHECKPOINT env var and restart.",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    t0 = time.time()
    result = direct_pipeline.extract_bytes(audio_bytes)
    duration_ms = (time.time() - t0) * 1000

    return DirectPipelineResponse(
        observations=result.get("observations", []),
        cds_alerts=result.get("cds_alerts", []),
        duration_ms=round(duration_ms, 1),
    )


@app.get("/api/concepts")
async def list_concepts():
    """List loaded CIEL concept categories and counts."""
    if extractor is None or extractor._ciel_data is None:
        raise HTTPException(503, "Concept data not loaded")
    return extractor._ciel_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8321)
