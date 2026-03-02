# ClinicDx — Architecture Deep Dive

## System Components

### 1. OpenMRS ESM Module (`openmrs-module/`)

A [single-spa](https://single-spa.js.org/) microfrontend built with the OpenMRS O3 framework. Registers four workspaces into the OpenMRS patient chart via `routes.json`:

- **CDS workspace** — sends patient context to `/cds/generate`, renders streaming markdown
- **Scribe workspace** — records audio, posts to `/scribe/process_audio`, shows confirmed observations
- **OCR workspace** — image/document upload and text extraction
- **Imaging workspace** — clinical image analysis

**Configuration schema** (`index.ts`):
```json
{
  "middlewareUrl": "http://localhost:8321"
}
```

### 2. Unified Model Server (`services/unified-model-server/`)

A single FastAPI process that owns the full model stack in GPU memory:

```
GPU Memory Layout:
  MedASR Wav2Vec2 encoder     ~105M params  (frozen)
  AudioProjector (custom)      ~11.8M params (trained, loaded from .pt)
  MedGemma 4B language model   ~4.3B params  (frozen, LoRA merged)
```

**AudioProjector architecture** (`modeling/gemma3_audio.py`):
```
Input: [B, T_enc, 512]  (MedASR Conformer output)
  Frame stacking k=4  →  [B, T/4, 2048]
  Linear(2048 → 2560)
  RMSNorm(2560)
  GELU
  Linear(2560 → 2560)
  Pad/truncate to 64 tokens
Output: [B, 64, 2560]  (MedGemma embedding dimension)
```

Token budget: 1s audio → ~13 projected frames, 5s → ~63.

### 3. Python Middleware (`services/middleware/`)

FastAPI service that routes frontend requests to the model server and KB daemon.

**CDS Router** (`service/cds_router.py`):
- Implements multi-turn ReAct loop (up to 4 turns)
- Model emits `<KB_QUERY>term</KB_QUERY>` → middleware resolves → injects `<KB_RESULT>` back
- Streaming endpoint via SSE (`/cds/generate_stream`)

**Scribe Router** (`service/scribe_router.py`):
- `/scribe/manifest` — builds CIEL concept manifest from live OpenMRS encounter
- `/scribe/process_audio` — sends audio to model server `/v1/audio/extract`, maps results to FHIR
- `/scribe/confirm` — POSTs confirmed FHIR R4 Observation resources to OpenMRS

### 4. Knowledge Base Daemon (`services/knowledge-base/`)

Minimal stdlib-only HTTP server (no FastAPI dependency) serving two memvid indexes:

| Index | Size | Contents |
|---|---|---|
| `who_knowledge.mv2` | ~large | WHO Africa clinical guidelines |
| `wikimed.mv2` | ~large | WikiMed medical reference |

Search mode: lexical (BM25-style) via `memvid_sdk`. Scores ≥ 15.0 are injected into the CDS context (threshold in `cds_router.py`).

## Data Formats

### CDS Prompt Format (Gemma chat template)
```
<bos><start_of_turn>user
{patient_case_xml}<end_of_turn>
<start_of_turn>model
<think>
  QUERY_ESTIMATE: 2
  <KB_QUERY>malaria diagnosis criteria</KB_QUERY>
</think>
<end_of_turn>
<start_of_turn>user
<KB_RESULT source="WHO Guidelines" score="18.5">
  ...WHO malaria treatment content...
</KB_RESULT><end_of_turn>
<start_of_turn>model
## Clinical Assessment
...
```

### Scribe Output Format
```
temperature: 37.8
blood_pressure: 120/80
chief_complaint: headache and fever
malaria_test_result: positive
```

### FHIR R4 Observation (built by `fhir_builder.py`)
```json
{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [
      {"code": "<local_uuid>"},
      {"system": "https://cielterminology.org", "code": "5088"}
    ]
  },
  "subject": {"reference": "Patient/<uuid>"},
  "encounter": {"reference": "Encounter/<uuid>"},
  "valueQuantity": {"value": 37.8, "unit": "Cel"}
}
```

## Deployment Topology

Production runs on a GCP VM (internal IP `10.128.0.4`):
- Port 8000: Unified Model Server (GPU)
- Port 4276: KB Daemon (CPU)
- Port 8321: Middleware (CPU)
- OpenMRS frontend served by O3 dev server / nginx
