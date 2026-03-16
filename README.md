<div align="center">

<img src="docs/assets/clinicdx-logo.png" alt="ClinicDx" width="120" />

# ClinicDx

### Offline AI Clinical Intelligence for Sub-Saharan Africa

**Clinical Decision Support · Voice Scribe · Document OCR · Imaging Analysis**

Powered by [MedGemma](https://huggingface.co/google/medgemma-4b-it) · Works with [OpenMRS O3](https://openmrs.org/) or any EMR · Runs fully offline

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![OpenMRS](https://img.shields.io/badge/OpenMRS-O3-blue?logo=data:image/svg+xml;base64,)](https://openmrs.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-ClinicDx1%2FClinicDx-yellow)](https://huggingface.co/ClinicDx1/ClinicDx)
[![Website](https://img.shields.io/badge/Website-clinicdx.org-informational)](https://clinicdx.org)

[**Website**](https://clinicdx.org) · [**Model on HuggingFace**](https://huggingface.co/ClinicDx1/ClinicDx) · [**Documentation**](docs/) · [**Report a Bug**](https://github.com/brookyale0512/ClinicDx-/issues)

</div>

---

## Overview

ClinicDx is an **open-source clinical AI engine** designed for resource-limited settings in sub-Saharan Africa. It ships as **two independent components**:

| Component | What it is | How it ships |
|---|---|---|
| **CDS Engine** | Model inference + knowledge base + CDS/Scribe middleware | `docker compose up` — works with any EMR |
| **OpenMRS Module** | React frontend (CDS workspace, Scribe, OCR, Imaging) | npm package — `@openmrs/esm-clinicdx-app` |

The engine is **EMR-agnostic**. OpenMRS is one supported integration, not a requirement. Any EMR can call the engine's REST API to get clinical decision support and voice scribe capabilities.

| Capability | Description |
|---|---|
| **Clinical Decision Support (CDS)** | Structured 6-section assessments with evidence citations from a local WHO knowledge base, generated via a multi-turn retrieval-augmented reasoning loop |
| **Voice Scribe** | Speak clinical observations in natural language — Scribe extracts structured FHIR observations and writes them directly to the EMR |
| **Document OCR** | Digitise referral letters, lab reports, and prescriptions |
| **Imaging Analysis** | AI-assisted interpretation of clinical images |

Everything runs on a single 4B-parameter multimodal model fine-tuned from [Google MedGemma](https://huggingface.co/google/medgemma-4b-it). The model, knowledge base, and all services operate **fully offline** — no data leaves the facility.

---

## Key Features

- **Truly offline** — after initial model download, zero network dependency
- **EMR-agnostic** — REST API works with any EMR; OpenMRS module is one integration example
- **Zero-config first start** — `docker compose up` auto-downloads all artifacts (~6.6 GB)
- **FHIR R4 native** — Scribe output posts as structured observations
- **CIEL terminology** — maps to the full CIEL clinical concept vocabulary
- **Multimodal** — single model handles text (CDS), audio (Scribe), and vision (OCR/Imaging)
- **Evidence-grounded** — CDS cites WHO guidelines and clinical references from a local vector knowledge base
- **Docker Compose deployment** — the entire stack starts with a single command
- **Designed for low-resource settings** — runs on a single consumer GPU (≥ 8 GB VRAM); CPU mode available

---

## Model

ClinicDx is powered by a single fine-tuned multimodal model:

**[`ClinicDx1/ClinicDx`](https://huggingface.co/ClinicDx1/ClinicDx)** on HuggingFace

| File | Size | Purpose |
|---|---|---|
| [`clinicdx-v1-q8.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/clinicdx-v1-q8.gguf) | 3.9 GB | Language model (Q8 — only variant, no quality degradation) |
| [`medasr-encoder.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/medasr-encoder.gguf) | 401 MB | MedASR Conformer audio encoder (frozen, 105M params) |
| [`audio-projector-v3-best.gguf`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/audio-projector-v3-best.gguf) | 46 MB | AudioProjector v3 — best checkpoint (step 40,000, val LM 0.1042) |
| [`who_knowledge_vec_v2.mv2`](https://huggingface.co/ClinicDx1/ClinicDx/blob/main/who_knowledge_vec_v2.mv2) | 1.1 GB | WHO/MSF knowledge base v2.1 (27,860 chunks, BM25 + semantic hybrid) |

### Training Stages

```
Google MedGemma 4B-IT  (base)
       │
       ▼  Stage 1 — CDS LoRA Fine-Tuning
       │  LoRA (r=64, α=128) on 27,592 quality-filtered clinical conversations
       │  Single training run with anteoc reasoning and KB tool-use integrated
       │  Input masking: only model output turns trained (64% context masked)
       │  Best checkpoint: step 4,000  |  eval loss 0.4758  |  accuracy 86.25%
       │
       ▼  Merge LoRA  →  medgemma_cds_think_v1  (production CDS model)
       │
       ▼  Stage 2 — AudioProjector Training  (base model frozen)
          2-layer MLP + LayerNorm projector: MedASR (512-dim) → LLM space (2560-dim)
          50,000 synthetic clinical audio clips  |  10 epochs
          Best checkpoint: step 40,000  |  val LM 0.1042  |  key accuracy 84%
          11,806,720 trainable parameters
```

---

## System Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                Component 1: CDS Engine (Docker)                  │
 │                                                                  │
 │  ┌────────────────────────────────────────────────────────────┐ │
 │  │  FastAPI Middleware  :8321 (exposed to host)                │ │
 │  │   cds_router.py    — Multi-turn ReAct CDS + SSE streaming  │ │
 │  │   scribe_router.py — Audio → structured observations       │ │
 │  └──────────────────┬────────────────────┬────────────────────┘ │
 │                     │                    │                       │
 │           ┌─────────┘                    └──────────┐           │
 │           ▼                                         ▼           │
 │  ┌─────────────────────┐         ┌─────────────────────────┐   │
 │  │  KB Daemon v2 :4276 │         │  llama-server :8180      │   │
 │  │                     │         │                          │   │
 │  │  WHO Guidelines     │         │  MedASR Encoder (105M)   │   │
 │  │  MSF Protocols      │         │  AudioProjector v3       │   │
 │  │  BM25 + Semantic    │         │  ClinicDx LLM (4.3B, Q8) │   │
 │  │  Hybrid Retrieval   │         │                          │   │
 │  └─────────────────────┘         └─────────────────────────┘   │
 └─────────────────────────────┬───────────────────────────────────┘
                               │
                    REST API (any EMR can call)
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
         ▼                     ▼                      ▼
 ┌───────────────┐    ┌──────────────┐     ┌──────────────────┐
 │ OpenMRS O3    │    │ Other EMR    │     │ curl / any       │
 │ @openmrs/esm- │    │ (FHIR, REST) │     │ HTTP client      │
 │ clinicdx-app  │    │              │     │                  │
 │ (Component 2) │    │              │     │                  │
 └───────────────┘    └──────────────┘     └──────────────────┘
```

### CDS Flow — Multi-Turn ReAct with KB Tool-Use

```
1.  Frontend sends patient case  →  POST /cds/generate_stream
2.  Middleware builds Gemma chat prompt with encounter context
3.  Model streams thinking block  →  emits <KB_QUERY>term</KB_QUERY>
4.  Middleware queries KB daemon  →  injects <KB_RESULT>evidence</KB_RESULT>
5.  Up to 5 retrieval turns until structured response is complete
6.  SSE stream delivers 6-section markdown response with WHO citations
```

**CDS Response Schema:**
1. **Alert Level** — Routine / Urgent / Emergency
2. **Clinical Assessment** — Findings summary and reasoning
3. **Differential Considerations** — Ranked diagnoses with rationale
4. **Recommended Actions** — Investigations and management steps
5. **Safety Alerts** — Red-flag signs, interactions, contraindications
6. **Key Points** — Concise handover summary

### Scribe Flow — Direct Audio to FHIR

```
1.  Doctor records voice note in browser  →  POST /scribe/process_audio
2.  Middleware transcodes to 16kHz mono PCM-16 WAV (via ffmpeg)
3.  WAV → POST /v1/audio/extract on llama-server
4.  MedASR Conformer encodes audio  →  [T_enc, 512]
5.  AudioProjector projects  →  [64, 2560]  (fixed token budget)
6.  LLM decodes structured observations  →  "key: value" lines
7.  Middleware maps keys to CIEL concept codes  →  FHIR R4 payloads
8.  Doctor reviews and confirms  →  POST to OpenMRS FHIR API
```

---

## Repository Layout

```
ClinicDx/
├── docker-compose.yml           CDS Engine stack (model + KB + middleware)
├── docker-compose.full.yml      Full stack (engine + nginx for OpenMRS)
├── Makefile                     up / up-cpu / up-full / up-full-cpu / down / logs
├── .env.example                 Configuration template
│
├── openmrs-module/              OpenMRS O3 ESM microfrontend (TypeScript/React)
│   └── src/
│       ├── cds/                 CDS workspace and action button
│       │   └── case-builder/    OpenMRS patient API + middleware API types
│       ├── scribe/              Voice Scribe workspace
│       ├── imaging/             Imaging analysis workspace
│       └── ocr/                 OCR workspace
│
├── services/
│   ├── middleware/              FastAPI middleware (CDS orchestration + Scribe)
│   │   ├── api.py               FastAPI app entry point
│   │   └── service/
│   │       ├── cds_router.py    Multi-turn CDS + SSE streaming
│   │       ├── scribe_router.py Audio → FHIR pipeline (OpenMRS endpoints guarded)
│   │       ├── manifest.py      Encounter → concept manifest (29 CIEL concepts)
│   │       ├── fhir_builder.py  FHIR R4 resource construction
│   │       └── ciel_mappings.json  CIEL concept → OpenMRS UUID map
│   │
│   └── knowledge-base/          Knowledge Base daemon (v2.1 index)
│       └── kb/
│           ├── daemon_v2.py     HTTP server (port 4276)
│           ├── retrieval_core_v2.py  BM25 + semantic hybrid retrieval
│           └── embedder.py      EmbedGemma 300M wrapper
│
├── docker/
│   ├── kb/                      KB Dockerfile + entrypoint (auto-downloads MV2 + EmbedGemma)
│   ├── model/                   Model Dockerfile + entrypoint (auto-downloads GGUFs)
│   ├── middleware/              Middleware Dockerfile
│   └── nginx/                   Nginx Dockerfile + config (full stack only)
│
├── training/
│   ├── cds-lora/                CDS LoRA fine-tuning (Stage 1)
│   └── scribe-projector/        AudioProjector training (Stage 2)
│
└── dataset/
    ├── cds/                     CDS conversation dataset pipeline
    └── speech/                  Audio clip generation and assembly
```

---

## Quick Start

### Requirements

| Requirement | Version |
|---|---|
| Docker Engine | ≥ 24 |
| Docker Compose plugin | ≥ 2.20 |
| NVIDIA Container Toolkit *(GPU mode)* | CUDA 12.x |
| GPU VRAM *(GPU mode)* | ≥ 8 GB |
| Disk space | ~20 GB (models + KB index + embeddings) |

### Option A: Engine Only (any EMR)

```bash
git clone https://github.com/brookyale0512/ClinicDx-.git && cd ClinicDx-
cp .env.example .env
make up           # GPU
# or: make up-cpu # CPU only
```

On first start, all artifacts (~6.6 GB) auto-download from HuggingFace. Subsequent starts are instant.

| What downloads | From | Size |
|---|---|---|
| `clinicdx-v1-q8.gguf` | `ClinicDx1/ClinicDx` | 3.9 GB |
| `medasr-encoder.gguf` | `ClinicDx1/ClinicDx` | 401 MB |
| `audio-projector-v3-best.gguf` | `ClinicDx1/ClinicDx` | 46 MB |
| `who_knowledge_vec_v2.mv2` | `ClinicDx1/ClinicDx` | 1.1 GB |
| EmbedGemma 300M | `google/embeddinggemma-300m` | ~1.2 GB |

Once running, the engine API is at `http://localhost:8321`:

```bash
curl http://localhost:8321/api/health
curl -X POST http://localhost:8321/cds/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\nPatient: 5-year-old, fever 39°C for 3 days, cough, rapid breathing.\n<end_of_turn>\n<start_of_turn>model\n"}'
```

### Option B: Full Stack (with nginx + OpenMRS)

```bash
git clone https://github.com/brookyale0512/ClinicDx-.git && cd ClinicDx-
cp .env.example .env

# Generate SSL certs (development)
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/server.key -out certs/server.crt -subj "/CN=localhost"

# Uncomment OPENMRS_URL in .env and set it to your OpenMRS instance

make up-full           # GPU
# or: make up-full-cpu # CPU only
```

| Service | URL |
|---|---|
| ClinicDx API health | `https://localhost/clinicdx-api/api/health` |
| CDS streaming | `https://localhost/clinicdx-api/cds/generate_stream` |

### Option C: OpenMRS Frontend Module

Install the frontend module into your existing OpenMRS O3 instance:

```json
// In spa-assemble-config.json:
{
  "frontendModules": {
    "@openmrs/esm-clinicdx-app": "latest"
  }
}
```

Configure the engine URL in OpenMRS admin:
```json
{
  "@openmrs/esm-clinicdx-app": {
    "middlewareUrl": "http://your-engine-host:8321"
  }
}
```

---

## Air-Gap / Offline Deployment

For facilities without internet access, pre-download all artifacts on a connected machine:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
# Model + KB artifacts
snapshot_download(
    repo_id='ClinicDx1/ClinicDx',
    allow_patterns=['*.gguf', '*.mv2'],
    local_dir='./artifacts/clinicdx'
)
# EmbedGemma 300M
snapshot_download(
    repo_id='google/embeddinggemma-300m',
    local_dir='./artifacts/embeddinggemma-300m'
)
"

# Transfer to target machine
tar -czf clinicdx_artifacts.tar.gz artifacts/
```

On the target machine, copy artifacts into the Docker volumes before starting:

```bash
# Start containers (they will wait for artifacts)
docker compose --profile gpu up -d

# Copy into named volumes
docker cp artifacts/clinicdx/. $(docker volume inspect clinicdx-engine_model_data -f '{{.Mountpoint}}')
docker cp artifacts/clinicdx/. $(docker volume inspect clinicdx-engine_kb_data -f '{{.Mountpoint}}')
docker cp artifacts/embeddinggemma-300m/. $(docker volume inspect clinicdx-engine_hf_cache -f '{{.Mountpoint}}')/embeddinggemma-300m/
```

---

## API Reference

### Middleware — Engine endpoints (EMR-agnostic, port `ENGINE_PORT` default 8321)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Full stack health — model, KB, and middleware status |
| `POST` | `/cds/generate` | Clinical decision support (blocking) |
| `POST` | `/cds/generate_stream` | CDS with SSE token streaming |
| `POST` | `/scribe/process` | Transcription text → structured observations |
| `POST` | `/scribe/process_audio` | Raw audio → structured observations (direct pipeline) |

### Middleware — OpenMRS endpoints (require `OPENMRS_URL` to be set)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/scribe/manifest?encounter_uuid=` | Build encounter concept manifest from OpenMRS |
| `POST` | `/scribe/confirm` | POST confirmed FHIR observations to OpenMRS |

> **Note:** The OpenMRS-dependent endpoints return HTTP 501 if `OPENMRS_URL` is not configured. This allows the engine to run standalone without an OpenMRS instance.

### Knowledge Base Daemon  (`port 4276`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns index version |
| `POST` | `/search` | Query KB: `{"query": "...", "top_k": 5}` |
| `GET` | `/stats` | Index statistics |

### Model Server  (`port 8180`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health |
| `POST` | `/v1/completions` | Text generation (OpenAI-compatible) |
| `POST` | `/v1/audio/extract` | Audio WAV → structured observations (multipart/form-data) |

---

## Configuration

All configuration is via environment variables (copy `.env.example` to `.env`):

**Engine (always required):**

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace token (only if repo is gated) |
| `HF_MODEL_REPO` | `ClinicDx1/ClinicDx` | HuggingFace model repository |
| `HF_KB_REPO` | `ClinicDx1/ClinicDx` | HuggingFace KB repository |
| `ENGINE_PORT` | `8321` | Port the middleware exposes to the host |
| `N_GPU_LAYERS` | `999` | GPU offload layers (0 = CPU only) |
| `MODEL_CTX` | `8192` | Context window size (tokens) |
| `MODEL_PARALLEL` | `1` | Inference slots — **must remain 1** (audio pipeline requirement) |
| `MODEL_THREADS` | `8` | CPU inference threads |
| `KB_SEARCH_MODE` | `rrf` | `rrf` (hybrid), `semantic`, or `lexical` |
| `KB_K` | `5` | Number of KB results per retrieval turn |
| `KB_SNIPPET_CHARS` | `15000` | Max characters per KB snippet |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARN` / `ERROR` |

**Full stack only (with nginx + OpenMRS):**

| Variable | Default | Description |
|---|---|---|
| `OPENMRS_URL` | *(empty)* | OpenMRS backend URL (enables `/scribe/manifest` and `/scribe/confirm`) |
| `OPENMRS_USER` | `admin` | OpenMRS credentials |
| `OPENMRS_PASSWORD` | `Admin123` | OpenMRS credentials |
| `SSL_CERT_PATH` | `./certs/server.crt` | SSL certificate for nginx |
| `SSL_KEY_PATH` | `./certs/server.key` | SSL key for nginx |
| `NGINX_HTTP_PORT` | `80` | Nginx HTTP port (redirects to HTTPS) |
| `NGINX_HTTPS_PORT` | `443` | Nginx HTTPS port |

> **Note:** `MODEL_PARALLEL` must be set to `1`. The audio extraction endpoint (`/v1/audio/extract`) performs blocking KV-cache operations that conflict with parallel slot inference.

---

## OpenMRS Frontend Configuration

The frontend module reads `middlewareUrl` from the OpenMRS config system:

```json
{
  "@openmrs/esm-clinicdx-app": {
    "middlewareUrl": "/clinicdx-api"
  }
}
```

Configure via **System Administration → Advanced Settings** in the OpenMRS admin UI.

- **Full stack (engine + nginx on same host):** use `/clinicdx-api` (default, nginx proxies to middleware)
- **Engine on a separate host:** use `http://<engine-host>:8321`

---

## Testing

### Unit Tests (no running stack required)

```bash
# All unit tests
make test-unit

# Middleware
pytest tests/unit/middleware/ -v

# Knowledge Base
pytest tests/unit/kb/ -v

# Frontend (Jest)
npx jest --config tests/unit/frontend/jest.config.js -v
```

### Integration Tests (requires running stack)

```bash
make up
make test-int

# Individual suites
pytest tests/integration/test_kb_endpoint.py -v
pytest tests/integration/test_model_health.py -v
pytest tests/integration/test_middleware_cds.py -v
pytest tests/integration/test_e2e_scribe.py -v
```

---

## Logging

Every service emits **newline-delimited JSON** on stdout:

```json
{"ts":"2026-03-13T14:00:00Z","level":"INFO","service":"middleware","trace_id":"abc-123","msg":"CDS request received","elapsed_ms":1842}
```

```bash
# Follow all service logs
make logs

# Errors only — across all services
docker compose logs | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.split(' ', 3)[-1])
        if d.get('level') in ('ERROR', 'WARN'):
            print(json.dumps(d, indent=2))
    except: pass
"
```

---

## Training

Training code lives in [`training/`](training/). All three stages can be reproduced independently:

| Stage | Directory | Script | What it trains |
|---|---|---|---|
| 1 — CDS LoRA | `training/cds-lora/` | `train_cds_lora.py` | Single-stage LoRA on 27,592 clinical conversations with reasoning and KB tool-use |
| 2 — AudioProjector | `training/scribe-projector/` | `train_audio_projector.py` | MedASR → LLM projector on 50,000 audio clips |

Dataset preparation pipelines are in [`dataset/`](dataset/).

---

## Supported Concepts (Scribe)

The Scribe module supports 29 CIEL clinical concepts across 7 encounter types:

**Vital Signs:** temperature, blood pressure (systolic/diastolic), pulse, SpO₂, respiratory rate, BMI, weight, height

**Lab Results:** CD4 count, CD4 percent, fasting glucose, finger-stick glucose, post-prandial glucose, serum glucose

**Clinical Observations:** duration of illness, Glasgow Coma Scale, visual analogue pain score, missed medication doses, urine output, fundal height, fetal heart rate, estimated gestational age

**Obstetric / Pediatric:** pre-gestational weight, birth weight, weight gain since last visit, weight on admission, head circumference, MUAC

---

## Roadmap

- [ ] Multi-language Scribe (Swahili, Amharic, Hausa, Yoruba)
- [ ] Differential diagnosis confidence scores
- [ ] Federated learning for multi-facility model improvement
- [ ] Medication reconciliation and drug interaction alerts
- [ ] Integration with OpenMRS Reporting Framework
- [ ] Offline model update via USB/sneakernet

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

```bash
# Clone and set up development environment
git clone https://github.com/brookyale0512/ClinicDx-.git
cd ClinicDx-

# Run unit tests
make test-unit
```

Areas where contributions are especially needed:
- Clinical validation studies
- Additional CIEL concept mappings
- Regional language support for Scribe
- Documentation translations

---

## Built With

| Component | Project |
|---|---|
| EMR Platform | [OpenMRS O3](https://openmrs.org/) |
| Base LLM | [Google MedGemma 4B-IT](https://huggingface.co/google/medgemma-4b-it) |
| Inference Runtime | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Clinical Terminology | [CIEL](https://www.cielterminology.org/) |
| Knowledge Base | [memvid](https://github.com/Oaynerad/memvid) |
| Audio Encoder | MedASR (Conformer, LASR architecture) |
| FHIR Standard | [HL7 FHIR R4](https://hl7.org/fhir/R4/) |

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see [LICENSE](LICENSE) for details.

Model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

<div align="center">

**[clinicdx.org](https://clinicdx.org)** · **[HuggingFace](https://huggingface.co/ClinicDx1/ClinicDx)** · **[GitHub](https://github.com/brookyale0512/ClinicDx-)** · **[Issues](https://github.com/brookyale0512/ClinicDx-/issues)**

*Built for clinicians in under-resourced settings. Every observation captured, every diagnosis supported.*

</div>
