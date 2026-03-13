<div align="center">

<img src="docs/assets/clinicdx-logo.png" alt="ClinicDx" width="120" />

# ClinicDx

### Offline AI Clinical Intelligence for Sub-Saharan Africa

**Clinical Decision Support · Voice Scribe · Document OCR · Imaging Analysis**

Built on [OpenMRS O3](https://openmrs.org/) · Powered by [MedGemma](https://huggingface.co/google/medgemma-4b-it) · Runs fully offline

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)
[![OpenMRS](https://img.shields.io/badge/OpenMRS-O3-blue?logo=data:image/svg+xml;base64,)](https://openmrs.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-ClinicDx1%2FClinicDx-yellow)](https://huggingface.co/ClinicDx1/ClinicDx)
[![Website](https://img.shields.io/badge/Website-clinicdx.org-informational)](https://clinicdx.org)

[**Website**](https://clinicdx.org) · [**Model on HuggingFace**](https://huggingface.co/ClinicDx1/ClinicDx) · [**Documentation**](docs/) · [**Report a Bug**](https://github.com/brookyale0512/ClinicDx-/issues)

</div>

---

## Overview

ClinicDx is an **open-source AI layer for OpenMRS O3** designed specifically for resource-limited clinical settings in sub-Saharan Africa. It brings four AI-powered capabilities into a single microfrontend module — all running on local hardware with no internet dependency after initial setup.

| Capability | Description |
|---|---|
| **Clinical Decision Support (CDS)** | Structured 6-section assessments with evidence citations from a local WHO knowledge base, generated via a multi-turn retrieval-augmented reasoning loop |
| **Voice Scribe** | Speak clinical observations in natural language — Scribe extracts structured FHIR observations and writes them directly to OpenMRS |
| **Document OCR** | Digitise referral letters, lab reports, and prescriptions |
| **Imaging Analysis** | AI-assisted interpretation of clinical images |

Everything runs on a single 4B-parameter multimodal model fine-tuned from [Google MedGemma](https://huggingface.co/google/medgemma-4b-it). The model, knowledge base, and all services operate **fully offline** — no data leaves the facility.

---

## Key Features

- **Truly offline** — after initial model download, zero network dependency
- **FHIR R4 native** — all Scribe output posts directly to OpenMRS as structured observations
- **CIEL terminology** — maps to the full CIEL clinical concept vocabulary
- **Multimodal** — single model handles text (CDS), audio (Scribe), and vision (OCR/Imaging)
- **Evidence-grounded** — CDS cites WHO guidelines and clinical references from a local vector knowledge base
- **OpenMRS O3 native** — ships as a standard ESM microfrontend, installs like any other OpenMRS module
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

### Training Stages

```
Google MedGemma 4B-IT  (base)
       │
       ▼  Stage 1 — CDS SFT
       │  LoRA (r=64, α=128) on 27,592 quality-filtered clinical conversations
       │  Input masking: only model output turns trained (64% context masked)
       │  Best checkpoint: step 4,000  |  eval loss 0.4758  |  accuracy 86.25%
       │
       ▼  Stage 2 — KB Tool-Use LoRA
       │  Trained on multi-turn ReAct format with KB retrieval traces
       │  Teaches the model when and how to query the knowledge base
       │
       ▼  Merge LoRA adapters  →  medgemma_cds_think_v1  (production CDS model)
       │
       ▼  Stage 3 — AudioProjector Training  (base model frozen)
          2-layer MLP + LayerNorm projector: MedASR (512-dim) → LLM space (2560-dim)
          50,000 synthetic clinical audio clips  |  10 epochs
          Best checkpoint: step 40,000  |  val LM 0.1042  |  key accuracy 84%
          11,806,720 trainable parameters
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OpenMRS O3 Frontend (Browser)                  │
│   @openmrs/esm-clinicdx-app  v2.0.0  (TypeScript / React / MFE) │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │   CDS    │  │  Scribe  │  │   OCR    │  │    Imaging     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬────────┘  │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │  HTTPS      │             │                 │
        ▼             ▼             ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│              Nginx Reverse Proxy  (:443 HTTPS)                    │
│  /clinicdx-api/*  →  middleware:8080                             │
│  /openmrs/*       →  openmrs:8080                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│         FastAPI Middleware  (Python 3.11, port 8080)              │
│                                                                   │
│   cds_router.py       ─── Multi-turn ReAct CDS + SSE streaming  │
│   scribe_router.py    ─── Audio → FHIR pipeline                  │
│   manifest.py         ─── OpenMRS encounter → concept manifest   │
│   fhir_builder.py     ─── FHIR R4 observation construction       │
└───────────────────┬──────────────────────┬────────────────────── ┘
                    │                      │
          ┌─────────┘                      └─────────┐
          ▼                                          ▼
┌─────────────────────┐              ┌───────────────────────────┐
│   KB Daemon v2      │              │  llama-server  (port 8180) │
│   port 4276         │              │                            │
│                     │              │  ┌──────────────────────┐  │
│  WHO Guidelines     │              │  │  MedASR Encoder      │  │
│  MSF Protocols      │◄─ queries ──│  │  (frozen, 105M)      │  │
│  WikiMed Corpus     │              │  └──────────┬───────────┘  │
│                     │              │             ▼              │
│  BM25 + Semantic    │              │  ┌──────────────────────┐  │
│  Hybrid Retrieval   │              │  │  AudioProjector v3   │  │
│  (v2 index)         │              │  │  (11.8M, trained)    │  │
└─────────────────────┘              │  └──────────┬───────────┘  │
                                     │             ▼              │
                                     │  ┌──────────────────────┐  │
                                     │  │  ClinicDx V1 LLM     │  │
                                     │  │  (4.3B, Q8 GGUF)     │  │
                                     │  └──────────────────────┘  │
                                     └───────────────────────────┘
```

### CDS Flow — Multi-Turn ReAct with KB Tool-Use

```
1.  Frontend sends patient case  →  POST /cds/generate_stream
2.  Middleware builds Gemma chat prompt with encounter context
3.  Model streams thinking block  →  emits <KB_QUERY>term</KB_QUERY>
4.  Middleware queries KB daemon  →  injects <KB_RESULT>evidence</KB_RESULT>
5.  Up to 4 retrieval turns until structured response is complete
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
├── openmrs-module/              OpenMRS O3 ESM microfrontend (TypeScript/React)
│   └── src/
│       ├── cds/                 CDS workspace and action button
│       │   └── case-builder/    OpenMRS patient API + middleware API types
│       ├── scribe/              Voice Scribe workspace
│       ├── imaging/             Imaging analysis workspace
│       └── ocr/                 OCR workspace
│
├── services/
│   ├── middleware/              FastAPI middleware (canonical V1 source)
│   │   ├── api.py               FastAPI app entry point
│   │   ├── cds_router.py        Multi-turn CDS + SSE streaming
│   │   ├── scribe_router.py     Audio → FHIR pipeline
│   │   ├── manifest.py          Encounter → concept manifest (29 CIEL concepts)
│   │   ├── fhir_builder.py      FHIR R4 resource construction
│   │   └── ciel_mappings.json   CIEL concept → OpenMRS UUID map
│   │
│   └── kb/                      Knowledge Base daemon (v2 index)
│       └── kb/
│           ├── daemon_v2.py     HTTP server (port 4276)
│           └── retrieval_core_v2.py  BM25 + semantic hybrid retrieval
│
├── training/
│   ├── cds-lora/                CDS LoRA fine-tuning (Stage 1 + 2)
│   ├── scribe-projector/        AudioProjector training (Stage 3)
│   └── kb-tool-use-lora/        KB tool-use LoRA
│
└── dataset/
    ├── cds/                     CDS conversation dataset pipeline
    └── speech/                  Audio clip generation and assembly
```

---

## Quick Start (Docker Compose)

### Requirements

| Requirement | Version |
|---|---|
| Docker Engine | ≥ 24 |
| Docker Compose plugin | ≥ 2.20 |
| NVIDIA Container Toolkit *(GPU mode)* | CUDA 12.x |
| GPU VRAM *(GPU mode)* | ≥ 8 GB |
| Disk space | ~20 GB (models + KB index) |

### 1 — Configure

```bash
cp .env.example .env
# Set HF_TOKEN if the HuggingFace repo requires authentication
```

### 2 — SSL Certificates (development)

```bash
mkdir -p certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout certs/server.key -out certs/server.crt -subj "/CN=localhost"
```

### 3 — Start the Stack

**GPU (recommended):**
```bash
make up
# or: docker compose --profile gpu up -d
```

**CPU only:**
```bash
make up-cpu
# or: docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

On first start, the model container automatically downloads the three GGUF files from [`ClinicDx1/ClinicDx`](https://huggingface.co/ClinicDx1/ClinicDx). Subsequent starts are instant.

### 4 — Verify

```bash
make smoke
```

### 5 — Access

| Service | URL |
|---|---|
| OpenMRS O3 | `https://localhost/openmrs` |
| ClinicDx API health | `https://localhost/clinicdx-api/api/health` |
| Model server health | `http://localhost:8180/health` |

---

## Manual / Air-Gap Deployment

For deployments without internet access, pre-download all artifacts:

```bash
# On a machine with internet access
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ClinicDx1/ClinicDx",
    allow_patterns=["*.gguf"],
    local_dir="./artifacts/gguf"
)
EOF

# Transfer artifacts to target machine and map to Docker volumes
tar -czf clinicdx_artifacts.tar.gz artifacts/
```

---

## API Reference

### Middleware  (`/clinicdx-api`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Full stack health — model, KB, and middleware status |
| `POST` | `/cds/generate` | Clinical decision support (blocking) |
| `POST` | `/cds/generate_stream` | CDS with SSE token streaming |
| `GET` | `/scribe/manifest?encounter_uuid=` | Build encounter concept manifest |
| `POST` | `/scribe/process` | Transcription text → FHIR observations |
| `POST` | `/scribe/process_audio` | Raw audio → FHIR observations (direct pipeline) |
| `POST` | `/scribe/confirm` | POST confirmed observations to OpenMRS |

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

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace token for model download |
| `HF_MODEL_REPO` | `ClinicDx1/ClinicDx` | HuggingFace model repository |
| `N_GPU_LAYERS` | `999` | GPU offload layers (0 = CPU only) |
| `MODEL_CTX` | `8192` | Context window size (tokens) |
| `MODEL_PARALLEL` | `1` | Inference slots — **must remain 1** (audio pipeline requirement) |
| `OPENMRS_URL` | `https://172.18.0.1/openmrs` | OpenMRS backend URL |
| `OPENMRS_USER` | `admin` | OpenMRS credentials |
| `OPENMRS_PASSWORD` | `Admin123` | OpenMRS credentials |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARN` / `ERROR` |

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

Configure via **System Administration → Advanced Settings** in the OpenMRS admin UI. For local development without Docker, set it to `http://localhost:8321`.

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
| 1 — CDS SFT | `training/cds-lora/` | `train_cds_lora.py` | CDS reasoning LoRA on 27,592 clinical conversations |
| 2 — KB Tool-Use | `training/kb-tool-use-lora/` | `train.py` | KB query format LoRA |
| 3 — AudioProjector | `training/scribe-projector/` | `train_audio_projector.py` | MedASR → LLM projector on 50,000 audio clips |

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

This project is licensed under the **Mozilla Public License 2.0** — see [LICENSE](LICENSE) for details.

Model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

<div align="center">

**[clinicdx.org](https://clinicdx.org)** · **[HuggingFace](https://huggingface.co/ClinicDx1/ClinicDx)** · **[GitHub](https://github.com/brookyale0512/ClinicDx-)** · **[Issues](https://github.com/brookyale0512/ClinicDx-/issues)**

*Built for clinicians in under-resourced settings. Every observation captured, every diagnosis supported.*

</div>
