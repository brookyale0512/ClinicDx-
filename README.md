# ClinicDx — Unified Clinical AI for OpenMRS

**ClinicDx** is an OpenMRS microfrontend module that brings AI-powered clinical intelligence directly into the EMR workflow. It integrates a single fine-tuned multimodal model (`medgemma_cds_think_v1`) to provide four capabilities in a single system:

| Feature | Description |
|---|---|
| **CDS** | Clinical Decision Support with multi-turn KB tool-use |
| **Scribe** | Voice-to-FHIR: audio → structured OpenMRS observations |
| **OCR** | Document/prescription digitisation |
| **Imaging** | Clinical image analysis |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  OpenMRS O3 Frontend                    │
│  openmrs-esm-clinicdx-app  (TypeScript / React)         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │   CDS    │ │  Scribe  │ │   OCR    │ │  Imaging  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
└───────┼────────────┼────────────┼──────────────┼────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│           Python Middleware  (FastAPI : 8321)            │
│   cds_router.py   scribe_router.py   concept_extractor  │
└──────────────┬──────────────────────────────────────────┘
               │ Multi-turn ReAct loop (KB tool-use for CDS)
               │ Direct audio → observations (Scribe)
     ┌─────────┴──────────┐
     │                    │
     ▼                    ▼
┌──────────────┐   ┌──────────────────────────────────────┐
│  KB Daemon   │   │  Unified Model Server  (FastAPI:8000) │
│  (port 4276) │   │                                       │
│              │   │  MedASR encoder (105M, frozen)        │
│  who_know.   │   │     ↓                                 │
│  mv2         │   │  AudioProjector (11.8M, trained)      │
│  wikimed.mv2 │   │     ↓                                 │
│              │   │  MedGemma CDS (4.3B, LoRA merged)     │
│  /search     │   │                                       │
│  (lex BM25)  │   │  POST /v1/completions  (CDS text)     │
└──────────────┘   │  POST /v1/audio/extract (Scribe)      │
                   └──────────────────────────────────────┘
```

### CDS Data Flow (multi-turn ReAct)
1. Frontend sends patient case → middleware `/cds/generate`
2. Middleware builds Gemma chat prompt and calls model server
3. Model emits `<KB_QUERY>diagnosis term</KB_QUERY>`
4. Middleware queries KB daemon → injects `<KB_RESULT>` back
5. Up to 4 turns until `## Clinical Assessment` section appears
6. Cleaned markdown response returned to frontend

### Scribe Data Flow (direct audio)
1. Doctor records audio in browser → frontend sends to middleware `/scribe/process_audio`
2. Middleware POSTs audio bytes to model server `/v1/audio/extract`
3. MedASR encodes audio → AudioProjector projects to LLM space → MedGemma decodes
4. Structured `label: value` observations returned
5. Middleware maps observations to CIEL concept codes → builds FHIR R4 payloads
6. Doctor confirms → middleware POSTs FHIR observations to OpenMRS

---

## Repository Layout

```
clinicdx/
├── openmrs-module/          # TypeScript/React ESM microfrontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── webpack.config.js
│   ├── translations/
│   └── src/
│       ├── index.ts         # Module entry + config schema
│       ├── routes.json      # OpenMRS slot registrations
│       ├── cds/             # CDS workspace + action button
│       │   └── case-builder/  # OpenMRS API + CDS API layer
│       ├── scribe/          # Voice Scribe workspace
│       ├── imaging/         # Imaging analysis workspace
│       └── ocr/             # OCR workspace
│
├── services/
│   ├── unified-model-server/  # Main inference server (port 8000)
│   │   ├── serve_unified.py   # FastAPI app — CDS + Scribe endpoints
│   │   └── modeling/
│   │       ├── gemma3_audio.py  # Gemma3WithAudioModel + AudioProjector
│   │       └── processor.py
│   │
│   ├── middleware/            # Request routing + FHIR building (port 8321)
│   │   └── service/
│   │       ├── api.py           # FastAPI app entry
│   │       ├── cds_router.py    # Multi-turn KB tool-use CDS
│   │       ├── scribe_router.py # Audio→FHIR Scribe pipeline
│   │       ├── fhir_builder.py  # FHIR R4 resource construction
│   │       ├── manifest.py      # OpenMRS encounter → concept manifest
│   │       ├── concept_extractor.py
│   │       ├── transcribe.py
│   │       ├── audio_pipeline.py
│   │       ├── projector.py
│   │       └── ciel_mappings.json  # CIEL concept → UUID/code map
│   │
│   └── knowledge-base/        # Local KB HTTP daemon (port 4276)
│       ├── kb/
│       │   ├── retrieval_core.py  # KBRetriever (memvid, thread-safe)
│       │   ├── daemon.py          # Stdlib ThreadingHTTPServer
│       │   └── client.py          # HTTP client helpers
│       └── tests/
│
├── training/
│   ├── cds-lora/              # LoRA fine-tuning for CDS reasoning
│   │   ├── train_cds_lora.py
│   │   ├── config.yaml
│   │   ├── data_loader.py
│   │   ├── merge_lora.py
│   │   ├── eval_medqa.py
│   │   └── scripts/           # run_training.sh, deploy.sh, setup_env.sh
│   │
│   ├── scribe-projector/      # Audio projector training
│   │   ├── train_projector.py
│   │   ├── train_text_sft.py
│   │   ├── data_loader.py
│   │   ├── evaluate.py
│   │   └── configs/
│   │
│   └── kb-tool-use-lora/      # KB-aware LoRA (2-query tool-use format)
│       ├── train.py
│       ├── config.yaml
│       └── validate_kb_live.py
│
└── dataset/
    ├── cds/                   # CDS dataset preparation
    │   └── build_sft_dataset.py
    └── speech/                # Speech dataset pipeline
        ├── generate_audio.py
        ├── assemble_training.py
        └── build_text_sft_data.py
```

---

## The Model

ClinicDx uses a single fine-tuned model: **`medgemma_cds_think_v1`**

This model is derived from [Google's MedGemma](https://huggingface.co/google/medgemma-4b-it) through three training stages:

```
MedGemma 4B-IT (base)
      ↓  Stage 1: SFT on CDS reasoning dataset (200k examples)
MedGemma-SFT
      ↓  Stage 2: LoRA + GRPO on KB tool-use format
medgemma_cds_think_v1  ← production model (LoRA merged)
      ↓  Stage 3: AudioProjector training (11.8M params, frozen base)
medgemma_cds_think_v1 + projector_final.pt  ← full Scribe model
```

**Model weights:** Hosted on HuggingFace (see model card). The repository contains only training code.

### Knowledge Base

The model queries a local KB during inference (CDS mode). The KB contains two [memvid](https://github.com/Oaynerad/memvid) indexes:

| File | Contents |
|---|---|
| `who_knowledge.mv2` | WHO clinical guidelines, Africa-focused protocols |
| `wikimed.mv2` | WikiMed medical reference corpus |

Default location: `/var/www/kbToolUseLora/kb/` (override with `KB_INDEX_DIR` env var).

---

## Quick Start

### 1. Start the Knowledge Base Daemon

```bash
cd services/knowledge-base
pip install -r requirements.txt
export KB_INDEX_DIR=/path/to/kb/indexes
python -m kb.daemon 4276
```

### 2. Start the Unified Model Server

```bash
cd services/unified-model-server
pip install -r requirements.txt
python serve_unified.py \
  --model-dir /path/to/medgemma_cds_think_v1 \
  --projector /path/to/projector_final.pt \
  --port 8000
```

### 3. Start the Middleware

```bash
cd services/middleware
pip install -r requirements.txt
export MODEL_SERVER_URL=http://localhost:8000
export KB_URL=http://localhost:4276
export OPENMRS_URL=http://localhost:8080/openmrs
export OPENMRS_USER=admin
export OPENMRS_PASSWORD=Admin123
uvicorn service.api:app --host 0.0.0.0 --port 8321
```

### 4. Build and Deploy the OpenMRS Module

```bash
cd openmrs-module
npm install
npm run build
# Copy dist/ to your OpenMRS frontend deployment
```

Set `middlewareUrl` in OpenMRS config to point to your middleware (`http://localhost:8321`).

---

## API Reference

### Unified Model Server (`port 8000`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List models (OpenAI compat) |
| `POST` | `/v1/completions` | Text generation for CDS (OpenAI compat) |
| `POST` | `/v1/audio/extract` | Audio → structured observations |

### Middleware (`port 8321`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Full health check |
| `POST` | `/cds/generate` | Multi-turn CDS with KB tool-use |
| `POST` | `/cds/generate_stream` | CDS with SSE token streaming |
| `GET` | `/scribe/manifest?encounter_uuid=` | Build concept manifest |
| `POST` | `/scribe/process` | Transcription → FHIR observations |
| `POST` | `/scribe/process_audio` | Audio → FHIR (direct, no text step) |
| `POST` | `/scribe/confirm` | POST confirmed observations to OpenMRS |

### KB Daemon (`port 4276`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Index statistics |
| `POST` | `/search` | Query KB (`{"query": "...", "k": 3}`) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KB_INDEX_DIR` | `/var/www/kbToolUseLora/kb` | Path to `.mv2` KB index files |
| `MODEL_SERVER_URL` | `http://10.128.0.4:8000` | Unified model server URL |
| `KB_URL` | `http://10.128.0.4:4276` | KB daemon URL |
| `MODEL_NAME` | `/var/www/ClinicDx/model/medgemma_cds_think_v1` | Model identifier |
| `OPENMRS_URL` | `http://localhost:8080/openmrs` | OpenMRS base URL |
| `OPENMRS_USER` | `admin` | OpenMRS credentials |
| `OPENMRS_PASSWORD` | `Admin123` | OpenMRS credentials |

---

## Training

See [`training/`](training/) for all training code.

| Folder | What it trains |
|---|---|
| `training/cds-lora/` | CDS reasoning LoRA on MedGemma (`train_cds_lora.py`) |
| `training/scribe-projector/` | AudioProjector mapping MedASR → MedGemma space |
| `training/kb-tool-use-lora/` | KB tool-use format LoRA (2-query ReAct style) |

Dataset preparation scripts are in [`dataset/`](dataset/).

---

## License

MPL-2.0 — see [LICENSE](LICENSE).

This module is built on top of [OpenMRS O3](https://openmrs.org/), [MedGemma](https://huggingface.co/google/medgemma-4b-it) (Google), and the [CIEL Clinical Terminology](https://www.cielterminology.org/).
