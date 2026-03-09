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
│ (port 4276) │   │                                       │
│              │   │  MedASR encoder (105M, frozen)        │
│ who_know_vec │   │     ↓                                 │
│  _v2.mv2     │   │  AudioProjector (11.8M, trained)      │
│              │   │     ↓                                 │
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
│   └── knowledge-base/        # Local KB HTTP daemon (port 4278, v2)
│       ├── kb/
│       │   ├── retrieval_core_v2.py  # KBRetriever v2 (WHO v2 index, RRF, intent-rerank)
│       │   ├── daemon_v2.py          # ThreadingHTTPServer (port 4278, v2 index)
│       │   ├── retrieval_core.py     # [legacy] v1 retriever
│       │   ├── daemon.py             # [legacy] v1 daemon
│       │   └── client.py             # HTTP client helpers
│       └── tests/
│
├── training/
│   ├── cds-lora/              # CDS KB tool-use LoRA fine-tuning
│   │   ├── train.py              # SFT trainer (v2, single-node, medgemma-4b-it)
│   │   ├── config.yaml
│   │   ├── data_loader.py
│   │   ├── merge_lora.py
│   │   ├── validate.py
│   │   ├── prep_cycle1.py        # Enriched shards → quality filter → train/val split
│   │   └── scripts/              # [stale] run_training.sh, deploy.sh, setup_env.sh
│   │
│   ├── scribe-projector/      # AudioProjector training (only trainable component)
│   │   ├── train_audio_projector.py
│   │   ├── train_projector.py
│   │   ├── data_loader.py
│   │   ├── validate_scribe.py
│   │   └── configs/
│   │
│   └── kb-tool-use-lora/      # [ARCHIVED] superseded by cds-lora/ (v2)
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

ClinicDx uses a single fine-tuned model: **[`medgemma_cds_think_v1`](https://huggingface.co/ClinicDx1/ClinicDx)**

This model is derived from [Google's MedGemma](https://huggingface.co/google/medgemma-4b-it) through two training stages:

```
MedGemma 4B-IT (Google base, multimodal, 4.3B params)
      │
      ↓  Stage 1: CDS KB tool-use LoRA SFT (r=64, all 7 projection modules)
      │    Base loaded as causal LM (vision tower frozen)
      │    Format: [CDS] tag + <think>/<KB_QUERY> XML multi-turn tool-use
      │    Dataset: ~8.5k CDS cases, think_format
      │    Best checkpoint: checkpoint-3712 (eval_loss 0.192, token_acc 92.3%)
medgemma_cds_think_v1  ← deployed CDS model (LoRA merged, Feb 28 2026)
      │
      ↓  Stage 2: AudioProjector training only (11.8M params)
      │    Base LLM frozen — CDS capability unchanged
      │    Trained on 23k pre-cached audio clips (MedASR encoder output)
      │    from CEIL speech dataset (537k clips total)
medgemma_cds_think_v1 + projector_final.pt  ← full Scribe model (deployed)
```

**Model weights:** Hosted on Hugging Face. This repository contains training and serving code only.

### Model Registry

- **Repository:** `https://huggingface.co/ClinicDx1/ClinicDx`
- **Latest published revision:** `a8f17b3f3caf3f30319d2ef42a8caf9523304ddf`
- **Deployment recommendation:** Pin to an explicit revision in production for reproducibility.

```bash
# Pull model artifacts to the expected server path
hf download ClinicDx1/ClinicDx --repo-type model \
  --local-dir /var/www/ClinicDx/model/medgemma_cds_think_v1
```

### Knowledge Base

The model queries a local KB during inference (CDS mode). The KB uses a single [memvid](https://github.com/Oaynerad/memvid) v2 index:

| File | Contents |
|---|---|
| `who_knowledge_vec_v2.mv2` | WHO/MSF clinical guidelines, Africa-focused protocols (v2, RRF-ready, 132K frames) |

Default location: `/var/www/kbToolUseLora/kb/` (override with `KB_INDEX_DIR` env var).

---

## Quick Start

### 1. Start the Knowledge Base Daemon

```bash
cd services/knowledge-base
pip install -r requirements.txt
export KB_INDEX_DIR=/path/to/kb/indexes
python3 -m kb.daemon_v2
```

### 2. Start the Unified Model Server

```bash
cd services/unified-model-server
pip install -r requirements.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python serve_unified.py \
  --model-dir /path/to/medgemma_cds_think_v1 \
  --projector /path/to/projector_final.pt \
  --port 8000
```

The server loads three components: MedASR encoder (105M, frozen), AudioProjector (11.8M, trained), and the merged CDS model (4.3B, frozen). Only the projector weights vary between deployments.

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
| `MODEL_NAME` | `/var/www/ClinicDx/model/medgemma_cds_think_v1` | Path to merged CDS model |
| `OPENMRS_URL` | `http://localhost:8080/openmrs` | OpenMRS base URL |
| `OPENMRS_USER` | `admin` | OpenMRS credentials |
| `OPENMRS_PASSWORD` | `Admin123` | OpenMRS credentials |

---

## Training

See [`training/`](training/) for all training code.

### CDS Model Training Pipeline

The next CDS model (`medgemma_cds_v2`) is trained using a **cyclic SFT approach** on production-enriched clinical cases:

1. **Dataset enrichment** (`enrich_production.py`): 29,690 real OpenMRS patient cases are processed by `agent2.py` — a multi-turn ReAct agent that queries the live KB (WHO guidelines + WikiMed) and generates structured CDS outputs in `<think>/<KB_QUERY>/<KB_RESULT>` format. Enrichment runs continuously; training cycles start as batches complete.

2. **Quality filtering** (`prep_cycle1.py`): Enriched records are filtered for structural completeness (`has_exact_six_sections`, `actions_evidence_mapped`, no `meta_contamination`) before being passed to the trainer.

3. **LoRA SFT** (`training/cds-lora/`): SFT on the enriched dataset using LoRA (r=64, all 7 projection modules) on `medgemma-4b-it` (causal LM, vision frozen). Training is cyclic — each cycle is one pass through the current batch of enriched cases. When all 29k cases have been seen once, a final 2-epoch run trains on the full dataset.

4. **Merge** (`training/cds-lora/merge_lora.py`): Best LoRA checkpoint is merged into the base model weights to produce the standalone `medgemma_cds_v2` model.

### AudioProjector Training

The Scribe projector is trained separately, after the CDS LoRA is already merged:

- **Base model**: `medgemma_cds_think_v1` (frozen — CDS capability cannot degrade)
- **Only trainable component**: `AudioProjector` (11.8M params)
- **Training data**: 23k pre-cached MedASR encoder embeddings from the CEIL speech dataset (537k clips available; training uses the pre-computed subset)
- **Script**: `training/scribe-projector/train_audio_projector.py`
- **Output**: `projector_final.pt` — loaded alongside the frozen model at inference

### Training Data Format

Each training record is a pre-formatted Gemma 3 multi-turn conversation:

```
<bos><start_of_turn>user
[CDS]
Provide evidence-based clinical decision support for this patient.
<patient case data>
<end_of_turn>
<start_of_turn>model
<think>Clinical reasoning...</think>
<KB_QUERY>search query terms</KB_QUERY><end_of_turn>
<start_of_turn>user
<KB_RESULT source="WikiMed" score="51.2">...retrieved text...</KB_RESULT><end_of_turn>
<start_of_turn>model
<think>Further reasoning...</think>
<KB_QUERY>second query</KB_QUERY><end_of_turn>
... (2–3 KB query turns) ...
<start_of_turn>model
<think>Final synthesis...</think>
## Clinical Assessment
...
## Differential Diagnoses
...
## Recommended Investigations
...
## Treatment Plan
...
## Patient Education & Follow-Up
...<end_of_turn>
```

### Training Folders

| Folder | What it trains |
|---|---|
| `training/cds-lora/` | CDS KB tool-use LoRA on MedGemma (`train.py`) |
| `training/scribe-projector/` | AudioProjector mapping MedASR → MedGemma space |
| `training/kb-tool-use-lora/` | \[ARCHIVED\] KB tool-use format LoRA (earlier experiment, superseded by `cds-lora/`) |

Dataset preparation scripts are in [`dataset/`](dataset/).

---

## License

MPL-2.0 — see [LICENSE](LICENSE).

This module is built on top of [OpenMRS O3](https://openmrs.org/), [MedGemma](https://huggingface.co/google/medgemma-4b-it) (Google), and the [CIEL Clinical Terminology](https://www.cielterminology.org/).
