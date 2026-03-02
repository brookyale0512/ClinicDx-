# ClinicDx — Deployment Guide

## Prerequisites

| Component | Requirement |
|---|---|
| GPU server | CUDA-capable, ≥ 16GB VRAM (for 4B model in bfloat16) |
| CPU server | Can be same machine or separate |
| OpenMRS | O3 (3.x) with FHIR2 module |
| Python | 3.10+ |
| Node.js | 18+ |
| ffmpeg | Required for non-WAV audio conversion |

---

## Step 1 — Obtain Model Weights

Download from HuggingFace (links in model card):

```
medgemma_cds_think_v1/     ← merged CDS model (LoRA baked in)
medASR/                    ← MedASR Wav2Vec2 encoder
projector_final.pt         ← trained AudioProjector checkpoint
```

Place them at paths referenced by config, or set env vars.

---

## Step 2 — Knowledge Base Indexes

Download `who_knowledge.mv2` and `wikimed.mv2` from the release assets
and place them in `/var/www/kbToolUseLora/kb/` or set `KB_INDEX_DIR`.

Start the daemon:
```bash
cd services/knowledge-base
pip install memvid-sdk
export KB_INDEX_DIR=/path/to/kb
python -m kb.daemon 4276
```

Verify:
```bash
curl http://localhost:4276/health
# {"ok": true}
```

---

## Step 3 — Unified Model Server

```bash
cd services/unified-model-server
pip install -r requirements.txt

python serve_unified.py \
  --model-dir /path/to/medgemma_cds_think_v1 \
  --projector /path/to/projector_final.pt \
  --host 0.0.0.0 \
  --port 8000
```

The server loads three components at startup (may take 1-2 minutes on first run).

Verify:
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Step 4 — Middleware

```bash
cd services/middleware
pip install -r requirements.txt

export MODEL_SERVER_URL=http://localhost:8000
export KB_URL=http://localhost:4276
export OPENMRS_URL=http://your-openmrs-host:8080/openmrs
export OPENMRS_USER=admin
export OPENMRS_PASSWORD=your_password

uvicorn service.api:app --host 0.0.0.0 --port 8321 --workers 1
```

> **Note:** Use `--workers 1` — the service delegates compute to the model server, and multiple workers would not share GPU state correctly.

Verify:
```bash
curl http://localhost:8321/api/health
```

---

## Step 5 — OpenMRS Module

```bash
cd openmrs-module
npm install
npm run build
```

Copy `dist/` to your OpenMRS frontend asset server, then register the module in `importmap.json`:

```json
{
  "@openmrs/esm-clinicdx-app": "http://your-host/dist/openmrs-esm-clinicdx-app.js"
}
```

In OpenMRS O3 admin, configure:
```
middlewareUrl = http://your-middleware-host:8321
```

---

## Systemd Service Files (example)

### KB Daemon
```ini
[Unit]
Description=ClinicDx KB Daemon
After=network.target

[Service]
WorkingDirectory=/opt/clinicdx/services/knowledge-base
ExecStart=/usr/bin/python3 -m kb.daemon 4276
Restart=always
Environment=KB_INDEX_DIR=/var/www/kbToolUseLora/kb

[Install]
WantedBy=multi-user.target
```

### Unified Model Server
```ini
[Unit]
Description=ClinicDx Unified Model Server
After=network.target

[Service]
WorkingDirectory=/opt/clinicdx/services/unified-model-server
ExecStart=/usr/bin/python3 serve_unified.py --port 8000
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```
