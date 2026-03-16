# =============================================================================
# ClinicDx V1 — KB Daemon Container
# =============================================================================
# Runs the KB daemon v2, serving who_knowledge_vec_v2.mv2 on port 4276.
# The index file is stored in a named Docker volume (/kb_data) and auto-
# downloaded from HuggingFace on first start if absent.
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="ClinicDx" \
      version="1.0.0" \
      description="ClinicDx Knowledge Base Daemon v2"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── HuggingFace CLI (for artifact download) ───────────────────────────────────
RUN pip install --no-cache-dir huggingface_hub[cli]

# ── Application dependencies ──────────────────────────────────────────────────
WORKDIR /app
COPY services/knowledge-base/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY services/knowledge-base/ /app/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/kb/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Data volume (index files) ─────────────────────────────────────────────────
VOLUME ["/kb_data"]

# ── Runtime config (overridable via .env or docker-compose) ───────────────────
ENV KB_PORT=4276 \
    KB_SEARCH_MODE=rrf \
    KB_K=5 \
    KB_SNIPPET_CHARS=15000 \
    KB_THRESHOLD=0.0 \
    KB_DATA_DIR=/kb_data \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 4276

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:4276/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
