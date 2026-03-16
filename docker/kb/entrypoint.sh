#!/usr/bin/env bash
# =============================================================================
# KB container entrypoint
# =============================================================================
# 1. Ensures the MV2 knowledge base index is available (downloads if missing)
# 2. Ensures EmbedGemma 300M is available (downloads if missing)
# 3. Starts the KB daemon v2
# =============================================================================
set -euo pipefail

WORK_DIR="/kb_data"
INDEX_FILENAME="who_knowledge_vec_v2.mv2"
WORK_INDEX="${WORK_DIR}/${INDEX_FILENAME}"
HF_KB_REPO="${HF_KB_REPO:-ClinicDx1/ClinicDx}"
HF_EMBED_MODEL="google/embeddinggemma-300m"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-/hf_cache/embeddinggemma-300m}"
KB_PORT="${KB_PORT:-4276}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

log() {
  printf '{"ts":"%s","level":"INFO","service":"kb-entrypoint","msg":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}
err() {
  printf '{"ts":"%s","level":"ERROR","service":"kb-entrypoint","msg":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

mkdir -p "${WORK_DIR}"

# ── Step 1: MV2 knowledge base index ──────────────────────────────────────────

if [ -f "${WORK_INDEX}" ]; then
  log "Found MV2 index: ${WORK_INDEX} ($(du -sh "${WORK_INDEX}" | cut -f1))"
else
  log "MV2 index not found. Downloading from ${HF_KB_REPO}..."
  if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
  fi
  huggingface-cli download "${HF_KB_REPO}" \
    "${INDEX_FILENAME}" \
    --local-dir "${WORK_DIR}" \
    --local-dir-use-symlinks False
  if [ ! -f "${WORK_INDEX}" ]; then
    err "Download failed: ${WORK_INDEX} still missing"
    exit 1
  fi
  log "MV2 download complete ($(du -sh "${WORK_INDEX}" | cut -f1))"
fi

# ── Step 2: EmbedGemma 300M embedding model ───────────────────────────────────

if [ -d "${EMBED_MODEL_PATH}" ] && [ -f "${EMBED_MODEL_PATH}/config.json" ]; then
  log "Found EmbedGemma at ${EMBED_MODEL_PATH}"
else
  log "EmbedGemma not found at ${EMBED_MODEL_PATH}. Downloading ${HF_EMBED_MODEL}..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_EMBED_MODEL}',
    local_dir='${EMBED_MODEL_PATH}',
)
print('EmbedGemma download complete.')
"
  if [ ! -f "${EMBED_MODEL_PATH}/config.json" ]; then
    err "EmbedGemma download failed"
    exit 1
  fi
  log "EmbedGemma download complete"
fi

# ── Step 3: Start KB daemon ───────────────────────────────────────────────────

log "Starting KB daemon v2 on port ${KB_PORT}"
export KB_INDEX_PATH="${WORK_INDEX}"
export KB_PORT="${KB_PORT}"
export EMBED_MODEL_PATH="${EMBED_MODEL_PATH}"
export LOG_LEVEL="${LOG_LEVEL}"

exec python3 -m kb.daemon_v2 "${KB_PORT}"
