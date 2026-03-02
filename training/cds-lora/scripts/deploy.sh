#!/usr/bin/env bash
set -euo pipefail

# CDS KB LoRA — Deploy to A100 + local prep
# Runs data prep, then rsyncs everything needed for training + eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A100="RAZERBLADE@10.128.0.5"
REMOTE_DIR="/var/www/ClinicDx/training/cds_kb_lora"
KB_SRC="/var/www/kbToolUseLora/kb"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Step 1: Prepare training data ==="
python3 "$SCRIPT_DIR/prep_cds_training.py"

log "=== Step 2: Create remote directories ==="
ssh "$A100" "mkdir -p $REMOTE_DIR/{data,logs,checkpoints,kb,results}"

log "=== Step 3: Rsync scripts + config + data ==="
rsync -avP --exclude='.venv' --exclude='__pycache__' \
    "$SCRIPT_DIR/"*.py "$SCRIPT_DIR/"*.yaml "$SCRIPT_DIR/"*.sh \
    "$A100:$REMOTE_DIR/"

rsync -avP "$SCRIPT_DIR/data/" "$A100:$REMOTE_DIR/data/"

log "=== Step 4: Rsync KB service ==="
rsync -avP "$SCRIPT_DIR/kb/" "$A100:$REMOTE_DIR/kb/"

log "=== Step 5: Rsync KB indices (if not already on A100) ==="
ssh "$A100" "test -f $REMOTE_DIR/kb/wikimed.mv2" 2>/dev/null || {
    log "Sending KB indices (~5.6 GB) ..."
    rsync -avP "$KB_SRC/wikimed.mv2" "$A100:$REMOTE_DIR/kb/"
    rsync -avP "$KB_SRC/who_knowledge.mv2" "$A100:$REMOTE_DIR/kb/"
}

log "=== Step 6: Install deps on A100 ==="
ssh "$A100" "source /opt/conda/bin/activate && pip install --quiet tensorboard safetensors memvid_sdk 2>&1 | tail -3" || true

log "=== Step 7: Verify A100 readiness ==="
ssh "$A100" "source /opt/conda/bin/activate && python3 -c \"
import torch, transformers, trl, peft
print(f'torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB)')
\""

log ""
log "================================================"
log "  Deployment complete. Ready to train."
log "================================================"
log ""
log "To start DDP training (from this server):"
log "  bash $SCRIPT_DIR/run_training.sh"
log ""
log "To start solo training (on A100 only):"
log "  ssh $A100 'cd $REMOTE_DIR && source /opt/conda/bin/activate && nohup python3 train_cds_lora.py --config config.yaml > logs/train.log 2>&1 &'"
