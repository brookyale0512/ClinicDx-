#!/usr/bin/env bash
set -euo pipefail

# CDS KB LoRA — DDP Training Launcher
# Launches torchrun on both nodes simultaneously.
# Falls back to solo A100 if DDP init fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A100="RAZERBLADE@10.128.0.5"
REMOTE_DIR="/var/www/ClinicDx/training/cds_kb_lora"
LOCAL_IP=$(hostname -I | awk '{print $1}')
MASTER_PORT=29500
WORLD_SIZE=2

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_ddp() {
    log "Starting DDP training: $WORLD_SIZE nodes"
    log "Master: $LOCAL_IP:$MASTER_PORT"

    # Activate local venv
    source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null || true

    # Launch remote worker first (background)
    log "Launching rank 1 on A100 ..."
    ssh "$A100" "source /opt/conda/bin/activate && cd $REMOTE_DIR && \
        nohup torchrun \
            --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=1 \
            --master_addr=$LOCAL_IP --master_port=$MASTER_PORT \
            train_cds_lora.py --config config.yaml \
            > logs/train_rank1.log 2>&1 &
        echo \$!" &
    REMOTE_PID=$!
    sleep 3

    # Launch local master (foreground with log tee)
    log "Launching rank 0 (master) locally ..."
    torchrun \
        --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=0 \
        --master_addr=$LOCAL_IP --master_port=$MASTER_PORT \
        "$SCRIPT_DIR/train_cds_lora.py" --config "$SCRIPT_DIR/config.yaml" \
        2>&1 | tee "$SCRIPT_DIR/logs/train_rank0.log"

    local exit_code=$?
    wait $REMOTE_PID 2>/dev/null || true

    if [ $exit_code -eq 0 ]; then
        log "DDP training complete."
    else
        log "DDP training failed (exit=$exit_code)."
        return $exit_code
    fi
}

run_solo() {
    log "Starting SOLO training on A100 ..."
    ssh "$A100" "source /opt/conda/bin/activate && cd $REMOTE_DIR && \
        nohup python3 train_cds_lora.py --config config.yaml \
            > logs/train_solo.log 2>&1 &
        echo Started PID: \$!"
    log "Training launched. Monitor with:"
    log "  ssh $A100 'tail -f $REMOTE_DIR/logs/train_solo.log'"
}

run_post_training() {
    log "=== Post-training: merge + eval ==="
    ssh "$A100" "source /opt/conda/bin/activate && cd $REMOTE_DIR && \
        python3 merge_lora.py > logs/merge.log 2>&1"
    log "Merge complete. Starting eval ..."

    # Start KB daemon on A100
    ssh "$A100" "source /opt/conda/bin/activate && cd $REMOTE_DIR && \
        python3 kb/daemon.py --who-index kb/who_knowledge.mv2 --wiki-index kb/wikimed.mv2 &
        sleep 5 && \
        python3 eval_medqa.py --model /var/www/ClinicDx/model/medgemma_cds_kb --max 1273 \
            > logs/eval.log 2>&1"
    log "Eval complete."
}

case "${1:-ddp}" in
    ddp)   run_ddp && run_post_training ;;
    solo)  run_solo ;;
    eval)  run_post_training ;;
    *)     echo "Usage: $0 {ddp|solo|eval}"; exit 1 ;;
esac
