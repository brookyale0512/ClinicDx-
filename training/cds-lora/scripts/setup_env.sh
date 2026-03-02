#!/usr/bin/env bash
set -euo pipefail

# CDS KB LoRA — Environment Setup
# Creates matched Python environments on both nodes for DDP training.
# A100 driver 550.90 limits us to CUDA 12.4; use torch 2.6.0+cu124 everywhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A100_HOST="RAZERBLADE@10.128.0.5"
A100_DIR="/var/www/ClinicDx/training/cds_kb_lora"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

PINNED_DEPS=(
    "torch==2.6.0+cu124"
    "torchvision==0.21.0+cu124"
    "torchaudio==2.6.0+cu124"
)
PIP_DEPS=(
    "transformers>=4.45,<6"
    "trl>=0.28,<1"
    "peft>=0.18,<1"
    "datasets>=4.0"
    "accelerate>=1.0"
    "tensorboard"
    "safetensors"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_local() {
    log "=== Setting up LOCAL environment ==="
    VENV="$SCRIPT_DIR/.venv"

    if [ -d "$VENV" ] && "$VENV/bin/python3" -c "import torch; assert torch.__version__.startswith('2.6')" 2>/dev/null; then
        log "Local venv already configured (torch $(\"$VENV/bin/python3\" -c 'import torch;print(torch.__version__)'))"
        return 0
    fi

    log "Creating venv at $VENV ..."
    python3 -m venv "$VENV" --system-site-packages
    source "$VENV/bin/activate"

    log "Installing pinned torch 2.6.0+cu124 ..."
    pip install --quiet "${PINNED_DEPS[@]}" --index-url "$TORCH_INDEX"

    log "Installing training dependencies ..."
    pip install --quiet "${PIP_DEPS[@]}"

    local tv
    tv=$(python3 -c "import torch; print(torch.__version__)")
    log "Local venv ready: torch=$tv"
    deactivate
}

setup_a100() {
    log "=== Setting up A100 environment ==="

    local remote_torch
    remote_torch=$(ssh "$A100_HOST" "source /opt/conda/bin/activate && python3 -c 'import torch; print(torch.__version__)'" 2>/dev/null || echo "MISSING")
    log "A100 current torch: $remote_torch"

    if [[ "$remote_torch" == 2.6.* ]]; then
        log "A100 torch version OK"
    else
        log "ERROR: A100 torch is $remote_torch, expected 2.6.x. Install manually:"
        log "  pip install torch==2.6.0+cu124 --index-url $TORCH_INDEX"
        return 1
    fi

    log "Installing extra deps on A100 ..."
    ssh "$A100_HOST" "source /opt/conda/bin/activate && pip install --quiet tensorboard safetensors 2>&1 | tail -1"

    log "A100 environment ready"
}

test_nccl() {
    log "=== Testing NCCL connectivity ==="
    local LOCAL_IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    cat > /tmp/_nccl_test.py << 'PYEOF'
import os, sys, torch, torch.distributed as dist
rank = int(os.environ["RANK"])
dist.init_process_group("nccl", init_method=os.environ["MASTER_INIT"],
                        rank=rank, world_size=2)
t = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(t)
expected = 3.0  # 1+2
assert abs(t.item() - expected) < 0.01, f"rank {rank}: got {t.item()}, expected {expected}"
print(f"rank {rank}: NCCL all_reduce OK (result={t.item()})")
dist.destroy_process_group()
PYEOF

    scp -q /tmp/_nccl_test.py "$A100_HOST:/tmp/_nccl_test.py"

    local MASTER_INIT="tcp://${LOCAL_IP}:29500"
    log "Master init: $MASTER_INIT"

    MASTER_INIT="$MASTER_INIT" RANK=0 \
        "$SCRIPT_DIR/.venv/bin/python3" /tmp/_nccl_test.py &
    local pid0=$!

    ssh "$A100_HOST" "source /opt/conda/bin/activate && \
        MASTER_INIT='$MASTER_INIT' RANK=1 python3 /tmp/_nccl_test.py" &
    local pid1=$!

    wait $pid0 && wait $pid1 && log "NCCL test PASSED" || {
        log "ERROR: NCCL test FAILED. Check firewall on port 29500."
        return 1
    }
}

main() {
    log "CDS KB LoRA — Environment Setup"
    log "================================"

    local cmd="${1:-all}"
    case "$cmd" in
        local)  setup_local ;;
        a100)   setup_a100 ;;
        nccl)   test_nccl ;;
        all)
            setup_local
            setup_a100
            log ""
            log "Both environments ready. To test NCCL: $0 nccl"
            ;;
        *)
            echo "Usage: $0 {local|a100|nccl|all}"
            exit 1
            ;;
    esac
}

main "$@"
