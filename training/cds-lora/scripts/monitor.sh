#!/usr/bin/env bash
# CDS KB LoRA — Live Training Monitor
# Displays training progress from JSON log, GPU usage on both nodes.

A100="RAZERBLADE@10.128.0.5"
LOG_DIR="/var/www/ClinicDx/training/cds_kb_lora/logs"

latest_log() {
    local dir="${1:-$LOG_DIR}"
    ls -t "$dir"/train_*.jsonl 2>/dev/null | head -1
}

show_status() {
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo "  CDS KB LoRA — Training Monitor  $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"

    local log
    log=$(latest_log)
    if [ -z "$log" ]; then
        # Try remote
        log=$(ssh "$A100" "ls -t /var/www/ClinicDx/training/cds_kb_lora/logs/train_*.jsonl 2>/dev/null | head -1" 2>/dev/null)
        if [ -n "$log" ]; then
            echo "  [Reading from A100: $log]"
            ssh "$A100" "tail -5 '$log'" 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        if d.get('type') == 'train':
            print(f\"  Step {d.get('step',0):>6}  loss={d.get('loss',0):.4f}  lr={d.get('learning_rate',0):.2e}  gpu={d.get('gpu_mem_gb',0):.1f}GB  wall={d.get('wall_s',0)/3600:.1f}h\")
        elif d.get('type') == 'eval':
            best = d.get('best_eval_loss', '?')
            new = ' [NEW BEST]' if d.get('new_best') else ''
            print(f\"  Step {d.get('step',0):>6}  eval_loss={d.get('eval_loss',0):.4f}  best={best}{new}\")
    except: pass
" 2>/dev/null
        else
            echo "  No training log found."
        fi
    else
        echo "  [Local log: $(basename "$log")]"
        tail -5 "$log" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        if d.get('type') == 'train':
            print(f\"  Step {d.get('step',0):>6}  loss={d.get('loss',0):.4f}  lr={d.get('learning_rate',0):.2e}  gpu={d.get('gpu_mem_gb',0):.1f}GB  wall={d.get('wall_s',0)/3600:.1f}h\")
        elif d.get('type') == 'eval':
            best = d.get('best_eval_loss', '?')
            new = ' [NEW BEST]' if d.get('new_best') else ''
            print(f\"  Step {d.get('step',0):>6}  eval_loss={d.get('eval_loss',0):.4f}  best={best}{new}\")
    except: pass
" 2>/dev/null
    fi

    echo ""
    echo "  GPU (local):"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | \
        awk -F, '{printf "    %s util | %s / %s | %s\n", $1, $2, $3, $4}'

    echo "  GPU (A100):"
    ssh "$A100" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null | \
        awk -F, '{printf "    %s util | %s / %s | %s\n", $1, $2, $3, $4}'

    echo ""
    echo "  Network: $(ping -c 1 -W 1 10.128.0.5 2>/dev/null | grep 'time=' | sed 's/.*time=//' || echo 'unreachable')"
    echo "════════════════════════════════════════════════════════════════"
}

while true; do
    show_status
    sleep 30
done
