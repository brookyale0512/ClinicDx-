#!/usr/bin/env python3
"""
CDS KB LoRA v2 — Merge LoRA adapter into base model and save as standalone model.

Loads the best LoRA checkpoint, merges into medgemma-4b-it (text-only causal),
saves the full merged model to medgemma_cds_v2.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_MODEL = "/var/www/ClinicDx/model/medgemma-4b-it"
CKPT_DIR = Path("/var/www/ClinicDx/training/lora_cds_kb_v2/checkpoints")
OUTPUT = "/var/www/ClinicDx/model/medgemma_cds_v2"


def find_best_checkpoint() -> Path:
    ckpts = sorted(
        [d for d in CKPT_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        log.error("No checkpoints found in %s", CKPT_DIR)
        sys.exit(1)

    for ckpt in reversed(ckpts):
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            best = state.get("best_model_checkpoint", "")
            if best and Path(best).exists():
                log.info("Best checkpoint (from trainer_state): %s", best)
                return Path(best)

    log.info("Using latest checkpoint: %s", ckpts[-1])
    return ckpts[-1]


def main():
    checkpoint = find_best_checkpoint() if len(sys.argv) < 2 else Path(sys.argv[1])
    log.info("=" * 60)
    log.info("Merging LoRA: %s", checkpoint.name)
    log.info("Base model: %s", BASE_MODEL)
    log.info("Output: %s", OUTPUT)
    log.info("=" * 60)

    log.info("[1/3] Loading base model + LoRA adapter ...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(checkpoint))

    log.info("[2/3] Merging LoRA weights ...")
    model = model.merge_and_unload()
    total = sum(p.numel() for p in model.parameters())
    log.info("  Merged model: %s params", f"{total:,}")

    log.info("[3/3] Saving merged model ...")
    model.save_pretrained(OUTPUT, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT)

    size_gb = sum(f.stat().st_size for f in Path(OUTPUT).rglob("*") if f.is_file()) / 1e9
    log.info("  Saved: %s (%.1f GB)", OUTPUT, size_gb)

    log.info("[Validate] Quick generation test ...")
    del model
    torch.cuda.empty_cache()
    test_model = AutoModelForCausalLM.from_pretrained(OUTPUT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    test_tok = AutoTokenizer.from_pretrained(OUTPUT, trust_remote_code=True)
    prompt = "<bos><start_of_turn>user\nA 45-year-old presents with fever and cough.<end_of_turn>\n<start_of_turn>model\n"
    ids = test_tok(prompt, return_tensors="pt").to(test_model.device)
    with torch.no_grad():
        out = test_model.generate(**ids, max_new_tokens=100, do_sample=False)
    text = test_tok.decode(out[0], skip_special_tokens=False)
    has_kb_query = "<KB_QUERY>" in text
    log.info("  Generated %d tokens, has KB_QUERY: %s", out.shape[1], has_kb_query)
    if has_kb_query:
        log.info("  Validation PASSED")
    else:
        log.warning("  Validation WARNING: no <KB_QUERY> in output — model may need more training")

    log.info("Done.")


if __name__ == "__main__":
    main()
