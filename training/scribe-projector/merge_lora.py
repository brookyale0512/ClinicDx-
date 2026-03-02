#!/usr/bin/env python3
"""Merge Phase 1 LoRA weights into full multimodal MedGemma for Phase 2."""

import torch, time, os, sys

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("  Merging LoRA checkpoint-1200 into MedGemma")
log("=" * 60)

log("[1/4] Loading text-only base + LoRA...")
from peft import PeftModel
from transformers import Gemma3ForCausalLM, AutoTokenizer

base = Gemma3ForCausalLM.from_pretrained(
    "/var/www/ClinicDx/model/medgemma_text_only",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(
    base,
    "/var/www/ClinicDx/training/checkpoints/text_sft_200k/checkpoint-1200",
)
log("  LoRA loaded")

log("[2/4] Merging LoRA weights...")
merged = model.merge_and_unload()
log(f"  Merged: {sum(p.numel() for p in merged.parameters()):,} params")

log("[3/4] Injecting into full multimodal model...")
from transformers import Gemma3ForConditionalGeneration

full = Gemma3ForConditionalGeneration.from_pretrained(
    "/var/www/ClinicDx/model/medgemma",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
full.model.language_model.load_state_dict(merged.model.state_dict())
full.lm_head.load_state_dict(merged.lm_head.state_dict())
log("  Language model weights replaced")

log("[4/4] Saving medgemma_sft...")
out = "/var/www/ClinicDx/model/medgemma_sft"
os.makedirs(out, exist_ok=True)
full.save_pretrained(out, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained("/var/www/ClinicDx/model/medgemma")
tokenizer.save_pretrained(out)

size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith('.safetensors')) / 1e9
log(f"  Saved: {out} ({size:.1f} GB)")

del base, model, merged, full
log("Done.")
