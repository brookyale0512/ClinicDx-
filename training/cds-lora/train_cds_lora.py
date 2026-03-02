#!/usr/bin/env python3
"""
CDS KB Tool-Use LoRA — Production Training Script
===================================================
DDP-aware LoRA fine-tuning on MedGemma with loss masking.
Supports 2-node training (torchrun) with solo fallback.

Usage:
    # DDP (launched by run_training.sh via torchrun):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=10.128.0.4 --master_port=29500 \
        train_cds_lora.py --config config.yaml

    # Solo:
    python3 train_cds_lora.py --config config.yaml

    # Dry run:
    python3 train_cds_lora.py --config config.yaml --dry-run
"""

import os
import sys
import json
import yaml
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import CdsDataLoader, DataConfig

import sys as _sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=_sys.stdout,
)
for _h in logging.root.handlers:
    _h.flush = _sys.stdout.flush
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemma 3 Data Collator (handles token_type_ids required for causal mask)
# ---------------------------------------------------------------------------
@dataclass
class Gemma3DataCollator:
    tokenizer: Any
    max_length: int = 5120
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "token_type_ids": []}
        for f in features:
            ids = f["input_ids"]
            batch["input_ids"].append(ids)
            batch["attention_mask"].append(f.get("attention_mask") or [1] * len(ids))
            batch["labels"].append(f.get("labels", ids))
            batch["token_type_ids"].append(f.get("token_type_ids") or [0] * len(ids))

        max_len = max(len(ids) for ids in batch["input_ids"])
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_len = min(max_len, self.max_length)

        pad_id = self.tokenizer.pad_token_id or 0
        out = {"input_ids": [], "attention_mask": [], "labels": [], "token_type_ids": []}
        for i in range(len(features)):
            n = len(batch["input_ids"][i])
            pad = max_len - n
            if pad > 0:
                out["input_ids"].append(batch["input_ids"][i] + [pad_id] * pad)
                out["attention_mask"].append(batch["attention_mask"][i] + [0] * pad)
                out["labels"].append(batch["labels"][i] + [-100] * pad)
                out["token_type_ids"].append(batch["token_type_ids"][i] + [0] * pad)
            else:
                out["input_ids"].append(batch["input_ids"][i][:max_len])
                out["attention_mask"].append(batch["attention_mask"][i][:max_len])
                out["labels"].append(batch["labels"][i][:max_len])
                out["token_type_ids"].append(batch["token_type_ids"][i][:max_len])

        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# ---------------------------------------------------------------------------
# JSON logging callback (structured log for monitoring)
# ---------------------------------------------------------------------------
class JsonLogCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a")
        self._best_eval_loss = float("inf")
        self._eval_increases = 0
        self._start = time.time()

    def _write(self, event: dict):
        event["ts"] = datetime.now().isoformat()
        event["wall_s"] = round(time.time() - self._start, 1)
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            entry = {
                "type": "train" if "loss" in logs else "eval",
                "step": state.global_step,
                "epoch": round(state.epoch or 0, 4),
            }
            for k in ["loss", "eval_loss", "learning_rate", "grad_norm",
                       "mean_token_accuracy", "eval_mean_token_accuracy",
                       "entropy", "eval_entropy"]:
                if k in logs:
                    entry[k] = logs[k]

            mem = torch.cuda.memory_allocated() / 1e9
            entry["gpu_mem_gb"] = round(mem, 2)

            if "eval_loss" in logs:
                el = logs["eval_loss"]
                if el < self._best_eval_loss:
                    self._best_eval_loss = el
                    self._eval_increases = 0
                    entry["new_best"] = True
                else:
                    self._eval_increases += 1
                    if self._eval_increases >= 3:
                        logger.warning("eval_loss increased %d consecutive evals — possible overfitting", self._eval_increases)
                entry["best_eval_loss"] = round(self._best_eval_loss, 6)

            self._write(entry)

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


def freeze_vision_encoder(model, cfg: dict):
    if not cfg.get("vision", {}).get("freeze", True):
        return
    frozen = 0
    for attr in ["vision_tower", "vision_model", "multi_modal_projector", "mm_projector"]:
        for prefix in [model, getattr(model, "model", None), getattr(model, "base_model", None)]:
            if prefix is None:
                continue
            mod = getattr(prefix, attr, None)
            if mod is not None:
                count = sum(p.numel() for p in mod.parameters())
                for p in mod.parameters():
                    p.requires_grad = False
                frozen += count
                logger.info("Frozen %s: %s params", attr, f"{count:,}")
    if frozen:
        logger.info("Total frozen vision/projector: %s", f"{frozen:,}")


def load_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    logger.info("Loading model: %s (dtype=%s)", name, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model class: %s", type(model).__name__)
    return model, tokenizer


def apply_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]
    targets = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        target_modules=targets,
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA applied: r=%d, alpha=%d, trainable=%s (%.2f%%)",
                lora_cfg["r"], lora_cfg["lora_alpha"], f"{trainable:,}", trainable / total * 100)
    return model


def create_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    return SFTConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t.get("weight_decay", 0.01),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        optim=t.get("optim", "adamw_torch"),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 250),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 250),
        save_total_limit=t.get("save_total_limit", 5),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=t.get("logging_first_step", True),
        seed=t.get("seed", 42),
        data_seed=t.get("data_seed", 42),
        remove_unused_columns=False,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=cfg["lora"].get("use_gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        logging_dir=str(Path(t["output_dir"]).parent / "logs" / "tensorboard"),
        ddp_timeout=cfg.get("distributed", {}).get("nccl_timeout", 600),
    )


def run_dry_run(model, tokenizer, train_ds, val_ds, cfg):
    import sys
    logger.info("=" * 60)
    logger.info("DRY RUN")
    logger.info("=" * 60)
    sys.stdout.flush()

    logger.info("[dry-run] Moving model to CUDA ...")
    sys.stdout.flush()
    model = model.to("cuda")

    mem_model = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info("[dry-run] Model on GPU: %.1f / %.1f GB", mem_model, mem_total)
    sys.stdout.flush()

    logger.info("[dry-run] Building test batch (2 samples) ...")
    sys.stdout.flush()
    collator = Gemma3DataCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_seq_length"])
    batch = collator([train_ds[i] for i in range(min(2, len(train_ds)))])
    batch = {k: v.to("cuda") for k, v in batch.items()}

    logger.info("[dry-run] Batch shapes: %s", {k: tuple(v.shape) for k, v in batch.items()})
    sys.stdout.flush()

    logger.info("[dry-run] Running forward pass ...")
    sys.stdout.flush()
    with torch.no_grad():
        out = model(**batch)
    logger.info("[dry-run] Forward pass OK — loss=%.4f", out.loss.item())

    mem_after = torch.cuda.memory_allocated() / 1e9
    logger.info("[dry-run] GPU after forward: %.1f / %.1f GB (%.1f GB free)",
                mem_after, mem_total, mem_total - mem_after)
    sys.stdout.flush()

    labels = batch["labels"][0].cpu().tolist()
    masked = sum(1 for l in labels if l == -100)
    logger.info("[dry-run] Loss masking: %d/%d tokens masked (%.1f%%)",
                masked, len(labels), masked / len(labels) * 100)

    t_cfg = cfg["training"]
    eff_batch = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"]
    world = cfg.get("distributed", {}).get("world_size", 1)
    eff_batch *= world
    steps = (len(train_ds) * t_cfg["num_train_epochs"]) // eff_batch
    logger.info("[dry-run] Samples: %d train, %d val", len(train_ds), len(val_ds) if val_ds else 0)
    logger.info("[dry-run] Effective batch: %d, Total steps: %d, Epochs: %d",
                eff_batch, steps, t_cfg["num_train_epochs"])
    logger.info("[dry-run] DRY RUN COMPLETE — ready for training")
    sys.stdout.flush()


def run_training(model, tokenizer, train_ds, val_ds, cfg):
    training_args = create_training_args(cfg)

    log_path = Path(cfg["training"]["output_dir"]).parent / "logs" / f"train_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    json_logger = JsonLogCallback(str(log_path))
    logger.info("JSON log: %s", log_path)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg["training"].get("early_stopping_patience", 5))]
    callbacks.append(json_logger)

    collator = Gemma3DataCollator(tokenizer=tokenizer, max_length=cfg["model"]["max_seq_length"])

    resume_checkpoint = None
    ckpt_dir = Path(cfg["training"]["output_dir"])
    if ckpt_dir.exists():
        ckpts = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                       key=lambda d: int(d.name.split("-")[1]))
        if ckpts:
            resume_checkpoint = str(ckpts[-1])
            logger.info("Resuming from %s", resume_checkpoint)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    t_cfg = cfg["training"]
    world = int(os.environ.get("WORLD_SIZE", cfg.get("distributed", {}).get("world_size", 1)))
    eff = t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"] * world
    logger.info("Starting training: %d samples, batch=%d, accum=%d, world=%d, eff_batch=%d, epochs=%d",
                len(train_ds), t_cfg["per_device_train_batch_size"],
                t_cfg["gradient_accumulation_steps"], world, eff, t_cfg["num_train_epochs"])

    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    if val_ds:
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)

    json_logger.close()
    return trainer, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model, tokenizer = load_model_and_tokenizer(cfg)
    freeze_vision_encoder(model, cfg)
    model = apply_lora(model, cfg)

    if cfg["lora"].get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    data_cfg = DataConfig(
        train_file=cfg["data"]["train_file"],
        val_file=cfg["data"].get("val_file"),
        max_seq_length=cfg["model"]["max_seq_length"],
    )
    loader = CdsDataLoader(tokenizer, data_cfg)

    logger.info("[data] Loading train JSONL ...")
    train_samples = loader.load_jsonl(data_cfg.train_file)
    logger.info("[data] Tokenizing train (%d samples) ...", len(train_samples))
    train_ds = loader.prepare_dataset(train_samples, for_training=True)
    logger.info("[data] Train ready: %d samples, %d columns", len(train_ds), len(train_ds.column_names))

    val_ds = None
    if data_cfg.val_file:
        logger.info("[data] Loading val JSONL ...")
        val_samples = loader.load_jsonl(data_cfg.val_file)
        logger.info("[data] Tokenizing val (%d samples) ...", len(val_samples))
        val_ds = loader.prepare_dataset(val_samples, for_training=True)
        logger.info("[data] Val ready: %d samples", len(val_ds))

    logger.info("[data] All data loaded. Proceeding to %s ...", "dry run" if args.dry_run else "training")

    if args.dry_run:
        run_dry_run(model, tokenizer, train_ds, val_ds, cfg)
    else:
        run_training(model, tokenizer, train_ds, val_ds, cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
