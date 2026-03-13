#!/usr/bin/env python3
"""
SFT training — supports tagged multi-task ([CDS] + [SCRIBE]) and single-task configs.
Reuses the existing SFT data loader and training logic.
"""

import sys
import yaml
import logging
import torch
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import CdsDataLoader, DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "config.yaml")
    logger.info("Config: %s", cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    logger.info("=" * 60)
    logger.info("Short SFT — Concise Output Training")
    logger.info("Model: %s", model_cfg["name"])
    logger.info("Data: %s", cfg["data"]["train_file"])
    logger.info("=" * 60)

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze vision
    for attr in ["vision_tower", "vision_model", "multi_modal_projector"]:
        for prefix in [model, getattr(model, "model", None)]:
            if prefix is None:
                continue
            mod = getattr(prefix, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

    # Apply LoRA (or resume from existing LoRA checkpoint)
    resume_ckpt = train_cfg.get("resume_from_checkpoint")
    if resume_ckpt:
        from peft import PeftModel
        logger.info("Resuming LoRA from: %s", resume_ckpt)
        model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"), task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: r=%d, trainable=%s (%.2f%%)", lora_cfg["r"], f"{trainable:,}", trainable / total * 100)

    # Load data
    data_cfg = DataConfig(
        train_file=cfg["data"]["train_file"],
        max_seq_length=model_cfg["max_seq_length"],
    )
    loader = CdsDataLoader(tokenizer, data_cfg)
    train_samples = loader.load_jsonl(data_cfg.train_file)
    train_ds = loader.prepare_dataset(train_samples, for_training=True,
                                       jsonl_path=data_cfg.train_file, split="train")
    logger.info("Train: %d samples", len(train_ds))
    stats = loader.get_dataset_stats(train_ds)
    logger.info("Stats: %s", stats)

    val_ds = None
    if cfg["data"].get("val_file"):
        val_samples = loader.load_jsonl(cfg["data"]["val_file"])
        val_ds = loader.prepare_dataset(val_samples, for_training=True,
                                         jsonl_path=cfg["data"]["val_file"], split="val")
        logger.info("Val: %d samples", len(val_ds))

    # Collator (same as main SFT)
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class Gemma3DataCollator:
        tokenizer: Any
        max_length: int = 4096
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

    # Training args
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        optim=train_cfg.get("optim", "adamw_torch"),
        bf16=True,
        logging_steps=train_cfg.get("logging_steps", 10),
        logging_first_step=True,
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 5),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if val_ds else "no",
        eval_steps=train_cfg.get("eval_steps", 200) if val_ds else None,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        seed=train_cfg.get("seed", 42),
        remove_unused_columns=False,
        packing=False,
        gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    collator = Gemma3DataCollator(tokenizer=tokenizer, max_length=model_cfg["max_seq_length"])

    callbacks = []
    early_stop_patience = train_cfg.get("early_stopping_patience")
    if early_stop_patience and val_ds:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stop_patience))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    logger.info("Starting training...")
    result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", result.metrics)

    logger.info("Training complete. Metrics: %s", result.metrics)
    logger.info("Checkpoint: %s", train_cfg["output_dir"])


if __name__ == "__main__":
    main()
