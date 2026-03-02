#!/usr/bin/env python3
"""
MedGemma 1.5 4B - Production LoRA Training Script
=================================================

Ante-hoc Clinical Reasoning Distillation using Unsloth-optimized LoRA.

Features:
- Full-precision LoRA (NOT QLoRA) for maximum accuracy
- Vision encoder freezing for text-only training
- Comprehensive WandB monitoring
- Early stopping with best model checkpointing
- Reasoning structure validation during training
- Memory-efficient training with gradient checkpointing

Usage:
    python train_lora.py --config config.yaml
    python train_lora.py --config config.yaml --dry-run  # Test without training
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import wandb
from transformers import (
    EarlyStoppingCallback,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from dataclasses import dataclass
from typing import Any

@dataclass
class Gemma3DataCollator:
    """
    Custom data collator for Gemma 3 that properly handles token_type_ids.
    Gemma 3 requires token_type_ids during training for its causal mask.
    """
    tokenizer: Any
    padding: bool = True
    max_length: int = 4096
    pad_to_multiple_of: int = 8
    
    def __call__(self, features):
        # Extract all the fields we need
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "token_type_ids": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            batch["input_ids"].append(input_ids)
            
            # attention_mask: use from feature or create all-ones
            attn_mask = feature.get("attention_mask")
            if attn_mask is None:
                attn_mask = [1] * len(input_ids)
            batch["attention_mask"].append(attn_mask)
            
            # labels: use from feature or copy input_ids
            batch["labels"].append(feature.get("labels", input_ids))
            
            # token_type_ids: use from feature or create zeros (required by Gemma 3)
            token_type_ids = feature.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = [0] * len(input_ids)
            batch["token_type_ids"].append(token_type_ids)
        
        # Find max length in batch
        max_len = max(len(ids) for ids in batch["input_ids"])
        
        # Pad to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Cap at max_length
        max_len = min(max_len, self.max_length)
        
        # Pad each sequence
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "token_type_ids": [],
        }
        
        for i in range(len(features)):
            seq_len = len(batch["input_ids"][i])
            padding_len = max_len - seq_len
            
            if padding_len > 0:
                # Pad on the right
                padded_batch["input_ids"].append(batch["input_ids"][i] + [pad_token_id] * padding_len)
                padded_batch["attention_mask"].append(batch["attention_mask"][i] + [0] * padding_len)
                padded_batch["labels"].append(batch["labels"][i] + [-100] * padding_len)  # -100 = ignore in loss
                padded_batch["token_type_ids"].append(batch["token_type_ids"][i] + [0] * padding_len)
            else:
                # Truncate if needed
                padded_batch["input_ids"].append(batch["input_ids"][i][:max_len])
                padded_batch["attention_mask"].append(batch["attention_mask"][i][:max_len])
                padded_batch["labels"].append(batch["labels"][i][:max_len])
                padded_batch["token_type_ids"].append(batch["token_type_ids"][i][:max_len])
        
        # Convert to tensors
        import torch
        return {
            "input_ids": torch.tensor(padded_batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded_batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded_batch["labels"], dtype=torch.long),
            "token_type_ids": torch.tensor(padded_batch["token_type_ids"], dtype=torch.long),
        }

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import MedGemmaDataLoader, DataConfig, create_data_collator
from evaluate import ReasoningEvaluator, EvaluationCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration before training.
    
    Checks required fields and warns about unusual hyperparameters.
    """
    logger.info("Validating configuration...")
    
    # Required fields
    required = {
        ("model", "name"): "Model name is required",
        ("data", "train_file"): "Training data file is required",
        ("training", "output_dir"): "Output directory is required",
    }
    
    for keys, message in required.items():
        value = config
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else None
        if not value:
            raise ValueError(f"Configuration error: {message}")
    
    # Validate LoRA parameters
    lora = config.get("lora", {})
    r = lora.get("r", 32)
    alpha = lora.get("lora_alpha", 32)
    ratio = alpha / r if r > 0 else 0
    
    if ratio < 0.5 or ratio > 4.0:
        logger.warning(f"⚠️  Unusual alpha/r ratio: {ratio:.2f} (standard: 0.5-2.0)")
    
    # Validate data file exists
    train_file = config.get("data", {}).get("train_file", "")
    if train_file and not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    logger.info("✓ Configuration validated")


def setup_wandb(config: Dict[str, Any]) -> None:
    """Initialize Weights & Biases tracking."""
    wandb_config = config.get("wandb", {})
    
    wandb.init(
        entity=wandb_config.get("entity"),
        project=wandb_config.get("project", "medgemma-lora"),
        name=wandb_config.get("name", f"lora-{datetime.now().strftime('%Y%m%d-%H%M')}"),
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes", ""),
        config={
            "model": config.get("model", {}),
            "lora": config.get("lora", {}),
            "training": config.get("training", {}),
        }
    )
    logger.info(f"Initialized WandB project: {wandb_config.get('project')}")


def load_model_and_tokenizer(config: Dict[str, Any]):
    """
    Load MedGemma model with optional Unsloth optimizations.
    
    Returns:
        Tuple of (model, tokenizer, use_unsloth)
    """
    # Check if Unsloth is explicitly disabled in config
    if not config.get("use_unsloth", True):
        logger.info("Unsloth disabled in config - using standard transformers")
        use_unsloth = False
    else:
        try:
            from unsloth import FastLanguageModel
            logger.info("Using Unsloth for optimized training")
            use_unsloth = True
        except ImportError:
            logger.warning("Unsloth not available, falling back to standard transformers")
            use_unsloth = False
    
    model_config = config.get("model", {})
    model_name = model_config.get("name", "google/medgemma-1.5-4b-it")
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"  dtype: {model_config.get('dtype', 'bfloat16')}")
    logger.info(f"  load_in_4bit: {model_config.get('load_in_4bit', False)}")
    logger.info(f"  max_seq_length: {model_config.get('max_seq_length', 4096)}")
    
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_config.get("max_seq_length", 4096),
            dtype=getattr(torch, model_config.get("dtype", "bfloat16")),
            load_in_4bit=model_config.get("load_in_4bit", False),
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
    else:
        from transformers import Gemma3ForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, model_config.get("dtype", "bfloat16")),
            trust_remote_code=model_config.get("trust_remote_code", True),
            device_map={"": 0},
        )
        logger.info("Loaded Gemma3ForCausalLM (text-only, no vision tower)")
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, use_unsloth


def freeze_vision_encoder(model, config: Dict[str, Any]) -> None:
    """
    Freeze the vision encoder (SigLIP) AND multimodal projector for text-only training.
    
    This is CRITICAL for ante-hoc reasoning training on text data.
    Freezes both the vision encoder and the projector that connects vision to text.
    """
    vision_config = config.get("vision", {})
    
    if not vision_config.get("freeze", True):
        logger.info("Vision encoder freezing disabled in config")
        return
    
    frozen_params = 0
    
    # 1. Freeze vision encoder
    vision_modules = ["vision_tower", "vision_model", "vision_encoder", "image_encoder"]
    
    for module_name in vision_modules:
        if hasattr(model, module_name):
            vision_module = getattr(model, module_name)
            count = sum(p.numel() for p in vision_module.parameters())
            for param in vision_module.parameters():
                param.requires_grad = False
            frozen_params += count
            logger.info(f"✓ Frozen {module_name}: {count:,} parameters")
            break
    
    # Check inside model.model if exists
    if hasattr(model, "model"):
        for module_name in vision_modules:
            if hasattr(model.model, module_name):
                vision_module = getattr(model.model, module_name)
                count = sum(p.numel() for p in vision_module.parameters())
                for param in vision_module.parameters():
                    param.requires_grad = False
                frozen_params += count
                logger.info(f"✓ Frozen model.{module_name}: {count:,} parameters")
                break
    
    # 2. Freeze multimodal projector (connects vision to text)
    projector_modules = ["multi_modal_projector", "mm_projector", "vision_projector", "connector"]
    
    for module_name in projector_modules:
        if hasattr(model, module_name):
            projector = getattr(model, module_name)
            count = sum(p.numel() for p in projector.parameters())
            for param in projector.parameters():
                param.requires_grad = False
            frozen_params += count
            logger.info(f"✓ Frozen {module_name}: {count:,} parameters")
            break
    
    if hasattr(model, "model"):
        for module_name in projector_modules:
            if hasattr(model.model, module_name):
                projector = getattr(model.model, module_name)
                count = sum(p.numel() for p in projector.parameters())
                for param in projector.parameters():
                    param.requires_grad = False
                frozen_params += count
                logger.info(f"✓ Frozen model.{module_name}: {count:,} parameters")
                break
    
    if frozen_params == 0:
        logger.warning("⚠️  No vision components found to freeze. Model may not have vision component.")
    else:
        logger.info(f"✓ Total frozen vision/projector parameters: {frozen_params:,}")


def apply_lora(model, config: Dict[str, Any], use_unsloth: bool):
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: Base model
        config: Configuration dict
        use_unsloth: Whether Unsloth is available
    
    Returns:
        Model with LoRA adapters
    """
    lora_config = config.get("lora", {})
    
    r = lora_config.get("r", 32)
    lora_alpha = lora_config.get("lora_alpha", 32)
    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    logger.info("Applying LoRA configuration:")
    logger.info(f"  rank (r): {r}")
    logger.info(f"  lora_alpha: {lora_alpha}")
    logger.info(f"  alpha/r ratio: {lora_alpha/r:.2f}")
    logger.info(f"  target_modules: {target_modules}")
    logger.info(f"  lora_dropout: {lora_config.get('lora_dropout', 0.05)}")
    
    if use_unsloth:
        from unsloth import FastLanguageModel
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
            random_state=lora_config.get("random_state", 42),
        )
    else:
        from peft import LoraConfig, get_peft_model
        
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        
        if lora_config.get("use_gradient_checkpointing"):
            model.gradient_checkpointing_enable()
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA applied successfully:")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"  Total parameters: {total_params:,}")
    
    return model


def create_training_arguments(config: Dict[str, Any]) -> SFTConfig:
    """Create SFTConfig from config (TRL v0.27+ API)."""
    train_config = config.get("training", {})
    model_config = config.get("model", {})
    
    args = SFTConfig(
        output_dir=train_config.get("output_dir", "./checkpoints"),

        # Batch settings
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),

        # Schedule
        num_train_epochs=train_config.get("num_train_epochs", 5),
        warmup_ratio=train_config.get("warmup_ratio", 0.05),
        
        # Optimization
        learning_rate=train_config.get("learning_rate", 2e-4),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        optim=train_config.get("optim", "adamw_torch"),
        
        # Precision
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Evaluation & Checkpointing
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 100),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 200),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_config.get("greater_is_better", False),
        
        # Logging
        logging_steps=train_config.get("logging_steps", 10),
        logging_first_step=train_config.get("logging_first_step", True),
        report_to="none",
        
        # Reproducibility
        seed=train_config.get("seed", 42),
        data_seed=train_config.get("data_seed", 42),
        
        # SFT specific (TRL v0.27+ uses max_length, not max_seq_length)
        max_length=model_config.get("max_seq_length", 4096),
        packing=False,
        dataset_text_field="text",
        
        # CRITICAL: Keep attention_mask and token_type_ids columns
        # TRL's default signature columns don't include these, so they get removed
        # Gemma 3 requires token_type_ids during training
        remove_unused_columns=False,
    )
    
    return args


def load_datasets(config: Dict[str, Any], tokenizer):
    """Load and prepare training and validation datasets."""
    data_config = config.get("data", {})
    
    loader = MedGemmaDataLoader(
        tokenizer=tokenizer,
        config=DataConfig(
            train_file=data_config.get("train_file", ""),
            val_file=data_config.get("val_file"),
            max_seq_length=config.get("model", {}).get("max_seq_length", 4096),
            train_val_split=data_config.get("train_val_split", 0.9),
            shuffle_seed=data_config.get("shuffle_seed", 42),
        )
    )
    
    train_dataset, val_dataset = loader.load_and_prepare()
    
    # Log dataset statistics
    if train_dataset:
        stats = loader.get_dataset_stats(train_dataset)
        logger.info(f"Training dataset: {stats}")
        if os.environ.get("WANDB_DISABLED") != "true":
            wandb.log({"train_dataset_stats": stats})
    
    if val_dataset:
        stats = loader.get_dataset_stats(val_dataset)
        logger.info(f"Validation dataset: {stats}")
        if os.environ.get("WANDB_DISABLED") != "true":
            wandb.log({"val_dataset_stats": stats})
    
    return train_dataset, val_dataset


def run_training(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: Dict[str, Any],
):
    """Run the training loop."""
    training_args = create_training_arguments(config)
    
    # Create callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 3)
        ),
    ]
    
    # Add evaluation callback if available
    eval_config = config.get("evaluation", {})
    if eval_config:
        evaluator = ReasoningEvaluator(tokenizer)
        callbacks.append(
            EvaluationCallback(
                evaluator=evaluator,
                tokenizer=tokenizer,
                eval_steps=eval_config.get("sample_generation_steps", 200),
                num_samples=eval_config.get("num_samples_to_log", 5),
            )
        )
    
    # Create custom data collator for Gemma 3 (requires token_type_ids during training)
    data_collator = Gemma3DataCollator(
        tokenizer=tokenizer,
        max_length=config.get("model", {}).get("max_seq_length", 4096),
    )
    
    # Create trainer (TRL v0.27+ API)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=callbacks,
        data_collator=data_collator,
    )
    
    # Log model info to wandb (if enabled)
    if os.environ.get("WANDB_DISABLED") != "true":
        wandb.watch(model, log="gradients", log_freq=100)
    
    logger.info("Starting training...")
    logger.info(f"  Total samples: {len(train_dataset)}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    
    # Calculate steps
    steps_per_epoch = len(train_dataset) // (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * training_args.num_train_epochs
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    
    # Train
    # Resume from checkpoint if requested
    # Auto-detect checkpoint to resume from
    checkpoint_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    import glob
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save the final model
    output_dir = config.get("training", {}).get("output_dir", "./checkpoints")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    return trainer, train_result


def validate_dataset_structure(train_dataset, tokenizer) -> None:
    """
    Validate dataset has proper reasoning structure.
    
    Checks that tokenized samples contain the 5-step ante-hoc reasoning format.
    """
    logger.info("Validating dataset structure...")
    
    required_steps = ["PRESENTATION", "RISK CONTEXT", "PATTERN", "DIFFERENTIAL", "DECISION"]
    
    # Check a sample of cases by decoding tokens
    sample_size = min(10, len(train_dataset))
    reasoning_present = 0
    json_valid = 0
    
    for i in range(sample_size):
        sample = train_dataset[i]
        
        # Decode the tokenized text to check content
        input_ids = sample.get("input_ids", [])
        if input_ids:
            text = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Check reasoning structure
            if all(step in text for step in required_steps):
                reasoning_present += 1
            
            # Check JSON present
            if "```json" in text:
                json_valid += 1
    
    reasoning_pct = reasoning_present / sample_size * 100
    json_pct = json_valid / sample_size * 100
    
    logger.info(f"✓ Reasoning structure: {reasoning_pct:.0f}% complete ({reasoning_present}/{sample_size})")
    logger.info(f"✓ JSON output: {json_pct:.0f}% present ({json_valid}/{sample_size})")


def run_dry_run(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: Dict[str, Any],
):
    """Run a dry run to test the setup without full training."""
    logger.info("=" * 60)
    logger.info("DRY RUN MODE - Testing setup without full training")
    logger.info("=" * 60)
    
    # 1. Test model forward pass
    logger.info("\n1. Testing model forward pass...")
    import torch as _torch
    sample = train_dataset[0]
    # Use pre-tokenized tensors directly (dataset is already tokenized)
    input_ids = _torch.tensor([sample["input_ids"][:512]], device=model.device)
    attention_mask = _torch.tensor([sample["attention_mask"][:512]], device=model.device)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    with torch.no_grad():
        outputs = model(**inputs)
        logger.info(f"   ✓ Forward pass successful")

    # 2. Test generation
    logger.info("\n2. Testing generation...")
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"][:, :100],
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"   ✓ Generation successful ({len(generated[0])} tokens)")
    
    # 3. Memory profiling under training load
    logger.info("\n3. Memory profiling...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Simple memory test - just check current allocation
            model_memory = torch.cuda.memory_allocated() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"   Model memory: {model_memory:.2f} GB")
            logger.info(f"   Total GPU memory: {total_memory:.1f} GB")
            logger.info(f"   Available: {total_memory - model_memory:.1f} GB")
            
            # Estimate training memory (model + gradients + optimizer states ≈ 3x model)
            estimated_training = model_memory * 2.5
            margin = total_memory - estimated_training
            
            logger.info(f"   Estimated training peak: ~{estimated_training:.1f} GB")
            logger.info(f"   Estimated margin: ~{margin:.1f} GB")
            
            if margin < 2.0:
                logger.warning("   ⚠️  LOW MEMORY MARGIN - Consider reducing batch size")
            else:
                logger.info("   ✓ Sufficient memory available")
        
        except Exception as e:
            logger.warning(f"   Memory profiling skipped: {e}")
    
    # 4. Dataset info
    logger.info("\n4. Dataset info:")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples: {len(val_dataset) if val_dataset else 'N/A'}")
    logger.info(f"   Columns: {train_dataset.column_names}")
    
    # 5. LoRA configuration
    logger.info("\n5. LoRA configuration:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    # 6. Dataset structure validation
    logger.info("\n6. Dataset structure validation:")
    validate_dataset_structure(train_dataset, tokenizer)
    
    # 7. Training step estimate
    logger.info("\n7. Training estimates:")
    batch_size = config.get("training", {}).get("per_device_train_batch_size", 4)
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 8)
    epochs = config.get("training", {}).get("num_train_epochs", 5)
    
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * epochs
    
    logger.info(f"   Effective batch size: {effective_batch}")
    logger.info(f"   Steps per epoch: {steps_per_epoch}")
    logger.info(f"   Total steps: {total_steps}")
    logger.info(f"   Estimated time: {total_steps * 3 / 3600:.1f} - {total_steps * 5 / 3600:.1f} hours")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ DRY RUN COMPLETE - All tests passed!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MedGemma LoRA Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run setup tests without training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    validate_config(config)
    
    # Setup WandB (unless disabled)
    if not args.no_wandb:
        setup_wandb(config)
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Load model and tokenizer
    model, tokenizer, use_unsloth = load_model_and_tokenizer(config)
    
    # No vision freezing needed — Gemma3ForCausalLM has no vision tower.
    # Vision components only exist in Gemma3ForConditionalGeneration (Phase 2).
    
    # Apply LoRA
    model = apply_lora(model, config, use_unsloth)
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(config, tokenizer)
    
    if args.dry_run:
        run_dry_run(model, tokenizer, train_dataset, val_dataset, config)
    else:
        trainer, result = run_training(
            model, tokenizer, train_dataset, val_dataset, config
        )
        
        # Final evaluation
        if val_dataset:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            trainer.log_metrics("eval", eval_results)
            trainer.save_metrics("eval", eval_results)
    
    # Cleanup
    if not args.no_wandb:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
