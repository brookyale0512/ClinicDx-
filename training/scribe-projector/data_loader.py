#!/usr/bin/env python3
"""
Data Loader for MedGemma Ante-hoc LoRA Training
================================================

Handles dataset loading, formatting with Gemma 3 chat template,
and preprocessing for instruction tuning.

Features:
- Proper Gemma 3 chat format with <bos>, <start_of_turn>, <end_of_turn>
- Ante-hoc reasoning structure formatting
- Token length validation and filtering
- Memory-efficient streaming support
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_file: str
    val_file: Optional[str] = None
    max_seq_length: int = 4096
    train_val_split: float = 0.9
    shuffle_seed: int = 42


class MedGemmaDataLoader:
    """
    Data loader for MedGemma ante-hoc reasoning training.
    
    Formats data using the official Gemma 3 chat template structure.
    """
    
    # Gemma 3 special tokens (from tokenizer_config.json)
    BOS_TOKEN = "<bos>"
    START_OF_TURN = "<start_of_turn>"
    END_OF_TURN = "<end_of_turn>"
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: DataConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config

        # Unwrap multimodal processor (e.g. Gemma3Processor) to get text tokenizer
        # for .encode() calls. The processor itself is kept for __call__ compatibility.
        if hasattr(tokenizer, 'tokenizer'):
            self._text_tokenizer = tokenizer.tokenizer
        else:
            self._text_tokenizer = tokenizer

        # Ensure tokenizer has proper padding
        if self._text_tokenizer.pad_token is None:
            self._text_tokenizer.pad_token = self._text_tokenizer.eos_token

        logger.info(f"Initialized MedGemmaDataLoader with max_seq_length={config.max_seq_length}")
    
    def format_training_sample(self, sample: Dict[str, Any]) -> str:
        """
        Format a single training sample using Gemma 3 chat template.

        If the sample already has a 'text' field, use it directly
        (supports pre-formatted multi-turn formats like KB-query PoC).

        Structure:
        <bos><start_of_turn>user
        {instruction}

        {input}<end_of_turn>
        <start_of_turn>model
        ## Clinical Reasoning

        {reasoning}

        ## Structured Output

        ```json
        {output}
        ```<end_of_turn>
        """
        # Support pre-formatted text (e.g., multi-turn KB-query format)
        if 'text' in sample and sample['text']:
            return sample['text']

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reasoning = sample.get("reasoning", "")
        output = sample.get("output", "")
        
        # Build the formatted text using Gemma 3 chat template
        formatted = f"""{self.BOS_TOKEN}{self.START_OF_TURN}user
{instruction}

{input_text}{self.END_OF_TURN}
{self.START_OF_TURN}model
## Clinical Reasoning

{reasoning}

## Structured Output

```json
{output}
```{self.END_OF_TURN}"""
        
        return formatted
    
    def format_inference_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a sample for inference (user turn only, no model response).
        
        Used for evaluation and generation testing.
        """
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        
        formatted = f"""{self.BOS_TOKEN}{self.START_OF_TURN}user
{instruction}

{input_text}{self.END_OF_TURN}
{self.START_OF_TURN}model
"""
        return formatted
    
    def tokenize_sample(
        self,
        sample: Dict[str, Any],
        for_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single sample.
        
        Args:
            sample: Raw sample dict with instruction, input, reasoning, output
            for_training: If True, include full response; if False, prompt only
        
        Returns:
            Dict with input_ids, attention_mask, and labels (for training)
        """
        if for_training:
            text = self.format_training_sample(sample)
        else:
            text = self.format_inference_prompt(sample)
        
        # Tokenize
        encoded = self._text_tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        if for_training:
            # For training, labels = input_ids (causal LM)
            encoded["labels"] = encoded["input_ids"].copy()
        
        return encoded
    
    def prepare_dataset(
        self,
        samples: List[Dict[str, Any]],
        for_training: bool = True,
        show_progress: bool = True
    ) -> Dataset:
        """
        Prepare a HuggingFace Dataset from raw samples.
        
        Args:
            samples: List of raw sample dicts
            for_training: Whether to prepare for training (includes labels)
            show_progress: Show tokenization progress
        
        Returns:
            HuggingFace Dataset ready for training
        """
        logger.info(f"Preparing dataset with {len(samples)} samples...")
        
        # Format all samples
        formatted_samples = []
        skipped = 0
        
        for i, sample in enumerate(samples):
            try:
                if for_training:
                    text = self.format_training_sample(sample)
                else:
                    text = self.format_inference_prompt(sample)
                
                # Check token length before adding
                tokens = self._text_tokenizer.encode(text, add_special_tokens=False)
                
                if len(tokens) > self.config.max_seq_length:
                    skipped += 1
                    if skipped <= 5:
                        logger.warning(
                            f"Sample {sample.get('id', i)} exceeds max_seq_length "
                            f"({len(tokens)} > {self.config.max_seq_length}), skipping"
                        )
                    continue
                
                formatted_samples.append({
                    "id": sample.get("id", f"sample_{i}"),
                    "category": sample.get("category", "unknown"),
                    "text": text,
                    "token_count": len(tokens),
                })
                
            except Exception as e:
                logger.error(f"Error formatting sample {i}: {e}")
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} samples due to length or errors")
        
        logger.info(f"Formatted {len(formatted_samples)} samples successfully")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(formatted_samples)
        
        # Tokenize using map (use text tokenizer for reliable encode)
        def tokenize_function(examples):
            tokenized = self._text_tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )
            # Add token_type_ids for Gemma 3 (required during training)
            # All zeros since we're doing causal LM, not distinguishing segments
            tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]
            return tokenized
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        # Add labels for training
        if for_training:
            dataset = dataset.map(
                lambda x: {"labels": x["input_ids"]},
                desc="Adding labels"
            )
        
        return dataset
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(samples)} samples from {file_path}")
        return samples
    
    def load_and_prepare(
        self,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Load and prepare train/validation datasets.
        
        Args:
            train_file: Path to training JSONL (uses config if None)
            val_file: Path to validation JSONL (uses config if None)
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_path = train_file or self.config.train_file
        val_path = val_file or self.config.val_file
        
        # Load training data
        train_samples = self.load_jsonl(train_path)
        train_dataset = self.prepare_dataset(train_samples, for_training=True)
        
        # Load or split validation data
        val_dataset = None
        if val_path and Path(val_path).exists():
            val_samples = self.load_jsonl(val_path)
            val_dataset = self.prepare_dataset(val_samples, for_training=True)
        
        return train_dataset, val_dataset
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """Get statistics about a dataset."""
        token_counts = dataset["token_count"]
        
        return {
            "num_samples": len(dataset),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "total_tokens": sum(token_counts),
        }


def create_data_collator(tokenizer: AutoTokenizer, max_length: int = 4096):
    """
    Create a data collator for dynamic padding.
    
    This ensures efficient batching by padding to the longest
    sequence in each batch rather than max_length.
    """
    from transformers import DataCollatorForLanguageModeling
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # Efficient tensor operations
    )


# =============================================================================
# Utility Functions
# =============================================================================

def validate_sample_format(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a sample has all required fields.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    required_fields = ["instruction", "input", "reasoning", "output"]
    issues = []
    
    for field in required_fields:
        if field not in sample:
            issues.append(f"Missing field: {field}")
        elif not sample[field] or len(str(sample[field]).strip()) < 10:
            issues.append(f"Empty or too short: {field}")
    
    # Check reasoning has the 5-step structure
    reasoning = sample.get("reasoning", "")
    required_steps = ["PRESENTATION", "RISK CONTEXT", "PATTERN", "DIFFERENTIAL", "DECISION"]
    
    for step in required_steps:
        if step not in reasoning:
            issues.append(f"Missing reasoning step: {step}")
    
    return len(issues) == 0, issues


def estimate_training_tokens(
    samples: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Estimate total training tokens from a sample.
    
    Args:
        samples: List of raw samples
        tokenizer: Tokenizer to use
        sample_size: Number of samples to use for estimation
    
    Returns:
        Dict with token statistics and estimates
    """
    import random
    
    loader = MedGemmaDataLoader(
        tokenizer=tokenizer,
        config=DataConfig(train_file="", max_seq_length=4096)
    )
    
    # Sample for estimation
    sample_subset = random.sample(samples, min(sample_size, len(samples)))
    
    token_counts = []
    for sample in sample_subset:
        text = loader.format_training_sample(sample)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
    
    avg_tokens = sum(token_counts) / len(token_counts)
    
    return {
        "samples_analyzed": len(token_counts),
        "avg_tokens_per_sample": avg_tokens,
        "estimated_total_tokens": avg_tokens * len(samples),
        "max_tokens_in_sample": max(token_counts),
        "min_tokens_in_sample": min(token_counts),
    }


if __name__ == "__main__":
    # Test the data loader
    print("Testing MedGemmaDataLoader...")
    
    # Create a test sample
    test_sample = {
        "id": "TEST-001",
        "category": "diabetes",
        "instruction": "You are a clinical decision support assistant...",
        "input": "<case>\n<demographics>Patient: 65yo male</demographics>\n</case>",
        "reasoning": "1. PRESENTATION: ...\n2. RISK CONTEXT: ...\n3. PATTERN: ...\n4. DIFFERENTIAL: ...\n5. DECISION: ...",
        "output": '{"urgency": {"level": "emergency"}}',
    }
    
    # Format and print
    loader = MedGemmaDataLoader(
        tokenizer=None,  # Will fail, just for format testing
        config=DataConfig(train_file="", max_seq_length=4096)
    )
    
    formatted = loader.format_training_sample(test_sample)
    print("Formatted sample preview (first 500 chars):")
    print(formatted[:500])
    print("...")
    print(f"\nTotal length: {len(formatted)} characters")
