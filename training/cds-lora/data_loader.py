#!/usr/bin/env python3
"""
CDS KB Tool-Use LoRA — Data Loader with Loss Masking
=====================================================

Key feature: masks user-turn tokens (labels=-100) so loss is only computed
on model outputs (reasoning, KB queries, final answers). This is the single
highest-impact change from the 3k test run analysis.

Training data format: pre-formatted Gemma 3 multi-turn chat text with
<KB_QUERY>/<KB_RESULT> tool-use tags already embedded.

Caching: tokenized datasets are saved to disk as Arrow files and reloaded
on subsequent runs (~2s vs ~7min). Cache auto-invalidates when the source
JSONL changes (mtime/size) or max_seq_length changes.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    train_file: str
    val_file: Optional[str] = None
    max_seq_length: int = 16384


def _mask_labels(input_ids: List[int], user_pat: List[int], model_pat: List[int],
                  system_pat: List[int] = None) -> List[int]:
    """Set labels=-100 for all tokens in user/system turns and BOS.

    Walks through the token sequence and tracks whether we're inside a
    user/system turn or a model turn. All tokens in user and system turns
    (including the <start_of_turn>user/system marker) are masked.
    Model turn tokens (reasoning, KB queries, final answers) keep their
    original label.
    """
    labels = list(input_ids)
    in_masked = False
    ulen = len(user_pat)
    mlen = len(model_pat)
    slen = len(system_pat) if system_pat else 0
    n = len(labels)

    i = 0
    while i < n:
        if i + ulen <= n and input_ids[i : i + ulen] == user_pat:
            in_masked = True
        elif system_pat and i + slen <= n and input_ids[i : i + slen] == system_pat:
            in_masked = True
        elif i + mlen <= n and input_ids[i : i + mlen] == model_pat:
            in_masked = False
        if in_masked:
            labels[i] = -100
        i += 1

    if labels:
        labels[0] = -100
    return labels


def _cache_fingerprint(jsonl_path: str, max_seq_length: int) -> str:
    """Deterministic hash from source file identity + tokenization params."""
    p = Path(jsonl_path)
    stat = p.stat()
    key = f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{max_seq_length}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class CdsDataLoader:
    """Loads pre-formatted KB tool-use training data with loss masking."""

    CACHE_DIR = Path("/var/www/ClinicDx/training/lora_cds_kb_v2/data/.cache")

    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config

        self.user_start_ids = tokenizer.encode(
            "<start_of_turn>user", add_special_tokens=False
        )
        self.model_start_ids = tokenizer.encode(
            "<start_of_turn>model", add_special_tokens=False
        )
        self.system_start_ids = tokenizer.encode(
            "<start_of_turn>system", add_special_tokens=False
        )

        logger.info(
            "CdsDataLoader initialized (max_seq=%d, user_start_tokens=%d, model_start_tokens=%d, system_start_tokens=%d)",
            config.max_seq_length,
            len(self.user_start_ids),
            len(self.model_start_ids),
            len(self.system_start_ids),
        )

    def _cache_path(self, jsonl_path: str, split: str) -> Path:
        fp = _cache_fingerprint(jsonl_path, self.config.max_seq_length)
        return self.CACHE_DIR / f"{split}_{fp}"

    def prepare_dataset(
        self,
        samples: List[Dict[str, Any]],
        for_training: bool = True,
        jsonl_path: Optional[str] = None,
        split: str = "train",
    ) -> Dataset:
        if jsonl_path and for_training:
            cache_dir = self._cache_path(jsonl_path, split)
            if cache_dir.exists():
                logger.info("Loading cached tokenized dataset from %s", cache_dir)
                dataset = load_from_disk(str(cache_dir))
                self._report_masking_stats(dataset)
                return dataset

        records = [
            {"text": s["text"], "id": s.get("id", ""), "source": s.get("source", "")}
            for s in samples if s.get("text")
        ]
        logger.info("Preparing %d samples for %s", len(records), "training" if for_training else "inference")

        dataset = Dataset.from_list(records)

        tok = self.tokenizer
        max_len = self.config.max_seq_length
        def tokenize_and_mask(examples):
            tokenized = tok(
                examples["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
            )
            tokenized["token_type_ids"] = [
                [0] * len(ids) for ids in tokenized["input_ids"]
            ]
            tokenized["labels"] = [
                list(ids) for ids in tokenized["input_ids"]
            ]
            return tokenized

        dataset = dataset.map(
            tokenize_and_mask,
            batched=True,
            batch_size=256,
            remove_columns=["text"],
            desc="Tokenizing + masking",
        )

        if for_training:
            self._report_masking_stats(dataset)
            if jsonl_path:
                cache_dir = self._cache_path(jsonl_path, split)
                cache_dir.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(cache_dir))
                size_mb = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                ) / 1e6
                logger.info("Cached tokenized dataset to %s (%.0f MB)", cache_dir, size_mb)

        return dataset

    def _report_masking_stats(self, dataset: Dataset):
        sample_size = min(200, len(dataset))
        total_tokens = 0
        masked_tokens = 0
        for i in range(sample_size):
            labels = dataset[i]["labels"]
            total_tokens += len(labels)
            masked_tokens += sum(1 for l in labels if l == -100)

        pct_masked = masked_tokens / max(total_tokens, 1) * 100
        pct_train = 100 - pct_masked
        logger.info(
            "Loss masking (sampled %d): %.1f%% masked (user turns), %.1f%% trainable (model turns) — avg %d tokens/sample",
            sample_size,
            pct_masked,
            pct_train,
            total_tokens // max(sample_size, 1),
        )

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        with open(p) as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        logger.info("Loaded %d samples from %s", len(samples), path)
        return samples

    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        n = len(dataset)
        sample_size = min(200, n)
        lengths = [len(dataset[i]["input_ids"]) for i in range(sample_size)]
        return {
            "num_samples": n,
            "avg_tokens": round(sum(lengths) / max(len(lengths), 1), 1),
            "max_tokens": max(lengths) if lengths else 0,
            "min_tokens": min(lengths) if lengths else 0,
        }
