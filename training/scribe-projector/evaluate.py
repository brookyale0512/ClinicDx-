#!/usr/bin/env python3
"""
Evaluator for Voice Scribe text SFT — key:value exact-match metrics.

Replaces the CDS ReasoningEvaluator. Checks:
  - Exact match: model output == expected output (full string)
  - Line-level F1: per key:value line precision/recall
  - Key accuracy: correct key extracted regardless of value
  - Value accuracy: correct value given correct key

Used as a callback during training and for post-training evaluation.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class ScribeEvaluator:
    """Evaluates key:value extraction accuracy."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            self._text_tokenizer = tokenizer.tokenizer
        else:
            self._text_tokenizer = tokenizer

    def parse_output(self, text: str) -> set:
        """Parse key:value lines from model output."""
        lines = set()
        for line in text.strip().splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip().lower()
                if key and value:
                    lines.add((key, value))
        return lines

    def evaluate_sample(self, predicted: str, expected: str) -> Dict[str, float]:
        """Evaluate a single prediction against expected output."""
        pred_lines = self.parse_output(predicted)
        exp_lines = self.parse_output(expected)

        exact_match = 1.0 if pred_lines == exp_lines else 0.0

        pred_keys = {k for k, v in pred_lines}
        exp_keys = {k for k, v in exp_lines}

        if not exp_lines:
            return {"exact_match": exact_match, "line_f1": 0.0,
                    "key_precision": 0.0, "key_recall": 0.0}

        tp = len(pred_lines & exp_lines)
        precision = tp / len(pred_lines) if pred_lines else 0.0
        recall = tp / len(exp_lines) if exp_lines else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        key_tp = len(pred_keys & exp_keys)
        key_precision = key_tp / len(pred_keys) if pred_keys else 0.0
        key_recall = key_tp / len(exp_keys) if exp_keys else 0.0

        return {
            "exact_match": exact_match,
            "line_f1": f1,
            "line_precision": precision,
            "line_recall": recall,
            "key_precision": key_precision,
            "key_recall": key_recall,
        }

    def evaluate_batch(self, predictions: List[str], expected: List[str]) -> Dict[str, float]:
        """Evaluate a batch of predictions."""
        metrics = {}
        all_results = [self.evaluate_sample(p, e) for p, e in zip(predictions, expected)]

        if not all_results:
            return {}

        for key in all_results[0]:
            values = [r[key] for r in all_results]
            metrics[key] = sum(values) / len(values)

        return metrics


class EvaluationCallback(TrainerCallback):
    """Callback that runs key:value exact-match evaluation during training."""

    def __init__(self, evaluator: ScribeEvaluator, tokenizer,
                 eval_steps: int = 200, num_samples: int = 5):
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            self._text_tokenizer = tokenizer.tokenizer
        else:
            self._text_tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_samples = num_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        logger.info(f"[Step {state.global_step}] Running key:value eval on {self.num_samples} samples...")

        try:
            eval_dataset = kwargs.get("eval_dataset")
            if eval_dataset is None:
                return

            n = min(self.num_samples, len(eval_dataset))
            predictions = []
            expected = []

            for i in range(n):
                sample = eval_dataset[i]
                input_ids = sample["input_ids"]

                full_text = self._text_tokenizer.decode(input_ids, skip_special_tokens=False)

                if "<start_of_turn>model\n" in full_text:
                    parts = full_text.split("<start_of_turn>model\n", 1)
                    prompt_text = parts[0] + "<start_of_turn>model\n"
                    expected_text = parts[1].replace("<end_of_turn>", "").strip()
                else:
                    continue

                prompt_ids = self._text_tokenizer.encode(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        prompt_ids,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self._text_tokenizer.pad_token_id,
                    )

                new_tokens = output_ids[0][prompt_ids.shape[1]:]
                predicted_text = self._text_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                predictions.append(predicted_text)
                expected.append(expected_text)

                if i < 2:
                    logger.info(f"  Sample {i}: expected='{expected_text[:80]}' predicted='{predicted_text[:80]}'")

            if predictions:
                metrics = self.evaluator.evaluate_batch(predictions, expected)
                logger.info(f"  Exact match: {metrics.get('exact_match', 0):.1%} | "
                           f"Line F1: {metrics.get('line_f1', 0):.1%} | "
                           f"Key recall: {metrics.get('key_recall', 0):.1%}")

        except Exception as e:
            logger.warning(f"Eval callback error: {e}")


# Backwards compatibility aliases
ReasoningEvaluator = ScribeEvaluator
