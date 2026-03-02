#!/usr/bin/env python3
"""Train the AudioProjector to map MedASR encoder embeddings to MedGemma space.

Only the AudioProjector (~11.8M params) is trainable. Everything else is frozen:
  - MedASR encoder (105M params)
  - Gemma3 language model + vision tower + vision projector (4.3B params)

Uses the same masked_scatter integration pattern as vision in Gemma3.

Training sequence layout:
  [system_prompt + manifest]  [<boa> <audio_soft_token>x64 <eoa>]  [target output]
       text tokens (frozen)        audio embeddings (projected)      text tokens (frozen)
       labels = -100               labels = -100                     labels = token_ids

Losses:
  L_lm:          Cross-entropy on target output tokens.
  L_contrastive: Cosine similarity for single-phrase clips (weight=0.1).
  L_total = L_lm + lambda * L_contrastive

Usage:
    python scripts/train_audio_projector.py configs/train_config.yaml
"""

import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.gemma3_audio import Gemma3WithAudioModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class PrecomputedClipDataset(Dataset):
    """Loads clip definitions with pre-computed MedASR encoder embeddings."""

    def __init__(self, clips_jsonl, precomputed_dir, manifest_jsonl=None, max_clips=0):
        self.precomputed_dir = Path(precomputed_dir)
        self.samples = []

        if manifest_jsonl:
            available = set()
            with open(manifest_jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    available.add(f"{r['clip_id']}_{r['voice']}")
            logger.info("Manifest filter: %d available pairs", len(available))
        else:
            available = None

        logger.info("Pre-scanning encoder cache directory...")
        cached_keys = {p.stem for p in self.precomputed_dir.iterdir() if p.suffix == ".pt"}
        logger.info("Found %d cached .pt files", len(cached_keys))

        with open(clips_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                clip = json.loads(line)
                clip_id = clip["clip_id"]

                for voice in clip["voices"]:
                    key = f"{clip_id}_{voice}"

                    if available is not None and key not in available:
                        continue
                    if key not in cached_keys:
                        continue
                    pt_path = self.precomputed_dir / f"{key}.pt"

                    concept_keys = []
                    for c in clip.get("concepts", []):
                        ml = c.get("manifest_line", "")
                        if ml:
                            cleaned = re.sub(r"^\[(dx|drug|test|value|order)\]\s*", "", ml)
                            cleaned = re.sub(r"\s*\(.*\)$", "", cleaned).strip()
                            concept_keys.append(cleaned)

                    self.samples.append({
                        "key": key,
                        "pt_path": str(pt_path),
                        "manifest": clip.get("manifest", ""),
                        "target": clip.get("output", ""),
                        "is_single_phrase": clip.get("phrase_count", 0) == 1,
                        "concept_keys": concept_keys,
                        "negative": clip.get("negative", False),
                    })

                    if max_clips > 0 and len(self.samples) >= max_clips:
                        break
                if max_clips > 0 and len(self.samples) >= max_clips:
                    break

        logger.info("Dataset: %d samples (precomputed)", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc_embs = torch.load(s["pt_path"], map_location="cpu", weights_only=True)
        if enc_embs.dim() == 2:
            enc_embs = enc_embs.unsqueeze(0)
        return {
            "enc_embs": enc_embs.float(),
            "manifest": s["manifest"],
            "target": s["target"],
            "is_single_phrase": s["is_single_phrase"],
            "concept_keys": s["concept_keys"],
        }


def collate_fn(batch):
    return {
        "enc_embs_list": [item["enc_embs"] for item in batch],
        "manifests": [item["manifest"] for item in batch],
        "targets": [item["target"] for item in batch],
        "is_single_phrase": [item["is_single_phrase"] for item in batch],
        "concept_keys": [item["concept_keys"] for item in batch],
    }


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    logger.info("=" * 60)
    logger.info("  AudioProjector Training")
    logger.info("  Base model: %s", model_cfg["base_model_path"])
    logger.info("  Audio encoder: %s", model_cfg["audio_encoder_path"])
    logger.info("=" * 60)

    model_config_path = Path(model_cfg["base_model_path"]) / "config.json"
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = Gemma3WithAudioModel(
        base_model_path=model_cfg["base_model_path"],
        audio_encoder_path=model_cfg["audio_encoder_path"],
        config=model_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
    )
    model.load_base_model()
    model._base_model.gradient_checkpointing_enable()
    device = next(model._base_model.parameters()).device
    model.audio_projector = model.audio_projector.to(device)
    logger.info(
        "AudioProjector initialized (%d trainable params) — skipping MedASR encoder (using precomputed embeddings)",
        model.audio_projector.param_count(),
    )

    tokenizer = model._base_model.config
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_path"])

    embed_layer = model._base_model.get_input_embeddings()
    audio_token_id = model.audio_token_id
    mm_tokens_per_audio = model.mm_tokens_per_audio

    logger.info("Audio token ID: %d", audio_token_id)
    logger.info("Tokens per audio: %d", mm_tokens_per_audio)
    logger.info("Projector params: %d", model.audio_projector.param_count())

    dataset = PrecomputedClipDataset(
        clips_jsonl=data_cfg["clips_jsonl"],
        precomputed_dir=data_cfg["precomputed_dir"],
        manifest_jsonl=data_cfg.get("manifest_jsonl"),
        max_clips=data_cfg.get("max_clips", 0),
    )

    val_size = int(len(dataset) * data_cfg.get("val_fraction", 0.05))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    dataloader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.audio_projector.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = (len(dataloader) * train_cfg["epochs"]) // train_cfg.get("gradient_accumulation_steps", 1)
    warmup_steps = min(train_cfg["warmup_steps"], total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    contrastive_w = train_cfg["contrastive_weight"]
    system_prompt = cfg.get("system_prompt", "").strip()
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    global_step = 0
    micro_step = 0
    start_epoch = 0

    resume_ckpt = train_cfg.get("resume_checkpoint")
    if resume_ckpt and Path(resume_ckpt).exists():
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        model.audio_projector.load_state_dict(ckpt["projector_state_dict"])
        model.audio_projector = model.audio_projector.to(device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        micro_step = global_step * grad_accum_steps
        logger.info("Resumed at epoch %d, global_step %d", start_epoch, global_step)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.audio_projector.train()
        epoch_lm_loss = 0.0
        epoch_con_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            enc_embs_list = batch["enc_embs_list"]
            manifests = batch["manifests"]
            targets = batch["targets"]
            is_single = batch["is_single_phrase"]
            concept_keys_batch = batch["concept_keys"]

            all_inputs_embeds = []
            all_labels = []
            contrastive_pairs = []

            for enc_embs, manifest, target, single, concept_keys in zip(
                enc_embs_list, manifests, targets, is_single, concept_keys_batch
            ):
                enc_embs = enc_embs.to(device)
                projected = model.audio_projector(enc_embs)

                prompt_text = system_prompt + "\n" + manifest + "\n\nOUTPUT:\n"
                prompt_ids = tokenizer.encode(
                    prompt_text, return_tensors="pt", add_special_tokens=True,
                ).to(device)

                target_ids = tokenizer.encode(
                    target, return_tensors="pt", add_special_tokens=False,
                ).to(device)
                eos_id = torch.tensor(
                    [[tokenizer.eos_token_id]], device=device,
                )
                target_ids = torch.cat([target_ids, eos_id], dim=1)

                with torch.no_grad():
                    prompt_embeds = embed_layer(prompt_ids)
                    target_embeds = embed_layer(target_ids)

                projected_cast = projected.to(
                    dtype=prompt_embeds.dtype, device=prompt_embeds.device,
                )

                inputs_embeds = torch.cat(
                    [prompt_embeds, projected_cast, target_embeds], dim=1,
                )

                prompt_len = prompt_embeds.shape[1]
                audio_len = projected_cast.shape[1]
                target_len = target_ids.shape[1]

                labels = torch.full(
                    (1, prompt_len + audio_len + target_len),
                    -100, dtype=torch.long, device=device,
                )
                labels[0, prompt_len + audio_len:] = target_ids[0]

                all_inputs_embeds.append(inputs_embeds)
                all_labels.append(labels)

                if single and concept_keys and contrastive_w > 0:
                    audio_mean = projected_cast[0].mean(dim=0)
                    contrastive_pairs.append((audio_mean, concept_keys[0]))

            max_len = max(ie.shape[1] for ie in all_inputs_embeds)
            embed_dim = all_inputs_embeds[0].shape[2]

            padded_embeds = torch.zeros(
                len(all_inputs_embeds), max_len, embed_dim,
                dtype=all_inputs_embeds[0].dtype, device=device,
            )
            padded_labels = torch.full(
                (len(all_labels), max_len), -100,
                dtype=torch.long, device=device,
            )
            attention_mask = torch.zeros(
                len(all_inputs_embeds), max_len,
                dtype=torch.long, device=device,
            )

            for i, (ie, lb) in enumerate(zip(all_inputs_embeds, all_labels)):
                seq_len = ie.shape[1]
                padded_embeds[i, :seq_len] = ie[0]
                padded_labels[i, :seq_len] = lb[0]
                attention_mask[i, :seq_len] = 1

            outputs = model._base_model(
                inputs_embeds=padded_embeds,
                attention_mask=attention_mask,
                labels=padded_labels,
            )
            lm_loss = outputs.loss

            con_loss = torch.tensor(0.0, device=device)
            if contrastive_pairs and contrastive_w > 0:
                cos_losses = []
                for audio_mean, concept_key in contrastive_pairs:
                    concept_ids = tokenizer.encode(
                        concept_key, add_special_tokens=False, return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        concept_embeds = embed_layer(concept_ids)
                        concept_mean = concept_embeds.mean(dim=1).squeeze(0)
                    cos_sim = F.cosine_similarity(
                        audio_mean.unsqueeze(0).float(),
                        concept_mean.unsqueeze(0).float(),
                    )
                    cos_losses.append(1.0 - cos_sim)
                if cos_losses:
                    con_loss = torch.stack(cos_losses).mean()

            loss = (lm_loss + contrastive_w * con_loss) / grad_accum_steps

            loss.backward()

            epoch_lm_loss += lm_loss.item()
            epoch_con_loss += con_loss.item()
            num_batches += 1
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.audio_projector.parameters(),
                    train_cfg["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg["log_every_steps"] == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %d | LM: %.4f | Con: %.4f | LR: %.2e",
                        global_step, lm_loss.item(), con_loss.item(), lr,
                    )

                if global_step % train_cfg["checkpoint_every_steps"] == 0:
                    ckpt_path = ckpt_dir / f"projector_step{global_step}.pt"
                    model.save_projector_checkpoint(
                        str(ckpt_path), optimizer, scheduler, global_step, epoch,
                    )

        avg_lm = epoch_lm_loss / max(num_batches, 1)
        avg_con = epoch_con_loss / max(num_batches, 1)
        logger.info(
            "Epoch %d/%d | Avg LM: %.4f | Avg Con: %.4f",
            epoch + 1, train_cfg["epochs"], avg_lm, avg_con,
        )

        ckpt_path = ckpt_dir / f"projector_epoch{epoch + 1}.pt"
        model.save_projector_checkpoint(
            str(ckpt_path), optimizer, scheduler, global_step, epoch + 1,
        )

    final_path = ckpt_dir / "projector_final.pt"
    model.save_projector_checkpoint(str(final_path), global_step=global_step)
    logger.info("Training complete. Final checkpoint: %s", final_path)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"
    )
    logger.info("Config: %s", cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
