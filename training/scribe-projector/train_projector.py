#!/usr/bin/env python3
"""Train the AudioProjector to map MedASR encoder embeddings to MedGemma space.

Architecture:
  MedASR encoder (frozen) → AudioProjector (trained, 11.8M) → MedGemma (frozen, 4-bit)

Training sequence layout (matches inference exactly):
  [system_prompt + manifest] [<audio_start> audio_frames <audio_end>] [target_output]
        text embeddings              projected embeddings              text embeddings
        labels = -100                labels = -100                     labels = token_ids

Losses:
  L_lm:          Causal LM cross-entropy on target output tokens.
  L_contrastive: (single-phrase clips only) Cosine similarity between
                 mean-pooled audio embeddings and the concept's text embedding.
                 Provides direct gradient to projector independent of frozen LLM.

  L_total = L_lm + λ_contrastive * L_contrastive

MedGemma has 128K context — we use this to include full manifests (25-45 concept
lines ≈ 200-400 tokens), keeping total sequence well under 1K tokens.

Optimizations (v2):
  - Pre-computed encoder embeddings via --precomputed-dir (skips encoder entirely)
  - Gradient checkpointing on MedGemma (frees ~25GB VRAM for larger batches)
  - Full checkpoint saving (optimizer, scheduler, global_step for seamless resume)
  - --original-total-steps to preserve LR schedule across batch size changes

Usage:
    python scripts/train_projector.py \
        --clips-jsonl /medASR/data/audio/clips.jsonl \
        --audio-dir /medASR/data/audio/clips \
        --precomputed-dir /medASR/data/audio/encoder_cache \
        --epochs 3 \
        --batch-size 8 \
        --lr 1e-3 \
        --contrastive-weight 0.1 \
        --resume-checkpoint checkpoints/projector/projector_step45000.pt
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

SYSTEM_PROMPT = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "Audio embeddings from a clinical recording are provided between "
    "<audio_start> and <audio_end> markers.\n"
    "Extract structured medical observations from the audio.\n"
    "Return ONLY key: value lines matching concepts from the manifest.\n\n"
)


class ClipAudioDataset(Dataset):
    """Dataset that loads clip definitions and returns encoder embeddings.

    Supports two modes:
      - precomputed_dir set: loads pre-computed .pt encoder embeddings (fast)
      - precomputed_dir None: loads WAV files for live encoding (original behavior)
    """

    def __init__(self, clips_jsonl: str, audio_dir: str,
                 manifest_jsonl: str = None, precomputed_dir: str = None):
        self.audio_dir = Path(audio_dir)
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None
        self.samples = []

        logger.info("Loading clip definitions from %s ...", clips_jsonl)

        if manifest_jsonl:
            available = set()
            with open(manifest_jsonl) as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    key = f"{r['clip_id']}_{r['voice']}"
                    available.add(key)
            logger.info("Manifest has %d available (clip, voice) pairs", len(available))
        else:
            available = None

        with open(clips_jsonl) as f:
            for line in f:
                if not line.strip(): continue
                clip = json.loads(line)
                clip_id = clip["clip_id"]

                for voice in clip["voices"]:
                    key = f"{clip_id}_{voice}"
                    wav_path = self.audio_dir / f"{key}.wav"

                    if available is not None and key not in available:
                        continue
                    if not wav_path.exists():
                        continue
                    if self.precomputed_dir and not (self.precomputed_dir / f"{key}.pt").exists():
                        continue

                    concept_keys = []
                    for c in clip.get("concepts", []):
                        ml = c.get("manifest_line", "")
                        if ml:
                            key_clean = re.sub(r"^\[(dx|drug|test|value|order)\]\s*", "", ml)
                            key_clean = re.sub(r"\s*\(.*\)$", "", key_clean).strip()
                            concept_keys.append(key_clean)

                    self.samples.append({
                        "key": key,
                        "wav_path": str(wav_path),
                        "manifest": clip.get("manifest", ""),
                        "target": clip.get("output", ""),
                        "is_single_phrase": clip["phrase_count"] == 1,
                        "concept_keys": concept_keys,
                        "negative": clip.get("negative", False),
                    })

        logger.info("Dataset: %d samples (precomputed=%s)", len(self.samples),
                     "yes" if self.precomputed_dir else "no")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.precomputed_dir:
            enc_embs = torch.load(
                self.precomputed_dir / f"{s['key']}.pt",
                map_location="cpu", weights_only=True,
            )
        else:
            audio, _ = librosa.load(s["wav_path"], sr=SAMPLE_RATE)
            enc_embs = audio.astype(np.float32)

        return {
            "enc_embs": enc_embs,
            "manifest": s["manifest"],
            "target": s["target"],
            "is_single_phrase": s["is_single_phrase"],
            "concept_keys": s["concept_keys"],
            "needs_encoding": not bool(self.precomputed_dir),
        }


def collate_fn(batch):
    return {
        "enc_embs_list": [item["enc_embs"] for item in batch],
        "manifests": [item["manifest"] for item in batch],
        "targets": [item["target"] for item in batch],
        "is_single_phrase": [item["is_single_phrase"] for item in batch],
        "concept_keys": [item["concept_keys"] for item in batch],
        "needs_encoding": batch[0]["needs_encoding"],
    }


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_precomputed = args.precomputed_dir is not None

    # ── Load encoder (only if no pre-computed embeddings) ─────────────────────
    encoder = None
    processor = None
    if not use_precomputed:
        logger.info("Loading MedASR encoder from %s ...", args.encoder_path)
        from transformers import AutoModel, AutoProcessor
        encoder_model = AutoModel.from_pretrained(
            args.encoder_path, torch_dtype=torch.float32
        ).to(device)
        encoder_model.eval()
        encoder = encoder_model.encoder if hasattr(encoder_model, "encoder") else encoder_model
        for p in encoder.parameters():
            p.requires_grad = False
        processor = AutoProcessor.from_pretrained(args.encoder_path)
    else:
        logger.info("Using pre-computed encoder embeddings from %s", args.precomputed_dir)

    # ── Load MedGemma ─────────────────────────────────────────────────────────
    logger.info("Loading MedGemma from %s (frozen)...", args.medgemma_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.medgemma_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.medgemma_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    llm.eval()

    for p in llm.parameters():
        p.requires_grad = False

    vision_params = 0
    if hasattr(llm.model, "vision_tower"):
        for p in llm.model.vision_tower.parameters():
            vision_params += p.numel()
            assert not p.requires_grad
    if hasattr(llm.model, "multi_modal_projector"):
        for p in llm.model.multi_modal_projector.parameters():
            vision_params += p.numel()
            assert not p.requires_grad
    logger.info("Vision encoder + MM projector frozen: %d params", vision_params)

    # ── Gradient checkpointing (only when using pre-computed embeddings) ─────
    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on MedGemma")
    else:
        logger.info("Gradient checkpointing disabled")

    embed_layer = llm.get_input_embeddings()
    logger.info("Embed layer: %s", type(embed_layer).__name__)
    logger.info("MedGemma loaded (frozen, gradient_checkpointing=%s)",
                 "on" if args.gradient_checkpointing else "off")

    # ── AudioProjector ────────────────────────────────────────────────────────
    logger.info("Initializing AudioProjector...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from service.projector import AudioProjector

    projector = AudioProjector(encoder_dim=512, llm_dim=2560, stack_factor=4).to(device)
    logger.info("Projector params: %d", projector.param_count())

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ClipAudioDataset(
        args.clips_jsonl, args.audio_dir,
        manifest_jsonl=args.manifest_jsonl,
        precomputed_dir=args.precomputed_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 if use_precomputed else 2,
        pin_memory=True,
    )
    logger.info("Training samples: %d | Batches/epoch: %d | Batch size: %d",
                len(dataset), len(dataloader), args.batch_size)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=0.01)

    if args.original_total_steps:
        total_steps = args.original_total_steps
        logger.info("Using original total_steps=%d for LR schedule (preserving cosine curve)", total_steps)
    else:
        total_steps = len(dataloader) * args.epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0

    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "projector_state_dict" in ckpt:
            projector.load_state_dict(ckpt["projector_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                logger.info("Restored optimizer state")
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                logger.info("Restored scheduler state")
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", 0)
        else:
            projector.load_state_dict(ckpt)
            global_step = int(
                re.search(r"step(\d+)", str(args.resume_checkpoint)).group(1)
            ) if re.search(r"step(\d+)", str(args.resume_checkpoint)) else 0

            for _ in range(global_step):
                scheduler.step()
            logger.info("Fast-forwarded scheduler to step %d", global_step)

        logger.info("Resumed from %s at global_step=%d, epoch=%d",
                     args.resume_checkpoint, global_step, start_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    contrastive_w = args.contrastive_weight

    for epoch in range(start_epoch, args.epochs):
        projector.train()
        epoch_lm_loss = 0.0
        epoch_con_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            enc_embs_list = batch["enc_embs_list"]
            manifests = batch["manifests"]
            targets = batch["targets"]
            is_single = batch["is_single_phrase"]
            concept_keys_batch = batch["concept_keys"]
            needs_encoding = batch["needs_encoding"]

            all_inputs_embeds = []
            all_labels = []
            contrastive_pairs = []

            for enc_embs_raw, manifest, target, single, concept_keys in zip(
                enc_embs_list, manifests, targets, is_single, concept_keys_batch
            ):
                # 1. Get encoder embeddings → project
                if needs_encoding:
                    inputs = processor(enc_embs_raw, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                    input_features = inputs["input_features"].to(device)
                    with torch.no_grad():
                        enc_out = encoder(input_features=input_features)
                        enc_embs = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
                else:
                    enc_embs = enc_embs_raw.unsqueeze(0).float().to(device)

                projected = projector(enc_embs)

                # 2. Build prompt
                prompt_text = SYSTEM_PROMPT + manifest + "\n\nOUTPUT:\n"
                prompt_ids = tokenizer.encode(
                    prompt_text, return_tensors="pt", add_special_tokens=True
                ).to(llm.device)

                # 3. Encode target output
                target_ids = tokenizer.encode(
                    target, return_tensors="pt", add_special_tokens=False
                ).to(llm.device)
                eos_id = torch.tensor([[tokenizer.eos_token_id]], device=llm.device)
                target_ids = torch.cat([target_ids, eos_id], dim=1)

                # 4. Get text embeddings (frozen)
                with torch.no_grad():
                    prompt_embeds = embed_layer(prompt_ids)
                    target_embeds = embed_layer(target_ids)

                projected_cast = projected.to(
                    dtype=prompt_embeds.dtype, device=prompt_embeds.device
                )

                # 5. Concatenate
                inputs_embeds = torch.cat(
                    [prompt_embeds, projected_cast, target_embeds], dim=1
                )

                # 6. Labels
                prompt_len = prompt_embeds.shape[1]
                audio_len = projected_cast.shape[1]
                target_len = target_ids.shape[1]

                labels = torch.full(
                    (1, prompt_len + audio_len + target_len),
                    -100, dtype=torch.long, device=llm.device,
                )
                labels[0, prompt_len + audio_len:] = target_ids[0]

                all_inputs_embeds.append(inputs_embeds)
                all_labels.append(labels)

                # 7. Contrastive pairs (single-phrase only)
                if single and concept_keys and contrastive_w > 0:
                    audio_only = projected_cast[0, 1:-1, :]
                    audio_mean = audio_only.mean(dim=0)
                    contrastive_pairs.append((audio_mean, concept_keys[0]))

            # ── Pad batch ─────────────────────────────────────────────────────
            max_len = max(ie.shape[1] for ie in all_inputs_embeds)
            embed_dim = all_inputs_embeds[0].shape[2]

            padded_embeds = torch.zeros(
                len(all_inputs_embeds), max_len, embed_dim,
                dtype=all_inputs_embeds[0].dtype,
                device=all_inputs_embeds[0].device,
            )
            padded_labels = torch.full(
                (len(all_labels), max_len), -100,
                dtype=torch.long, device=all_labels[0].device,
            )
            attention_mask = torch.zeros(
                len(all_inputs_embeds), max_len,
                dtype=torch.long, device=all_inputs_embeds[0].device,
            )

            for i, (ie, lb) in enumerate(zip(all_inputs_embeds, all_labels)):
                seq_len = ie.shape[1]
                padded_embeds[i, :seq_len] = ie[0]
                padded_labels[i, :seq_len] = lb[0]
                attention_mask[i, :seq_len] = 1

            # ── LM loss ──────────────────────────────────────────────────────
            outputs = llm(
                inputs_embeds=padded_embeds,
                attention_mask=attention_mask,
                labels=padded_labels,
            )
            lm_loss = outputs.loss

            # ── Contrastive loss ──────────────────────────────────────────────
            con_loss = torch.tensor(0.0, device=device)
            if contrastive_pairs and contrastive_w > 0:
                cos_losses = []
                for audio_mean, concept_key in contrastive_pairs:
                    concept_ids = tokenizer.encode(
                        concept_key, add_special_tokens=False, return_tensors="pt"
                    ).to(llm.device)
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

            # ── Total loss ────────────────────────────────────────────────────
            loss = lm_loss + contrastive_w * con_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_lm_loss += lm_loss.item()
            epoch_con_loss += con_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Step %d | LM: %.4f | Con: %.4f | Total: %.4f | LR: %.2e",
                    global_step, lm_loss.item(), con_loss.item(), loss.item(), lr,
                )

            if global_step % 5000 == 0 and global_step > 0:
                ckpt_path = checkpoint_dir / f"projector_step{global_step}.pt"
                torch.save({
                    "projector_state_dict": projector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_path)
                logger.info("Mid-epoch checkpoint: %s", ckpt_path)

        avg_lm = epoch_lm_loss / max(num_batches, 1)
        avg_con = epoch_con_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d | LM: %.4f | Contrastive: %.4f",
                     epoch + 1, args.epochs, avg_lm, avg_con)

        ckpt_path = checkpoint_dir / f"projector_epoch{epoch + 1}.pt"
        torch.save({
            "projector_state_dict": projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch + 1,
        }, ckpt_path)
        logger.info("Checkpoint: %s", ckpt_path)

    final_path = checkpoint_dir / "projector_final.pt"
    torch.save({
        "projector_state_dict": projector.state_dict(),
        "global_step": global_step,
    }, final_path)
    logger.info("Training complete. Final: %s", final_path)


def main():
    parser = argparse.ArgumentParser(description="Train AudioProjector")
    parser.add_argument("--clips-jsonl", type=str, required=True,
                        help="Path to clips.jsonl (clip definitions)")
    parser.add_argument("--audio-dir", type=str, default="/medASR/data/audio/clips",
                        help="Directory containing WAV files")
    parser.add_argument("--precomputed-dir", type=str, default=None,
                        help="Directory with pre-computed encoder .pt files (skips encoder)")
    parser.add_argument("--manifest-jsonl", type=str, default=None,
                        help="Optional manifest.jsonl to filter available (clip,voice) pairs")
    parser.add_argument("--encoder-path", type=str, default="/medASR/model")
    parser.add_argument("--medgemma-path", type=str, default="/medASR/medgemma")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss on single-phrase clips (0=disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="/var/www/ClinicDx/training/checkpoints")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--original-total-steps", type=int, default=None,
                        help="Preserve original LR schedule total_steps across batch size changes")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing on MedGemma (saves VRAM, slower backward)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
