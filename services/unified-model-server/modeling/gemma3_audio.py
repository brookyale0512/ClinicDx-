"""Gemma3 extended with audio modality via MedASR encoder + AudioProjector.

Mirrors how vision is integrated in Gemma3ForConditionalGeneration:
  - Vision:  SigLIP (frozen) -> MultiModalProjector -> LLM
  - Audio:   MedASR (frozen) -> AudioProjector      -> LLM

The AudioProjector is the ONLY trainable component. The MedASR encoder and
the entire Gemma3 model (language_model + vision_tower + multi_modal_projector)
remain frozen during audio projector training.

Architecture (AudioProjector, ~11.8M params):
  Input:  [B, T_enc, 512]  (MedASR Conformer encoder output)
    | Frame stacking (k=4): concat 4 adjacent frames
  [B, T_enc/4, 2048]
    | Linear(2048 -> 2560, no bias)
    | RMSNorm(2560)
    | GELU
    | Linear(2560 -> 2560, no bias)
  [B, T_enc/4, 2560]
    | Pad/truncate to mm_tokens_per_audio (default 64)
  Output: [B, 64, 2560]  (MedGemma embedding space)

Token budget per audio duration (at 16kHz, hop=160, subsample=2x, stack=4x):
  1s -> 13 projected frames | 3s -> 38 | 5s -> 63 | 10s -> 125
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class Gemma3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (matches Gemma3 internal impl)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class Gemma3AudioProjector(nn.Module):
    """Projects MedASR encoder embeddings into Gemma3's embedding space.

    Mirrors Gemma3MultiModalProjector but for audio instead of vision.
    Uses frame stacking to reduce temporal resolution before projection.

    Args:
        encoder_dim: MedASR encoder hidden size (512).
        llm_dim: Gemma3 text hidden size (2560).
        stack_factor: Adjacent frames to concatenate (4).
        mm_tokens_per_audio: Fixed output token count (64).
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 2560,
        stack_factor: int = 4,
        mm_tokens_per_audio: int = 64,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.stack_factor = stack_factor
        self.mm_tokens_per_audio = mm_tokens_per_audio

        stacked_dim = encoder_dim * stack_factor  # 2048

        self.proj = nn.Sequential(
            nn.Linear(stacked_dim, llm_dim, bias=False),
            Gemma3RMSNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False),
        )

        self.audio_padding_emb = nn.Parameter(torch.zeros(1, 1, llm_dim))

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Project and normalize encoder embeddings to fixed token count.

        Args:
            encoder_output: [B, T_enc, encoder_dim] from MedASR encoder.

        Returns:
            [B, mm_tokens_per_audio, llm_dim] — fixed-size audio embeddings.
        """
        B, T, D = encoder_output.shape
        if D != self.encoder_dim:
            raise ValueError(
                f"Expected encoder_dim={self.encoder_dim}, got {D}"
            )

        k = self.stack_factor

        remainder = T % k
        if remainder != 0:
            pad_len = k - remainder
            padding = encoder_output.new_zeros(B, pad_len, D)
            encoder_output = torch.cat([encoder_output, padding], dim=1)
            T = encoder_output.shape[1]

        stacked = encoder_output.reshape(B, T // k, D * k)
        projected = self.proj(stacked)  # [B, T/k, llm_dim]

        projected = self._adjust_to_expected_length(projected)
        return projected

    def _adjust_to_expected_length(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Pad or truncate to mm_tokens_per_audio (mirrors Gemma3n approach)."""
        B, seq_len, dim = features.shape
        expected = self.mm_tokens_per_audio

        if seq_len < expected:
            pad_count = expected - seq_len
            padding = self.audio_padding_emb.expand(B, pad_count, dim)
            features = torch.cat([features, padding], dim=1)
        elif seq_len > expected:
            features = features[:, :expected, :]

        return features

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Gemma3WithAudioModel(nn.Module):
    """Gemma3 model extended with audio encoder and projector.

    Wraps a frozen Gemma3ForConditionalGeneration and adds:
      - audio_encoder: MedASR Conformer encoder (frozen)
      - audio_projector: Gemma3AudioProjector (trainable)

    The forward pass handles audio_values using the same masked_scatter
    pattern that Gemma3Model uses for pixel_values.
    """

    def __init__(
        self,
        base_model_path: str,
        audio_encoder_path: str,
        config: Optional[dict] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
    ):
        super().__init__()

        self.base_model_path = base_model_path
        self.audio_encoder_path = audio_encoder_path

        if config is None:
            import json
            with open(Path(base_model_path) / "config.json") as f:
                config = json.load(f)
        self.config = config

        audio_cfg = config.get("audio_config", {})
        text_cfg = config.get("text_config", {})

        self.audio_token_id = config.get("audio_token_index", 256003)
        self.mm_tokens_per_audio = config.get("mm_tokens_per_audio", 64)

        self.audio_projector = Gemma3AudioProjector(
            encoder_dim=audio_cfg.get("hidden_size", 512),
            llm_dim=text_cfg.get("hidden_size", 2560),
            stack_factor=config.get("audio_projector_stack_factor", 4),
            mm_tokens_per_audio=self.mm_tokens_per_audio,
        )

        self._base_model = None
        self._audio_encoder = None
        self._audio_processor = None
        self._torch_dtype = torch_dtype
        self._device_map = device_map

    def load_base_model(self) -> None:
        """Load the frozen Gemma3 base model."""
        logger.info("Loading base model from %s", self.base_model_path)
        self._base_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map or "auto",
        )
        self._base_model.eval()
        for p in self._base_model.parameters():
            p.requires_grad = False
        logger.info(
            "Base model loaded (frozen, %d params)",
            sum(p.numel() for p in self._base_model.parameters()),
        )

    def load_audio_encoder(self) -> None:
        """Load the frozen MedASR encoder."""
        logger.info("Loading MedASR encoder from %s", self.audio_encoder_path)
        full_model = AutoModel.from_pretrained(
            self.audio_encoder_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if hasattr(full_model, "encoder"):
            self._audio_encoder = full_model.encoder
        else:
            self._audio_encoder = full_model

        device = next(self._base_model.parameters()).device
        self._audio_encoder = self._audio_encoder.to(device)
        self._audio_encoder.eval()
        for p in self._audio_encoder.parameters():
            p.requires_grad = False

        self._audio_processor = AutoProcessor.from_pretrained(
            self.audio_encoder_path, trust_remote_code=True,
        )
        logger.info(
            "MedASR encoder loaded (frozen, %d params)",
            sum(p.numel() for p in self._audio_encoder.parameters()),
        )

    def load_all(self) -> None:
        """Load all components."""
        self.load_base_model()
        self.load_audio_encoder()
        device = next(self._base_model.parameters()).device
        self.audio_projector = self.audio_projector.to(device)
        logger.info(
            "AudioProjector initialized (%d trainable params)",
            self.audio_projector.param_count(),
        )

    def load_projector_checkpoint(self, path: str) -> None:
        """Load trained audio projector weights."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "projector_state_dict" in ckpt:
            self.audio_projector.load_state_dict(ckpt["projector_state_dict"])
        else:
            self.audio_projector.load_state_dict(ckpt)
        logger.info("Audio projector weights loaded from %s", path)

    def save_projector_checkpoint(
        self, path: str, optimizer=None, scheduler=None, global_step=0, epoch=0,
    ) -> None:
        """Save audio projector weights (and optionally optimizer state)."""
        ckpt = {
            "projector_state_dict": self.audio_projector.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, path)
        logger.info("Audio projector checkpoint saved to %s", path)

    def get_audio_features(
        self, audio_values: torch.Tensor
    ) -> torch.Tensor:
        """Run MedASR encoder + AudioProjector on audio input features.

        Args:
            audio_values: [B, n_mels, T_mel] mel spectrogram from MedASR processor.

        Returns:
            [B, mm_tokens_per_audio, llm_dim] projected audio embeddings.
        """
        with torch.no_grad():
            encoder_out = self._audio_encoder(input_features=audio_values)
            if hasattr(encoder_out, "last_hidden_state"):
                enc_embs = encoder_out.last_hidden_state
            else:
                enc_embs = encoder_out[0]

        projected = self.audio_projector(enc_embs)
        return projected

    def get_audio_features_from_precomputed(
        self, precomputed_embs: torch.Tensor
    ) -> torch.Tensor:
        """Run AudioProjector on pre-computed encoder embeddings.

        Args:
            precomputed_embs: [B, T_enc, 512] pre-computed MedASR encoder output.

        Returns:
            [B, mm_tokens_per_audio, llm_dim] projected audio embeddings.
        """
        return self.audio_projector(precomputed_embs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        precomputed_audio_embs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        """Forward pass with optional audio input.

        Handles audio using the same masked_scatter pattern as vision:
        1. Embed input_ids (with audio placeholder tokens getting dummy embeds)
        2. Compute audio features via encoder + projector
        3. Replace placeholder embeddings with projected audio features
        4. Pass unified embedding sequence to the language model

        Audio can be provided as:
          - audio_values: raw mel spectrogram (runs encoder + projector)
          - precomputed_audio_embs: pre-computed encoder output (runs projector only)
        """
        base = self._base_model
        model = base.model  # Gemma3Model

        vocab_size = model.vocab_size
        if input_ids is not None and self.audio_token_id >= vocab_size:
            special_audio_mask = input_ids == self.audio_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_audio_mask] = 0
        else:
            llm_input_ids = input_ids
            special_audio_mask = (
                (input_ids == self.audio_token_id) if input_ids is not None else None
            )

        image_token_id = getattr(base.config, "image_token_index", 262144)
        if input_ids is not None and image_token_id >= vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = llm_input_ids.clone()
            llm_input_ids[special_image_mask] = 0

        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None:
            image_features = model.get_image_features(pixel_values)
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            img_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                img_mask, image_features
            )

        has_audio = (audio_values is not None) or (precomputed_audio_embs is not None)
        if has_audio:
            if precomputed_audio_embs is not None:
                audio_features = self.get_audio_features_from_precomputed(
                    precomputed_audio_embs
                )
            else:
                audio_features = self.get_audio_features(audio_values)

            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            if special_audio_mask is not None:
                n_audio_tokens = special_audio_mask.sum().item()
                n_audio_features = audio_features.shape[0] * audio_features.shape[1]
                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio placeholder count ({n_audio_tokens}) != "
                        f"audio feature count ({n_audio_features}). "
                        f"Expected {self.mm_tokens_per_audio} placeholders per audio clip."
                    )
                audio_mask_expanded = special_audio_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask_expanded, audio_features
                )

        if token_type_ids is not None and has_audio:
            is_audio = (token_type_ids == 2)
            if is_audio.any():
                token_type_ids = token_type_ids.clone()

        return base(
            input_ids=None,
            pixel_values=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        precomputed_audio_embs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        """Generate with optional audio input.

        Builds the inputs_embeds with audio features injected, then delegates
        to the base model's generate method.
        """
        base = self._base_model
        model = base.model

        vocab_size = model.vocab_size
        llm_input_ids = input_ids
        if input_ids is not None:
            llm_input_ids = input_ids.clone()
            if self.audio_token_id >= vocab_size:
                llm_input_ids[input_ids == self.audio_token_id] = 0
            image_token_id = getattr(base.config, "image_token_index", 262144)
            if image_token_id >= vocab_size:
                llm_input_ids[input_ids == image_token_id] = 0

        inputs_embeds = model.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None:
            image_features = model.get_image_features(pixel_values)
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            img_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(img_mask, image_features)

        has_audio = (audio_values is not None) or (precomputed_audio_embs is not None)
        if has_audio:
            if precomputed_audio_embs is not None:
                audio_features = self.get_audio_features_from_precomputed(
                    precomputed_audio_embs
                )
            else:
                audio_features = self.get_audio_features(audio_values)

            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_audio_mask = (input_ids == self.audio_token_id)
            audio_mask_expanded = special_audio_mask.unsqueeze(-1).expand_as(
                inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask_expanded, audio_features
            )

        return base.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **generate_kwargs,
        )


Gemma3WithAudioForConditionalGeneration = Gemma3WithAudioModel
