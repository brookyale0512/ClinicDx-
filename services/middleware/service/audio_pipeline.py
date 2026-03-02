"""Direct audio-to-concept pipeline: bypass text intermediate.

Flow: Audio -> MedASR Encoder (512-dim) -> AudioProjector -> MedGemma (2560-dim) -> JSON

This module connects the MedASR Conformer encoder directly to MedGemma
via a learned MLP projector, avoiding the lossy text bottleneck.

Training/inference sequence layout (must match train_projector.py):
  [system_prompt + manifest] [<audio_start> audio <audio_end>] [output]
"""

import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .projector import AudioProjector

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"
MEDGEMMA_DIR = Path(__file__).parent.parent / "medgemma"
SAMPLE_RATE = 16000

SYSTEM_PROMPT = (
    "You are a medical concept extractor for an OpenMRS clinic in Africa.\n"
    "Audio embeddings from a clinical recording are provided between "
    "<audio_start> and <audio_end> markers.\n"
    "Extract structured medical observations from the audio.\n"
    "Return ONLY key: value lines matching concepts from the manifest.\n\n"
)


class DirectAudioPipeline:
    """End-to-end audio-to-concept extraction without text intermediate.

    Components (loaded in load()):
      1. MedASR encoder (frozen) — extracts 512-dim audio embeddings
      2. AudioProjector (trained) — projects to 2560-dim MedGemma space
         with <audio_start>/<audio_end> delimiters
      3. MedGemma (frozen, 4-bit) — generates structured key:value output
    """

    def __init__(
        self,
        encoder_path: str = str(MODEL_DIR),
        medgemma_path: str = str(MEDGEMMA_DIR),
        projector_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.encoder_path = encoder_path
        self.medgemma_path = medgemma_path
        self.projector_checkpoint = projector_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder = None
        self._processor = None
        self._projector = None
        self._llm = None
        self._tokenizer = None
        self._embed_layer = None

    def load(self) -> None:
        """Load all pipeline components."""
        self._load_encoder()
        self._load_projector()
        self._load_llm()

    def _load_encoder(self) -> None:
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading MedASR encoder from %s", self.encoder_path)
        model = AutoModel.from_pretrained(
            self.encoder_path, torch_dtype=torch.float32,
        ).to(self.device)
        model.eval()

        self._encoder = model.encoder if hasattr(model, "encoder") else model
        for param in self._encoder.parameters():
            param.requires_grad = False

        self._processor = AutoProcessor.from_pretrained(self.encoder_path)
        logger.info("MedASR encoder loaded (frozen)")

    def _load_projector(self) -> None:
        self._projector = AudioProjector(
            encoder_dim=512, llm_dim=2560, stack_factor=4,
        ).to(self.device)

        if self.projector_checkpoint and Path(self.projector_checkpoint).exists():
            state_dict = torch.load(
                self.projector_checkpoint,
                map_location=self.device, weights_only=True,
            )
            self._projector.load_state_dict(state_dict)
            logger.info("Projector weights loaded from %s", self.projector_checkpoint)
        else:
            logger.warning("No projector checkpoint — using random weights (untrained)")

        logger.info("AudioProjector: %d params", self._projector.param_count())

    def _load_llm(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading MedGemma from %s (full precision)", self.medgemma_path)

        self._tokenizer = AutoTokenizer.from_pretrained(self.medgemma_path)

        self._llm = AutoModelForCausalLM.from_pretrained(
            self.medgemma_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._llm.eval()
        for param in self._llm.parameters():
            param.requires_grad = False

        self._embed_layer = self._llm.get_input_embeddings()
        logger.info("Embed layer: %s", type(self._embed_layer).__name__)
        logger.info("MedGemma loaded (full precision, all frozen)")

    def extract_encoder_embeddings(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> torch.Tensor:
        """Extract encoder embeddings from audio waveform.

        Returns: Tensor of shape [1, seq_len, 512].
        """
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        inputs = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)

        with torch.no_grad():
            encoder_out = self._encoder(input_features=input_features)

        if hasattr(encoder_out, "last_hidden_state"):
            return encoder_out.last_hidden_state
        return encoder_out[0]

    @torch.inference_mode()
    def extract(
        self,
        audio: np.ndarray,
        manifest_string: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict:
        """Extract structured observations from audio with manifest context.

        Args:
            audio: Audio waveform (mono, float32).
            manifest_string: The concept manifest for this encounter.
            sample_rate: Sample rate of audio.

        Returns:
            dict with "observations", "cds_alerts", and per-item "confidence".
        """
        # Step 1: Audio → encoder → projector (includes delimiters)
        encoder_embs = self.extract_encoder_embeddings(audio, sample_rate)
        projected = self._projector(encoder_embs)  # [1, T/4+2, 2560]

        # Step 2: Build prompt — matches training layout exactly
        prompt_text = SYSTEM_PROMPT + manifest_string + "\n\nOUTPUT:\n"
        prompt_tokens = self._tokenizer.encode(
            prompt_text, return_tensors="pt", add_special_tokens=True
        ).to(self._llm.device)

        prompt_embeds = self._embed_layer(prompt_tokens)  # [1, P, 2560]

        projected_cast = projected.to(
            dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )

        # Concat: [prompt+manifest | <audio_start> audio <audio_end>]
        inputs_embeds = torch.cat([prompt_embeds, projected_cast], dim=1)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device,
        )

        # Step 3: Generate
        output_ids = self._llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )

        response_text = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        # Step 4: Parse + compute confidence per item
        result = self._parse_response(response_text)
        result["confidence_scores"] = self._compute_confidence(
            projected_cast, result.get("observations", [])
        )

        return result

    def _compute_confidence(
        self, projected: torch.Tensor, observations: list
    ) -> dict:
        """Compute cosine similarity confidence for each extracted concept.

        Uses mean-pooled audio embedding (excluding delimiters) vs concept
        key text embedding. The pool covers the entire clip, so for
        multi-concept clips the score is noisier — the embedding for
        "malaria: absent" also contains signal from other concepts in the
        same clip. Scores are most reliable for single-concept utterances.
        Frontend should treat low confidence as "needs review", not "wrong".
        """
        scores = {}
        if not observations:
            return scores

        audio_only = projected[0, 1:-1, :]  # strip delimiters
        audio_mean = audio_only.mean(dim=0).float()  # [2560]

        for obs in observations:
            label = obs.get("label", "")
            if not label:
                continue

            concept_key = label.lower().replace(" ", "_")
            concept_ids = self._tokenizer.encode(
                concept_key, add_special_tokens=False, return_tensors="pt"
            ).to(self._llm.device)
            concept_embeds = self._embed_layer(concept_ids)  # [1, N, 2560]
            concept_mean = concept_embeds.mean(dim=1).squeeze(0).float()

            cos_sim = F.cosine_similarity(
                audio_mean.unsqueeze(0), concept_mean.unsqueeze(0)
            ).item()

            scores[label] = round(max(0.0, min(1.0, (cos_sim + 1) / 2)), 3)

        return scores

    def extract_bytes(self, audio_bytes: bytes, manifest_string: str = "") -> dict:
        """Extract observations from raw audio bytes."""
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        return self.extract(audio, manifest_string=manifest_string, sample_rate=SAMPLE_RATE)

    def _parse_response(self, response_text: str) -> dict:
        """Parse key:value lines or JSON from LLM response."""
        observations = []

        for line in response_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if not key or not value:
                continue

            if key == "NOT_IN_MANIFEST":
                observations.append({
                    "label": value,
                    "value": "NOT_IN_MANIFEST",
                    "not_in_manifest": True,
                })
            elif value == "NONE":
                continue
            else:
                observations.append({
                    "label": key,
                    "value": value,
                })

        return {
            "observations": observations,
            "cds_alerts": [],
            "raw_output": response_text,
        }

    @property
    def is_loaded(self) -> bool:
        return all([
            self._encoder is not None,
            self._projector is not None,
            self._llm is not None,
            self._tokenizer is not None,
        ])
