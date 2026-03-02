"""Extended Gemma3 processor with audio support.

Handles audio inputs alongside text and images in chat templates.
Mirrors how Gemma3Processor handles images:
  1. Chat template inserts <start_of_audio> marker
  2. Processor expands it to <start_of_audio><audio_soft_token>x64<end_of_audio>
  3. Audio waveform is processed into mel spectrogram features
  4. token_type_ids marks audio positions with 2 (text=0, image=1, audio=2)

Usage:
    processor = Gemma3AudioProcessor.from_model_dir("scribe_version/model", "scribe_version/medASR")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "[SCRIBE]\\n<manifest>...</manifest>"},
        {"type": "audio", "audio": audio_array},
    ]}]
    inputs = processor(messages=messages, return_tensors="pt")
    # inputs has: input_ids, attention_mask, token_type_ids, audio_values
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Gemma3AudioProcessor:
    """Processor that handles text, images, and audio for Gemma3WithAudioModel.

    Audio processing:
      - Accepts raw waveforms (numpy float32 at 16kHz)
      - Converts to mel spectrogram via MedASR's LASRFeatureExtractor
      - Inserts audio placeholder tokens into the text sequence
      - Sets token_type_ids=2 for audio positions
    """

    def __init__(
        self,
        tokenizer,
        audio_processor,
        audio_seq_length: int = 64,
        image_seq_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.audio_seq_length = audio_seq_length
        self.image_seq_length = image_seq_length

        self.boa_token = "<start_of_audio>"
        self.eoa_token = "<end_of_audio>"
        self.audio_token = "<audio_soft_token>"

        self.boa_token_id = tokenizer.convert_tokens_to_ids(self.boa_token)
        self.eoa_token_id = tokenizer.convert_tokens_to_ids(self.eoa_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

        self.boi_token = getattr(tokenizer, "boi_token", "<start_of_image>")
        self.eoi_token = getattr(tokenizer, "eoi_token", "<end_of_image>")
        self.image_token = getattr(tokenizer, "image_token", "<image_soft_token>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        audio_tokens = self.audio_token * audio_seq_length
        self.full_audio_sequence = (
            f"\n\n{self.boa_token}{audio_tokens}{self.eoa_token}\n\n"
        )

        image_tokens = self.image_token * image_seq_length
        self.full_image_sequence = (
            f"\n\n{self.boi_token}{image_tokens}{self.eoi_token}\n\n"
        )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        audio_encoder_dir: str,
        audio_seq_length: int = 64,
        image_seq_length: int = 256,
    ) -> "Gemma3AudioProcessor":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        preproc_path = Path(audio_encoder_dir) / "preprocessor_config.json"
        if preproc_path.exists():
            with open(preproc_path) as f:
                pcfg = json.load(f)
        else:
            pcfg = {}

        audio_processor = WhisperFeatureExtractor(
            feature_size=pcfg.get("feature_size", 128),
            sampling_rate=pcfg.get("sampling_rate", 16000),
            hop_length=pcfg.get("hop_length", 160),
            n_fft=pcfg.get("n_fft", 512),
            win_length=pcfg.get("win_length", 400),
        )

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            audio_seq_length=audio_seq_length,
            image_seq_length=image_seq_length,
        )

    def process_audio_waveform(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram features.

        Args:
            audio: float32 waveform array.
            sample_rate: input sample rate (resampled to 16kHz if different).

        Returns:
            input_features tensor [1, n_mels, T_mel] for MedASR encoder.
        """
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE,
            )
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        features = self.audio_processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        return features["input_features"]

    def __call__(
        self,
        messages: list[dict],
        audio: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        audio_sample_rate: int = SAMPLE_RATE,
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
    ) -> dict:
        """Process messages with optional audio into model inputs.

        Args:
            messages: Chat messages in OpenAI format. Audio content items
                should have type="audio" with an "audio" key containing
                the waveform array, or pass audio separately.
            audio: Optional audio waveform(s) to use for <start_of_audio> markers.
            audio_sample_rate: Sample rate of provided audio.
            add_generation_prompt: Whether to add <start_of_turn>model prompt.
            return_tensors: Tensor format ("pt" for PyTorch).

        Returns:
            Dict with input_ids, attention_mask, token_type_ids,
            and optionally audio_values.
        """
        audio_waveforms = []
        processed_messages = []

        for msg in messages:
            if isinstance(msg.get("content"), str):
                processed_messages.append(msg)
                continue

            new_content = []
            for item in msg["content"]:
                if item.get("type") == "audio":
                    waveform = item.get("audio")
                    if waveform is not None:
                        audio_waveforms.append(waveform)
                    new_content.append({"type": "audio"})
                else:
                    new_content.append(item)
            processed_messages.append({**msg, "content": new_content})

        if audio is not None:
            if isinstance(audio, np.ndarray):
                audio_waveforms = [audio]
            else:
                audio_waveforms = list(audio)

        text = self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        text = text.replace(self.boa_token, self.full_audio_sequence)
        text = text.replace(self.boi_token, self.full_image_sequence)

        encoding = self.tokenizer(
            text, return_tensors=return_tensors, add_special_tokens=False,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == self.audio_token_id] = 2
        token_type_ids[input_ids == self.image_token_id] = 1

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if audio_waveforms:
            audio_features_list = []
            for waveform in audio_waveforms:
                features = self.process_audio_waveform(
                    waveform, sample_rate=audio_sample_rate,
                )
                audio_features_list.append(features)
            result["audio_values"] = torch.cat(audio_features_list, dim=0)

        return result

    def decode(self, token_ids, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids, **kwargs) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, **kwargs)
