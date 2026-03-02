"""MedASR transcription wrapper.

Wraps the MedASR Conformer CTC model for medical speech-to-text.
Supports both greedy decoding and beam search with KenLM language model.
"""

import dataclasses
import io
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"
SAMPLE_RATE = 16000


class MedASRTranscriber:
    """Wraps MedASR for medical speech-to-text transcription."""

    def __init__(
        self,
        model_path: str = str(MODEL_DIR),
        use_lm: bool = True,
        device: Optional[str] = None,
        chunk_length_s: float = 20.0,
        stride_length_s: float = 2.0,
    ):
        self.model_path = model_path
        self.use_lm = use_lm
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._pipe = None
        self._pipe_with_lm = None

    def load(self) -> None:
        """Load model and processor. Call once at startup."""
        import transformers

        logger.info("Loading MedASR model from %s on %s", self.model_path, self.device)

        # Greedy decoding pipeline
        self._pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=self.model_path,
            device=self.device,
        )

        # Beam search with KenLM (better accuracy)
        if self.use_lm:
            lm_path = str(Path(self.model_path) / "lm_6.kenlm")
            if Path(lm_path).exists():
                try:
                    self._pipe_with_lm = self._build_beam_search_pipe(lm_path)
                    logger.info("KenLM language model loaded for beam search decoding")
                except ImportError as e:
                    logger.warning("pyctcdecode not available, using greedy decoding: %s", e)
                except Exception as e:
                    logger.warning("Failed to build beam search pipeline, using greedy: %s", e)
            else:
                logger.warning("KenLM model not found at %s, using greedy decoding", lm_path)

        logger.info("MedASR model loaded successfully")

    def _build_beam_search_pipe(self, lm_path: str):
        """Build beam search pipeline with KenLM language model."""
        import pyctcdecode
        import transformers

        def _restore_text(text: str) -> str:
            return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()

        class LasrCtcBeamSearchDecoder:
            def __init__(self, tokenizer, kenlm_model_path=None, **kwargs):
                vocab = [None for _ in range(tokenizer.vocab_size)]
                for k, v in tokenizer.vocab.items():
                    if v < tokenizer.vocab_size:
                        vocab[v] = k

                for i in range(len(vocab)):
                    if vocab[i] is None:
                        vocab[i] = f"<MISSING_{i}>"
                    piece = vocab[i]
                    if not piece.startswith("<") and not piece.endswith(">"):
                        piece = "\u2581" + piece.replace("\u2581", "#")
                    vocab[i] = piece

                self._decoder = pyctcdecode.build_ctcdecoder(
                    vocab, kenlm_model_path, **kwargs
                )

            def decode_beams(self, *args, **kwargs):
                beams = self._decoder.decode_beams(*args, **kwargs)
                return [
                    dataclasses.replace(i, text=_restore_text(i.text)) for i in beams
                ]

        feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(
            self.model_path
        )
        feature_extractor._processor_class = "LasrProcessorWithLM"

        pipe = transformers.pipeline(
            task="automatic-speech-recognition",
            model=self.model_path,
            feature_extractor=feature_extractor,
            decoder=LasrCtcBeamSearchDecoder(
                transformers.AutoTokenizer.from_pretrained(self.model_path), lm_path
            ),
            device=self.device,
        )
        assert pipe.type == "ctc_with_lm"
        return pipe

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, use_lm: Optional[bool] = None
    ) -> dict:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array (mono, float32).
            sample_rate: Sample rate of the audio. Will resample to 16kHz if different.
            use_lm: Override whether to use LM beam search. Defaults to instance setting.

        Returns:
            dict with keys: text (str), confidence (float)
        """
        if self._pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Choose pipeline
        should_use_lm = use_lm if use_lm is not None else self.use_lm
        pipe = self._pipe_with_lm if (should_use_lm and self._pipe_with_lm) else self._pipe

        result = pipe(
            audio,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
        )

        text = result.get("text", "").strip()

        return {"text": text}

    def transcribe_file(self, file_path: str, use_lm: Optional[bool] = None) -> dict:
        """Transcribe an audio file.

        Args:
            file_path: Path to audio file (wav, mp3, flac, etc.)
            use_lm: Override whether to use LM beam search.

        Returns:
            dict with keys: text (str), confidence (float)
        """
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return self.transcribe(audio, sample_rate=SAMPLE_RATE, use_lm=use_lm)

    def transcribe_bytes(self, audio_bytes: bytes, use_lm: Optional[bool] = None) -> dict:
        """Transcribe audio from raw bytes (wav/webm uploaded via API).

        Args:
            audio_bytes: Raw audio file bytes.
            use_lm: Override whether to use LM beam search.

        Returns:
            dict with keys: text (str), confidence (float)
        """
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        return self.transcribe(audio, sample_rate=SAMPLE_RATE, use_lm=use_lm)
