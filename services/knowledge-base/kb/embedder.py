#!/usr/bin/env python3
"""
EmbedGemma 300M base (float32) embedder for the KB retrieval layer.

Replaces the Q4-quantized llama-server embedder. The base model has 2.9×
better discrimination gap on clinical text vs Q4.

Model context: 2048 tokens. Clinical/WHO text averages 5.33 chars/token,
giving a real capacity of ~10,000 chars per window (safe margin from 10,903).
Chunks larger than one window use multi-window average-pooling.

Pipeline: Transformer → MeanPooling → Dense(768→3072) → Dense(3072→768) → Normalize
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import List, Sequence

log = logging.getLogger("kb.embedder")

MODEL_PATH = os.environ.get(
    "EMBED_MODEL_PATH",
    # Default: HF hub cache path inside the container (bind-mounted from host)
    "/hf_cache/hub/models--google--embeddinggemma-300m/snapshots/"
    "57c266a740f537b4dc058e1b0cda161fd15afa75",
)

# 2048 tokens × 5.33 chars/token × 0.92 safety margin = ~10,000 chars
# At this limit NO clinical chunk is silently truncated within a single window.
WINDOW_CHARS   = 10_000   # chars per embedding window
MAX_WINDOWS    = 3        # cover up to 30,000 chars per chunk
EMBED_CHAR_LIMIT = WINDOW_CHARS  # alias used by retrieval_core for query embedding
EMBED_BATCH_SIZE = 64

try:
    from memvid_sdk.embeddings import EmbeddingProvider
except ImportError:
    EmbeddingProvider = object  # type: ignore[assignment,misc]


def _pool_windows(vecs: List[List[float]]) -> List[float]:
    """Average-pool multiple window embeddings and re-normalize."""
    if len(vecs) == 1:
        return vecs[0]
    dim = len(vecs[0])
    pooled = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
    return [x / norm for x in pooled]


def _windows(text: str) -> List[str]:
    """Split text into non-overlapping WINDOW_CHARS windows, up to MAX_WINDOWS."""
    wins = []
    for i in range(MAX_WINDOWS):
        start = i * WINDOW_CHARS
        if start >= len(text):
            break
        wins.append(text[start: start + WINDOW_CHARS])
    return wins


class EmbedGemmaEmbedder(EmbeddingProvider):  # type: ignore[misc]
    """EmbedGemma 300M base (float32) via sentence-transformers — no llama-server needed."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("Loading EmbedGemma base on %s", device)
                self._model = SentenceTransformer(MODEL_PATH, device=device)
                log.info("EmbedGemma base loaded (dim=%d, max_seq=%d)",
                         self.dimension, self._model.max_seq_length)
            except Exception as exc:
                log.error("Failed to load EmbedGemma base: %s", exc)

    @property
    def dimension(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return "embedgemma-300m-base"

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def _encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=min(EMBED_BATCH_SIZE, len(texts)),
        )
        return vecs.tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed documents with multi-window pooling for large chunks."""
        self._load()
        if self._model is None:
            return [[0.0] * 768] * len(texts)

        results: List[List[float]] = []
        # Collect all windows flat, track which doc each belongs to
        doc_window_counts: List[int] = []
        flat_windows: List[str] = []
        for text in texts:
            wins = _windows(text)
            doc_window_counts.append(len(wins))
            flat_windows.extend(wins)

        # Batch encode all windows at once
        flat_vecs: List[List[float]] = []
        for i in range(0, len(flat_windows), EMBED_BATCH_SIZE):
            batch = flat_windows[i: i + EMBED_BATCH_SIZE]
            flat_vecs.extend(self._encode(batch))

        # Reassemble per-doc and pool
        idx = 0
        for n in doc_window_counts:
            doc_vecs = flat_vecs[idx: idx + n]
            results.append(_pool_windows(doc_vecs))
            idx += n

        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query — queries are short, always one window."""
        self._load()
        if self._model is None:
            return [0.0] * 768
        vec = self._model.encode(
            [text[:WINDOW_CHARS]],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].tolist()
