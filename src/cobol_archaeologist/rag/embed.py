"""Embed regulation chunks. Falls back to a deterministic hashing embedder
when sentence-transformers is unavailable, so tests run without heavy deps.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

from ..schemas import RegulationChunk


class HashingEmbedder:
    """Cheap deterministic text embedder for tests / offline runs."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


class STEmbedder:
    """sentence-transformers wrapper (optional)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=normalize, convert_to_numpy=True)
        return vecs.astype(np.float32)


def get_embedder(prefer_st: bool = True, model_name: str = "BAAI/bge-small-en-v1.5"):
    if prefer_st:
        try:
            return STEmbedder(model_name=model_name)
        except Exception:
            pass
    return HashingEmbedder()


def embed_chunks(chunks: Iterable[RegulationChunk], embedder=None) -> np.ndarray:
    chunks = list(chunks)
    if embedder is None:
        embedder = get_embedder()
    return embedder.encode([c.text for c in chunks])
