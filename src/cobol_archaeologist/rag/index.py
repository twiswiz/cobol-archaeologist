"""FAISS-backed vector index with a numpy fallback.

The numpy fallback keeps the whole pipeline runnable without faiss-cpu installed,
which matters for CI on Windows where faiss wheels are flaky.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..schemas import RegulationChunk


@dataclass
class SearchHit:
    chunk: RegulationChunk
    score: float


class _NumpyIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[RegulationChunk]):
        self.vectors = vectors
        self.chunks = chunks

    def search(self, query: np.ndarray, k: int) -> list[SearchHit]:
        scores = (self.vectors @ query.T).ravel()
        top = np.argsort(-scores)[:k]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in top]


class RegulationIndex:
    def __init__(self, vectors: np.ndarray, chunks: Sequence[RegulationChunk]):
        import os
        self._chunks = list(chunks)
        self._faiss = None
        # faiss-cpu is opt-in: it can segfault on small Linux containers (e.g. Render
        # free tier) due to AVX/numpy ABI mismatches. With <10k chunks the numpy
        # fallback is fast enough.
        if os.getenv("ENABLE_FAISS", "").lower() not in ("1", "true", "yes"):
            self._numpy = _NumpyIndex(vectors, self._chunks)
            return
        try:
            import faiss  # type: ignore

            idx = faiss.IndexFlatIP(vectors.shape[1])
            idx.add(vectors)
            self._faiss = idx
            self._dim = vectors.shape[1]
        except Exception:
            self._numpy = _NumpyIndex(vectors, self._chunks)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[SearchHit]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)
            return [
                SearchHit(chunk=self._chunks[i], score=float(s))
                for s, i in zip(scores[0], idxs[0])
                if i >= 0
            ]
        return self._numpy.search(q, k)

    def save(self, directory: Path, vectors: np.ndarray) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(c.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path) -> "RegulationIndex":
        vectors = np.load(directory / "vectors.npy")
        chunks = []
        with (directory / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(RegulationChunk.model_validate(json.loads(line)))
        return cls(vectors, chunks)
