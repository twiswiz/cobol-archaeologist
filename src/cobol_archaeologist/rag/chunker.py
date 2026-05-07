"""Chunk regulation pages into overlapping windows for retrieval."""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from ..schemas import RegulationChunk
from .pdf_loader import PdfPage

_SECTION_RE = re.compile(r"^\s*(?:Section|Chapter|Article|Para(?:graph)?)\s+([A-Z0-9.\-]+)", re.I | re.M)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Word-window chunker. ``chunk_size`` and ``overlap`` are in *words*."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _detect_section(text: str) -> str | None:
    m = _SECTION_RE.search(text)
    return m.group(0).strip() if m else None


def _chunk_id(source: str, page: int, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{page}:{idx}:{text[:40]}".encode()).hexdigest()[:10]
    return f"reg_{h}"


def chunk_pages(pages: Iterable[PdfPage], chunk_size: int = 800, overlap: int = 100) -> list[RegulationChunk]:
    out: list[RegulationChunk] = []
    for page in pages:
        section = _detect_section(page.text)
        for i, c in enumerate(_chunk_text(page.text, chunk_size=chunk_size, overlap=overlap)):
            out.append(
                RegulationChunk(
                    id=_chunk_id(page.source, page.page, i, c),
                    source=page.source,
                    section=section,
                    page=page.page,
                    text=c,
                )
            )
    return out
