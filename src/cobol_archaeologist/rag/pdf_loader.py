"""Extract text from regulatory PDFs (RBI KYC, Basel III)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfPage:
    source: str
    page: int
    text: str


def load_pdf(path: Path, source_label: str | None = None) -> list[PdfPage]:
    from pypdf import PdfReader  # local import to keep base deps light

    reader = PdfReader(str(path))
    label = source_label or path.stem
    pages: list[PdfPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(PdfPage(source=label, page=i, text=text))
    return pages
