"""Turn paragraphs + static facts into LogicBlock records."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..ingest.cleaner import read_cobol_file
from ..parser.copybooks import find_copybooks
from ..parser.paragraphs import Paragraph, split_paragraphs
from ..schemas import LogicBlock
from ..static_analysis.extract import extract_facts


def _block_id(source_file: str, paragraph: str, start: int) -> str:
    h = hashlib.sha1(f"{source_file}:{paragraph}:{start}".encode()).hexdigest()[:10]
    return f"lb_{h}"


def block_from_paragraph(source_file: str, paragraph: Paragraph, copybooks: list[str]) -> LogicBlock:
    facts = extract_facts(paragraph.code)
    return LogicBlock(
        id=_block_id(source_file, paragraph.name, paragraph.start_line),
        source_file=source_file,
        paragraph=paragraph.name,
        code=paragraph.code,
        start_line=paragraph.start_line,
        end_line=paragraph.end_line,
        vars_read=facts.vars_read,
        vars_written=facts.vars_written,
        conditions=facts.conditions,
        perform_calls=facts.perform_calls,
        file_refs=facts.file_refs,
        copybooks=copybooks,
    )


def segment_file(path: Path) -> list[LogicBlock]:
    cleaned = read_cobol_file(path)
    paragraphs = split_paragraphs(cleaned)
    copybooks = find_copybooks(cleaned)
    rel = str(path)
    return [block_from_paragraph(rel, p, copybooks) for p in paragraphs]
