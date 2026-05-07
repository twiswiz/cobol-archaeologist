"""Robust paragraph extractor.

Lark grammars over real-world COBOL are flaky, so we provide a regex-based
paragraph splitter that operates on cleaned source. This always succeeds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Paragraph header: a single label followed by a period at the start of a line.
# Allowed inside PROCEDURE DIVISION only.
_PARA_HEADER = re.compile(r"^\s*([A-Z0-9][A-Z0-9-]*)\s*\.\s*$", re.IGNORECASE)
_DIV_HEADER = re.compile(r"^\s*(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION\b", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^\s*([A-Z0-9][A-Z0-9-]*)\s+SECTION\s*\.", re.IGNORECASE)

# Reserved words that should never be treated as paragraph names even if they
# appear before a period at the start of a line.
_RESERVED = {
    "END-IF", "END-PERFORM", "END-EVALUATE", "END-READ", "END-WRITE",
    "END-CALL", "END-SEARCH", "END-COMPUTE", "END-ADD", "END-SUBTRACT",
    "ELSE", "EXIT", "STOP", "GOBACK", "CONTINUE", "PROGRAM-ID",
    "AUTHOR", "INSTALLATION", "DATE-WRITTEN", "DATE-COMPILED", "REMARKS",
    "SECURITY", "ENVIRONMENT", "DATA", "PROCEDURE", "IDENTIFICATION",
    "WORKING-STORAGE", "FILE", "LINKAGE", "LOCAL-STORAGE", "INPUT-OUTPUT",
    "CONFIGURATION", "SOURCE-COMPUTER", "OBJECT-COMPUTER", "FILE-CONTROL",
}


@dataclass
class Paragraph:
    name: str
    section: str | None
    code: str
    start_line: int
    end_line: int


def split_paragraphs(source: str) -> list[Paragraph]:
    """Split cleaned COBOL source into procedure-division paragraphs."""
    lines = source.splitlines()
    proc_start = None
    for i, line in enumerate(lines):
        m = _DIV_HEADER.match(line)
        if m and m.group(1).upper() == "PROCEDURE":
            proc_start = i + 1
            break
    if proc_start is None:
        return []

    paragraphs: list[Paragraph] = []
    current: Paragraph | None = None
    section: str | None = None

    def flush(end_line: int) -> None:
        nonlocal current
        if current is not None:
            current.end_line = end_line
            current.code = "\n".join(lines[current.start_line - 1 : current.end_line])
            paragraphs.append(current)
            current = None

    for idx in range(proc_start, len(lines)):
        line = lines[idx]
        line_no = idx + 1
        stripped = line.strip()
        if not stripped:
            continue

        sm = _SECTION_HEADER.match(line)
        if sm:
            flush(idx)
            section = sm.group(1).upper()
            continue

        pm = _PARA_HEADER.match(line)
        if pm and pm.group(1).upper() not in _RESERVED:
            flush(idx)
            current = Paragraph(
                name=pm.group(1).upper(),
                section=section,
                code="",
                start_line=line_no,
                end_line=line_no,
            )
            continue

        if current is None:
            # Lines before any paragraph header (rare) are dropped.
            continue

    flush(len(lines))
    return paragraphs
