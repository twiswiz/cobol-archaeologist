"""Normalize COBOL source files: strip column 1-6 sequence numbers, handle col 7 indicators."""
from __future__ import annotations

from pathlib import Path


COMMENT_INDICATORS = {"*", "/"}
DEBUG_INDICATOR = "D"
CONTINUATION_INDICATOR = "-"


def clean_line(line: str, drop_comments: bool = False) -> str | None:
    """Clean a single fixed-format COBOL line.

    Returns None when the line should be dropped.
    """
    # Remove trailing line breaks but keep internal whitespace.
    raw = line.rstrip("\r\n")
    # Tabs -> spaces (8-col tab stops are conventional, but for cleaning a single space is fine).
    raw = raw.expandtabs(4)

    # Free-format heuristic: line shorter than 7 chars or no leading 6 spaces -> treat as free.
    if len(raw) < 7 or not raw[:6].isspace() and not raw[:6].isdigit() and not raw[:6].isalnum():
        # Even in free format, some files start with comment markers.
        stripped = raw.lstrip()
        if stripped.startswith(("*>", "*")) and drop_comments:
            return None
        return raw

    indicator = raw[6] if len(raw) > 6 else " "
    body = raw[7:] if len(raw) > 7 else ""

    if indicator in COMMENT_INDICATORS:
        if drop_comments:
            return None
        return "      *" + body  # keep but normalized
    if indicator == DEBUG_INDICATOR:
        # Keep as a comment-equivalent unless dropped.
        if drop_comments:
            return None
        return "      *" + body
    if indicator == CONTINUATION_INDICATOR:
        return "       " + body  # treat as plain continuation; downstream regex tolerates it
    return "       " + body


def clean_text(text: str, drop_comments: bool = False, uppercase: bool = False) -> str:
    out: list[str] = []
    for line in text.splitlines():
        cleaned = clean_line(line, drop_comments=drop_comments)
        if cleaned is None:
            continue
        out.append(cleaned.upper() if uppercase else cleaned)
    return "\n".join(out)


def read_cobol_file(path: Path, drop_comments: bool = False, uppercase: bool = False) -> str:
    """Read a COBOL source file with EBCDIC fallback."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            text = path.read_bytes().decode("cp037", errors="replace")  # EBCDIC US
    return clean_text(text, drop_comments=drop_comments, uppercase=uppercase)
