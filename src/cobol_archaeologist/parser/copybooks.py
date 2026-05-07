"""Detect COPY references inside a COBOL program."""
from __future__ import annotations

import re

_COPY = re.compile(r"\bCOPY\s+([A-Z0-9][A-Z0-9-]*)\b", re.IGNORECASE)


def find_copybooks(source: str) -> list[str]:
    seen: list[str] = []
    for m in _COPY.finditer(source):
        name = m.group(1).upper()
        if name not in seen:
            seen.append(name)
    return seen
