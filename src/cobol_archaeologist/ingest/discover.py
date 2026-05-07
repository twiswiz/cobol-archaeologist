"""Walk extracted dataset folders and produce a CSV manifest of COBOL sources."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

COBOL_EXTS = {".cbl", ".cob", ".cobol", ".cpy", ".pco"}


def iter_cobol_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in COBOL_EXTS:
            yield p


def discover(root: Path, out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "rel_path", "size_bytes", "lines", "is_copybook"])
        for path in iter_cobol_files(root):
            try:
                size = path.stat().st_size
                lines = sum(1 for _ in path.open("rb"))
            except OSError:
                continue
            writer.writerow(
                [
                    str(path),
                    str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                    size,
                    lines,
                    path.suffix.lower() == ".cpy",
                ]
            )
            n += 1
    return n
