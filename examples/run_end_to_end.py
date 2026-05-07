"""Run the full pipeline against the bundled fixture and print a Business Intent Card.

Usage::

    python -m examples.run_end_to_end
"""
from __future__ import annotations

import json
from pathlib import Path

from cobol_archaeologist.labels.weak import label_blocks
from cobol_archaeologist.model.backend import EchoBackend
from cobol_archaeologist.model.runner import generate_card
from cobol_archaeologist.segmenter.segment import segment_file


def main() -> None:
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample.cbl"
    blocks = label_blocks(segment_file(fixture))
    backend = EchoBackend()
    print(f"Found {len(blocks)} logic blocks in {fixture.name}\n")
    for b in blocks:
        card = generate_card(b, backend)
        print(f"== {b.paragraph} (label={b.weak_label}) ==")
        print(json.dumps(card.model_dump(), indent=2))
        print()


if __name__ == "__main__":
    main()
