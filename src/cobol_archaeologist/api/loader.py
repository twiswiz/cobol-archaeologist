"""Load and cache JSONL data files at startup."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from cobol_archaeologist.schemas import BusinessIntentCard, LogicBlock

# Default paths — override via env vars
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
BLOCKS_FILES = [
    DATA_DIR / "processed" / "logic_blocks.jsonl",
    DATA_DIR / "processed" / "carddemo_blocks.jsonl",
    DATA_DIR / "generated" / "generated_cobol_logic_blocks.jsonl",
]
CARDS_FILES = [
    DATA_DIR / "reports" / "cbsa_cards.jsonl",
    DATA_DIR / "reports" / "carddemo_cards.jsonl",
    DATA_DIR / "reports" / "cards.jsonl",
]
INDEX_DIR = DATA_DIR / "index"


def _load_jsonl(paths: list[Path], model: type) -> list:
    records = []
    for path in paths:
        if not path.exists():
            # also check reports/ at top level
            alt = Path("reports") / path.name
            if alt.exists():
                path = alt
            else:
                continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(model.model_validate_json(line))
                    except Exception:
                        pass
    return records


@lru_cache(maxsize=1)
def get_blocks() -> list[LogicBlock]:
    return _load_jsonl(BLOCKS_FILES, LogicBlock)


@lru_cache(maxsize=1)
def get_cards() -> list[BusinessIntentCard]:
    cards = _load_jsonl(CARDS_FILES, BusinessIntentCard)
    return cards


@lru_cache(maxsize=1)
def get_cards_by_block_id() -> dict[str, BusinessIntentCard]:
    return {c.logic_block_id: c for c in get_cards() if c.logic_block_id}


def get_block_by_id(block_id: str) -> Optional[LogicBlock]:
    for b in get_blocks():
        if b.id == block_id:
            return b
    return None
