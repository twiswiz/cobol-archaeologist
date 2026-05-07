"""Parse and validate model output into a BusinessIntentCard."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..schemas import BusinessIntentCard


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class CardParseError(ValueError):
    """Raised when the model output cannot be parsed as a Business Intent Card."""


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = _JSON_BLOCK.search(text)
    if not m:
        raise CardParseError("No JSON object found in model output.")
    return m.group(0)


def parse_card(text: str, logic_block_id: str | None = None) -> BusinessIntentCard:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CardParseError(f"Invalid JSON: {exc}") from exc
    if logic_block_id is not None:
        data.setdefault("logic_block_id", logic_block_id)
    try:
        return BusinessIntentCard.model_validate(data)
    except ValidationError as exc:
        raise CardParseError(str(exc)) from exc
