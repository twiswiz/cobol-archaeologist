"""End-to-end inference: LogicBlock + retrieved chunks -> BusinessIntentCard."""
from __future__ import annotations

from typing import Optional

from ..schemas import BusinessIntentCard, LogicBlock, RegulationChunk
from .backend import LLMBackend
from .parse_output import CardParseError, parse_card
from .prompt import render_prompt


REPAIR_INSTRUCTION = (
    "\nThe previous response was not valid JSON for a Business Intent Card. "
    "Re-emit ONLY a single JSON object with the required keys."
)


def generate_card(
    block: LogicBlock,
    backend: LLMBackend,
    retrieved: Optional[list[RegulationChunk]] = None,
    repair_attempts: int = 1,
) -> BusinessIntentCard:
    prompt = render_prompt(block, retrieved=retrieved)
    last_error: Exception | None = None
    for attempt in range(repair_attempts + 1):
        text = backend.generate(prompt if attempt == 0 else prompt + REPAIR_INSTRUCTION)
        try:
            return parse_card(text, logic_block_id=block.id)
        except CardParseError as exc:
            last_error = exc
            continue
    raise CardParseError(f"Model failed to produce a valid card: {last_error}")
