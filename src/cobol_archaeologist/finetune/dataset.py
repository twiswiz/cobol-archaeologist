"""Build a supervised fine-tuning corpus from infer envelope JSONL + logic blocks.

Inputs
------
- A JSONL of envelope records produced by `cli.py infer ...` (each line has
  `logic_block_id`, `model`, `mode`, `retrieved_chunk_ids`, `card`).
- The same `logic_blocks.jsonl` used for inference (we re-render the prompt
  from the block + the retrieved chunks so the training prompt is identical
  to the runtime prompt).
- Optionally the regulation chunks JSONL to fetch the actual chunk texts for
  the RAG section. When the file is missing we render `(none)` so the model
  still learns the JSON shape.

Output
------
A JSONL where each line is a chat-style record:

    {"messages": [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user",   "content": "<rest of prompt>"},
        {"role": "assistant", "content": "<card json>"}
    ]}

The format matches what `transformers` chat templates and TRL's `SFTTrainer`
consume for Qwen / Llama style instruct models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..model.prompt import SYSTEM_INSTRUCTION, render_prompt
from ..schemas import BusinessIntentCard, LogicBlock, RegulationChunk


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _load_blocks(path: Path) -> dict[str, LogicBlock]:
    return {b.id: b for b in (LogicBlock.model_validate(r) for r in _iter_jsonl(path))}


def _load_chunks(path: Path | None) -> dict[str, RegulationChunk]:
    if path is None or not path.exists():
        return {}
    return {
        c.id: c
        for c in (RegulationChunk.model_validate(r) for r in _iter_jsonl(path))
    }


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Split the rendered prompt into (system, user) by stripping SYSTEM_INSTRUCTION."""
    sys = SYSTEM_INSTRUCTION.rstrip()
    if prompt.startswith(sys):
        rest = prompt[len(sys):].lstrip("\n")
    else:
        rest = prompt
    return sys, rest


def _envelope_to_card(env: dict) -> BusinessIntentCard | None:
    card = env.get("card")
    if not card:
        return None
    try:
        return BusinessIntentCard.model_validate(card)
    except Exception:
        return None


def _is_quality(card: BusinessIntentCard) -> bool:
    """Filter obviously bad cards out of the training set."""
    what = (card.what or "").strip()
    why = (card.why or "").strip()
    if len(what) < 5 or len(why) < 5:
        return False
    if "Performs business logic over the listed COBOL variables" in what:
        return False
    if what.lower() == "unknown" or why.lower() == "unknown":
        return False
    return True


def build_corpus(
    envelopes: Path,
    logic_blocks: Path,
    out_path: Path,
    chunks_path: Path | None = None,
    include_static: bool = True,
    include_rag: bool = True,
    quality_filter: bool = True,
) -> dict:
    """Build a chat-format SFT JSONL.

    Returns a small summary dict (counts, dropped reasons).
    """
    blocks = _load_blocks(logic_blocks)
    chunks = _load_chunks(chunks_path)

    kept = 0
    dropped_no_block = 0
    dropped_no_card = 0
    dropped_quality = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for env in _iter_jsonl(envelopes):
            block_id = env.get("logic_block_id")
            block = blocks.get(block_id) if block_id else None
            if block is None:
                dropped_no_block += 1
                continue
            card = _envelope_to_card(env)
            if card is None:
                dropped_no_card += 1
                continue
            if quality_filter and not _is_quality(card):
                dropped_quality += 1
                continue

            retrieved = [
                chunks[cid] for cid in env.get("retrieved_chunk_ids", []) if cid in chunks
            ]
            prompt = render_prompt(
                block,
                retrieved=retrieved,
                include_static=include_static,
                include_rag=include_rag,
            )
            sys, user = _split_prompt(prompt)
            assistant = card.model_dump_json()
            record = {
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user.rstrip()},
                    {"role": "assistant", "content": assistant},
                ],
                "logic_block_id": block.id,
                "source_model": env.get("model"),
                "mode": env.get("mode"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "kept": kept,
        "dropped_no_block": dropped_no_block,
        "dropped_no_card": dropped_no_card,
        "dropped_quality": dropped_quality,
        "out_path": str(out_path),
    }
