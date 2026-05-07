"""Prompt template for generating Business Intent Cards."""
from __future__ import annotations

import json

from ..schemas import LogicBlock, RegulationChunk

SYSTEM_INSTRUCTION = """You are COBOL-Archaeologist, an analyst that recovers business intent
from undocumented COBOL programs running in banks and financial institutions.

For the COBOL block below, produce a SINGLE JSON object with EXACTLY these keys:
  what               (string) - the business operation implemented by the block
  why                (string) - the likely financial/operational reason
  code_evidence      (array of strings) - cite variables, conditions and statements in the block
  regulation_link    (string|null) - if a retrieved regulation snippet clearly applies; else null
  regulation_sources (array of strings) - chunk ids of retrieved snippets you actually used
  confidence         (object with keys "level": High|Medium|Low and "justification": string)

Hard rules:
- Output JSON only. No prose, no markdown fences.
- Every variable mentioned in code_evidence MUST appear in the static context.
- Set regulation_link to null unless a retrieved snippet is clearly relevant.
"""


def render_prompt(block: LogicBlock, retrieved: list[RegulationChunk] | None = None) -> str:
    ctx = {
        "paragraph": block.paragraph,
        "vars_read": block.vars_read,
        "vars_written": block.vars_written,
        "conditions": block.conditions,
        "perform_calls": block.perform_calls,
        "file_refs": block.file_refs,
        "copybooks": block.copybooks,
        "weak_label": block.weak_label,
        "tags": block.tags,
    }
    retrieved = retrieved or []
    reg_blob = "\n\n".join(
        f"[{r.id}] source={r.source} page={r.page}\n{r.text}" for r in retrieved
    ) or "(none)"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"COBOL CODE:\n{block.code}\n"
        f"STATIC CONTEXT:\n{json.dumps(ctx, ensure_ascii=False)}\n"
        f"RETRIEVED REGULATIONS:\n{reg_blob}\n"
        f"OUTPUT JSON:\n"
    )
