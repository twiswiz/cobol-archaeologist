"""Deterministic + reference-based metrics for Business Intent Cards."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..schemas import BusinessIntentCard, LogicBlock


_IDENT = re.compile(r"[A-Z][A-Z0-9-]{2,}")


def _idents(text: str) -> set[str]:
    return {m.group(0).upper() for m in _IDENT.finditer(text or "")}


def evidence_faithfulness(card: BusinessIntentCard, block: LogicBlock) -> float:
    """Fraction of identifiers cited in code_evidence that exist in the block's static context."""
    pool = set(block.vars_read) | set(block.vars_written)
    pool |= {tok for c in block.conditions for tok in _idents(c)}
    pool |= set(block.file_refs) | set(block.perform_calls)

    cited: set[str] = set()
    for line in card.code_evidence:
        cited |= _idents(line)
    if not cited:
        return 0.0
    grounded = cited & pool
    return len(grounded) / len(cited)


def regulation_citation_precision(card: BusinessIntentCard, relevant_ids: set[str]) -> float | None:
    cited = set(card.regulation_sources or [])
    if not cited:
        return None
    return len(cited & relevant_ids) / len(cited)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(len(a)):
        prev = 0
        for j in range(len(b)):
            tmp = dp[j + 1]
            if a[i] == b[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = tmp
    return dp[-1]


def rouge_l_f(pred: str, ref: str) -> float:
    p, r = _tokenize(pred), _tokenize(ref)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec)


@dataclass
class CardEvalResult:
    json_valid: bool
    faithfulness: float
    rouge_what: float
    rouge_why: float
    reg_precision: float | None


def evaluate_card(
    card: BusinessIntentCard,
    block: LogicBlock,
    reference: dict | None = None,
) -> CardEvalResult:
    reference = reference or {}
    return CardEvalResult(
        json_valid=True,
        faithfulness=evidence_faithfulness(card, block),
        rouge_what=rouge_l_f(card.what, reference.get("what", "")),
        rouge_why=rouge_l_f(card.why, reference.get("why", "")),
        reg_precision=regulation_citation_precision(
            card, set(reference.get("relevant_regulation_ids", []))
        )
        if "relevant_regulation_ids" in reference
        else None,
    )


def aggregate(results: list[CardEvalResult]) -> dict:
    if not results:
        return {}
    valid = [r for r in results if r.json_valid]
    n = len(results)
    counts = Counter(r.json_valid for r in results)
    avg = lambda f: sum(getattr(r, f) for r in valid) / max(1, len(valid))
    reg = [r.reg_precision for r in valid if r.reg_precision is not None]
    return {
        "n": n,
        "json_validity": counts[True] / n,
        "faithfulness": avg("faithfulness"),
        "rouge_l_what": avg("rouge_what"),
        "rouge_l_why": avg("rouge_why"),
        "regulation_precision": (sum(reg) / len(reg)) if reg else None,
    }
