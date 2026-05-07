"""Heuristic weak labeler.

Two layers:
1. Rule layer: keyword + variable-name patterns map to a candidate intent tag.
2. Optional LLM-assist layer: if a backend is provided, the labeler can ask the
   model for a refined tag when the rules return ``unknown``. This realizes the
   "rules + small-LLM auto-label" path discussed in the plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..schemas import LogicBlock


@dataclass(frozen=True)
class Rule:
    label: str
    tags: tuple[str, ...]
    pattern: re.Pattern[str]
    confidence: float = 0.6


def _p(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)


RULES: list[Rule] = [
    Rule(
        "balance_check",
        ("banking", "balance_check", "control_flow"),
        _p("BALANCE", "AVAILABLE-BALANCE", "INSUFFICIENT", "OVERDRAFT"),
        0.75,
    ),
    Rule(
        "late_fee",
        ("credit_card", "late_fee", "date_rule"),
        _p("LATE-FEE", "DUE-DATE", "LATE-PAYMENT", "PAYMENT-DATE"),
        0.75,
    ),
    Rule(
        "kyc_screening",
        ("kyc", "compliance", "risk_review"),
        _p("KYC", "HIGH-RISK", "RISK-COUNTRY", "EDD", "PEP", "SANCTION"),
        0.8,
    ),
    Rule(
        "interest_calculation",
        ("banking", "interest", "computation"),
        _p("INTEREST", "ACCRUED-INTEREST", "INTEREST-RATE", "APR"),
        0.7,
    ),
    Rule(
        "loan_eligibility",
        ("loan", "eligibility", "underwriting"),
        _p("LOAN", "ELIGIBILITY", "CREDIT-SCORE", "DTI"),
        0.65,
    ),
    Rule(
        "transaction_validation",
        ("banking", "transaction", "validation"),
        _p("TRANSACTION", "VALIDATE", "REJECT", "APPROVE", "DECLINE"),
        0.55,
    ),
    Rule(
        "fraud_check",
        ("banking", "fraud", "risk"),
        _p("FRAUD", "SUSPICIOUS", "AML", "ANTI-MONEY"),
        0.8,
    ),
    Rule(
        "payroll",
        ("payroll", "hr"),
        _p("PAYROLL", "SALARY", "GROSS-PAY", "NET-PAY", "TAX-DEDUCTION"),
        0.75,
    ),
]


LLMRefiner = Callable[[LogicBlock], Optional[tuple[str, list[str], float]]]


def label_block(block: LogicBlock, llm_refiner: Optional[LLMRefiner] = None) -> LogicBlock:
    haystack = " ".join([block.paragraph, block.code, " ".join(block.vars_read + block.vars_written)])
    best: Rule | None = None
    for rule in RULES:
        if rule.pattern.search(haystack):
            if best is None or rule.confidence > best.confidence:
                best = rule

    if best is not None:
        block.weak_label = best.label
        block.weak_label_confidence = best.confidence
        block.tags = list(best.tags)
        return block

    if llm_refiner is not None:
        result = llm_refiner(block)
        if result is not None:
            label, tags, conf = result
            block.weak_label = label
            block.tags = list(tags)
            block.weak_label_confidence = conf
            return block

    block.weak_label = "unknown"
    block.weak_label_confidence = 0.0
    block.tags = []
    return block


def label_blocks(
    blocks: list[LogicBlock], llm_refiner: Optional[LLMRefiner] = None
) -> list[LogicBlock]:
    return [label_block(b, llm_refiner=llm_refiner) for b in blocks]
