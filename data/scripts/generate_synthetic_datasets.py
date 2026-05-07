"""Golden seed dataset.

This module replaces the original synthetic generator. It writes:
- generated_cobol_logic_blocks.jsonl  (LogicBlock examples to feed the pipeline)
- business_intent_cards.jsonl          (reference cards)
- business_intent_cards.csv            (flat view)
- golden_eval.jsonl                    (paired LogicBlock + reference for eval)

These are hand-curated and meant for instruction tuning + the eval harness.
The bulk processed dataset is now produced by the real pipeline
(`cobol-archaeologist segment ...`).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GOLDEN = [
    {
        "block": {
            "id": "lb_golden_001",
            "source_file": "synthetic/banking.cbl",
            "paragraph": "VALIDATE-WITHDRAWAL",
            "code": (
                "IF WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE\n"
                "    MOVE 'Y' TO INSUFFICIENT-FUNDS-FLAG\n"
                "    MOVE 'DECLINED' TO TRANSACTION-STATUS\n"
                "ELSE\n"
                "    SUBTRACT WITHDRAWAL-AMOUNT FROM AVAILABLE-BALANCE\n"
                "    MOVE 'APPROVED' TO TRANSACTION-STATUS\n"
                "END-IF"
            ),
            "vars_read": ["WITHDRAWAL-AMOUNT", "AVAILABLE-BALANCE"],
            "vars_written": ["INSUFFICIENT-FUNDS-FLAG", "TRANSACTION-STATUS", "AVAILABLE-BALANCE"],
            "conditions": ["WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE"],
            "tags": ["banking", "balance_check"],
        },
        "reference": {
            "what": "Reject a withdrawal when the requested amount exceeds the available balance.",
            "why": "Prevents overdrafts and supports the bank's transaction approval policy.",
            "relevant_regulation_ids": [],
        },
    },
    {
        "block": {
            "id": "lb_golden_002",
            "source_file": "synthetic/credit_card.cbl",
            "paragraph": "APPLY-LATE-FEE",
            "code": (
                "IF PAYMENT-DATE > DUE-DATE\n"
                "    ADD LATE-FEE-AMOUNT TO CURRENT-BALANCE\n"
                "    MOVE 'LATE-FEE-APPLIED' TO ACCOUNT-ACTION\n"
                "ELSE\n"
                "    MOVE 'NO-FEE' TO ACCOUNT-ACTION\n"
                "END-IF"
            ),
            "vars_read": ["PAYMENT-DATE", "DUE-DATE", "LATE-FEE-AMOUNT", "CURRENT-BALANCE"],
            "vars_written": ["CURRENT-BALANCE", "ACCOUNT-ACTION"],
            "conditions": ["PAYMENT-DATE > DUE-DATE"],
            "tags": ["credit_card", "late_fee"],
        },
        "reference": {
            "what": "Apply a late fee when a payment is received after the due date.",
            "why": "Enforces credit-card payment timeliness and fee-recovery policy.",
            "relevant_regulation_ids": [],
        },
    },
    {
        "block": {
            "id": "lb_golden_003",
            "source_file": "synthetic/kyc.cbl",
            "paragraph": "KYC-RISK-ROUTE",
            "code": (
                "IF CUSTOMER-COUNTRY = HIGH-RISK-COUNTRY\n"
                "    MOVE 'EDD' TO REVIEW-TYPE\n"
                "    MOVE 'MANUAL' TO APPROVAL-ROUTE\n"
                "ELSE\n"
                "    MOVE 'STANDARD' TO REVIEW-TYPE\n"
                "END-IF"
            ),
            "vars_read": ["CUSTOMER-COUNTRY", "HIGH-RISK-COUNTRY"],
            "vars_written": ["REVIEW-TYPE", "APPROVAL-ROUTE"],
            "conditions": ["CUSTOMER-COUNTRY = HIGH-RISK-COUNTRY"],
            "tags": ["kyc", "compliance"],
        },
        "reference": {
            "what": "Route customers from high-risk countries to enhanced due diligence and manual approval.",
            "why": "Implements risk-based KYC review per AML/KYC obligations.",
            "relevant_regulation_ids": [],
        },
    },
    {
        "block": {
            "id": "lb_golden_004",
            "source_file": "synthetic/loan.cbl",
            "paragraph": "CHECK-LOAN-ELIGIBILITY",
            "code": (
                "IF CREDIT-SCORE < MIN-CREDIT-SCORE\n"
                "    MOVE 'INELIGIBLE' TO LOAN-DECISION\n"
                "ELSE\n"
                "    IF DTI-RATIO > MAX-DTI\n"
                "        MOVE 'INELIGIBLE' TO LOAN-DECISION\n"
                "    ELSE\n"
                "        MOVE 'ELIGIBLE' TO LOAN-DECISION\n"
                "    END-IF\n"
                "END-IF"
            ),
            "vars_read": ["CREDIT-SCORE", "MIN-CREDIT-SCORE", "DTI-RATIO", "MAX-DTI"],
            "vars_written": ["LOAN-DECISION"],
            "conditions": ["CREDIT-SCORE < MIN-CREDIT-SCORE", "DTI-RATIO > MAX-DTI"],
            "tags": ["loan", "eligibility"],
        },
        "reference": {
            "what": "Mark a loan applicant ineligible when credit score is too low or DTI too high.",
            "why": "Implements basic underwriting thresholds for loan approval.",
            "relevant_regulation_ids": [],
        },
    },
    {
        "block": {
            "id": "lb_golden_005",
            "source_file": "synthetic/payroll.cbl",
            "paragraph": "COMPUTE-NET-PAY",
            "code": (
                "COMPUTE GROSS-PAY = HOURS-WORKED * HOURLY-RATE.\n"
                "COMPUTE TAX-DEDUCTION = GROSS-PAY * TAX-RATE.\n"
                "COMPUTE NET-PAY = GROSS-PAY - TAX-DEDUCTION."
            ),
            "vars_read": ["HOURS-WORKED", "HOURLY-RATE", "TAX-RATE"],
            "vars_written": ["GROSS-PAY", "TAX-DEDUCTION", "NET-PAY"],
            "conditions": [],
            "tags": ["payroll"],
        },
        "reference": {
            "what": "Compute gross pay, tax deduction, and net pay for an employee.",
            "why": "Standard payroll calculation flow producing a take-home figure.",
            "relevant_regulation_ids": [],
        },
    },
]


# Back-compat exports used by older callers / docs.
LOGIC_BLOCKS = [
    {
        "id": g["block"]["id"],
        "domain": g["block"]["tags"][0] if g["block"]["tags"] else "general",
        "intent": g["reference"]["what"],
        "cobol": g["block"]["code"].splitlines(),
        "inputs": g["block"]["vars_read"],
        "outputs": g["block"]["vars_written"],
    }
    for g in GOLDEN
]

BUSINESS_INTENT_CARDS = [
    {
        "id": f"intent_card_{i+1:03d}",
        "task": "summarize_cobol",
        "instruction": "Explain the business rule implemented by this COBOL block.",
        "context_id": g["block"]["id"],
        "expected_response": g["reference"]["what"],
        "evaluation_tags": g["block"]["tags"],
    }
    for i, g in enumerate(GOLDEN)
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_intent_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(
            h,
            fieldnames=["id", "task", "instruction", "context_id", "expected_response", "evaluation_tags"],
        )
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["evaluation_tags"] = "|".join(r["evaluation_tags"])
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    blocks = [dict(g["block"]) for g in GOLDEN]
    write_jsonl(out / "generated_cobol_logic_blocks.jsonl", blocks)
    write_jsonl(out / "business_intent_cards.jsonl", BUSINESS_INTENT_CARDS)
    write_intent_csv(out / "business_intent_cards.csv", BUSINESS_INTENT_CARDS)
    write_jsonl(out / "golden_eval.jsonl", GOLDEN)


if __name__ == "__main__":
    main()
