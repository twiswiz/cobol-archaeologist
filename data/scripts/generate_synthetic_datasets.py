import argparse
import csv
import json
from pathlib import Path


LOGIC_BLOCKS = [
    {
        "id": "logic_block_001",
        "domain": "banking",
        "intent": "Reject withdrawals when available balance is insufficient.",
        "cobol": [
            "IF WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE",
            "    MOVE 'Y' TO INSUFFICIENT-FUNDS-FLAG",
            "    MOVE 'DECLINED' TO TRANSACTION-STATUS",
            "ELSE",
            "    SUBTRACT WITHDRAWAL-AMOUNT FROM AVAILABLE-BALANCE",
            "    MOVE 'APPROVED' TO TRANSACTION-STATUS",
            "END-IF",
        ],
        "inputs": ["WITHDRAWAL-AMOUNT", "AVAILABLE-BALANCE"],
        "outputs": ["INSUFFICIENT-FUNDS-FLAG", "TRANSACTION-STATUS", "AVAILABLE-BALANCE"],
    },
    {
        "id": "logic_block_002",
        "domain": "credit_card",
        "intent": "Apply a late fee when the payment is received after the due date.",
        "cobol": [
            "IF PAYMENT-DATE > DUE-DATE",
            "    ADD LATE-FEE-AMOUNT TO CURRENT-BALANCE",
            "    MOVE 'LATE-FEE-APPLIED' TO ACCOUNT-ACTION",
            "ELSE",
            "    MOVE 'NO-FEE' TO ACCOUNT-ACTION",
            "END-IF",
        ],
        "inputs": ["PAYMENT-DATE", "DUE-DATE", "LATE-FEE-AMOUNT", "CURRENT-BALANCE"],
        "outputs": ["CURRENT-BALANCE", "ACCOUNT-ACTION"],
    },
    {
        "id": "logic_block_003",
        "domain": "kyc",
        "intent": "Mark customer review as enhanced due diligence for high-risk countries.",
        "cobol": [
            "IF CUSTOMER-COUNTRY = HIGH-RISK-COUNTRY",
            "    MOVE 'EDD' TO REVIEW-TYPE",
            "    MOVE 'MANUAL' TO APPROVAL-ROUTE",
            "ELSE",
            "    MOVE 'STANDARD' TO REVIEW-TYPE",
            "END-IF",
        ],
        "inputs": ["CUSTOMER-COUNTRY", "HIGH-RISK-COUNTRY"],
        "outputs": ["REVIEW-TYPE", "APPROVAL-ROUTE"],
    },
]


BUSINESS_INTENT_CARDS = [
    {
        "id": "intent_card_001",
        "task": "summarize_cobol",
        "instruction": "Explain the business rule implemented by this COBOL block.",
        "context_id": "logic_block_001",
        "expected_response": "The block approves a withdrawal only when the requested amount does not exceed the available balance; otherwise it flags insufficient funds and declines the transaction.",
        "evaluation_tags": ["control_flow", "banking", "balance_check"],
    },
    {
        "id": "intent_card_002",
        "task": "generate_tests",
        "instruction": "Create positive and negative test scenarios for this late-fee rule.",
        "context_id": "logic_block_002",
        "expected_response": "Test an on-time payment that leaves the balance unchanged and a late payment that adds the configured late fee to the current balance.",
        "evaluation_tags": ["test_generation", "credit_card", "date_rule"],
    },
    {
        "id": "intent_card_003",
        "task": "map_regulation_to_code",
        "instruction": "Identify why this customer review route might matter for KYC compliance.",
        "context_id": "logic_block_003",
        "expected_response": "Customers associated with high-risk countries are routed to enhanced due diligence and manual approval, supporting risk-based KYC review.",
        "evaluation_tags": ["kyc", "risk_review", "compliance"],
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_intent_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "task",
                "instruction",
                "context_id",
                "expected_response",
                "evaluation_tags",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["evaluation_tags"] = "|".join(row["evaluation_tags"])
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logic_blocks = []
    for block in LOGIC_BLOCKS:
        row = dict(block)
        row["cobol"] = "\n".join(block["cobol"])
        logic_blocks.append(row)

    write_jsonl(output_dir / "generated_cobol_logic_blocks.jsonl", logic_blocks)
    write_jsonl(output_dir / "business_intent_cards.jsonl", BUSINESS_INTENT_CARDS)
    write_intent_csv(output_dir / "business_intent_cards.csv", BUSINESS_INTENT_CARDS)


if __name__ == "__main__":
    main()
