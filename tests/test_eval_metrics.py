from cobol_archaeologist.eval.metrics import (
    aggregate,
    evaluate_card,
    evidence_faithfulness,
    rouge_l_f,
)
from cobol_archaeologist.schemas import BusinessIntentCard, Confidence, LogicBlock


def _block():
    return LogicBlock(
        id="x",
        source_file="f",
        paragraph="P",
        code="",
        vars_read=["WITHDRAWAL-AMOUNT", "AVAILABLE-BALANCE"],
        vars_written=["TRANSACTION-STATUS"],
        conditions=["WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE"],
    )


def test_faithfulness_perfect():
    card = BusinessIntentCard(
        what="ok", why="ok",
        code_evidence=["Reads WITHDRAWAL-AMOUNT", "Writes TRANSACTION-STATUS"],
        confidence=Confidence(level="High", justification=""),
    )
    assert evidence_faithfulness(card, _block()) == 1.0


def test_faithfulness_partial():
    card = BusinessIntentCard(
        what="ok", why="ok",
        code_evidence=["Reads UNKNOWN-VAR", "Writes TRANSACTION-STATUS"],
        confidence=Confidence(level="High", justification=""),
    )
    assert 0.0 < evidence_faithfulness(card, _block()) < 1.0


def test_rouge_l_basic():
    assert rouge_l_f("a b c d", "a b c d") == 1.0
    assert rouge_l_f("", "x") == 0.0


def test_aggregate_empty():
    assert aggregate([]) == {}


def test_evaluate_card_smoke():
    card = BusinessIntentCard(
        what="validate withdrawal",
        why="check balance",
        code_evidence=["Reads WITHDRAWAL-AMOUNT"],
        confidence=Confidence(level="High", justification=""),
    )
    res = evaluate_card(card, _block(), {"what": "validate withdrawal", "why": "check balance"})
    assert res.json_valid
    assert res.faithfulness == 1.0
