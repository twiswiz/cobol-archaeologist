from cobol_archaeologist.labels.weak import label_block
from cobol_archaeologist.schemas import LogicBlock


def _block(**kw) -> LogicBlock:
    base = dict(
        id="lb_test",
        source_file="x.cbl",
        paragraph="P",
        code="",
        vars_read=[],
        vars_written=[],
        conditions=[],
    )
    base.update(kw)
    return LogicBlock(**base)


def test_balance_check_rule():
    b = _block(
        paragraph="VALIDATE-WITHDRAWAL",
        vars_read=["WITHDRAWAL-AMOUNT", "AVAILABLE-BALANCE"],
        code="IF WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE",
    )
    out = label_block(b)
    assert out.weak_label == "balance_check"
    assert "banking" in out.tags


def test_kyc_rule():
    b = _block(
        paragraph="KYC-REVIEW",
        vars_read=["CUSTOMER-COUNTRY", "HIGH-RISK-COUNTRY"],
        code="IF CUSTOMER-COUNTRY = HIGH-RISK-COUNTRY MOVE 'EDD' TO REVIEW-TYPE",
    )
    out = label_block(b)
    assert out.weak_label == "kyc_screening"


def test_unknown_when_no_keywords():
    b = _block(paragraph="X", code="MOVE 1 TO Y")
    out = label_block(b)
    assert out.weak_label == "unknown"
