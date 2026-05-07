from cobol_archaeologist.model.backend import EchoBackend
from cobol_archaeologist.model.runner import generate_card
from cobol_archaeologist.schemas import LogicBlock


def test_echo_backend_produces_valid_card():
    block = LogicBlock(
        id="lb_x",
        source_file="x.cbl",
        paragraph="VALIDATE-WITHDRAWAL",
        code="IF WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE MOVE 'DECLINED' TO TRANSACTION-STATUS",
        vars_read=["WITHDRAWAL-AMOUNT", "AVAILABLE-BALANCE"],
        vars_written=["TRANSACTION-STATUS"],
        conditions=["WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE"],
    )
    card = generate_card(block, EchoBackend())
    assert card.confidence.level in {"High", "Medium", "Low"}
    assert card.what
    assert card.why
    assert card.code_evidence
