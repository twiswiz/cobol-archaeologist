from cobol_archaeologist.schemas import BusinessIntentCard, Confidence, LogicBlock


def test_logic_block_roundtrip():
    b = LogicBlock(id="x", source_file="f", paragraph="P", code="MOVE 1 TO X.")
    data = b.model_dump_json()
    b2 = LogicBlock.model_validate_json(data)
    assert b2.paragraph == "P"


def test_business_intent_card_required():
    c = BusinessIntentCard(
        what="x", why="y",
        confidence=Confidence(level="High", justification="ok"),
    )
    assert c.code_evidence == []
    assert c.regulation_link is None
