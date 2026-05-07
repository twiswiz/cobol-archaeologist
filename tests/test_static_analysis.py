from cobol_archaeologist.static_analysis.extract import extract_facts


CODE = """
       VALIDATE-WITHDRAWAL.
           IF WITHDRAWAL-AMOUNT > AVAILABLE-BALANCE
               MOVE 'Y' TO INSUFFICIENT-FUNDS-FLAG
               MOVE 'DECLINED' TO TRANSACTION-STATUS
           ELSE
               SUBTRACT WITHDRAWAL-AMOUNT FROM AVAILABLE-BALANCE
               MOVE 'APPROVED' TO TRANSACTION-STATUS
           END-IF.
"""


def test_extract_finds_vars_and_conditions():
    facts = extract_facts(CODE)
    assert "WITHDRAWAL-AMOUNT" in facts.vars_read
    assert "AVAILABLE-BALANCE" in facts.vars_read
    assert "INSUFFICIENT-FUNDS-FLAG" in facts.vars_written
    assert "TRANSACTION-STATUS" in facts.vars_written
    assert any("WITHDRAWAL-AMOUNT" in c.upper() for c in facts.conditions)


def test_perform_capture():
    facts = extract_facts("       MAIN. PERFORM VALIDATE-WITHDRAWAL.")
    assert "VALIDATE-WITHDRAWAL" in facts.perform_calls
