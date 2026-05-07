from cobol_archaeologist.ingest.cleaner import clean_line, clean_text


def test_strip_sequence_and_comment_indicator():
    line = "000100* This is a comment line"
    assert clean_line(line, drop_comments=True) is None
    assert clean_line(line, drop_comments=False).startswith("      *")


def test_normal_line_keeps_body():
    line = "000200 IF A > B"
    out = clean_line(line)
    assert "IF A > B" in out


def test_clean_text_uppercase():
    text = "000100 if a > b"
    cleaned = clean_text(text, uppercase=True)
    assert "IF A > B" in cleaned
