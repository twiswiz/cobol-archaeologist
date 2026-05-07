from pathlib import Path

from cobol_archaeologist.ingest.cleaner import read_cobol_file
from cobol_archaeologist.parser.paragraphs import split_paragraphs


FIXTURE = Path(__file__).parent / "fixtures" / "sample.cbl"


def test_split_paragraphs_finds_main_and_validate():
    cleaned = read_cobol_file(FIXTURE)
    paragraphs = split_paragraphs(cleaned)
    names = [p.name for p in paragraphs]
    assert "MAIN-PARA" in names
    assert "VALIDATE-WITHDRAWAL" in names


def test_paragraph_code_contains_if():
    cleaned = read_cobol_file(FIXTURE)
    paragraphs = split_paragraphs(cleaned)
    val = next(p for p in paragraphs if p.name == "VALIDATE-WITHDRAWAL")
    assert "IF WITHDRAWAL-AMOUNT" in val.code.upper()
