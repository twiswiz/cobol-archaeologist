from cobol_archaeologist.rag.chunker import _chunk_text, chunk_pages
from cobol_archaeologist.rag.pdf_loader import PdfPage


def test_chunk_text_overlaps():
    text = " ".join([f"w{i}" for i in range(50)])
    chunks = _chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    # Overlap: last 5 words of chunk 0 should appear in chunk 1.
    tail = chunks[0].split()[-5:]
    head = chunks[1].split()[:10]
    assert any(t in head for t in tail)


def test_chunk_pages_yields_chunks_with_metadata():
    pages = [PdfPage(source="rbi-kyc", page=1, text=" ".join(["term"] * 200))]
    chunks = chunk_pages(pages, chunk_size=80, overlap=20)
    assert chunks
    assert chunks[0].source == "rbi-kyc"
    assert chunks[0].page == 1
