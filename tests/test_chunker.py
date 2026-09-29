from app.ingestion.chunker import chunk_text


def test_chunk_text_creates_chunks():
    text = "This is a test document. " * 100

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1


def test_chunk_text_returns_empty_for_empty_input():
    assert chunk_text("") == []


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("test", chunk_size=100, overlap=100)
        assert False
    except ValueError:
        assert True