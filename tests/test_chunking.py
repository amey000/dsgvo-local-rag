import itertools

import pytest

from dsgvo_rag.chunking import chunk_text, normalize


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("Kurzer Text.", 100, 10) == ["Kurzer Text."]


def test_empty_text() -> None:
    assert chunk_text("   \n\n ", 100, 10) == []


def test_chunks_respect_size_and_overlap() -> None:
    sentences = [f"Satz Nummer {i} enthält etwas Inhalt." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share some text (overlap)
    for a, b in itertools.pairwise(chunks):
        assert any(word in a for word in b.split()[:3])
    # nothing is lost
    for sentence in sentences:
        assert any(sentence.split(" enthält")[0] in c for c in chunks)


def test_paragraph_boundaries_preferred() -> None:
    text = "Absatz eins. " * 5 + "\n\n" + "Absatz zwei. " * 5
    chunks = chunk_text(text, chunk_size=80, overlap=0)
    assert chunks[0].startswith("Absatz eins")
    assert all("eins" not in c or "zwei" not in c for c in chunks)


def test_placeholders_are_never_split() -> None:
    text = " ".join(f"<PERSON_{i}> hat den Vertrag unterschrieben." for i in range(1, 40))
    for chunk in chunk_text(text, chunk_size=120, overlap=40):
        for token in chunk.split():
            if "PERSON" in token:
                assert token.startswith("<PERSON_"), token


def test_normalize_joins_pdf_hyphenation_and_whitespace() -> None:
    raw = "Die Kündi-\ngung   erfolgt\r\n\n\n\nschriftlich."
    assert normalize(raw) == "Die Kündigung erfolgt\n\nschriftlich."


def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)
