from dsgvo_rag.prompts import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    build_messages,
    extract_citations,
    is_no_answer,
)
from dsgvo_rag.vectorstore import SearchHit


def _hit(source: str, page: int | None, text: str) -> SearchHit:
    return SearchHit(score=0.9, doc_id="d", source=source, page=page, chunk_index=0, text=text)


def test_extract_citations_formats() -> None:
    assert extract_citations("A [1]. B [2][3]. C [1, 4].", 5) == [1, 2, 3, 4]
    assert extract_citations("Siehe [3; 2].", 3) == [3, 2]


def test_extract_citations_drops_invalid_numbers() -> None:
    assert extract_citations("Erfunden [7] und [0], echt [2].", 3) == [2]
    assert extract_citations("Keine Quelle.", 3) == []


def test_no_answer_detection() -> None:
    assert is_no_answer(NO_ANSWER)
    assert is_no_answer(f"Leider: {NO_ANSWER.lower()}")
    assert not is_no_answer("Es gibt 30 Urlaubstage [1].")


def test_build_messages_numbers_sources_with_pages() -> None:
    messages = build_messages(
        "Wie viele Urlaubstage?",
        [_hit("handbuch.pdf", 3, "30 Tage Urlaub."), _hit("notiz.md", None, "Gleitzeit.")],
    )
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "[1] Quelle: handbuch.pdf, Seite 3\n30 Tage Urlaub." in user
    assert "[2] Quelle: notiz.md\nGleitzeit." in user
    assert user.rstrip().endswith("Antwort mit Quellenangaben:")


def test_system_prompt_forbids_reidentification() -> None:
    assert "<PERSON_1>" in SYSTEM_PROMPT
    assert "niemals" in SYSTEM_PROMPT
    assert NO_ANSWER in SYSTEM_PROMPT
