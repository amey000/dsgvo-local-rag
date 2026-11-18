"""End-to-end pipeline tests with real Presidio redaction, in-memory Qdrant and a fake LLM."""

import pytest
from qdrant_client import QdrantClient

from dsgvo_rag.pipeline import RagService
from dsgvo_rag.prompts import NO_ANSWER
from dsgvo_rag.vectorstore import VectorStore

from conftest import SAMPLE_PII, SAMPLES, FakeLLM, make_pdf


def _all_payload_texts(service: RagService) -> list[str]:
    points, _ = service.store.client.scroll(service.store.collection, limit=10_000)
    return [f"{p.payload['source']} {p.payload['text']}" for p in points]


@pytest.fixture
def ingested(service: RagService) -> RagService:
    for path in sorted(SAMPLES.iterdir()):
        service.ingest_path(path)
    return service


def test_no_raw_pii_reaches_vector_store_or_embedder(ingested: RagService, embedder) -> None:
    stored = "\n".join(_all_payload_texts(ingested))
    embedded = "\n".join(embedder.seen)
    assert stored and embedded
    for secret in SAMPLE_PII:
        assert secret not in stored, f"PII im Vektorspeicher: {secret}"
        assert secret not in embedded, f"PII an das Embedding-Modell übergeben: {secret}"
    # ...while the actual knowledge is still there
    assert "30 Arbeitstage" in stored
    assert "72 Stunden" in stored


def test_ingest_reports_redactions(service: RagService) -> None:
    result = service.ingest_path(SAMPLES / "kundenbeschwerde_mueller.txt")
    assert result.chunks >= 1
    assert result.redacted_entities["PERSON"] >= 3
    assert result.redacted_entities["IBAN_CODE"] == 1
    assert result.redacted_entities["DE_ADDRESS"] == 1
    # the file name itself contains a surname and must be redacted as well
    assert "mueller" not in result.source.lower()


def test_pseudonyms_consistent_across_document(service: RagService) -> None:
    service.ingest_path(SAMPLES / "kundenbeschwerde_mueller.txt")
    text = " ".join(_all_payload_texts(service))
    # Jürgen Müller appears three times and must always get the same placeholder.
    assert text.count("<PERSON_1>") >= 2


def test_query_sends_only_redacted_data_to_llm(ingested: RagService, llm: FakeLLM) -> None:
    result = ingested.query("Wie viele Arbeitstage Urlaub hat Sabine Wagner?")
    assert "Sabine Wagner" not in result.redacted_question
    assert result.question_entities == {"PERSON": 1}
    prompt = "\n".join(m["content"] for m in llm.calls[-1])
    for secret in SAMPLE_PII:
        assert secret not in prompt
    assert result.answer == "Es gibt 30 Urlaubstage [1]."
    assert result.grounded
    assert [s.index for s in result.sources if s.cited] == [1]
    assert any("Urlaub" in s.text for s in result.sources)


def test_query_without_relevant_chunks_skips_llm(settings, redactor, embedder) -> None:
    llm = FakeLLM()
    strict = settings.model_copy(update={"min_score": 0.99})
    store = VectorStore(QdrantClient(":memory:"), "strict", embedder.dim)
    svc = RagService(strict, redactor, embedder, store, llm)
    svc.ingest_path(SAMPLES / "mitarbeiterhandbuch.md")
    result = svc.query("Welche Farbe hat der Mond?")
    assert result.answer == NO_ANSWER
    assert not result.llm_called and llm.calls == []


def test_uncited_answer_is_flagged(ingested: RagService, llm: FakeLLM) -> None:
    llm.answer = "Es gibt 30 Urlaubstage."
    result = ingested.query("Wie viele Urlaubstage gibt es?")
    assert not result.grounded
    assert result.warnings


def test_declined_answer_counts_as_grounded(ingested: RagService, llm: FakeLLM) -> None:
    llm.answer = NO_ANSWER
    result = ingested.query("Wie hoch ist das Budget für Firmenwagen?")
    assert result.grounded and not result.warnings


def test_reingest_is_idempotent_and_delete_erases(service: RagService) -> None:
    path = SAMPLES / "datenschutzrichtlinie.md"
    first = service.ingest_path(path)
    second = service.ingest_path(path)
    assert first.doc_id == second.doc_id
    assert service.store.count() == first.chunks
    assert [d.doc_id for d in service.list_documents()] == [first.doc_id]

    assert service.delete_document(first.doc_id) == first.chunks
    assert service.store.count() == 0
    assert service.delete_document(first.doc_id) == 0


def test_pdf_ingestion_keeps_page_numbers(service: RagService) -> None:
    pdf = make_pdf(
        [
            "Herr Thomas Becker genehmigt alle Reisekosten.",
            "Die Probezeit betraegt sechs Monate.",
        ]
    )
    result = service.ingest_bytes(pdf, "vertrag.pdf")
    assert result.pages == 2
    points, _ = service.store.client.scroll(service.store.collection, limit=10)
    by_page = {p.payload["page"]: p.payload["text"] for p in points}
    assert set(by_page) == {1, 2}
    assert "Thomas Becker" not in by_page[1] and "<PERSON_1>" in by_page[1]
    assert "Probezeit" in by_page[2]
