"""Retrieval quality with the real multilingual-e5-large model (≈2 GB download).

Run with: pytest -m slow
"""

import pytest
from qdrant_client import QdrantClient

from dsgvo_rag.config import Settings
from dsgvo_rag.pipeline import RagService
from dsgvo_rag.vectorstore import VectorStore

from conftest import SAMPLES, FakeLLM

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def e5_service(redactor) -> RagService:
    from dsgvo_rag.embeddings import E5Embedder

    settings = Settings(_env_file=None)
    embedder = E5Embedder(settings.embedding_model)
    store = VectorStore(QdrantClient(":memory:"), "e5", embedder.dim)
    svc = RagService(settings, redactor, embedder, store, FakeLLM())
    for path in sorted(SAMPLES.iterdir()):
        svc.ingest_path(path)
    return svc


@pytest.mark.parametrize(
    ("question", "expected_source", "expected_text"),
    [
        ("Wie viele Urlaubstage habe ich?", "mitarbeiterhandbuch.md", "30 Arbeitstage"),
        ("Wie hoch ist die Verpflegungspauschale?", "mitarbeiterhandbuch.md", "14 Euro"),
        ("Bis wann müssen Datenpannen gemeldet werden?", "datenschutzrichtlinie.md", "72 Stunden"),
        ("Wie behebe ich den DATEV-Export-Fehler?", "kundenbeschwerde", "SKR03"),
    ],
)
def test_relevant_chunk_is_retrieved(e5_service, question, expected_source, expected_text):
    hits = e5_service.retrieve(question, top_k=3)
    assert hits, "kein Treffer über min_score"
    assert hits[0].source.startswith(expected_source)
    assert any(expected_text in h.text for h in hits)


@pytest.mark.parametrize(
    "question", ["Welche Farbe hat der Mond?", "Wer hat die Fußball-WM 2014 gewonnen?"]
)
def test_off_topic_questions_fall_below_threshold(e5_service, question):
    assert e5_service.retrieve(question) == []
