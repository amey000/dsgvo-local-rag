from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from dsgvo_rag.config import Settings
from dsgvo_rag.pipeline import RagService
from dsgvo_rag.redaction import Redactor
from dsgvo_rag.vectorstore import VectorStore

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"

# Every piece of personal data contained in data/samples – none of it may ever be stored,
# embedded or sent to the LLM.
SAMPLE_PII = [
    "Sabine Wagner",
    "Wagner",
    "s.wagner@example.com",
    "089 99998123",
    "Thomas Becker",
    "030 23125000",
    "Jürgen Müller",
    "Lindenstraße 14",
    "0221 4710123",
    "juergen.mueller@example.com",
    "DE89 3704 0044 0532 0130 00",
    "Laura Fischer",
    "Markus Weber",
    "069 90009456",
    "datenschutz@example.com",
]


_STOPWORDS_TEXT = (
    "der die das den dem des ein eine einer und oder in im an am auf aus bei mit von vor zu zum "
    "zur für ist sind hat haben wie viele was wer wo wann ich sie es nicht nur auch person"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests. Records everything it is asked to embed."""

    dim = 512

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[^\W\d_]{2,}", text.lower()):
            if token in STOPWORDS:
                continue
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.seen.append(text)
        return self._vec(text)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, answer: str = "Es gibt 30 Urlaubstage [1].") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.answer

    def is_ready(self) -> bool:
        return True


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, min_score=0.0, chunk_size=500, chunk_overlap=80)


@pytest.fixture(scope="session")
def redactor(settings: Settings) -> Redactor:
    # Loading spaCy takes a few seconds – share one instance across all tests.
    return Redactor(
        spacy_model=settings.spacy_model,
        entities=settings.pii_entities,
        score_threshold=settings.pii_score_threshold,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(settings, redactor, embedder, llm) -> RagService:
    store = VectorStore(QdrantClient(":memory:"), "test", embedder.dim)
    return RagService(settings, redactor, embedder, store, llm)


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid multi-page PDF (Helvetica, WinAnsi) without extra dependencies."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for pid, text in zip(page_ids, pages, strict=True):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
