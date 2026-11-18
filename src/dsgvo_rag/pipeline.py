"""Orchestration: extract → redact → chunk → embed → store, and retrieve → prompt → cite.

Privacy invariant: raw document text and raw questions never leave this module. The embedder,
the vector store, the LLM and the logs only ever receive pseudonymised text.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .llm import LLM
from .loaders import load_bytes
from .prompts import NO_ANSWER, build_messages, extract_citations, is_no_answer
from .redaction import Pseudonymizer, Redactor
from .vectorstore import ChunkRecord, DocumentInfo, SearchHit, VectorStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    source: str
    pages: int
    chunks: int
    redacted_entities: dict[str, int]


@dataclass(frozen=True)
class Source:
    index: int
    source: str
    page: int | None
    doc_id: str
    score: float
    text: str
    cited: bool


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[Source]
    redacted_question: str
    question_entities: dict[str, int]
    model: str
    grounded: bool  # True if the answer cites at least one source (or correctly declines)
    llm_called: bool = True
    warnings: list[str] = field(default_factory=list)


class RagService:
    def __init__(
        self,
        settings: Settings,
        redactor: Redactor,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
    ) -> None:
        self.settings = settings
        self.redactor = redactor
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.store.ensure_collection()

    # ------------------------------------------------------------------ ingestion
    def ingest_bytes(self, data: bytes, filename: str) -> IngestResult:
        doc_id = hashlib.sha256(data).hexdigest()[:32]
        pages = load_bytes(data, filename)

        # One pseudonymizer per document → "<PERSON_1>" means the same person on every page.
        pseudonymizer = Pseudonymizer()
        entity_counts: Counter[str] = Counter()
        redacted_pages = []
        for page in pages:
            redacted = self.redactor.redact(page.text, pseudonymizer)
            entity_counts.update(redacted.entities)
            redacted_pages.append((page.number, redacted.text))
        # After the body, so names found in the text are also caught in the file name.
        source = self.redactor.redact_filename(filename, pseudonymizer)

        now = datetime.now(UTC).isoformat(timespec="seconds")
        records: list[ChunkRecord] = []
        for page_number, text in redacted_pages:
            for chunk in chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap):
                records.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        source=source,
                        page=page_number,
                        chunk_index=len(records),
                        text=chunk,
                        ingested_at=now,
                    )
                )

        # Re-ingesting replaces the previous version of the same content.
        self.store.delete_document(doc_id)
        if records:
            vectors = self.embedder.embed_passages([r.text for r in records])
            self.store.upsert(records, vectors)

        # Log metadata only – never document content.
        log.info(
            "Ingested doc_id=%s pages=%d chunks=%d redacted=%s",
            doc_id,
            len(pages),
            len(records),
            dict(entity_counts),
        )
        return IngestResult(doc_id, source, len(pages), len(records), dict(entity_counts))

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest_bytes(path.read_bytes(), path.name)

    # ------------------------------------------------------------------ querying
    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        vector = self.embedder.embed_query(question)
        return self.store.search(vector, top_k or self.settings.top_k, self.settings.min_score)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        if self.settings.redact_queries:
            redacted = self.redactor.redact(question)
            safe_question, question_entities = redacted.text, redacted.entities
        else:
            safe_question, question_entities = question, {}

        hits = self.retrieve(safe_question, top_k)
        if not hits:
            log.info("No chunk above min_score=%.2f – LLM not called", self.settings.min_score)
            return QueryResult(
                answer=NO_ANSWER,
                sources=[],
                redacted_question=safe_question,
                question_entities=question_entities,
                model=self.llm.model,
                grounded=True,
                llm_called=False,
            )

        answer = self.llm.chat(build_messages(safe_question, hits))
        cited = set(extract_citations(answer, len(hits)))
        declined = is_no_answer(answer)
        warnings = []
        if not cited and not declined:
            warnings.append("Die Antwort enthält keine Quellenangaben – bitte manuell prüfen.")

        sources = [
            Source(
                index=i,
                source=h.source,
                page=h.page,
                doc_id=h.doc_id,
                score=round(h.score, 4),
                text=h.text,
                cited=i in cited,
            )
            for i, h in enumerate(hits, start=1)
        ]
        log.info("Answered query: hits=%d cited=%s", len(hits), sorted(cited))
        return QueryResult(
            answer=answer,
            sources=sources,
            redacted_question=safe_question,
            question_entities=question_entities,
            model=self.llm.model,
            grounded=bool(cited) or declined,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ management
    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        deleted = self.store.delete_document(doc_id)
        log.info("Deleted doc_id=%s chunks=%d", doc_id, deleted)
        return deleted


def build_service(settings: Settings) -> RagService:
    """Wire up the production components (Presidio, e5, Qdrant, Ollama)."""
    from qdrant_client import QdrantClient

    from .embeddings import E5Embedder
    from .llm import OllamaClient

    embedder = E5Embedder(
        settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
    )
    return RagService(
        settings=settings,
        redactor=Redactor(
            spacy_model=settings.spacy_model,
            entities=settings.pii_entities,
            score_threshold=settings.pii_score_threshold,
            allow_list=settings.pii_allow_list,
        ),
        embedder=embedder,
        store=VectorStore(
            QdrantClient(url=settings.qdrant_url), settings.qdrant_collection, embedder.dim
        ),
        llm=OllamaClient(
            settings.ollama_base_url,
            settings.llm_model,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            timeout_s=settings.llm_timeout_s,
        ),
    )
