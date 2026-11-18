"""Qdrant wrapper. Only redacted text ever reaches this module."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

_ID_NAMESPACE = uuid.UUID("6f1c2a52-9a43-4c1e-9d0b-2f7e8c3b1a10")


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    source: str  # redacted file name
    page: int | None
    chunk_index: int
    text: str  # redacted chunk text
    ingested_at: str

    @property
    def point_id(self) -> str:
        # Deterministic IDs make re-ingesting the same document idempotent.
        return str(uuid.uuid5(_ID_NAMESPACE, f"{self.doc_id}:{self.chunk_index}"))


@dataclass(frozen=True)
class SearchHit:
    score: float
    doc_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source: str
    chunks: int
    ingested_at: str


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            existing = self.client.get_collection(self.collection).config.params.vectors
            size = getattr(existing, "size", None)
            if size is not None and size != self.dim:
                raise RuntimeError(
                    f"Collection '{self.collection}' hat Dimension {size}, "
                    f"das Embedding-Modell liefert {self.dim}."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        self.client.create_payload_index(
            self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )

    @staticmethod
    def _doc_filter(doc_id: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )

    def upsert(self, records: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if len(records) != len(vectors):
            raise ValueError("Anzahl Chunks und Vektoren stimmt nicht überein")
        points = [
            models.PointStruct(
                id=r.point_id,
                vector=v,
                payload={
                    "doc_id": r.doc_id,
                    "source": r.source,
                    "page": r.page,
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                    "ingested_at": r.ingested_at,
                },
            )
            for r, v in zip(records, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[start : start + 256], wait=True)

    def count(self, doc_id: str | None = None) -> int:
        query_filter = self._doc_filter(doc_id) if doc_id else None
        return self.client.count(self.collection, count_filter=query_filter, exact=True).count

    def delete_document(self, doc_id: str) -> int:
        """Erase every chunk of one document (Art. 17 DSGVO – right to erasure)."""
        n = self.count(doc_id)
        if n:
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(filter=self._doc_filter(doc_id)),
                wait=True,
            )
        return n

    def search(self, vector: list[float], top_k: int, min_score: float = 0.0) -> list[SearchHit]:
        response = self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score or None,
        )
        return [self._to_hit(p.score, p.payload or {}) for p in response.points]

    @staticmethod
    def _to_hit(score: float, payload: dict[str, Any]) -> SearchHit:
        return SearchHit(
            score=float(score),
            doc_id=payload.get("doc_id", ""),
            source=payload.get("source", ""),
            page=payload.get("page"),
            chunk_index=int(payload.get("chunk_index", 0)),
            text=payload.get("text", ""),
        )

    def list_documents(self) -> list[DocumentInfo]:
        docs: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": 0})
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=512,
                offset=offset,
                with_payload=["doc_id", "source", "ingested_at"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                doc = docs[payload.get("doc_id", "")]
                doc["chunks"] += 1
                doc["source"] = payload.get("source", "")
                doc["ingested_at"] = payload.get("ingested_at", "")
            if offset is None:
                break
        return sorted(
            (DocumentInfo(doc_id=k, **v) for k, v in docs.items()), key=lambda d: d.source
        )
