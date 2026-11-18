"""REST API (FastAPI). Interactive docs at /docs."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .loaders import UnsupportedFileTypeError
from .pipeline import RagService, build_service

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Wie viele Urlaubstage?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class RedactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


def get_service(request: Request) -> RagService:
    return request.app.state.service


Service = Annotated[RagService, Depends(get_service)]


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        if getattr(app.state, "service", None) is None:
            log.info("Loading models (Presidio/spaCy, multilingual-e5-large) ...")
            app.state.service = build_service(get_settings())
        yield

    app = FastAPI(
        title="DSGVO-konformes lokales RAG",
        version=__version__,
        description=(
            "Lokale Retrieval-Augmented-Generation mit automatischer Schwärzung "
            "personenbezogener Daten vor der Vektorisierung."
        ),
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health", tags=["Betrieb"])
    def health(svc: Service) -> dict:
        try:
            svc.store.client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False
        is_ready = getattr(svc.llm, "is_ready", None)
        llm_ok = bool(is_ready()) if callable(is_ready) else True
        return {
            "status": "ok" if qdrant_ok and llm_ok else "degraded",
            "qdrant": qdrant_ok,
            "llm": llm_ok,
            "llm_model": svc.llm.model,
            "embedding_model": svc.settings.embedding_model,
        }

    @app.post("/redact", tags=["Datenschutz"])
    def redact(body: RedactRequest, svc: Service) -> dict:
        """Zeigt, wie ein Text vor der Speicherung geschwärzt wird (nichts wird gespeichert)."""
        result = svc.redactor.redact(body.text)
        return {"text": result.text, "entities": result.entities}

    @app.post("/documents", tags=["Dokumente"])
    def upload_documents(files: Annotated[list[UploadFile], File()], svc: Service) -> list[dict]:
        limit = svc.settings.max_upload_mb * 1024 * 1024
        results = []
        for upload in files:
            data = upload.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"Datei größer als {svc.settings.max_upload_mb} MB")
            try:
                results.append(asdict(svc.ingest_bytes(data, upload.filename or "upload.txt")))
            except UnsupportedFileTypeError as exc:
                raise HTTPException(415, str(exc)) from exc
        return results

    @app.get("/documents", tags=["Dokumente"])
    def list_documents(svc: Service) -> list[dict]:
        return [asdict(d) for d in svc.list_documents()]

    @app.delete("/documents/{doc_id}", tags=["Datenschutz"])
    def delete_document(doc_id: str, svc: Service) -> dict:
        """Löscht alle Chunks eines Dokuments (Recht auf Löschung, Art. 17 DSGVO)."""
        deleted = svc.delete_document(doc_id)
        if not deleted:
            raise HTTPException(404, "Dokument nicht gefunden")
        return {"doc_id": doc_id, "deleted_chunks": deleted}

    @app.post("/query", tags=["RAG"])
    def query(body: QueryRequest, svc: Service) -> dict:
        try:
            return asdict(svc.query(body.question, body.top_k))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


app = create_app()
