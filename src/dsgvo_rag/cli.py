"""Command line interface: ``dsgvo-rag {ingest,query,redact,docs,delete,serve}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings


def _service():
    from .pipeline import build_service

    return build_service(get_settings())


def cmd_ingest(args: argparse.Namespace) -> int:
    from .loaders import iter_files

    files = iter_files([Path(p) for p in args.paths])
    if not files:
        print("Keine unterstützten Dateien gefunden (.pdf, .txt, .md).", file=sys.stderr)
        return 1
    svc = _service()
    for path in files:
        result = svc.ingest_path(path)
        entities = ", ".join(f"{k}={v}" for k, v in sorted(result.redacted_entities.items()))
        print(
            f"✓ {result.source}: {result.chunks} Chunks, doc_id={result.doc_id}"
            f"  | geschwärzt: {entities or '–'}"
        )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    result = _service().query(args.question, args.top_k)
    if args.json:
        from dataclasses import asdict

        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    if result.redacted_question != args.question:
        print(f"(Frage geschwärzt: {result.redacted_question})\n")
    print(result.answer, "\n")
    for s in result.sources:
        if s.cited:
            page = f", S. {s.page}" if s.page else ""
            print(f"  [{s.index}] {s.source}{page}  (Score {s.score:.3f})")
    for w in result.warnings:
        print(f"⚠ {w}")
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    from .redaction import Redactor

    s = get_settings()
    text = args.text if args.text is not None else sys.stdin.read()
    redactor = Redactor(s.spacy_model, s.pii_entities, s.pii_score_threshold, s.pii_allow_list)
    result = redactor.redact(text)
    print(result.text)
    print(json.dumps(result.entities, ensure_ascii=False), file=sys.stderr)
    return 0


def cmd_docs(_: argparse.Namespace) -> int:
    for d in _service().list_documents():
        print(f"{d.doc_id}  {d.chunks:4d} Chunks  {d.ingested_at}  {d.source}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    n = _service().delete_document(args.doc_id)
    print(f"{n} Chunks gelöscht." if n else "Dokument nicht gefunden.")
    return 0 if n else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("dsgvo_rag.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsgvo-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Dokumente schwärzen, vektorisieren und speichern")
    p.add_argument("paths", nargs="+", help="Dateien oder Verzeichnisse")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", help="Frage an die Dokumente stellen")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("redact", help="Text schwärzen (Argument oder stdin)")
    p.add_argument("text", nargs="?")
    p.set_defaults(func=cmd_redact)

    sub.add_parser("docs", help="Gespeicherte Dokumente auflisten").set_defaults(func=cmd_docs)

    p = sub.add_parser("delete", help="Dokument löschen (Art. 17 DSGVO)")
    p.add_argument("doc_id")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("serve", help="REST-API starten")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
