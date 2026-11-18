"""Text extraction for PDF, Markdown and plain-text files."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFileTypeError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    text: str
    number: int | None  # 1-based page number for PDFs, None for text files


def load_bytes(data: bytes, filename: str) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return [Page(page.extract_text() or "", i) for i, page in enumerate(reader.pages, start=1)]
    if suffix in {".txt", ".md"}:
        return [Page(data.decode("utf-8", errors="replace"), None)]
    raise UnsupportedFileTypeError(
        f"Dateityp '{suffix or filename}' wird nicht unterstützt "
        f"(erlaubt: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
    )


def iter_files(paths: list[Path]) -> list[Path]:
    """Expand directories recursively into supported files (sorted for reproducibility)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files
