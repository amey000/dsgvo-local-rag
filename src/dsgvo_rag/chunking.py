"""Recursive text splitter (paragraph → line → sentence → word) with character overlap."""

from __future__ import annotations

import re

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"(\w)-\n([a-zäöüß])", r"\1\2", text)  # PDF hyphenation: "Kündi-\ngung"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sep, *rest = separators
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, rest, size))
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ``overlap`` characters, starting at a word boundary (keeps placeholders intact)."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and " " in tail:
        tail = tail[tail.index(" ") + 1 :]
    return tail


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap muss kleiner als chunk_size sein")
    text = normalize(text)
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for piece in _split(text, SEPARATORS, chunk_size):
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = _overlap_tail(current, overlap)
        current = tail + piece if len(tail) + len(piece) <= chunk_size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
