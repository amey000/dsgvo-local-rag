"""System prompt and citation handling for source-grounded answers."""

from __future__ import annotations

import re

from .vectorstore import SearchHit

NO_ANSWER = "Die bereitgestellten Dokumente enthalten dazu keine Informationen."

SYSTEM_PROMPT = f"""Du bist ein sachlicher Assistent, der Fragen ausschließlich anhand interner \
Dokumente beantwortet.

Regeln:
1. Nutze NUR die nummerierten Quellen in der Nachricht. Verwende kein Vorwissen.
2. Belege jede inhaltliche Aussage direkt mit der Quellennummer in eckigen Klammern, \
z. B. [1] oder [2][3].
3. Wenn die Quellen die Frage nicht beantworten, antworte exakt: "{NO_ANSWER}"
4. Erfinde keine Fakten, Zahlen, Fristen, Namen oder Quellen.
5. Platzhalter wie <PERSON_1>, <IBAN_1> oder <TELEFON_1> stehen für geschwärzte \
personenbezogene Daten. Übernimm sie unverändert und versuche niemals, die echten Daten zu \
erraten oder zu rekonstruieren.
6. Antworte präzise in der Sprache der Frage (standardmäßig Deutsch)."""


def format_source_label(hit: SearchHit) -> str:
    return f"{hit.source}, Seite {hit.page}" if hit.page else hit.source


def build_messages(question: str, hits: list[SearchHit]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i}] Quelle: {format_source_label(hit)}\n{hit.text}" for i, hit in enumerate(hits, 1)
    )
    user = f"Quellen:\n\n{context}\n\n---\nFrage: {question}\n\nAntwort mit Quellenangaben:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def extract_citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers cited in the answer, in order of first appearance; invalid ones dropped."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(answer):
        for number in re.split(r"\s*[,;]\s*", group):
            idx = int(number)
            if 1 <= idx <= n_sources and idx not in seen:
                seen.append(idx)
    return seen


def is_no_answer(answer: str) -> bool:
    return NO_ANSWER.casefold().rstrip(".") in answer.casefold()
