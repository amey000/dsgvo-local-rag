"""PII redaction: Presidio detects, we pseudonymise.

Every detected entity is replaced by a numbered placeholder such as ``<PERSON_1>``. Numbers are
consistent within one document (the same IBAN always becomes ``<IBAN_1>``), so the LLM can still
reason about "who did what" without ever seeing the real data. The value→placeholder mapping
lives only in memory for the duration of one document and is never persisted, i.e. the stored
text cannot be re-identified from the system itself.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePath

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .recognizers import LANGUAGE, build_registry

# German placeholder labels shown to the LLM and stored in the vector database.
LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "E_MAIL",
    "PHONE_NUMBER": "TELEFON",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "KREDITKARTE",
    "IP_ADDRESS": "IP_ADRESSE",
    "DE_ADDRESS": "ADRESSE",
    "DE_TAX_ID": "STEUER_ID",
    "DE_SOCIAL_SECURITY": "SV_NUMMER",
    "LOCATION": "ORT",
    "ORGANIZATION": "ORGANISATION",
}

PLACEHOLDER_RE = re.compile(r"<[A-Z_]+_\d+>")

# Entity types that come from the statistical spaCy NER model (as opposed to patterns).
NER_TYPES = {"PERSON", "LOCATION", "ORGANIZATION"}

_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def _name_key(token: str) -> str:
    """``Müller`` and ``mueller`` → ``mueller`` (used to match names in file names)."""
    return token.casefold().translate(_TRANSLIT).strip(".,;:()[]\"'")


@dataclass(frozen=True)
class RedactionResult:
    text: str
    entities: dict[str, int]  # entity type -> number of redacted occurrences

    @property
    def total(self) -> int:
        return sum(self.entities.values())


@dataclass
class Pseudonymizer:
    """Hands out stable placeholders per (entity type, value) within one scope (= document)."""

    _maps: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    _name_tokens: dict[str, str] = field(default_factory=dict)

    def placeholder(self, entity_type: str, value: str) -> str:
        mapping = self._maps[entity_type]
        key = re.sub(r"\s+", "", value).casefold()
        if key not in mapping:
            label = LABELS.get(entity_type, entity_type)
            mapping[key] = f"<{label}_{len(mapping) + 1}>"
            if entity_type == "PERSON":
                for token in re.split(r"[\s\-]+", value):
                    if len(token) >= 3:
                        self._name_tokens.setdefault(_name_key(token), mapping[key])
        return mapping[key]

    def known_name(self, token: str) -> str | None:
        """Placeholder of a person whose first/last name equals ``token`` (if seen before)."""
        return self._name_tokens.get(_name_key(token))


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    entity_type: str
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def merge_spans(results: list[RecognizerResult]) -> list[_Span]:
    """Union of overlapping detections (privacy-first: never leak a partially covered value).

    The merged span takes the entity type of its longest member (ties: higher score), so
    ``Hauptstraße 5, 10115 Berlin`` (DE_ADDRESS) wins over ``Berlin`` (LOCATION).
    """
    spans = sorted(
        (_Span(r.start, r.end, r.entity_type, r.score) for r in results),
        key=lambda s: (s.start, -s.end),
    )
    merged: list[_Span] = []
    for span in spans:
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            dominant = max(last, span, key=lambda s: (s.length, s.score))
            merged[-1] = _Span(
                last.start,
                max(last.end, span.end),
                dominant.entity_type,
                max(last.score, span.score),
            )
        else:
            merged.append(span)
    return merged


class Redactor:
    def __init__(
        self,
        spacy_model: str = "de_core_news_md",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
        allow_list: list[str] | None = None,
    ) -> None:
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": LANGUAGE, "model_name": spacy_model}],
                "ner_model_configuration": {
                    "model_to_presidio_entity_mapping": {
                        "PER": "PERSON",
                        "PERSON": "PERSON",
                        "LOC": "LOCATION",
                        "GPE": "LOCATION",
                        "ORG": "ORGANIZATION",
                    },
                    "labels_to_ignore": ["MISC"],
                },
            }
        ).create_engine()
        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=build_registry(),
            supported_languages=[LANGUAGE],
        )
        self.entities = entities
        self.score_threshold = score_threshold
        self.allow_list = allow_list or []

    def analyze(self, text: str) -> list[RecognizerResult]:
        if not text.strip():
            return []
        results = self._analyzer.analyze(
            text=text,
            language=LANGUAGE,
            entities=self.entities,
            score_threshold=self.score_threshold,
            allow_list=self.allow_list or None,
        )
        # Never re-redact our own placeholders (e.g. when a query is redacted twice).
        protected = [m.span() for m in PLACEHOLDER_RE.finditer(text)]
        cleaned = []
        for r in results:
            if any(r.start >= s and r.end <= e for s, e in protected):
                continue
            if r.entity_type in NER_TYPES:
                r = self._trim_ner_span(text, r)
                if r is None:
                    continue
            cleaned.append(r)
        return cleaned

    @staticmethod
    def _trim_ner_span(text: str, r: RecognizerResult) -> RecognizerResult | None:
        """NER sometimes swallows numbers, e.g. ``Tel. 030 2312…`` or ``Anna Schmidt 030 …``.

        Names contain no digits: cut the span at the first digit/``@``. Keep the rest only if it
        still looks like a name (≥ 2 capitalised words), so ``Anna Schmidt`` survives while the
        false positive ``Tel.`` is dropped. The number itself is covered by pattern recognizers.
        """
        value = text[r.start : r.end]
        cut = re.search(r"[\d@]", value)
        if not cut:
            return r
        head = value[: cut.start()].rstrip(" ,;:(/-")
        if len(re.findall(r"\b[A-ZÄÖÜ][\wäöüß-]+", head)) < 2:
            return None
        return RecognizerResult(r.entity_type, r.start, r.start + len(head), r.score)

    def redact(self, text: str, pseudonymizer: Pseudonymizer | None = None) -> RedactionResult:
        """Replace all PII in ``text``. Pass a shared ``pseudonymizer`` for document-wide labels."""
        return self._replace(text, self.analyze(text), pseudonymizer or Pseudonymizer())

    @staticmethod
    def _replace(
        text: str, results: list[RecognizerResult], pseudonymizer: Pseudonymizer
    ) -> RedactionResult:
        spans = merge_spans(results)
        counts: Counter[str] = Counter()
        parts: list[str] = []
        cursor = 0
        for span in spans:
            parts.append(text[cursor : span.start])
            parts.append(pseudonymizer.placeholder(span.entity_type, text[span.start : span.end]))
            counts[span.entity_type] += 1
            cursor = span.end
        parts.append(text[cursor:])
        return RedactionResult(text="".join(parts), entities=dict(counts))

    def redact_filename(self, filename: str, pseudonymizer: Pseudonymizer | None = None) -> str:
        """``Bewerbung_Max_Mustermann.pdf`` → ``Bewerbung <PERSON_1>.pdf``.

        File names end up in citations, so they must not leak PII either. NER is unreliable on
        short, lower-case file names, therefore a name is redacted if
        (a) the token is a first/last name already found in the document body (pass the
            document's ``pseudonymizer``; ``mueller`` matches ``Müller``), or
        (b) NER finds a multi-word name (``Max Mustermann``), or
        (c) a pattern recognizer fires (e-mail, IBAN, phone, ...).
        """
        pseudonymizer = pseudonymizer or Pseudonymizer()
        path = PurePath(filename)
        original = re.sub(r"_+", " ", path.stem).strip()
        readable = re.sub(
            r"[^\W\d_]{3,}", lambda m: pseudonymizer.known_name(m.group()) or m.group(), original
        )
        # "<PERSON_1>-<PERSON_1>" (first and last name) → "<PERSON_1>"
        readable = re.sub(r"(<[A-Z_]+_\d+>)(?:[\s\-.]+\1)+", r"\1", readable)
        results = [
            r
            for r in self.analyze(readable)
            if r.entity_type not in NER_TYPES or len(readable[r.start : r.end].split()) >= 2
        ]
        redacted = self._replace(readable, results, pseudonymizer).text
        if redacted == original:
            return path.name  # nothing found → keep the original spelling
        return f"{redacted}{path.suffix}"
