"""German-specific Presidio recognizers.

Presidio ships most predefined recognizers for English only, so we register both German
variants of the generic recognizers (IBAN, phone, e-mail, ...) and a few custom pattern
recognizers for identifiers that only exist in Germany (Steuer-ID, Rentenversicherungsnummer,
street addresses).
"""

from __future__ import annotations

import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)

LANGUAGE = "de"

_STREET_SUFFIX = r"(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer|chaussee|pfad)"
_STREET_WORD = r"(?:Straße|Strasse|Str\.|Weg|Allee|Platz|Gasse|Ring|Damm|Ufer|Chaussee)"
_NAME = r"[A-ZÄÖÜ][a-zäöüß]+"
_HOUSE_NO = r"\d{1,4}\s?[a-zA-Z]?(?:\s?[-/]\s?\d{1,4}[a-zA-Z]?)?"
_ZIP_CITY = rf"(?:,?\s+\d{{5}}\s+{_NAME}(?:[- ]{_NAME})?)?"


class GermanAddressRecognizer(PatternRecognizer):
    """Street + house number, optionally followed by ``PLZ Ort``.

    Matches e.g. ``Hauptstraße 5``, ``Berliner Allee 12a, 40212 Düsseldorf``, ``Am-Ring-Weg 3``.
    """

    PATTERNS = [  # noqa: RUF012 (Presidio convention)
        Pattern(
            "de_street_compound",
            rf"\b{_NAME}(?:-{_NAME})*{_STREET_SUFFIX}\s+{_HOUSE_NO}\b{_ZIP_CITY}",
            0.6,
        ),
        Pattern(
            "de_street_two_words",
            rf"\b{_NAME}\s+{_STREET_WORD}\s+{_HOUSE_NO}\b{_ZIP_CITY}",
            0.6,
        ),
    ]
    CONTEXT = ["adresse", "anschrift", "wohnhaft", "wohnt", "straße", "postanschrift"]  # noqa: RUF012

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DE_ADDRESS",
            supported_language=LANGUAGE,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


def is_valid_tax_id(digits: str) -> bool:
    """Checksum of the German *Steuerliche Identifikationsnummer* (ISO/IEC 7064, MOD 11,10)."""
    digits = re.sub(r"\D", "", digits)
    if len(digits) != 11 or digits[0] == "0":
        return False
    product = 10
    for char in digits[:10]:
        total = (int(char) + product) % 10 or 10
        product = (total * 2) % 11
    check = (11 - product) % 10
    return check == int(digits[10])


class GermanTaxIdRecognizer(PatternRecognizer):
    """Steuer-ID: 11 digits, first digit != 0, valid check digit.

    A valid checksum keeps the pattern score instead of jumping to 1.0, because ~10 % of all
    random 11-digit numbers pass the checksum; context words ("Steuer-ID") boost the score.
    """

    PATTERNS = [Pattern("de_tax_id", r"\b[1-9]\d(?:\s?\d{3}){3}\b", 0.4)]  # noqa: RUF012
    CONTEXT = [  # noqa: RUF012
        "steuer",
        "steuer-id",
        "steuerid",
        "steueridentifikationsnummer",
        "identifikationsnummer",
        "idnr",
        "tin",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DE_TAX_ID",
            supported_language=LANGUAGE,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return None if is_valid_tax_id(pattern_text) else False


class GermanSocialSecurityRecognizer(PatternRecognizer):
    """Rentenversicherungsnummer, e.g. ``12 150485 M 123``: area, birth date, initial, serial."""

    PATTERNS = [  # noqa: RUF012
        Pattern(
            "de_rvnr",
            r"\b\d{2}\s?(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}\s?[A-Z]\s?\d{3}\b",
            0.5,
        )
    ]
    CONTEXT = [  # noqa: RUF012
        "rentenversicherung",
        "rentenversicherungsnummer",
        "sozialversicherung",
        "sozialversicherungsnummer",
        "sv-nummer",
        "versicherungsnummer",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="DE_SOCIAL_SECURITY",
            supported_language=LANGUAGE,
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


def build_registry() -> RecognizerRegistry:
    """All recognizers used for German text."""
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    recognizers: list[EntityRecognizer] = [
        # Names, places and organisations come from the spaCy NER model (de_core_news_*).
        SpacyRecognizer(
            supported_language=LANGUAGE,
            supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
        ),
        IbanRecognizer(
            supported_language=LANGUAGE,
            context=["iban", "konto", "kontonummer", "bankverbindung", "überweisung"],
        ),
        PhoneRecognizer(
            supported_language=LANGUAGE,
            supported_regions=("DE", "AT", "CH"),
            context=["telefon", "tel", "mobil", "handy", "fax", "rufnummer", "erreichbar"],
        ),
        EmailRecognizer(supported_language=LANGUAGE, context=["e-mail", "email", "mail"]),
        CreditCardRecognizer(
            supported_language=LANGUAGE, context=["kreditkarte", "karte", "visa", "mastercard"]
        ),
        IpRecognizer(supported_language=LANGUAGE),
        GermanAddressRecognizer(),
        GermanTaxIdRecognizer(),
        GermanSocialSecurityRecognizer(),
    ]
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)
    return registry
