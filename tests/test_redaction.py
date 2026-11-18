import pytest
from presidio_analyzer import RecognizerResult

from dsgvo_rag.redaction import PLACEHOLDER_RE, Pseudonymizer, Redactor, is_valid_tax_id
from dsgvo_rag.redaction.engine import merge_spans


@pytest.mark.parametrize(
    ("text", "secret", "label"),
    [
        ("Sehr geehrte Frau Anna Schmidt, vielen Dank.", "Anna Schmidt", "<PERSON_1>"),
        ("Herr Jürgen Müller-Lüdenscheidt ist zuständig.", "Müller-Lüdenscheidt", "<PERSON_1>"),
        ("Bitte an DE89 3704 0044 0532 0130 00 überweisen.", "DE89 3704", "<IBAN_1>"),
        ("Bitte an DE89370400440532013000 überweisen.", "DE8937040044", "<IBAN_1>"),
        ("Österreich: AT61 1904 3002 3457 3201.", "AT61 1904", "<IBAN_1>"),
        ("Rufen Sie uns an: 030 23125000.", "23125000", "<TELEFON_1>"),
        ("Mobil erreichbar unter +49 171 2345678.", "2345678", "<TELEFON_1>"),
        ("Schreiben Sie an max.muster@example.com.", "max.muster@", "<E_MAIL_1>"),
        ("Wohnhaft in der Hauptstraße 5, 10115 Berlin.", "Hauptstraße 5", "<ADRESSE_1>"),
        ("Neue Anschrift: Berliner Allee 12a.", "Berliner Allee 12a", "<ADRESSE_1>"),
        ("Ihre Steuer-ID lautet 86 095 742 719.", "86 095 742 719", "<STEUER_ID_1>"),
        ("Rentenversicherungsnummer 12 150485 M 123.", "150485", "<SV_NUMMER_1>"),
        ("Kreditkarte 4111 1111 1111 1111 belastet.", "4111", "<KREDITKARTE_1>"),
        ("Login von IP 192.168.10.5 erkannt.", "192.168", "<IP_ADRESSE_1>"),
    ],
)
def test_detects_german_pii(redactor: Redactor, text: str, secret: str, label: str) -> None:
    result = redactor.redact(text)
    assert secret not in result.text
    assert label in result.text


def test_non_personal_text_is_untouched(redactor: Redactor) -> None:
    text = (
        "Die Kündigungsfrist beträgt drei Monate zum Quartalsende. Der Urlaubsanspruch "
        "beträgt 30 Tage, die Verpflegungspauschale 14 Euro."
    )
    result = redactor.redact(text)
    assert result.text == text
    assert result.total == 0


def test_pseudonyms_are_consistent_within_a_document(redactor: Redactor) -> None:
    text = (
        "Anna Schmidt beauftragte Peter Klein. Später rief Anna Schmidt erneut an. "
        "IBAN DE89 3704 0044 0532 0130 00, also DE89370400440532013000."
    )
    result = redactor.redact(text)
    assert result.text.count("<PERSON_1>") == 2
    assert "<PERSON_2>" in result.text
    assert result.text.count("<IBAN_1>") == 2  # spacing variants map to the same placeholder
    assert result.entities["PERSON"] == 3


def test_shared_pseudonymizer_spans_pages(redactor: Redactor) -> None:
    pseudo = Pseudonymizer()
    page1 = redactor.redact("Vertrag mit Anna Schmidt.", pseudo)
    page2 = redactor.redact("Unterschrift: Peter Klein und Anna Schmidt.", pseudo)
    assert "<PERSON_1>" in page1.text
    assert "<PERSON_2>" in page2.text and "<PERSON_1>" in page2.text


def test_redaction_is_idempotent(redactor: Redactor) -> None:
    once = redactor.redact("Herr Thomas Becker, Tel. 030 23125000").text
    assert redactor.redact(once).text == once
    assert PLACEHOLDER_RE.findall(once) == ["<PERSON_1>", "<TELEFON_1>"]


def test_allow_list(settings) -> None:
    redactor = Redactor(settings.spacy_model, settings.pii_entities, allow_list=["Anna Schmidt"])
    assert "Anna Schmidt" in redactor.redact("Produktname: Anna Schmidt").text


def test_filename_redaction(redactor: Redactor) -> None:
    assert redactor.redact_filename("Bewerbung_Max_Mustermann.pdf") == "Bewerbung <PERSON_1>.pdf"
    assert redactor.redact_filename("urlaubsrichtlinie-2026.md") == "urlaubsrichtlinie-2026.md"
    assert redactor.redact_filename("urlaub.txt") == "urlaub.txt"
    assert redactor.redact_filename("kontakt_a.b@example.com.txt").startswith("kontakt <E_MAIL_1>")


def test_filename_uses_names_from_document_body(redactor: Redactor) -> None:
    pseudo = Pseudonymizer()
    redactor.redact("Absender: Herr Jürgen Müller aus Köln.", pseudo)
    # lower-case, transliterated surname is only recognisable via the document body
    assert redactor.redact_filename("beschwerde_mueller.txt", pseudo) == "beschwerde <PERSON_1>.txt"
    assert redactor.redact_filename("juergen-mueller.pdf", pseudo) == "<PERSON_1>.pdf"


def test_ner_spans_with_digits_are_ignored(redactor: Redactor) -> None:
    result = redactor.redact("Herr Thomas Becker, Tel. 030 23125000")
    assert result.text == "Herr <PERSON_1>, Tel. <TELEFON_1>"


def test_ner_span_is_trimmed_not_dropped() -> None:
    from presidio_analyzer import RecognizerResult as R

    text = "Kontakt Anna Schmidt 030 23125000"
    trimmed = Redactor._trim_ner_span(text, R("PERSON", 8, len(text), 0.85))
    assert trimmed is not None and text[trimmed.start : trimmed.end] == "Anna Schmidt"
    assert Redactor._trim_ner_span("Tel. 030 1", R("PERSON", 0, 10, 0.85)) is None


def test_tax_id_checksum() -> None:
    assert is_valid_tax_id("86095742719")
    assert is_valid_tax_id("86 095 742 719")
    assert not is_valid_tax_id("86095742718")  # wrong check digit
    assert not is_valid_tax_id("06095742719")  # leading zero
    assert not is_valid_tax_id("1234")


def test_random_11_digit_number_with_bad_checksum_is_kept(redactor: Redactor) -> None:
    text = "Bestellnummer 86095742718 wurde versandt."
    assert redactor.redact(text).text == text


def test_overlapping_spans_are_merged_privacy_first() -> None:
    results = [
        RecognizerResult("LOCATION", 20, 26, 0.85),  # "Berlin"
        RecognizerResult("DE_ADDRESS", 0, 26, 0.6),  # "Hauptstraße 5, 10115 Berlin"
        RecognizerResult("PERSON", 30, 40, 0.85),
    ]
    spans = merge_spans(results)
    assert [(s.start, s.end, s.entity_type) for s in spans] == [
        (0, 26, "DE_ADDRESS"),
        (30, 40, "PERSON"),
    ]
