import pytest
from fastapi.testclient import TestClient

from dsgvo_rag.api import create_app


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_model"] == "fake-llm"


def test_redact_endpoint(client) -> None:
    body = client.post("/redact", json={"text": "Anruf von Anna Schmidt: 030 23125000"}).json()
    assert body["text"] == "Anruf von <PERSON_1>: <TELEFON_1>"
    assert body["entities"] == {"PERSON": 1, "PHONE_NUMBER": 1}


def test_document_lifecycle(client) -> None:
    files = [
        ("files", ("urlaub.txt", "Mitarbeitende haben 30 Tage Urlaub. Kontakt: Anna Schmidt.")),
        ("files", ("reise.md", "Die Verpflegungspauschale beträgt 14 Euro.")),
    ]
    uploaded = client.post("/documents", files=files)
    assert uploaded.status_code == 200, uploaded.text
    results = uploaded.json()
    assert [r["source"] for r in results] == ["urlaub.txt", "reise.md"]
    assert results[0]["redacted_entities"] == {"PERSON": 1}

    docs = client.get("/documents").json()
    assert {d["doc_id"] for d in docs} == {r["doc_id"] for r in results}

    answer = client.post("/query", json={"question": "Wie viele Urlaubstage?"}).json()
    assert answer["answer"] == "Es gibt 30 Urlaubstage [1]."
    assert answer["sources"][0]["cited"] is True

    doc_id = results[0]["doc_id"]
    deleted = client.delete(f"/documents/{doc_id}")
    assert deleted.json() == {"doc_id": doc_id, "deleted_chunks": 1}
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_unsupported_file_type(client) -> None:
    response = client.post("/documents", files=[("files", ("bild.png", b"\x89PNG"))])
    assert response.status_code == 415


def test_query_validation(client) -> None:
    assert client.post("/query", json={"question": "?"}).status_code == 422
