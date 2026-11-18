LLM_MODEL ?= llama3.1:8b
export LLM_MODEL

.PHONY: help setup build pull-model up down logs ingest ask docs verify-isolation install test lint format clean

help: ## Diese Hilfe anzeigen
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: build pull-model ## Einmalig: Image bauen und LLM herunterladen

build: ## App-Image bauen (lädt spaCy- und e5-Modell ins Image)
	docker compose build app

pull-model: ## LLM einmalig über das separate Egress-Netz laden
	docker compose --profile setup run --rm model-pull

up: ## Stack starten (API unter http://127.0.0.1:8000/docs)
	docker compose up -d --wait

down: ## Stack stoppen
	docker compose down

logs: ## Logs verfolgen
	docker compose logs -f app

ingest: ## Beispieldokumente aus data/samples indexieren
	docker compose exec app dsgvo-rag ingest /data/samples

ask: ## Frage stellen: make ask Q="Wie viele Urlaubstage habe ich?"
	docker compose exec app dsgvo-rag query "$(Q)"

docs: ## Indexierte Dokumente anzeigen
	docker compose exec app dsgvo-rag docs

verify-isolation: ## Prüfen, dass App/Qdrant/Ollama keinen Internetzugang haben
	./scripts/verify_isolation.sh

install: ## Lokale Entwicklungsumgebung
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/python -m spacy download de_core_news_md

test: ## Tests ausführen (ohne Docker)
	.venv/bin/pytest -m "not slow"

lint: ## Ruff
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:
	.venv/bin/ruff format . && .venv/bin/ruff check --fix .

clean: ## Stack inkl. Volumes (Vektoren + Modelle) entfernen
	docker compose down -v
