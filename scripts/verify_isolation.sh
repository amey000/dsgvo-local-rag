#!/usr/bin/env bash
# Checks that the containers processing personal data cannot reach the internet,
# while still reaching each other inside the internal Docker network.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Netzwerk-Isolation des app-Containers =="
docker compose exec -T app python - <<'PY'
import sys
import urllib.request

def reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False

ok = True
for url in ["https://huggingface.co", "https://registry.ollama.ai", "http://1.1.1.1", "http://8.8.8.8"]:
    blocked = not reachable(url)
    ok &= blocked
    print(f"{'OK  ' if blocked else 'FAIL'} kein Internetzugriff: {url}")
for url in ["http://qdrant:6333/readyz", "http://ollama:11434/api/tags"]:
    up = reachable(url)
    ok &= up
    print(f"{'OK  ' if up else 'FAIL'} intern erreichbar:    {url}")
sys.exit(0 if ok else 1)
PY

echo "== Netzwerk-Definition =="
docker network inspect dsgvo-rag_rag_internal --format 'rag_internal internal={{.Internal}}'
echo "Ports, die am Host lauschen:"
docker compose ps --format '{{.Service}}: {{.Ports}}' | grep -v ': $' || true
