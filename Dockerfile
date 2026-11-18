# syntax=docker/dockerfile:1.7
#
# All models (spaCy de_core_news_md, multilingual-e5-large) are baked into the image at build
# time. At runtime the container has no internet access and runs fully offline.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------- builder
FROM base AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/models/huggingface

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

ARG SPACY_MODEL=de_core_news_md
RUN python -m spacy download ${SPACY_MODEL}

ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

# ---------------------------------------------------------------- runtime
FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    XDG_CACHE_HOME=/tmp/.cache

USER app
WORKDIR /home/app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "dsgvo_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
