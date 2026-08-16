FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS builder

RUN python -m pip install --no-cache-dir uv==0.11.29

WORKDIR /app/processor/service

COPY processor/runtime/pyproject.toml processor/runtime/models.toml processor/runtime/languages.toml /app/processor/runtime/
COPY processor/runtime/src /app/processor/runtime/src
COPY processor/service/pyproject.toml processor/service/uv.lock ./
COPY processor/service/src ./src

RUN uv sync --frozen --no-dev --no-editable

FROM builder AS test

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libopus0 \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --frozen --no-editable

COPY processor/service/tests ./tests
COPY integration/fixtures /app/integration/fixtures
COPY processor/service/rules ./rules
COPY processor/service/hypercorn.toml ./hypercorn.toml

CMD ["/app/processor/service/.venv/bin/pytest", "-q", "-p", "no:cacheprovider"]

FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 libopus0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 voicesniffer \
    && useradd --uid 10001 --gid voicesniffer --no-create-home --shell /usr/sbin/nologin voicesniffer

WORKDIR /app

COPY --from=builder /app/processor/service/.venv /app/.venv
COPY processor/service/hypercorn.toml /app/hypercorn.toml
COPY processor/service/rules /app/rules

RUN install -d -o 10001 -g 10001 -m 0775 /models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VOICESNIFFER_HYPERCORN_CONFIG=/app/hypercorn.toml \
    VOICESNIFFER_MODEL_ID=parakeet-tdt-0.6b-v3-fp32 \
    VOICESNIFFER_MODELS_DIR=/models \
    VOICESNIFFER_RULES_DIR=/app/rules

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()"]

ENTRYPOINT ["python", "-m", "voicesniffer_processor"]
