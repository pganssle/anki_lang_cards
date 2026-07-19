# syntax=docker/dockerfile:1.7

FROM python:3.13-slim-bookworm@sha256:fcbd8dfc2605ba7c2eca646846c5e892b2931e41f6227985154a596f26ab8ed7 AS test-dependencies

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN python -m pip install --no-cache-dir uv==0.11.28

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group calibre-e2e --no-install-project \
    && rm -rf /root/.cache/uv


FROM test-dependencies AS cards-test-dependencies

RUN apt-get update \
    && apt-get install --yes --no-install-recommends g++ libhunspell-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --extra books --extra testing --no-install-project \
    && rm -rf /root/.cache/uv

COPY tests/fakes/llm-fake-cards ./tests/fakes/llm-fake-cards
RUN uv pip install --python /opt/venv/bin/python --no-cache-dir \
    ./tests/fakes/llm-fake-cards


FROM python:3.13-slim-bookworm@sha256:fcbd8dfc2605ba7c2eca646846c5e892b2931e41f6227985154a596f26ab8ed7 AS cards-e2e

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    HOME=/tmp/test-user

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libhunspell-1.7-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /tmp/test-user --uid 10001 test-user

COPY --from=cards-test-dependencies /opt/venv /opt/venv
WORKDIR /app
COPY --chown=test-user:test-user pyproject.toml ./
COPY --chown=test-user:test-user src ./src
COPY --chown=test-user:test-user tests ./tests

USER test-user
CMD ["pytest", "-p", "no:cacheprovider", "-m", "e2e", "-q", "tests/test_card_generation.py"]


FROM python:3.13-slim-bookworm@sha256:fcbd8dfc2605ba7c2eca646846c5e892b2931e41f6227985154a596f26ab8ed7 AS calibre-e2e

ARG CALIBRE_VERSION=6.13.0+repack-2+deb12u9
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    CALIBRE_CONFIG_DIRECTORY=/tmp/calibre-config \
    HOME=/tmp/test-user

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        calibre=${CALIBRE_VERSION} \
        calibre-bin=${CALIBRE_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /tmp/test-user --uid 10001 test-user

COPY --from=test-dependencies /opt/venv /opt/venv
WORKDIR /app
COPY --chown=test-user:test-user pyproject.toml ./
COPY --chown=test-user:test-user src ./src
COPY --chown=test-user:test-user tests ./tests

USER test-user
CMD ["pytest", "-p", "no:cacheprovider", "-m", "e2e", "-q", "tests/test_annotation_extractor.py"]
