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


FROM test-dependencies AS anki-test-dependencies

RUN uv sync --frozen --extra testing --no-install-project \
    && rm -rf /root/.cache/uv

RUN uv pip install --python /opt/venv/bin/python --no-cache-dir "aqt[qt68]==26.5"

ADD https://github.com/FooSoft/anki-connect/archive/4064fa142785975255457abd6a496015f5b71f38.tar.gz /tmp/anki-connect.tar.gz
RUN mkdir -p /opt/anki-connect \
    && tar -xzf /tmp/anki-connect.tar.gz -C /opt/anki-connect --strip-components=1 \
    && rm /tmp/anki-connect.tar.gz


FROM python:3.13-slim-bookworm@sha256:fcbd8dfc2605ba7c2eca646846c5e892b2931e41f6227985154a596f26ab8ed7 AS anki-e2e

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    HOME=/tmp/test-user \
    ANKI_CONNECT_PLUGIN=/opt/anki-connect/plugin \
    QTWEBENGINE_CHROMIUM_FLAGS="--no-sandbox" \
    ANKI_WAYLAND=0

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        xvfb xauth libgl1 libegl1 libopengl0 libglib2.0-0 libfontconfig1 libfreetype6 \
        libgssapi-krb5-2 libatomic1 libcups2 \
        libdbus-1-3 libnss3 libnspr4 libasound2 libxkbcommon0 libxkbcommon-x11-0 \
        libx11-xcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
        libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxcomposite1 libxdamage1 \
        libxrandr2 libxtst6 libxi6 libsm6 libice6 libxrender1 libxext6 libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /tmp/test-user --uid 10001 test-user

COPY --from=anki-test-dependencies /opt/venv /opt/venv
COPY --from=anki-test-dependencies /opt/anki-connect /opt/anki-connect
WORKDIR /app
COPY --chown=test-user:test-user pyproject.toml ./
COPY --chown=test-user:test-user src ./src
COPY --chown=test-user:test-user tests ./tests

USER test-user
CMD ["pytest", "-p", "no:cacheprovider", "-m", "e2e", "-q", "tests/test_anki_import.py", "tests/test_anki_connect.py"]


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
