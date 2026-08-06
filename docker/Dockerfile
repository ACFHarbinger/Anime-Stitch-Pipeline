# Example multi-stage image for the Python module's service.
#
# Build context is the repository ROOT (see docker/docker-compose.yml:
#   build: { context: .., dockerfile: docker/Dockerfile }).
#
# Replace/extend with additional stages for other language modules
# (typescript/, rust/, go/, cpp/, kotlin/) as the project needs them.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY python/pyproject.toml python/uv.lock ./python/
RUN cd python && uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml

COPY python/ ./python/

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/python ./python

USER appuser
EXPOSE 8000

CMD ["python", "-m", "python.src.main"]
