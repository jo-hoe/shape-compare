# Minimal, multi-stage image for shape-compare.
#
# Stage 1 installs the package (and its runtime deps) into an isolated venv.
# Stage 2 copies only that venv into a slim runtime and runs as a non-root user.
# opencv-python-headless pulls no GUI libraries, so the runtime stays small.

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV=/opt/venv

WORKDIR /build

# Create an isolated venv and put it first on PATH.
RUN python -m venv "$VENV"
ENV PATH="$VENV/bin:$PATH"

# Copy only what the install needs first, to maximize layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install .


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV VENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /work
USER appuser

# The console script is the entry point; pass TARGET + candidate dir as args.
ENTRYPOINT ["shape-compare"]
CMD ["--help"]
