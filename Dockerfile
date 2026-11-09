# syntax=docker/dockerfile:1.7
#
# Linux image: faster-whisper (CPU by default) + optional pyannote diarization,
# served as an HTTP API + MCP endpoint on :8000.
#
#   docker build -t transcribation .
#   docker build --build-arg WITH_DIARIZATION=false -t transcribation:slim .   # no torch, ~4x smaller
#
ARG PYTHON_VERSION=3.12
ARG DENO_VERSION=2.6.0

# yt-dlp needs a JavaScript runtime to solve YouTube's player challenges
FROM denoland/deno:bin-${DENO_VERSION} AS deno

FROM python:${PYTHON_VERSION}-slim

ARG WITH_DIARIZATION=true
# CPU-only torch wheels (the default PyPI torch for Linux bundles ~3 GB of CUDA libs).
# For an NVIDIA GPU build use e.g. https://download.pytorch.org/whl/cu128 (see compose.gpu.yaml).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSCRIBE_DATA_DIR=/data \
    TRANSCRIBE_HOST=0.0.0.0 \
    TRANSCRIBE_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deno /deno /usr/local/bin/deno

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data /cache/huggingface \
    && chown -R app:app /data /cache

WORKDIR /app

# 1) dependencies (cached layer: only invalidated by pyproject.toml changes)
COPY pyproject.toml README.md LICENSE ./
COPY src/transcribation/__init__.py src/transcribation/__init__.py
RUN extras="server,faster-whisper"; \
    if [ "$WITH_DIARIZATION" = "true" ]; then extras="$extras,diarize"; fi; \
    pip install --extra-index-url "$TORCH_INDEX_URL" ".[$extras]"

# 2) the application itself
COPY src ./src
RUN pip install --no-deps --force-reinstall .

# cuBLAS/cuDNN from the nvidia-* wheels (GPU builds only; harmless otherwise)
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

USER app
EXPOSE 8000
VOLUME ["/data", "/cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["transcribation-server"]
