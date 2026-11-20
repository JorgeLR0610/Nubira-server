# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    OMP_NUM_THREADS=1 \
    NUBIRA_INFERENCE_CONCURRENCY=1 \
    NUBIRA_REQUIRE_POSTGRES=1 \
    NUBIRA_MONITOR_INTERVAL=900 \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Use the CPU index exclusively, so PyPI cannot select a CUDA build.
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu

COPY *.py ./
COPY artifacts/ ./artifacts/
RUN python fetch_artifacts.py --check

HEALTHCHECK --start-period=30s --timeout=3s --interval=30s \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ["PORT"] + "/health", timeout=2)'

EXPOSE 8000
CMD ["python", "start.py"]
