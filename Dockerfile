# syntax=docker/dockerfile:1

# Base image with Python
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    OMP_NUM_THREADS=2 \
    NUBIRA_BIND_HOST=0.0.0.0

# Set workdir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install python deps (except torch here)
RUN pip install --no-cache-dir -r requirements-api.txt

# Install CPU-only PyTorch from official index
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . .

HEALTHCHECK --start-period=15s --timeout=3s --interval=30s \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ["PORT"] + "/health", timeout=2)'

# Hugging Face Spaces provides $PORT
ENV PORT=7860 

# Expose for local run (HF ignores EXPOSE)
EXPOSE 7860

# Default command for HF Spaces: iniciar mediante start.py, que lee PORT en Python
CMD ["python", "start.py"]
