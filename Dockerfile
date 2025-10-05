# syntax=docker/dockerfile:1

# Base image with Python
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requeriments.txt ./requirements.txt

# Install python deps (except torch here)
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch from official index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . .

# Hugging Face Spaces provides $PORT
ENV PORT=7860 

# Expose for local run (HF ignores EXPOSE)
EXPOSE 7860

# Default command for HF Spaces: iniciar mediante start.py, que lee PORT en Python
CMD ["python", "start.py"]
