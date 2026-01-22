# syntax=docker/dockerfile:1.4
# Optimized Multi-stage Railway Deployment (CPU-only)
FROM python:3.12-slim AS builder

# Avoid interactive prompts and reduce build layers
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install build dependencies with minimal packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated virtual environment for clean package isolation
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy pinned requirements file (CPU-only PyTorch, no duplicates)
COPY requirements-railway-pinned.txt /tmp/requirements-railway-pinned.txt

# Single-pass installation: upgrade pip and install all packages efficiently
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements-railway-pinned.txt

# Runtime stage: minimal footprint with only necessary dependencies
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install minimal runtime libraries required by ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy complete virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables for Railway deployment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV USE_GPU=false
ENV PYTHONPATH=/app

WORKDIR /app

# Copy application code (filtered by .dockerignore)
COPY . .

# Expose port
EXPOSE 8000

# Use shell form to support Railway's PORT environment variable
CMD ["/bin/sh", "-c", "python -m chainlit run chainlit-app/app.py --host 0.0.0.0 --port ${PORT:-8000} --headless"]
