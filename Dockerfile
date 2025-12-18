# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

# System deps (minimal). Add gcc/libpq-dev only if you build psycopg from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./
COPY uv.lock ./
# If you don't have uv.lock, comment that line out.

# Install deps into the image (no venv needed; uv will manage)
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy the rest of the repo (including prompts/)
COPY . .

# Chainlit default port
EXPOSE 8000

# Important: point Chainlit at your entry file
CMD ["uv", "run", "chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]

