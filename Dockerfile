# start from python base image
FROM python:3.10-slim

# ── Build args ────────────────────────────────────────────────────────
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────
# libpq-dev: PostgreSQL client headers (needed by psycopg2)
# gcc: C compiler (needed by some Python packages with C extensions)
# curl: health check in docker-compose
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────
# Copy requirements first for better Docker layer caching.
# If only application code changes, this layer is reused.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────
COPY . .

# ── Create directories expected at runtime ────────────────────────────
RUN mkdir -p logs results/prophet models/prophet/forecasts

# ── Non-root user for security ────────────────────────────────────────
RUN useradd -m -u 1000 demandsurge && chown -R demandsurge:demandsurge /app
USER demandsurge

# ── Default environment variables ─────────────────────────────────────
# These are overridden by docker-compose.yml or runtime --env-file
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_BASE_URL=http://api:8000

# ── Expose ports ──────────────────────────────────────────────────────
# 8000: FastAPI
# 8501: Streamlit
EXPOSE 8000 8501

# ── Default command ───────────────────────────────────────────────────
# Overridden per-service in docker-compose.yml
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

