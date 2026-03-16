# =============================================================================
# ClinicDx V1 — Middleware Container (FastAPI)
# =============================================================================
# Runs the FastAPI middleware service that routes between the OpenMRS frontend,
# the KB daemon, and the llama-server model.
# All logs are newline-delimited JSON on stdout.
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="ClinicDx" \
      version="1.0.0" \
      description="ClinicDx FastAPI Middleware"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY services/middleware/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source (as a proper Python package)
COPY services/middleware/ /app/service/

# The entrypoint runs the package as a module
ENV MODEL_SERVER_URL=http://model:8180 \
    KB_URL=http://kb:4276 \
    OPENMRS_URL="" \
    OPENMRS_USER="" \
    OPENMRS_PASSWORD="" \
    MIDDLEWARE_PORT=8080 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:8080/api/health || exit 1

CMD ["uvicorn", "service.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--no-access-log"]
