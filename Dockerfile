# =============================================================================
# Dockerfile - Acne Advisor AI Backend (FastAPI)
# Deploy lên Railway: builder = DOCKERFILE (xem railway.json)
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Thư viện hệ thống tối thiểu cho package có C-extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Railway inject $PORT lúc runtime — không hardcode 8000
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
