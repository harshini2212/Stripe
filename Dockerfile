# Stripify — FastAPI spend-management platform.
# Pinned to Python 3.11 so numpy / pandas / scikit-learn install from prebuilt
# wheels (no source compile), which is what breaks Nixpacks auto-detected builds.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source.
COPY . .

# Railway injects $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn comptroller.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
