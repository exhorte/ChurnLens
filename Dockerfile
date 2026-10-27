# Container image for the ChurnLens scoring API (service/main.py).
# Build:  docker build -t churnlens-api .
# Run:    docker run --rm -p 8000:8000 churnlens-api   ->   http://localhost:8000/docs
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURNLENS_MODELS_DIR=/app/models

WORKDIR /app

# Install dependencies first for better layer caching (API only: no Streamlit/Plotly).
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs to serve. The model is a pickle: build the image from a model you trained.
COPY src ./src
COPY service ./service
COPY models ./models

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 churn && chown -R churn /app
USER churn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
