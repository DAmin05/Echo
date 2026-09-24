# Build context is the repo root (see docker-compose.yml) so proto/ is reachable.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY embedding-svc/requirements.txt .
RUN pip install -r requirements.txt

# Bake the model into the image: no download at startup, and the container
# works without internet access.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1 EMBEDDING_MODEL=${EMBEDDING_MODEL}

COPY proto/embedding.proto proto/
RUN python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/embedding.proto
COPY embedding-svc/main.py .

EXPOSE 50051 8001
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/healthz', timeout=2)"
CMD ["python", "main.py"]
