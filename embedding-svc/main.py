"""embedding-svc: text -> vector over gRPC, using sentence-transformers.

Serves two ports from one asyncio loop:
  - gRPC (EMBEDDING_GRPC_PORT, default 50051): EmbeddingService.Embed, used by the gateway.
  - HTTP (EMBEDDING_HTTP_PORT, default 8001): FastAPI with /healthz (Docker
    healthcheck) and POST /embed (for poking at the service with curl).
"""
import asyncio
import logging
import os
import time

import grpc
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import embedding_pb2
import embedding_pb2_grpc

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GRPC_PORT = int(os.getenv("EMBEDDING_GRPC_PORT", "50051"))
HTTP_PORT = int(os.getenv("EMBEDDING_HTTP_PORT", "8001"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("embedding-svc")


class Embedder:
    def __init__(self, model_name: str):
        started = time.perf_counter()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        log.info("loaded %s (dim=%d) in %.1fs", model_name, self.dim, time.perf_counter() - started)

    async def embed(self, text: str) -> list[float]:
        # encode() is CPU-bound; running it in a worker thread keeps the event
        # loop (and the other port) responsive. torch releases the GIL while computing.
        # Normalized so cosine similarity in Qdrant equals the dot product.
        vector = await asyncio.to_thread(self.model.encode, text, normalize_embeddings=True)
        return vector.tolist()


class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def Embed(self, request, context):
        if not request.text.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text must not be empty")
        started = time.perf_counter()
        vector = await self.embedder.embed(request.text)
        log.debug("embedded %d chars in %.1fms", len(request.text), (time.perf_counter() - started) * 1000)
        return embedding_pb2.EmbedResponse(vector=vector, model=self.embedder.model_name)


class EmbedBody(BaseModel):
    text: str


def http_app(embedder: Embedder) -> FastAPI:
    app = FastAPI(title="embedding-svc")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": embedder.model_name, "dim": embedder.dim}

    @app.post("/embed")
    async def embed(body: EmbedBody):
        if not body.text.strip():
            raise HTTPException(400, "text must not be empty")
        return {"model": embedder.model_name, "vector": await embedder.embed(body.text)}

    return app


async def main() -> None:
    # Load before opening any port, so /healthz only succeeds once we can serve.
    embedder = Embedder(MODEL_NAME)

    grpc_server = grpc.aio.server()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingService(embedder), grpc_server)
    grpc_server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await grpc_server.start()
    log.info("gRPC listening on :%d, HTTP on :%d", GRPC_PORT, HTTP_PORT)

    # uvicorn owns SIGINT/SIGTERM; when it returns, drain gRPC and exit.
    http_server = uvicorn.Server(uvicorn.Config(http_app(embedder), host="0.0.0.0", port=HTTP_PORT, log_level="warning"))
    try:
        await http_server.serve()
    finally:
        await grpc_server.stop(grace=5)


if __name__ == "__main__":
    asyncio.run(main())
