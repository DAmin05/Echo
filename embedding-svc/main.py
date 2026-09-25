"""embedding-svc: the ML models behind Echo's cache, over gRPC.

  - Embed: bi-encoder (all-MiniLM-L6-v2) text -> vector, for nearest-neighbour search.
  - ScoreDuplicates: cross-encoder (quora-roberta-base) that reads a query and a
    candidate together and scores whether they ask the same question. It catches
    the swaps vector similarity misses ("miles to km" vs "km to miles").

Serves two ports from one asyncio loop:
  - gRPC (EMBEDDING_GRPC_PORT, default 50051): used by the gateway.
  - HTTP (EMBEDDING_HTTP_PORT, default 8001): FastAPI with /healthz (Docker
    healthcheck), POST /embed and POST /score-duplicates (for curl and the eval script).
"""
import asyncio
import logging
import os
import time

import grpc
import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

import embedding_pb2
import embedding_pb2_grpc

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VERIFIER_NAME = os.getenv("VERIFIER_MODEL", "cross-encoder/quora-roberta-base")
GRPC_PORT = int(os.getenv("EMBEDDING_GRPC_PORT", "50051"))
HTTP_PORT = int(os.getenv("EMBEDDING_HTTP_PORT", "8001"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("embedding-svc")
tracer = trace.get_tracer("embedding-svc")


def init_tracing() -> None:
    """Export spans over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set (as the
    Rust services do). The gRPC server instrumentation reads the caller's
    `traceparent`, so these spans join the gateway's trace."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorServer
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "embedding-svc"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    GrpcAioInstrumentorServer().instrument()
    log.info("exporting traces to %s", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])


class Embedder:
    def __init__(self, model_name: str):
        started = time.perf_counter()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        # The first inference is several times slower (lazy init inside torch);
        # pay it here, before /healthz passes, not on the first real request.
        self.model.encode("warm-up")
        log.info("loaded %s (dim=%d) in %.1fs", model_name, self.dim, time.perf_counter() - started)

    async def embed(self, text: str) -> list[float]:
        # encode() is CPU-bound; running it in a worker thread keeps the event
        # loop (and the other port) responsive. torch releases the GIL while computing.
        # Normalized so cosine similarity in Qdrant equals the dot product.
        with tracer.start_as_current_span("minilm.encode") as span:
            span.set_attribute("text.chars", len(text))
            vector = await asyncio.to_thread(self.model.encode, text, normalize_embeddings=True)
        return vector.tolist()


class Verifier:
    def __init__(self, model_name: str):
        started = time.perf_counter()
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu")
        self.model.predict([("warm-up", "warm-up")])  # see Embedder.__init__
        log.info("loaded %s in %.1fs", model_name, time.perf_counter() - started)

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        # One batched forward pass over all pairs; sigmoid output in [0, 1].
        with tracer.start_as_current_span("cross_encoder.predict") as span:
            span.set_attribute("pairs", len(candidates))
            scores = await asyncio.to_thread(self.model.predict, [(query, c) for c in candidates])
            span.set_attribute("max_score", float(max(scores)))
        return [float(s) for s in scores]


class EmbeddingService(embedding_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, embedder: Embedder, verifier: Verifier):
        self.embedder = embedder
        self.verifier = verifier

    async def Embed(self, request, context):
        if not request.text.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text must not be empty")
        started = time.perf_counter()
        vector = await self.embedder.embed(request.text)
        log.debug("embedded %d chars in %.1fms", len(request.text), (time.perf_counter() - started) * 1000)
        return embedding_pb2.EmbedResponse(vector=vector, model=self.embedder.model_name)

    async def ScoreDuplicates(self, request, context):
        if not request.query.strip():
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        scores = await self.verifier.score(request.query, list(request.candidates))
        return embedding_pb2.ScoreDuplicatesResponse(scores=scores, model=self.verifier.model_name)


class EmbedBody(BaseModel):
    text: str


class ScoreBody(BaseModel):
    query: str
    candidates: list[str]


def http_app(embedder: Embedder, verifier: Verifier) -> FastAPI:
    app = FastAPI(title="embedding-svc")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": embedder.model_name, "dim": embedder.dim, "verifier": verifier.model_name}

    @app.post("/embed")
    async def embed(body: EmbedBody):
        if not body.text.strip():
            raise HTTPException(400, "text must not be empty")
        return {"model": embedder.model_name, "vector": await embedder.embed(body.text)}

    @app.post("/score-duplicates")
    async def score_duplicates(body: ScoreBody):
        return {"model": verifier.model_name, "scores": await verifier.score(body.query, body.candidates)}

    return app


async def main() -> None:
    # Before creating the gRPC server, so the instrumentation wraps it.
    init_tracing()
    # Load before opening any port, so /healthz only succeeds once we can serve.
    embedder = Embedder(MODEL_NAME)
    verifier = Verifier(VERIFIER_NAME)

    grpc_server = grpc.aio.server()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingService(embedder, verifier), grpc_server)
    grpc_server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await grpc_server.start()
    log.info("gRPC listening on :%d, HTTP on :%d", GRPC_PORT, HTTP_PORT)

    # uvicorn owns SIGINT/SIGTERM; when it returns, drain gRPC and exit.
    http_server = uvicorn.Server(uvicorn.Config(http_app(embedder, verifier), host="0.0.0.0", port=HTTP_PORT, log_level="warning"))
    try:
        await http_server.serve()
    finally:
        await grpc_server.stop(grace=5)


if __name__ == "__main__":
    asyncio.run(main())
