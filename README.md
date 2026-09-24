# Echo — LLM Gateway with Semantic Caching

Echo sits between clients and LLM providers. It embeds each prompt, checks a vector cache for semantically similar past requests, and returns the cached response on a hit, skipping the LLM call entirely. Clients talk to it with the OpenAI chat completions API, so any OpenAI SDK works by changing only the base URL.

## Status

- [x] Phase 0 — Infra bootstrap (Qdrant + Redis via Docker Compose)
- [x] Phase 1 — Monolith proof of concept
- [x] Phase 2 — Split into services (gRPC)
- [ ] Phase 3 — Concurrency correctness (in-flight dedup, threshold tuning)
- [ ] Phase 4 — Observability (OpenTelemetry + Jaeger)
- [ ] Phase 5 — Load test and results

## Architecture

```
                         ┌──────────────────────┐
  OpenAI-format HTTP ──▶ │ gateway (Rust, Axum) │  :8080  public API, orchestration
                         └──┬────────┬────────┬─┘
                     gRPC   │        │        │   gRPC
          ┌─────────────────┘        │        └──────────────────┐
          ▼                          ▼                           ▼
┌───────────────────┐   ┌──────────────────────┐   ┌───────────────────────────┐
│ embedding-svc     │   │ cache-svc (Rust)     │   │ provider-adapter (Rust)   │
│ (Python, FastAPI, │   │ :50052               │   │ :50053                    │
│  sentence-        │   │ Query / Store / Stats│   │ Generate                  │
│  transformers)    │   └──────┬─────────┬─────┘   └──┬──────────┬──────────┬──┘
│ :50051 Embed      │          │         │            │          │          │
└───────────────────┘      Qdrant      Redis      Anthropic    OpenAI     Ollama
                          (vectors)  (responses,
                                      TTLs, stats)
```

Per request, the gateway:

1. **Embeds** the final user message (embedding-svc → 384-dim `all-MiniLM-L6-v2` vector).
2. **Queries** the cache (cache-svc → nearest neighbour in Qdrant with cosine similarity ≥ 0.92, restricted to entries whose other request fields match exactly).
3. On a **hit**, returns the cached response (`x-echo-cache: hit`, `x-echo-similarity: 0.95xx`).
4. On a **miss**, **generates** the answer (provider-adapter → Claude / OpenAI / Ollama), **stores** it (cache-svc), and returns it (`x-echo-cache: miss`).

| Service | Language | Port(s) | Owns |
|---|---|---|---|
| `gateway` | Rust (Axum, tonic) | 8080 | Public API, cache key, orchestration. Holds no secrets. |
| `embedding-svc` | Python (sentence-transformers, grpcio, FastAPI) | 50051 gRPC, 8001 HTTP | The embedding model |
| `cache-svc` | Rust (tonic) | 50052 | Qdrant + Redis, similarity threshold, TTL, hit/miss stats |
| `provider-adapter` | Rust (tonic, reqwest) | 50053 | Provider API keys and request/response translation |
| `qdrant` | — | 6333 REST, 6334 gRPC | Vectors |
| `redis` | — | 6379 | Cached responses with TTLs |

The contracts are in [`proto/`](proto/). Rust stubs are generated at build time with [protox](https://crates.io/crates/protox), a pure-Rust protobuf compiler, so no `protoc` install is needed; Python stubs are generated in the embedding-svc image.

## Running

Requires Docker (Docker Desktop or OrbStack).

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY
docker compose up -d --build      # first build takes a few minutes
python3 scripts/phase1_demo.py    # paraphrase demo against localhost:8080
```

Any OpenAI SDK works:

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
client.chat.completions.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])
```

`GET /stats` returns running hit/miss counters. `POST localhost:8001/embed` with `{"text": "..."}` returns a raw embedding. Qdrant's dashboard is at http://localhost:6333/dashboard.

Stop with `docker compose down` (add `-v` to wipe stored vectors and responses).

### Choosing a provider

provider-adapter picks the provider from the model name: `claude-*` → Anthropic, `gpt-*` / `o3-*` etc. → OpenAI, anything else → Ollama. A prefix forces one: `ollama/llama3.2`, `anthropic/claude-sonnet-5`. Each provider is enabled by its setting in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_URL`). Requesting one that isn't configured returns a 400 that names the missing variable.

## Design decisions

### What makes two requests "the same"

- **Only the final user message is embedded.** That's the part expected to vary in wording.
- **Everything else must match exactly**: model, max_tokens, sampling params, and all earlier messages (system prompt, prior turns), serialized as canonical JSON and stored as a Qdrant payload filter. Embedding the whole conversation would let a long shared system prompt pull unrelated questions toward each other and cause false hits. It would also let the same question asked in a different conversation return an answer written for another context.

### Fail open

The cache only exists to save time and money, so its failures never fail a request. If embedding-svc or cache-svc is down or slow, the gateway logs a warning and calls the provider anyway (`x-echo-cache: bypass`). gRPC channels connect lazily with per-service deadlines (embedding 5 s, cache 2 s, provider 600 s), so the gateway starts and serves even while a dependency is still coming up.

### Only complete answers are cached

Only answers that ended naturally (`FINISH_REASON_STOP`) are stored. Refusals, `max_tokens` truncations and errors are returned to the client but never replayed to later callers.

### Storage layout (cache-svc)

Each cache entry has one UUID used in both stores:

| Store | Key | Holds |
|---|---|---|
| Qdrant | point `{id}` in `echo_cache` | embedding, payload `{prompt, params}` (`params` indexed) |
| Redis | hash `echo:entry:{id}` (TTL) | `response`, `prompt`, `hits` |

Redis decides whether an entry is still live. When its TTL expires, the matching Qdrant point is stale, and it gets deleted the next time a lookup lands on it. On a write, Redis is updated before Qdrant, so a failure partway through leaves a harmless orphaned vector rather than a response nothing can reach.

### Errors keep their meaning across hops

provider-adapter maps provider HTTP errors to gRPC codes (400 → `INVALID_ARGUMENT`, 401 → `UNAUTHENTICATED`, 429 → `RESOURCE_EXHAUSTED`, 5xx/529 → `UNAVAILABLE`, …), and the gateway maps them back to HTTP. A client gets the same status and message class it would have got from the provider directly, in OpenAI's `{"error": {...}}` shape.

### OpenAI → provider translation

The gateway converts OpenAI requests into a provider-neutral `GenerateRequest`, and provider-adapter translates that for each API:

| Field | Anthropic | OpenAI | Ollama (`/api/chat`) |
|---|---|---|---|
| system / developer messages | top-level `system` | `system` messages | `system` messages |
| max tokens | `max_tokens` (default 16000; required) | `max_completion_tokens` | `options.num_predict` |
| `stop` | `stop_sequences` | `stop` | `options.stop` |
| `temperature`, `top_p` | passed through¹ | passed through | `options.*` |

¹ Opus 5 rejects sampling params with a 400, which reaches the client unchanged. Opus 5 / Fable 5 requests also get `fallbacks: "default"`, so a safety-classifier refusal is retried server-side on a fallback model.

## Results

### Phase 1 — monolith, `claude-opus-5`

`scripts/phase1_demo.py`: 4 groups of paraphrased prompts plus 3 unrelated prompts.

| | Result |
|---|---|
| Paraphrases of a cached prompt | 7 of 8 hit (similarity 0.93–0.99) |
| Unrelated prompts | 3 of 3 missed, no false hits |
| Median latency, hit | **15 ms** |
| Median latency, miss (Claude call) | **2,171 ms**: hits are ~145× faster |

The one paraphrase that missed was "What is the boiling point of water at sea level?" vs. "At sea level, what temperature does water boil at?". Claude gave the same answer both times, so this was a missed saving, not a correctness problem. It's evidence that 0.92 may be too strict for MiniLM; Phase 3 tunes the threshold on a proper paraphrase set.

### Phase 2 — services

- **The cache survived the split unchanged.** Phase 1 embedded in-process with fastembed (ONNX); embedding-svc uses sentence-transformers (PyTorch). Re-embedding all 8 cached prompts gave cosine similarity 1.000000 against the stored vectors, and replaying the Phase 1 Opus 5 demo was served 15 of 15 from cache with zero provider calls.
- **Network cost of the split:** median hit latency went from 15 ms to **29 ms** (two gRPC hops plus Docker networking). Phase 4's tracing breaks this down per service.
- **Same demo on `claude-haiku-4-5`:** 7 of 8 paraphrases hit again, median miss **789 ms** (vs 2,171 ms on Opus 5).

Failure behaviour, tested by stopping containers:

| Scenario | Result |
|---|---|
| embedding-svc down | 200, `x-echo-cache: bypass` |
| cache-svc down | 200, `x-echo-cache: bypass`, no store attempted; `/stats` returns 503 |
| Both restored | next request hits the cache |
| Unknown model | 404 `not_found_error` from Anthropic |
| Provider not configured | 400 naming the missing env var |
| Opus 5 with `temperature` | 400 from Anthropic, message passed through |

## Local development

The Rust services are one Cargo workspace:

```bash
cargo test --workspace
cargo run -p cache-svc          # or gateway / provider-adapter
```

Each binary reads `.env` from the repo root and defaults to `localhost` addresses, so you can stop one container (`docker compose stop gateway`) and run that service from source against the rest of the stack.

## Known limitations (addressed in later phases)

- No in-flight dedup: two concurrent misses for the same prompt both call the LLM (Phase 3).
- The 0.92 threshold hasn't been tuned yet (Phase 3 eval).
- `stream: true` and tool calling are rejected (Phase 6).
- If the nearest match turns out to be expired, the lookup counts as a miss, even when a live entry further down the list would have matched.
- Stale Qdrant points that no lookup ever hits again are never cleaned up; a periodic sweep in cache-svc would fix it.
- The embedding-svc image is ~2 GB, mostly PyTorch. Serving the same model through ONNX Runtime would shrink it a lot, and Phase 2 showed the vectors match exactly.
