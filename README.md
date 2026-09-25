# Echo — LLM Gateway with Semantic Caching

Echo sits between clients and LLM providers. It embeds each prompt, checks a vector cache for semantically similar past requests, and returns the cached response on a hit, skipping the LLM call entirely. Clients talk to it with the OpenAI chat completions API, so any OpenAI SDK works by changing only the base URL.

On a 1,000-request mixed workload (k6, `claude-haiku-4-5`), Echo served **62% of requests without calling the LLM**, cut **LLM cost by 70%**, and brought **median latency from 690 ms to 82 ms**, with 1 wrong answer out of 623 served from cache. Details in [Phase 5 — load test](#phase-5--load-test).

## Status

- [x] Phase 0 — Infra bootstrap (Qdrant + Redis via Docker Compose)
- [x] Phase 1 — Monolith proof of concept
- [x] Phase 2 — Split into services (gRPC)
- [x] Phase 3 — Concurrency correctness (in-flight dedup, threshold tuning)
- [x] Phase 4 — Observability (OpenTelemetry + Jaeger)
- [x] Phase 5 — Load test and results

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
2. **Joins in-flight requests**: if a matching request is already being answered, waits for that answer instead (`x-echo-cache: coalesced`).
3. **Queries** the cache (cache-svc → up to 3 candidates from Qdrant with cosine similarity ≥ 0.70, restricted to entries whose other request fields match exactly).
4. **Verifies** the candidates (embedding-svc → a cross-encoder reads the question and each candidate together and must score them as the same question, ≥ 0.80).
5. On a verified **hit**, fetches and returns the cached response (`x-echo-cache: hit`, `x-echo-similarity`, `x-echo-verify-score`).
6. On a **miss**, **generates** the answer (provider-adapter → Claude / OpenAI / Ollama), **stores** it (cache-svc), and returns it (`x-echo-cache: miss`).

| Service | Language | Port(s) | Owns |
|---|---|---|---|
| `gateway` | Rust (Axum, tonic) | 8080 | Public API, cache key, matching thresholds, in-flight dedup, orchestration. Holds no secrets. |
| `embedding-svc` | Python (sentence-transformers, grpcio, FastAPI) | 50051 gRPC, 8001 HTTP | The models: bi-encoder for vectors (`all-MiniLM-L6-v2`), cross-encoder for verification (`quora-roberta-base`) |
| `cache-svc` | Rust (tonic) | 50052 | Qdrant + Redis, TTL, hit/miss stats |
| `provider-adapter` | Rust (tonic, reqwest) | 50053 | Provider API keys and request/response translation |
| `qdrant` | — | 6333 REST, 6334 gRPC | Vectors |
| `redis` | — | 6379 | Cached responses with TTLs |
| `jaeger` | — | 16686 UI, 4317 OTLP | Distributed traces from all four services |

The contracts are in [`proto/`](proto/). Rust stubs are generated at build time with [protox](https://crates.io/crates/protox), a pure-Rust protobuf compiler, so no `protoc` install is needed; Python stubs are generated in the embedding-svc image.

## Running

Requires Docker (Docker Desktop or OrbStack).

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY
docker compose up -d --build      # first build takes a few minutes
python3 scripts/phase1_demo.py    # paraphrase demo against localhost:8080
./loadtest/run.sh                 # 1,000-request k6 load test + results tables
```

Any OpenAI SDK works:

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
client.chat.completions.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "Hi"}])
```

`GET /stats` returns running hit/miss counters, plus how many requests were coalesced and how many are in flight. `POST localhost:8001/embed` with `{"text": "..."}` returns a raw embedding, and `POST localhost:8001/score-duplicates` with `{"query": "...", "candidates": [...]}` returns verifier scores. Qdrant's dashboard is at http://localhost:6333/dashboard.

Traces are at http://localhost:16686 (pick service `gateway`), or as text with `python3 scripts/trace_tree.py`.

Stop with `docker compose down` (add `-v` to wipe stored vectors and responses).

### Choosing a provider

provider-adapter picks the provider from the model name: `claude-*` → Anthropic, `gpt-*` / `o3-*` etc. → OpenAI, anything else → Ollama. A prefix forces one: `ollama/llama3.2`, `anthropic/claude-sonnet-5`. Each provider is enabled by its setting in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_URL`). Requesting one that isn't configured returns a 400 that names the missing variable.

## Design decisions

### What makes two requests "the same"

- **Only the final user message is embedded.** That's the part expected to vary in wording.
- **Everything else must match exactly**: model, max_tokens, sampling params, and all earlier messages (system prompt, prior turns), serialized as canonical JSON and stored as a Qdrant payload filter. Embedding the whole conversation would let a long shared system prompt pull unrelated questions toward each other and cause false hits. It would also let the same question asked in a different conversation return an answer written for another context.

### In-flight deduplication (thundering herd)

Without dedup, N near-identical requests arriving together all miss the cache, since nothing is stored until the first one finishes, and all N call the LLM. [`gateway/src/dedup.rs`](gateway/src/dedup.rs) makes the first request the **leader**. Later similar requests become **followers** and wait for the leader's result on a `tokio::sync::broadcast` channel.

- **"Similar" means what it means for the cache:** identical `params`, cosine similarity ≥ the candidate threshold against in-flight vectors, then cross-encoder verification. An exact-match key would only catch byte-identical prompts.
- **Verification happens outside the lock.** `join` returns the in-flight candidates, already subscribed to their results. The gateway verifies them, then either waits on an accepted one or calls `join` again with the rejected ones excluded. Only a `join` that finds no un-rejected candidate registers a new leader, so two requests that both rejected a third one still find each other.
- **No gap between in-flight and cached.** A request joins the in-flight set *before* its cache lookup, and a leader stores its answer *before* leaving the set. A later similar request therefore always finds either the leader or its cached answer.
- **Check-and-register is one critical section** under a `Mutex`, so two simultaneous requests can't both become leader. Nothing is awaited while the lock is held.
- **The leader's work runs as a detached task.** If the leader's client disconnects, the answer is still cached and followers still receive it. If the task dies anyway, followers see the channel close and handle the request themselves.
- **Followers share the leader's outcome, errors included.** A 429 for the leader is a 429 for its followers too, rather than N more requests into a rate limit.
- **Scope: one gateway process.** Multiple gateway replicas would each elect their own leaders; cross-replica dedup would need a shared lock (e.g. Redis `SET NX` with a TTL).

`DEDUP_ENABLED=false` turns it off, which exists only to measure what it prevents (see Results).

### Matching: vector search, then verification

Deciding that two questions are "the same" has to be both cheap (it runs on every request) and safe: serving the answer to a *different* question is worse than a cache miss. Echo does it in two stages:

1. **Recall: vector similarity.** Bi-encoder embeddings (MiniLM) find up to 3 candidates with cosine ≥ `CANDIDATE_THRESHOLD` (0.70). Fast, but blind to details.
2. **Precision: a cross-encoder.** `cross-encoder/quora-roberta-base`, trained on Quora's duplicate-question pairs, reads the new question and each candidate *together* and must score them ≥ `VERIFY_THRESHOLD` (0.80).

It **fails closed**: if the verifier is unavailable, nothing is served from cache or coalesced, and the request goes to the provider. `VERIFY_ENABLED=false` switches back to a single `SIMILARITY_THRESHOLD` (0.90).

#### Why one threshold isn't enough

[`scripts/threshold_eval.py`](scripts/threshold_eval.py) runs [`eval/paraphrases.json`](eval/paraphrases.json) through both strategies using the live models. The set has 50 questions, each with 2 paraphrases that *should* hit and 1 hard negative that *must not*: a question worded almost identically that needs a different answer ("capital of France" / "capital of Germany", "10 km to miles" / "10 miles to km"). It simulates a cache holding the 50 originals.

Vector similarity alone:

| Threshold | Paraphrases served (savings) | Wrong answers served | Precision |
|---|---|---|---|
| 0.80 | 91% | 20 | 82.0% |
| 0.85 | 80% | 11 | 87.9% |
| **0.90** | **60%** | **5** | **92.3%** |
| 0.92 | 42% | 5 | 89.4% |
| 0.95 | 25% | 3 | 89.3% |
| 0.97 | 11% | 2 | 84.6% |

No threshold is safe. The worst hard negatives score *higher* than almost every real paraphrase: "convert an integer to a string in JavaScript" vs "a string to an integer" scores 0.996, and "10 miles to km" vs "10 km to miles" scores 0.993. MiniLM embeddings barely register direction, word order or which number appears, so even at 0.97 two of thirteen hits are wrong. Raising the threshold mostly throws away savings. (The best single threshold, 0.90, is the `VERIFY_ENABLED=false` fallback; it beats the original 0.92 by catching 18 more paraphrases for the same 5 wrong answers.)

With verification (candidate ≥ 0.70, top 3):

| Verify threshold | Paraphrases served | Wrong answers served | Precision |
|---|---|---|---|
| 0.50 | 72% | 2 | 97.3% |
| 0.70 | 67% | 2 | 97.1% |
| **0.80** | **60%** | **1** | **98.4%** |
| 0.90 | 32% | 0 | 100% |

At the default, it serves **the same 60% of paraphrases as the best single threshold, with 1 wrong answer instead of 5**. The verifier scores "10 miles to km" vs "10 km to miles" at 0.435 (rejected), where cosine had it at 0.993. The one wrong answer left is int↔string in JavaScript (0.839). 0.90 would remove it too, at the cost of half the savings. I picked 0.80 because the project favours correctness, but not at any price.

Other verifiers were tried with the same harness: the smaller `quora-distilroberta-base` passes both swaps (0.98, 0.97), and the general-purpose `stsb-roberta-base` scores them 0.997. It's the duplicate-question training that matters, not model size.

**Costs:**
- Verification adds ~40 ms to a hit on CPU (median hit latency: 29 ms → 68 ms, still ~10× faster than a Haiku call), and runs only when there is a candidate.
- The verifier is conservative with some phrasings. It rejected "How many miles is 10 km?" as a match for "Convert 10 kilometers to miles." (0.03).
- Its scores are asymmetric. "Which city is Australia's capital?" is accepted when it's the new question (0.84–0.88 against the other phrasings) but scores 0.32–0.79 when it's the one already cached or in flight. What gets reused can therefore depend on which phrasing arrived first. Scoring both directions and taking the max would raise recall, but it needs re-checking against the hard negatives first.
- The eval set is small and hand-written, so the numbers are indicative, not a benchmark.

### Observability

Every service exports OpenTelemetry spans over OTLP to Jaeger, and a request is one trace across all four services.

- **Propagation:** the gateway starts a server span per HTTP request, joining the caller's trace if it sent a W3C `traceparent`. Each gRPC call runs in a client span, and a tonic interceptor ([`telemetry::inject`](telemetry/src/lib.rs)) writes its context into the request metadata. The Rust servers pick it up in `Server::builder().trace_fn(telemetry::server_span)`; the Python server uses OpenTelemetry's gRPC instrumentation.
- **Async boundaries:** the dedup leader's work runs in a spawned task, which doesn't inherit the span, so it's spawned with `.in_current_span()`. A follower's wait is its own span, `dedup.wait_for_leader`.
- **Shared setup:** the three Rust services use one small workspace crate, [`telemetry/`](telemetry/), for tracer setup, propagation and the server span. With `OTEL_EXPORTER_OTLP_ENDPOINT` unset (e.g. a local `cargo run` without Jaeger), they just log.
- **Attributes:** `cache` (hit / miss / coalesced / bypass) and `model` on the root span; `gen_ai.*` on the LLM call, following OpenTelemetry's GenAI conventions (request/response model, finish reason, input/output tokens); `candidates` on Qdrant queries; `pairs` and `max_score` on verification.
- **Sampling:** every request is traced, which is fine for development. Production would sample, e.g. by ratio at the gateway.

`scripts/trace_tree.py` prints recent traces as trees (`--cache hit` to filter). A miss and a hit, back to back:

```
trace c51d8ed3be7c1897  cache=miss  total=651.0ms
  POST /v1/chat/completions                          gateway            651.0ms
    embed                                            gateway             25.9ms
      /echo.embedding.v1.EmbeddingService/Embed      embedding-svc       19.5ms
        minilm.encode                                embedding-svc       19.2ms
    cache.query                                      gateway              4.8ms
      echo.cache.v1.CacheService/Query               cache-svc            3.6ms
        qdrant.query                                 cache-svc            3.2ms
    generate                                         gateway            606.8ms
      echo.provider.v1.ProviderService/Generate      provider-adapter   605.1ms
        anthropic claude-haiku-4-5                   provider-adapter   604.0ms
    cache.store                                      gateway             12.9ms
      echo.cache.v1.CacheService/Store               cache-svc           11.7ms
        redis.put                                    cache-svc            3.4ms
        qdrant.upsert                                cache-svc            7.4ms

trace 16b0f84c8b2e3faa  cache=hit  total=80.0ms
  POST /v1/chat/completions                          gateway             80.0ms
    embed                                            gateway             31.2ms
      /echo.embedding.v1.EmbeddingService/Embed      embedding-svc       29.1ms
        minilm.encode                                embedding-svc       28.9ms
    cache.query                                      gateway              1.2ms
      echo.cache.v1.CacheService/Query               cache-svc            0.9ms
        qdrant.query                                 cache-svc            0.9ms
    verify                                           gateway             46.9ms
      /echo.embedding.v1.EmbeddingService/ScoreDuplicates  embedding-svc 46.1ms
        cross_encoder.predict                        embedding-svc       46.1ms
    cache.fetch                                      gateway              0.5ms
      echo.cache.v1.CacheService/Fetch               cache-svc            0.3ms
        redis.get                                    cache-svc            0.2ms
```

A request that joined an in-flight leader shows `embed` → `verify` → `dedup.wait_for_leader`, with no cache or provider spans.

What the traces showed:

- **On a hit, the ML models are ~95% of the time.** MiniLM (~20–30 ms) and the cross-encoder (~45 ms) dominate; Qdrant and Redis together take ~1–2 ms, and each gRPC hop adds under 2 ms. Faster hits mean faster inference (ONNX Runtime, batching concurrent requests, a GPU), not a faster cache.
- **On a miss, the provider is ~93% of the time.** The cache's own overhead (embed, query, store) is ~45 ms on top of the LLM call.
- **They caught a cold-start bug.** The first hit after a deploy spent 136 ms in `cross_encoder.predict`, against ~45 ms warm, because the first inference in each model pays lazy initialisation. embedding-svc now runs one warm-up inference per model before `/healthz` passes, and the first hit after a deploy dropped from 158 ms to 80 ms.

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

### Phase 5 — load test

[k6](https://k6.io) (in Docker, on the Compose network) replays a 1,000-request mixed workload at 10 requests/s against a cold cache, with `claude-haiku-4-5`, on an Apple M3 Pro (Docker: 11 CPUs). The workload is generated by [`loadtest/make_workload.py`](loadtest/make_workload.py):

- **40% exact repeats** of 50 popular questions, Zipf-weighted so a few are asked far more often, like real traffic.
- **30% paraphrases** of those questions.
- **30% unique questions**, each asked once. They're deliberately look-alikes ("What is the capital of Peru?" / "...of Chile?", "Who wrote *Dune*?" / "...*Dracula*?"), so any cache hit on one is a wrong answer. This is the correctness check.

```bash
./loadtest/run.sh                   # ~2 min, ~$0.06 in Haiku calls; prints the tables below
```

**Results**

| | |
|---|---|
| Served without calling the LLM | **62.3%** (617 cache hits + 6 coalesced) |
| LLM cost per 1,000 requests (Haiku 4.5) | $0.189 → **$0.056 (−70%)** |
| Median latency | 690 ms → **82 ms** overall; cache hits 77 ms |
| Wrong answers served from cache | 1 of 623 (see below) |
| Errors | 0 |

By request type:

| Request type | Requests | Served from cache | Provider calls |
|---|---|---|---|
| Exact repeat | 400 | 91.0% | 36 (first time each question was asked) |
| Paraphrase | 300 | 86.0% | 42 |
| Unique (look-alike) | 300 | 0.3% (1, a wrong answer) | 299 |

The paraphrase rate is higher than the eval's 60% because a popular paraphrase recurs, and after its first appearance it matches itself exactly.

Latency by path:

| Path | Requests | p50 | p95 | p99 |
|---|---|---|---|---|
| Cache hit | 617 | **77 ms** | 87 ms | 107 ms |
| Provider call (miss) | 377 | 690 ms | 1,182 ms | 1,964 ms |
| All requests | 1,000 | **82 ms** | 957 ms | 1,556 ms |

Cost per 1,000 requests, from the token usage on every response (a hit replays the usage of the call that produced it):

| Model | Without Echo | With Echo | Saved |
|---|---|---|---|
| `claude-haiku-4-5` (measured) | $0.189 | $0.056 | 70% |
| `claude-sonnet-5` (projected) | $0.378 | $0.113 | 70% |
| `claude-opus-5` (projected) | $0.946 | $0.282 | 70% |

Projections reuse the measured token counts at each model's list price; real counts differ somewhat by model. The saving (70%) is higher than the share of requests served (62%) because the unique questions, which always call the provider, have shorter answers than the popular ones.

**The wrong answer.** "Who wrote Frankenstein?" was served the cached answer to "Who wrote Dracula?". Vector similarity was only 0.719, but the cross-encoder scored the pair 0.967. It's the same weakness the eval found, in a new form: two questions with the same structure about closely related subjects (two Gothic novels) look like duplicates to a model trained on Quora question pairs.

**Hit-path capacity.** [`loadtest/capacity.sh`](loadtest/capacity.sh) replays questions the run above already cached, at increasing rates, so every request should be a hit:

| Rate | p50 | p95 | Provider calls |
|---|---|---|---|
| 10 req/s (mixed run above) | 77 ms | 87 ms | — |
| 25 req/s | 137 ms | 196 ms | 0 |
| 50 req/s | 3,938 ms | 7,430 ms | 0 (k6 dropped 203 requests: no free virtual users) |
| 100 req/s | 4,782 ms | 23,463 ms | **225** |

The traces from the overloaded runs pin down the cause:
- **The bottleneck is model inference.** `minilm.encode` and `cross_encoder.predict` sit at exactly 5 s (p50 and p95), which is the gateway's deadline for embedding-svc. Requests queue for CPU inside embedding-svc until the gateway cancels them. Meanwhile `qdrant.query` stays at 1–2 ms and `redis.get` under 1 ms.
- **Overload turns into LLM spend.** A cancelled embed fails open (bypass, 115 times) and a cancelled verification fails closed (no match, 498 times). Both send the request to the provider, so at 100 req/s, questions that were already cached caused 225 provider calls. Failing open keeps the service available, but the cache's savings disappear exactly when traffic is highest.
- **The overload compounds.** embedding-svc keeps computing requests the gateway has already cancelled, spending CPU on answers nobody will read.

**What would fix it**, in order of impact:
1. Micro-batch concurrent requests into one forward pass per model in embedding-svc. Both models are far cheaper per item in batches.
2. Admission control: bound in-flight inference and reject fast when full, instead of queueing past the caller's deadline.
3. Cheaper inference: ONNX Runtime; the Phase 2 check showed its vectors are identical.
4. More embedding-svc replicas. It's stateless, so it scales horizontally.

### Phase 1 — monolith, `claude-opus-5`

`scripts/phase1_demo.py`: 4 groups of paraphrased prompts plus 3 unrelated prompts.

| | Result |
|---|---|
| Paraphrases of a cached prompt | 7 of 8 hit (similarity 0.93–0.99) |
| Unrelated prompts | 3 of 3 missed, no false hits |
| Median latency, hit | **15 ms** |
| Median latency, miss (Claude call) | **2,171 ms**: hits are ~145× faster |

The one paraphrase that missed was "What is the boiling point of water at sea level?" vs. "At sea level, what temperature does water boil at?". Claude gave the same answer both times, so this was a missed saving, not a correctness problem. The demo's paraphrases turned out to be easy ones: on Phase 3's larger eval set, a 0.92 similarity threshold caught only 42% of paraphrases.

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

### Phase 3 — concurrency and matching

[`scripts/thundering_herd.py`](scripts/thundering_herd.py) fires 20 simultaneous requests, drawn from 4 paraphrases of one question, at a cold cache:

| | LLM calls | Distinct answers returned | Median latency |
|---|---|---|---|
| `DEDUP_ENABLED=false` | 20 | 5 | 759 ms |
| Dedup, similarity only (≥ 0.90) | 2 | 2 | 777 ms |
| **Dedup, verified (default)** | **1** | **1** | 931 ms |

That's 95% fewer provider calls, and everyone asking at the same moment gets the same answer. Similarity-only dedup needed 2 leaders because two of the paraphrases ("Which city is Australia's capital?" / "Tell me the capital of Australia.") score only 0.878 against each other. With verification they're candidates at ≥ 0.70, and the cross-encoder confirms they're the same question. The unit test `fifty_concurrent_requests_make_one_backend_call` checks the single-leader case directly.

Live check of the verifier through the gateway, cold cache:

| Request (in order) | Result |
|---|---|
| Convert 10 kilometers to miles. | miss → "≈ 6.21 miles" cached |
| Convert 10 miles to kilometers. | **miss** (cosine 0.993, verifier rejected); similarity-only would have served "6.21 miles" |
| Convert 10 miles to kilometers. (again) | hit, verify score 0.967 |

Matching results are in [Matching: vector search, then verification](#matching-vector-search-then-verification).

## Local development

The Rust services are one Cargo workspace:

```bash
cargo test --workspace
cargo run -p cache-svc          # or gateway / provider-adapter
```

Each binary reads `.env` from the repo root and defaults to `localhost` addresses, so you can stop one container (`docker compose stop gateway`) and run that service from source against the rest of the stack.

## Known limitations

- Matching still serves the occasional wrong answer (1 in 61 hits on the eval set), and the verifier misses some valid paraphrases. A larger verifier, or one fine-tuned on this traffic, is the next lever.
- In-flight dedup is per gateway process, not across replicas.
- The hit path saturates between 25 and 50 requests/s on one embedding-svc, and under overload the cache's savings disappear (see [hit-path capacity](#phase-5--load-test)). Micro-batching and admission control in embedding-svc are the fix.
- `stream: true` and tool calling are rejected (Phase 6).
- Only the top 3 candidates are considered. If all of them fail verification or have expired, the lookup misses, even when a matching entry further down the list would have passed.
- Stale Qdrant points that no lookup ever hits again are never cleaned up; a periodic sweep in cache-svc would fix it.
- The embedding-svc image is ~2 GB, mostly PyTorch. Serving the same model through ONNX Runtime would shrink it a lot, and Phase 2 showed the vectors match exactly.
