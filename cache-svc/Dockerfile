# Build context is the repo root: the Cargo workspace and proto/ are shared.
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY . .
# Cache mounts keep the cargo registry and build artifacts between image
# builds, so rebuilding after a code change doesn't recompile every dependency.
# `locked` serializes parallel builds of the three Rust services on the cache.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release --locked -p cache-svc \
    && cp target/release/cache-svc /usr/local/bin/cache-svc

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/cache-svc /usr/local/bin/cache-svc
CMD ["cache-svc"]
