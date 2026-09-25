//! Echo gateway: the public, OpenAI-compatible HTTP API.
//!
//! A pure orchestrator since Phase 2: embedding, caching and LLM calls are
//! gRPC calls to embedding-svc, cache-svc and provider-adapter.

mod clients;
mod config;
mod dedup;
mod handlers;
mod matching;
mod openai;

use std::sync::{atomic::AtomicU64, Arc};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use tracing::info;

use crate::{clients::Clients, config::Config, dedup::InFlight, handlers::AppState};

#[tokio::main]
async fn main() -> Result<()> {
    // Looks in the current directory and its parents, so the repo-root .env works.
    dotenvy::dotenv().ok();
    let _telemetry = telemetry::init("gateway", "gateway=info")?;

    let cfg = Config::from_env()?;
    let state = Arc::new(AppState {
        clients: Clients::connect_lazy(&cfg)?,
        inflight: InFlight::new(cfg.matcher.candidate_threshold()),
        dedup_enabled: cfg.dedup_enabled,
        matcher: cfg.matcher,
        coalesced: AtomicU64::new(0),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/stats", get(handlers::stats))
        .route("/healthz", get(handlers::healthz))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port)).await?;
    info!(
        addr = %listener.local_addr()?,
        matching = %cfg.matcher,
        dedup = cfg.dedup_enabled,
        embedding_svc = %cfg.embedding_svc_url,
        cache_svc = %cfg.cache_svc_url,
        provider_adapter = %cfg.provider_adapter_url,
        "echo gateway listening"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;
    Ok(())
}
