//! Echo gateway: the public, OpenAI-compatible HTTP API.
//!
//! A pure orchestrator since Phase 2: embedding, caching and LLM calls are
//! gRPC calls to embedding-svc, cache-svc and provider-adapter.

mod clients;
mod config;
mod handlers;
mod openai;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::{clients::Clients, config::Config};

#[tokio::main]
async fn main() -> Result<()> {
    // Looks in the current directory and its parents, so the repo-root .env works.
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "gateway=info".into()))
        .init();

    let cfg = Config::from_env()?;
    let clients = Clients::connect_lazy(&cfg)?;

    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/stats", get(handlers::stats))
        .route("/healthz", get(handlers::healthz))
        .with_state(clients);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port)).await?;
    info!(
        addr = %listener.local_addr()?,
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
