//! cache-svc: semantic cache behind gRPC.
//!
//! Each entry has one UUID shared by both stores:
//!   - Qdrant point `{id}`: prompt embedding + payload `{prompt, params}`
//!   - Redis hash `echo:entry:{id}`: `response`, `prompt`, `hits`, with a TTL
//!
//! Redis is the source of truth for liveness. When its TTL expires, the Qdrant
//! point becomes stale; it is deleted the next time a lookup lands on it.

mod qdrant;
mod redis_client;

use std::{env, fmt::Display, net::SocketAddr, str::FromStr};

use anyhow::Result;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{qdrant::VectorIndex, redis_client::EntryStore};

pub mod pb {
    tonic::include_proto!("echo.cache.v1");
}
use pb::{
    cache_service_server::{CacheService, CacheServiceServer},
    CacheHit, QueryRequest, QueryResponse, StatsRequest, StatsResponse, StoreRequest, StoreResponse,
};

struct Cache {
    index: VectorIndex,
    entries: EntryStore,
    dim: usize,
    default_threshold: f32,
    default_ttl_secs: u64,
}

impl Cache {
    fn check_vector(&self, vector: &[f32]) -> Result<(), Status> {
        if vector.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "vector has {} dimensions, expected {}",
                vector.len(),
                self.dim
            )));
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl CacheService for Cache {
    async fn query(&self, req: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let req = req.into_inner();
        self.check_vector(&req.vector)?;
        let threshold = req.threshold.unwrap_or(self.default_threshold);
        if !(-1.0..=1.0).contains(&threshold) {
            return Err(Status::invalid_argument("threshold must be within [-1, 1]"));
        }

        let Some((id, similarity)) = self
            .index
            .nearest(req.vector, &req.params, threshold)
            .await
            .map_err(unavailable)?
        else {
            return Ok(Response::new(QueryResponse { hit: None }));
        };

        let Some(response) = self.entries.get(&id).await.map_err(unavailable)? else {
            // TTL expired in Redis; drop the orphaned vector so it stops matching.
            match self.index.delete(&id).await {
                Ok(()) => info!(entry = %id, "evicted stale cache entry"),
                Err(e) => warn!(entry = %id, error = %e, "failed to evict stale cache entry"),
            }
            return Ok(Response::new(QueryResponse { hit: None }));
        };

        if let Err(e) = self.entries.record(true).await {
            warn!(error = %e, "failed to record hit");
        }
        Ok(Response::new(QueryResponse {
            hit: Some(CacheHit { entry_id: id, similarity, response }),
        }))
    }

    async fn store(&self, req: Request<StoreRequest>) -> Result<Response<StoreResponse>, Status> {
        let req = req.into_inner();
        self.check_vector(&req.vector)?;
        let ttl_secs = req.ttl_secs.unwrap_or(self.default_ttl_secs);
        let id = Uuid::new_v4().to_string();

        // Redis first: a Qdrant point without a Redis entry is harmless (it is
        // treated as expired), but the reverse would be an unreachable entry.
        self.entries
            .put(&id, &req.prompt, &req.response, ttl_secs)
            .await
            .map_err(unavailable)?;
        self.index
            .insert(&id, req.vector, &req.prompt, &req.params)
            .await
            .map_err(unavailable)?;

        if let Err(e) = self.entries.record(false).await {
            warn!(error = %e, "failed to record miss");
        }
        Ok(Response::new(StoreResponse { entry_id: id }))
    }

    async fn stats(&self, _: Request<StatsRequest>) -> Result<Response<StatsResponse>, Status> {
        let (hits, misses) = self.entries.stats().await.map_err(unavailable)?;
        Ok(Response::new(StatsResponse { hits, misses }))
    }
}

/// Backing-store failures are reported as UNAVAILABLE: the caller (gateway)
/// treats that as "no cache right now" and carries on without it.
fn unavailable(e: anyhow::Error) -> Status {
    warn!(error = %format!("{e:#}"), "backing store error");
    Status::unavailable(format!("{e:#}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "cache_svc=info".into()),
        )
        .init();

    let addr: SocketAddr = var("CACHE_SVC_ADDR", "0.0.0.0:50052").parse()?;
    let dim: u64 = parse("EMBEDDING_DIM", 384)?;
    let collection = var("QDRANT_COLLECTION", "echo_cache");

    let cache = Cache {
        index: VectorIndex::connect(&var("QDRANT_URL", "http://localhost:6334"), &collection, dim).await?,
        entries: EntryStore::connect(&var("REDIS_URL", "redis://localhost:6379")).await?,
        dim: dim as usize,
        default_threshold: parse("SIMILARITY_THRESHOLD", 0.90)?,
        default_ttl_secs: parse("CACHE_TTL_SECS", 86_400)?,
    };

    info!(
        %addr,
        collection,
        dim,
        threshold = %cache.default_threshold,
        ttl_secs = cache.default_ttl_secs,
        "cache-svc listening"
    );
    Server::builder()
        .add_service(CacheServiceServer::new(cache))
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.ok();
}

fn var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn parse<T>(key: &str, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: Display,
{
    match env::var(key) {
        Ok(raw) => raw.parse().map_err(|e| anyhow::anyhow!("invalid {key}={raw:?}: {e}")),
        Err(_) => Ok(default),
    }
}
