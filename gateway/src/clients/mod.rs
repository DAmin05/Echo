//! gRPC clients for the three internal services.
//!
//! Channels connect lazily, so the gateway starts even if a dependency isn't
//! up yet; calls to it fail until it is, and the handler degrades around that.
//! Each channel has its own deadline, sized to what that service does.
//!
//! Every call runs in a client span, and the `telemetry::inject` interceptor
//! passes that span's context to the callee, so the whole request is one trace.

use std::time::Duration;

use anyhow::Result;
use tonic::{
    service::interceptor::InterceptedService,
    transport::{Channel, Endpoint},
    Request, Status,
};
use tracing::instrument;

use crate::config::Config;

pub mod pb {
    pub mod embedding {
        tonic::include_proto!("echo.embedding.v1");
    }
    pub mod cache {
        tonic::include_proto!("echo.cache.v1");
    }
    pub mod provider {
        tonic::include_proto!("echo.provider.v1");
    }
}
use pb::{
    cache::{
        cache_service_client::CacheServiceClient, Candidate, FetchRequest, QueryRequest, StatsRequest, StatsResponse,
        StoreRequest,
    },
    embedding::{embedding_service_client::EmbeddingServiceClient, EmbedRequest, ScoreDuplicatesRequest},
    provider::{provider_service_client::ProviderServiceClient, GenerateRequest, GenerateResponse},
};

type Traced = InterceptedService<Channel, fn(Request<()>) -> Result<Request<()>, Status>>;

#[derive(Clone)]
pub struct Clients {
    embedding: EmbeddingServiceClient<Traced>,
    cache: CacheServiceClient<Traced>,
    provider: ProviderServiceClient<Traced>,
}

impl Clients {
    pub fn connect_lazy(cfg: &Config) -> Result<Self> {
        Ok(Self {
            embedding: EmbeddingServiceClient::with_interceptor(
                channel(&cfg.embedding_svc_url, Duration::from_secs(5))?,
                telemetry::inject as _,
            ),
            cache: CacheServiceClient::with_interceptor(channel(&cfg.cache_svc_url, Duration::from_secs(2))?, telemetry::inject as _),
            // Generous: a long answer with extended thinking can take minutes.
            provider: ProviderServiceClient::with_interceptor(
                channel(&cfg.provider_adapter_url, Duration::from_secs(600))?,
                telemetry::inject as _,
            ),
        })
    }

    #[instrument(name = "embed", skip_all, fields(otel.kind = "client", peer.service = "embedding-svc"))]
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Status> {
        let resp = self
            .embedding
            .clone()
            .embed(EmbedRequest { text: text.to_string() })
            .await?;
        Ok(resp.into_inner().vector)
    }

    /// Cross-encoder duplicate probability for each candidate, in order.
    #[instrument(name = "verify", skip_all, fields(otel.kind = "client", peer.service = "embedding-svc", candidates = candidates.len()))]
    pub async fn score_duplicates(&self, query: &str, candidates: Vec<String>) -> Result<Vec<f32>, Status> {
        let resp = self
            .embedding
            .clone()
            .score_duplicates(ScoreDuplicatesRequest { query: query.to_string(), candidates })
            .await?;
        Ok(resp.into_inner().scores)
    }

    /// Up to `limit` cached entries with identical params and similarity >= `threshold`.
    #[instrument(name = "cache.query", skip_all, fields(otel.kind = "client", peer.service = "cache-svc"))]
    pub async fn cache_query(
        &self,
        vector: Vec<f32>,
        params: &str,
        threshold: f32,
        limit: u32,
    ) -> Result<Vec<Candidate>, Status> {
        let resp = self
            .cache
            .clone()
            .query(QueryRequest { vector, params: params.to_string(), threshold: Some(threshold), limit })
            .await?;
        Ok(resp.into_inner().candidates)
    }

    /// The entry's response (counted as a hit), or `None` if it expired.
    #[instrument(name = "cache.fetch", skip_all, fields(otel.kind = "client", peer.service = "cache-svc"))]
    pub async fn cache_fetch(&self, entry_id: &str) -> Result<Option<String>, Status> {
        let resp = self
            .cache
            .clone()
            .fetch(FetchRequest { entry_id: entry_id.to_string() })
            .await?;
        Ok(resp.into_inner().response)
    }

    /// Uses cache-svc's configured TTL.
    #[instrument(name = "cache.store", skip_all, fields(otel.kind = "client", peer.service = "cache-svc"))]
    pub async fn cache_store(&self, vector: Vec<f32>, prompt: &str, params: &str, response: String) -> Result<String, Status> {
        let resp = self
            .cache
            .clone()
            .store(StoreRequest {
                vector,
                prompt: prompt.to_string(),
                params: params.to_string(),
                response,
                ttl_secs: None,
            })
            .await?;
        Ok(resp.into_inner().entry_id)
    }

    pub async fn cache_stats(&self) -> Result<StatsResponse, Status> {
        Ok(self.cache.clone().stats(StatsRequest {}).await?.into_inner())
    }

    #[instrument(name = "generate", skip_all, fields(otel.kind = "client", peer.service = "provider-adapter", model = %req.model))]
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse, Status> {
        Ok(self.provider.clone().generate(req).await?.into_inner())
    }
}

fn channel(url: &str, timeout: Duration) -> Result<Channel> {
    Ok(Endpoint::from_shared(url.to_string())?
        .connect_timeout(Duration::from_secs(1))
        .timeout(timeout)
        .connect_lazy())
}
