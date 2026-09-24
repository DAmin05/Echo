//! gRPC clients for the three internal services.
//!
//! Channels connect lazily, so the gateway starts even if a dependency isn't
//! up yet; calls to it fail until it is, and the handler degrades around that.
//! Each channel has its own deadline, sized to what that service does.

use std::time::Duration;

use anyhow::Result;
use tonic::{
    transport::{Channel, Endpoint},
    Status,
};

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
    cache::{cache_service_client::CacheServiceClient, CacheHit, QueryRequest, StatsRequest, StatsResponse, StoreRequest},
    embedding::{embedding_service_client::EmbeddingServiceClient, EmbedRequest},
    provider::{provider_service_client::ProviderServiceClient, GenerateRequest, GenerateResponse},
};

#[derive(Clone)]
pub struct Clients {
    embedding: EmbeddingServiceClient<Channel>,
    cache: CacheServiceClient<Channel>,
    provider: ProviderServiceClient<Channel>,
}

impl Clients {
    pub fn connect_lazy(cfg: &Config) -> Result<Self> {
        Ok(Self {
            embedding: EmbeddingServiceClient::new(channel(&cfg.embedding_svc_url, Duration::from_secs(5))?),
            cache: CacheServiceClient::new(channel(&cfg.cache_svc_url, Duration::from_secs(2))?),
            // Generous: a long answer with extended thinking can take minutes.
            provider: ProviderServiceClient::new(channel(&cfg.provider_adapter_url, Duration::from_secs(600))?),
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, Status> {
        let resp = self
            .embedding
            .clone()
            .embed(EmbedRequest { text: text.to_string() })
            .await?;
        Ok(resp.into_inner().vector)
    }

    /// Uses cache-svc's configured similarity threshold.
    pub async fn cache_query(&self, vector: Vec<f32>, params: &str) -> Result<Option<CacheHit>, Status> {
        let resp = self
            .cache
            .clone()
            .query(QueryRequest { vector, params: params.to_string(), threshold: None })
            .await?;
        Ok(resp.into_inner().hit)
    }

    /// Uses cache-svc's configured TTL.
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
