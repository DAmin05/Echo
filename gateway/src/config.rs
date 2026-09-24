use std::{env, fmt::Display, str::FromStr};

use anyhow::Result;

/// Gateway configuration: where it listens, where the internal services are,
/// and the similarity threshold (used for cache lookups and in-flight dedup).
/// The cache TTL lives in cache-svc; provider keys live in provider-adapter.
/// The gateway holds no secrets.
pub struct Config {
    pub port: u16,
    pub similarity_threshold: f32,
    /// Off only to measure the thundering herd it prevents.
    pub dedup_enabled: bool,
    pub embedding_svc_url: String,
    pub cache_svc_url: String,
    pub provider_adapter_url: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            port: parse("PORT", 8080)?,
            similarity_threshold: parse("SIMILARITY_THRESHOLD", 0.90)?,
            dedup_enabled: parse("DEDUP_ENABLED", true)?,
            embedding_svc_url: var("EMBEDDING_SVC_URL", "http://localhost:50051"),
            cache_svc_url: var("CACHE_SVC_URL", "http://localhost:50052"),
            provider_adapter_url: var("PROVIDER_ADAPTER_URL", "http://localhost:50053"),
        })
    }
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
        Ok(raw) => raw
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid {key}={raw:?}: {e}")),
        Err(_) => Ok(default),
    }
}
