//! Response side of the cache: one Redis hash per entry, with a TTL, plus
//! global hit/miss counters.

use anyhow::{Context, Result};
use redis::{aio::ConnectionManager, AsyncCommands};

const STATS_HITS: &str = "echo:stats:hits";
const STATS_MISSES: &str = "echo:stats:misses";

#[derive(Clone)]
pub struct EntryStore {
    conn: ConnectionManager,
}

impl EntryStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).context("parsing REDIS_URL")?;
        let conn = ConnectionManager::new(client).await.context("connecting to Redis")?;
        Ok(Self { conn })
    }

    /// Writes the entry and its TTL atomically.
    pub async fn put(&self, id: &str, prompt: &str, response: &str, ttl_secs: u64) -> Result<()> {
        let key = entry_key(id);
        redis::pipe()
            .atomic()
            .hset_multiple(&key, &[("response", response), ("prompt", prompt), ("hits", "0")])
            .ignore()
            .expire(&key, ttl_secs as i64)
            .ignore()
            .query_async::<()>(&mut self.conn.clone())
            .await
            .context("writing cache entry to Redis")
    }

    /// The cached response, or `None` if the entry's TTL has expired.
    /// Bumps the entry's hit count when found.
    pub async fn get(&self, id: &str) -> Result<Option<String>> {
        let key = entry_key(id);
        let mut conn = self.conn.clone();
        let response: Option<String> = conn.hget(&key, "response").await?;
        if response.is_some() {
            let _: i64 = conn.hincr(&key, "hits", 1).await?;
        }
        Ok(response)
    }

    pub async fn record(&self, hit: bool) -> Result<()> {
        let key = if hit { STATS_HITS } else { STATS_MISSES };
        let _: i64 = self.conn.clone().incr(key, 1).await?;
        Ok(())
    }

    pub async fn stats(&self) -> Result<(u64, u64)> {
        let (hits, misses): (Option<u64>, Option<u64>) = redis::pipe()
            .get(STATS_HITS)
            .get(STATS_MISSES)
            .query_async(&mut self.conn.clone())
            .await?;
        Ok((hits.unwrap_or(0), misses.unwrap_or(0)))
    }
}

fn entry_key(id: &str) -> String {
    format!("echo:entry:{id}")
}
