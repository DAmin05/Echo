//! In-flight request deduplication ("thundering herd" protection).
//!
//! Without this, N near-identical requests that arrive together all miss the
//! cache (nothing is stored until the first one finishes) and all call the
//! LLM. With it, the first becomes the *leader* and does the work; the others
//! become *followers* and wait for the leader's result.
//!
//! "Near-identical" uses the same rule as the cache: identical `params`, and
//! cosine similarity of the prompt embeddings >= the threshold. An exact-match
//! key would only catch byte-identical prompts.
//!
//! Correctness depends on ordering in the caller:
//!   1. `join` happens *before* the cache lookup, and
//!   2. the leader stores its answer in the cache *before* `finish` removes it.
//!
//! So a later similar request either finds the leader in flight or finds its
//! answer in the cache. There is no window where it sees neither.
//!
//! Scope: one gateway process. Multiple replicas would each have their own
//! leaders; cross-replica dedup would need a shared lock (e.g. Redis SET NX).

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use tokio::sync::broadcast;

pub struct InFlight<T> {
    /// Keyed by `params`; within a key, a linear scan over in-flight vectors.
    /// The number of concurrent requests with identical params is small, so a
    /// scan is cheaper than any index.
    requests: Mutex<HashMap<String, Vec<Pending<T>>>>,
    threshold: f32,
    next_id: AtomicU64,
}

struct Pending<T> {
    id: u64,
    vector: Vec<f32>,
    tx: broadcast::Sender<T>,
}

pub enum Joined<T: Clone> {
    /// No similar request is in flight: do the work, then call `finish`.
    Leader(Leader<T>),
    /// A similar request is in flight: wait for its result. `Err(Closed)`
    /// means the leader went away without one; handle the request yourself.
    Follower(broadcast::Receiver<T>),
}

impl<T: Clone> InFlight<T> {
    pub fn new(threshold: f32) -> Arc<Self> {
        Arc::new(Self { requests: Mutex::default(), threshold, next_id: AtomicU64::new(0) })
    }

    /// Joins the most similar in-flight request above the threshold, or
    /// registers this one as a new leader. Check and register happen under one
    /// lock, so two simultaneous callers can't both become leader.
    pub fn join(self: &Arc<Self>, params: &str, vector: &[f32]) -> Joined<T> {
        let mut requests = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        let bucket = requests.entry(params.to_string()).or_default();

        let closest = bucket
            .iter()
            .map(|p| (p, cosine(&p.vector, vector)))
            .filter(|(_, sim)| *sim >= self.threshold)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((pending, _)) = closest {
            return Joined::Follower(pending.tx.subscribe());
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Capacity 1: exactly one value is ever sent per leader.
        let (tx, _) = broadcast::channel(1);
        bucket.push(Pending { id, vector: vector.to_vec(), tx: tx.clone() });
        Joined::Leader(Leader { inflight: Arc::clone(self), params: params.to_string(), id, tx: Some(tx) })
    }

    /// Number of leaders currently in flight.
    pub fn len(&self) -> usize {
        self.requests.lock().unwrap_or_else(|e| e.into_inner()).values().map(Vec::len).sum()
    }

    fn remove(&self, params: &str, id: u64) {
        let mut requests = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(bucket) = requests.get_mut(params) {
            bucket.retain(|p| p.id != id);
            if bucket.is_empty() {
                requests.remove(params);
            }
        }
    }
}

pub struct Leader<T: Clone> {
    inflight: Arc<InFlight<T>>,
    params: String,
    id: u64,
    /// `None` once `finish` has run.
    tx: Option<broadcast::Sender<T>>,
}

impl<T: Clone> Leader<T> {
    /// Unregisters, then hands `value` to every follower.
    ///
    /// Unregistering first means every follower subscribed before the send
    /// and receives it; anyone arriving later goes to the cache instead.
    pub fn finish(mut self, value: T) {
        self.inflight.remove(&self.params, self.id);
        if let Some(tx) = self.tx.take() {
            // Err only means there were no followers.
            let _ = tx.send(value);
        }
    }
}

impl<T: Clone> Drop for Leader<T> {
    /// A leader dropped without `finish` (e.g. its task panicked) unregisters
    /// and closes the channel, so followers stop waiting and fall back.
    fn drop(&mut self) {
        if self.tx.is_some() {
            self.inflight.remove(&self.params, self.id);
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return -1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicUsize, time::Duration};

    use tokio::sync::broadcast::error::RecvError;

    use super::*;

    const A: [f32; 3] = [1.0, 0.0, 0.0];
    const A_ISH: [f32; 3] = [0.99, 0.1, 0.0]; // cosine with A ≈ 0.995
    const B: [f32; 3] = [0.0, 1.0, 0.0]; // orthogonal to A

    fn leader(j: Joined<&'static str>) -> Leader<&'static str> {
        match j {
            Joined::Leader(l) => l,
            Joined::Follower(_) => panic!("expected leader"),
        }
    }

    fn follower(j: Joined<&'static str>) -> broadcast::Receiver<&'static str> {
        match j {
            Joined::Follower(rx) => rx,
            Joined::Leader(_) => panic!("expected follower"),
        }
    }

    #[tokio::test]
    async fn similar_request_follows_and_gets_leaders_result() {
        let inflight = InFlight::new(0.9);
        let lead = leader(inflight.join("p", &A));
        let mut rx = follower(inflight.join("p", &A_ISH));
        lead.finish("answer");
        assert_eq!(rx.recv().await.unwrap(), "answer");
        assert_eq!(inflight.len(), 0);
    }

    #[test]
    fn dissimilar_or_different_params_lead_separately() {
        let inflight = InFlight::<&str>::new(0.9);
        let _a = leader(inflight.join("p", &A));
        let _b = leader(inflight.join("p", &B));
        let _c = leader(inflight.join("other params", &A));
        assert_eq!(inflight.len(), 3);
    }

    #[test]
    fn finished_leader_is_no_longer_joinable() {
        let inflight = InFlight::new(0.9);
        leader(inflight.join("p", &A)).finish("done");
        let _next = leader(inflight.join("p", &A));
    }

    #[tokio::test]
    async fn dropped_leader_releases_followers() {
        let inflight = InFlight::<&str>::new(0.9);
        let lead = leader(inflight.join("p", &A));
        let mut rx = follower(inflight.join("p", &A));
        drop(lead);
        assert!(matches!(rx.recv().await, Err(RecvError::Closed)));
        assert_eq!(inflight.len(), 0);
    }

    /// The thundering herd itself: 50 concurrent near-identical requests,
    /// exactly one call to the (slow) backend.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fifty_concurrent_requests_make_one_backend_call() {
        let inflight = InFlight::<&'static str>::new(0.9);
        let backend_calls = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..50)
            .map(|i| {
                let inflight = Arc::clone(&inflight);
                let calls = Arc::clone(&backend_calls);
                tokio::spawn(async move {
                    let v = if i % 2 == 0 { A } else { A_ISH };
                    match inflight.join("p", &v) {
                        Joined::Leader(lead) => {
                            calls.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            lead.finish("answer");
                            "answer"
                        }
                        Joined::Follower(mut rx) => rx.recv().await.unwrap(),
                    }
                })
            })
            .collect();

        for t in tasks {
            assert_eq!(t.await.unwrap(), "answer");
        }
        assert_eq!(backend_calls.load(Ordering::SeqCst), 1);
        assert_eq!(inflight.len(), 0);
    }
}
