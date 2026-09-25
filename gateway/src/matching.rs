//! Deciding whether a candidate (a cached entry or an in-flight request) asks
//! the same question as the incoming request.
//!
//! Embedding similarity alone can't do this safely: the eval set showed
//! "convert 10 miles to km" vs "10 km to miles" at cosine 0.993, above almost
//! every genuine paraphrase. So by default matching is two-stage:
//!
//!   1. recall: vector search finds candidates at a loose similarity (0.70),
//!   2. precision: a cross-encoder reads the query and each candidate together
//!      and must score it as a duplicate (>= 0.80).
//!
//! `VERIFY_ENABLED=false` falls back to a single vector-similarity threshold.

use tracing::{debug, warn};

use crate::clients::Clients;

#[derive(Clone, Copy)]
pub enum Matcher {
    Verify { candidate_threshold: f32, verify_threshold: f32 },
    Similarity { threshold: f32 },
}

/// A candidate that passed matching.
#[derive(Clone, Copy)]
pub struct Accepted {
    /// Index into the candidates passed to `accept`.
    pub index: usize,
    /// Cross-encoder score, when verification ran.
    pub verify_score: Option<f32>,
}

impl Matcher {
    /// Minimum vector similarity for something to be considered at all.
    pub fn candidate_threshold(&self) -> f32 {
        match *self {
            Matcher::Verify { candidate_threshold, .. } => candidate_threshold,
            Matcher::Similarity { threshold } => threshold,
        }
    }

    /// The candidates that ask the same question as `prompt`, best first.
    /// `candidates` are `(prompt, vector similarity)`.
    ///
    /// Fails closed: if the verifier is unavailable, nothing is accepted and
    /// the request goes to the provider. Serving an unverified answer is the
    /// failure this step exists to prevent.
    pub async fn accept(&self, clients: &Clients, prompt: &str, candidates: &[(&str, f32)]) -> Vec<Accepted> {
        if candidates.is_empty() {
            return Vec::new();
        }
        match *self {
            Matcher::Similarity { threshold } => {
                let mut accepted: Vec<_> = (0..candidates.len())
                    .filter(|&i| candidates[i].1 >= threshold)
                    .collect();
                accepted.sort_by(|&a, &b| candidates[b].1.total_cmp(&candidates[a].1));
                accepted.into_iter().map(|index| Accepted { index, verify_score: None }).collect()
            }
            Matcher::Verify { verify_threshold, .. } => {
                let texts: Vec<String> = candidates.iter().map(|(p, _)| p.to_string()).collect();
                let scores = match clients.score_duplicates(prompt, texts).await {
                    Ok(s) if s.len() == candidates.len() => s,
                    Ok(s) => {
                        warn!(expected = candidates.len(), got = s.len(), "verifier returned wrong number of scores");
                        return Vec::new();
                    }
                    Err(e) => {
                        warn!(code = ?e.code(), error = e.message(), "verifier failed; treating candidates as non-matches");
                        return Vec::new();
                    }
                };
                for ((candidate, similarity), score) in candidates.iter().zip(&scores) {
                    debug!(%candidate, similarity, score, accepted = *score >= verify_threshold, "verified candidate");
                }
                let mut accepted: Vec<_> = (0..candidates.len())
                    .filter(|&i| scores[i] >= verify_threshold)
                    .collect();
                accepted.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                accepted
                    .into_iter()
                    .map(|index| Accepted { index, verify_score: Some(scores[index]) })
                    .collect()
            }
        }
    }
}

impl std::fmt::Display for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Matcher::Verify { candidate_threshold, verify_threshold } => {
                write!(f, "similarity>={candidate_threshold} then verifier>={verify_threshold}")
            }
            Matcher::Similarity { threshold } => write!(f, "similarity>={threshold}"),
        }
    }
}
