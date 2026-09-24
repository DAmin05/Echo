use std::{collections::BTreeMap, time::Instant};

use axum::{
    extract::State,
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::{
    clients::{pb::provider::FinishReason, Clients},
    openai::{content_text, http_error, to_chat_completion, to_generate_request},
};

/// `POST /v1/chat/completions` — OpenAI-compatible, with a semantic cache in
/// front of the LLM. Orchestrates the three internal services:
///
///   embed (embedding-svc) → query (cache-svc) → hit: return
///                                              → miss: generate (provider-adapter)
///                                                      → store (cache-svc) → return
///
/// The cache is an optimisation, so it fails open: if embedding-svc or
/// cache-svc is down or slow, the request still goes to the provider.
pub async fn chat_completions(
    State(clients): State<Clients>,
    Json(body): Json<Value>,
) -> Result<Response, AppError> {
    let started = Instant::now();
    let (prompt, params) = cache_inputs(&body)?;

    let t = Instant::now();
    let vector = match clients.embed(&prompt).await {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(code = ?e.code(), error = e.message(), "embedding-svc failed; bypassing cache");
            None
        }
    };
    let embed_ms = t.elapsed().as_millis();

    let t = Instant::now();
    // Set to None if cache-svc fails, so we don't also try to store into it:
    // that would add a second timeout to a request already on the slow path.
    let mut vector = vector;
    if let Some(v) = &vector {
        match clients.cache_query(v.clone(), &params).await {
            Ok(Some(hit)) => {
                info!(
                    cache_hit = true,
                    similarity = hit.similarity,
                    entry = %hit.entry_id,
                    embed_ms,
                    lookup_ms = t.elapsed().as_millis(),
                    total_ms = started.elapsed().as_millis(),
                    "served from cache"
                );
                let cached: Value = serde_json::from_str(&hit.response)
                    .map_err(|e| AppError::Internal(format!("corrupt cached response: {e}")))?;
                return Ok(respond(StatusCode::OK, cached, "hit", Some(hit.similarity)));
            }
            Ok(None) => {}
            Err(e) => {
                warn!(code = ?e.code(), error = e.message(), "cache-svc query failed; bypassing cache");
                vector = None;
            }
        }
    }
    let lookup_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let generated = clients.generate(to_generate_request(&body)).await.map_err(|s| {
        let (status, kind) = http_error(&s);
        warn!(code = ?s.code(), error = s.message(), "provider-adapter failed");
        AppError::Upstream { status, kind, message: s.message().to_string() }
    })?;
    let llm_ms = t.elapsed().as_millis();
    let completion = to_chat_completion(&generated);

    // Only complete answers are cached: refusals and max_tokens truncations
    // are returned but never replayed to later callers. "miss" means the
    // answer is now cached; "bypass" means the cache played no part.
    let cache_status = match (vector, generated.finish_reason()) {
        (Some(v), FinishReason::Stop) => match clients.cache_store(v, &prompt, &params, completion.to_string()).await {
            Ok(_) => "miss",
            Err(e) => {
                warn!(code = ?e.code(), error = e.message(), "cache-svc store failed");
                "bypass"
            }
        },
        _ => "bypass",
    };

    info!(
        cache_hit = false,
        cache = cache_status,
        model = %generated.model,
        finish = generated.finish_reason().as_str_name(),
        embed_ms,
        lookup_ms,
        llm_ms,
        total_ms = started.elapsed().as_millis(),
        "forwarded to provider"
    );
    Ok(respond(StatusCode::OK, completion, cache_status, None))
}

/// `GET /stats` — running hit/miss counters from cache-svc.
pub async fn stats(State(clients): State<Clients>) -> Result<Json<Value>, AppError> {
    let s = clients.cache_stats().await.map_err(|e| AppError::Upstream {
        status: StatusCode::SERVICE_UNAVAILABLE,
        kind: "api_error",
        message: format!("cache-svc: {}", e.message()),
    })?;
    let total = s.hits + s.misses;
    Ok(Json(json!({
        "hits": s.hits,
        "misses": s.misses,
        "hit_rate": if total == 0 { 0.0 } else { s.hits as f64 / total as f64 },
    })))
}

pub async fn healthz() -> &'static str {
    "ok"
}

/// Splits a chat request into the text to embed and the exact-match key.
///
/// - `prompt`: the final user message — the part expected to vary in wording.
/// - `params`: everything else as canonical JSON — model, sampling settings,
///   and all earlier messages (system prompt, prior turns). Only entries with
///   identical params can match, so a shared system prompt can't make unrelated
///   questions look similar, and the same question asked in a different
///   conversation never gets an answer written for another context.
fn cache_inputs(body: &Value) -> Result<(String, String), AppError> {
    let obj = body
        .as_object()
        .ok_or_else(|| AppError::BadRequest("request body must be a JSON object".into()))?;
    if obj.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(AppError::BadRequest("stream=true is not supported yet".into()));
    }
    if obj.contains_key("tools") || obj.contains_key("functions") {
        return Err(AppError::BadRequest("tool calling is not supported yet".into()));
    }
    if !obj.get("model").is_some_and(Value::is_string) {
        return Err(AppError::BadRequest("`model` must be a string".into()));
    }
    let messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| AppError::BadRequest("`messages` must be a non-empty array".into()))?;
    if messages.iter().any(|m| {
        let role = m.get("role").and_then(Value::as_str);
        !matches!(role, Some("system" | "developer" | "user" | "assistant"))
    }) {
        return Err(AppError::BadRequest(
            "each message needs a role of system, developer, user or assistant".into(),
        ));
    }

    let (last, context) = messages.split_last().expect("checked non-empty");
    if last.get("role").and_then(Value::as_str) != Some("user") {
        return Err(AppError::BadRequest("the last message must have role `user`".into()));
    }
    let prompt = content_text(last.get("content").unwrap_or(&Value::Null));

    let mut params = obj.clone();
    params.insert("messages".into(), Value::Array(context.to_vec()));
    Ok((prompt, canonical_json(&Value::Object(params))))
}

/// JSON with object keys sorted at every level, so logically equal params
/// always produce the same string regardless of client key order.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map.iter().map(|(k, v)| (k, canonical_json(v))).collect();
            let fields: Vec<_> = sorted
                .into_iter()
                .map(|(k, v)| format!("{}:{v}", Value::String(k.clone())))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => {
            format!("[{}]", items.iter().map(canonical_json).collect::<Vec<_>>().join(","))
        }
        other => other.to_string(),
    }
}

fn respond(status: StatusCode, body: Value, cache: &'static str, similarity: Option<f32>) -> Response {
    let mut res = (status, Json(body)).into_response();
    let headers = res.headers_mut();
    headers.insert("x-echo-cache", HeaderValue::from_static(cache));
    if let Some(score) = similarity {
        if let Ok(v) = HeaderValue::from_str(&format!("{score:.4}")) {
            headers.insert("x-echo-similarity", v);
        }
    }
    res
}

/// Errors rendered in OpenAI's `{"error": {...}}` shape so SDK clients parse them.
pub enum AppError {
    BadRequest(String),
    Upstream { status: StatusCode, kind: &'static str, message: String },
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, kind, message) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            AppError::Upstream { status, kind, message } => (status, kind, message),
            AppError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m),
        };
        (status, Json(json!({ "error": { "message": message, "type": kind } }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_last_user_message_and_keys_on_the_rest() {
        let a = json!({"model": "m", "max_tokens": 50, "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]});
        let b = json!({"messages": [
            {"content": "be brief", "role": "system"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ], "max_tokens": 50, "model": "m"});
        let (pa, ka) = cache_inputs(&a).ok().unwrap();
        let (pb, kb) = cache_inputs(&b).ok().unwrap();
        assert_eq!((pa.as_str(), pb.as_str()), ("hi", "hello"));
        assert_eq!(ka, kb, "key order must not matter");
        assert_eq!(
            ka,
            r#"{"max_tokens":50,"messages":[{"content":"be brief","role":"system"}],"model":"m"}"#
        );
    }

    #[test]
    fn different_context_gives_different_key() {
        let with_system = |s: &str| {
            json!({"model": "m", "messages": [
                {"role": "system", "content": s},
                {"role": "user", "content": "hi"},
            ]})
        };
        let (_, k1) = cache_inputs(&with_system("be brief")).ok().unwrap();
        let (_, k2) = cache_inputs(&with_system("be verbose")).ok().unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn rejects_unsupported_requests() {
        let user = json!([{"role": "user", "content": "x"}]);
        for body in [
            json!({"model": "m", "stream": true, "messages": user}),
            json!({"model": "m", "tools": [], "messages": user}),
            json!({"model": "m", "messages": []}),
            json!({"messages": user}),
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}),
            json!({"model": "m", "messages": [{"role": "tool", "content": "x"}, {"role": "user", "content": "y"}]}),
        ] {
            assert!(cache_inputs(&body).is_err(), "should reject {body}");
        }
    }
}
