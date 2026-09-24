//! Claude Messages API. Rust has no official Anthropic SDK, so this speaks
//! the REST API directly.

use serde_json::{json, Map, Value};
use tonic::Status;

use crate::{
    pb::{FinishReason, GenerateRequest, GenerateResponse, Provider, Role},
    send_json, token_count,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Claude requires max_tokens; used when the caller doesn't set one.
const DEFAULT_MAX_TOKENS: u32 = 16_000;
/// Opt-in for server-side refusal fallbacks (`fallbacks: "default"`).
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub struct Anthropic {
    pub http: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
}

impl Anthropic {
    pub async fn generate(&self, model: &str, req: &GenerateRequest) -> Result<GenerateResponse, Status> {
        let body = request_body(model, req);
        let mut builder = self
            .http
            .post(format!("{}/v1/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION);
        if body.get("fallbacks").is_some() {
            builder = builder.header("anthropic-beta", FALLBACK_BETA);
        }
        let msg = send_json(builder.json(&body)).await?;
        Ok(parse_response(&msg))
    }
}

/// System messages move to the top-level `system` field.
fn request_body(model: &str, req: &GenerateRequest) -> Value {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for m in &req.messages {
        match m.role() {
            Role::System => system.push(m.content.as_str()),
            Role::Assistant => messages.push(json!({ "role": "assistant", "content": m.content })),
            Role::User | Role::Unspecified => messages.push(json!({ "role": "user", "content": m.content })),
        }
    }

    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("max_tokens".into(), json!(req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)));
    out.insert("messages".into(), Value::Array(messages));
    if !system.is_empty() {
        out.insert("system".into(), json!(system.join("\n\n")));
    }
    // Passed through as sent; models that removed sampling params (Opus 5+)
    // reject them with a 400, which reaches the caller as INVALID_ARGUMENT.
    if let Some(t) = req.temperature {
        out.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        out.insert("top_p".into(), json!(p));
    }
    if !req.stop.is_empty() {
        out.insert("stop_sequences".into(), json!(req.stop));
    }
    // Models with safety classifiers can decline with stop_reason "refusal";
    // "default" lets the API retry on Anthropic's recommended fallback model.
    if model.starts_with("claude-opus-5") || model.starts_with("claude-fable-5") {
        out.insert("fallbacks".into(), json!("default"));
    }
    Value::Object(out)
}

/// Only `text` blocks become the answer; thinking and fallback-marker blocks are dropped.
fn parse_response(msg: &Value) -> GenerateResponse {
    let text = msg
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<String>()
        })
        .unwrap_or_default();

    let finish_reason = match msg.get("stop_reason").and_then(Value::as_str) {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    };
    let usage = msg.get("usage");

    GenerateResponse {
        id: msg.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
        model: msg.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        text,
        finish_reason: finish_reason.into(),
        prompt_tokens: token_count(usage, "input_tokens"),
        completion_tokens: token_count(usage, "output_tokens"),
        provider: Provider::Anthropic.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::Message;

    fn msg(role: Role, content: &str) -> Message {
        Message { role: role.into(), content: content.into() }
    }

    #[test]
    fn request_moves_system_and_maps_params() {
        let req = GenerateRequest {
            messages: vec![msg(Role::System, "be brief"), msg(Role::User, "hi")],
            max_tokens: Some(200),
            stop: vec!["END".into()],
            ..Default::default()
        };
        let out = request_body("claude-opus-5", &req);
        assert_eq!(out["system"], "be brief");
        assert_eq!(out["messages"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(out["max_tokens"], 200);
        assert_eq!(out["stop_sequences"], json!(["END"]));
        assert_eq!(out["fallbacks"], "default");
        assert!(out.get("temperature").is_none());
    }

    #[test]
    fn request_defaults_max_tokens_and_skips_fallbacks_for_other_models() {
        let req = GenerateRequest { messages: vec![msg(Role::User, "hi")], ..Default::default() };
        let out = request_body("claude-haiku-4-5", &req);
        assert_eq!(out["max_tokens"], DEFAULT_MAX_TOKENS);
        assert!(out.get("fallbacks").is_none());
    }

    #[test]
    fn response_keeps_only_text_blocks() {
        let out = parse_response(&json!({
            "id": "msg_1", "model": "claude-haiku-4-5", "stop_reason": "max_tokens",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Paris."}],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }));
        assert_eq!(out.text, "Paris.");
        assert_eq!(out.finish_reason(), FinishReason::Length);
        assert_eq!((out.prompt_tokens, out.completion_tokens), (10, 3));
    }
}
