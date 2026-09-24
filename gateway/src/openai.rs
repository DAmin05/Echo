//! The gateway's public API is OpenAI's chat completions shape. This module
//! converts between it and the internal provider-adapter protobufs.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use serde_json::{json, Value};
use tonic::{Code, Status};

use crate::clients::pb::provider::{FinishReason, GenerateRequest, GenerateResponse, Message, Role};

/// Assumes the body already passed `handlers::cache_inputs` validation.
pub fn to_generate_request(body: &Value) -> GenerateRequest {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .map(|msgs| {
            msgs.iter()
                .map(|m| {
                    let role = match m.get("role").and_then(Value::as_str) {
                        Some("system" | "developer") => Role::System,
                        Some("assistant") => Role::Assistant,
                        _ => Role::User,
                    };
                    Message { role: role.into(), content: content_text(m.get("content").unwrap_or(&Value::Null)) }
                })
                .collect()
        })
        .unwrap_or_default();

    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        _ => Vec::new(),
    };

    GenerateRequest {
        // Unspecified: provider-adapter picks from the model name.
        provider: 0,
        model: body.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        messages,
        max_tokens: body
            .get("max_completion_tokens")
            .or_else(|| body.get("max_tokens"))
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok()),
        temperature: body.get("temperature").and_then(Value::as_f64),
        top_p: body.get("top_p").and_then(Value::as_f64),
        stop,
    }
}

pub fn to_chat_completion(resp: &GenerateResponse) -> Value {
    let finish_reason = match resp.finish_reason() {
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        _ => "stop",
    };
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    json!({
        "id": resp.id,
        "object": "chat.completion",
        "created": created,
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": resp.text },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.prompt_tokens + resp.completion_tokens,
        },
    })
}

/// provider-adapter's gRPC status → the HTTP status and OpenAI error type a
/// client would have seen from the provider directly.
pub fn http_error(status: &Status) -> (StatusCode, &'static str) {
    match status.code() {
        Code::InvalidArgument | Code::FailedPrecondition => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        Code::Unauthenticated => (StatusCode::UNAUTHORIZED, "authentication_error"),
        Code::PermissionDenied => (StatusCode::FORBIDDEN, "permission_error"),
        Code::NotFound => (StatusCode::NOT_FOUND, "not_found_error"),
        Code::ResourceExhausted => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        Code::DeadlineExceeded => (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
        Code::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "api_error"),
        _ => (StatusCode::BAD_GATEWAY, "api_error"),
    }
}

/// Message content is either a string or an array of typed parts.
pub fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_maps_roles_and_params() {
        let req = to_generate_request(&json!({
            "model": "claude-haiku-4-5",
            "max_completion_tokens": 100,
            "temperature": 0.2,
            "stop": "END",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }));
        assert_eq!(req.model, "claude-haiku-4-5");
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.2));
        assert_eq!(req.stop, vec!["END"]);
        assert_eq!(req.messages[0].role(), Role::System);
        assert_eq!(req.messages[1].role(), Role::User);
        assert_eq!(req.messages[1].content, "hi");
    }

    #[test]
    fn response_has_openai_shape() {
        let out = to_chat_completion(&GenerateResponse {
            id: "msg_1".into(),
            model: "claude-haiku-4-5".into(),
            text: "Paris.".into(),
            finish_reason: FinishReason::Length.into(),
            prompt_tokens: 10,
            completion_tokens: 3,
            provider: 1,
        });
        assert_eq!(out["choices"][0]["message"]["content"], "Paris.");
        assert_eq!(out["choices"][0]["finish_reason"], "length");
        assert_eq!(out["usage"]["total_tokens"], 13);
    }

    #[test]
    fn provider_errors_map_back_to_http() {
        assert_eq!(http_error(&Status::unauthenticated("x")).0, StatusCode::UNAUTHORIZED);
        assert_eq!(http_error(&Status::resource_exhausted("x")).0, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(http_error(&Status::unavailable("x")).0, StatusCode::SERVICE_UNAVAILABLE);
    }
}
