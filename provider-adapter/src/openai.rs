//! OpenAI Chat Completions API.

use serde_json::{json, Map, Value};
use tonic::Status;

use crate::{
    pb::{FinishReason, GenerateRequest, GenerateResponse, Provider},
    role_name, send_json, token_count,
};

pub struct OpenAi {
    pub http: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
}

impl OpenAi {
    pub async fn generate(&self, model: &str, req: &GenerateRequest) -> Result<GenerateResponse, Status> {
        let builder = self
            .http
            .post(format!("{}/chat/completions", self.base_url.trim_end_matches('/')))
            .bearer_auth(&self.api_key)
            .json(&request_body(model, req));
        let resp = send_json(builder).await?;
        Ok(parse_response(&resp))
    }
}

fn request_body(model: &str, req: &GenerateRequest) -> Value {
    let messages: Vec<_> = req
        .messages
        .iter()
        .map(|m| json!({ "role": role_name(m.role()), "content": m.content }))
        .collect();

    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("messages".into(), Value::Array(messages));
    // `max_tokens` is deprecated in favour of this, and rejected by reasoning models.
    if let Some(n) = req.max_tokens {
        out.insert("max_completion_tokens".into(), json!(n));
    }
    if let Some(t) = req.temperature {
        out.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        out.insert("top_p".into(), json!(p));
    }
    if !req.stop.is_empty() {
        out.insert("stop".into(), json!(req.stop));
    }
    Value::Object(out)
}

fn parse_response(resp: &Value) -> GenerateResponse {
    let choice = resp.pointer("/choices/0");
    let finish_reason = match choice.and_then(|c| c.get("finish_reason")).and_then(Value::as_str) {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    };
    let usage = resp.get("usage");

    GenerateResponse {
        id: resp.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
        model: resp.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        text: choice
            .and_then(|c| c.pointer("/message/content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        finish_reason: finish_reason.into(),
        prompt_tokens: token_count(usage, "prompt_tokens"),
        completion_tokens: token_count(usage, "completion_tokens"),
        provider: Provider::Openai.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::{Message, Role};

    #[test]
    fn request_uses_max_completion_tokens() {
        let req = GenerateRequest {
            messages: vec![Message { role: Role::User.into(), content: "hi".into() }],
            max_tokens: Some(50),
            ..Default::default()
        };
        let out = request_body("gpt-4o-mini", &req);
        assert_eq!(out["max_completion_tokens"], 50);
        assert!(out.get("max_tokens").is_none());
        assert_eq!(out["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[test]
    fn response_maps_finish_reason_and_usage() {
        let out = parse_response(&json!({
            "id": "chatcmpl-1", "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2},
        }));
        assert_eq!(out.text, "Paris.");
        assert_eq!(out.finish_reason(), FinishReason::Stop);
        assert_eq!((out.prompt_tokens, out.completion_tokens), (9, 2));
    }
}
