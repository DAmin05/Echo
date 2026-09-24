//! Ollama's native chat API (`/api/chat`), for locally hosted models.

use serde_json::{json, Map, Value};
use tonic::Status;
use uuid::Uuid;

use crate::{
    pb::{FinishReason, GenerateRequest, GenerateResponse, Provider},
    role_name, send_json, token_count,
};

pub struct Ollama {
    pub http: reqwest::Client,
    pub base_url: String,
}

impl Ollama {
    pub async fn generate(&self, model: &str, req: &GenerateRequest) -> Result<GenerateResponse, Status> {
        let builder = self
            .http
            .post(format!("{}/api/chat", self.base_url.trim_end_matches('/')))
            .json(&request_body(model, req));
        let resp = send_json(builder).await?;
        Ok(parse_response(&resp))
    }
}

/// Sampling settings live under `options`, and max tokens is `num_predict`.
fn request_body(model: &str, req: &GenerateRequest) -> Value {
    let messages: Vec<_> = req
        .messages
        .iter()
        .map(|m| json!({ "role": role_name(m.role()), "content": m.content }))
        .collect();

    let mut options = Map::new();
    if let Some(n) = req.max_tokens {
        options.insert("num_predict".into(), json!(n));
    }
    if let Some(t) = req.temperature {
        options.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        options.insert("top_p".into(), json!(p));
    }
    if !req.stop.is_empty() {
        options.insert("stop".into(), json!(req.stop));
    }

    json!({ "model": model, "messages": messages, "stream": false, "options": options })
}

fn parse_response(resp: &Value) -> GenerateResponse {
    let finish_reason = match resp.get("done_reason").and_then(Value::as_str) {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        _ => FinishReason::Other,
    };

    GenerateResponse {
        // Ollama doesn't return an id.
        id: format!("ollama-{}", Uuid::new_v4()),
        model: resp.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        text: resp
            .pointer("/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        finish_reason: finish_reason.into(),
        prompt_tokens: token_count(Some(resp), "prompt_eval_count"),
        completion_tokens: token_count(Some(resp), "eval_count"),
        provider: Provider::Ollama.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::{Message, Role};

    #[test]
    fn request_puts_sampling_under_options() {
        let req = GenerateRequest {
            messages: vec![Message { role: Role::User.into(), content: "hi".into() }],
            max_tokens: Some(64),
            temperature: Some(0.0),
            ..Default::default()
        };
        let out = request_body("llama3.2", &req);
        assert_eq!(out["stream"], false);
        assert_eq!(out["options"], json!({"num_predict": 64, "temperature": 0.0}));
    }

    #[test]
    fn response_maps_counts_and_done_reason() {
        let out = parse_response(&json!({
            "model": "llama3.2", "message": {"role": "assistant", "content": "Paris."},
            "done_reason": "length", "prompt_eval_count": 11, "eval_count": 4,
        }));
        assert_eq!(out.text, "Paris.");
        assert_eq!(out.finish_reason(), FinishReason::Length);
        assert_eq!((out.prompt_tokens, out.completion_tokens), (11, 4));
    }
}
