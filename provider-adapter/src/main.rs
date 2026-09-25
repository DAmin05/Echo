//! provider-adapter: one gRPC `Generate` call over Anthropic, OpenAI and Ollama.
//!
//! Each provider module translates the normalized request into its own API
//! shape and back. Upstream HTTP errors become gRPC status codes so callers
//! can tell a bad request from an outage without knowing which provider ran.

mod anthropic;
mod ollama;
mod openai;

use std::{env, net::SocketAddr, time::{Duration, Instant}};

use anyhow::Result;
use reqwest::StatusCode;
use serde_json::Value;
use tonic::{transport::Server, Code, Request, Response, Status};
use tracing::{info, info_span, field::Empty, warn, Instrument};

use crate::{anthropic::Anthropic, ollama::Ollama, openai::OpenAi};

pub mod pb {
    tonic::include_proto!("echo.provider.v1");
}
use pb::{
    provider_service_server::{ProviderService, ProviderServiceServer},
    GenerateRequest, GenerateResponse, Provider, Role,
};

/// A provider is `None` when its credentials aren't configured.
struct Adapter {
    anthropic: Option<Anthropic>,
    openai: Option<OpenAi>,
    ollama: Ollama,
}

#[tonic::async_trait]
impl ProviderService for Adapter {
    async fn generate(&self, req: Request<GenerateRequest>) -> Result<Response<GenerateResponse>, Status> {
        let req = req.into_inner();
        if req.messages.is_empty() {
            return Err(Status::invalid_argument("messages must not be empty"));
        }
        if req.messages.iter().any(|m| m.role() == Role::Unspecified) {
            return Err(Status::invalid_argument("every message needs a role"));
        }
        let (provider, model) = resolve(req.provider(), &req.model)?;

        // Attribute names follow OpenTelemetry's GenAI semantic conventions.
        let span = info_span!(
            "llm.generate",
            otel.name = %format!("{} {model}", provider_name(provider)),
            otel.kind = "client",
            gen_ai.system = provider_name(provider),
            gen_ai.request.model = %model,
            gen_ai.response.model = Empty,
            gen_ai.response.finish_reason = Empty,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
        );
        let started = Instant::now();
        let result = async { match provider {
            Provider::Anthropic => {
                let p = self.anthropic.as_ref().ok_or_else(|| not_configured("ANTHROPIC_API_KEY"))?;
                p.generate(&model, &req).await
            }
            Provider::Openai => {
                let p = self.openai.as_ref().ok_or_else(|| not_configured("OPENAI_API_KEY"))?;
                p.generate(&model, &req).await
            }
            Provider::Ollama => self.ollama.generate(&model, &req).await,
            Provider::Unspecified => unreachable!("resolve() always picks a provider"),
        } }
        .instrument(span.clone())
        .await;
        let latency_ms = started.elapsed().as_millis();
        if let Ok(r) = &result {
            span.record("gen_ai.response.model", r.model.as_str());
            span.record("gen_ai.response.finish_reason", r.finish_reason().as_str_name());
            span.record("gen_ai.usage.input_tokens", r.prompt_tokens);
            span.record("gen_ai.usage.output_tokens", r.completion_tokens);
        }

        match &result {
            Ok(r) => info!(
                provider = provider.as_str_name(),
                model = %r.model,
                finish = r.finish_reason().as_str_name(),
                prompt_tokens = r.prompt_tokens,
                completion_tokens = r.completion_tokens,
                latency_ms,
                "generated"
            ),
            Err(s) => warn!(provider = provider.as_str_name(), %model, code = ?s.code(), message = s.message(), latency_ms, "provider error"),
        }
        result.map(Response::new)
    }
}

/// Picks the provider and the model name to send it.
///
/// An explicit `provider` wins; otherwise a `provider/` prefix on the model;
/// otherwise the model name decides.
fn resolve(provider: Provider, model: &str) -> Result<(Provider, String), Status> {
    if model.is_empty() {
        return Err(Status::invalid_argument("model must not be empty"));
    }
    if provider != Provider::Unspecified {
        return Ok((provider, model.to_string()));
    }
    if let Some((prefix, rest)) = model.split_once('/') {
        let explicit = match prefix {
            "anthropic" => Some(Provider::Anthropic),
            "openai" => Some(Provider::Openai),
            "ollama" => Some(Provider::Ollama),
            _ => None,
        };
        if let Some(p) = explicit {
            return Ok((p, rest.to_string()));
        }
    }
    let is_openai_reasoning = model.len() > 1
        && model.starts_with('o')
        && model[1..].starts_with(|c: char| c.is_ascii_digit());
    let provider = if model.starts_with("claude-") {
        Provider::Anthropic
    } else if model.starts_with("gpt-") || model.starts_with("chatgpt-") || is_openai_reasoning {
        Provider::Openai
    } else {
        Provider::Ollama
    };
    Ok((provider, model.to_string()))
}

fn provider_name(p: Provider) -> &'static str {
    match p {
        Provider::Anthropic => "anthropic",
        Provider::Openai => "openai",
        Provider::Ollama => "ollama",
        Provider::Unspecified => "unspecified",
    }
}

fn not_configured(var: &str) -> Status {
    Status::failed_precondition(format!("provider not configured: set {var} on provider-adapter"))
}

/// Sends a request and returns the JSON body, or the upstream error as a Status.
pub(crate) async fn send_json(builder: reqwest::RequestBuilder) -> Result<Value, Status> {
    let resp = builder.send().await.map_err(|e| {
        if e.is_timeout() {
            Status::deadline_exceeded(format!("provider timed out: {e}"))
        } else {
            Status::unavailable(format!("provider unreachable: {e}"))
        }
    })?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| Status::unavailable(format!("reading provider response: {e}")))?;
    let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);

    if !status.is_success() {
        return Err(Status::new(grpc_code(status), upstream_message(&body, &text)));
    }
    if body.is_null() {
        return Err(Status::internal(format!("provider returned non-JSON body: {text}")));
    }
    Ok(body)
}

/// Upstream HTTP status → gRPC code. The gateway maps these back to HTTP, so
/// the client sees the same class of error the provider returned.
fn grpc_code(status: StatusCode) -> Code {
    match status.as_u16() {
        400 | 413 | 422 => Code::InvalidArgument,
        401 => Code::Unauthenticated,
        403 => Code::PermissionDenied,
        404 => Code::NotFound,
        408 | 504 => Code::DeadlineExceeded,
        429 => Code::ResourceExhausted,
        _ => Code::Unavailable, // 5xx, including Anthropic's 529 "overloaded"
    }
}

/// Anthropic and OpenAI send `{"error": {"message": ...}}`; Ollama sends `{"error": "..."}`.
fn upstream_message(body: &Value, raw: &str) -> String {
    body.pointer("/error/message")
        .or_else(|| body.get("error"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| raw.chars().take(500).collect())
}

pub(crate) fn token_count(usage: Option<&Value>, key: &str) -> u32 {
    usage.and_then(|u| u.get(key)).and_then(Value::as_u64).unwrap_or(0) as u32
}

pub(crate) fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Assistant => "assistant",
        Role::User | Role::Unspecified => "user",
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let _telemetry = telemetry::init("provider-adapter", "provider_adapter=info")?;

    let addr: SocketAddr = var("PROVIDER_ADAPTER_ADDR", "0.0.0.0:50053").parse()?;
    // Long ceiling: with extended thinking a 16K-token answer can take minutes.
    let http = reqwest::Client::builder().timeout(Duration::from_secs(600)).build()?;
    let key = |name: &str| env::var(name).ok().filter(|k| !k.is_empty());

    let adapter = Adapter {
        anthropic: key("ANTHROPIC_API_KEY").map(|api_key| Anthropic {
            http: http.clone(),
            // Not ANTHROPIC_BASE_URL: other tools (e.g. Claude Code) set that
            // in the environment, and it would silently redirect Echo's calls.
            base_url: var("ECHO_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            api_key,
        }),
        openai: key("OPENAI_API_KEY").map(|api_key| OpenAi {
            http: http.clone(),
            base_url: var("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key,
        }),
        ollama: Ollama { http, base_url: var("OLLAMA_URL", "http://localhost:11434") },
    };

    info!(
        %addr,
        anthropic = adapter.anthropic.is_some(),
        openai = adapter.openai.is_some(),
        ollama_url = %adapter.ollama.base_url,
        "provider-adapter listening"
    );
    Server::builder()
        .trace_fn(telemetry::server_span)
        .add_service(ProviderServiceServer::new(adapter))
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;
    Ok(())
}

fn var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_provider_from_model_name() {
        let cases = [
            ("claude-haiku-4-5", Provider::Anthropic, "claude-haiku-4-5"),
            ("gpt-4o-mini", Provider::Openai, "gpt-4o-mini"),
            ("o3-mini", Provider::Openai, "o3-mini"),
            ("llama3.2", Provider::Ollama, "llama3.2"),
            ("ollama/gpt-oss", Provider::Ollama, "gpt-oss"),
            ("anthropic/claude-sonnet-5", Provider::Anthropic, "claude-sonnet-5"),
            // Unknown prefixes are part of the model name (e.g. HF-style Ollama tags).
            ("hf.co/foo/bar", Provider::Ollama, "hf.co/foo/bar"),
            ("orca-mini", Provider::Ollama, "orca-mini"),
        ];
        for (model, provider, name) in cases {
            assert_eq!(resolve(Provider::Unspecified, model).unwrap(), (provider, name.to_string()), "{model}");
        }
    }

    #[test]
    fn explicit_provider_wins() {
        assert_eq!(
            resolve(Provider::Ollama, "claude-haiku-4-5").unwrap(),
            (Provider::Ollama, "claude-haiku-4-5".to_string())
        );
        assert!(resolve(Provider::Unspecified, "").is_err());
    }

    #[test]
    fn maps_http_errors_to_grpc_codes() {
        assert_eq!(grpc_code(StatusCode::BAD_REQUEST), Code::InvalidArgument);
        assert_eq!(grpc_code(StatusCode::UNAUTHORIZED), Code::Unauthenticated);
        assert_eq!(grpc_code(StatusCode::TOO_MANY_REQUESTS), Code::ResourceExhausted);
        assert_eq!(grpc_code(StatusCode::from_u16(529).unwrap()), Code::Unavailable);
    }

    #[test]
    fn extracts_error_messages_from_each_provider_shape() {
        let nested = serde_json::json!({"type": "error", "error": {"type": "x", "message": "bad key"}});
        let flat = serde_json::json!({"error": "model 'x' not found"});
        assert_eq!(upstream_message(&nested, ""), "bad key");
        assert_eq!(upstream_message(&flat, ""), "model 'x' not found");
        assert_eq!(upstream_message(&Value::Null, "plain text"), "plain text");
    }
}
