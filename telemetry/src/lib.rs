//! Shared observability setup for Echo's Rust services.
//!
//! - `init` installs logging (as before) plus an OpenTelemetry layer that
//!   exports spans over OTLP/gRPC when `OTEL_EXPORTER_OTLP_ENDPOINT` is set
//!   (e.g. `http://jaeger:4317`). Unset, services log locally and export nothing.
//! - `inject` is a tonic client interceptor that writes the current span's
//!   W3C `traceparent` into outgoing gRPC metadata.
//! - `server_span` starts a server span for an incoming gRPC request, parented
//!   to the caller's `traceparent`, so one request is one trace across services.

use std::env;

use anyhow::Result;
use opentelemetry::{
    global,
    propagation::{Extractor, Injector},
    trace::TracerProvider as _,
};
use opentelemetry_sdk::{propagation::TraceContextPropagator, trace::SdkTracerProvider, Resource};
use tonic::{metadata::MetadataMap, Request, Status};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Flushes buffered spans on drop. Keep it alive for the life of `main`.
pub struct Guard(Option<SdkTracerProvider>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(provider) = self.0.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("failed to flush traces: {e}");
            }
        }
    }
}

/// `service` names the service in traces; `default_filter` is the log filter
/// used when `RUST_LOG` is unset.
pub fn init(service: &'static str, default_filter: &str) -> Result<Guard> {
    global::set_text_map_propagator(TraceContextPropagator::new());
    // `telemetry` must pass the filter too: server_span's spans are created here.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| format!("{default_filter},telemetry=info").into());
    let fmt = tracing_subscriber::fmt::layer();

    if env::var("OTEL_EXPORTER_OTLP_ENDPOINT").map_or(true, |v| v.is_empty()) {
        tracing_subscriber::registry().with(filter).with(fmt).init();
        return Ok(Guard(None));
    }

    // Reads OTEL_EXPORTER_OTLP_ENDPOINT itself.
    let exporter = opentelemetry_otlp::SpanExporter::builder().with_tonic().build()?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name(service).build())
        .build();
    let otel = tracing_opentelemetry::layer().with_tracer(provider.tracer(service));
    tracing_subscriber::registry().with(filter).with(fmt).with(otel).init();
    global::set_tracer_provider(provider.clone());
    Ok(Guard(Some(provider)))
}

/// tonic client interceptor: propagates the current span to the callee.
/// Use with `Client::with_interceptor(channel, telemetry::inject)`.
pub fn inject(mut req: Request<()>) -> Result<Request<()>, Status> {
    let cx = Span::current().context();
    global::get_text_map_propagator(|p| p.inject_context(&cx, &mut MetadataInjector(req.metadata_mut())));
    Ok(req)
}

/// For `tonic::transport::Server::builder().trace_fn(telemetry::server_span)`:
/// one span per RPC, named after the method and parented to the caller.
pub fn server_span(req: &http::Request<()>) -> Span {
    let path = req.uri().path();
    let span = tracing::info_span!("grpc", otel.name = %path.trim_start_matches('/'), otel.kind = "server", rpc.method = %path);
    set_parent_from_headers(&span, req.headers());
    span
}

/// Parents `span` to the trace context in incoming HTTP headers, if any.
/// The gateway uses this so a client that sends `traceparent` sees Echo's
/// spans inside its own trace.
pub fn set_parent_from_headers(span: &Span, headers: &http::HeaderMap) {
    let parent = global::get_text_map_propagator(|p| p.extract(&HeaderExtractor(headers)));
    // Only fails if the span is disabled by the filter, in which case there's nothing to parent.
    let _ = span.set_parent(parent);
}

struct MetadataInjector<'a>(&'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(k), Ok(v)) = (key.parse::<tonic::metadata::MetadataKey<_>>(), value.parse()) {
            self.0.insert(k, v);
        }
    }
}

struct HeaderExtractor<'a>(&'a http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}
