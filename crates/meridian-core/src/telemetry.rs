//! OpenTelemetry OTLP export wiring (feature `otel`).
//!
//! Meridian already emits structured spans through `tracing` on the hot
//! scheduling path. This module bridges those spans to an OTLP collector
//! without adding any new instrumentation: [`install`] builds an OTLP/HTTP
//! span exporter, registers it as the OpenTelemetry global provider, and
//! attaches a `tracing` layer so existing spans flow out unchanged.
//!
//! The HTTP + blocking transport is deliberate — it keeps `meridian-core`
//! free of an async runtime, which the synchronous scheduler core does not
//! otherwise need.

use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{Config, TracerProvider};
use tracing_subscriber::prelude::*;

/// Boxed error so telemetry wiring failures do not leak transport-specific
/// types into the core error enum.
pub type TelemetryError = Box<dyn std::error::Error + Send + Sync>;

/// Build an OTLP/HTTP tracer provider exporting to `endpoint`
/// (e.g. `http://localhost:4318/v1/traces`), tagged with `service_name`.
///
/// The caller owns the returned provider; dropping it flushes pending spans.
/// Most callers want [`install`] instead, which also wires the `tracing`
/// bridge and sets the OpenTelemetry global.
///
/// # Errors
/// Returns an error if the OTLP exporter cannot be constructed.
pub fn otlp_tracer_provider(
    endpoint: &str,
    service_name: &str,
) -> Result<TracerProvider, TelemetryError> {
    let provider = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .http()
                .with_endpoint(endpoint),
        )
        .with_trace_config(Config::default().with_resource(Resource::new(vec![
            KeyValue::new("service.name", service_name.to_string()),
        ])))
        .install_simple()?;
    Ok(provider)
}

/// Install OTLP export for the process: build the provider, register it as the
/// OpenTelemetry global, and attach a `tracing` layer so the spans the
/// scheduler already emits are exported.
///
/// Uses `try_init` for the `tracing` subscriber, so calling this when a
/// subscriber is already installed is a no-op for the subscriber (the OTLP
/// provider is still registered as the global).
///
/// # Errors
/// Returns an error if the OTLP exporter cannot be constructed.
pub fn install(endpoint: &str, service_name: &str) -> Result<(), TelemetryError> {
    let provider = otlp_tracer_provider(endpoint, service_name)?;
    let tracer = provider.tracer("meridian");
    opentelemetry::global::set_tracer_provider(provider);

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let _ = tracing_subscriber::registry().with(otel_layer).try_init();
    Ok(())
}

/// Flush and shut down the OpenTelemetry global provider. Call once on
/// graceful shutdown so batched spans are not lost.
pub fn shutdown() {
    opentelemetry::global::shutdown_tracer_provider();
}
