//! Prometheus metrics + a tiny HTTP exporter.
//!
//! The exporter is intentionally minimal — a single hyper handler
//! serving `/metrics` (plain Prometheus text format) and `/health`
//! (returns 200 OK as long as the process is up). No JSON, no
//! authentication, no /admin endpoints. Bind to a non-public port
//! and scrape from your monitoring system.
//!
//! Counters:
//! - `duxx_resp_connections_total`     — TCP connections ever accepted
//! - `duxx_resp_commands_total{cmd}`   — command count, label = command name
//! - `duxx_resp_errors_total{kind}`    — server-side errors, label = kind
//! - `duxx_resp_remembers_total`       — REMEMBER calls
//! - `duxx_resp_recalls_total`         — RECALL calls
//!
//! Gauges:
//! - `duxx_resp_active_connections`    — currently-open TCP connections
//! - `duxx_resp_memory_count`          — memories in the store
//! - `duxx_resp_session_count`         — sessions tracked
//!
//! Histograms:
//! - `duxx_resp_command_duration_seconds{cmd}` — per-command latency

use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus::{
    register_histogram_vec_with_registry, register_int_counter_vec_with_registry,
    register_int_counter_with_registry, register_int_gauge_with_registry, Encoder, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, Registry, TextEncoder,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Bundle of all metrics surfaced by `duxx-server`.
#[derive(Clone)]
pub struct Metrics {
    pub registry: Arc<Registry>,
    pub connections_total: IntCounter,
    pub active_connections: IntGauge,
    pub commands_total: IntCounterVec,
    pub command_duration: HistogramVec,
    pub errors_total: IntCounterVec,
    pub remembers_total: IntCounter,
    pub recalls_total: IntCounter,
    pub memory_count: IntGauge,
    pub session_count: IntGauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Arc::new(Registry::new());

        let connections_total = register_int_counter_with_registry!(
            "duxx_resp_connections_total",
            "Total TCP connections accepted since boot",
            registry
        )
        .unwrap();
        let active_connections = register_int_gauge_with_registry!(
            "duxx_resp_active_connections",
            "Currently-open TCP connections",
            registry
        )
        .unwrap();
        let commands_total = register_int_counter_vec_with_registry!(
            "duxx_resp_commands_total",
            "Command count, by command name",
            &["cmd"],
            registry
        )
        .unwrap();
        let command_duration = register_histogram_vec_with_registry!(
            "duxx_resp_command_duration_seconds",
            "Per-command server-side latency",
            &["cmd"],
            // Buckets in seconds: 100us, 250us, 500us, 1ms, 2.5ms, 5ms, 10ms, 25ms, 100ms.
            vec![
                0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.1
            ],
            registry
        )
        .unwrap();
        let errors_total = register_int_counter_vec_with_registry!(
            "duxx_resp_errors_total",
            "Server-side error count, by kind",
            &["kind"],
            registry
        )
        .unwrap();
        let remembers_total = register_int_counter_with_registry!(
            "duxx_resp_remembers_total",
            "REMEMBER calls",
            registry
        )
        .unwrap();
        let recalls_total = register_int_counter_with_registry!(
            "duxx_resp_recalls_total",
            "RECALL calls",
            registry
        )
        .unwrap();
        let memory_count = register_int_gauge_with_registry!(
            "duxx_resp_memory_count",
            "Memories currently in the store",
            registry
        )
        .unwrap();
        let session_count = register_int_gauge_with_registry!(
            "duxx_resp_session_count",
            "Sessions currently tracked",
            registry
        )
        .unwrap();

        Self {
            registry,
            connections_total,
            active_connections,
            commands_total,
            command_duration,
            errors_total,
            remembers_total,
            recalls_total,
            memory_count,
            session_count,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Bind a tiny hyper server to `addr` and serve `/metrics` + `/health`.
/// Loops until the process exits — wrap in `tokio::spawn` from the
/// caller. Logs and ignores per-connection errors.
pub async fn serve(metrics: Metrics, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "metrics endpoint listening (/metrics, /health)");
    loop {
        let (stream, _) = listener.accept().await?;
        let metrics = metrics.clone();
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req: Request<Incoming>| {
                        let m = metrics.clone();
                        async move { handle(req, m).await }
                    }),
                )
                .await
            {
                tracing::debug!(error = %e, "metrics conn error");
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    m: Metrics,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let body = match req.uri().path() {
        "/metrics" => {
            let mfs = m.registry.gather();
            let encoder = TextEncoder::new();
            let mut buf = Vec::with_capacity(2048);
            if let Err(e) = encoder.encode(&mfs, &mut buf) {
                return Ok(plain_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("encode error: {e}"),
                ));
            }
            return Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", encoder.format_type())
                .body(Full::from(buf))
                .unwrap());
        }
        "/health" => "ok\n",
        _ => {
            return Ok(plain_response(
                StatusCode::NOT_FOUND,
                "not found\n".to_string(),
            ));
        }
    };
    Ok(plain_response(StatusCode::OK, body.to_string()))
}

fn plain_response(status: StatusCode, body: String) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Full::from(body.into_bytes()))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_register_without_panic() {
        let m = Metrics::new();
        m.connections_total.inc();
        m.commands_total.with_label_values(&["PING"]).inc();
        m.errors_total.with_label_values(&["parse"]).inc();
        m.memory_count.set(42);
        // Encode and assert the names appear.
        let mfs = m.registry.gather();
        let mut buf = Vec::new();
        TextEncoder::new().encode(&mfs, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("duxx_resp_connections_total"));
        assert!(s.contains("duxx_resp_commands_total"));
        assert!(s.contains("duxx_resp_memory_count 42"));
    }
}
