//! OTLP/HTTP (JSON) trace bridge — flow traces **in** and **out** of DuxxDB.
//!
//! - **In** (receiver): `POST /v1/traces` accepts an OTLP/HTTP JSON
//!   `ExportTraceServiceRequest` (what any OpenTelemetry SDK or Collector
//!   emits) and records every span into the caller's workspace `TraceStore`.
//!   Authenticated by a workspace JWT, so spans land in the right tenant.
//! - **Out** (exporter): [`spans_to_otlp_json`] converts DuxxDB spans back into
//!   OTLP/HTTP JSON, and [`push_otlp`] POSTs them to an external collector
//!   (Langfuse, Datadog, Grafana, or an OTel Collector).
//!
//! JSON (not protobuf) so there's no `opentelemetry-proto` / `prost`
//! dependency. `Span` already mirrors the OTel shape, so the mapping is thin.

use crate::{Server, SERVER_VERSION};
use duxx_trace::{Span, SpanKind, SpanStatus, TraceStore};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Ingest (OTLP JSON → TraceStore)
// ---------------------------------------------------------------------------

/// Parse an OTLP/HTTP JSON `ExportTraceServiceRequest` and record every span
/// into `traces`. Returns the number of spans ingested.
pub fn ingest_otlp_json(traces: &TraceStore, body: &[u8]) -> Result<usize, String> {
    let req: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let mut n = 0usize;
    for rs in array(req.get("resourceSpans")) {
        for ss in array(rs.get("scopeSpans")) {
            for sp in array(ss.get("spans")) {
                let span = otlp_span_to_span(sp)?;
                traces.record_span(span).map_err(|e| e.to_string())?;
                n += 1;
            }
        }
    }
    Ok(n)
}

fn array(v: Option<&Value>) -> Vec<&Value> {
    v.and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default()
}

fn otlp_span_to_span(sp: &Value) -> Result<Span, String> {
    let trace_id = sp
        .get("traceId")
        .and_then(Value::as_str)
        .ok_or("span missing traceId")?
        .to_string();
    let span_id = sp
        .get("spanId")
        .and_then(Value::as_str)
        .ok_or("span missing spanId")?
        .to_string();
    let parent_span_id = sp
        .get("parentSpanId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let name = sp.get("name").and_then(Value::as_str).unwrap_or("").to_string();
    // OTLP SpanKind: 0 UNSPECIFIED, 1 INTERNAL, 2 SERVER, 3 CLIENT, 4 PRODUCER, 5 CONSUMER.
    let kind = match sp.get("kind").and_then(Value::as_i64).unwrap_or(0) {
        2 => SpanKind::Server,
        3 => SpanKind::Client,
        4 => SpanKind::Producer,
        5 => SpanKind::Consumer,
        _ => SpanKind::Internal,
    };
    // OTLP StatusCode: 0 UNSET, 1 OK, 2 ERROR.
    let status = match sp
        .get("status")
        .and_then(|s| s.get("code"))
        .and_then(Value::as_i64)
        .unwrap_or(0)
    {
        1 => SpanStatus::Ok,
        2 => SpanStatus::Error,
        _ => SpanStatus::Unset,
    };
    Ok(Span {
        trace_id,
        span_id,
        parent_span_id,
        thread_id: None,
        name,
        kind,
        start_unix_ns: parse_nano(sp.get("startTimeUnixNano")).unwrap_or(0),
        end_unix_ns: parse_nano(sp.get("endTimeUnixNano")),
        status,
        attributes: otlp_attrs_to_value(sp.get("attributes")),
    })
}

/// OTLP encodes uint64 nanos as a JSON **string** (proto3 JSON). Accept both
/// string and number; `None`/`"0"` → `None`.
fn parse_nano(v: Option<&Value>) -> Option<u128> {
    match v {
        Some(Value::String(s)) if !s.is_empty() && s != "0" => s.parse().ok(),
        Some(Value::Number(n)) => n.as_u64().filter(|x| *x != 0).map(u128::from),
        _ => None,
    }
}

fn otlp_attrs_to_value(attrs: Option<&Value>) -> Value {
    let mut map = serde_json::Map::new();
    for kv in array(attrs) {
        if let (Some(k), Some(v)) = (kv.get("key").and_then(Value::as_str), kv.get("value")) {
            map.insert(k.to_string(), otlp_anyvalue(v));
        }
    }
    Value::Object(map)
}

/// Flatten an OTLP `AnyValue` (`{stringValue|intValue|doubleValue|boolValue}`)
/// to a plain JSON scalar.
fn otlp_anyvalue(v: &Value) -> Value {
    if let Some(s) = v.get("stringValue") {
        return s.clone();
    }
    if let Some(i) = v.get("intValue") {
        // intValue is a string in proto3 JSON.
        if let Some(s) = i.as_str() {
            return s.parse::<i64>().map(Value::from).unwrap_or_else(|_| json!(s));
        }
        return i.clone();
    }
    if let Some(d) = v.get("doubleValue") {
        return d.clone();
    }
    if let Some(b) = v.get("boolValue") {
        return b.clone();
    }
    Value::Null
}

// ---------------------------------------------------------------------------
// Export (TraceStore → OTLP JSON)
// ---------------------------------------------------------------------------

/// Convert spans into an OTLP/HTTP JSON `ExportTraceServiceRequest`.
pub fn spans_to_otlp_json(spans: &[Span]) -> Value {
    let otlp_spans: Vec<Value> = spans.iter().map(span_to_otlp).collect();
    json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [{ "key": "service.name", "value": { "stringValue": "duxxdb" } }]
            },
            "scopeSpans": [{
                "scope": { "name": "duxxdb", "version": SERVER_VERSION },
                "spans": otlp_spans,
            }],
        }],
    })
}

fn span_to_otlp(s: &Span) -> Value {
    let kind = match s.kind {
        SpanKind::Internal => 1,
        SpanKind::Server => 2,
        SpanKind::Client => 3,
        SpanKind::Producer => 4,
        SpanKind::Consumer => 5,
    };
    let status_code = match s.status {
        SpanStatus::Unset => 0,
        SpanStatus::Ok => 1,
        SpanStatus::Error => 2,
    };
    let mut obj = json!({
        "traceId": s.trace_id,
        "spanId": s.span_id,
        "name": s.name,
        "kind": kind,
        "startTimeUnixNano": s.start_unix_ns.to_string(),
        "status": { "code": status_code },
        "attributes": value_to_otlp_attrs(&s.attributes),
    });
    if let Some(p) = &s.parent_span_id {
        obj["parentSpanId"] = json!(p);
    }
    if let Some(e) = s.end_unix_ns {
        obj["endTimeUnixNano"] = json!(e.to_string());
    }
    obj
}

fn value_to_otlp_attrs(v: &Value) -> Value {
    let mut out = Vec::new();
    if let Some(map) = v.as_object() {
        for (k, val) in map {
            out.push(json!({ "key": k, "value": value_to_anyvalue(val) }));
        }
    }
    Value::Array(out)
}

fn value_to_anyvalue(v: &Value) -> Value {
    match v {
        Value::String(s) => json!({ "stringValue": s }),
        Value::Bool(b) => json!({ "boolValue": b }),
        Value::Number(n) if n.is_i64() || n.is_u64() => json!({ "intValue": n.to_string() }),
        Value::Number(n) => json!({ "doubleValue": n.as_f64().unwrap_or(0.0) }),
        other => json!({ "stringValue": other.to_string() }),
    }
}

/// POST spans as OTLP/HTTP JSON to an external collector (Langfuse, Datadog,
/// Grafana, OTel Collector). Blocking — call from a background thread or
/// `spawn_blocking`. `headers` carries auth (e.g. `Authorization: Bearer …`).
pub fn push_otlp(
    endpoint: &str,
    headers: &[(String, String)],
    spans: &[Span],
) -> Result<(), String> {
    let payload = spans_to_otlp_json(spans);
    let client = reqwest::blocking::Client::new();
    let mut req = client.post(endpoint).json(&payload);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let resp = req.send().map_err(|e| format!("otlp push: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("otlp push: HTTP {}", resp.status()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP receiver
// ---------------------------------------------------------------------------

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Serve the OTLP/HTTP trace receiver on `addr` until the process exits.
/// `POST /v1/traces` (Bearer-JWT authenticated) records spans into the token's
/// workspace; `GET /healthz` is open.
pub async fn serve(server: Server, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "OTLP/HTTP trace receiver listening (POST /v1/traces)");
    loop {
        let (stream, _) = listener.accept().await?;
        let server = server.clone();
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            let svc = service_fn(move |req: Request<Incoming>| {
                let server = server.clone();
                async move { handle(req, server).await }
            });
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!(error = %e, "otlp conn error");
            }
        });
    }
}

async fn handle(req: Request<Incoming>, server: Server) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if method == hyper::Method::GET && path == "/healthz" {
        return Ok(reply(200, json!({ "status": "ok" })));
    }
    if !(method == hyper::Method::POST && path == "/v1/traces") {
        return Ok(reply(404, json!({ "error": "not found" })));
    }

    let bearer = req
        .headers()
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);

    let body = req
        .into_body()
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();

    match workspace_for(&server, bearer.as_deref()) {
        Ok(ws) => match ingest_otlp_json(ws.traces(), &body) {
            Ok(n) => Ok(reply(200, json!({ "partialSuccess": {}, "ingested": n }))),
            Err(e) => Ok(reply(400, json!({ "error": e }))),
        },
        Err((status, msg)) => Ok(reply(status, json!({ "error": msg }))),
    }
}

/// Resolve the workspace for a Bearer JWT (Ed25519 public key, then HS256).
fn workspace_for(
    server: &Server,
    bearer: Option<&str>,
) -> Result<std::sync::Arc<duxx_tenant::Workspace>, (u16, String)> {
    if server.jwt_secret.is_none() && server.jwt_public_key.is_none() {
        return Err((503, "OTLP ingest requires --jwt-secret or --jwt-public-key".into()));
    }
    let token = bearer.ok_or((401u16, "missing Authorization: Bearer <jwt>".to_string()))?;
    let claims = server
        .jwt_public_key
        .as_ref()
        .and_then(|pk| duxx_token::verify_ed25519(token, pk).ok())
        .or_else(|| {
            server
                .jwt_secret
                .as_ref()
                .and_then(|s| duxx_token::verify(token, s.as_slice()).ok())
        })
        .ok_or((401u16, "invalid or expired token".to_string()))?;
    let ns = duxx_tenant::Namespace::parse(&claims.tenant());
    Ok(server.tenants.workspace(&ns))
}

fn reply(status: u16, body: Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", "application/json")
        .body(Full::from(serde_json::to_vec(&body).unwrap_or_default()))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_ingest_then_export_round_trips() {
        let ts = TraceStore::new();
        let payload = json!({
            "resourceSpans": [{
                "scopeSpans": [{
                    "spans": [{
                        "traceId": "trace-abc",
                        "spanId": "span-1",
                        "name": "llm.openai.completion",
                        "kind": 3,
                        "startTimeUnixNano": "1700000000000000000",
                        "endTimeUnixNano": "1700000000500000000",
                        "status": { "code": 1 },
                        "attributes": [
                            { "key": "model", "value": { "stringValue": "gpt-4o" } },
                            { "key": "tokens", "value": { "intValue": "42" } }
                        ]
                    }]
                }]
            }]
        });

        let n = ingest_otlp_json(&ts, payload.to_string().as_bytes()).unwrap();
        assert_eq!(n, 1);

        let spans = ts.get_trace("trace-abc");
        assert_eq!(spans.len(), 1);
        let s = &spans[0];
        assert_eq!(s.name, "llm.openai.completion");
        assert_eq!(s.kind, SpanKind::Client);
        assert_eq!(s.status, SpanStatus::Ok);
        assert_eq!(s.start_unix_ns, 1_700_000_000_000_000_000u128);
        assert_eq!(s.attributes["model"], "gpt-4o");
        assert_eq!(s.attributes["tokens"], 42);

        // Export back to OTLP JSON and confirm the shape round-trips.
        let out = spans_to_otlp_json(&spans);
        let sp = &out["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(sp["traceId"], "trace-abc");
        assert_eq!(sp["kind"], 3);
        assert_eq!(sp["startTimeUnixNano"], "1700000000000000000");
        assert_eq!(sp["status"]["code"], 1);
        // attributes became OTLP key/value pairs again.
        let attrs = sp["attributes"].as_array().unwrap();
        assert!(attrs.iter().any(|a| a["key"] == "model"
            && a["value"]["stringValue"] == "gpt-4o"));
    }

    #[test]
    fn ingest_rejects_malformed_json() {
        let ts = TraceStore::new();
        assert!(ingest_otlp_json(&ts, b"{not json").is_err());
    }

    #[test]
    fn empty_request_ingests_zero() {
        let ts = TraceStore::new();
        assert_eq!(ingest_otlp_json(&ts, b"{}").unwrap(), 0);
    }
}
