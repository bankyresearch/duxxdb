//! Replication transport — ship the change feed between nodes over HTTP.
//!
//! A **leader** serves `POST /replication/pull` (a follower asks for everything
//! after a sequence). A **follower** uses [`ReplicationClient`] to pull and
//! [`Server::apply_replicated`] to converge. Auth is a shared cluster token
//! (`--replication-token`), distinct from client JWTs.
//!
//! Implemented over hyper HTTP for consistency with the server's other service
//! endpoints (metrics / studio / otlp) and so the wire format is unit-testable
//! without sockets. Swappable to gRPC via `duxx-grpc` if you prefer streaming.

use crate::Server;
use duxx_cluster::{Change, ChangeLog};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Wire format (pure, testable)
// ---------------------------------------------------------------------------

/// Serialize every change after `after_seq` as the pull response body:
/// `{ latestSeq, changes: [{ seq, namespace, op: [u8…] }] }`.
pub fn pull_changes(log: &dyn ChangeLog, after_seq: u64) -> Value {
    let changes: Vec<Value> = log
        .since(after_seq)
        .into_iter()
        .map(|c| json!({ "seq": c.seq, "namespace": c.namespace, "op": c.op }))
        .collect();
    json!({
        "latestSeq": log.latest_seq(),
        // A follower whose cursor < earliestSeq-1 has fallen off the compacted
        // log and must bootstrap from a snapshot instead of tailing.
        "earliestSeq": log.earliest_seq(),
        "changes": changes,
    })
}

/// Parse one change from the pull response.
pub fn parse_change(v: &Value) -> Result<Change, String> {
    let seq = v
        .get("seq")
        .and_then(Value::as_u64)
        .ok_or("change missing seq")?;
    let namespace = v
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let op = v
        .get("op")
        .and_then(Value::as_array)
        .ok_or("change missing op")?
        .iter()
        .map(|n| n.as_u64().unwrap_or(0) as u8)
        .collect();
    Ok(Change { seq, namespace, op })
}

// ---------------------------------------------------------------------------
// Follower client
// ---------------------------------------------------------------------------

/// Pulls the change feed from a remote leader.
pub struct ReplicationClient {
    endpoint: String,
    token: Option<String>,
    http: reqwest::blocking::Client,
}

impl ReplicationClient {
    /// `endpoint` is the leader's replication base URL, e.g.
    /// `http://leader-node:7075`.
    pub fn new(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            token,
            http: reqwest::blocking::Client::new(),
        }
    }

    /// Pull every change after `after_seq`. Blocking.
    pub fn pull(&self, after_seq: u64) -> Result<Vec<Change>, String> {
        let url = format!("{}/replication/pull", self.endpoint.trim_end_matches('/'));
        let mut req = self.http.post(url).json(&json!({ "afterSeq": after_seq }));
        if let Some(t) = &self.token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let resp = req.send().map_err(|e| format!("pull: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("pull: HTTP {}", resp.status()));
        }
        let body: Value = resp.json().map_err(|e| format!("pull decode: {e}"))?;
        body.get("changes")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(parse_change).collect::<Result<Vec<_>, _>>())
            .unwrap_or_else(|| Ok(Vec::new()))
    }
}

/// Pull from the leader and apply to `server`, advancing `applied_seq`. Returns
/// the number of changes applied. Call this on a timer / in a background loop.
pub fn sync_from_leader(
    client: &ReplicationClient,
    server: &Server,
    applied_seq: &mut u64,
) -> Result<usize, String> {
    let changes = client.pull(*applied_seq)?;
    let mut n = 0;
    for c in &changes {
        server.apply_replicated(c);
        *applied_seq = c.seq;
        n += 1;
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// Leader HTTP server
// ---------------------------------------------------------------------------

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Serve the replication feed on `addr` until the process exits. Requires the
/// server to have a change log (`with_replication`) and a token
/// (`with_replication_token`).
pub async fn serve(server: Server, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "replication transport listening (POST /replication/pull)");
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
                tracing::debug!(error = %e, "replication conn error");
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    server: Server,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if method == hyper::Method::GET && path == "/healthz" {
        return Ok(reply(200, json!({ "status": "ok" })));
    }
    if !(method == hyper::Method::POST && path == "/replication/pull") {
        return Ok(reply(404, json!({ "error": "not found" })));
    }

    // Cluster-token auth.
    let bearer = req
        .headers()
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match server.replication_token() {
        None => {
            return Ok(reply(
                503,
                json!({ "error": "replication transport not configured" }),
            ));
        }
        Some(expected) if bearer == Some(expected) => {}
        Some(_) => return Ok(reply(401, json!({ "error": "invalid replication token" }))),
    }

    let log = match server.replication_log() {
        Some(l) => l.clone(),
        None => {
            return Ok(reply(
                503,
                json!({ "error": "replication not enabled on this node" }),
            ))
        }
    };

    let body = req
        .into_body()
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();
    let after_seq = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("afterSeq").and_then(Value::as_u64))
        .unwrap_or(0);

    Ok(reply(200, pull_changes(&*log, after_seq)))
}

fn reply(status: u16, body: Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", "application/json")
        .body(Full::from(serde_json::to_vec(&body).unwrap_or_default()))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Tests — exercise the wire format + apply path without sockets.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resp::RespValue;
    use crate::Response as DispatchResponse;
    use duxx_cluster::{MemoryLog, StaticCoordinator};
    use std::sync::Arc;

    fn dispatch(server: &Server, items: Vec<RespValue>) -> RespValue {
        match server.dispatch(RespValue::Array(items)) {
            DispatchResponse::Reply(v) | DispatchResponse::CloseAfter(v) => v,
            _ => RespValue::Null,
        }
    }

    #[test]
    fn pull_serializes_feed_and_follower_applies_over_the_wire_format() {
        // Leader logs a mutation.
        let log = Arc::new(MemoryLog::new());
        let leader =
            Server::new().with_replication(log.clone(), Arc::new(StaticCoordinator::solo("l")));
        dispatch(
            &leader,
            vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("u1"),
                RespValue::bulk("wire me owl"),
            ],
        );

        // Serialize exactly as the HTTP endpoint would.
        let payload = pull_changes(&*log, 0);
        assert_eq!(payload["latestSeq"], 1);
        let changes = payload["changes"].as_array().unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0]["seq"], 1);

        // Follower parses the JSON changes and applies them.
        let follower = Server::new();
        for cv in changes {
            let c = parse_change(cv).unwrap();
            follower.apply_replicated(&c);
        }

        // Follower converged.
        let r = dispatch(
            &follower,
            vec![
                RespValue::bulk("RECALL"),
                RespValue::bulk("u1"),
                RespValue::bulk("owl"),
                RespValue::bulk("5"),
            ],
        );
        match r {
            RespValue::Array(items) => assert!(!items.is_empty(), "follower must converge"),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn parse_change_round_trips() {
        let c = Change {
            seq: 7,
            namespace: "org/p/prod".into(),
            op: vec![1, 2, 3, 255],
        };
        let v = json!({ "seq": c.seq, "namespace": c.namespace, "op": c.op });
        assert_eq!(parse_change(&v).unwrap(), c);
    }
}
