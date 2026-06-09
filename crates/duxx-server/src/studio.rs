//! Duxx Studio read-API (Phase C kickoff).
//!
//! Read-only HTTP/JSON views of a tenant's **own** workspace — the backend a
//! future Studio web UI renders. Every request is authenticated by a
//! control-plane-issued workspace JWT (`Authorization: Bearer <jwt>`, verified
//! with the same `--jwt-secret` the data plane already uses), and is scoped to
//! exactly that JWT's `(org, project, env)` workspace. There is no way to read
//! another tenant's data — the namespace comes from the verified claims, never
//! from the request.
//!
//! Like the control-plane API, routing is a **pure, synchronous** function
//! ([`route`]) so the whole surface is unit-testable without sockets, with a
//! thin `hyper` wrapper ([`serve`]).
//!
//! | Method · Path | Returns |
//! |---|---|
//! | `GET /studio/healthz` | `{"status":"ok"}` (open) |
//! | `GET /studio/overview` | `{tenant, memory_count, prompts:[…]}` |
//! | `GET /studio/memory?q=…&k=…` | `{hits:[{id,score,text}]}` (hybrid recall) |
//! | `GET /studio/cost` | `{tenant, total_usd, by_model:[…]}` |
//! | `GET /studio/evals` | `{runs:[EvalRun…]}` |
//! | `GET /studio/datasets` | `{datasets:[name…]}` |
//! | `GET /studio/replay` | `{sessions:[ReplaySession…]}` |
//! | `GET /studio/traces?trace_id=…` | `{trace_id, spans:[Span…]}` |
//! | `GET /studio/audit?limit=…` | `{events:[…]}` (this tenant's audit trail) |

use crate::Server;
use duxx_cost::{CostFilter, GroupBy};
use serde_json::json;

/// The single-file Studio UI, embedded at build time.
const STUDIO_HTML: &str = include_str!("ui/studio.html");

/// A status code + body + content type.
pub struct StudioResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

fn json_body<T: serde::Serialize>(status: u16, value: &T) -> StudioResponse {
    StudioResponse {
        status,
        content_type: "application/json",
        body: serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec()),
    }
}

fn html(status: u16, body: &str) -> StudioResponse {
    StudioResponse {
        status,
        content_type: "text/html; charset=utf-8",
        body: body.as_bytes().to_vec(),
    }
}

fn err(status: u16, msg: &str) -> StudioResponse {
    json_body(status, &json!({ "error": msg }))
}

/// Verify the bearer token against the server's JWT secret and return its
/// claims, or an error response (401/503).
fn authorize(server: &Server, bearer: Option<&str>) -> Result<duxx_token::Claims, StudioResponse> {
    if server.jwt_secret.is_none() && server.jwt_public_key.is_none() {
        return Err(err(
            503,
            "studio requires the server to be started with --jwt-secret or --jwt-public-key",
        ));
    }
    let token = match bearer {
        Some(t) if !t.is_empty() => t,
        _ => return Err(err(401, "missing Authorization: Bearer <jwt>")),
    };
    // Asymmetric (Ed25519) first, then symmetric (HS256).
    server
        .jwt_public_key
        .as_ref()
        .and_then(|pk| duxx_token::verify_ed25519(token, pk).ok())
        .or_else(|| {
            server
                .jwt_secret
                .as_ref()
                .and_then(|s| duxx_token::verify(token, s.as_slice()).ok())
        })
        .ok_or_else(|| err(401, "invalid or expired token"))
}

/// Tiny `&`-free query-string lookup: `first=1&q=foo&k=5` → value for `key`.
fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then_some(v)
    })
}

/// Route one Studio request. Pure + synchronous: the `hyper` wrapper supplies
/// the method, path, query string, and bearer token.
pub fn route(
    server: &Server,
    method: &str,
    path: &str,
    query: &str,
    bearer: Option<&str>,
) -> StudioResponse {
    // The Studio UI page and liveness are open; data routes are scoped.
    if method == "GET" && matches!(path, "/" | "/index.html" | "/studio" | "/studio/") {
        return html(200, STUDIO_HTML);
    }
    if method == "GET" && path == "/studio/healthz" {
        return json_body(200, &json!({ "status": "ok" }));
    }

    let claims = match authorize(server, bearer) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let ns = duxx_tenant::Namespace::parse(&claims.tenant());
    let ws = server.tenants.workspace(&ns);

    match (method, path) {
        ("GET", "/studio/overview") => json_body(
            200,
            &json!({
                "tenant": claims.tenant(),
                "role": claims.role,
                "memory_count": ws.memory().len(),
                "prompts": ws.prompts().names(),
            }),
        ),

        ("GET", "/studio/memory") => {
            let q = query_param(query, "q").unwrap_or("");
            if q.is_empty() {
                return err(400, "missing required query parameter 'q'");
            }
            let k = query_param(query, "k")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(10);
            let qvec = match server.embedder.embed(q) {
                Ok(v) => v,
                Err(e) => return err(500, &format!("embed: {e}")),
            };
            let hits = match ws.memory().recall("", q, &qvec, k) {
                Ok(h) => h,
                Err(e) => return err(500, &e.to_string()),
            };
            let hits: Vec<_> = hits
                .into_iter()
                .map(|h| {
                    json!({
                        "id": h.memory.id,
                        "score": h.score,
                        "text": h.memory.text,
                    })
                })
                .collect();
            json_body(200, &json!({ "hits": hits }))
        }

        ("GET", "/studio/cost") => {
            // Cost entries for this workspace are keyed by its tenant string
            // (`org/project/env`), which is exactly `claims.tenant()`.
            let tenant = claims.tenant();
            let total = ws.costs().total_for(&tenant, None, None);
            let by_model = ws.costs().aggregate(
                &CostFilter {
                    tenant: Some(tenant.clone()),
                    ..Default::default()
                },
                GroupBy::Model,
            );
            json_body(
                200,
                &json!({ "tenant": tenant, "total_usd": total, "by_model": by_model }),
            )
        }

        ("GET", "/studio/evals") => json_body(200, &json!({ "runs": ws.evals().list_runs() })),

        ("GET", "/studio/datasets") => {
            json_body(200, &json!({ "datasets": ws.datasets().names() }))
        }

        ("GET", "/studio/documents") => {
            let documents: Vec<_> = ws
                .docs()
                .list_documents()
                .into_iter()
                .map(|d| {
                    json!({
                        "id": d.id,
                        "uri": d.uri,
                        "content_type": d.content_type,
                        "version": d.version,
                        "chunks": d.chunk_ids.len(),
                    })
                })
                .collect();
            json_body(200, &json!({ "documents": documents }))
        }

        ("GET", "/studio/replay") => {
            json_body(200, &json!({ "sessions": ws.replays().list_sessions() }))
        }

        ("GET", "/studio/traces") => match query_param(query, "trace_id") {
            Some(tid) if !tid.is_empty() => json_body(
                200,
                &json!({ "trace_id": tid, "spans": ws.traces().get_trace(tid) }),
            ),
            _ => err(400, "missing required query parameter 'trace_id'"),
        },

        // Per-tenant audit trail (governance) — recent activity for THIS
        // workspace's tenant only.
        ("GET", "/studio/audit") => {
            let limit = query_param(query, "limit")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(200)
                .min(1000);
            let events = server.audit_trail.for_tenant(&claims.tenant(), limit);
            json_body(200, &json!({ "events": events }))
        }

        _ => err(404, "not found"),
    }
}

// ---------------------------------------------------------------------------
// hyper wrapper
// ---------------------------------------------------------------------------

use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Serve the Studio read-API on `addr` until the process exits. Spawn from the
/// caller (e.g. `tokio::spawn`).
pub async fn serve(server: Server, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "studio read-API listening (/studio/*)");
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
                tracing::debug!(error = %e, "studio conn error");
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    server: Server,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();
    let bearer = req
        .headers()
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);

    let resp = route(&server, &method, &path, &query, bearer.as_deref());
    Ok(Response::builder()
        .status(StatusCode::from_u16(resp.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", resp.content_type)
        .body(Full::from(resp.body))
        .unwrap())
}

// ---------------------------------------------------------------------------
// Tests — drive the pure router directly, no sockets.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resp::RespValue, security, Response as DispatchResponse};
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn mint(secret: &[u8], project: &str) -> String {
        let claims =
            duxx_token::Claims::new("key_s", "org1", project, "prod", "developer", now(), 3600);
        duxx_token::sign(&claims, secret).unwrap()
    }

    #[test]
    fn studio_reads_only_the_callers_workspace() {
        let secret = b"studio-jwt-secret".to_vec();
        let s = Server::new().with_jwt_secret(secret.clone());
        let jwt_a = mint(&secret, "projA");

        // Populate projA's workspace through the normal RESP path.
        let mut auth = security::AuthState::unauthenticated();
        s.dispatch_with_auth_state(
            RespValue::Array(vec![
                RespValue::bulk("AUTH"),
                RespValue::bulk(jwt_a.clone()),
            ]),
            false,
            &mut auth,
        );
        let put = s.dispatch_with_auth_state(
            RespValue::Array(vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("u1"),
                RespValue::bulk("studio falcon memory"),
            ]),
            false,
            &mut auth,
        );
        assert!(matches!(
            put,
            DispatchResponse::Reply(RespValue::Integer(_))
        ));

        // Overview reflects projA's workspace.
        let r = route(&s, "GET", "/studio/overview", "", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["tenant"], "org1/projA/prod");
        assert_eq!(v["memory_count"], 1);

        // Memory search finds it.
        let r = route(&s, "GET", "/studio/memory", "q=falcon&k=5", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert!(!v["hits"].as_array().unwrap().is_empty());

        // Cost + eval read views return 200 with the right shape.
        let r = route(&s, "GET", "/studio/cost", "", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["tenant"], "org1/projA/prod");
        assert!(v["total_usd"].is_number());
        assert!(v["by_model"].is_array());

        let r = route(&s, "GET", "/studio/evals", "", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert!(v["runs"].is_array());

        // Datasets + replay list views.
        for path in ["/studio/datasets", "/studio/replay"] {
            let r = route(&s, "GET", path, "", Some(&jwt_a));
            assert_eq!(r.status, 200, "{path}");
        }

        // Traces require a trace_id; absent → 400, present → 200 (spans array).
        assert_eq!(
            route(&s, "GET", "/studio/traces", "", Some(&jwt_a)).status,
            400
        );
        let r = route(&s, "GET", "/studio/traces", "trace_id=t1", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert!(v["spans"].is_array());

        // Audit trail: the REMEMBER we ran above is recorded for this tenant.
        let r = route(&s, "GET", "/studio/audit", "", Some(&jwt_a));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert!(
            v["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["command"] == "REMEMBER"),
            "audit trail should include the REMEMBER"
        );

        // A different project's JWT sees an empty workspace (isolation).
        let jwt_b = mint(&secret, "projB");
        let r = route(&s, "GET", "/studio/overview", "", Some(&jwt_b));
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["tenant"], "org1/projB/prod");
        assert_eq!(v["memory_count"], 0);
    }

    #[test]
    fn studio_auth_is_enforced() {
        let secret = b"studio-jwt-secret".to_vec();
        let s = Server::new().with_jwt_secret(secret.clone());

        // The UI page and health are open (no token).
        let page = route(&s, "GET", "/", "", None);
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html; charset=utf-8");
        assert!(String::from_utf8_lossy(&page.body).contains("Duxx Studio"));
        assert_eq!(route(&s, "GET", "/studio/healthz", "", None).status, 200);

        // No token / bad token → 401.
        assert_eq!(route(&s, "GET", "/studio/overview", "", None).status, 401);
        assert_eq!(
            route(&s, "GET", "/studio/overview", "", Some("not.a.jwt")).status,
            401
        );

        // Unknown route → 404 (with a valid token).
        let jwt = mint(&secret, "projA");
        assert_eq!(route(&s, "GET", "/studio/nope", "", Some(&jwt)).status, 404);

        // Missing q → 400.
        assert_eq!(
            route(&s, "GET", "/studio/memory", "", Some(&jwt)).status,
            400
        );
    }

    #[test]
    fn studio_without_jwt_secret_is_unavailable() {
        // A server with no JWT secret cannot serve Studio (no way to auth).
        let s = Server::new();
        assert_eq!(
            route(&s, "GET", "/studio/overview", "", Some("x")).status,
            503
        );
    }
}
