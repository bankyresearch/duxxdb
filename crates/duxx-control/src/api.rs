//! HTTP API for the control plane (Phase B service).
//!
//! Routing is a **pure, synchronous** function ([`route`]) — request in,
//! response out — so the whole API surface is unit-testable without sockets or
//! an async client. A thin `hyper` wrapper ([`serve`]) reads the body, stamps
//! the time, calls `route`, and writes the response.
//!
//! All mutating endpoints are `POST` with a JSON body (IDs in the body, not the
//! path) so routing is exact-string matching — no path-param parser needed.
//!
//! | Method · Path | Body | Returns |
//! |---|---|---|
//! | `GET  /healthz` | — | `{"status":"ok"}` |
//! | `POST /v1/orgs` | `{name}` | `Org` |
//! | `POST /v1/projects` | `{org_id,name}` | `Project` |
//! | `POST /v1/keys` | `{project_id,env,role,name}` | `ApiKey` (incl. secret) |
//! | `POST /v1/keys/revoke` | `{key_id}` | `{revoked}` |
//! | `POST /v1/keys/rotate` | `{key_id}` | `ApiKey` |
//! | `POST /v1/tokens` | `{api_key_secret,ttl_secs}` | `{jwt}` |
//! | `POST /v1/placements` | `{project_id,node,mode}` | `Placement` |
//! | `POST /v1/auth-entries` | `{node}` | `{entries:[…]}` |
//! | `POST /v1/usage` | `{project_id,tokens_in,tokens_out,cost_usd}` | `{ok}` |
//! | `POST /v1/usage/query` | `{project_id}` | `Usage` |

use crate::{ControlError, ControlPlane, Env, PlacementMode, Role};
use serde::de::DeserializeOwned;
use serde::Deserialize;

/// The single-file Control Console UI, embedded at build time.
const CONSOLE_HTML: &str = include_str!("ui/console.html");

/// A status code + body + content type, ready to write to any transport.
pub struct ApiResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

fn json<T: serde::Serialize>(status: u16, value: &T) -> ApiResponse {
    let body = serde_json::to_vec(value)
        .unwrap_or_else(|e| format!("{{\"error\":\"encode: {e}\"}}").into_bytes());
    ApiResponse {
        status,
        content_type: "application/json",
        body,
    }
}

fn html(status: u16, body: &str) -> ApiResponse {
    ApiResponse {
        status,
        content_type: "text/html; charset=utf-8",
        body: body.as_bytes().to_vec(),
    }
}

fn error(status: u16, msg: impl Into<String>) -> ApiResponse {
    json(status, &serde_json::json!({ "error": msg.into() }))
}

fn status_for(err: &ControlError) -> u16 {
    match err {
        ControlError::OrgNotFound(_)
        | ControlError::ProjectNotFound(_)
        | ControlError::KeyNotFound(_) => 404,
        ControlError::EmptyName => 400,
        ControlError::BadSecret => 401,
        ControlError::NoSigningKey => 503,
        ControlError::Token(_) => 500,
    }
}

fn from_control(err: ControlError) -> ApiResponse {
    let status = status_for(&err);
    error(status, err.to_string())
}

fn parse_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, ApiResponse> {
    serde_json::from_slice(body).map_err(|e| error(400, format!("invalid JSON body: {e}")))
}

fn parse_env(s: &str) -> Result<Env, ApiResponse> {
    match s.trim().to_ascii_lowercase().as_str() {
        "dev" | "development" => Ok(Env::Dev),
        "staging" | "stage" => Ok(Env::Staging),
        "prod" | "production" => Ok(Env::Prod),
        other => Err(error(
            400,
            format!("unknown env {other:?} (dev|staging|prod)"),
        )),
    }
}

fn parse_role(s: &str) -> Result<Role, ApiResponse> {
    match s.trim().to_ascii_lowercase().as_str() {
        "owner" => Ok(Role::Owner),
        "admin" => Ok(Role::Admin),
        "developer" | "dev" => Ok(Role::Developer),
        "evaluator" => Ok(Role::Evaluator),
        "observer" => Ok(Role::Observer),
        "service" | "service-account" => Ok(Role::ServiceAccount),
        other => Err(error(
            400,
            format!("unknown role {other:?} (owner|admin|developer|evaluator|observer|service)"),
        )),
    }
}

fn parse_mode(s: &str) -> Result<PlacementMode, ApiResponse> {
    match s.trim().to_ascii_lowercase().as_str() {
        "shared" => Ok(PlacementMode::Shared),
        "dedicated" => Ok(PlacementMode::Dedicated),
        other => Err(error(
            400,
            format!("unknown mode {other:?} (shared|dedicated)"),
        )),
    }
}

// ---- request bodies -------------------------------------------------------

#[derive(Deserialize)]
struct NewOrg {
    name: String,
}
#[derive(Deserialize)]
struct OrgRef {
    org_id: String,
}
#[derive(Deserialize)]
struct NewProject {
    org_id: String,
    name: String,
}
#[derive(Deserialize)]
struct NewKey {
    project_id: String,
    env: String,
    role: String,
    name: String,
}
#[derive(Deserialize)]
struct KeyRef {
    key_id: String,
}
#[derive(Deserialize)]
struct MintToken {
    api_key_secret: String,
    ttl_secs: u64,
}
#[derive(Deserialize)]
struct NewPlacement {
    project_id: String,
    node: String,
    mode: String,
}
#[derive(Deserialize)]
struct NodeRef {
    node: String,
}
#[derive(Deserialize)]
struct UsageReport {
    project_id: String,
    tokens_in: u64,
    tokens_out: u64,
    cost_usd: f64,
}
#[derive(Deserialize)]
struct ProjectRef {
    project_id: String,
}
#[derive(Deserialize)]
struct InviteMember {
    org_id: String,
    email: String,
    role: String,
}
#[derive(Deserialize)]
struct AcceptInvite {
    invite_token: String,
}
#[derive(Deserialize)]
struct MemberRef {
    member_id: String,
}

/// Route one request. Pure and synchronous: `now_unix` is supplied by the
/// caller (the `hyper` wrapper stamps the real clock) so this stays testable.
pub fn route(
    cp: &ControlPlane,
    method: &str,
    path: &str,
    body: &[u8],
    now_unix: u64,
) -> ApiResponse {
    inner(cp, method, path, body, now_unix).unwrap_or_else(|resp| resp)
}

fn inner(
    cp: &ControlPlane,
    method: &str,
    path: &str,
    body: &[u8],
    now_unix: u64,
) -> Result<ApiResponse, ApiResponse> {
    Ok(match (method, path) {
        // Control Console UI.
        ("GET", "/") | ("GET", "/index.html") => html(200, CONSOLE_HTML),

        ("GET", "/healthz") => json(200, &serde_json::json!({ "status": "ok" })),

        ("GET", "/v1/orgs") => json(200, &serde_json::json!({ "orgs": cp.list_orgs() })),

        ("POST", "/v1/orgs") => {
            let req: NewOrg = parse_body(body)?;
            match cp.create_org(&req.name) {
                Ok(o) => json(201, &o),
                Err(e) => from_control(e),
            }
        }

        // List projects in an org. POST-with-body keeps routing param-free.
        ("POST", "/v1/projects/list") => {
            let req: OrgRef = parse_body(body)?;
            json(
                200,
                &serde_json::json!({ "projects": cp.projects_in_org(&req.org_id) }),
            )
        }

        // List a project's keys, with secrets redacted.
        ("POST", "/v1/keys/list") => {
            let req: ProjectRef = parse_body(body)?;
            let keys: Vec<_> = cp
                .keys_for_project(&req.project_id)
                .into_iter()
                .map(|k| {
                    serde_json::json!({
                        "id": k.id,
                        "project_id": k.project_id,
                        "env": k.env,
                        "role": k.role,
                        "name": k.name,
                        "revoked": k.revoked,
                    })
                })
                .collect();
            json(200, &serde_json::json!({ "keys": keys }))
        }

        ("POST", "/v1/projects") => {
            let req: NewProject = parse_body(body)?;
            match cp.create_project(&req.org_id, &req.name) {
                Ok(p) => json(201, &p),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/keys") => {
            let req: NewKey = parse_body(body)?;
            let env = parse_env(&req.env)?;
            let role = parse_role(&req.role)?;
            match cp.issue_key(&req.project_id, env, role, &req.name) {
                Ok(k) => json(201, &k),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/keys/revoke") => {
            let req: KeyRef = parse_body(body)?;
            match cp.revoke_key(&req.key_id) {
                Ok(revoked) => json(200, &serde_json::json!({ "revoked": revoked })),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/keys/rotate") => {
            let req: KeyRef = parse_body(body)?;
            match cp.rotate_key(&req.key_id) {
                Ok(k) => json(200, &k),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/tokens") => {
            let req: MintToken = parse_body(body)?;
            match cp.mint_jwt(&req.api_key_secret, now_unix, req.ttl_secs) {
                Ok(jwt) => json(200, &serde_json::json!({ "jwt": jwt })),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/members/invite") => {
            let req: InviteMember = parse_body(body)?;
            let role = parse_role(&req.role)?;
            match cp.invite_member(&req.org_id, &req.email, role) {
                Ok((member, invite_token)) => json(
                    201,
                    &serde_json::json!({ "member": member, "invite_token": invite_token }),
                ),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/members/accept") => {
            let req: AcceptInvite = parse_body(body)?;
            match cp.accept_invite(&req.invite_token) {
                Ok(member) => json(200, &member),
                Err(_) => error(404, "unknown or already-used invite token"),
            }
        }

        ("POST", "/v1/members/list") => {
            let req: OrgRef = parse_body(body)?;
            json(
                200,
                &serde_json::json!({ "members": cp.list_members(&req.org_id) }),
            )
        }

        ("POST", "/v1/members/remove") => {
            let req: MemberRef = parse_body(body)?;
            json(
                200,
                &serde_json::json!({ "removed": cp.remove_member(&req.member_id) }),
            )
        }

        ("POST", "/v1/placements") => {
            let req: NewPlacement = parse_body(body)?;
            let mode = parse_mode(&req.mode)?;
            match cp.place_project(&req.project_id, &req.node, mode) {
                Ok(pl) => json(201, &pl),
                Err(e) => from_control(e),
            }
        }

        ("POST", "/v1/auth-entries") => {
            let req: NodeRef = parse_body(body)?;
            let entries = cp.data_plane_auth_entries(&req.node);
            json(200, &serde_json::json!({ "entries": entries }))
        }

        ("POST", "/v1/usage") => {
            let req: UsageReport = parse_body(body)?;
            cp.record_usage(&req.project_id, req.tokens_in, req.tokens_out, req.cost_usd);
            json(200, &serde_json::json!({ "ok": true }))
        }

        ("POST", "/v1/usage/query") => {
            let req: ProjectRef = parse_body(body)?;
            json(200, &cp.usage(&req.project_id))
        }

        _ => error(404, "not found"),
    })
}

// ---------------------------------------------------------------------------
// hyper wrapper
// ---------------------------------------------------------------------------

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

/// Serve the control-plane HTTP API on `addr` until the process exits.
pub async fn serve(cp: Arc<ControlPlane>, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    loop {
        let (stream, _) = listener.accept().await?;
        let cp = cp.clone();
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            let svc = service_fn(move |req: Request<Incoming>| {
                let cp = cp.clone();
                async move { handle(req, cp).await }
            });
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                eprintln!("control-api conn error: {e}");
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    cp: Arc<ControlPlane>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let body = req
        .into_body()
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let resp = route(&cp, &method, &path, &body, now);
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
    use serde_json::Value;

    fn call(cp: &ControlPlane, method: &str, path: &str, body: Value) -> (u16, Value) {
        let bytes = serde_json::to_vec(&body).unwrap();
        let r = route(cp, method, path, &bytes, 1_700_000_000);
        let v: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
        (r.status, v)
    }

    #[test]
    fn health_ok() {
        let cp = ControlPlane::new();
        let (status, v) = call(&cp, "GET", "/healthz", Value::Null);
        assert_eq!(status, 200);
        assert_eq!(v["status"], "ok");
    }

    #[test]
    fn serves_console_ui_at_root() {
        let cp = ControlPlane::new();
        let r = route(&cp, "GET", "/", &[], 0);
        assert_eq!(r.status, 200);
        assert_eq!(r.content_type, "text/html; charset=utf-8");
        let html = String::from_utf8_lossy(&r.body);
        assert!(html.contains("DuxxDB Cloud"));
        // Full feature coverage is wired in the page.
        assert!(html.contains("/v1/members/invite"));
        assert!(html.contains("/v1/keys/rotate"));
    }

    #[test]
    fn full_provisioning_flow_over_http() {
        let cp = ControlPlane::with_signing_key(b"api-test-secret".to_vec());

        // Org → project → key.
        let (s, org) = call(&cp, "POST", "/v1/orgs", serde_json::json!({"name":"Acme"}));
        assert_eq!(s, 201);
        let org_id = org["id"].as_str().unwrap().to_string();

        let (s, proj) = call(
            &cp,
            "POST",
            "/v1/projects",
            serde_json::json!({"org_id": org_id, "name":"bot"}),
        );
        assert_eq!(s, 201);
        let project_id = proj["id"].as_str().unwrap().to_string();

        let (s, key) = call(
            &cp,
            "POST",
            "/v1/keys",
            serde_json::json!({"project_id": project_id, "env":"prod", "role":"service", "name":"agent"}),
        );
        assert_eq!(s, 201);
        let secret = key["secret"].as_str().unwrap().to_string();
        assert!(secret.starts_with("sk_"));

        // Place it and pull the node's auth entries.
        let (s, _) = call(
            &cp,
            "POST",
            "/v1/placements",
            serde_json::json!({"project_id": project_id, "node":"n1:6380", "mode":"shared"}),
        );
        assert_eq!(s, 201);
        let (s, entries) = call(
            &cp,
            "POST",
            "/v1/auth-entries",
            serde_json::json!({"node":"n1:6380"}),
        );
        assert_eq!(s, 200);
        assert_eq!(entries["entries"].as_array().unwrap().len(), 1);

        // List endpoints (what a UI renders).
        let (s, orgs) = call(&cp, "GET", "/v1/orgs", Value::Null);
        assert_eq!(s, 200);
        assert_eq!(orgs["orgs"].as_array().unwrap().len(), 1);

        let (s, projs) = call(
            &cp,
            "POST",
            "/v1/projects/list",
            serde_json::json!({"org_id": org_id}),
        );
        assert_eq!(s, 200);
        assert_eq!(projs["projects"].as_array().unwrap().len(), 1);

        let (s, keys) = call(
            &cp,
            "POST",
            "/v1/keys/list",
            serde_json::json!({"project_id": project_id}),
        );
        assert_eq!(s, 200);
        let listed = &keys["keys"].as_array().unwrap()[0];
        assert_eq!(listed["name"], "agent");
        assert!(listed.get("secret").is_none(), "list must redact secrets");

        // Mint a JWT for the key secret.
        let (s, tok) = call(
            &cp,
            "POST",
            "/v1/tokens",
            serde_json::json!({"api_key_secret": secret, "ttl_secs": 900}),
        );
        assert_eq!(s, 200);
        assert!(tok["jwt"].as_str().unwrap().contains('.'));

        // Record + query usage.
        let (s, _) = call(
            &cp,
            "POST",
            "/v1/usage",
            serde_json::json!({"project_id": project_id, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.01}),
        );
        assert_eq!(s, 200);
        let (s, usage) = call(
            &cp,
            "POST",
            "/v1/usage/query",
            serde_json::json!({"project_id": project_id}),
        );
        assert_eq!(s, 200);
        assert_eq!(usage["requests"], 1);
        assert_eq!(usage["tokens_in"], 10);
    }

    #[test]
    fn errors_map_to_status_codes() {
        let cp = ControlPlane::new();

        // Missing org → 404.
        let (s, v) = call(
            &cp,
            "POST",
            "/v1/projects",
            serde_json::json!({"org_id":"org_nope","name":"x"}),
        );
        assert_eq!(s, 404);
        assert!(v["error"].as_str().unwrap().contains("not found"));

        // Bad role → 400.
        let org = cp.create_org("A").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let (s, _) = call(
            &cp,
            "POST",
            "/v1/keys",
            serde_json::json!({"project_id": proj.id, "env":"prod", "role":"wizard", "name":"k"}),
        );
        assert_eq!(s, 400);

        // Mint without a signing key → 503.
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Developer, "k")
            .unwrap();
        let (s, _) = call(
            &cp,
            "POST",
            "/v1/tokens",
            serde_json::json!({"api_key_secret": key.secret, "ttl_secs": 60}),
        );
        assert_eq!(s, 503);

        // Unknown route → 404.
        let (s, _) = call(&cp, "GET", "/nope", Value::Null);
        assert_eq!(s, 404);

        // Malformed JSON body → 400.
        let r = route(&cp, "POST", "/v1/orgs", b"{not json", 0);
        assert_eq!(r.status, 400);
    }
}
