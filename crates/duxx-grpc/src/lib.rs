//! # duxx-grpc
//!
//! Tonic gRPC daemon for DuxxDB. Wraps `MemoryStore` with the typed
//! schema in `proto/duxx.proto`. Server streaming is used for
//! `Subscribe` so any gRPC client (Python `grpcio`, Go `grpc-go`,
//! Node `@grpc/grpc-js`, …) can consume memory change events directly.
//!
//! See `proto/duxx.proto` for the schema. Generated Rust types live in
//! `pb::duxx::v1::*` (this module re-exports them).

use duxx_embed::{Embedder, HashEmbedder};
use duxx_memory::MemoryStore;
use duxx_reactive::{ChangeEvent as InternalChangeEvent, ChangeKind};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

/// Generated protobuf types.
pub mod pb {
    pub mod duxx {
        pub mod v1 {
            tonic::include_proto!("duxx.v1");
        }
    }
}

pub use pb::duxx::v1 as proto;
use proto::duxx_server::{Duxx, DuxxServer};
use proto::{
    ChangeEvent, MemoryHit, PingRequest, PingResponse, RecallRequest, RecallResponse,
    RememberRequest, RememberResponse, StatsRequest, StatsResponse, SubscribeRequest,
};

pub const SERVER_NAME: &str = "duxxdb-grpc";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Convenience type for the streaming `Subscribe` reply.
type SubscribeStream =
    Pin<Box<dyn futures::Stream<Item = Result<ChangeEvent, Status>> + Send + 'static>>;

/// Tonic service implementation.
#[derive(Clone)]
pub struct DuxxService {
    memory: MemoryStore,
    embedder: Arc<dyn Embedder>,
    dim: usize,
    /// Optional bearer token. When `Some`, every request must carry
    /// `authorization: Bearer <token>` in metadata. When `None`, the
    /// service is unauthenticated.
    auth_token: Option<Arc<str>>,
}

impl DuxxService {
    /// Build a service with the default `HashEmbedder` (toy, 32-d).
    pub fn new() -> Self {
        Self::with_provider(Arc::new(HashEmbedder::new(32)))
    }

    /// Build with an explicit embedder; in-memory store.
    pub fn with_provider(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            memory: MemoryStore::with_capacity(dim, 100_000),
            embedder,
            dim,
            auth_token: None,
        }
    }

    /// Build with an embedder + a fully-persistent on-disk store at `dir`.
    pub fn open_at(
        embedder: Arc<dyn Embedder>,
        dir: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let dim = embedder.dim();
        let memory = MemoryStore::open_at(dim, 100_000, dir)?;
        Ok(Self {
            memory,
            embedder,
            dim,
            auth_token: None,
        })
    }

    /// Require clients to send `authorization: Bearer <token>` on
    /// every RPC. Pass `None` (or omit) to disable auth.
    pub fn with_auth(mut self, token: impl Into<Arc<str>>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Build the bearer-token check used by [`Self::serve`]. Public
    /// so tests / custom serve loops can reuse it.
    pub fn auth_interceptor(&self) -> impl tonic::service::Interceptor + Clone {
        let expected: Option<Arc<str>> = self.auth_token.clone();
        move |req: tonic::Request<()>| -> Result<tonic::Request<()>, tonic::Status> {
            let Some(want) = expected.as_ref() else {
                return Ok(req); // auth disabled
            };
            let got = req
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .unwrap_or("");
            if got.is_empty() {
                return Err(tonic::Status::unauthenticated(
                    "missing authorization header (Bearer token required)",
                ));
            }
            if !ct_eq(got.as_bytes(), want.as_bytes()) {
                return Err(tonic::Status::unauthenticated("invalid bearer token"));
            }
            Ok(req)
        }
    }

    /// Wrap as a `tonic` server ready for `serve()`.
    pub fn into_server(self) -> DuxxServer<Self> {
        DuxxServer::new(self)
    }

    /// Convenience: serve on `addr` until cancellation. Includes the
    /// standard `grpc.health.v1.Health` service for k8s / load
    /// balancers and the bearer-token interceptor when auth is set.
    pub async fn serve(self, addr: &str) -> anyhow::Result<()> {
        let parsed: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid addr {addr}: {e}"))?;
        tracing::info!(%addr, auth = self.auth_token.is_some(), "duxx-grpc listening");

        // Standard health protocol -- k8s livenessProbe / readinessProbe
        // can hit `grpc.health.v1.Health/Check` directly.
        let (mut health_reporter, health_svc) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<DuxxServer<DuxxService>>()
            .await;

        let interceptor = self.auth_interceptor();
        let duxx_with_auth = DuxxServer::with_interceptor(self, interceptor);

        tonic::transport::Server::builder()
            .add_service(health_svc)
            .add_service(duxx_with_auth)
            .serve(parsed)
            .await?;
        Ok(())
    }
}

/// Constant-time byte comparison (auth tokens, defense against timing
/// side-channels).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

impl Default for DuxxService {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl Duxx for DuxxService {
    async fn ping(
        &self,
        request: Request<PingRequest>,
    ) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse {
            nonce: request.into_inner().nonce,
        }))
    }

    async fn stats(
        &self,
        _: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse {
            memories: self.memory.len() as u64,
            sessions: 0, // SessionStore not exposed via gRPC yet
            dim: self.dim as u32,
            version: SERVER_VERSION.to_string(),
        }))
    }

    async fn remember(
        &self,
        request: Request<RememberRequest>,
    ) -> Result<Response<RememberResponse>, Status> {
        let req = request.into_inner();
        if req.key.is_empty() {
            return Err(Status::invalid_argument("key is required"));
        }
        if req.text.is_empty() {
            return Err(Status::invalid_argument("text is required"));
        }
        let embedding = if req.embedding.is_empty() {
            self.embedder
                .embed(&req.text)
                .map_err(|e| Status::internal(format!("embed: {e}")))?
        } else {
            req.embedding
        };
        if embedding.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "embedding has dim {}, server expects {}",
                embedding.len(),
                self.dim
            )));
        }
        let id = self
            .memory
            .remember(req.key, req.text, embedding)
            .map_err(|e| Status::internal(format!("remember: {e}")))?;
        Ok(Response::new(RememberResponse { id }))
    }

    async fn recall(
        &self,
        request: Request<RecallRequest>,
    ) -> Result<Response<RecallResponse>, Status> {
        let req = request.into_inner();
        if req.key.is_empty() {
            return Err(Status::invalid_argument("key is required"));
        }
        if req.query.is_empty() {
            return Err(Status::invalid_argument("query is required"));
        }
        let embedding = if req.embedding.is_empty() {
            self.embedder
                .embed(&req.query)
                .map_err(|e| Status::internal(format!("embed: {e}")))?
        } else {
            req.embedding
        };
        if embedding.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "embedding has dim {}, server expects {}",
                embedding.len(),
                self.dim
            )));
        }
        let k = if req.k == 0 { 10 } else { req.k as usize };
        let hits = self
            .memory
            .recall(&req.key, &req.query, &embedding, k)
            .map_err(|e| Status::internal(format!("recall: {e}")))?;
        let pb_hits: Vec<MemoryHit> = hits
            .into_iter()
            .map(|h| MemoryHit {
                id: h.memory.id,
                key: h.memory.key,
                text: h.memory.text,
                score: h.score,
            })
            .collect();
        Ok(Response::new(RecallResponse { hits: pb_hits }))
    }

    type SubscribeStream = SubscribeStream;

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<SubscribeStream>, Status> {
        let pattern = request.into_inner().pattern;
        let rx = self.memory.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(move |item| match item {
            Ok(event) => {
                let chan = event.channel();
                if pattern.is_empty() || matches_pattern(&pattern, &chan) {
                    Some(Ok(to_pb_event(event, chan)))
                } else {
                    None
                }
            }
            Err(_lagged) => None, // skip dropped messages on slow consumers
        });
        Ok(Response::new(Box::pin(stream)))
    }
}

/// Glob match copied from duxx-server::glob (kept private to avoid a
/// circular dep). Tiny enough that duplication is cheaper than
/// extracting a shared crate.
fn matches_pattern(pattern: &str, text: &str) -> bool {
    glob_match(pattern.as_bytes(), text.as_bytes())
}

fn glob_match(mut pat: &[u8], mut s: &[u8]) -> bool {
    let mut star_pat: Option<&[u8]> = None;
    let mut star_s: &[u8] = &[];
    loop {
        match (pat.first(), s.first()) {
            (Some(b'\\'), _) if pat.len() >= 2 => {
                let want = pat[1];
                if let Some(&got) = s.first() {
                    if got == want {
                        pat = &pat[2..];
                        s = &s[1..];
                        continue;
                    }
                }
            }
            (Some(b'?'), Some(_)) => {
                pat = &pat[1..];
                s = &s[1..];
                continue;
            }
            (Some(b'*'), _) => {
                while let Some(b'*') = pat.first() {
                    pat = &pat[1..];
                }
                if pat.is_empty() {
                    return true;
                }
                star_pat = Some(pat);
                star_s = s;
                continue;
            }
            (Some(&pc), Some(&sc)) if pc == sc => {
                pat = &pat[1..];
                s = &s[1..];
                continue;
            }
            (None, None) => return true,
            _ => {}
        }
        if let Some(sp) = star_pat {
            if !star_s.is_empty() {
                star_s = &star_s[1..];
                pat = sp;
                s = star_s;
                continue;
            }
        }
        return false;
    }
}

fn to_pb_event(e: InternalChangeEvent, channel: String) -> ChangeEvent {
    use proto::change_event::Kind as PbKind;
    let kind = match e.kind {
        ChangeKind::Insert => PbKind::Insert,
        ChangeKind::Update => PbKind::Update,
        ChangeKind::Delete => PbKind::Delete,
    };
    ChangeEvent {
        table: e.table,
        key: e.key.unwrap_or_default(),
        row_id: e.row_id,
        kind: kind as i32,
        channel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_redis_examples() {
        assert!(matches_pattern("memory.*", "memory.alice"));
        assert!(!matches_pattern("memory.*", "memory"));
        assert!(matches_pattern("memory.a*", "memory.alice"));
        assert!(!matches_pattern("memory.a*", "memory.bob"));
        assert!(matches_pattern("*", "anything"));
    }

    #[test]
    fn pb_event_round_trip() {
        let e = InternalChangeEvent {
            table: "memory".to_string(),
            key: Some("alice".to_string()),
            row_id: 42,
            kind: ChangeKind::Insert,
        };
        let chan = e.channel();
        let pb = to_pb_event(e, chan);
        assert_eq!(pb.table, "memory");
        assert_eq!(pb.key, "alice");
        assert_eq!(pb.row_id, 42);
        assert_eq!(pb.kind, proto::change_event::Kind::Insert as i32);
        assert_eq!(pb.channel, "memory.alice");
    }

    #[tokio::test]
    async fn ping_echoes_nonce() {
        let svc = DuxxService::new();
        let resp = svc
            .ping(Request::new(PingRequest {
                nonce: "abc".into(),
            }))
            .await
            .unwrap();
        assert_eq!(resp.into_inner().nonce, "abc");
    }

    #[tokio::test]
    async fn remember_then_recall_via_grpc() {
        let svc = DuxxService::new();
        let id = svc
            .remember(Request::new(RememberRequest {
                key: "alice".into(),
                text: "I lost my wallet at the cafe".into(),
                embedding: vec![],
            }))
            .await
            .unwrap()
            .into_inner()
            .id;
        assert_eq!(id, 1);

        let hits = svc
            .recall(Request::new(RecallRequest {
                key: "alice".into(),
                query: "wallet".into(),
                embedding: vec![],
                k: 5,
            }))
            .await
            .unwrap()
            .into_inner()
            .hits;
        assert!(!hits.is_empty());
        assert!(hits[0].text.to_lowercase().contains("wallet"));
    }
}
