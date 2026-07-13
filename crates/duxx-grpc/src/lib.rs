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
    ChangeEvent, CompactRequest, CompactResponse, EraseRequest, EraseResponse, MemoryHit,
    PingRequest, PingResponse, RecallRequest, RecallResponse, RememberBatchRequest,
    RememberBatchResponse, RememberRequest, RememberResponse, ScanRequest, ScanResponse,
    StatsRequest, StatsResponse, SubscribeRequest,
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
    /// Optional TLS identity (Phase 6.2). When `Some`, the listener
    /// uses tonic's tls support (rustls under the hood) so clients
    /// connect over h2 + TLS.
    tls_identity: Option<tonic::transport::Identity>,
    /// Optional client CA root for mTLS client certificate verification.
    tls_client_ca: Option<tonic::transport::Certificate>,
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
            tls_identity: None,
            tls_client_ca: None,
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
            tls_identity: None,
            tls_client_ca: None,
        })
    }

    /// Require clients to send `authorization: Bearer <token>` on
    /// every RPC. Pass `None` (or omit) to disable auth.
    /// Enable native TLS termination (Phase 6.2). Pass paths to a
    /// PEM cert chain and PEM private key. The listener uses tonic's
    /// rustls-backed tls support; clients must connect over `https://`
    /// (or with the equivalent `with_root_certificates` / `--tls`
    /// flag for grpcurl etc.).
    pub fn with_tls_files(
        mut self,
        cert_path: impl AsRef<std::path::Path>,
        key_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let cert = std::fs::read(cert_path.as_ref())
            .map_err(|e| anyhow::anyhow!("read TLS cert {}: {e}", cert_path.as_ref().display()))?;
        let key = std::fs::read(key_path.as_ref())
            .map_err(|e| anyhow::anyhow!("read TLS key {}: {e}", key_path.as_ref().display()))?;
        self.tls_identity = Some(tonic::transport::Identity::from_pem(cert, key));
        Ok(self)
    }

    /// Enable native TLS and require clients to present certificates
    /// chaining to `client_ca_path`.
    pub fn with_mtls_files(
        mut self,
        cert_path: impl AsRef<std::path::Path>,
        key_path: impl AsRef<std::path::Path>,
        client_ca_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let cert = std::fs::read(cert_path.as_ref())
            .map_err(|e| anyhow::anyhow!("read TLS cert {}: {e}", cert_path.as_ref().display()))?;
        let key = std::fs::read(key_path.as_ref())
            .map_err(|e| anyhow::anyhow!("read TLS key {}: {e}", key_path.as_ref().display()))?;
        let client_ca = std::fs::read(client_ca_path.as_ref()).map_err(|e| {
            anyhow::anyhow!(
                "read TLS client CA {}: {e}",
                client_ca_path.as_ref().display()
            )
        })?;
        self.tls_identity = Some(tonic::transport::Identity::from_pem(cert, key));
        self.tls_client_ca = Some(tonic::transport::Certificate::from_pem(client_ca));
        Ok(self)
    }

    pub fn with_auth(mut self, token: impl Into<Arc<str>>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Build the bearer-token check used by [`Self::serve`]. Public
    /// so tests / custom serve loops can reuse it.
    #[allow(clippy::result_large_err)]
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
        tracing::info!(
            %addr,
            auth = self.auth_token.is_some(),
            tls = self.tls_identity.is_some(),
            mtls = self.tls_client_ca.is_some(),
            "duxx-grpc listening"
        );

        // Standard health protocol -- k8s livenessProbe / readinessProbe
        // can hit `grpc.health.v1.Health/Check` directly.
        let (mut health_reporter, health_svc) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<DuxxServer<DuxxService>>()
            .await;

        let tls_identity = self.tls_identity.clone();
        let tls_client_ca = self.tls_client_ca.clone();
        let interceptor = self.auth_interceptor();
        let duxx_with_auth = DuxxServer::with_interceptor(self, interceptor);

        let mut builder = tonic::transport::Server::builder();
        if let Some(identity) = tls_identity {
            let mut tls_cfg = tonic::transport::ServerTlsConfig::new().identity(identity);
            if let Some(client_ca) = tls_client_ca {
                tls_cfg = tls_cfg.client_ca_root(client_ca);
            }
            builder = builder
                .tls_config(tls_cfg)
                .map_err(|e| anyhow::anyhow!("gRPC TLS config: {e}"))?;
        }

        builder
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
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse {
            nonce: request.into_inner().nonce,
        }))
    }

    async fn stats(&self, _: Request<StatsRequest>) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse {
            memories: self.memory.len() as u64,
            sessions: 0, // SessionStore not exposed via gRPC yet
            dim: self.dim as u32,
            version: SERVER_VERSION.to_string(),
            tombstone_ratio: self.memory.tombstone_ratio(),
            compactions: self.memory.compactions_total(),
            protocol_version: duxx_core::PROTOCOL_VERSION,
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
        let id = if req.idempotency_key.is_empty() {
            self.memory.remember(req.key, req.text, embedding)
        } else {
            self.memory
                .remember_idempotent(req.key, req.text, embedding, req.idempotency_key)
        }
        .map_err(|e| Status::internal(format!("remember: {e}")))?;
        Ok(Response::new(RememberResponse { id }))
    }

    async fn remember_batch(
        &self,
        request: Request<RememberBatchRequest>,
    ) -> Result<Response<RememberBatchResponse>, Status> {
        let req = request.into_inner();
        if req.items.is_empty() {
            return Ok(Response::new(RememberBatchResponse { ids: vec![] }));
        }
        let mut items = Vec::with_capacity(req.items.len());
        for it in req.items {
            if it.key.is_empty() {
                return Err(Status::invalid_argument("key is required"));
            }
            if it.text.is_empty() {
                return Err(Status::invalid_argument("text is required"));
            }
            let emb = if it.embedding.is_empty() {
                self.embedder
                    .embed(&it.text)
                    .map_err(|e| Status::internal(format!("embed: {e}")))?
            } else {
                it.embedding
            };
            if emb.len() != self.dim {
                return Err(Status::invalid_argument(format!(
                    "embedding has dim {}, server expects {}",
                    emb.len(),
                    self.dim
                )));
            }
            items.push((it.key, it.text, emb));
        }
        let ids = self
            .memory
            .remember_batch(items)
            .map_err(|e| Status::internal(format!("remember_batch: {e}")))?;
        Ok(Response::new(RememberBatchResponse { ids }))
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

    async fn scan(&self, request: Request<ScanRequest>) -> Result<Response<ScanResponse>, Status> {
        let req = request.into_inner();
        let limit = if req.limit == 0 {
            10
        } else {
            req.limit as usize
        };
        let (page, next) = self.memory.scan(req.cursor, limit);
        let memories: Vec<MemoryHit> = page
            .into_iter()
            .map(|m| MemoryHit {
                id: m.id,
                key: m.key,
                text: m.text,
                score: 0.0,
            })
            .collect();
        Ok(Response::new(ScanResponse {
            memories,
            next_cursor: next.unwrap_or(0),
            has_more: next.is_some(),
        }))
    }

    async fn compact(
        &self,
        _request: Request<CompactRequest>,
    ) -> Result<Response<CompactResponse>, Status> {
        let reclaimed = self
            .memory
            .compact()
            .map_err(|e| Status::internal(format!("compact: {e}")))?;
        Ok(Response::new(CompactResponse {
            reclaimed: reclaimed as u64,
            tombstone_ratio: self.memory.tombstone_ratio(),
        }))
    }

    async fn erase(
        &self,
        request: Request<EraseRequest>,
    ) -> Result<Response<EraseResponse>, Status> {
        let req = request.into_inner();
        let count = if !req.key.is_empty() {
            self.memory.forget_by_key(&req.key)
        } else if req.older_than_secs > 0.0 {
            self.memory
                .forget_older_than(std::time::Duration::from_secs_f64(req.older_than_secs))
        } else {
            return Err(Status::invalid_argument(
                "set either key or older_than_secs",
            ));
        };
        Ok(Response::new(EraseResponse {
            count: count as u64,
        }))
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
                idempotency_key: String::new(),
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

    #[tokio::test]
    async fn compact_via_grpc_reclaims_tombstones() {
        let svc = DuxxService::new();
        svc.memory.set_auto_compact_ratio(None);
        svc.memory.set_max_rows(Some(2));
        svc.memory
            .set_eviction_half_life(std::time::Duration::from_micros(1));
        for i in 0..5 {
            svc.remember(Request::new(RememberRequest {
                key: "u".into(),
                text: format!("note {i}"),
                embedding: vec![],
                idempotency_key: String::new(),
            }))
            .await
            .unwrap();
        }
        // 5 inserts at cap 2 → 3 evictions → 3 tombstones reclaimed.
        let resp = svc
            .compact(Request::new(CompactRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.reclaimed, 3);
        assert_eq!(resp.tombstone_ratio, 0.0);
    }

    #[tokio::test]
    async fn erase_by_key_via_grpc() {
        let svc = DuxxService::new();
        for i in 0..5 {
            svc.remember(Request::new(RememberRequest {
                key: "alice".into(),
                text: format!("note {i}"),
                embedding: vec![],
                idempotency_key: String::new(),
            }))
            .await
            .unwrap();
        }
        let resp = svc
            .erase(Request::new(EraseRequest {
                key: "alice".into(),
                older_than_secs: 0.0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.count, 5);
        assert_eq!(svc.memory.len(), 0);
    }

    #[tokio::test]
    async fn remember_batch_via_grpc() {
        use proto::RememberItem;
        let svc = DuxxService::new();
        let items: Vec<RememberItem> = (0..20)
            .map(|i| RememberItem {
                key: "u".into(),
                text: format!("note {i}"),
                embedding: vec![],
            })
            .collect();
        let ids = svc
            .remember_batch(Request::new(RememberBatchRequest { items }))
            .await
            .unwrap()
            .into_inner()
            .ids;
        assert_eq!(ids.len(), 20);
        assert_eq!(svc.memory.len(), 20);
    }

    #[tokio::test]
    async fn scan_pages_all_via_grpc() {
        let svc = DuxxService::new();
        for i in 0..25 {
            svc.remember(Request::new(RememberRequest {
                key: "u".into(),
                text: format!("m{i}"),
                embedding: vec![],
                idempotency_key: String::new(),
            }))
            .await
            .unwrap();
        }
        let mut count = 0usize;
        let mut cursor = 0u64;
        loop {
            let resp = svc
                .scan(Request::new(ScanRequest { cursor, limit: 10 }))
                .await
                .unwrap()
                .into_inner();
            count += resp.memories.len();
            if !resp.has_more {
                break;
            }
            cursor = resp.next_cursor;
        }
        assert_eq!(count, 25);
    }

    #[tokio::test]
    async fn remember_idempotency_key_dedupes() {
        let svc = DuxxService::new();
        let mk = || {
            svc.remember(Request::new(RememberRequest {
                key: "u".into(),
                text: "I lost my wallet".into(),
                embedding: vec![],
                idempotency_key: "req-1".into(),
            }))
        };
        let id1 = mk().await.unwrap().into_inner().id;
        let id2 = mk().await.unwrap().into_inner().id;
        assert_eq!(id1, id2);
        assert_eq!(svc.memory.len(), 1);
    }

    #[tokio::test]
    async fn stats_reports_compaction_health() {
        let svc = DuxxService::new();
        svc.memory.set_auto_compact_ratio(None);
        svc.memory.set_max_rows(Some(2));
        svc.memory
            .set_eviction_half_life(std::time::Duration::from_micros(1));
        for i in 0..5 {
            svc.remember(Request::new(RememberRequest {
                key: "u".into(),
                text: format!("note {i}"),
                embedding: vec![],
                idempotency_key: String::new(),
            }))
            .await
            .unwrap();
        }
        // Before compaction: 2 live rows, 3 tombstones -> ratio 1.5, 0 compactions.
        let s = svc
            .stats(Request::new(StatsRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(s.memories, 2);
        assert!(s.tombstone_ratio > 0.0, "ratio={}", s.tombstone_ratio);
        assert_eq!(s.compactions, 0);
        assert_eq!(s.protocol_version, duxx_core::PROTOCOL_VERSION);

        svc.compact(Request::new(CompactRequest {})).await.unwrap();
        let s2 = svc
            .stats(Request::new(StatsRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(s2.tombstone_ratio, 0.0);
        assert_eq!(s2.compactions, 1);
    }
}
