//! # duxx-server
//!
//! TCP server speaking RESP2/3, the wire protocol of Valkey and Redis.
//!
//! Any client that knows Valkey/Redis (`valkey-cli`, `redis-cli`, the
//! many redis-rs / node-redis / go-redis libraries, …) connects to
//! DuxxDB without any code changes — it just sees a server with a few
//! extra commands.
//!
//! Implemented commands:
//!
//! - `PING [msg]`            — health check
//! - `HELLO [version]`       — RESP version handshake
//! - `COMMAND`               — return command list (minimal)
//! - `INFO`                  — server stats
//! - `QUIT`                  — close connection
//! - `SET key value`         — session-store put
//! - `GET key`               — session-store get
//! - `DEL key`               — session-store delete
//! - `REMEMBER key text...`  — store a memory
//! - `RECALL key query [k]`  — top-k hybrid recall
//!
//! `REMEMBER` and `RECALL` use a deterministic toy embedder so the
//! server is self-contained for demos. Production swaps in a real
//! provider via [`Server::with_embedder`].

pub mod glob;
pub mod metrics;
pub mod resp;

use duxx_embed::{Embedder, HashEmbedder};
use duxx_memory::{MemoryStore, SessionStore};
use duxx_reactive::{ChangeEvent, ChangeKind};
use resp::RespValue;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;

pub const SERVER_NAME: &str = "duxxdb";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_DIM: usize = 32;

/// The server state — cheaply cloned (Arc internals).
#[derive(Clone)]
pub struct Server {
    memory: MemoryStore,
    sessions: SessionStore,
    embedder: Arc<dyn Embedder>,
    dim: usize,
    /// When `Some`, every connection must `AUTH <token>` before any
    /// other command (PING is always allowed). When `None`, the
    /// server runs unauthenticated — fine for localhost dev, NOT
    /// safe for any network-exposed deployment.
    auth_token: Option<Arc<str>>,
    /// Active connection count for graceful-shutdown drain. Dropped
    /// to 0 by `serve_with_shutdown` on each conn close.
    active_conns: Arc<std::sync::atomic::AtomicUsize>,
    /// Optional Prometheus metrics. When Some, accept-loop + handler
    /// hooks update counters. Use [`Server::with_metrics`] to attach.
    metrics: Option<metrics::Metrics>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("dim", &self.dim)
            .field("memories", &self.memory.len())
            .field("sessions", &self.sessions.len())
            .finish_non_exhaustive()
    }
}

impl Server {
    /// Build with the default `HashEmbedder` (toy, 32-d), no
    /// durable storage, and no auth.
    pub fn new() -> Self {
        Self::with_provider(Arc::new(HashEmbedder::new(DEFAULT_DIM)))
    }

    /// Build with an explicit embedder; in-memory only.
    pub fn with_provider(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            memory: MemoryStore::with_capacity(dim, 100_000),
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
        }
    }

    /// Require clients to issue `AUTH <token>` before any non-PING
    /// command. Pass `None` to disable auth (the default).
    pub fn with_auth(mut self, token: impl Into<Arc<str>>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Attach Prometheus metrics. The accept loop will increment
    /// connection counters; the dispatch path increments command
    /// counters and records latency histograms.
    pub fn with_metrics(mut self, m: metrics::Metrics) -> Self {
        self.metrics = Some(m);
        self
    }

    /// Number of currently-open client connections.
    pub fn active_connections(&self) -> usize {
        self.active_conns
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Build with an embedder AND a durable storage backend (redb,
    /// or any other `Storage` impl). Memories survive process restart;
    /// indices are kept in memory and rebuilt from the row store on open.
    pub fn with_provider_and_storage(
        embedder: Arc<dyn Embedder>,
        storage: Arc<dyn duxx_storage::Storage>,
    ) -> anyhow::Result<Self> {
        let dim = embedder.dim();
        let memory = MemoryStore::with_storage(dim, 100_000, storage)?;
        Ok(Self {
            memory,
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
        })
    }

    /// Build with an embedder AND a fully-persistent on-disk store at `dir`.
    /// Both the row store (redb) AND the indices (tantivy + HNSW dump)
    /// are persisted, so reopens skip the rebuild.
    pub fn open_at(
        embedder: Arc<dyn Embedder>,
        dir: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let dim = embedder.dim();
        let memory = MemoryStore::open_at(dim, 100_000, dir)?;
        Ok(Self {
            memory,
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
        })
    }

    /// Convenience: wrap a closure as an embedder. Useful for tests.
    pub fn with_embedder<F>(self, dim: usize, embedder: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        struct ClosureEmbedder<F> {
            dim: usize,
            f: F,
        }
        impl<F: Fn(&str) -> Vec<f32> + Send + Sync> Embedder for ClosureEmbedder<F> {
            fn embed(&self, text: &str) -> duxx_core::Result<Vec<f32>> {
                Ok((self.f)(text))
            }
            fn dim(&self) -> usize {
                self.dim
            }
        }
        let _ = self;
        Self::with_provider(Arc::new(ClosureEmbedder { dim, f: embedder }))
    }

    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    pub fn sessions(&self) -> &SessionStore {
        &self.sessions
    }

    /// Bind and serve. Returns when the listener errors (typically on
    /// process shutdown).
    pub async fn serve(&self, addr: &str) -> anyhow::Result<()> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!(
            addr = %addr,
            auth = self.auth_token.is_some(),
            "duxx-server listening"
        );
        loop {
            let (socket, peer) = listener.accept().await?;
            tracing::debug!(?peer, "accepted");
            let server = self.clone();
            self.active_conns
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let counter = self.active_conns.clone();
            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(socket).await {
                    tracing::warn!(?peer, error = %e, "connection ended with error");
                }
                counter.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            });
        }
    }

    /// Bind and serve until `shutdown` resolves. After the signal:
    /// stop accepting new connections, then wait up to `drain` for
    /// active connections to close. Returns the count still open at
    /// the deadline (0 = clean drain).
    pub async fn serve_with_shutdown(
        &self,
        addr: &str,
        shutdown: impl std::future::Future<Output = ()>,
        drain: std::time::Duration,
    ) -> anyhow::Result<usize> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!(
            addr = %addr,
            auth = self.auth_token.is_some(),
            "duxx-server listening"
        );
        let server = self.clone();
        let counter = self.active_conns.clone();
        tokio::pin!(shutdown);

        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (socket, peer) = accept?;
                    tracing::debug!(?peer, "accepted");
                    let s = server.clone();
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Some(m) = &s.metrics {
                        m.connections_total.inc();
                        m.active_connections.inc();
                        m.memory_count.set(s.memory.len() as i64);
                        m.session_count.set(s.sessions.len() as i64);
                    }
                    let c = counter.clone();
                    let m = s.metrics.clone();
                    tokio::spawn(async move {
                        if let Err(e) = s.handle_connection(socket).await {
                            tracing::warn!(?peer, error = %e, "connection ended with error");
                            if let Some(ref m) = m {
                                m.errors_total.with_label_values(&["conn"]).inc();
                            }
                        }
                        c.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(m) = m {
                            m.active_connections.dec();
                        }
                    });
                }
                _ = &mut shutdown => {
                    tracing::info!("shutdown signal received; draining");
                    break;
                }
            }
        }

        // Drain.
        let deadline = tokio::time::Instant::now() + drain;
        loop {
            let n = counter.load(std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                tracing::info!("drain complete: all connections closed");
                return Ok(0);
            }
            if tokio::time::Instant::now() >= deadline {
                tracing::warn!(
                    open = n,
                    "drain deadline hit; {n} connections still open"
                );
                return Ok(n);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    async fn handle_connection(&self, mut socket: TcpStream) -> anyhow::Result<()> {
        let mut buf = bytes::BytesMut::with_capacity(4096);
        let mut out = Vec::with_capacity(1024);
        let mut state = SubState::default();
        // Pre-authed when no token is configured. Otherwise stays
        // false until the client issues AUTH <token>.
        let mut authed = self.auth_token.is_none();

        loop {
            // Drain any complete commands already in `buf`.
            loop {
                match resp::parse(&mut buf) {
                    Ok(Some(v)) => {
                        out.clear();
                        let action =
                            self.dispatch_with_auth(v, state.is_subscribed(), &mut authed);
                        match action {
                            Response::Reply(value) => {
                                value.write_to(&mut out);
                                socket.write_all(&out).await?;
                            }
                            Response::CloseAfter(value) => {
                                value.write_to(&mut out);
                                socket.write_all(&out).await?;
                                socket.shutdown().await.ok();
                                return Ok(());
                            }
                            Response::Subscribe { channel, ack } => {
                                ack.write_to(&mut out);
                                socket.write_all(&out).await?;
                                if state.rx.is_none() {
                                    state.rx = Some(self.memory.subscribe());
                                }
                                if !state.exact.contains(&channel) {
                                    state.exact.push(channel);
                                }
                            }
                            Response::Unsubscribe { channels, ack } => {
                                ack.write_to(&mut out);
                                socket.write_all(&out).await?;
                                if channels.is_empty() {
                                    state.exact.clear();
                                } else {
                                    state.exact.retain(|c| !channels.contains(c));
                                }
                                state.maybe_drop_rx();
                            }
                            Response::PSubscribe { pattern, ack } => {
                                ack.write_to(&mut out);
                                socket.write_all(&out).await?;
                                if state.rx.is_none() {
                                    state.rx = Some(self.memory.subscribe());
                                }
                                if !state.patterns.contains(&pattern) {
                                    state.patterns.push(pattern);
                                }
                            }
                            Response::PUnsubscribe { patterns, ack } => {
                                ack.write_to(&mut out);
                                socket.write_all(&out).await?;
                                if patterns.is_empty() {
                                    state.patterns.clear();
                                } else {
                                    state.patterns.retain(|p| !patterns.contains(p));
                                }
                                state.maybe_drop_rx();
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        out.clear();
                        RespValue::Error(format!("ERR {e}")).write_to(&mut out);
                        socket.write_all(&out).await?;
                        return Ok(());
                    }
                }
            }

            // Wait for the next event.
            match state.rx.as_mut() {
                Some(rx) => {
                    tokio::select! {
                        read = socket.read_buf(&mut buf) => {
                            let n = read?;
                            if n == 0 { return Ok(()); }
                        }
                        recv = rx.recv() => {
                            match recv {
                                Ok(event) => {
                                    let event_channel = event.channel();
                                    // Exact subscriptions: match channel exactly OR by table name
                                    // (so existing `SUBSCRIBE memory` keeps receiving keyed events).
                                    for ch in &state.exact {
                                        if *ch == event_channel || *ch == event.table {
                                            out.clear();
                                            push_message(ch, &event).write_to(&mut out);
                                            socket.write_all(&out).await?;
                                        }
                                    }
                                    // Pattern subscriptions: glob match against channel.
                                    for pat in &state.patterns {
                                        if glob::glob_match(pat, &event_channel) {
                                            out.clear();
                                            push_pmessage(pat, &event_channel, &event)
                                                .write_to(&mut out);
                                            socket.write_all(&out).await?;
                                        }
                                    }
                                }
                                Err(broadcast::error::RecvError::Lagged(n)) => {
                                    tracing::warn!(missed = n, "subscriber lagged; events lost");
                                }
                                Err(broadcast::error::RecvError::Closed) => {
                                    state.rx = None;
                                }
                            }
                        }
                    }
                }
                None => {
                    let n = socket.read_buf(&mut buf).await?;
                    if n == 0 { return Ok(()); }
                }
            }
        }
    }

    /// Pure dispatch — no subscription / auth side-effects. Used by tests.
    pub fn dispatch(&self, value: RespValue) -> Response {
        let mut authed = true;
        self.dispatch_with_auth(value, false, &mut authed)
    }

    /// Dispatch with auth + subscribe-mode awareness. `authed` is
    /// flipped to true on a successful AUTH command.
    fn dispatch_with_auth(
        &self,
        value: RespValue,
        subscribed: bool,
        authed: &mut bool,
    ) -> Response {
        // Authentication gate: when a token is configured, only
        // PING / AUTH / QUIT / HELLO work pre-auth. Everything else
        // returns NOAUTH to match Redis convention.
        if !*authed {
            // Peek at the command name without consuming `value`.
            let cmd_upper = match &value {
                RespValue::Array(items) if !items.is_empty() => arg_string(&items[0])
                    .map(|s| s.to_ascii_uppercase())
                    .unwrap_or_default(),
                _ => String::new(),
            };
            let allowed_pre_auth =
                matches!(cmd_upper.as_str(), "AUTH" | "PING" | "QUIT" | "HELLO");
            if !allowed_pre_auth {
                return Response::Reply(err(
                    "NOAUTH Authentication required.",
                ));
            }
        }
        self.dispatch_with_sub_inner(value, subscribed, authed)
    }

    /// Dispatch in connection context. `subscribed` is true if the
    /// connection has any active SUBSCRIBE/PSUBSCRIBE.
    fn dispatch_with_sub(&self, value: RespValue, subscribed: bool) -> Response {
        let mut authed = true;
        self.dispatch_with_sub_inner(value, subscribed, &mut authed)
    }

    fn dispatch_with_sub_inner(
        &self,
        value: RespValue,
        subscribed: bool,
        authed: &mut bool,
    ) -> Response {
        let args = match value {
            RespValue::Array(items) => items,
            other => return Response::Reply(err(format!("ERR expected array, got {other:?}"))),
        };
        if args.is_empty() {
            return Response::Reply(err("ERR empty command"));
        }
        let cmd = match arg_string(&args[0]) {
            Some(s) => s.to_ascii_uppercase(),
            None => return Response::Reply(err("ERR command name must be a string")),
        };

        // While subscribed, only the subscribe-mode commands are valid (matches Redis).
        if subscribed {
            match cmd.as_str() {
                "SUBSCRIBE" | "UNSUBSCRIBE" | "PSUBSCRIBE" | "PUNSUBSCRIBE" | "PING" | "QUIT" => {}
                other => {
                    return Response::Reply(err(format!(
                        "ERR can't execute '{other}' in subscribe mode"
                    )));
                }
            }
        }

        match cmd.as_str() {
            "PING" => self.cmd_ping(&args),
            "HELLO" => self.cmd_hello(&args),
            "AUTH" => self.cmd_auth(&args, authed),
            "COMMAND" => self.cmd_command(),
            "INFO" => self.cmd_info(),
            "QUIT" => Response::CloseAfter(simple("OK")),
            "SET" => self.cmd_set(&args),
            "GET" => self.cmd_get(&args),
            "DEL" => self.cmd_del(&args),
            "REMEMBER" => self.cmd_remember(&args),
            "RECALL" => self.cmd_recall(&args),
            "SUBSCRIBE" => self.cmd_subscribe(&args),
            "UNSUBSCRIBE" => self.cmd_unsubscribe(&args),
            "PSUBSCRIBE" => self.cmd_psubscribe(&args),
            "PUNSUBSCRIBE" => self.cmd_punsubscribe(&args),
            other => Response::Reply(err(format!("ERR unknown command '{other}'"))),
        }
    }

    fn cmd_auth(&self, args: &[RespValue], authed: &mut bool) -> Response {
        // Redis AUTH is `AUTH <password>` or `AUTH <user> <password>`.
        // We accept both shapes and ignore the user part (single-tenant).
        let token = match args.len() {
            2 => arg_string(&args[1]),
            3 => arg_string(&args[2]),
            _ => return Response::Reply(err("ERR wrong number of arguments for AUTH")),
        };
        let token = match token {
            Some(s) => s,
            None => return Response::Reply(err("ERR AUTH token must be a string")),
        };
        match &self.auth_token {
            None => Response::Reply(err(
                "ERR Client sent AUTH, but no password is set",
            )),
            Some(expected) => {
                if constant_time_eq(token.as_bytes(), expected.as_bytes()) {
                    *authed = true;
                    Response::Reply(simple("OK"))
                } else {
                    Response::Reply(err("WRONGPASS invalid token"))
                }
            }
        }
    }

    // ----- command handlers ------------------------------------------------

    fn cmd_ping(&self, args: &[RespValue]) -> Response {
        if args.len() == 1 {
            Response::Reply(simple("PONG"))
        } else if let Some(s) = arg_string(&args[1]) {
            Response::Reply(RespValue::bulk(s))
        } else {
            Response::Reply(err("ERR PING arg must be a string"))
        }
    }

    fn cmd_hello(&self, _args: &[RespValue]) -> Response {
        // Minimal HELLO reply: server info as a flat array.
        // (Full RESP3 Map type would be richer; RESP2 array is fine for now.)
        Response::Reply(RespValue::Array(vec![
            RespValue::bulk("server"),
            RespValue::bulk(SERVER_NAME),
            RespValue::bulk("version"),
            RespValue::bulk(SERVER_VERSION),
            RespValue::bulk("proto"),
            RespValue::Integer(2),
            RespValue::bulk("mode"),
            RespValue::bulk("standalone"),
        ]))
    }

    fn cmd_command(&self) -> Response {
        // Very minimal: just list our command names.
        let names = [
            "PING", "HELLO", "AUTH", "COMMAND", "INFO", "QUIT", "SET", "GET", "DEL",
            "REMEMBER", "RECALL", "SUBSCRIBE", "UNSUBSCRIBE", "PSUBSCRIBE", "PUNSUBSCRIBE",
        ];
        let arr: Vec<RespValue> = names.iter().map(|n| RespValue::bulk(*n)).collect();
        Response::Reply(RespValue::Array(arr))
    }

    fn cmd_info(&self) -> Response {
        let info = format!(
            "# Server\r\nserver:{SERVER_NAME}\r\nversion:{SERVER_VERSION}\r\n# Stats\r\nmemories:{}\r\nsessions:{}\r\ndim:{}\r\n",
            self.memory.len(),
            self.sessions.len(),
            self.dim
        );
        Response::Reply(RespValue::BulkString(info.into_bytes()))
    }

    fn cmd_set(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR SET expects 2 args: key, value"));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR SET key must be a string")),
        };
        let value = match &args[2] {
            RespValue::BulkString(b) => b.clone(),
            RespValue::SimpleString(s) => s.clone().into_bytes(),
            _ => return Response::Reply(err("ERR SET value must be a string")),
        };
        self.sessions.put(key.to_string(), value);
        Response::Reply(simple("OK"))
    }

    fn cmd_get(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR GET expects 1 arg: key"));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR GET key must be a string")),
        };
        match self.sessions.get(key) {
            Some(v) => Response::Reply(RespValue::BulkString(v)),
            None => Response::Reply(RespValue::Null),
        }
    }

    fn cmd_del(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR DEL expects 1 arg: key"));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR DEL key must be a string")),
        };
        let removed = self.sessions.delete(key);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    fn cmd_remember(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err("ERR REMEMBER expects: key text..."));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR REMEMBER key must be a string")),
        };
        // Concat remaining args with spaces, so REMEMBER alice hello world works.
        let parts: Vec<String> = args[2..]
            .iter()
            .filter_map(|v| arg_string(v).map(|s| s.to_string()))
            .collect();
        if parts.is_empty() {
            return Response::Reply(err("ERR REMEMBER text required"));
        }
        let text = parts.join(" ");
        let emb = match self.embedder.embed(&text) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR embed: {e}"))),
        };
        if emb.len() != self.dim {
            return Response::Reply(err(format!(
                "ERR embedder dim {} != server dim {}",
                emb.len(),
                self.dim
            )));
        }
        match self.memory.remember(key, text, emb) {
            Ok(id) => Response::Reply(RespValue::Integer(id as i64)),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    fn cmd_subscribe(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR SUBSCRIBE expects 1 arg: channel"));
        }
        let channel = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR SUBSCRIBE channel must be a string")),
        };
        let ack = RespValue::Array(vec![
            RespValue::bulk("subscribe"),
            RespValue::bulk(channel.clone()),
            RespValue::Integer(1),
        ]);
        Response::Subscribe { channel, ack }
    }

    fn cmd_unsubscribe(&self, args: &[RespValue]) -> Response {
        let channels: Vec<String> = args[1..]
            .iter()
            .filter_map(|v| arg_string(v).map(|s| s.to_string()))
            .collect();
        let ack_channel = match channels.first() {
            Some(c) => RespValue::bulk(c.clone()),
            None => RespValue::Null,
        };
        let ack = RespValue::Array(vec![
            RespValue::bulk("unsubscribe"),
            ack_channel,
            RespValue::Integer(0),
        ]);
        Response::Unsubscribe { channels, ack }
    }

    fn cmd_psubscribe(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR PSUBSCRIBE expects 1 arg: pattern"));
        }
        let pattern = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR PSUBSCRIBE pattern must be a string")),
        };
        let ack = RespValue::Array(vec![
            RespValue::bulk("psubscribe"),
            RespValue::bulk(pattern.clone()),
            RespValue::Integer(1),
        ]);
        Response::PSubscribe { pattern, ack }
    }

    fn cmd_punsubscribe(&self, args: &[RespValue]) -> Response {
        let patterns: Vec<String> = args[1..]
            .iter()
            .filter_map(|v| arg_string(v).map(|s| s.to_string()))
            .collect();
        let ack_pattern = match patterns.first() {
            Some(p) => RespValue::bulk(p.clone()),
            None => RespValue::Null,
        };
        let ack = RespValue::Array(vec![
            RespValue::bulk("punsubscribe"),
            ack_pattern,
            RespValue::Integer(0),
        ]);
        Response::PUnsubscribe { patterns, ack }
    }

    fn cmd_recall(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err("ERR RECALL expects: key query [k]"));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR RECALL key must be a string")),
        };
        let query = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR RECALL query must be a string")),
        };
        let k = if args.len() >= 4 {
            arg_string(&args[3])
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(10)
        } else {
            10
        };
        let qvec = match self.embedder.embed(query) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR embed: {e}"))),
        };
        if qvec.len() != self.dim {
            return Response::Reply(err(format!(
                "ERR embedder dim {} != server dim {}",
                qvec.len(),
                self.dim
            )));
        }
        let hits = match self.memory.recall(key, query, &qvec, k) {
            Ok(h) => h,
            Err(e) => return Response::Reply(err(format!("ERR {e}"))),
        };
        // Reply as an array of nested arrays: [id_int, score_str, text_str].
        let arr: Vec<RespValue> = hits
            .into_iter()
            .map(|h| {
                RespValue::Array(vec![
                    RespValue::Integer(h.memory.id as i64),
                    RespValue::bulk(format!("{:.6}", h.score)),
                    RespValue::bulk(h.memory.text),
                ])
            })
            .collect();
        Response::Reply(RespValue::Array(arr))
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

/// What `dispatch` returns. The connection-loop interprets each variant.
#[derive(Debug)]
pub enum Response {
    Reply(RespValue),
    CloseAfter(RespValue),
    Subscribe { channel: String, ack: RespValue },
    Unsubscribe { channels: Vec<String>, ack: RespValue },
    PSubscribe { pattern: String, ack: RespValue },
    PUnsubscribe { patterns: Vec<String>, ack: RespValue },
}

/// Per-connection subscription state.
#[derive(Default)]
struct SubState {
    rx: Option<broadcast::Receiver<ChangeEvent>>,
    exact: Vec<String>,
    patterns: Vec<String>,
}

impl SubState {
    fn is_subscribed(&self) -> bool {
        !self.exact.is_empty() || !self.patterns.is_empty()
    }
    fn maybe_drop_rx(&mut self) {
        if !self.is_subscribed() {
            self.rx = None;
        }
    }
}

fn payload_json(event: &ChangeEvent) -> String {
    let kind = match event.kind {
        ChangeKind::Insert => "insert",
        ChangeKind::Update => "update",
        ChangeKind::Delete => "delete",
    };
    let key = event.key.as_deref().unwrap_or("");
    format!(
        r#"{{"table":"{}","key":"{}","row_id":{},"kind":"{}"}}"#,
        event.table, key, event.row_id, kind
    )
}

/// Redis pub/sub `message` push: `*3 message <channel> <payload>`.
fn push_message(channel: &str, event: &ChangeEvent) -> RespValue {
    RespValue::Array(vec![
        RespValue::bulk("message"),
        RespValue::bulk(channel.to_string()),
        RespValue::bulk(payload_json(event)),
    ])
}

/// Redis pub/sub `pmessage` push: `*4 pmessage <pattern> <channel> <payload>`.
fn push_pmessage(pattern: &str, channel: &str, event: &ChangeEvent) -> RespValue {
    RespValue::Array(vec![
        RespValue::bulk("pmessage"),
        RespValue::bulk(pattern.to_string()),
        RespValue::bulk(channel.to_string()),
        RespValue::bulk(payload_json(event)),
    ])
}

/// Compatibility re-export: older tests still call this by its old name.
#[doc(hidden)]
fn push_event_message(channel: &str, event: &ChangeEvent) -> RespValue {
    push_message(channel, event)
}

fn simple(s: &str) -> RespValue {
    RespValue::SimpleString(s.to_string())
}

fn err(s: impl Into<String>) -> RespValue {
    RespValue::Error(s.into())
}

/// Constant-time byte comparison — keeps token validation immune to
/// timing side-channel attacks. Pulled in here rather than via the
/// `subtle` crate to keep the dep tree small.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn arg_string(v: &RespValue) -> Option<&str> {
    match v {
        RespValue::BulkString(b) => std::str::from_utf8(b).ok(),
        RespValue::SimpleString(s) => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BytesMut;

    fn dispatch_array(server: &Server, items: Vec<RespValue>) -> RespValue {
        match server.dispatch(RespValue::Array(items)) {
            Response::Reply(v)
            | Response::CloseAfter(v)
            | Response::Subscribe { ack: v, .. }
            | Response::Unsubscribe { ack: v, .. }
            | Response::PSubscribe { ack: v, .. }
            | Response::PUnsubscribe { ack: v, .. } => v,
        }
    }

    #[test]
    fn ping_returns_pong() {
        let s = Server::new();
        let r = dispatch_array(&s, vec![RespValue::bulk("PING")]);
        assert_eq!(r, simple("PONG"));
    }

    #[test]
    fn ping_with_arg_echoes() {
        let s = Server::new();
        let r = dispatch_array(
            &s,
            vec![RespValue::bulk("PING"), RespValue::bulk("hello")],
        );
        assert_eq!(r, RespValue::bulk("hello"));
    }

    #[test]
    fn unknown_command_errors() {
        let s = Server::new();
        let r = dispatch_array(&s, vec![RespValue::bulk("WHATEVER")]);
        assert!(matches!(r, RespValue::Error(_)));
    }

    #[test]
    fn set_get_del_session() {
        let s = Server::new();
        let set = dispatch_array(
            &s,
            vec![
                RespValue::bulk("SET"),
                RespValue::bulk("k"),
                RespValue::bulk("v"),
            ],
        );
        assert_eq!(set, simple("OK"));
        let get = dispatch_array(&s, vec![RespValue::bulk("GET"), RespValue::bulk("k")]);
        assert_eq!(get, RespValue::BulkString(b"v".to_vec()));
        let del = dispatch_array(&s, vec![RespValue::bulk("DEL"), RespValue::bulk("k")]);
        assert_eq!(del, RespValue::Integer(1));
        let get2 = dispatch_array(&s, vec![RespValue::bulk("GET"), RespValue::bulk("k")]);
        assert_eq!(get2, RespValue::Null);
    }

    #[test]
    fn remember_then_recall() {
        let s = Server::new();
        let r1 = dispatch_array(
            &s,
            vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("alice"),
                RespValue::bulk("I lost my wallet at the cafe"),
            ],
        );
        assert!(matches!(r1, RespValue::Integer(_)));
        dispatch_array(
            &s,
            vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("alice"),
                RespValue::bulk("My favorite color is blue"),
            ],
        );
        let r2 = dispatch_array(
            &s,
            vec![
                RespValue::bulk("RECALL"),
                RespValue::bulk("alice"),
                RespValue::bulk("wallet"),
                RespValue::bulk("2"),
            ],
        );
        let items = match r2 {
            RespValue::Array(items) => items,
            other => panic!("expected array, got {other:?}"),
        };
        assert!(!items.is_empty());
        // Top result text should mention "wallet".
        let first = match &items[0] {
            RespValue::Array(inner) => inner,
            other => panic!("expected nested array, got {other:?}"),
        };
        let text = match &first[2] {
            RespValue::BulkString(b) => std::str::from_utf8(b).unwrap().to_string(),
            other => panic!("expected bulk text, got {other:?}"),
        };
        assert!(text.to_lowercase().contains("wallet"));
    }

    #[test]
    fn quit_closes_after_ok() {
        let s = Server::new();
        let resp = s.dispatch(RespValue::Array(vec![RespValue::bulk("QUIT")]));
        assert!(matches!(resp, Response::CloseAfter(RespValue::SimpleString(s)) if s == "OK"));
    }

    #[test]
    fn subscribe_returns_three_element_ack() {
        let s = Server::new();
        let resp = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("SUBSCRIBE"),
            RespValue::bulk("memory"),
        ]));
        match resp {
            Response::Subscribe { channel, ack } => {
                assert_eq!(channel, "memory");
                if let RespValue::Array(items) = ack {
                    assert_eq!(items.len(), 3);
                    assert_eq!(items[0], RespValue::bulk("subscribe"));
                    assert_eq!(items[1], RespValue::bulk("memory"));
                    assert_eq!(items[2], RespValue::Integer(1));
                } else {
                    panic!("ack is not array");
                }
            }
            other => panic!("expected Subscribe, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn unsubscribe_returns_ack() {
        let s = Server::new();
        let resp = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("UNSUBSCRIBE"),
            RespValue::bulk("memory"),
        ]));
        assert!(matches!(resp, Response::Unsubscribe { .. }));
    }

    #[test]
    fn remember_publishes_to_subscribers() {
        let s = Server::new();
        let mut rx = s.memory().subscribe();
        s.memory()
            .remember("u", "test memory", vec![0.0; DEFAULT_DIM])
            .unwrap();
        let event = rx.try_recv().expect("expected published event");
        assert_eq!(event.table, "memory");
        assert_eq!(event.key.as_deref(), Some("u"));
        assert_eq!(event.row_id, 1);
        assert_eq!(event.channel(), "memory.u");
    }

    #[test]
    fn push_message_format() {
        let event = ChangeEvent {
            table: "memory".to_string(),
            key: Some("u".to_string()),
            row_id: 42,
            kind: ChangeKind::Insert,
        };
        let msg = push_message("memory.u", &event);
        if let RespValue::Array(items) = msg {
            assert_eq!(items[0], RespValue::bulk("message"));
            assert_eq!(items[1], RespValue::bulk("memory.u"));
            if let RespValue::BulkString(b) = &items[2] {
                let s = std::str::from_utf8(b).unwrap();
                assert!(s.contains(r#""row_id":42"#));
                assert!(s.contains(r#""kind":"insert""#));
                assert!(s.contains(r#""key":"u""#));
            } else {
                panic!("expected bulk payload");
            }
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn pipelined_dispatch_via_codec() {
        // Multi-line input; each command produces one response.
        let s = Server::new();
        let mut buf = BytesMut::from(&b"PING\r\nINFO\r\n"[..]);
        let v1 = resp::parse(&mut buf).unwrap().unwrap();
        let v2 = resp::parse(&mut buf).unwrap().unwrap();
        let unwrap = |r| match r {
            Response::Reply(v)
            | Response::CloseAfter(v)
            | Response::Subscribe { ack: v, .. }
            | Response::Unsubscribe { ack: v, .. }
            | Response::PSubscribe { ack: v, .. }
            | Response::PUnsubscribe { ack: v, .. } => v,
        };
        let r1 = unwrap(s.dispatch(v1));
        let r2 = unwrap(s.dispatch(v2));
        assert_eq!(r1, simple("PONG"));
        assert!(matches!(r2, RespValue::BulkString(_)));
    }

    #[test]
    fn auth_rejects_unauthed_commands_when_token_set() {
        let s = Server::new().with_auth("s3cret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("u"),
                RespValue::bulk("hi"),
            ]),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.starts_with("NOAUTH"), "got: {msg}");
            }
            _ => panic!("expected NOAUTH error"),
        }
    }

    #[test]
    fn auth_accepts_correct_token() {
        let s = Server::new().with_auth("s3cret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![RespValue::bulk("AUTH"), RespValue::bulk("s3cret")]),
            false,
            &mut authed,
        );
        assert!(authed, "should be authed after AUTH");
        match r {
            Response::Reply(RespValue::SimpleString(s)) if s == "OK" => {}
            other => panic!("expected +OK, got {other:?}"),
        }
    }

    #[test]
    fn auth_rejects_wrong_token() {
        let s = Server::new().with_auth("s3cret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![RespValue::bulk("AUTH"), RespValue::bulk("wrong")]),
            false,
            &mut authed,
        );
        assert!(!authed, "should NOT be authed after wrong token");
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.starts_with("WRONGPASS"), "got: {msg}");
            }
            _ => panic!("expected WRONGPASS"),
        }
    }

    #[test]
    fn auth_no_token_set_errors_on_auth_command() {
        // Redis convention: AUTH on an un-passworded server is a client error.
        let s = Server::new();
        let mut authed = true;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![RespValue::bulk("AUTH"), RespValue::bulk("anything")]),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("no password is set"), "got: {msg}");
            }
            _ => panic!("expected error"),
        }
    }

    #[test]
    fn ping_allowed_pre_auth() {
        let s = Server::new().with_auth("s3cret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![RespValue::bulk("PING")]),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::SimpleString(s)) if s == "PONG" => {}
            _ => panic!("PING must work before auth"),
        }
    }

    #[test]
    fn constant_time_eq_basic() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn psubscribe_returns_three_element_ack() {
        let s = Server::new();
        let resp = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PSUBSCRIBE"),
            RespValue::bulk("memory.*"),
        ]));
        match resp {
            Response::PSubscribe { pattern, ack } => {
                assert_eq!(pattern, "memory.*");
                if let RespValue::Array(items) = ack {
                    assert_eq!(items.len(), 3);
                    assert_eq!(items[0], RespValue::bulk("psubscribe"));
                    assert_eq!(items[1], RespValue::bulk("memory.*"));
                    assert_eq!(items[2], RespValue::Integer(1));
                } else {
                    panic!("ack is not array");
                }
            }
            _ => panic!("expected PSubscribe"),
        }
    }

    #[test]
    fn pmessage_push_format() {
        let event = ChangeEvent {
            table: "memory".to_string(),
            key: Some("alice".to_string()),
            row_id: 99,
            kind: ChangeKind::Insert,
        };
        let push = push_pmessage("memory.*", "memory.alice", &event);
        if let RespValue::Array(items) = push {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0], RespValue::bulk("pmessage"));
            assert_eq!(items[1], RespValue::bulk("memory.*"));
            assert_eq!(items[2], RespValue::bulk("memory.alice"));
            if let RespValue::BulkString(b) = &items[3] {
                let s = std::str::from_utf8(b).unwrap();
                assert!(s.contains(r#""key":"alice""#));
                assert!(s.contains(r#""row_id":99"#));
            } else {
                panic!("expected bulk payload");
            }
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn glob_helper_works() {
        use crate::glob::glob_match as m;
        assert!(m("memory.*", "memory.alice"));
        assert!(!m("memory.*", "session.alice"));
        assert!(m("memory.a*", "memory.alice"));
        assert!(!m("memory.a*", "memory.bob"));
    }
}
