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
pub mod tls;

use duxx_datasets::{DatasetRegistry, DatasetRow};
use duxx_embed::{Embedder, HashEmbedder};
use duxx_eval::EvalRegistry;
use duxx_replay::{
    InvocationKind as ReplayInvocationKind, OverrideKind as ReplayOverrideKind, ReplayInvocation,
    ReplayMode, ReplayOverride, ReplayRegistry,
};
use duxx_cost::{BudgetPeriod, CostEntry, CostFilter, CostLedger, GroupBy};
use duxx_memory::{MemoryStore, SessionStore};
use duxx_prompts::PromptRegistry;
use duxx_reactive::{ChangeEvent, ChangeKind};
use duxx_trace::{Span, SpanKind, SpanStatus, TraceSearch, TraceStore};
use resp::RespValue;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
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
    /// When Some, the accept loop wraps each TcpStream in a rustls
    /// TlsStream before handing it to `handle_connection`. Phase 6.2.
    tls_config: Option<Arc<rustls::ServerConfig>>,
    /// Phase 7.1: in-process agent trace store. Surfaced via the
    /// TRACE.* RESP commands.
    traces: TraceStore,
    /// Phase 7.2: versioned prompt registry with semantic search.
    /// Surfaced via the PROMPT.* RESP commands. Shares the same
    /// embedder as the memory store so prompts compete in the same
    /// semantic space as memories.
    prompts: PromptRegistry,
    /// Phase 7.3: versioned eval datasets. Surfaced via DATASET.*
    /// commands. Same shared embedder so dataset rows can be searched
    /// semantically alongside memories and prompts.
    datasets: DatasetRegistry,
    /// Phase 7.4: eval runs + scores + regressions + semantic failure
    /// clustering. Surfaced via EVAL.* commands.
    evals: EvalRegistry,
    /// Phase 7.5: deterministic agent replay. Captures LLM/tool
    /// invocations against a trace_id and lets callers re-execute
    /// with overrides. Surfaced via REPLAY.* commands.
    replays: ReplayRegistry,
    /// Phase 7.6: token + cost ledger with per-tenant budgets and
    /// semantic clustering of expensive queries. Surfaced via the
    /// COST.* RESP commands.
    costs: CostLedger,
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
        let prompts = PromptRegistry::new(embedder.clone());
        let datasets = DatasetRegistry::new(embedder.clone());
        let evals = EvalRegistry::new(embedder.clone());
        let replays = ReplayRegistry::new();
        let costs = CostLedger::new(embedder.clone());
        Self {
            memory: MemoryStore::with_capacity(dim, 100_000),
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
            tls_config: None,
            traces: TraceStore::new(),
            prompts,
            datasets,
            evals,
            replays,
            costs,
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

    /// Enable TLS termination on the listener. Pass paths to a
    /// PEM-encoded certificate chain and private key. Once set,
    /// every accepted TCP stream is upgraded to rustls before any
    /// RESP traffic flows.
    ///
    /// Errors if the cert / key files don't exist, can't be parsed,
    /// or don't form a valid pair.
    pub fn with_tls_files(
        mut self,
        cert_path: impl AsRef<std::path::Path>,
        key_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let cfg = tls::load_server_config(cert_path, key_path)?;
        self.tls_config = Some(cfg);
        Ok(self)
    }

    /// Whether this server will terminate TLS on accept.
    pub fn tls_enabled(&self) -> bool {
        self.tls_config.is_some()
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
        let prompts = PromptRegistry::new(embedder.clone());
        let datasets = DatasetRegistry::new(embedder.clone());
        let evals = EvalRegistry::new(embedder.clone());
        let replays = ReplayRegistry::new();
        let costs = CostLedger::new(embedder.clone());
        Ok(Self {
            memory,
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
            tls_config: None,
            traces: TraceStore::new(),
            prompts,
            datasets,
            evals,
            replays,
            costs,
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
        let prompts = PromptRegistry::new(embedder.clone());
        let datasets = DatasetRegistry::new(embedder.clone());
        let evals = EvalRegistry::new(embedder.clone());
        let replays = ReplayRegistry::new();
        let costs = CostLedger::new(embedder.clone());
        Ok(Self {
            memory,
            sessions: SessionStore::new(),
            embedder,
            dim,
            auth_token: None,
            active_conns: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: None,
            tls_config: None,
            traces: TraceStore::new(),
            prompts,
            datasets,
            evals,
            replays,
            costs,
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
            tls = self.tls_config.is_some(),
            "duxx-server listening"
        );
        let acceptor = self
            .tls_config
            .clone()
            .map(tokio_rustls::TlsAcceptor::from);
        loop {
            let (socket, peer) = listener.accept().await?;
            tracing::debug!(?peer, "accepted");
            let server = self.clone();
            self.active_conns
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let counter = self.active_conns.clone();
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let result = match acceptor {
                    Some(acc) => match acc.accept(socket).await {
                        Ok(tls) => server.handle_connection(tls).await,
                        Err(e) => Err(anyhow::anyhow!("TLS handshake: {e}")),
                    },
                    None => server.handle_connection(socket).await,
                };
                if let Err(e) = result {
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
            tls = self.tls_config.is_some(),
            "duxx-server listening"
        );
        let server = self.clone();
        let counter = self.active_conns.clone();
        let acceptor = self
            .tls_config
            .clone()
            .map(tokio_rustls::TlsAcceptor::from);
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
                    let acc = acceptor.clone();
                    tokio::spawn(async move {
                        let result = match acc {
                            Some(a) => match a.accept(socket).await {
                                Ok(tls) => s.handle_connection(tls).await,
                                Err(e) => Err(anyhow::anyhow!("TLS handshake: {e}")),
                            },
                            None => s.handle_connection(socket).await,
                        };
                        if let Err(e) = result {
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

    async fn handle_connection<S>(&self, mut socket: S) -> anyhow::Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
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
            // Phase 7.1: agent observability
            "TRACE.RECORD" => self.cmd_trace_record(&args),
            "TRACE.CLOSE" => self.cmd_trace_close(&args),
            "TRACE.GET" => self.cmd_trace_get(&args),
            "TRACE.SUBTREE" => self.cmd_trace_subtree(&args),
            "TRACE.THREAD" => self.cmd_trace_thread(&args),
            "TRACE.SEARCH" => self.cmd_trace_search(&args),
            // Phase 7.2: versioned prompt registry
            "PROMPT.PUT" => self.cmd_prompt_put(&args),
            "PROMPT.GET" => self.cmd_prompt_get(&args),
            "PROMPT.LIST" => self.cmd_prompt_list(&args),
            "PROMPT.NAMES" => self.cmd_prompt_names(),
            "PROMPT.TAG" => self.cmd_prompt_tag(&args),
            "PROMPT.UNTAG" => self.cmd_prompt_untag(&args),
            "PROMPT.DELETE" => self.cmd_prompt_delete(&args),
            "PROMPT.SEARCH" => self.cmd_prompt_search(&args),
            "PROMPT.DIFF" => self.cmd_prompt_diff(&args),
            // Phase 7.3: versioned eval datasets
            "DATASET.CREATE" => self.cmd_dataset_create(&args),
            "DATASET.ADD" => self.cmd_dataset_add(&args),
            "DATASET.GET" => self.cmd_dataset_get(&args),
            "DATASET.LIST" => self.cmd_dataset_list(&args),
            "DATASET.NAMES" => self.cmd_dataset_names(),
            "DATASET.TAG" => self.cmd_dataset_tag(&args),
            "DATASET.UNTAG" => self.cmd_dataset_untag(&args),
            "DATASET.DELETE" => self.cmd_dataset_delete(&args),
            "DATASET.SAMPLE" => self.cmd_dataset_sample(&args),
            "DATASET.SIZE" => self.cmd_dataset_size(&args),
            "DATASET.SPLITS" => self.cmd_dataset_splits(&args),
            "DATASET.SEARCH" => self.cmd_dataset_search(&args),
            "DATASET.FROM_RECALL" => self.cmd_dataset_from_recall(&args),
            // Phase 7.4: eval runs + regressions + failure clustering
            "EVAL.START" => self.cmd_eval_start(&args),
            "EVAL.SCORE" => self.cmd_eval_score(&args),
            "EVAL.COMPLETE" => self.cmd_eval_complete(&args),
            "EVAL.FAIL" => self.cmd_eval_fail(&args),
            "EVAL.GET" => self.cmd_eval_get(&args),
            "EVAL.SCORES" => self.cmd_eval_scores(&args),
            "EVAL.LIST" => self.cmd_eval_list(&args),
            "EVAL.COMPARE" => self.cmd_eval_compare(&args),
            "EVAL.CLUSTER_FAILURES" => self.cmd_eval_cluster_failures(&args),
            // Phase 7.5: deterministic agent replay
            "REPLAY.CAPTURE" => self.cmd_replay_capture(&args),
            "REPLAY.START" => self.cmd_replay_start(&args),
            "REPLAY.STEP" => self.cmd_replay_step(&args),
            "REPLAY.RECORD" => self.cmd_replay_record(&args),
            "REPLAY.COMPLETE" => self.cmd_replay_complete(&args),
            "REPLAY.FAIL" => self.cmd_replay_fail(&args),
            "REPLAY.GET_SESSION" => self.cmd_replay_get_session(&args),
            "REPLAY.GET_RUN" => self.cmd_replay_get_run(&args),
            "REPLAY.LIST_SESSIONS" => self.cmd_replay_list_sessions(),
            "REPLAY.LIST_RUNS" => self.cmd_replay_list_runs(&args),
            "REPLAY.DIFF" => self.cmd_replay_diff(&args),
            "REPLAY.SET_TRACE" => self.cmd_replay_set_trace(&args),
            // Phase 7.6: token + cost ledger
            "COST.RECORD" => self.cmd_cost_record(&args),
            "COST.QUERY" => self.cmd_cost_query(&args),
            "COST.AGGREGATE" => self.cmd_cost_aggregate(&args),
            "COST.TOTAL" => self.cmd_cost_total(&args),
            "COST.SET_BUDGET" => self.cmd_cost_set_budget(&args),
            "COST.GET_BUDGET" => self.cmd_cost_get_budget(&args),
            "COST.DELETE_BUDGET" => self.cmd_cost_delete_budget(&args),
            "COST.STATUS" => self.cmd_cost_status(&args),
            "COST.ALERTS" => self.cmd_cost_alerts(),
            "COST.CLUSTER_EXPENSIVE" => self.cmd_cost_cluster_expensive(&args),
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
            "TRACE.RECORD", "TRACE.CLOSE", "TRACE.GET", "TRACE.SUBTREE", "TRACE.THREAD",
            "TRACE.SEARCH",
            "PROMPT.PUT", "PROMPT.GET", "PROMPT.LIST", "PROMPT.NAMES", "PROMPT.TAG",
            "PROMPT.UNTAG", "PROMPT.DELETE", "PROMPT.SEARCH", "PROMPT.DIFF",
            "DATASET.CREATE", "DATASET.ADD", "DATASET.GET", "DATASET.LIST", "DATASET.NAMES",
            "DATASET.TAG", "DATASET.UNTAG", "DATASET.DELETE", "DATASET.SAMPLE",
            "DATASET.SIZE", "DATASET.SPLITS", "DATASET.SEARCH", "DATASET.FROM_RECALL",
            "EVAL.START", "EVAL.SCORE", "EVAL.COMPLETE", "EVAL.FAIL", "EVAL.GET",
            "EVAL.SCORES", "EVAL.LIST", "EVAL.COMPARE", "EVAL.CLUSTER_FAILURES",
            "REPLAY.CAPTURE", "REPLAY.START", "REPLAY.STEP", "REPLAY.RECORD",
            "REPLAY.COMPLETE", "REPLAY.FAIL", "REPLAY.GET_SESSION", "REPLAY.GET_RUN",
            "REPLAY.LIST_SESSIONS", "REPLAY.LIST_RUNS", "REPLAY.DIFF", "REPLAY.SET_TRACE",
            "COST.RECORD", "COST.QUERY", "COST.AGGREGATE", "COST.TOTAL",
            "COST.SET_BUDGET", "COST.GET_BUDGET", "COST.DELETE_BUDGET",
            "COST.STATUS", "COST.ALERTS", "COST.CLUSTER_EXPENSIVE",
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

    // ---------------------------------------------------------------- Phase 7.1: TRACE.*

    /// Access the underlying TraceStore (used by tests / external
    /// integrations that want to push spans directly).
    pub fn traces(&self) -> &TraceStore {
        &self.traces
    }

    /// `TRACE.RECORD trace_id span_id parent_span_id name attrs_json
    /// [start_unix_ns] [end_unix_ns] [status] [kind] [thread_id]`
    ///
    /// `parent_span_id` may be the literal "null" / "-" to mean "no
    /// parent" (root span). `attrs_json` may be `"-"` to mean an empty
    /// object. Optional args default to: now / open / unset / internal /
    /// no-thread.
    fn cmd_trace_record(&self, args: &[RespValue]) -> Response {
        if args.len() < 5 {
            return Response::Reply(err(
                "ERR TRACE.RECORD expects: trace_id span_id parent_span_id name attrs_json \
                 [start_ns] [end_ns] [status] [kind] [thread_id]",
            ));
        }
        let trace_id = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR trace_id must be a string")),
        };
        let span_id = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR span_id must be a string")),
        };
        let parent_span_id = arg_string(&args[3]).and_then(|s| {
            if s.is_empty() || s == "-" || s.eq_ignore_ascii_case("null") {
                None
            } else {
                Some(s.to_string())
            }
        });
        let name = match arg_string(&args[4]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let attributes: serde_json::Value = if args.len() >= 6 {
            match arg_string(&args[5]) {
                Some("-") | Some("") => serde_json::Value::Null,
                Some(s) => match serde_json::from_str(s) {
                    Ok(v) => v,
                    Err(e) => {
                        return Response::Reply(err(format!("ERR attrs_json: {e}")));
                    }
                },
                None => serde_json::Value::Null,
            }
        } else {
            serde_json::Value::Null
        };
        let now_ns = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        };
        let start_unix_ns = args
            .get(6)
            .and_then(arg_string)
            .and_then(|s| s.parse::<u128>().ok())
            .unwrap_or_else(now_ns);
        let end_unix_ns = args
            .get(7)
            .and_then(arg_string)
            .and_then(|s| {
                if s == "-" || s.is_empty() {
                    None
                } else {
                    s.parse::<u128>().ok()
                }
            });
        let status = args
            .get(8)
            .and_then(arg_string)
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "ok" => SpanStatus::Ok,
                "error" | "err" => SpanStatus::Error,
                _ => SpanStatus::Unset,
            })
            .unwrap_or(SpanStatus::Unset);
        let kind = args
            .get(9)
            .and_then(arg_string)
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "server" => SpanKind::Server,
                "client" => SpanKind::Client,
                "producer" => SpanKind::Producer,
                "consumer" => SpanKind::Consumer,
                _ => SpanKind::Internal,
            })
            .unwrap_or(SpanKind::Internal);
        let thread_id = args.get(10).and_then(arg_string).and_then(|s| {
            if s.is_empty() || s == "-" {
                None
            } else {
                Some(s.to_string())
            }
        });
        let span = Span {
            trace_id,
            span_id,
            parent_span_id,
            thread_id,
            name,
            kind,
            start_unix_ns,
            end_unix_ns,
            status,
            attributes,
        };
        match self.traces.record_span(span) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `TRACE.CLOSE span_id end_unix_ns [status]`
    fn cmd_trace_close(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err("ERR TRACE.CLOSE expects: span_id end_unix_ns [status]"));
        }
        let span_id = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR span_id must be a string")),
        };
        let end_unix_ns = match args
            .get(2)
            .and_then(arg_string)
            .and_then(|s| s.parse::<u128>().ok())
        {
            Some(n) => n,
            None => return Response::Reply(err("ERR end_unix_ns must be a u128")),
        };
        let status = args
            .get(3)
            .and_then(arg_string)
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "ok" => SpanStatus::Ok,
                "error" | "err" => SpanStatus::Error,
                _ => SpanStatus::Unset,
            })
            .unwrap_or(SpanStatus::Ok);
        match self.traces.close_span(&span_id, end_unix_ns, status) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `TRACE.GET trace_id` — returns an array of span JSON blobs (one per span).
    fn cmd_trace_get(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR TRACE.GET expects: trace_id"));
        }
        let trace_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR trace_id must be a string")),
        };
        spans_to_array_reply(self.traces.get_trace(trace_id))
    }

    /// `TRACE.SUBTREE span_id` — every span under (inclusive) `span_id`.
    fn cmd_trace_subtree(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR TRACE.SUBTREE expects: span_id"));
        }
        let span_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR span_id must be a string")),
        };
        spans_to_array_reply(self.traces.subtree(span_id))
    }

    /// `TRACE.THREAD thread_id` — every span across every trace in `thread_id`.
    fn cmd_trace_thread(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR TRACE.THREAD expects: thread_id"));
        }
        let thread_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR thread_id must be a string")),
        };
        spans_to_array_reply(self.traces.thread(thread_id))
    }

    /// `TRACE.SEARCH filter_json` — linear-scan filter. `filter_json`
    /// keys mirror [`duxx_trace::TraceSearch`]:
    ///   name_prefix, since, until, status, trace_id, kind, limit.
    fn cmd_trace_search(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR TRACE.SEARCH expects: filter_json"));
        }
        let filter_json = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR filter_json must be a string")),
        };
        let filter = match parse_trace_search(filter_json) {
            Ok(f) => f,
            Err(e) => return Response::Reply(err(format!("ERR filter_json: {e}"))),
        };
        let mut hits = self.traces.search(&filter);
        if filter.limit > 0 && hits.len() > filter.limit {
            hits.truncate(filter.limit);
        }
        spans_to_array_reply(hits)
    }

    // ---------------------------------------------------------------- Phase 7.2: PROMPT.*

    /// Access the underlying PromptRegistry (used by tests / external
    /// integrations that want to call the typed API directly).
    pub fn prompts(&self) -> &PromptRegistry {
        &self.prompts
    }

    /// `PROMPT.PUT name content [metadata_json]`
    ///
    /// Inserts a new version of `name`. The registry assigns a
    /// monotonic version number (1, 2, 3, ...). `metadata_json` may be
    /// `"-"` / `""` to mean an empty object. Returns the assigned
    /// version as an integer reply.
    fn cmd_prompt_put(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err(
                "ERR PROMPT.PUT expects: name content [metadata_json]",
            ));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let content = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR content must be a string")),
        };
        let metadata: serde_json::Value = match args.get(3).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Object(serde_json::Map::new()),
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        match self.prompts.put(name, content, metadata) {
            Ok(version) => Response::Reply(RespValue::Integer(version as i64)),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `PROMPT.GET name [version | tag]`
    ///
    /// Without the second argument, returns the latest version. With
    /// a numeric second argument, returns that exact version. With a
    /// non-numeric second argument, treats it as a tag.
    /// Returns the prompt as a single bulk JSON payload (or a nil
    /// bulk-string when not found).
    fn cmd_prompt_get(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err("ERR PROMPT.GET expects: name [version|tag]"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let prompt = match args.get(2).and_then(arg_string) {
            None => self.prompts.get_latest(&name),
            Some(s) => match s.parse::<u64>() {
                Ok(v) => self.prompts.get(&name, v),
                Err(_) => self.prompts.get_by_tag(&name, s),
            },
        };
        match prompt {
            Some(p) => match serde_json::to_vec(&p) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `PROMPT.LIST name` — every version of `name`, ascending,
    /// each as a JSON bulk payload.
    fn cmd_prompt_list(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR PROMPT.LIST expects: name"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let arr: Vec<RespValue> = self
            .prompts
            .list(name)
            .into_iter()
            .filter_map(|p| serde_json::to_vec(&p).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `PROMPT.NAMES` — every prompt name in the registry,
    /// lexicographic.
    fn cmd_prompt_names(&self) -> Response {
        let arr: Vec<RespValue> = self
            .prompts
            .names()
            .into_iter()
            .map(RespValue::bulk)
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `PROMPT.TAG name version tag` — point `tag` at `version`.
    /// Moves the tag if it already exists.
    fn cmd_prompt_tag(&self, args: &[RespValue]) -> Response {
        if args.len() != 4 {
            return Response::Reply(err("ERR PROMPT.TAG expects: name version tag"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let tag = match arg_string(&args[3]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tag must be a string")),
        };
        match self.prompts.tag(name, version, tag) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `PROMPT.UNTAG name tag` — remove a tag. Returns the integer 1
    /// if the tag existed, 0 otherwise.
    fn cmd_prompt_untag(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR PROMPT.UNTAG expects: name tag"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let tag = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tag must be a string")),
        };
        let removed = self.prompts.untag(name, tag);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    /// `PROMPT.DELETE name version` — hard-delete one version. The
    /// version number is NOT reused on subsequent `PROMPT.PUT`.
    /// Returns 1 if deleted, 0 otherwise.
    fn cmd_prompt_delete(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR PROMPT.DELETE expects: name version"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let removed = self.prompts.delete(name, version);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    /// `PROMPT.SEARCH query [k]` — semantic search across the
    /// catalog. Returns up to `k` hits as nested arrays:
    /// `[name, version, score_str, content]`.
    fn cmd_prompt_search(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err("ERR PROMPT.SEARCH expects: query [k]"));
        }
        let query = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR query must be a string")),
        };
        let k: usize = args
            .get(2)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let hits = match self.prompts.search(query, k) {
            Ok(h) => h,
            Err(e) => return Response::Reply(err(format!("ERR {e}"))),
        };
        let arr: Vec<RespValue> = hits
            .into_iter()
            .map(|h| {
                RespValue::Array(vec![
                    RespValue::bulk(h.prompt.name),
                    RespValue::Integer(h.prompt.version as i64),
                    RespValue::bulk(format!("{:.6}", h.score)),
                    RespValue::bulk(h.prompt.content),
                ])
            })
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `PROMPT.DIFF name version_a version_b` — line diff between
    /// two versions. Each output line is prefixed with " ", "-",
    /// or "+".
    fn cmd_prompt_diff(&self, args: &[RespValue]) -> Response {
        if args.len() != 4 {
            return Response::Reply(err(
                "ERR PROMPT.DIFF expects: name version_a version_b",
            ));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let va: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version_a must be a positive integer")),
        };
        let vb: u64 = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version_b must be a positive integer")),
        };
        match self.prompts.diff(name, va, vb) {
            Ok(d) => Response::Reply(RespValue::BulkString(d.into_bytes())),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    // ---------------------------------------------------------------- Phase 7.3: DATASET.*

    /// Access the underlying DatasetRegistry.
    pub fn datasets(&self) -> &DatasetRegistry {
        &self.datasets
    }

    /// `DATASET.CREATE name [schema_json]`
    fn cmd_dataset_create(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err("ERR DATASET.CREATE expects: name [schema_json]"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let schema: serde_json::Value = match args.get(2).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR schema_json: {e}"))),
            },
        };
        match self.datasets.create(name, schema) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `DATASET.ADD name rows_json [metadata_json]`
    ///
    /// `rows_json` is a JSON array. Each element either has the
    /// canonical row shape (`{id, text, data, split, annotations}`)
    /// or just `{text, split?}` for the minimal case.
    fn cmd_dataset_add(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err(
                "ERR DATASET.ADD expects: name rows_json [metadata_json]",
            ));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let rows_json = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR rows_json must be a string")),
        };
        let raw: serde_json::Value = match serde_json::from_str(rows_json) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR rows_json: {e}"))),
        };
        let rows: Vec<DatasetRow> = match parse_rows(raw) {
            Ok(rs) => rs,
            Err(e) => return Response::Reply(err(format!("ERR rows_json: {e}"))),
        };
        let metadata: serde_json::Value = match args.get(3).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        match self.datasets.add(name, rows, metadata) {
            Ok(version) => Response::Reply(RespValue::Integer(version as i64)),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `DATASET.GET name [version|tag]`
    fn cmd_dataset_get(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err("ERR DATASET.GET expects: name [version|tag]"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let ds = match args.get(2).and_then(arg_string) {
            None => self.datasets.get_latest(&name),
            Some(s) => match s.parse::<u64>() {
                Ok(v) => self.datasets.get(&name, v),
                Err(_) => self.datasets.get_by_tag(&name, s),
            },
        };
        match ds {
            Some(d) => match serde_json::to_vec(&d) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `DATASET.LIST name` — every version as a JSON bulk payload.
    fn cmd_dataset_list(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR DATASET.LIST expects: name"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let arr: Vec<RespValue> = self
            .datasets
            .list(name)
            .into_iter()
            .filter_map(|d| serde_json::to_vec(&d).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `DATASET.NAMES` — every known dataset name, lex order.
    fn cmd_dataset_names(&self) -> Response {
        let arr: Vec<RespValue> = self
            .datasets
            .names()
            .into_iter()
            .map(RespValue::bulk)
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `DATASET.TAG name version tag`
    fn cmd_dataset_tag(&self, args: &[RespValue]) -> Response {
        if args.len() != 4 {
            return Response::Reply(err("ERR DATASET.TAG expects: name version tag"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let tag = match arg_string(&args[3]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tag must be a string")),
        };
        match self.datasets.tag(name, version, tag) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `DATASET.UNTAG name tag`
    fn cmd_dataset_untag(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR DATASET.UNTAG expects: name tag"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let tag = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tag must be a string")),
        };
        let removed = self.datasets.untag(name, tag);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    /// `DATASET.DELETE name version`
    fn cmd_dataset_delete(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR DATASET.DELETE expects: name version"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let removed = self.datasets.delete(name, version);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    /// `DATASET.SAMPLE name version n [split]`
    fn cmd_dataset_sample(&self, args: &[RespValue]) -> Response {
        if args.len() < 4 {
            return Response::Reply(err("ERR DATASET.SAMPLE expects: name version n [split]"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let n: usize = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR n must be a non-negative integer")),
        };
        let split = args.get(4).and_then(arg_string).filter(|s| !s.is_empty() && *s != "-");
        let rows = self.datasets.sample(name, version, n, split);
        let arr: Vec<RespValue> = rows
            .into_iter()
            .filter_map(|r| serde_json::to_vec(&r).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `DATASET.SIZE name version [split]` — row count.
    fn cmd_dataset_size(&self, args: &[RespValue]) -> Response {
        if args.len() < 3 {
            return Response::Reply(err("ERR DATASET.SIZE expects: name version [split]"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let split = args.get(3).and_then(arg_string).filter(|s| !s.is_empty() && *s != "-");
        Response::Reply(RespValue::Integer(self.datasets.size(name, version, split) as i64))
    }

    /// `DATASET.SPLITS name version` — distinct split names in a version.
    fn cmd_dataset_splits(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR DATASET.SPLITS expects: name version"));
        }
        let name = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR name must be a string")),
        };
        let version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR version must be a positive integer")),
        };
        let splits = self.datasets.splits(name, version);
        let arr: Vec<RespValue> = splits.into_iter().map(RespValue::bulk).collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `DATASET.SEARCH query [k] [name_filter]` — semantic row search.
    fn cmd_dataset_search(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err("ERR DATASET.SEARCH expects: query [k] [name_filter]"));
        }
        let query = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR query must be a string")),
        };
        let k: usize = args
            .get(2)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let name_filter = args
            .get(3)
            .and_then(arg_string)
            .filter(|s| !s.is_empty() && *s != "-");
        let hits = match self.datasets.search(query, k, name_filter) {
            Ok(h) => h,
            Err(e) => return Response::Reply(err(format!("ERR {e}"))),
        };
        let arr: Vec<RespValue> = hits
            .into_iter()
            .map(|h| {
                RespValue::Array(vec![
                    RespValue::bulk(h.dataset),
                    RespValue::Integer(h.version as i64),
                    RespValue::bulk(h.row.id),
                    RespValue::bulk(format!("{:.6}", h.score)),
                    RespValue::bulk(h.row.text),
                ])
            })
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `DATASET.FROM_RECALL key query k dataset_name [split]`
    ///
    /// Run a memory recall, then store every hit as a row in a NEW
    /// version of `dataset_name`. Returns the assigned version.
    /// This is the killer move: turn any agent-recall result into a
    /// dataset in one round-trip. Memories, datasets, and prompts
    /// share the same vector space.
    fn cmd_dataset_from_recall(&self, args: &[RespValue]) -> Response {
        if args.len() < 5 {
            return Response::Reply(err(
                "ERR DATASET.FROM_RECALL expects: key query k dataset_name [split]",
            ));
        }
        let key = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR key must be a string")),
        };
        let query = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR query must be a string")),
        };
        let k: usize = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR k must be a positive integer")),
        };
        let dataset_name = match arg_string(&args[4]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR dataset_name must be a string")),
        };
        let split = args
            .get(5)
            .and_then(arg_string)
            .map(|s| s.to_string())
            .unwrap_or_default();

        let qvec = match self.embedder.embed(query) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR embed: {e}"))),
        };
        let hits = match self.memory.recall(key, query, &qvec, k) {
            Ok(h) => h,
            Err(e) => return Response::Reply(err(format!("ERR recall: {e}"))),
        };
        let rows: Vec<DatasetRow> = hits
            .into_iter()
            .map(|h| {
                DatasetRow::new(h.memory.text)
                    .with_split(split.clone())
                    .with_annotations(serde_json::json!({
                        "source": "memory.recall",
                        "memory_id": h.memory.id,
                        "key": h.memory.key,
                        "score": h.score,
                    }))
            })
            .collect();
        let metadata = serde_json::json!({
            "source": "memory.recall",
            "query": query,
            "key": key,
            "k": k,
        });
        match self.datasets.add(dataset_name, rows, metadata) {
            Ok(version) => Response::Reply(RespValue::Integer(version as i64)),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    // ---------------------------------------------------------------- Phase 7.4: EVAL.*

    /// Access the underlying EvalRegistry.
    pub fn evals(&self) -> &EvalRegistry {
        &self.evals
    }

    /// `EVAL.START dataset_name dataset_version prompt_name[-] prompt_version[-] model scorer [metadata_json]`
    fn cmd_eval_start(&self, args: &[RespValue]) -> Response {
        if args.len() < 7 {
            return Response::Reply(err(
                "ERR EVAL.START expects: dataset_name dataset_version prompt_name[-] prompt_version[-] model scorer [metadata_json]",
            ));
        }
        let dataset_name = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR dataset_name must be a string")),
        };
        let dataset_version: u64 = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR dataset_version must be a positive integer")),
        };
        let prompt_name = args
            .get(3)
            .and_then(arg_string)
            .filter(|s| !s.is_empty() && *s != "-")
            .map(|s| s.to_string());
        let prompt_version = args
            .get(4)
            .and_then(arg_string)
            .filter(|s| !s.is_empty() && *s != "-")
            .and_then(|s| s.parse::<u64>().ok());
        let model = match arg_string(&args[5]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR model must be a string")),
        };
        let scorer = match arg_string(&args[6]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR scorer must be a string")),
        };
        let metadata: serde_json::Value = match args.get(7).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        let id = self.evals.start(
            dataset_name,
            dataset_version,
            prompt_name,
            prompt_version,
            model,
            scorer,
            metadata,
        );
        Response::Reply(RespValue::BulkString(id.into_bytes()))
    }

    /// `EVAL.SCORE run_id row_id score [output_text] [notes_json]`
    fn cmd_eval_score(&self, args: &[RespValue]) -> Response {
        if args.len() < 4 {
            return Response::Reply(err(
                "ERR EVAL.SCORE expects: run_id row_id score [output_text] [notes_json]",
            ));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let row_id = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR row_id must be a string")),
        };
        let score: f32 = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR score must be a float in [0, 1]")),
        };
        let output_text = args
            .get(4)
            .and_then(arg_string)
            .map(|s| if s == "-" { "" } else { s }.to_string())
            .unwrap_or_default();
        let notes: serde_json::Value = match args.get(5).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR notes_json: {e}"))),
            },
        };
        match self.evals.score(run_id, row_id, score, output_text, notes) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `EVAL.COMPLETE run_id` — returns the summary JSON.
    fn cmd_eval_complete(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR EVAL.COMPLETE expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        match self.evals.complete(run_id) {
            Ok(summary) => match serde_json::to_vec(&summary) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `EVAL.FAIL run_id reason`
    fn cmd_eval_fail(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR EVAL.FAIL expects: run_id reason"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let reason = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR reason must be a string")),
        };
        match self.evals.fail(run_id, reason) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `EVAL.GET run_id` — returns the run as bulk JSON.
    fn cmd_eval_get(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR EVAL.GET expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        match self.evals.get(run_id) {
            Some(r) => match serde_json::to_vec(&r) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `EVAL.SCORES run_id` — every recorded score as JSON bulk payload.
    fn cmd_eval_scores(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR EVAL.SCORES expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let arr: Vec<RespValue> = self
            .evals
            .scores(run_id)
            .into_iter()
            .filter_map(|s| serde_json::to_vec(&s).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `EVAL.LIST [dataset_name] [dataset_version]` — all runs, or
    /// those for a specific dataset/version.
    fn cmd_eval_list(&self, args: &[RespValue]) -> Response {
        let runs = match (args.get(1).and_then(arg_string), args.get(2).and_then(arg_string).and_then(|s| s.parse::<u64>().ok())) {
            (Some(name), Some(v)) => self.evals.list_runs_for(name, v),
            _ => self.evals.list_runs(),
        };
        let arr: Vec<RespValue> = runs
            .into_iter()
            .filter_map(|r| serde_json::to_vec(&r).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `EVAL.COMPARE run_a run_b` — full comparison as bulk JSON.
    fn cmd_eval_compare(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR EVAL.COMPARE expects: run_a run_b"));
        }
        let run_a = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_a must be a string")),
        };
        let run_b = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_b must be a string")),
        };
        match self.evals.compare(run_a, run_b) {
            Ok(cmp) => match serde_json::to_vec(&cmp) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `EVAL.CLUSTER_FAILURES run_id [score_threshold] [sim_threshold] [max_clusters]`
    ///
    /// The differentiator — semantic clustering of failures using the
    /// shared HNSW. Defaults: score<0.5, sim>=0.8, max 10 clusters.
    fn cmd_eval_cluster_failures(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err(
                "ERR EVAL.CLUSTER_FAILURES expects: run_id [score_threshold] [sim_threshold] [max_clusters]",
            ));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let score_threshold: f32 = args
            .get(2)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let sim_threshold: f32 = args
            .get(3)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.8);
        let max_clusters: usize = args
            .get(4)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        match self
            .evals
            .cluster_failures(run_id, score_threshold, sim_threshold, max_clusters)
        {
            Ok(clusters) => {
                let arr: Vec<RespValue> = clusters
                    .into_iter()
                    .filter_map(|c| serde_json::to_vec(&c).ok().map(RespValue::BulkString))
                    .collect();
                Response::Reply(RespValue::Array(arr))
            }
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    // ---------------------------------------------------------------- Phase 7.5: REPLAY.*

    /// Access the underlying ReplayRegistry.
    pub fn replays(&self) -> &ReplayRegistry {
        &self.replays
    }

    /// `REPLAY.CAPTURE trace_id invocation_json`
    ///
    /// `invocation_json` is the JSON form of a [`ReplayInvocation`].
    /// `idx` is auto-assigned in capture order; any value the caller
    /// provides is overwritten.
    fn cmd_replay_capture(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err(
                "ERR REPLAY.CAPTURE expects: trace_id invocation_json",
            ));
        }
        let trace_id = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR trace_id must be a string")),
        };
        let invocation_json = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR invocation_json must be a string")),
        };
        let invocation: ReplayInvocation = match serde_json::from_str(invocation_json) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR invocation_json: {e}"))),
        };
        let idx = self.replays.capture(trace_id, invocation);
        Response::Reply(RespValue::Integer(idx as i64))
    }

    /// `REPLAY.START source_trace_id [mode] [overrides_json] [metadata_json]`
    ///
    /// `mode` is one of `cached` / `live` / `stepped` (default `live`).
    /// `overrides_json` is a JSON array of `ReplayOverride` objects.
    fn cmd_replay_start(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err(
                "ERR REPLAY.START expects: source_trace_id [mode] [overrides_json] [metadata_json]",
            ));
        }
        let source = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR source_trace_id must be a string")),
        };
        let mode = match args.get(2).and_then(arg_string) {
            None | Some("-") | Some("") | Some("live") => ReplayMode::Live,
            Some("cached") => ReplayMode::Cached,
            Some("stepped") => ReplayMode::Stepped,
            Some(other) => {
                return Response::Reply(err(format!("ERR unknown mode '{other}'")));
            }
        };
        let overrides: Vec<ReplayOverride> = match args.get(3).and_then(arg_string) {
            Some("-") | Some("") | None => Vec::new(),
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR overrides_json: {e}"))),
            },
        };
        let metadata: serde_json::Value = match args.get(4).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        match self.replays.start(source, mode, overrides, metadata) {
            Ok(id) => Response::Reply(RespValue::BulkString(id.into_bytes())),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.STEP run_id` -> next invocation JSON (or Null when done).
    fn cmd_replay_step(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR REPLAY.STEP expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        match self.replays.step(run_id) {
            Ok(Some(inv)) => match serde_json::to_vec(&inv) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            Ok(None) => Response::Reply(RespValue::Null),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.RECORD run_id invocation_idx output_json`
    fn cmd_replay_record(&self, args: &[RespValue]) -> Response {
        if args.len() != 4 {
            return Response::Reply(err(
                "ERR REPLAY.RECORD expects: run_id invocation_idx output_json",
            ));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let idx: usize = match args.get(2).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR invocation_idx must be a non-negative integer")),
        };
        let output_json = match arg_string(&args[3]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR output_json must be a string")),
        };
        let output: serde_json::Value = match serde_json::from_str(output_json) {
            Ok(v) => v,
            Err(e) => return Response::Reply(err(format!("ERR output_json: {e}"))),
        };
        match self.replays.record_output(run_id, idx, output) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.COMPLETE run_id`
    fn cmd_replay_complete(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR REPLAY.COMPLETE expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        match self.replays.complete(run_id) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.FAIL run_id reason`
    fn cmd_replay_fail(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err("ERR REPLAY.FAIL expects: run_id reason"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let reason = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR reason must be a string")),
        };
        match self.replays.fail(run_id, reason) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.GET_SESSION trace_id` -> bulk JSON or Null.
    fn cmd_replay_get_session(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR REPLAY.GET_SESSION expects: trace_id"));
        }
        let trace_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR trace_id must be a string")),
        };
        match self.replays.get_session(trace_id) {
            Some(s) => match serde_json::to_vec(&s) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `REPLAY.GET_RUN run_id` -> bulk JSON or Null.
    fn cmd_replay_get_run(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR REPLAY.GET_RUN expects: run_id"));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        match self.replays.get_run(run_id) {
            Some(r) => match serde_json::to_vec(&r) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `REPLAY.LIST_SESSIONS`
    fn cmd_replay_list_sessions(&self) -> Response {
        let arr: Vec<RespValue> = self
            .replays
            .list_sessions()
            .into_iter()
            .filter_map(|s| serde_json::to_vec(&s).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `REPLAY.LIST_RUNS [source_trace_id]`
    fn cmd_replay_list_runs(&self, args: &[RespValue]) -> Response {
        let runs = match args.get(1).and_then(arg_string) {
            Some(s) if !s.is_empty() && s != "-" => self.replays.list_runs_for(s),
            _ => self.replays.list_runs(),
        };
        let arr: Vec<RespValue> = runs
            .into_iter()
            .filter_map(|r| serde_json::to_vec(&r).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `REPLAY.DIFF source_trace_id replay_run_id` -> bulk JSON.
    fn cmd_replay_diff(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err(
                "ERR REPLAY.DIFF expects: source_trace_id replay_run_id",
            ));
        }
        let source = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR source_trace_id must be a string")),
        };
        let run = match arg_string(&args[2]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR replay_run_id must be a string")),
        };
        match self.replays.diff(source, run) {
            Ok(diff) => match serde_json::to_vec(&diff) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `REPLAY.SET_TRACE run_id replay_trace_id` — link a replay run
    /// to the trace it produced. Lets TRACE.* queries pull every
    /// replay of a source by joining on this id.
    fn cmd_replay_set_trace(&self, args: &[RespValue]) -> Response {
        if args.len() != 3 {
            return Response::Reply(err(
                "ERR REPLAY.SET_TRACE expects: run_id replay_trace_id",
            ));
        }
        let run_id = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR run_id must be a string")),
        };
        let replay_trace = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR replay_trace_id must be a string")),
        };
        match self.replays.set_replay_trace_id(run_id, replay_trace) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    // ---------------------------------------------------------------- Phase 7.6: COST.*

    /// Access the underlying CostLedger.
    pub fn costs(&self) -> &CostLedger {
        &self.costs
    }

    /// `COST.RECORD tenant model tokens_in tokens_out cost_usd [input_text] [metadata_json]`
    fn cmd_cost_record(&self, args: &[RespValue]) -> Response {
        if args.len() < 6 {
            return Response::Reply(err(
                "ERR COST.RECORD expects: tenant model tokens_in tokens_out cost_usd [input_text] [metadata_json]",
            ));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        let model = match arg_string(&args[2]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR model must be a string")),
        };
        let tokens_in: u64 = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR tokens_in must be a non-negative integer")),
        };
        let tokens_out: u64 = match args.get(4).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR tokens_out must be a non-negative integer")),
        };
        let cost_usd: f64 = match args.get(5).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR cost_usd must be a non-negative float")),
        };
        let input_text = args
            .get(6)
            .and_then(arg_string)
            .map(|s| if s == "-" { "" } else { s }.to_string())
            .unwrap_or_default();
        let metadata: serde_json::Value = match args.get(7).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        let entry = CostEntry {
            id: String::new(),
            tenant,
            model,
            tokens_in,
            tokens_out,
            cost_usd,
            trace_id: None,
            run_id: None,
            prompt_name: None,
            prompt_version: None,
            input_text,
            metadata,
            recorded_at_unix_ns: 0,
        };
        match self.costs.record(entry) {
            Ok(id) => Response::Reply(RespValue::BulkString(id.into_bytes())),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    fn parse_cost_filter(s: Option<&str>) -> std::result::Result<CostFilter, String> {
        match s {
            None | Some("-") | Some("") => Ok(CostFilter::default()),
            Some(json) => serde_json::from_str(json).map_err(|e| e.to_string()),
        }
    }

    /// `COST.QUERY [filter_json]` — returns entries newest first.
    fn cmd_cost_query(&self, args: &[RespValue]) -> Response {
        let filter = match Self::parse_cost_filter(args.get(1).and_then(arg_string)) {
            Ok(f) => f,
            Err(e) => return Response::Reply(err(format!("ERR filter_json: {e}"))),
        };
        let arr: Vec<RespValue> = self
            .costs
            .query(&filter)
            .into_iter()
            .filter_map(|e| serde_json::to_vec(&e).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `COST.AGGREGATE group_by [filter_json]`
    /// `group_by` ∈ tenant | model | prompt | day | none.
    fn cmd_cost_aggregate(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err(
                "ERR COST.AGGREGATE expects: group_by [filter_json]",
            ));
        }
        let group_by = match arg_string(&args[1]).map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("tenant") => GroupBy::Tenant,
            Some("model") => GroupBy::Model,
            Some("prompt") => GroupBy::Prompt,
            Some("day") | Some("day_utc") => GroupBy::DayUtc,
            Some("none") | Some("all") => GroupBy::None,
            other => {
                return Response::Reply(err(format!(
                    "ERR unknown group_by '{}' (use: tenant | model | prompt | day | none)",
                    other.unwrap_or("")
                )));
            }
        };
        let filter = match Self::parse_cost_filter(args.get(2).and_then(arg_string)) {
            Ok(f) => f,
            Err(e) => return Response::Reply(err(format!("ERR filter_json: {e}"))),
        };
        let arr: Vec<RespValue> = self
            .costs
            .aggregate(&filter, group_by)
            .into_iter()
            .filter_map(|b| serde_json::to_vec(&b).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `COST.TOTAL tenant [since_unix_ns] [until_unix_ns]`
    fn cmd_cost_total(&self, args: &[RespValue]) -> Response {
        if args.len() < 2 {
            return Response::Reply(err(
                "ERR COST.TOTAL expects: tenant [since_unix_ns] [until_unix_ns]",
            ));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        let since = args.get(2).and_then(arg_string).and_then(|s| s.parse::<u128>().ok());
        let until = args.get(3).and_then(arg_string).and_then(|s| s.parse::<u128>().ok());
        let total = self.costs.total_for(tenant, since, until);
        Response::Reply(RespValue::BulkString(format!("{total:.6}").into_bytes()))
    }

    /// `COST.SET_BUDGET tenant period amount_usd [warn_pct] [metadata_json]`
    /// period ∈ daily | weekly | monthly | <seconds>
    fn cmd_cost_set_budget(&self, args: &[RespValue]) -> Response {
        if args.len() < 4 {
            return Response::Reply(err(
                "ERR COST.SET_BUDGET expects: tenant period amount_usd [warn_pct] [metadata_json]",
            ));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s.to_string(),
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        let period = match arg_string(&args[2]).map(|s| s.to_ascii_lowercase()) {
            Some(s) if s == "daily" => BudgetPeriod::Daily,
            Some(s) if s == "weekly" => BudgetPeriod::Weekly,
            Some(s) if s == "monthly" => BudgetPeriod::Monthly,
            Some(s) => match s.parse::<u64>() {
                Ok(secs) => BudgetPeriod::Custom { secs },
                Err(_) => {
                    return Response::Reply(err(format!(
                        "ERR unknown period '{s}' (use: daily | weekly | monthly | <seconds>)"
                    )));
                }
            },
            None => return Response::Reply(err("ERR period must be a string")),
        };
        let amount: f64 = match args.get(3).and_then(arg_string).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return Response::Reply(err("ERR amount_usd must be a non-negative float")),
        };
        let warn_pct: f32 = args
            .get(4)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.8);
        let metadata: serde_json::Value = match args.get(5).and_then(arg_string) {
            Some("-") | Some("") | None => serde_json::Value::Null,
            Some(s) => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => return Response::Reply(err(format!("ERR metadata_json: {e}"))),
            },
        };
        match self.costs.set_budget(tenant, period, amount, warn_pct, metadata) {
            Ok(_) => Response::Reply(simple("OK")),
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }

    /// `COST.GET_BUDGET tenant`
    fn cmd_cost_get_budget(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR COST.GET_BUDGET expects: tenant"));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        match self.costs.get_budget(tenant) {
            Some(b) => match serde_json::to_vec(&b) {
                Ok(bytes) => Response::Reply(RespValue::BulkString(bytes)),
                Err(e) => Response::Reply(err(format!("ERR encode: {e}"))),
            },
            None => Response::Reply(RespValue::Null),
        }
    }

    /// `COST.DELETE_BUDGET tenant`
    fn cmd_cost_delete_budget(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR COST.DELETE_BUDGET expects: tenant"));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        let removed = self.costs.delete_budget(tenant);
        Response::Reply(RespValue::Integer(if removed { 1 } else { 0 }))
    }

    /// `COST.STATUS tenant`
    fn cmd_cost_status(&self, args: &[RespValue]) -> Response {
        if args.len() != 2 {
            return Response::Reply(err("ERR COST.STATUS expects: tenant"));
        }
        let tenant = match arg_string(&args[1]) {
            Some(s) => s,
            None => return Response::Reply(err("ERR tenant must be a string")),
        };
        let status = self.costs.budget_status(tenant);
        let s = match status {
            duxx_cost::BudgetStatus::NoBudget => "no_budget",
            duxx_cost::BudgetStatus::Ok => "ok",
            duxx_cost::BudgetStatus::Warning => "warning",
            duxx_cost::BudgetStatus::Exceeded => "exceeded",
        };
        Response::Reply(RespValue::BulkString(s.as_bytes().to_vec()))
    }

    /// `COST.ALERTS`
    fn cmd_cost_alerts(&self) -> Response {
        let arr: Vec<RespValue> = self
            .costs
            .alerts()
            .into_iter()
            .filter_map(|a| serde_json::to_vec(&a).ok().map(RespValue::BulkString))
            .collect();
        Response::Reply(RespValue::Array(arr))
    }

    /// `COST.CLUSTER_EXPENSIVE [filter_json] [sim_threshold] [max_clusters] [top_n]`
    fn cmd_cost_cluster_expensive(&self, args: &[RespValue]) -> Response {
        let filter = match Self::parse_cost_filter(args.get(1).and_then(arg_string)) {
            Ok(f) => f,
            Err(e) => return Response::Reply(err(format!("ERR filter_json: {e}"))),
        };
        let sim_threshold: f32 = args
            .get(2)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        let max_clusters: usize = args
            .get(3)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let top_n: usize = args
            .get(4)
            .and_then(arg_string)
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        match self.costs.cluster_expensive(&filter, sim_threshold, max_clusters, top_n) {
            Ok(clusters) => {
                let arr: Vec<RespValue> = clusters
                    .into_iter()
                    .filter_map(|c| serde_json::to_vec(&c).ok().map(RespValue::BulkString))
                    .collect();
                Response::Reply(RespValue::Array(arr))
            }
            Err(e) => Response::Reply(err(format!("ERR {e}"))),
        }
    }
}

// Silence dead-code warnings for the imported types we don't reference
// in this file but want to surface in the public API of duxx-server.
#[allow(dead_code)]
fn _replay_type_anchor(
    _: ReplayInvocationKind,
    _: ReplayOverrideKind,
) {
}

/// Parse the rows_json payload of DATASET.ADD. Accepts either a JSON
/// array of objects OR a JSON array of strings (treated as `text` only,
/// auto-id, no split).
fn parse_rows(raw: serde_json::Value) -> std::result::Result<Vec<DatasetRow>, String> {
    let arr = match raw {
        serde_json::Value::Array(a) => a,
        _ => return Err("expected a JSON array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        match v {
            serde_json::Value::String(s) => {
                out.push(DatasetRow::new(s));
            }
            serde_json::Value::Object(_) => {
                let row: DatasetRow = serde_json::from_value(v)
                    .map_err(|e| format!("row decode: {e}"))?;
                // Fill in an id if the caller omitted one.
                let row = if row.id.is_empty() {
                    DatasetRow { id: uuid::Uuid::new_v4().simple().to_string(), ..row }
                } else {
                    row
                };
                out.push(row);
            }
            other => {
                return Err(format!(
                    "expected string or object per row, got {other:?}"
                ))
            }
        }
    }
    Ok(out)
}

fn spans_to_array_reply(spans: Vec<Span>) -> Response {
    let arr: Vec<RespValue> = spans
        .into_iter()
        .filter_map(|s| serde_json::to_vec(&s).ok().map(RespValue::BulkString))
        .collect();
    Response::Reply(RespValue::Array(arr))
}

fn parse_trace_search(json: &str) -> Result<TraceSearch, String> {
    #[derive(serde::Deserialize, Default)]
    struct Raw {
        #[serde(default)]
        name_prefix: Option<String>,
        #[serde(default)]
        since: Option<u128>,
        #[serde(default)]
        until: Option<u128>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        trace_id: Option<String>,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        limit: usize,
    }
    let raw: Raw = serde_json::from_str(json).map_err(|e| e.to_string())?;
    Ok(TraceSearch {
        name_prefix: raw.name_prefix,
        since: raw.since,
        until: raw.until,
        status: raw.status.and_then(|s| match s.to_ascii_lowercase().as_str() {
            "ok" => Some(SpanStatus::Ok),
            "error" | "err" => Some(SpanStatus::Error),
            "unset" => Some(SpanStatus::Unset),
            _ => None,
        }),
        trace_id: raw.trace_id,
        kind: raw.kind.and_then(|s| match s.to_ascii_lowercase().as_str() {
            "server" => Some(SpanKind::Server),
            "client" => Some(SpanKind::Client),
            "internal" => Some(SpanKind::Internal),
            "producer" => Some(SpanKind::Producer),
            "consumer" => Some(SpanKind::Consumer),
            _ => None,
        }),
        limit: raw.limit,
    })
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

    // ---------- Phase 6.2: native TLS handshake end-to-end ----------

    /// Generate a self-signed cert + key pair for "localhost" in a
    /// fresh tempdir. Returns (tempdir, cert_pem_path, key_pem_path).
    /// Tempdir must outlive the file paths.
    fn self_signed_pair_for_test() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        use std::io::Write;
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::File::create(&cert_path)
            .unwrap()
            .write_all(cert.cert.pem().as_bytes())
            .unwrap();
        std::fs::File::create(&key_path)
            .unwrap()
            .write_all(cert.key_pair.serialize_pem().as_bytes())
            .unwrap();
        (dir, cert_path, key_path)
    }

    /// Drive a full RESP-over-TLS round-trip against a `Server` built
    /// with [`Server::with_tls_files`]. Confirms:
    ///   1. The accept loop upgrades each TCP stream to rustls.
    ///   2. The decrypted stream still talks RESP correctly.
    ///   3. PING works end-to-end over TLS.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tls_handshake_round_trips_resp() {
        // Make rustls happy. Idempotent — fine to call from multiple tests.
        let _ = rustls::crypto::ring::default_provider().install_default();

        // 1. Self-signed cert for "localhost".
        let (_dir, cert_path, key_path) = self_signed_pair_for_test();
        let server_cert_pem = std::fs::read(&cert_path).unwrap();

        // 2. Bind the server on an ephemeral port with TLS enabled.
        let server = Server::new().with_tls_files(&cert_path, &key_path).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let local_addr = listener.local_addr().unwrap();
        drop(listener);
        let addr = format!("{local_addr}");

        let server_handle = {
            let s = server.clone();
            let a = addr.clone();
            tokio::spawn(async move {
                let _ = s
                    .serve_with_shutdown(
                        &a,
                        async {
                            // Hold the server up for the whole test.
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        },
                        std::time::Duration::from_secs(1),
                    )
                    .await;
            })
        };

        // Give the listener a beat to come up.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // 3. Build a rustls client that trusts our self-signed cert.
        let mut roots = rustls::RootCertStore::empty();
        for cert in rustls_pemfile::certs(&mut server_cert_pem.as_slice()) {
            roots.add(cert.unwrap()).unwrap();
        }
        let client_cfg = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(client_cfg));
        let domain = rustls::pki_types::ServerName::try_from("localhost").unwrap();

        let tcp = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let mut tls = connector.connect(domain, tcp).await.expect("TLS handshake");

        // 4. Send PING and read PONG.
        let ping = b"*1\r\n$4\r\nPING\r\n";
        tls.write_all(ping).await.unwrap();
        tls.flush().await.unwrap();

        let mut buf = [0u8; 64];
        let n = tls.read(&mut buf).await.unwrap();
        let response = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(response, "+PONG\r\n", "PING response over TLS");

        server_handle.abort();
    }

    // ---------- Phase 7.1: TRACE.* commands ----------

    fn trace_record_args(
        trace_id: &str,
        span_id: &str,
        parent: &str,
        name: &str,
        attrs_json: &str,
    ) -> RespValue {
        RespValue::Array(vec![
            RespValue::bulk("TRACE.RECORD"),
            RespValue::bulk(trace_id),
            RespValue::bulk(span_id),
            RespValue::bulk(parent),
            RespValue::bulk(name),
            RespValue::bulk(attrs_json),
        ])
    }

    #[test]
    fn trace_record_then_get_returns_span_array() {
        let s = Server::new();
        let _ = s.dispatch(trace_record_args(
            "t1",
            "s1",
            "-",
            "agent.run",
            "{\"model\":\"gpt-4o\"}",
        ));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("TRACE.GET"),
            RespValue::bulk("t1"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 1, "one span in trace");
                if let RespValue::BulkString(bytes) = &items[0] {
                    let body = std::str::from_utf8(bytes).unwrap();
                    assert!(body.contains("\"name\":\"agent.run\""));
                    assert!(body.contains("\"model\":\"gpt-4o\""));
                } else {
                    panic!("expected bulk span payload");
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn trace_subtree_returns_descendants() {
        let s = Server::new();
        s.dispatch(trace_record_args("t1", "root", "-", "agent.run", "-"));
        s.dispatch(trace_record_args("t1", "child1", "root", "llm.call", "-"));
        s.dispatch(trace_record_args("t1", "child2", "root", "tool.search", "-"));
        s.dispatch(trace_record_args("t1", "grand", "child1", "embed", "-"));

        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("TRACE.SUBTREE"),
            RespValue::bulk("child1"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn trace_close_marks_span_closed() {
        let s = Server::new();
        s.dispatch(trace_record_args("t1", "s1", "-", "agent.run", "-"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("TRACE.CLOSE"),
            RespValue::bulk("s1"),
            RespValue::bulk("2000000"),
            RespValue::bulk("ok"),
        ]));
        match r {
            Response::Reply(RespValue::SimpleString(ref ok)) => assert_eq!(ok, "OK"),
            other => panic!("expected +OK, got {other:?}"),
        }
        let span = s.traces().get_span("s1").unwrap();
        assert_eq!(span.end_unix_ns, Some(2_000_000));
        assert_eq!(span.status, duxx_trace::SpanStatus::Ok);
    }

    #[test]
    fn trace_record_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            trace_record_args("t1", "s1", "-", "agent.run", "-"),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH error, got {other:?}"),
        }
    }

    // ---------- Phase 7.2: PROMPT.* commands ----------

    fn prompt_put_args(name: &str, content: &str, metadata: &str) -> RespValue {
        RespValue::Array(vec![
            RespValue::bulk("PROMPT.PUT"),
            RespValue::bulk(name),
            RespValue::bulk(content),
            RespValue::bulk(metadata),
        ])
    }

    #[test]
    fn prompt_put_returns_monotonic_versions() {
        let s = Server::new();
        let v1 = s.dispatch(prompt_put_args("greet", "Hi", "-"));
        let v2 = s.dispatch(prompt_put_args("greet", "Hi!", "-"));
        match (v1, v2) {
            (
                Response::Reply(RespValue::Integer(a)),
                Response::Reply(RespValue::Integer(b)),
            ) => {
                assert_eq!((a, b), (1, 2));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn prompt_get_returns_latest_by_default() {
        let s = Server::new();
        s.dispatch(prompt_put_args("g", "first", "-"));
        s.dispatch(prompt_put_args("g", "second", "-"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.GET"),
            RespValue::bulk("g"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"content\":\"second\""));
                assert!(body.contains("\"version\":2"));
            }
            other => panic!("expected bulk JSON, got {other:?}"),
        }
    }

    #[test]
    fn prompt_get_by_tag_resolves_alias() {
        let s = Server::new();
        s.dispatch(prompt_put_args("g", "first", "-"));
        s.dispatch(prompt_put_args("g", "second", "-"));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.TAG"),
            RespValue::bulk("g"),
            RespValue::bulk("1"),
            RespValue::bulk("prod"),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.GET"),
            RespValue::bulk("g"),
            RespValue::bulk("prod"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"content\":\"first\""));
            }
            other => panic!("expected bulk JSON, got {other:?}"),
        }
    }

    #[test]
    fn prompt_get_missing_returns_null() {
        let s = Server::new();
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.GET"),
            RespValue::bulk("does-not-exist"),
        ]));
        match r {
            Response::Reply(RespValue::Null) => {}
            other => panic!("expected Null, got {other:?}"),
        }
    }

    #[test]
    fn prompt_list_returns_all_versions() {
        let s = Server::new();
        for _ in 0..3 {
            s.dispatch(prompt_put_args("g", "x", "-"));
        }
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.LIST"),
            RespValue::bulk("g"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 3);
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn prompt_search_finds_semantically_close_prompts() {
        let s = Server::new();
        s.dispatch(prompt_put_args("a", "hello world how are you", "-"));
        s.dispatch(prompt_put_args("b", "goodbye see you later", "-"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.SEARCH"),
            RespValue::bulk("hello world"),
            RespValue::bulk("1"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert!(!items.is_empty());
                if let RespValue::Array(first) = &items[0] {
                    // First hit should be prompt "a" which contains "hello"
                    if let RespValue::BulkString(name_bytes) = &first[0] {
                        assert_eq!(name_bytes.as_slice(), b"a");
                    }
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn prompt_delete_does_not_reuse_version_number() {
        let s = Server::new();
        s.dispatch(prompt_put_args("g", "v1", "-"));
        s.dispatch(prompt_put_args("g", "v2", "-"));
        let _ = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.DELETE"),
            RespValue::bulk("g"),
            RespValue::bulk("2"),
        ]));
        let r = s.dispatch(prompt_put_args("g", "v3", "-"));
        match r {
            Response::Reply(RespValue::Integer(v)) => assert_eq!(v, 3),
            other => panic!("expected version=3, got {other:?}"),
        }
    }

    #[test]
    fn prompt_diff_marks_added_and_removed_lines() {
        let s = Server::new();
        s.dispatch(prompt_put_args("g", "line a\nshared\nline c", "-"));
        s.dispatch(prompt_put_args("g", "line a\nshared\nNEW line", "-"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("PROMPT.DIFF"),
            RespValue::bulk("g"),
            RespValue::bulk("1"),
            RespValue::bulk("2"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("-line c"), "missing removed line, got:\n{body}");
                assert!(body.contains("+NEW line"), "missing added line, got:\n{body}");
            }
            other => panic!("expected bulk diff, got {other:?}"),
        }
    }

    // ---------- Phase 7.3: DATASET.* commands ----------

    fn dataset_add_args(name: &str, rows_json: &str) -> RespValue {
        RespValue::Array(vec![
            RespValue::bulk("DATASET.ADD"),
            RespValue::bulk(name),
            RespValue::bulk(rows_json),
        ])
    }

    #[test]
    fn dataset_add_returns_monotonic_version() {
        let s = Server::new();
        let v1 = s.dispatch(dataset_add_args(
            "d",
            r#"[{"text": "hello", "split": "train"}]"#,
        ));
        let v2 = s.dispatch(dataset_add_args(
            "d",
            r#"[{"text": "world", "split": "train"}]"#,
        ));
        match (v1, v2) {
            (
                Response::Reply(RespValue::Integer(a)),
                Response::Reply(RespValue::Integer(b)),
            ) => {
                assert_eq!((a, b), (1, 2));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn dataset_get_returns_latest_with_rows() {
        let s = Server::new();
        s.dispatch(dataset_add_args(
            "d",
            r#"[{"text": "q1", "split": "eval"}, {"text": "q2", "split": "eval"}]"#,
        ));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.GET"),
            RespValue::bulk("d"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"version\":1"));
                assert!(body.contains("\"text\":\"q1\""));
            }
            other => panic!("expected bulk JSON, got {other:?}"),
        }
    }

    #[test]
    fn dataset_size_and_splits_reflect_per_split_counts() {
        let s = Server::new();
        s.dispatch(dataset_add_args(
            "d",
            r#"[
                {"text": "a", "split": "train"},
                {"text": "b", "split": "train"},
                {"text": "c", "split": "eval"}
            ]"#,
        ));
        let size = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SIZE"),
            RespValue::bulk("d"),
            RespValue::bulk("1"),
        ]));
        match size {
            Response::Reply(RespValue::Integer(n)) => assert_eq!(n, 3),
            other => panic!("unexpected: {other:?}"),
        }
        let size_eval = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SIZE"),
            RespValue::bulk("d"),
            RespValue::bulk("1"),
            RespValue::bulk("eval"),
        ]));
        match size_eval {
            Response::Reply(RespValue::Integer(n)) => assert_eq!(n, 1),
            other => panic!("unexpected: {other:?}"),
        }
        let splits = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SPLITS"),
            RespValue::bulk("d"),
            RespValue::bulk("1"),
        ]));
        if let Response::Reply(RespValue::Array(items)) = splits {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected array of splits");
        }
    }

    #[test]
    fn dataset_sample_filters_by_split() {
        let s = Server::new();
        s.dispatch(dataset_add_args(
            "d",
            r#"[
                {"text": "a", "split": "train"},
                {"text": "b", "split": "train"},
                {"text": "c", "split": "eval"},
                {"text": "d", "split": "eval"}
            ]"#,
        ));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SAMPLE"),
            RespValue::bulk("d"),
            RespValue::bulk("1"),
            RespValue::bulk("10"),
            RespValue::bulk("eval"),
        ]));
        if let Response::Reply(RespValue::Array(items)) = r {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected array of eval rows");
        }
    }

    #[test]
    fn dataset_search_finds_semantically_close_rows() {
        let s = Server::new();
        s.dispatch(dataset_add_args(
            "d",
            r#"[
                {"text": "hello world how are you", "split": "eval"},
                {"text": "goodbye see you later", "split": "eval"}
            ]"#,
        ));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SEARCH"),
            RespValue::bulk("hello world"),
            RespValue::bulk("1"),
        ]));
        if let Response::Reply(RespValue::Array(items)) = r {
            assert!(!items.is_empty());
            if let RespValue::Array(first) = &items[0] {
                if let RespValue::BulkString(text_bytes) = &first[4] {
                    let txt = std::str::from_utf8(text_bytes).unwrap();
                    assert!(txt.contains("hello"));
                }
            }
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn dataset_delete_does_not_reuse_version() {
        let s = Server::new();
        s.dispatch(dataset_add_args("d", r#"[{"text":"v1"}]"#));
        s.dispatch(dataset_add_args("d", r#"[{"text":"v2"}]"#));
        let _ = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.DELETE"),
            RespValue::bulk("d"),
            RespValue::bulk("2"),
        ]));
        let v3 = s.dispatch(dataset_add_args("d", r#"[{"text":"v3"}]"#));
        match v3 {
            Response::Reply(RespValue::Integer(v)) => assert_eq!(v, 3),
            other => panic!("expected v=3, got {other:?}"),
        }
    }

    #[test]
    fn dataset_from_recall_lifts_memories_into_a_new_dataset() {
        let s = Server::new();
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REMEMBER"),
            RespValue::bulk("alice"),
            RespValue::bulk("I lost my wallet at the cafe"),
        ]));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REMEMBER"),
            RespValue::bulk("alice"),
            RespValue::bulk("Favorite color is blue"),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.FROM_RECALL"),
            RespValue::bulk("alice"),
            RespValue::bulk("wallet"),
            RespValue::bulk("2"),
            RespValue::bulk("recall-dump"),
            RespValue::bulk("eval"),
        ]));
        match r {
            Response::Reply(RespValue::Integer(v)) => assert!(v >= 1),
            other => panic!("expected dataset version, got {other:?}"),
        }
        // The new dataset version should now exist with up to 2 rows.
        let size = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("DATASET.SIZE"),
            RespValue::bulk("recall-dump"),
            RespValue::bulk("1"),
        ]));
        match size {
            Response::Reply(RespValue::Integer(n)) => assert!(n >= 1),
            other => panic!("expected integer size, got {other:?}"),
        }
    }

    // ---------- Phase 7.4: EVAL.* commands ----------

    fn eval_start_run(s: &Server) -> String {
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.START"),
            RespValue::bulk("qa-set"),
            RespValue::bulk("1"),
            RespValue::bulk("classifier"),
            RespValue::bulk("3"),
            RespValue::bulk("gpt-4o"),
            RespValue::bulk("llm_judge"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                String::from_utf8(bytes).unwrap()
            }
            other => panic!("expected bulk run_id, got {other:?}"),
        }
    }

    #[test]
    fn eval_start_returns_a_run_id() {
        let s = Server::new();
        let id = eval_start_run(&s);
        assert!(!id.is_empty());
    }

    #[test]
    fn eval_score_records_and_completes() {
        let s = Server::new();
        let id = eval_start_run(&s);
        for (row, score) in [("a", "0.1"), ("b", "0.5"), ("c", "0.9")] {
            let r = s.dispatch(RespValue::Array(vec![
                RespValue::bulk("EVAL.SCORE"),
                RespValue::bulk(&id),
                RespValue::bulk(row),
                RespValue::bulk(score),
                RespValue::bulk("out"),
            ]));
            match r {
                Response::Reply(RespValue::SimpleString(s)) => assert_eq!(s, "OK"),
                other => panic!("expected +OK, got {other:?}"),
            }
        }
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPLETE"),
            RespValue::bulk(&id),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"total_scored\":3"));
                assert!(body.contains("\"mean\":"));
            }
            other => panic!("expected summary JSON, got {other:?}"),
        }
    }

    #[test]
    fn eval_get_returns_run_with_summary() {
        let s = Server::new();
        let id = eval_start_run(&s);
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.SCORE"),
            RespValue::bulk(&id),
            RespValue::bulk("a"),
            RespValue::bulk("0.5"),
            RespValue::bulk("out"),
        ]));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPLETE"),
            RespValue::bulk(&id),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.GET"),
            RespValue::bulk(&id),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"status\":\"completed\""));
                assert!(body.contains("\"summary\":"));
            }
            other => panic!("expected bulk JSON, got {other:?}"),
        }
    }

    #[test]
    fn eval_compare_detects_regressions() {
        let s = Server::new();
        let a = eval_start_run(&s);
        let b = eval_start_run(&s);
        for (id, row, score) in [
            (&a, "r1", "0.9"), (&a, "r2", "0.5"),
            (&b, "r1", "0.6"), (&b, "r2", "0.9"),
        ] {
            s.dispatch(RespValue::Array(vec![
                RespValue::bulk("EVAL.SCORE"),
                RespValue::bulk(id.as_str()),
                RespValue::bulk(row),
                RespValue::bulk(score),
                RespValue::bulk("out"),
            ]));
        }
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPLETE"),
            RespValue::bulk(&a),
        ]));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPLETE"),
            RespValue::bulk(&b),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPARE"),
            RespValue::bulk(&a),
            RespValue::bulk(&b),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"row_id\":\"r1\"")); // regressed
                assert!(body.contains("\"row_id\":\"r2\"")); // improved
            }
            other => panic!("expected bulk JSON, got {other:?}"),
        }
    }

    #[test]
    fn eval_cluster_failures_groups_similar_outputs() {
        let s = Server::new();
        let id = eval_start_run(&s);
        for (row, text) in [
            ("f1", "the model hallucinated a phone number"),
            ("f2", "model hallucinated phone number for user"),
            ("f3", "model hallucinated a different phone number"),
            ("f4", "the model rejected the prompt as unsafe"),
        ] {
            s.dispatch(RespValue::Array(vec![
                RespValue::bulk("EVAL.SCORE"),
                RespValue::bulk(&id),
                RespValue::bulk(row),
                RespValue::bulk("0.1"),
                RespValue::bulk(text),
            ]));
        }
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.COMPLETE"),
            RespValue::bulk(&id),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.CLUSTER_FAILURES"),
            RespValue::bulk(&id),
            RespValue::bulk("0.5"),
            RespValue::bulk("0.5"),
            RespValue::bulk("10"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert!(!items.is_empty(), "expected at least one cluster");
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn eval_list_filters_by_dataset() {
        let s = Server::new();
        let _a = eval_start_run(&s);
        // Second run on a different dataset.
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.START"),
            RespValue::bulk("other-set"),
            RespValue::bulk("1"),
            RespValue::bulk("-"),
            RespValue::bulk("-"),
            RespValue::bulk("gpt-4o"),
            RespValue::bulk("exact_match"),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.LIST"),
            RespValue::bulk("qa-set"),
            RespValue::bulk("1"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                // Only qa-set runs should come back.
                for item in &items {
                    if let RespValue::BulkString(bytes) = item {
                        let body = std::str::from_utf8(bytes).unwrap();
                        assert!(body.contains("\"dataset_name\":\"qa-set\""));
                    }
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn eval_score_invalid_out_of_range_errors() {
        let s = Server::new();
        let id = eval_start_run(&s);
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("EVAL.SCORE"),
            RespValue::bulk(&id),
            RespValue::bulk("row"),
            RespValue::bulk("1.5"),
            RespValue::bulk("out"),
        ]));
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("score must be in"), "got: {msg}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    // ---------- Phase 7.5: REPLAY.* commands ----------

    fn replay_capture_args(trace_id: &str, idx_seed: u64, input_text: &str, output_text: &str) -> RespValue {
        let inv = serde_json::json!({
            "idx": 0,
            "span_id": format!("span-{idx_seed}"),
            "kind": {"kind": "llm_call"},
            "model": "gpt-4o",
            "input": {"prompt": input_text},
            "output": output_text,
            "recorded_at_unix_ns": 0
        });
        RespValue::Array(vec![
            RespValue::bulk("REPLAY.CAPTURE"),
            RespValue::bulk(trace_id),
            RespValue::bulk(inv.to_string()),
        ])
    }

    fn capture_three(s: &Server, trace_id: &str) {
        s.dispatch(replay_capture_args(trace_id, 0, "hi", "hello world"));
        s.dispatch(replay_capture_args(trace_id, 1, "2+2?", "4"));
        s.dispatch(replay_capture_args(trace_id, 2, "bye", "goodbye"));
    }

    #[test]
    fn replay_capture_returns_assigned_idx() {
        let s = Server::new();
        let r0 = s.dispatch(replay_capture_args("t1", 0, "hi", "ok"));
        let r1 = s.dispatch(replay_capture_args("t1", 1, "again", "ok"));
        match (r0, r1) {
            (
                Response::Reply(RespValue::Integer(a)),
                Response::Reply(RespValue::Integer(b)),
            ) => assert_eq!((a, b), (0, 1)),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn replay_get_session_returns_captured_invocations() {
        let s = Server::new();
        capture_three(&s, "t1");
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.GET_SESSION"),
            RespValue::bulk("t1"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"trace_id\":\"t1\""));
                assert!(body.contains("\"fingerprint\":"));
                // Three invocations got captured.
                let count = body.matches("\"idx\":").count();
                assert_eq!(count, 3);
            }
            other => panic!("expected bulk session JSON, got {other:?}"),
        }
    }

    fn replay_start(s: &Server, source: &str, mode: &str, overrides_json: &str) -> String {
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.START"),
            RespValue::bulk(source),
            RespValue::bulk(mode),
            RespValue::bulk(overrides_json),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                String::from_utf8(bytes).unwrap()
            }
            other => panic!("expected run_id, got {other:?}"),
        }
    }

    #[test]
    fn replay_step_returns_invocations_in_order() {
        let s = Server::new();
        capture_three(&s, "t1");
        let run = replay_start(&s, "t1", "live", "-");
        for expected_idx in [0u64, 1, 2] {
            let r = s.dispatch(RespValue::Array(vec![
                RespValue::bulk("REPLAY.STEP"),
                RespValue::bulk(&run),
            ]));
            match r {
                Response::Reply(RespValue::BulkString(bytes)) => {
                    let body = std::str::from_utf8(&bytes).unwrap();
                    assert!(body.contains(&format!("\"idx\":{expected_idx}")));
                }
                other => panic!("expected invocation JSON, got {other:?}"),
            }
        }
        // 4th step is the end of the session.
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.STEP"),
            RespValue::bulk(&run),
        ]));
        matches!(r, Response::Reply(RespValue::Null));
    }

    #[test]
    fn replay_start_with_swap_model_override() {
        let s = Server::new();
        capture_three(&s, "t1");
        let overrides = r#"[{"at_idx": 1, "kind": {"kind": "swap_model", "model": "claude-4.5-sonnet"}}]"#;
        let run = replay_start(&s, "t1", "live", overrides);
        // Burn idx 0.
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.STEP"),
            RespValue::bulk(&run),
        ]));
        // Idx 1 should come back with the swapped model.
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.STEP"),
            RespValue::bulk(&run),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"model\":\"claude-4.5-sonnet\""));
            }
            other => panic!("expected invocation JSON, got {other:?}"),
        }
    }

    #[test]
    fn replay_record_and_diff_show_changes() {
        let s = Server::new();
        capture_three(&s, "t1");
        let run = replay_start(&s, "t1", "live", "-");
        // Match idx 0 (same as original), differ on idx 1.
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.RECORD"),
            RespValue::bulk(&run),
            RespValue::bulk("0"),
            RespValue::bulk("\"hello world\""),
        ]));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.RECORD"),
            RespValue::bulk(&run),
            RespValue::bulk("1"),
            RespValue::bulk("\"FIVE\""),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.DIFF"),
            RespValue::bulk("t1"),
            RespValue::bulk(&run),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                // idx 0 = same, idx 1 = differs, idx 2 = missing (differs).
                assert!(body.contains("\"differing_count\":2"));
            }
            other => panic!("expected diff JSON, got {other:?}"),
        }
    }

    #[test]
    fn replay_complete_freezes_run() {
        let s = Server::new();
        capture_three(&s, "t1");
        let run = replay_start(&s, "t1", "live", "-");
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.COMPLETE"),
            RespValue::bulk(&run),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.GET_RUN"),
            RespValue::bulk(&run),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("\"status\":\"completed\""));
            }
            other => panic!("expected run JSON, got {other:?}"),
        }
    }

    #[test]
    fn replay_list_runs_filters_by_source() {
        let s = Server::new();
        capture_three(&s, "t1");
        s.dispatch(replay_capture_args("t2", 0, "x", "y"));
        let _run_a = replay_start(&s, "t1", "live", "-");
        let _run_b = replay_start(&s, "t2", "live", "-");
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.LIST_RUNS"),
            RespValue::bulk("t1"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                // Only the t1 run comes back.
                assert_eq!(items.len(), 1);
                if let RespValue::BulkString(bytes) = &items[0] {
                    let body = std::str::from_utf8(bytes).unwrap();
                    assert!(body.contains("\"source_trace_id\":\"t1\""));
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn replay_start_on_missing_session_errors() {
        let s = Server::new();
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("REPLAY.START"),
            RespValue::bulk("does-not-exist"),
        ]));
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("not found"), "got: {msg}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    // ---------- Phase 7.6: COST.* commands ----------

    fn cost_record_args(tenant: &str, model: &str, tok_in: u64, tok_out: u64, usd: &str, input: &str) -> RespValue {
        RespValue::Array(vec![
            RespValue::bulk("COST.RECORD"),
            RespValue::bulk(tenant),
            RespValue::bulk(model),
            RespValue::bulk(tok_in.to_string()),
            RespValue::bulk(tok_out.to_string()),
            RespValue::bulk(usd),
            RespValue::bulk(input),
        ])
    }

    #[test]
    fn cost_record_returns_uuid() {
        let s = Server::new();
        let r = s.dispatch(cost_record_args("acme", "gpt-4o", 100, 200, "0.01", "hi"));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let id = String::from_utf8(bytes).unwrap();
                assert_eq!(id.len(), 32, "uuid simple form");
            }
            other => panic!("expected bulk id, got {other:?}"),
        }
    }

    #[test]
    fn cost_query_returns_entries() {
        let s = Server::new();
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "1.00", "a"));
        s.dispatch(cost_record_args("acme", "claude", 1, 1, "2.00", "b"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.QUERY"),
            RespValue::bulk(r#"{"tenant": "acme"}"#),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => assert_eq!(items.len(), 2),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn cost_aggregate_groups_correctly() {
        let s = Server::new();
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "1.00", ""));
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "2.00", ""));
        s.dispatch(cost_record_args("globex", "gpt-4o", 1, 1, "5.00", ""));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.AGGREGATE"),
            RespValue::bulk("tenant"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 2);
                if let RespValue::BulkString(bytes) = &items[0] {
                    let body = std::str::from_utf8(bytes).unwrap();
                    // Sorted by total desc → globex (5.0) first.
                    assert!(body.contains("\"key\":\"globex\""));
                    assert!(body.contains("\"total_usd\":5"));
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn cost_total_returns_dollar_sum() {
        let s = Server::new();
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "1.25", ""));
        s.dispatch(cost_record_args("acme", "claude", 1, 1, "0.50", ""));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.TOTAL"),
            RespValue::bulk("acme"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(bytes)) => {
                let body = std::str::from_utf8(&bytes).unwrap();
                let v: f64 = body.parse().unwrap();
                assert!((v - 1.75).abs() < 0.001);
            }
            other => panic!("expected bulk float, got {other:?}"),
        }
    }

    #[test]
    fn cost_budget_status_transitions() {
        let s = Server::new();
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.SET_BUDGET"),
            RespValue::bulk("acme"),
            RespValue::bulk("daily"),
            RespValue::bulk("10.0"),
            RespValue::bulk("0.8"),
        ]));
        // Empty → ok.
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.STATUS"),
            RespValue::bulk("acme"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(b)) => {
                assert_eq!(std::str::from_utf8(&b).unwrap(), "ok");
            }
            other => panic!("expected status, got {other:?}"),
        }
        // Cross warn line.
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "9.0", ""));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.STATUS"),
            RespValue::bulk("acme"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(b)) => {
                assert_eq!(std::str::from_utf8(&b).unwrap(), "warning");
            }
            other => panic!("expected status, got {other:?}"),
        }
        // Cross budget.
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "5.0", ""));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.STATUS"),
            RespValue::bulk("acme"),
        ]));
        match r {
            Response::Reply(RespValue::BulkString(b)) => {
                assert_eq!(std::str::from_utf8(&b).unwrap(), "exceeded");
            }
            other => panic!("expected status, got {other:?}"),
        }
    }

    #[test]
    fn cost_alerts_returns_only_at_or_above_warn() {
        let s = Server::new();
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.SET_BUDGET"),
            RespValue::bulk("acme"),
            RespValue::bulk("daily"),
            RespValue::bulk("10.0"),
            RespValue::bulk("0.8"),
        ]));
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.SET_BUDGET"),
            RespValue::bulk("globex"),
            RespValue::bulk("daily"),
            RespValue::bulk("100.0"),
            RespValue::bulk("0.8"),
        ]));
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "9.0", ""));
        s.dispatch(cost_record_args("globex", "gpt-4o", 1, 1, "1.0", ""));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.ALERTS"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 1);
                if let RespValue::BulkString(bytes) = &items[0] {
                    let body = std::str::from_utf8(bytes).unwrap();
                    assert!(body.contains("\"tenant\":\"acme\""));
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn cost_cluster_expensive_returns_clusters() {
        let s = Server::new();
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "5.0", "user asked phone number"));
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "4.0", "phone number lookup"));
        s.dispatch(cost_record_args("acme", "gpt-4o", 1, 1, "0.01", "what's the weather"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.CLUSTER_EXPENSIVE"),
            RespValue::bulk("-"),
            RespValue::bulk("0.5"),
            RespValue::bulk("5"),
            RespValue::bulk("50"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert!(!items.is_empty(), "expected at least one cluster");
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn cost_delete_budget_returns_one_when_existed() {
        let s = Server::new();
        s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.SET_BUDGET"),
            RespValue::bulk("acme"),
            RespValue::bulk("daily"),
            RespValue::bulk("10.0"),
        ]));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.DELETE_BUDGET"),
            RespValue::bulk("acme"),
        ]));
        match r {
            Response::Reply(RespValue::Integer(n)) => assert_eq!(n, 1),
            other => panic!("expected 1, got {other:?}"),
        }
    }

    #[test]
    fn cost_get_budget_missing_returns_null() {
        let s = Server::new();
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("COST.GET_BUDGET"),
            RespValue::bulk("no-such-tenant"),
        ]));
        matches!(r, Response::Reply(RespValue::Null));
    }

    #[test]
    fn cost_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            cost_record_args("acme", "gpt-4o", 1, 1, "0.01", "hi"),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH, got {other:?}"),
        }
    }

    #[test]
    fn replay_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            replay_capture_args("t1", 0, "hi", "ok"),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH, got {other:?}"),
        }
    }

    #[test]
    fn eval_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            RespValue::Array(vec![
                RespValue::bulk("EVAL.START"),
                RespValue::bulk("qa-set"),
                RespValue::bulk("1"),
                RespValue::bulk("-"),
                RespValue::bulk("-"),
                RespValue::bulk("gpt-4o"),
                RespValue::bulk("llm_judge"),
            ]),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH, got {other:?}"),
        }
    }

    #[test]
    fn dataset_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            dataset_add_args("d", r#"[{"text":"x"}]"#),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH error, got {other:?}"),
        }
    }

    #[test]
    fn prompt_rejected_pre_auth() {
        let s = Server::new().with_auth("secret");
        let mut authed = false;
        let r = s.dispatch_with_auth(
            prompt_put_args("g", "hi", "-"),
            false,
            &mut authed,
        );
        match r {
            Response::Reply(RespValue::Error(msg)) => {
                assert!(msg.contains("NOAUTH"), "got: {msg}");
            }
            other => panic!("expected NOAUTH error, got {other:?}"),
        }
    }

    #[test]
    fn trace_search_filters_by_name_prefix() {
        let s = Server::new();
        s.dispatch(trace_record_args("t1", "s1", "-", "llm.openai.completion", "-"));
        s.dispatch(trace_record_args("t1", "s2", "-", "tool.web_search", "-"));
        let r = s.dispatch(RespValue::Array(vec![
            RespValue::bulk("TRACE.SEARCH"),
            RespValue::bulk("{\"name_prefix\":\"llm.\"}"),
        ]));
        match r {
            Response::Reply(RespValue::Array(items)) => {
                assert_eq!(items.len(), 1);
                if let RespValue::BulkString(bytes) = &items[0] {
                    let body = std::str::from_utf8(bytes).unwrap();
                    assert!(body.contains("\"name\":\"llm.openai.completion\""));
                }
            }
            other => panic!("expected array, got {other:?}"),
        }
    }
}
