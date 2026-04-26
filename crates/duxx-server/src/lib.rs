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

pub mod resp;

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

/// Embedder closure type.
pub type Embedder = Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// The server state — cheaply cloned (Arc internals).
#[derive(Clone)]
pub struct Server {
    memory: MemoryStore,
    sessions: SessionStore,
    embedder: Embedder,
    dim: usize,
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
    pub fn new() -> Self {
        let dim = DEFAULT_DIM;
        Self {
            memory: MemoryStore::with_capacity(dim, 100_000),
            sessions: SessionStore::new(),
            embedder: Arc::new(toy_embed),
            dim,
        }
    }

    pub fn with_embedder<F>(mut self, dim: usize, embedder: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        self.dim = dim;
        self.memory = MemoryStore::with_capacity(dim, 100_000);
        self.embedder = Arc::new(embedder);
        self
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
        tracing::info!(addr = %addr, "duxx-server listening");
        loop {
            let (socket, peer) = listener.accept().await?;
            tracing::debug!(?peer, "accepted");
            let server = self.clone();
            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(socket).await {
                    tracing::warn!(?peer, error = %e, "connection ended with error");
                }
            });
        }
    }

    async fn handle_connection(&self, mut socket: TcpStream) -> anyhow::Result<()> {
        let mut buf = bytes::BytesMut::with_capacity(4096);
        let mut out = Vec::with_capacity(1024);
        // None until the client SUBSCRIBEs; Some after.
        let mut sub_rx: Option<broadcast::Receiver<ChangeEvent>> = None;
        let mut sub_channel: Option<String> = None;

        loop {
            // Drain any complete commands already in `buf`.
            loop {
                match resp::parse(&mut buf) {
                    Ok(Some(v)) => {
                        out.clear();
                        let action = self.dispatch_with_sub(v, sub_channel.as_deref());
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
                                if sub_rx.is_none() {
                                    sub_rx = Some(self.memory.subscribe());
                                }
                                sub_channel = Some(channel);
                            }
                            Response::Unsubscribe { ack } => {
                                ack.write_to(&mut out);
                                socket.write_all(&out).await?;
                                sub_rx = None;
                                sub_channel = None;
                            }
                        }
                    }
                    Ok(None) => break, // need more bytes
                    Err(e) => {
                        out.clear();
                        RespValue::Error(format!("ERR {e}")).write_to(&mut out);
                        socket.write_all(&out).await?;
                        return Ok(());
                    }
                }
            }

            // Wait for the next event: either more bytes, or a published
            // ChangeEvent (only when subscribed).
            match sub_rx.as_mut() {
                Some(rx) => {
                    tokio::select! {
                        read = socket.read_buf(&mut buf) => {
                            let n = read?;
                            if n == 0 { return Ok(()); }
                        }
                        recv = rx.recv() => {
                            match recv {
                                Ok(event) => {
                                    out.clear();
                                    push_event_message(
                                        sub_channel.as_deref().unwrap_or("memory"),
                                        &event,
                                    ).write_to(&mut out);
                                    socket.write_all(&out).await?;
                                }
                                Err(broadcast::error::RecvError::Lagged(n)) => {
                                    tracing::warn!(missed = n, "subscriber lagged; events lost");
                                }
                                Err(broadcast::error::RecvError::Closed) => {
                                    sub_rx = None;
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

    /// Pure dispatch — no subscription side-effects. Used by tests.
    pub fn dispatch(&self, value: RespValue) -> Response {
        self.dispatch_with_sub(value, None)
    }

    /// Dispatch in connection context. `sub_channel` is `Some` while
    /// the connection is in subscribe mode; certain commands reply
    /// differently (e.g. `SUBSCRIBE` to an already-subscribed channel
    /// is idempotent).
    fn dispatch_with_sub(&self, value: RespValue, sub_channel: Option<&str>) -> Response {
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

        // While subscribed, only a few commands are valid (matches Redis).
        if sub_channel.is_some() {
            match cmd.as_str() {
                "SUBSCRIBE" | "UNSUBSCRIBE" | "PING" | "QUIT" => {} // allowed
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
            "COMMAND" => self.cmd_command(),
            "INFO" => self.cmd_info(),
            "QUIT" => Response::CloseAfter(simple("OK")),
            "SET" => self.cmd_set(&args),
            "GET" => self.cmd_get(&args),
            "DEL" => self.cmd_del(&args),
            "REMEMBER" => self.cmd_remember(&args),
            "RECALL" => self.cmd_recall(&args),
            "SUBSCRIBE" => self.cmd_subscribe(&args),
            "UNSUBSCRIBE" => self.cmd_unsubscribe(&args, sub_channel),
            other => Response::Reply(err(format!("ERR unknown command '{other}'"))),
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
            "PING", "HELLO", "COMMAND", "INFO", "QUIT", "SET", "GET", "DEL", "REMEMBER", "RECALL",
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
        let emb = (self.embedder)(&text);
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
        // Redis convention: ack is a 3-element array (subscribe, channel, count).
        let ack = RespValue::Array(vec![
            RespValue::bulk("subscribe"),
            RespValue::bulk(channel.clone()),
            RespValue::Integer(1),
        ]);
        Response::Subscribe { channel, ack }
    }

    fn cmd_unsubscribe(&self, args: &[RespValue], current: Option<&str>) -> Response {
        // UNSUBSCRIBE without args clears all; with channel arg, just that one.
        let channel = if args.len() >= 2 {
            arg_string(&args[1]).unwrap_or("").to_string()
        } else {
            current.unwrap_or("").to_string()
        };
        let ack = RespValue::Array(vec![
            RespValue::bulk("unsubscribe"),
            if channel.is_empty() {
                RespValue::Null
            } else {
                RespValue::bulk(channel)
            },
            RespValue::Integer(0),
        ]);
        Response::Unsubscribe { ack }
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
        let qvec = (self.embedder)(query);
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
pub enum Response {
    /// Write the value, keep the connection open.
    Reply(RespValue),
    /// Write the value, then close the connection.
    CloseAfter(RespValue),
    /// Write the ack, then move into subscribe mode for `channel`.
    Subscribe { channel: String, ack: RespValue },
    /// Write the ack, then leave subscribe mode.
    Unsubscribe { ack: RespValue },
}

/// Build the Redis pub/sub `message` push for one ChangeEvent.
fn push_event_message(channel: &str, event: &ChangeEvent) -> RespValue {
    let kind = match event.kind {
        ChangeKind::Insert => "insert",
        ChangeKind::Update => "update",
        ChangeKind::Delete => "delete",
    };
    let payload = format!(
        r#"{{"table":"{}","row_id":{},"kind":"{}"}}"#,
        event.table, event.row_id, kind
    );
    RespValue::Array(vec![
        RespValue::bulk("message"),
        RespValue::bulk(channel.to_string()),
        RespValue::bulk(payload),
    ])
}

fn simple(s: &str) -> RespValue {
    RespValue::SimpleString(s.to_string())
}

fn err(s: impl Into<String>) -> RespValue {
    RespValue::Error(s.into())
}

fn arg_string(v: &RespValue) -> Option<&str> {
    match v {
        RespValue::BulkString(b) => std::str::from_utf8(b).ok(),
        RespValue::SimpleString(s) => Some(s),
        _ => None,
    }
}

/// Toy 32-d hash-bucket embedder. DO NOT use in production.
fn toy_embed(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = vec![0.0f32; DEFAULT_DIM];
    for token in text.to_lowercase().split_whitespace() {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        let bucket = (h.finish() as usize) % DEFAULT_DIM;
        v[bucket] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in &mut v {
        *x /= norm;
    }
    v
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
            | Response::Unsubscribe { ack: v } => v,
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
        // Synchronous remember publishes a ChangeEvent.
        s.memory()
            .remember("u", "test memory", vec![0.0; DEFAULT_DIM])
            .unwrap();
        let event = rx.try_recv().expect("expected published event");
        assert_eq!(event.table, "memory");
        assert_eq!(event.row_id, 1);
    }

    #[test]
    fn push_event_message_format() {
        let event = ChangeEvent {
            table: "memory".to_string(),
            row_id: 42,
            kind: ChangeKind::Insert,
        };
        let msg = push_event_message("memory", &event);
        if let RespValue::Array(items) = msg {
            assert_eq!(items[0], RespValue::bulk("message"));
            assert_eq!(items[1], RespValue::bulk("memory"));
            if let RespValue::BulkString(b) = &items[2] {
                let s = std::str::from_utf8(b).unwrap();
                assert!(s.contains(r#""row_id":42"#));
                assert!(s.contains(r#""kind":"insert""#));
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
            | Response::Unsubscribe { ack: v } => v,
        };
        let r1 = unwrap(s.dispatch(v1));
        let r2 = unwrap(s.dispatch(v2));
        assert_eq!(r1, simple("PONG"));
        assert!(matches!(r2, RespValue::BulkString(_)));
    }
}
