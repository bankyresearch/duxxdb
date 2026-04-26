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
use resp::RespValue;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

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

        loop {
            // Try to drain any complete commands already in `buf`.
            loop {
                match resp::parse(&mut buf) {
                    Ok(Some(v)) => {
                        out.clear();
                        let resp = self.dispatch(v);
                        let close = matches!(resp, Response::CloseAfter(_));
                        let value = match resp {
                            Response::Reply(v) | Response::CloseAfter(v) => v,
                        };
                        value.write_to(&mut out);
                        socket.write_all(&out).await?;
                        if close {
                            socket.shutdown().await.ok();
                            return Ok(());
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

            // Read more from the socket.
            let n = socket.read_buf(&mut buf).await?;
            if n == 0 {
                return Ok(()); // peer closed
            }
        }
    }

    fn dispatch(&self, value: RespValue) -> Response {
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

/// What `dispatch` returns: a reply value, plus a hint on whether to
/// close the connection after writing it.
enum Response {
    Reply(RespValue),
    CloseAfter(RespValue),
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
            Response::Reply(v) | Response::CloseAfter(v) => v,
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
    fn pipelined_dispatch_via_codec() {
        // Multi-line input; each command produces one response.
        let s = Server::new();
        let mut buf = BytesMut::from(&b"PING\r\nINFO\r\n"[..]);
        let v1 = resp::parse(&mut buf).unwrap().unwrap();
        let v2 = resp::parse(&mut buf).unwrap().unwrap();
        let r1 = match s.dispatch(v1) { Response::Reply(v) | Response::CloseAfter(v) => v };
        let r2 = match s.dispatch(v2) { Response::Reply(v) | Response::CloseAfter(v) => v };
        assert_eq!(r1, simple("PONG"));
        assert!(matches!(r2, RespValue::BulkString(_)));
    }
}
