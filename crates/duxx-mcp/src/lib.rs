//! # duxx-mcp
//!
//! Model Context Protocol (MCP) server for DuxxDB. Speaks JSON-RPC 2.0
//! over stdio (newline-delimited messages on stdin/stdout). Any
//! MCP-aware agent (Claude Desktop, Cline, etc.) can drive it as a
//! tool provider with zero glue code.
//!
//! Implemented MCP methods:
//! - `initialize`        — handshake
//! - `tools/list`        — advertise our tool set
//! - `tools/call`        — invoke a tool (`remember`, `recall`, `forget`)
//! - `notifications/initialized` — accepted, ignored
//! - `ping`              — health check
//!
//! Tool inputs use a deterministic toy embedder (32-d hashed buckets)
//! so the server is self-contained for demos. A production deployment
//! plugs in a real provider via [`McpServer::with_embedder`].

use duxx_memory::MemoryStore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::Arc;

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "duxxdb";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_DIM: usize = 32;

/// A function that turns a string into a vector. Caller provides;
/// the server doesn't pretend to ship a real embedding model.
pub type Embedder = Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>;

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// The MCP server. Cheap to clone (`Arc` internals).
#[derive(Clone)]
pub struct McpServer {
    store: MemoryStore,
    embedder: Embedder,
    dim: usize,
}

impl std::fmt::Debug for McpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("dim", &self.dim)
            .field("store_len", &self.store.len())
            .finish_non_exhaustive()
    }
}

impl McpServer {
    /// Build a server with the default toy embedder.
    pub fn new() -> Self {
        let dim = DEFAULT_DIM;
        Self {
            store: MemoryStore::with_capacity(dim, 100_000),
            embedder: Arc::new(toy_embed),
            dim,
        }
    }

    /// Plug in a real embedder. The vector length must match the store's `dim`.
    pub fn with_embedder<F>(mut self, dim: usize, embedder: F) -> Self
    where
        F: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        self.dim = dim;
        self.store = MemoryStore::with_capacity(dim, 100_000);
        self.embedder = Arc::new(embedder);
        self
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Run the server on the given reader/writer until EOF.
    pub fn run<R: BufRead, W: Write>(&self, reader: R, mut writer: W) -> anyhow::Result<()> {
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let response = match serde_json::from_str::<JsonRpcRequest>(line) {
                Ok(req) => self.handle(req),
                Err(e) => Some(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("parse error: {e}"),
                        data: None,
                    }),
                }),
            };
            if let Some(resp) = response {
                let s = serde_json::to_string(&resp)?;
                writeln!(writer, "{s}")?;
                writer.flush()?;
            }
        }
        Ok(())
    }

    /// Dispatch a single request. Returns `None` for notifications
    /// (no response required).
    pub fn handle(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone();
        let result: std::result::Result<Value, JsonRpcError> = match req.method.as_str() {
            "initialize" => Ok(self.initialize()),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.tools_list()),
            "tools/call" => self.tools_call(req.params),
            // Notifications never return a result.
            method if method.starts_with("notifications/") => return None,
            other => Err(JsonRpcError {
                code: -32601,
                message: format!("method not found: {other}"),
                data: None,
            }),
        };
        Some(match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(value),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(e),
            },
        })
    }

    fn initialize(&self) -> Value {
        json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": { "listChanged": false },
            },
        })
    }

    fn tools_list(&self) -> Value {
        json!({
            "tools": [
                {
                    "name": "remember",
                    "description":
                        "Store a memory under a user/agent key. Indexed for hybrid recall.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "key":  { "type": "string", "description": "User/agent partition key." },
                            "text": { "type": "string", "description": "Memory content." }
                        },
                        "required": ["key", "text"]
                    }
                },
                {
                    "name": "recall",
                    "description":
                        "Retrieve the top-k most relevant memories for a query (vector + BM25, RRF-fused).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "key":   { "type": "string", "description": "Partition key (currently informational)." },
                            "query": { "type": "string", "description": "Query string." },
                            "k":     { "type": "integer", "default": 10, "minimum": 1, "maximum": 100 }
                        },
                        "required": ["key", "query"]
                    }
                },
                {
                    "name": "stats",
                    "description": "Return basic server stats: stored memory count, dim, version.",
                    "inputSchema": { "type": "object", "properties": {} }
                }
            ]
        })
    }

    fn tools_call(&self, params: Value) -> std::result::Result<Value, JsonRpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'name'"))?;
        let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

        let body = match name {
            "remember" => self.tool_remember(&args)?,
            "recall" => self.tool_recall(&args)?,
            "stats" => self.tool_stats(),
            other => {
                return Err(JsonRpcError {
                    code: -32602,
                    message: format!("unknown tool: {other}"),
                    data: None,
                })
            }
        };
        Ok(json!({
            "content": [
                {"type": "text", "text": body.to_string()}
            ],
            "isError": false,
        }))
    }

    fn tool_remember(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let key = args.get("key").and_then(Value::as_str).ok_or_else(|| invalid_params("missing 'key'"))?;
        let text = args.get("text").and_then(Value::as_str).ok_or_else(|| invalid_params("missing 'text'"))?;
        let emb = (self.embedder)(text);
        if emb.len() != self.dim {
            return Err(internal(format!(
                "embedder produced dim {} but store expects {}",
                emb.len(),
                self.dim
            )));
        }
        let id = self
            .store
            .remember(key, text, emb)
            .map_err(|e| internal(e.to_string()))?;
        Ok(json!({ "id": id }))
    }

    fn tool_recall(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let key = args.get("key").and_then(Value::as_str).ok_or_else(|| invalid_params("missing 'key'"))?;
        let query = args.get("query").and_then(Value::as_str).ok_or_else(|| invalid_params("missing 'query'"))?;
        let k = args.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;
        let qvec = (self.embedder)(query);
        if qvec.len() != self.dim {
            return Err(internal(format!(
                "embedder produced dim {} but store expects {}",
                qvec.len(),
                self.dim
            )));
        }
        let hits = self
            .store
            .recall(key, query, &qvec, k)
            .map_err(|e| internal(e.to_string()))?;
        let serialized: Vec<Value> = hits
            .into_iter()
            .map(|h| {
                json!({
                    "id": h.memory.id,
                    "score": h.score,
                    "text": h.memory.text,
                })
            })
            .collect();
        Ok(json!({ "hits": serialized }))
    }

    fn tool_stats(&self) -> Value {
        json!({
            "memories":    self.store.len(),
            "dim":         self.dim,
            "server":      SERVER_NAME,
            "version":     SERVER_VERSION,
            "protocol":    MCP_PROTOCOL_VERSION,
        })
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

fn invalid_params(msg: impl Into<String>) -> JsonRpcError {
    JsonRpcError {
        code: -32602,
        message: msg.into(),
        data: None,
    }
}

fn internal(msg: impl Into<String>) -> JsonRpcError {
    JsonRpcError {
        code: -32603,
        message: msg.into(),
        data: None,
    }
}

/// Toy embedder: hash tokens into 32-d vector. Self-contained for demos.
fn toy_embed(text: &str) -> Vec<f32> {
    duxx_toy_embed(text, DEFAULT_DIM)
}

/// Public reusable hash-bucket embedder. Keeps the example crate from
/// having to redefine this helper.
pub fn duxx_toy_embed(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = vec![0.0f32; dim];
    for token in text.to_lowercase().split_whitespace() {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        let bucket = (h.finish() as usize) % dim;
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
    use std::io::Cursor;

    fn run_one(server: &McpServer, request_line: &str) -> serde_json::Value {
        let mut output = Vec::new();
        let input = Cursor::new(request_line.to_string() + "\n");
        server.run(input, &mut output).unwrap();
        let s = String::from_utf8(output).unwrap();
        let line = s.trim().lines().next().unwrap();
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn initialize_responds_with_protocol_version() {
        let s = McpServer::new();
        let resp = run_one(
            &s,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        );
        assert_eq!(resp["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(resp["result"]["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn tools_list_includes_three_tools() {
        let s = McpServer::new();
        let resp = run_one(&s, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"remember"));
        assert!(names.contains(&"recall"));
        assert!(names.contains(&"stats"));
    }

    #[test]
    fn remember_then_recall_via_tools_call() {
        let s = McpServer::new();
        // Remember
        let r1 = run_one(
            &s,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"remember","arguments":{"key":"u","text":"refund my broken order"}}}"#,
        );
        assert_eq!(r1["error"], Value::Null);
        // Recall
        let r2 = run_one(
            &s,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"recall","arguments":{"key":"u","query":"refund","k":5}}}"#,
        );
        // result.content[0].text is a stringified JSON; parse it.
        let body = r2["result"]["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(body).unwrap();
        let hits = parsed["hits"].as_array().unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0]["text"].as_str().unwrap().contains("refund"));
    }

    #[test]
    fn unknown_method_returns_minus_32601() {
        let s = McpServer::new();
        let resp = run_one(
            &s,
            r#"{"jsonrpc":"2.0","id":99,"method":"nonexistent"}"#,
        );
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn notifications_produce_no_response() {
        let s = McpServer::new();
        let mut output = Vec::new();
        let input = Cursor::new(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#,
        );
        s.run(input, &mut output).unwrap();
        assert!(String::from_utf8(output).unwrap().trim().is_empty());
    }

    #[test]
    fn parse_error_returns_minus_32700() {
        let s = McpServer::new();
        let resp = run_one(&s, "not json");
        assert_eq!(resp["error"]["code"], -32700);
    }
}
