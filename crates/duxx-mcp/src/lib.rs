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
//! - `tools/call`        — invoke a tool. The agent hot-tier is exposed as
//!   MCP tools: `remember` / `recall` / `forget` (memory), `session_set` /
//!   `session_get` (per-conversation state), `tool_cache_put` /
//!   `tool_cache_get` (exact + semantic tool-result cache), and `stats`.
//! - `notifications/initialized` — accepted, ignored
//! - `ping`              — health check
//!
//! Tool inputs use a deterministic toy embedder (32-d hashed buckets)
//! so the server is self-contained for demos. A production deployment
//! plugs in a real provider via [`McpServer::with_embedder`].

use duxx_embed::{Embedder, HashEmbedder};
use duxx_memory::{HitKind, MemoryStore, SessionStore, ToolCache};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::Arc;

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
pub const SERVER_NAME: &str = "duxxdb";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_DIM: usize = 32;

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
    sessions: SessionStore,
    tools: ToolCache,
    embedder: Arc<dyn Embedder>,
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
    /// Build a server with the default `HashEmbedder` (32-d).
    pub fn new() -> Self {
        Self::with_provider(Arc::new(HashEmbedder::new(DEFAULT_DIM)))
    }

    /// Build with an explicit embedder. The store's `dim` is taken from
    /// `embedder.dim()`.
    pub fn with_provider(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            store: MemoryStore::with_capacity(dim, 100_000),
            sessions: SessionStore::new(),
            tools: ToolCache::new(),
            embedder,
            dim,
        }
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
                    "name": "forget",
                    "description": "Delete a memory by its id.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "integer", "description": "Memory id returned by remember." }
                        },
                        "required": ["id"]
                    }
                },
                {
                    "name": "compact",
                    "description": "Rebuild the memory index from surviving rows, dropping tombstones left by forget/eviction. Restores recall quality. Returns rows reclaimed.",
                    "inputSchema": { "type": "object", "properties": {} }
                },
                {
                    "name": "session_set",
                    "description": "Store per-conversation working state under a session id (sliding TTL).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" },
                            "value":      { "type": "string", "description": "Opaque value (e.g. JSON state)." }
                        },
                        "required": ["session_id", "value"]
                    }
                },
                {
                    "name": "session_get",
                    "description": "Fetch per-conversation working state for a session id (null if absent/expired).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "session_id": { "type": "string" }
                        },
                        "required": ["session_id"]
                    }
                },
                {
                    "name": "tool_cache_put",
                    "description": "Cache an expensive tool/LLM result keyed by (tool, args), for later exact or semantic reuse.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool":   { "type": "string", "description": "Tool/function name." },
                            "args":   { "type": "string", "description": "Canonical args string." },
                            "output": { "type": "string", "description": "Result to cache." }
                        },
                        "required": ["tool", "args", "output"]
                    }
                },
                {
                    "name": "tool_cache_get",
                    "description": "Look up a cached tool result. Returns an exact hit, or a semantic near-hit (cosine >= 0.95), or a miss.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tool": { "type": "string" },
                            "args": { "type": "string" }
                        },
                        "required": ["tool", "args"]
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
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        let body = match name {
            "remember" => self.tool_remember(&args)?,
            "recall" => self.tool_recall(&args)?,
            "forget" => self.tool_forget(&args)?,
            "compact" => self.tool_compact()?,
            "session_set" => self.tool_session_set(&args)?,
            "session_get" => self.tool_session_get(&args)?,
            "tool_cache_put" => self.tool_cache_put(&args)?,
            "tool_cache_get" => self.tool_cache_get(&args)?,
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
        let key = args
            .get("key")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'key'"))?;
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'text'"))?;
        let emb = self
            .embedder
            .embed(text)
            .map_err(|e| internal(format!("embed: {e}")))?;
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
        let key = args
            .get("key")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'key'"))?;
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'query'"))?;
        let k = args.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;
        let qvec = self
            .embedder
            .embed(query)
            .map_err(|e| internal(format!("embed: {e}")))?;
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

    fn tool_forget(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let id = args
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid_params("missing 'id'"))?;
        Ok(json!({ "removed": self.store.forget(id) }))
    }

    fn tool_compact(&self) -> std::result::Result<Value, JsonRpcError> {
        let reclaimed = self.store.compact().map_err(|e| internal(e.to_string()))?;
        Ok(json!({
            "reclaimed": reclaimed,
            "tombstone_ratio": self.store.tombstone_ratio(),
        }))
    }

    fn tool_session_set(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let sid = args
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'session_id'"))?;
        let value = args
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'value'"))?;
        self.sessions
            .put(sid.to_string(), value.as_bytes().to_vec());
        Ok(json!({ "ok": true }))
    }

    fn tool_session_get(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let sid = args
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'session_id'"))?;
        let value = self
            .sessions
            .get(sid)
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        Ok(json!({ "value": value }))
    }

    fn tool_cache_put(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let tool = args
            .get("tool")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'tool'"))?;
        let input = args
            .get("args")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'args'"))?;
        let output = args
            .get("output")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'output'"))?;
        let emb = self
            .embedder
            .embed(input)
            .map_err(|e| internal(format!("embed: {e}")))?;
        self.tools
            .put(
                tool.to_string(),
                hash_str(input),
                emb,
                output.as_bytes().to_vec(),
                std::time::Duration::from_secs(3600),
            )
            .map_err(|e| internal(e.to_string()))?;
        Ok(json!({ "ok": true }))
    }

    fn tool_cache_get(&self, args: &Value) -> std::result::Result<Value, JsonRpcError> {
        let tool = args
            .get("tool")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'tool'"))?;
        let input = args
            .get("args")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_params("missing 'args'"))?;
        let emb = self
            .embedder
            .embed(input)
            .map_err(|e| internal(format!("embed: {e}")))?;
        Ok(match self.tools.get(tool, hash_str(input), &emb) {
            Some(hit) => json!({
                "hit": true,
                "kind": match hit.kind {
                    HitKind::Exact => "exact",
                    HitKind::SemanticNearHit => "semantic_near_hit",
                },
                "similarity": hit.similarity,
                "output": String::from_utf8_lossy(&hit.result),
            }),
            None => json!({ "hit": false }),
        })
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

/// Stable 64-bit hash of an args string, for the tool cache's exact-match index.
fn hash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
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
    fn tools_list_advertises_the_agent_hot_tier() {
        let s = McpServer::new();
        let resp = run_one(&s, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for expected in [
            "remember",
            "recall",
            "forget",
            "compact",
            "session_set",
            "session_get",
            "tool_cache_put",
            "tool_cache_get",
            "stats",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn compact_tool_returns_reclaimed_count() {
        let s = McpServer::new();
        s.store().set_auto_compact_ratio(None);
        s.store().set_max_rows(Some(2));
        s.store()
            .set_eviction_half_life(std::time::Duration::from_micros(1));
        for i in 0..5 {
            run_one(
                &s,
                &format!(
                    r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"remember","arguments":{{"key":"u","text":"note {i}"}}}}}}"#
                ),
            );
        }
        // 5 inserts at cap 2 → 3 evictions → 3 tombstones to reclaim.
        let resp = run_one(
            &s,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"compact","arguments":{}}}"#,
        );
        let body: Value =
            serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(body["reclaimed"], 3);
        assert_eq!(body["tombstone_ratio"], 0.0);
    }

    fn call_tool(s: &McpServer, id: u32, name: &str, args: serde_json::Value) -> Value {
        let req = json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": { "name": name, "arguments": args },
        })
        .to_string();
        let resp = run_one(s, &req);
        let body = resp["result"]["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn session_set_then_get_round_trips() {
        let s = McpServer::new();
        call_tool(
            &s,
            1,
            "session_set",
            json!({"session_id":"c1","value":"turn-buffer"}),
        );
        let got = call_tool(&s, 2, "session_get", json!({"session_id":"c1"}));
        assert_eq!(got["value"], "turn-buffer");
        let miss = call_tool(&s, 3, "session_get", json!({"session_id":"nope"}));
        assert_eq!(miss["value"], Value::Null);
    }

    #[test]
    fn tool_cache_put_then_get_is_an_exact_hit() {
        let s = McpServer::new();
        call_tool(
            &s,
            1,
            "tool_cache_put",
            json!({"tool":"web_search","args":"weather in paris","output":"18C, clear"}),
        );
        let hit = call_tool(
            &s,
            2,
            "tool_cache_get",
            json!({"tool":"web_search","args":"weather in paris"}),
        );
        assert_eq!(hit["hit"], true);
        assert_eq!(hit["kind"], "exact");
        assert_eq!(hit["output"], "18C, clear");

        let miss = call_tool(
            &s,
            3,
            "tool_cache_get",
            json!({"tool":"web_search","args":"a totally unrelated query xyzzy"}),
        );
        assert_eq!(miss["hit"], false);
    }

    #[test]
    fn forget_removes_a_memory() {
        let s = McpServer::new();
        let r = call_tool(&s, 1, "remember", json!({"key":"u","text":"delete me"}));
        let id = r["id"].as_u64().unwrap();
        let f = call_tool(&s, 2, "forget", json!({"id": id}));
        assert_eq!(f["removed"], true);
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
        let resp = run_one(&s, r#"{"jsonrpc":"2.0","id":99,"method":"nonexistent"}"#);
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn notifications_produce_no_response() {
        let s = McpServer::new();
        let mut output = Vec::new();
        let input =
            Cursor::new(r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#);
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
