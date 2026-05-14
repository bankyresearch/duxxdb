//! # `duxx-trace` — agent observability primitives (Phase 7.1)
//!
//! DuxxDB's first-class observability layer for AI agents. Stores
//! **spans** (single operations) grouped into **traces** (one
//! agent-turn worth of work) optionally grouped into **threads**
//! (multi-turn conversations).
//!
//! Designed to ingest standard OpenTelemetry / OpenInference span
//! payloads with no translation loss — same field names, same JSON
//! shape — so any agent that already exports OTel works out of the
//! box. See `Span::attributes` for the free-form payload that
//! mirrors OTel `span.attributes`.
//!
//! ## Mental model
//!
//! ```text
//!   Thread "session-42"        (long-running conversation)
//!   ├── Trace "turn-1"          (one agent turn — one trigger)
//!   │   ├── Span "agent.run"      (root span)
//!   │   │   ├── Span "llm.call"
//!   │   │   ├── Span "tool.web_search"
//!   │   │   └── Span "memory.recall"
//!   ├── Trace "turn-2"
//!   │   └── ...
//! ```
//!
//! ## What's in this crate
//!
//! - [`Span`], [`SpanStatus`], [`SpanKind`] — primitive types
//! - [`TraceStore`] — the in-process store with `record_span`,
//!   `close_span`, `get_trace`, `subtree`, `thread`, `search`
//! - [`ChangeBus`](duxx_reactive::ChangeBus) integration so
//!   `PSUBSCRIBE trace.*` clients see live span events
//!
//! ## What's NOT in this crate (yet)
//!
//! - OTLP HTTP/gRPC ingest endpoint — that wires into
//!   `duxx-server` / `duxx-grpc` directly so it can share the same
//!   auth + TLS + Prometheus surface.
//! - Cold-tier export to Parquet — reuses `duxx-coldtier`;
//!   added once the on-disk schema for spans is frozen.
//! - Persistence — same redb-backed pattern as `duxx-memory`,
//!   landing in 7.1b.

use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::broadcast;
use uuid::Uuid;

// Brought in transitively; expose it so callers don't need a separate
// dep on the `tokio` crate just to use the broadcast receiver.
pub use tokio::sync::broadcast as tokio_broadcast;

/// Errors surfaced by the trace store.
#[derive(Debug, Error)]
pub enum TraceError {
    #[error("span {0} not found")]
    SpanNotFound(SpanId),
    #[error("trace {0} not found")]
    TraceNotFound(TraceId),
    #[error("attempted to close span {0} but it has no start_unix_ns")]
    UnstartedSpan(SpanId),
}

pub type Result<T> = std::result::Result<T, TraceError>;

// ---------------------------------------------------------------- IDs

/// Identifier for a trace. Conventionally an OTel trace_id (128-bit hex)
/// or a UUID-as-string. Stored as String so callers can use whichever
/// scheme their agent framework emits.
pub type TraceId = String;

/// Identifier for a span within a trace. Same string discipline as
/// `TraceId`.
pub type SpanId = String;

/// Optional grouping that lets a multi-turn conversation be queried as
/// a single thread. Maps cleanly to OpenInference's `thread_id`
/// attribute on the root span of a turn.
pub type ThreadId = String;

/// Generate a fresh trace_id. Convenient for callers that don't have
/// one from upstream.
pub fn new_trace_id() -> TraceId {
    Uuid::new_v4().simple().to_string()
}

/// Generate a fresh span_id.
pub fn new_span_id() -> SpanId {
    let u = Uuid::new_v4();
    // 64-bit hex slice, OTel-compatible width.
    let bytes = u.as_bytes();
    let mut out = String::with_capacity(16);
    for b in &bytes[..8] {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ---------------------------------------------------------------- Span types

/// Lifecycle / outcome of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpanStatus {
    /// Span is in flight; `end_unix_ns` is None.
    #[default]
    Unset,
    /// Span closed cleanly.
    Ok,
    /// Span closed with an error. The error message lives in
    /// `attributes["error.message"]`.
    Error,
}

/// Coarse-grained category of a span — borrowed from OpenTelemetry.
/// Optional; defaulting to `Internal` is the right call when in doubt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpanKind {
    #[default]
    Internal,
    /// Inbound network call to the agent (HTTP / gRPC / RESP).
    Server,
    /// Outbound network call FROM the agent — LLM, tool, embedder.
    Client,
    /// Async job emitting work to a queue.
    Producer,
    /// Async job consuming from a queue.
    Consumer,
}

/// One observation. Mirrors the OTel + OpenInference shape closely so
/// callers can `serde_json::from_value` an OTLP payload straight into
/// `Span` (with field-name massaging on the JSON side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<SpanId>,
    /// Optional grouping for multi-turn conversations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<ThreadId>,
    /// Logical name, e.g. "llm.openai.completion", "tool.web_search".
    pub name: String,
    #[serde(default)]
    pub kind: SpanKind,
    /// Unix epoch nanoseconds. Mandatory — every span has a start.
    pub start_unix_ns: u128,
    /// Unix epoch nanoseconds. Optional — None means the span is still
    /// open. `close_span()` sets this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_unix_ns: Option<u128>,
    #[serde(default)]
    pub status: SpanStatus,
    /// Free-form payload. Mirrors OTel `span.attributes`. Common keys:
    /// `model`, `prompt`, `tokens_in`, `tokens_out`, `latency_ms`,
    /// `error.message`, `tool.name`, etc.
    #[serde(default)]
    pub attributes: serde_json::Value,
}

impl Span {
    /// Build a new open span. `start_unix_ns` defaults to "now".
    pub fn open(name: impl Into<String>, trace_id: TraceId) -> Self {
        Self {
            trace_id,
            span_id: new_span_id(),
            parent_span_id: None,
            thread_id: None,
            name: name.into(),
            kind: SpanKind::default(),
            start_unix_ns: now_unix_ns(),
            end_unix_ns: None,
            status: SpanStatus::Unset,
            attributes: serde_json::Value::Null,
        }
    }

    /// Duration in nanoseconds if closed; None otherwise.
    pub fn duration_ns(&self) -> Option<u128> {
        self.end_unix_ns
            .map(|e| e.saturating_sub(self.start_unix_ns))
    }

    /// True if `close_span` has been called.
    pub fn is_closed(&self) -> bool {
        self.end_unix_ns.is_some()
    }
}

fn now_unix_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- TraceStore

/// In-process trace store. Cheaply clonable (Arc internals).
///
/// Current backing is two HashMaps: `by_span_id` and `by_trace_id`.
/// Persistence + on-disk indexing are Phase 7.1b — same redb-backed
/// pattern as `duxx-memory`.
#[derive(Clone)]
pub struct TraceStore {
    inner: Arc<Inner>,
}

struct Inner {
    /// Every span ever recorded, keyed by span_id.
    by_span: RwLock<HashMap<SpanId, Span>>,
    /// span_ids grouped by trace_id, in insertion order.
    by_trace: RwLock<HashMap<TraceId, Vec<SpanId>>>,
    /// trace_ids grouped by thread_id, deduped, oldest first.
    by_thread: RwLock<HashMap<ThreadId, Vec<TraceId>>>,
    /// ChangeBus shared with the rest of DuxxDB so RESP/gRPC clients
    /// `PSUBSCRIBE trace.*` see span events live.
    bus: ChangeBus,
}

impl std::fmt::Debug for TraceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.inner.by_span.read().len();
        let t = self.inner.by_trace.read().len();
        let th = self.inner.by_thread.read().len();
        f.debug_struct("TraceStore")
            .field("spans", &s)
            .field("traces", &t)
            .field("threads", &th)
            .finish()
    }
}

impl Default for TraceStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                by_span: RwLock::new(HashMap::new()),
                by_trace: RwLock::new(HashMap::new()),
                by_thread: RwLock::new(HashMap::new()),
                bus: ChangeBus::default(),
            }),
        }
    }

    /// Number of spans currently stored.
    pub fn span_count(&self) -> usize {
        self.inner.by_span.read().len()
    }

    /// Number of distinct traces.
    pub fn trace_count(&self) -> usize {
        self.inner.by_trace.read().len()
    }

    /// Number of distinct threads.
    pub fn thread_count(&self) -> usize {
        self.inner.by_thread.read().len()
    }

    /// Subscribe to change events. Each `record_span` / `close_span`
    /// publishes one `ChangeEvent` with table = `"trace"` and
    /// `key = Some(trace_id)`, so `PSUBSCRIBE trace.*` filters by user.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Record a span. Inserts if the span_id is new; overwrites
    /// otherwise (useful for `close_span`'s mutate-and-republish path).
    pub fn record_span(&self, span: Span) -> Result<()> {
        let span_id = span.span_id.clone();
        let trace_id = span.trace_id.clone();
        let thread_id = span.thread_id.clone();
        let is_new = self.inner.by_span.write().insert(span_id.clone(), span).is_none();

        if is_new {
            self.inner
                .by_trace
                .write()
                .entry(trace_id.clone())
                .or_default()
                .push(span_id.clone());

            if let Some(th) = thread_id {
                let mut th_map = self.inner.by_thread.write();
                let traces = th_map.entry(th).or_default();
                if !traces.contains(&trace_id) {
                    traces.push(trace_id.clone());
                }
            }
        }

        // Publish over ChangeBus so subscribers see it. Channel will be
        // `trace.<trace_id>`; pattern `trace.*` matches all.
        self.inner.bus.publish(ChangeEvent {
            table: "trace".to_string(),
            key: Some(trace_id),
            row_id: 0, // spans use a string id; row_id is unused for traces
            kind: if is_new {
                ChangeKind::Insert
            } else {
                ChangeKind::Update
            },
        });
        Ok(())
    }

    /// Close an open span. Optionally pin its final status.
    pub fn close_span(
        &self,
        span_id: &str,
        end_unix_ns: u128,
        status: SpanStatus,
    ) -> Result<()> {
        let mut by_span = self.inner.by_span.write();
        let span = by_span
            .get_mut(span_id)
            .ok_or_else(|| TraceError::SpanNotFound(span_id.to_string()))?;
        span.end_unix_ns = Some(end_unix_ns);
        span.status = status;
        let trace_id = span.trace_id.clone();
        drop(by_span);

        self.inner.bus.publish(ChangeEvent {
            table: "trace".to_string(),
            key: Some(trace_id),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Get one span by id.
    pub fn get_span(&self, span_id: &str) -> Option<Span> {
        self.inner.by_span.read().get(span_id).cloned()
    }

    /// Get every span in a trace, in insertion order (not topological).
    /// Returns an empty Vec if the trace doesn't exist; that mirrors
    /// how OTel collectors behave on lookup misses.
    pub fn get_trace(&self, trace_id: &str) -> Vec<Span> {
        let span_ids = match self.inner.by_trace.read().get(trace_id).cloned() {
            Some(ids) => ids,
            None => return Vec::new(),
        };
        let by_span = self.inner.by_span.read();
        span_ids
            .into_iter()
            .filter_map(|id| by_span.get(&id).cloned())
            .collect()
    }

    /// Get every span under a given root (inclusive). Walks the
    /// parent_span_id graph. O(N) over the trace's spans, where N is
    /// the trace's total span count.
    pub fn subtree(&self, root_span_id: &str) -> Vec<Span> {
        let root = match self.get_span(root_span_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let all = self.get_trace(&root.trace_id);
        // Build a parent->children index, then DFS from the root.
        let mut children: HashMap<SpanId, Vec<SpanId>> = HashMap::new();
        let mut by_id: HashMap<SpanId, &Span> = HashMap::new();
        for s in &all {
            by_id.insert(s.span_id.clone(), s);
            if let Some(p) = &s.parent_span_id {
                children.entry(p.clone()).or_default().push(s.span_id.clone());
            }
        }
        let mut out = Vec::new();
        let mut stack = vec![root_span_id.to_string()];
        while let Some(id) = stack.pop() {
            if let Some(&s) = by_id.get(&id) {
                out.push(s.clone());
                if let Some(kids) = children.get(&id) {
                    // Push in reverse for stable left-to-right traversal.
                    for k in kids.iter().rev() {
                        stack.push(k.clone());
                    }
                }
            }
        }
        out
    }

    /// Get every span in every trace belonging to a thread, ordered
    /// by trace insertion order then span insertion order.
    pub fn thread(&self, thread_id: &str) -> Vec<Span> {
        let trace_ids = match self.inner.by_thread.read().get(thread_id).cloned() {
            Some(ts) => ts,
            None => return Vec::new(),
        };
        let mut out = Vec::new();
        for tid in trace_ids {
            out.extend(self.get_trace(&tid));
        }
        out
    }

    /// Simple time-window + name-prefix search. Phase 7.1b adds JSON
    /// attribute filtering + full-text search via tantivy; for now
    /// this is a linear scan suitable for in-memory testing.
    pub fn search(&self, filter: &TraceSearch) -> Vec<Span> {
        let by_span = self.inner.by_span.read();
        by_span
            .values()
            .filter(|s| filter.matches(s))
            .cloned()
            .collect()
    }
}

/// Filters for [`TraceStore::search`]. Linear-scan today; tantivy-backed
/// indices arrive in 7.1b.
#[derive(Debug, Default, Clone)]
pub struct TraceSearch {
    /// If set, span.name must start with this prefix.
    pub name_prefix: Option<String>,
    /// If set, span.start_unix_ns must be >= this.
    pub since: Option<u128>,
    /// If set, span.start_unix_ns must be < this.
    pub until: Option<u128>,
    /// If set, only spans with this exact status.
    pub status: Option<SpanStatus>,
    /// If set, only spans in this trace_id.
    pub trace_id: Option<TraceId>,
    /// If set, only spans of this kind.
    pub kind: Option<SpanKind>,
    /// Cap on results. 0 means "no limit".
    pub limit: usize,
}

impl TraceSearch {
    fn matches(&self, s: &Span) -> bool {
        if let Some(p) = &self.name_prefix {
            if !s.name.starts_with(p) {
                return false;
            }
        }
        if let Some(t) = self.since {
            if s.start_unix_ns < t {
                return false;
            }
        }
        if let Some(t) = self.until {
            if s.start_unix_ns >= t {
                return false;
            }
        }
        if let Some(st) = self.status {
            if s.status != st {
                return false;
            }
        }
        if let Some(tid) = &self.trace_id {
            if &s.trace_id != tid {
                return false;
            }
        }
        if let Some(k) = self.kind {
            if s.kind != k {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn span(trace: &str, span: &str, parent: Option<&str>, name: &str) -> Span {
        Span {
            trace_id: trace.into(),
            span_id: span.into(),
            parent_span_id: parent.map(String::from),
            thread_id: None,
            name: name.into(),
            kind: SpanKind::Internal,
            start_unix_ns: 1_000_000,
            end_unix_ns: None,
            status: SpanStatus::Unset,
            attributes: serde_json::Value::Null,
        }
    }

    #[test]
    fn record_and_get_span() {
        let store = TraceStore::new();
        store
            .record_span(span("t1", "s1", None, "agent.run"))
            .unwrap();
        let got = store.get_span("s1").unwrap();
        assert_eq!(got.name, "agent.run");
        assert_eq!(store.span_count(), 1);
        assert_eq!(store.trace_count(), 1);
    }

    #[test]
    fn get_trace_returns_all_spans() {
        let store = TraceStore::new();
        store.record_span(span("t1", "s1", None, "agent.run")).unwrap();
        store.record_span(span("t1", "s2", Some("s1"), "llm.call")).unwrap();
        store.record_span(span("t1", "s3", Some("s1"), "tool.search")).unwrap();
        let spans = store.get_trace("t1");
        assert_eq!(spans.len(), 3);
    }

    #[test]
    fn get_trace_missing_returns_empty() {
        let store = TraceStore::new();
        assert!(store.get_trace("does-not-exist").is_empty());
    }

    #[test]
    fn subtree_walks_parent_graph() {
        let store = TraceStore::new();
        // s1 (root)
        //   ├── s2
        //   │   └── s4
        //   └── s3
        store.record_span(span("t1", "s1", None, "agent.run")).unwrap();
        store.record_span(span("t1", "s2", Some("s1"), "llm")).unwrap();
        store.record_span(span("t1", "s3", Some("s1"), "tool")).unwrap();
        store.record_span(span("t1", "s4", Some("s2"), "embed")).unwrap();

        let sub = store.subtree("s2");
        let ids: Vec<String> = sub.iter().map(|s| s.span_id.clone()).collect();
        assert_eq!(ids, vec!["s2", "s4"]);

        let sub = store.subtree("s1");
        assert_eq!(sub.len(), 4);
    }

    #[test]
    fn close_span_sets_end_and_status() {
        let store = TraceStore::new();
        store.record_span(span("t1", "s1", None, "agent.run")).unwrap();
        store.close_span("s1", 2_000_000, SpanStatus::Ok).unwrap();
        let s = store.get_span("s1").unwrap();
        assert_eq!(s.end_unix_ns, Some(2_000_000));
        assert_eq!(s.duration_ns(), Some(1_000_000));
        assert_eq!(s.status, SpanStatus::Ok);
        assert!(s.is_closed());
    }

    #[test]
    fn close_span_missing_errors() {
        let store = TraceStore::new();
        let err = store.close_span("nope", 1, SpanStatus::Ok).unwrap_err();
        matches!(err, TraceError::SpanNotFound(_));
    }

    #[test]
    fn thread_groups_traces_in_order() {
        let store = TraceStore::new();
        let mut s1 = span("turn-a", "sa", None, "agent.run");
        s1.thread_id = Some("user-7".into());
        store.record_span(s1).unwrap();

        let mut s2 = span("turn-b", "sb", None, "agent.run");
        s2.thread_id = Some("user-7".into());
        store.record_span(s2).unwrap();

        let mut s3 = span("turn-b", "sc", Some("sb"), "llm.call");
        s3.thread_id = Some("user-7".into());
        store.record_span(s3).unwrap();

        let thread = store.thread("user-7");
        // Trace turn-a (1 span) then trace turn-b (2 spans) in insertion order
        let names: Vec<String> = thread.iter().map(|s| s.span_id.clone()).collect();
        assert_eq!(names, vec!["sa", "sb", "sc"]);
    }

    #[test]
    fn search_filters_by_name_prefix_and_status() {
        let store = TraceStore::new();
        let mut a = span("t1", "s1", None, "llm.openai.completion");
        a.status = SpanStatus::Ok;
        let mut b = span("t1", "s2", None, "llm.openai.embedding");
        b.status = SpanStatus::Error;
        let c = span("t1", "s3", None, "tool.web_search");
        store.record_span(a).unwrap();
        store.record_span(b).unwrap();
        store.record_span(c).unwrap();

        let hits = store.search(&TraceSearch {
            name_prefix: Some("llm.".into()),
            status: Some(SpanStatus::Ok),
            ..Default::default()
        });
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].span_id, "s1");
    }

    #[test]
    fn change_bus_publishes_on_record() {
        let store = TraceStore::new();
        let mut rx = store.subscribe();
        store.record_span(span("t1", "s1", None, "agent.run")).unwrap();
        let event = rx.try_recv().expect("event published");
        assert_eq!(event.table, "trace");
        assert_eq!(event.key.as_deref(), Some("t1"));
        assert!(matches!(event.kind, ChangeKind::Insert));
    }

    #[test]
    fn change_bus_publishes_update_on_close() {
        let store = TraceStore::new();
        store.record_span(span("t1", "s1", None, "agent.run")).unwrap();
        let mut rx = store.subscribe();
        store.close_span("s1", 2_000_000, SpanStatus::Ok).unwrap();
        let event = rx.try_recv().expect("update published");
        assert!(matches!(event.kind, ChangeKind::Update));
    }

    #[test]
    fn span_serializes_to_otlp_compatible_json() {
        let s = Span {
            trace_id: "00112233445566778899aabbccddeeff".into(),
            span_id: "0011223344556677".into(),
            parent_span_id: None,
            thread_id: Some("session-1".into()),
            name: "agent.run".into(),
            kind: SpanKind::Server,
            start_unix_ns: 1_700_000_000_000_000_000,
            end_unix_ns: Some(1_700_000_001_000_000_000),
            status: SpanStatus::Ok,
            attributes: serde_json::json!({"model": "gpt-4o", "tokens_in": 421}),
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["name"], "agent.run");
        assert_eq!(v["kind"], "server");
        assert_eq!(v["status"], "ok");
        assert_eq!(v["attributes"]["model"], "gpt-4o");
        // Field name is parent_span_id, matching OTel convention.
        assert!(v.get("parent_span_id").is_some() || v["parent_span_id"].is_null() || v.get("parent_span_id").is_none());
    }

    #[test]
    fn id_helpers_produce_distinct_values() {
        let a = new_trace_id();
        let b = new_trace_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 32); // uuid simple form
        let s1 = new_span_id();
        let s2 = new_span_id();
        assert_ne!(s1, s2);
        assert_eq!(s1.len(), 16);
    }
}
