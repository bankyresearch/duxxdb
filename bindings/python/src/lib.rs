//! # duxxdb-py
//!
//! Python bindings for DuxxDB. Exposes the agent-native primitives
//! (`MemoryStore`, `ToolCache`, `SessionStore`) as Python classes via
//! [PyO3]. Compiled into a stable-API extension module (`abi3-py38`)
//! so a single wheel works across Python 3.8 → 3.13.
//!
//! [PyO3]: https://github.com/PyO3/pyo3

use duxx_cost::{
    Budget as RustBudget, BudgetPeriod as RustBudgetPeriod, BudgetStatus as RustBudgetStatus,
    CostEntry as RustCostEntry, CostFilter as RustCostFilter, CostLedger as RustCostLedger,
    GroupBy as RustGroupBy,
};
use duxx_datasets::{
    Dataset as RustDataset, DatasetRegistry as RustDatasetRegistry, DatasetRow as RustDatasetRow,
};
use duxx_embed::{Embedder, HashEmbedder};
use duxx_eval::{
    EvalRegistry as RustEvalRegistry, EvalRun as RustEvalRun, EvalScore as RustEvalScore,
    EvalStatus as RustEvalStatus, EvalSummary as RustEvalSummary,
};
use duxx_memory::{
    HitKind as RustHitKind, MemoryStore as RustMemoryStore, SessionStore as RustSessionStore,
    ToolCache as RustToolCache,
};
use duxx_prompts::{Prompt as RustPrompt, PromptRegistry as RustPromptRegistry};
use duxx_replay::{
    InvocationKind as RustInvocationKind, ReplayInvocation as RustReplayInvocation,
    ReplayMode as RustReplayMode, ReplayRegistry as RustReplayRegistry,
    ReplayRun as RustReplayRun, ReplaySession as RustReplaySession,
    ReplayStatus as RustReplayStatus,
};
use duxx_storage::{open_backend, Backend, MemoryBackend};
use duxx_trace::{
    Span as RustSpan, SpanKind as RustSpanKind, SpanStatus as RustSpanStatus,
    TraceStore as RustTraceStore,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Long-term semantic memory store with hybrid recall.
///
/// >>> store = duxxdb.MemoryStore(dim=4)
/// >>> store.remember(key="alice", text="hi", embedding=[0.0]*4)
/// >>> hits = store.recall(key="alice", query="hi", embedding=[0.0]*4, k=5)
#[pyclass(name = "MemoryStore", module = "duxxdb._native")]
struct MemoryStore {
    inner: RustMemoryStore,
}

#[pymethods]
impl MemoryStore {
    #[new]
    #[pyo3(signature = (dim, capacity = 100_000))]
    fn new(dim: usize, capacity: usize) -> Self {
        Self {
            inner: RustMemoryStore::with_capacity(dim, capacity),
        }
    }

    /// Open a fully-persistent store rooted at `dir`. Rows + tantivy
    /// BM25 index + HNSW dump all live under the directory, so a
    /// graceful close + reopen skips the index rebuild (sub-second cold
    /// start on the order of 100k memories).
    ///
    /// Hard kills fall back to a row-rebuild path that takes time
    /// proportional to the corpus size. New in duxxdb v0.1.1.
    ///
    /// >>> store = duxxdb.MemoryStore.open_at(dim=4, capacity=100_000, dir="./data/duxx")
    #[staticmethod]
    #[pyo3(signature = (dim, capacity, dir))]
    fn open_at(dim: usize, capacity: usize, dir: &str) -> PyResult<Self> {
        let inner = RustMemoryStore::open_at(dim, capacity, dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn remember(&self, key: &str, text: &str, embedding: Vec<f32>) -> PyResult<u64> {
        if embedding.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "embedding has dim {}, store expects {}",
                embedding.len(),
                self.inner.dim()
            )));
        }
        self.inner
            .remember(key.to_string(), text.to_string(), embedding)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (key, query, embedding, k = 10))]
    fn recall(
        &self,
        key: &str,
        query: &str,
        embedding: Vec<f32>,
        k: usize,
    ) -> PyResult<Vec<MemoryHit>> {
        if embedding.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "embedding has dim {}, store expects {}",
                embedding.len(),
                self.inner.dim()
            )));
        }
        let hits = self
            .inner
            .recall(key, query, &embedding, k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(hits
            .into_iter()
            .map(|h| MemoryHit {
                id: h.memory.id,
                key: h.memory.key,
                text: h.memory.text,
                score: h.score,
            })
            .collect())
    }

    /// Forget one row by id. Returns True if the id existed.
    /// New in duxxdb v0.1.1.
    fn forget(&self, id: u64) -> bool {
        self.inner.forget(id)
    }

    /// Configure a soft maximum number of rows. When the cap is
    /// exceeded, every subsequent `remember` evicts the row with the
    /// lowest decayed importance until the count is back at the cap.
    /// Pass `None` to disable the cap.
    /// New in duxxdb v0.1.1.
    #[pyo3(signature = (cap = None))]
    fn set_max_rows(&self, cap: Option<usize>) {
        self.inner.set_max_rows(cap);
    }

    /// Currently configured row cap, or None if unlimited.
    /// New in duxxdb v0.1.1.
    fn max_rows(&self) -> Option<usize> {
        self.inner.max_rows()
    }

    /// Set the half-life (in seconds) used by the cap eviction policy
    /// when weighting effective importance. Default 24h.
    /// New in duxxdb v0.1.1.
    fn set_eviction_half_life(&self, seconds: f64) {
        let nanos = (seconds * 1_000_000_000f64).max(0.0) as u64;
        self.inner
            .set_eviction_half_life(Duration::from_nanos(nanos));
    }

    /// Total number of rows evicted by the cap since process start.
    /// New in duxxdb v0.1.1.
    fn evictions_total(&self) -> u64 {
        self.inner.evictions_total()
    }

    /// Whether this store is backed by durable on-disk storage.
    /// New in duxxdb v0.1.1.
    fn is_persistent(&self) -> bool {
        self.inner.is_persistent()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<MemoryStore dim={} memories={} persistent={}>",
            self.inner.dim(),
            self.inner.len(),
            self.inner.is_persistent()
        )
    }
}

/// One recall result.
#[pyclass(name = "MemoryHit", module = "duxxdb._native")]
#[derive(Clone)]
struct MemoryHit {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    key: String,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    score: f32,
}

#[pymethods]
impl MemoryHit {
    fn __repr__(&self) -> String {
        format!(
            "<MemoryHit id={} score={:.4} text={:?}>",
            self.id, self.score, self.text
        )
    }
}

// ---------------------------------------------------------------------------
// ToolCache
// ---------------------------------------------------------------------------

/// Two-stage tool-result cache: exact hash + semantic-near-hit.
#[pyclass(name = "ToolCache", module = "duxxdb._native")]
struct ToolCache {
    inner: RustToolCache,
}

#[pymethods]
impl ToolCache {
    #[new]
    #[pyo3(signature = (threshold = 0.95))]
    fn new(threshold: f32) -> Self {
        Self {
            inner: RustToolCache::with_threshold(threshold),
        }
    }

    #[pyo3(signature = (tool, args_hash, args_embedding, result, ttl_secs = 3600))]
    fn put(
        &self,
        tool: &str,
        args_hash: u64,
        args_embedding: Vec<f32>,
        result: Vec<u8>,
        ttl_secs: u64,
    ) -> PyResult<()> {
        self.inner
            .put(
                tool.to_string(),
                args_hash,
                args_embedding,
                result,
                Duration::from_secs(ttl_secs),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get(
        &self,
        tool: &str,
        args_hash: u64,
        args_embedding: Vec<f32>,
    ) -> Option<ToolCacheHit> {
        self.inner
            .get(tool, args_hash, &args_embedding)
            .map(|hit| ToolCacheHit {
                kind: match hit.kind {
                    RustHitKind::Exact => "exact".to_string(),
                    RustHitKind::SemanticNearHit => "semantic_near_hit".to_string(),
                },
                similarity: hit.similarity,
                result: hit.result,
            })
    }

    fn purge_expired(&self) -> usize {
        self.inner.purge_expired()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("<ToolCache entries={}>", self.inner.len())
    }
}

/// Cache hit returned by `ToolCache.get`.
#[pyclass(name = "ToolCacheHit", module = "duxxdb._native")]
#[derive(Clone)]
struct ToolCacheHit {
    /// Either `"exact"` or `"semantic_near_hit"`.
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    similarity: f32,
    #[pyo3(get)]
    result: Vec<u8>,
}

#[pymethods]
impl ToolCacheHit {
    fn __repr__(&self) -> String {
        format!(
            "<ToolCacheHit kind={} similarity={:.4} result_len={}>",
            self.kind,
            self.similarity,
            self.result.len()
        )
    }
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

/// Sliding-TTL key/value store for per-conversation working state.
#[pyclass(name = "SessionStore", module = "duxxdb._native")]
struct SessionStore {
    inner: RustSessionStore,
}

#[pymethods]
impl SessionStore {
    #[new]
    #[pyo3(signature = (ttl_secs = 1800))]
    fn new(ttl_secs: u64) -> Self {
        Self {
            inner: RustSessionStore::with_ttl(Duration::from_secs(ttl_secs)),
        }
    }

    fn put(&self, session_id: &str, data: Vec<u8>) {
        self.inner.put(session_id.to_string(), data);
    }

    fn get(&self, session_id: &str) -> Option<Vec<u8>> {
        self.inner.get(session_id)
    }

    fn delete(&self, session_id: &str) -> bool {
        self.inner.delete(session_id)
    }

    fn purge_expired(&self) -> usize {
        self.inner.purge_expired()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("<SessionStore sessions={}>", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// PromptRegistry (Phase 7.2, v0.2.0)
// ---------------------------------------------------------------------------

/// One persisted prompt. Returned by every `PromptRegistry` lookup.
#[pyclass(name = "Prompt", module = "duxxdb._native")]
#[derive(Clone)]
struct Prompt {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: u64,
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    tags: Vec<String>,
    /// Free-form metadata. Stored as JSON in Rust; surfaced here as
    /// a Python `dict` / `list` / scalar via `json.loads`-style
    /// conversion.
    metadata_json: String,
    #[pyo3(get)]
    created_at_unix_ns: u128,
}

#[pymethods]
impl Prompt {
    /// Decoded `metadata`. Returned as the native Python value
    /// (`dict`, `list`, `str`, …) that `json.loads(...)` would give
    /// you. Always evaluated lazily on attribute access.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Prompt name={:?} version={} content_len={} tags={:?}>",
            self.name,
            self.version,
            self.content.len(),
            self.tags,
        )
    }
}

impl Prompt {
    fn from_rust(p: RustPrompt) -> Self {
        let metadata_json = serde_json::to_string(&p.metadata).unwrap_or_else(|_| "null".into());
        Self {
            name: p.name,
            version: p.version,
            content: p.content,
            tags: p.tags,
            metadata_json,
            created_at_unix_ns: p.created_at_unix_ns,
        }
    }
}

/// One hit from `PromptRegistry.search`. Wraps a `Prompt` with the
/// cosine similarity score in `[0, 1]`.
#[pyclass(name = "PromptHit", module = "duxxdb._native")]
#[derive(Clone)]
struct PromptHit {
    #[pyo3(get)]
    prompt: Prompt,
    #[pyo3(get)]
    score: f32,
}

#[pymethods]
impl PromptHit {
    fn __repr__(&self) -> String {
        format!(
            "<PromptHit name={:?} version={} score={:.4}>",
            self.prompt.name, self.prompt.version, self.score
        )
    }
}

/// Versioned prompt registry with semantic search across the
/// catalog.
///
/// >>> r = duxxdb.PromptRegistry(dim=16)
/// >>> v1 = r.put("classifier", "You are a refund agent.")
/// >>> v2 = r.put("classifier", "You are a friendly refund agent.")
/// >>> r.tag("classifier", v2, "prod")
/// >>> r.get("classifier", "prod").content
/// 'You are a friendly refund agent.'
///
/// Pass ``storage=`` for persistence:
///
/// >>> r = duxxdb.PromptRegistry(dim=16, storage="redb:./prompts.redb")
#[pyclass(name = "PromptRegistry", module = "duxxdb._native")]
struct PromptRegistry {
    inner: RustPromptRegistry,
    dim: usize,
}

#[pymethods]
impl PromptRegistry {
    /// Build a new registry.
    ///
    /// * ``dim`` — embedding dimensionality. Must match the
    ///   embedder you've configured (defaults to the built-in
    ///   ``HashEmbedder`` for now; explicit embedder support lands
    ///   alongside the other Phase 7 primitives).
    /// * ``storage`` — optional persistence selector:
    ///   * ``None`` or ``"memory"`` → in-memory (default; matches
    ///     v0.1.x behavior).
    ///   * ``"redb:/path/to/file.redb"`` → durable, ACID.
    #[new]
    #[pyo3(signature = (dim = 32, storage = None))]
    fn new(dim: usize, storage: Option<&str>) -> PyResult<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(dim));
        let backend: Arc<dyn Backend> = match storage {
            None | Some("") | Some("memory") => Arc::new(MemoryBackend::new()),
            Some(spec) => open_backend(Some(spec))
                .map(Arc::from)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        let inner = RustPromptRegistry::open(embedder, backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner, dim })
    }

    /// Insert a new version of ``name``. Returns the assigned
    /// monotonic version number (starting at 1).
    #[pyo3(signature = (name, content, metadata = None))]
    fn put(
        &self,
        py: Python<'_>,
        name: &str,
        content: &str,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let metadata_value = py_to_json(py, metadata)?;
        self.inner
            .put(name.to_string(), content.to_string(), metadata_value)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Look up a prompt.
    ///
    /// ``version_or_tag`` accepts:
    ///   * ``None`` → latest version
    ///   * an ``int`` → that exact version
    ///   * a ``str`` → resolved as a tag
    #[pyo3(signature = (name, version_or_tag = None))]
    fn get(&self, name: &str, version_or_tag: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Prompt>> {
        let prompt = match version_or_tag {
            None => self.inner.get_latest(name),
            Some(obj) => {
                if let Ok(v) = obj.extract::<u64>() {
                    self.inner.get(name, v)
                } else {
                    let tag: String = obj.extract().map_err(|_| {
                        PyValueError::new_err(
                            "version_or_tag must be an int (version) or str (tag)",
                        )
                    })?;
                    self.inner.get_by_tag(name, &tag)
                }
            }
        };
        Ok(prompt.map(Prompt::from_rust))
    }

    /// Every version of ``name``, ascending.
    fn list(&self, name: &str) -> Vec<Prompt> {
        self.inner.list(name).into_iter().map(Prompt::from_rust).collect()
    }

    /// Every known prompt name, lex order.
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    /// Point ``tag`` at ``version`` of ``name``. If the tag exists
    /// it is moved.
    fn tag(&self, name: &str, version: u64, tag: &str) -> PyResult<()> {
        self.inner
            .tag(name, version, tag)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove a tag. Returns True if the tag existed.
    fn untag(&self, name: &str, tag: &str) -> bool {
        self.inner.untag(name, tag)
    }

    /// Hard-delete one version of a prompt. Returns True if the
    /// version existed. The version number is never reused.
    fn delete(&self, name: &str, version: u64) -> bool {
        self.inner.delete(name, version)
    }

    /// Semantic search across the catalog.
    #[pyo3(signature = (query, k = 10))]
    fn search(&self, query: &str, k: usize) -> PyResult<Vec<PromptHit>> {
        let hits = self
            .inner
            .search(query, k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(hits
            .into_iter()
            .map(|h| PromptHit {
                prompt: Prompt::from_rust(h.prompt),
                score: h.score,
            })
            .collect())
    }

    /// Line diff between two versions.
    fn diff(&self, name: &str, version_a: u64, version_b: u64) -> PyResult<String> {
        self.inner
            .diff(name, version_a, version_b)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Embedding dimensionality this registry was built with.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "<PromptRegistry names={} versions={} tags={} dim={}>",
            stats.names, stats.versions, stats.tags, self.dim
        )
    }
}

/// Convert a Python value (dict/list/str/number/bool/None) into a
/// `serde_json::Value` by way of `json.dumps`. PyO3 doesn't offer
/// a direct converter; routing through `json` keeps the encoding
/// exactly the same as what the user expects.
fn py_to_json(py: Python<'_>, obj: Option<&Bound<'_, PyAny>>) -> PyResult<serde_json::Value> {
    let Some(obj) = obj else {
        return Ok(serde_json::Value::Null);
    };
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    let json_mod = py.import_bound("json")?;
    let dumps = json_mod.getattr("dumps")?;
    let s: String = dumps.call1((obj,))?.extract()?;
    serde_json::from_str(&s)
        .map_err(|e| PyValueError::new_err(format!("metadata JSON encode: {e}")))
}

/// Round-trip the other direction. We store the metadata JSON as
/// a string in [`Prompt`] and only inflate it when the user
/// touches the attribute.
fn json_str_to_py(py: Python<'_>, json: &str) -> PyResult<PyObject> {
    let json_mod = py.import_bound("json")?;
    let loads = json_mod.getattr("loads")?;
    Ok(loads.call1((json,))?.unbind())
}

// ---------------------------------------------------------------------------
// CostLedger (Phase 7.6, v0.2.1)
// ---------------------------------------------------------------------------

/// One recorded LLM/tool cost row. Returned by every ledger lookup.
#[pyclass(name = "CostEntry", module = "duxxdb._native")]
#[derive(Clone)]
struct CostEntryPy {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    tenant: String,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    tokens_in: u64,
    #[pyo3(get)]
    tokens_out: u64,
    #[pyo3(get)]
    cost_usd: f64,
    #[pyo3(get)]
    trace_id: Option<String>,
    #[pyo3(get)]
    run_id: Option<String>,
    #[pyo3(get)]
    prompt_name: Option<String>,
    #[pyo3(get)]
    prompt_version: Option<u64>,
    #[pyo3(get)]
    input_text: String,
    metadata_json: String,
    #[pyo3(get)]
    recorded_at_unix_ns: u128,
}

#[pymethods]
impl CostEntryPy {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }

    fn __repr__(&self) -> String {
        format!(
            "<CostEntry tenant={:?} model={:?} cost_usd={:.6}>",
            self.tenant, self.model, self.cost_usd,
        )
    }
}

impl CostEntryPy {
    fn from_rust(e: RustCostEntry) -> Self {
        let metadata_json =
            serde_json::to_string(&e.metadata).unwrap_or_else(|_| "null".into());
        Self {
            id: e.id,
            tenant: e.tenant,
            model: e.model,
            tokens_in: e.tokens_in,
            tokens_out: e.tokens_out,
            cost_usd: e.cost_usd,
            trace_id: e.trace_id,
            run_id: e.run_id,
            prompt_name: e.prompt_name,
            prompt_version: e.prompt_version,
            input_text: e.input_text,
            metadata_json,
            recorded_at_unix_ns: e.recorded_at_unix_ns,
        }
    }
}

/// One per-tenant spending cap.
#[pyclass(name = "Budget", module = "duxxdb._native")]
#[derive(Clone)]
struct BudgetPy {
    #[pyo3(get)]
    tenant: String,
    /// One of: ``"daily"``, ``"weekly"``, ``"monthly"``, or
    /// ``"custom:<seconds>"`` for arbitrary windows.
    #[pyo3(get)]
    period: String,
    #[pyo3(get)]
    amount_usd: f64,
    #[pyo3(get)]
    warn_pct: f32,
    metadata_json: String,
    #[pyo3(get)]
    created_at_unix_ns: u128,
}

#[pymethods]
impl BudgetPy {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Budget tenant={:?} period={} amount_usd={}>",
            self.tenant, self.period, self.amount_usd,
        )
    }
}

impl BudgetPy {
    fn from_rust(b: RustBudget) -> Self {
        let metadata_json =
            serde_json::to_string(&b.metadata).unwrap_or_else(|_| "null".into());
        Self {
            tenant: b.tenant,
            period: budget_period_to_string(b.period),
            amount_usd: b.amount_usd,
            warn_pct: b.warn_pct,
            metadata_json,
            created_at_unix_ns: b.created_at_unix_ns,
        }
    }
}

fn budget_period_from_str(s: &str) -> PyResult<RustBudgetPeriod> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "daily" => Ok(RustBudgetPeriod::Daily),
        "weekly" => Ok(RustBudgetPeriod::Weekly),
        "monthly" => Ok(RustBudgetPeriod::Monthly),
        _ => {
            // accept "custom:<secs>" or a bare integer "<secs>"
            let rest = s.strip_prefix("custom:").unwrap_or(&s);
            rest.parse::<u64>()
                .map(|secs| RustBudgetPeriod::Custom { secs })
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "unknown period {s:?} (use: daily | weekly | monthly | custom:<secs> | <secs>)"
                    ))
                })
        }
    }
}

fn budget_period_to_string(p: RustBudgetPeriod) -> String {
    match p {
        RustBudgetPeriod::Daily => "daily".into(),
        RustBudgetPeriod::Weekly => "weekly".into(),
        RustBudgetPeriod::Monthly => "monthly".into(),
        RustBudgetPeriod::Custom { secs } => format!("custom:{secs}"),
    }
}

fn group_by_from_str(s: &str) -> PyResult<RustGroupBy> {
    match s.trim().to_ascii_lowercase().as_str() {
        "tenant" => Ok(RustGroupBy::Tenant),
        "model" => Ok(RustGroupBy::Model),
        "prompt" => Ok(RustGroupBy::Prompt),
        "day" | "day_utc" => Ok(RustGroupBy::DayUtc),
        "none" | "all" => Ok(RustGroupBy::None),
        other => Err(PyValueError::new_err(format!(
            "unknown group_by {other:?} (use: tenant | model | prompt | day | none)"
        ))),
    }
}

fn budget_status_to_string(s: RustBudgetStatus) -> &'static str {
    match s {
        RustBudgetStatus::NoBudget => "no_budget",
        RustBudgetStatus::Ok => "ok",
        RustBudgetStatus::Warning => "warning",
        RustBudgetStatus::Exceeded => "exceeded",
    }
}

fn cost_filter_from_kwargs(
    tenant: Option<String>,
    model: Option<String>,
    prompt_name: Option<String>,
    since_unix_ns: Option<u128>,
    until_unix_ns: Option<u128>,
    limit: usize,
) -> RustCostFilter {
    RustCostFilter {
        tenant,
        model,
        prompt_name,
        since_unix_ns,
        until_unix_ns,
        limit,
    }
}

/// Append-only token+cost ledger with per-tenant budgets.
///
/// >>> ledger = duxxdb.CostLedger(dim=16)
/// >>> eid = ledger.record(tenant="acme", model="gpt-4o-mini",
/// ...                     tokens_in=100, tokens_out=50, cost_usd=0.0023)
/// >>> ledger.total("acme")
/// 0.0023
///
/// Pass ``storage=`` for durability:
///
/// >>> ledger = duxxdb.CostLedger(dim=16, storage="redb:./cost.redb")
#[pyclass(name = "CostLedger", module = "duxxdb._native")]
struct CostLedger {
    inner: RustCostLedger,
    dim: usize,
}

#[pymethods]
impl CostLedger {
    #[new]
    #[pyo3(signature = (dim = 32, storage = None))]
    fn new(dim: usize, storage: Option<&str>) -> PyResult<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(dim));
        let backend = open_backend_arc(storage)?;
        let inner = RustCostLedger::open(embedder, backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner, dim })
    }

    /// Append one cost row. Returns the assigned UUID-form id.
    #[pyo3(signature = (
        tenant, model,
        tokens_in,
        tokens_out,
        cost_usd,
        trace_id = None,
        run_id = None,
        prompt_name = None,
        prompt_version = None,
        input_text = String::new(),
        metadata = None,
    ))]
    fn record(
        &self,
        py: Python<'_>,
        tenant: &str,
        model: &str,
        tokens_in: u64,
        tokens_out: u64,
        cost_usd: f64,
        trace_id: Option<String>,
        run_id: Option<String>,
        prompt_name: Option<String>,
        prompt_version: Option<u64>,
        input_text: String,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let metadata_v = py_to_json(py, metadata)?;
        let entry = RustCostEntry {
            id: String::new(),
            tenant: tenant.into(),
            model: model.into(),
            tokens_in,
            tokens_out,
            cost_usd,
            trace_id,
            run_id,
            prompt_name,
            prompt_version,
            input_text,
            metadata: metadata_v,
            recorded_at_unix_ns: 0,
        };
        self.inner
            .record(entry)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (
        tenant = None,
        model = None,
        prompt_name = None,
        since_unix_ns = None,
        until_unix_ns = None,
        limit = 0,
    ))]
    fn query(
        &self,
        tenant: Option<String>,
        model: Option<String>,
        prompt_name: Option<String>,
        since_unix_ns: Option<u128>,
        until_unix_ns: Option<u128>,
        limit: usize,
    ) -> Vec<CostEntryPy> {
        let f = cost_filter_from_kwargs(tenant, model, prompt_name, since_unix_ns, until_unix_ns, limit);
        self.inner.query(&f).into_iter().map(CostEntryPy::from_rust).collect()
    }

    /// Total spend for ``tenant`` in an optional time window.
    #[pyo3(signature = (tenant, since_unix_ns = None, until_unix_ns = None))]
    fn total(&self, tenant: &str, since_unix_ns: Option<u128>, until_unix_ns: Option<u128>) -> f64 {
        self.inner.total_for(tenant, since_unix_ns, until_unix_ns)
    }

    /// Aggregate by ``tenant`` / ``model`` / ``prompt`` / ``day`` / ``none``.
    /// Returns a list of ``(key, count, tokens_in, tokens_out, total_usd, mean_usd)`` tuples.
    #[pyo3(signature = (
        group_by,
        tenant = None,
        model = None,
        prompt_name = None,
        since_unix_ns = None,
        until_unix_ns = None,
    ))]
    fn aggregate(
        &self,
        group_by: &str,
        tenant: Option<String>,
        model: Option<String>,
        prompt_name: Option<String>,
        since_unix_ns: Option<u128>,
        until_unix_ns: Option<u128>,
    ) -> PyResult<Vec<(String, u64, u64, u64, f64, f64)>> {
        let gb = group_by_from_str(group_by)?;
        let f = cost_filter_from_kwargs(tenant, model, prompt_name, since_unix_ns, until_unix_ns, 0);
        Ok(self
            .inner
            .aggregate(&f, gb)
            .into_iter()
            .map(|b| {
                (
                    b.key,
                    b.count,
                    b.tokens_in,
                    b.tokens_out,
                    b.total_usd,
                    b.mean_usd,
                )
            })
            .collect())
    }

    /// Set or replace a budget for ``tenant``.
    ///
    /// ``period`` accepts ``"daily"`` / ``"weekly"`` / ``"monthly"`` /
    /// ``"custom:<seconds>"`` / a bare integer of seconds.
    #[pyo3(signature = (tenant, period, amount_usd, warn_pct = 0.8, metadata = None))]
    fn set_budget(
        &self,
        py: Python<'_>,
        tenant: &str,
        period: &str,
        amount_usd: f64,
        warn_pct: f32,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let p = budget_period_from_str(period)?;
        let m = py_to_json(py, metadata)?;
        self.inner
            .set_budget(tenant.to_string(), p, amount_usd, warn_pct, m)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get_budget(&self, tenant: &str) -> Option<BudgetPy> {
        self.inner.get_budget(tenant).map(BudgetPy::from_rust)
    }

    fn delete_budget(&self, tenant: &str) -> bool {
        self.inner.delete_budget(tenant)
    }

    /// One of ``"no_budget"`` / ``"ok"`` / ``"warning"`` / ``"exceeded"``.
    fn status(&self, tenant: &str) -> &'static str {
        budget_status_to_string(self.inner.budget_status(tenant))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "<CostLedger entries={} budgets={} total_usd={:.4}>",
            s.entries, s.tenants_with_budget, s.total_usd,
        )
    }
}

// ---------------------------------------------------------------------------
// DatasetRegistry (Phase 7.3, v0.2.1)
// ---------------------------------------------------------------------------

#[pyclass(name = "DatasetRow", module = "duxxdb._native")]
#[derive(Clone)]
struct DatasetRowPy {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    text: String,
    data_json: String,
    #[pyo3(get)]
    split: String,
    annotations_json: String,
}

#[pymethods]
impl DatasetRowPy {
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.data_json)
    }
    #[getter]
    fn annotations(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.annotations_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<DatasetRow id={:?} split={:?} text_len={}>",
            self.id,
            self.split,
            self.text.len()
        )
    }
}

impl DatasetRowPy {
    fn from_rust(r: RustDatasetRow) -> Self {
        Self {
            id: r.id,
            text: r.text,
            data_json: serde_json::to_string(&r.data).unwrap_or_else(|_| "null".into()),
            split: r.split,
            annotations_json: serde_json::to_string(&r.annotations).unwrap_or_else(|_| "null".into()),
        }
    }
}

#[pyclass(name = "Dataset", module = "duxxdb._native")]
#[derive(Clone)]
struct DatasetPy {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: u64,
    schema_json: String,
    #[pyo3(get)]
    rows: Vec<DatasetRowPy>,
    #[pyo3(get)]
    tags: Vec<String>,
    metadata_json: String,
    #[pyo3(get)]
    created_at_unix_ns: u128,
}

#[pymethods]
impl DatasetPy {
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.schema_json)
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<Dataset name={:?} version={} rows={} tags={:?}>",
            self.name,
            self.version,
            self.rows.len(),
            self.tags
        )
    }
}

impl DatasetPy {
    fn from_rust(d: RustDataset) -> Self {
        Self {
            name: d.name,
            version: d.version,
            schema_json: serde_json::to_string(&d.schema).unwrap_or_else(|_| "null".into()),
            rows: d.rows.into_iter().map(DatasetRowPy::from_rust).collect(),
            tags: d.tags,
            metadata_json: serde_json::to_string(&d.metadata).unwrap_or_else(|_| "null".into()),
            created_at_unix_ns: d.created_at_unix_ns,
        }
    }
}

/// Versioned eval datasets with semantic row search.
#[pyclass(name = "DatasetRegistry", module = "duxxdb._native")]
struct DatasetRegistry {
    inner: RustDatasetRegistry,
    dim: usize,
}

#[pymethods]
impl DatasetRegistry {
    #[new]
    #[pyo3(signature = (dim = 32, storage = None))]
    fn new(dim: usize, storage: Option<&str>) -> PyResult<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(dim));
        let backend = open_backend_arc(storage)?;
        let inner = RustDatasetRegistry::open(embedder, backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner, dim })
    }

    /// Register a dataset name with an optional schema hint.
    #[pyo3(signature = (name, schema = None))]
    fn create(&self, py: Python<'_>, name: &str, schema: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let sv = py_to_json(py, schema)?;
        self.inner
            .create(name.to_string(), sv)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Append a new immutable version with the supplied rows.
    ///
    /// Each row is a dict with optional ``id`` / ``data`` / ``annotations``
    /// keys; required: ``text``. ``split`` defaults to ``""``.
    #[pyo3(signature = (name, rows, metadata = None))]
    fn add(
        &self,
        py: Python<'_>,
        name: &str,
        rows: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let rows_val = py_to_json(py, Some(rows))?;
        let raw_rows = match rows_val {
            serde_json::Value::Array(a) => a,
            other => {
                return Err(PyValueError::new_err(format!(
                    "rows must be a list, got {other:?}"
                )));
            }
        };
        let mut parsed = Vec::with_capacity(raw_rows.len());
        for v in raw_rows {
            let row: RustDatasetRow = match v {
                serde_json::Value::String(s) => RustDatasetRow::new(s),
                serde_json::Value::Object(_) => serde_json::from_value(v)
                    .map_err(|e| PyValueError::new_err(format!("row decode: {e}")))?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "row must be a dict or string, got {other:?}"
                    )));
                }
            };
            // Auto-assign an id when the user omitted one.
            let row = if row.id.is_empty() {
                RustDatasetRow {
                    id: uuid::Uuid::new_v4().simple().to_string(),
                    ..row
                }
            } else {
                row
            };
            parsed.push(row);
        }
        let mv = py_to_json(py, metadata)?;
        self.inner
            .add(name.to_string(), parsed, mv)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, version_or_tag = None))]
    fn get(
        &self,
        name: &str,
        version_or_tag: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<DatasetPy>> {
        let ds = match version_or_tag {
            None => self.inner.get_latest(name),
            Some(obj) => {
                if let Ok(v) = obj.extract::<u64>() {
                    self.inner.get(name, v)
                } else {
                    let tag: String = obj.extract().map_err(|_| {
                        PyValueError::new_err(
                            "version_or_tag must be an int (version) or str (tag)",
                        )
                    })?;
                    self.inner.get_by_tag(name, &tag)
                }
            }
        };
        Ok(ds.map(DatasetPy::from_rust))
    }

    fn list(&self, name: &str) -> Vec<DatasetPy> {
        self.inner.list(name).into_iter().map(DatasetPy::from_rust).collect()
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn tag(&self, name: &str, version: u64, tag: &str) -> PyResult<()> {
        self.inner
            .tag(name, version, tag)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn untag(&self, name: &str, tag: &str) -> bool {
        self.inner.untag(name, tag)
    }

    fn delete(&self, name: &str, version: u64) -> bool {
        self.inner.delete(name, version)
    }

    /// Sample up to ``n`` rows from a specific version.
    #[pyo3(signature = (name, version, n, split = None))]
    fn sample(&self, name: &str, version: u64, n: usize, split: Option<String>) -> Vec<DatasetRowPy> {
        self.inner
            .sample(name, version, n, split.as_deref())
            .into_iter()
            .map(DatasetRowPy::from_rust)
            .collect()
    }

    #[pyo3(signature = (name, version, split = None))]
    fn size(&self, name: &str, version: u64, split: Option<String>) -> usize {
        self.inner.size(name, version, split.as_deref())
    }

    fn splits(&self, name: &str, version: u64) -> Vec<String> {
        self.inner.splits(name, version)
    }

    /// Semantic search across the catalog. Returns ``(dataset, version,
    /// row_id, score, text)`` tuples.
    #[pyo3(signature = (query, k = 10, name_filter = None))]
    fn search(&self, query: &str, k: usize, name_filter: Option<String>) -> PyResult<Vec<(String, u64, String, f32, String)>> {
        let hits = self
            .inner
            .search(query, k, name_filter.as_deref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(hits
            .into_iter()
            .map(|h| (h.dataset, h.version, h.row.id, h.score, h.row.text))
            .collect())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "<DatasetRegistry names={} versions={} rows={} tags={}>",
            s.names, s.versions, s.rows, s.tags,
        )
    }
}

// ---------------------------------------------------------------------------
// EvalRegistry (Phase 7.4, v0.2.1)
// ---------------------------------------------------------------------------

#[pyclass(name = "EvalSummary", module = "duxxdb._native")]
#[derive(Clone)]
struct EvalSummaryPy {
    #[pyo3(get)]
    total_scored: usize,
    #[pyo3(get)]
    mean: f32,
    #[pyo3(get)]
    p50: f32,
    #[pyo3(get)]
    p90: f32,
    #[pyo3(get)]
    p99: f32,
    #[pyo3(get)]
    min: f32,
    #[pyo3(get)]
    max: f32,
    #[pyo3(get)]
    pass_rate_50: f32,
}

#[pymethods]
impl EvalSummaryPy {
    fn __repr__(&self) -> String {
        format!(
            "<EvalSummary n={} mean={:.3} p99={:.3} pass_rate_50={:.3}>",
            self.total_scored, self.mean, self.p99, self.pass_rate_50,
        )
    }
}

impl EvalSummaryPy {
    fn from_rust(s: RustEvalSummary) -> Self {
        Self {
            total_scored: s.total_scored,
            mean: s.mean,
            p50: s.p50,
            p90: s.p90,
            p99: s.p99,
            min: s.min,
            max: s.max,
            pass_rate_50: s.pass_rate_50,
        }
    }
}

#[pyclass(name = "EvalRun", module = "duxxdb._native")]
#[derive(Clone)]
struct EvalRunPy {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    dataset_name: String,
    #[pyo3(get)]
    dataset_version: u64,
    #[pyo3(get)]
    prompt_name: Option<String>,
    #[pyo3(get)]
    prompt_version: Option<u64>,
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    scorer: String,
    /// One of ``"pending"`` / ``"running"`` / ``"completed"`` / ``"failed"``.
    #[pyo3(get)]
    status: String,
    metadata_json: String,
    #[pyo3(get)]
    created_at_unix_ns: u128,
    #[pyo3(get)]
    completed_at_unix_ns: Option<u128>,
    #[pyo3(get)]
    summary: Option<EvalSummaryPy>,
}

#[pymethods]
impl EvalRunPy {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<EvalRun id={:?} dataset={} model={} status={}>",
            self.id, self.dataset_name, self.model, self.status,
        )
    }
}

fn eval_status_to_string(s: RustEvalStatus) -> String {
    match s {
        RustEvalStatus::Pending => "pending",
        RustEvalStatus::Running => "running",
        RustEvalStatus::Completed => "completed",
        RustEvalStatus::Failed => "failed",
    }
    .into()
}

impl EvalRunPy {
    fn from_rust(r: RustEvalRun) -> Self {
        Self {
            id: r.id,
            name: r.name,
            dataset_name: r.dataset_name,
            dataset_version: r.dataset_version,
            prompt_name: r.prompt_name,
            prompt_version: r.prompt_version,
            model: r.model,
            scorer: r.scorer,
            status: eval_status_to_string(r.status),
            metadata_json: serde_json::to_string(&r.metadata).unwrap_or_else(|_| "null".into()),
            created_at_unix_ns: r.created_at_unix_ns,
            completed_at_unix_ns: r.completed_at_unix_ns,
            summary: r.summary.map(EvalSummaryPy::from_rust),
        }
    }
}

#[pyclass(name = "EvalScore", module = "duxxdb._native")]
#[derive(Clone)]
struct EvalScorePy {
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    row_id: String,
    #[pyo3(get)]
    score: f32,
    notes_json: String,
    #[pyo3(get)]
    output_text: String,
    #[pyo3(get)]
    recorded_at_unix_ns: u128,
}

#[pymethods]
impl EvalScorePy {
    #[getter]
    fn notes(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.notes_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<EvalScore run={:?} row={:?} score={:.3}>",
            self.run_id, self.row_id, self.score,
        )
    }
}

impl EvalScorePy {
    fn from_rust(s: RustEvalScore) -> Self {
        Self {
            run_id: s.run_id,
            row_id: s.row_id,
            score: s.score,
            notes_json: serde_json::to_string(&s.notes).unwrap_or_else(|_| "null".into()),
            output_text: s.output_text,
            recorded_at_unix_ns: s.recorded_at_unix_ns,
        }
    }
}

/// Eval runs with regressions + semantic failure clustering.
#[pyclass(name = "EvalRegistry", module = "duxxdb._native")]
struct EvalRegistry {
    inner: RustEvalRegistry,
    dim: usize,
}

#[pymethods]
impl EvalRegistry {
    #[new]
    #[pyo3(signature = (dim = 32, storage = None))]
    fn new(dim: usize, storage: Option<&str>) -> PyResult<Self> {
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(dim));
        let backend = open_backend_arc(storage)?;
        let inner = RustEvalRegistry::open(embedder, backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner, dim })
    }

    #[pyo3(signature = (
        dataset_name,
        dataset_version,
        model,
        scorer,
        prompt_name = None,
        prompt_version = None,
        metadata = None,
    ))]
    fn start(
        &self,
        py: Python<'_>,
        dataset_name: &str,
        dataset_version: u64,
        model: &str,
        scorer: &str,
        prompt_name: Option<String>,
        prompt_version: Option<u64>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let m = py_to_json(py, metadata)?;
        Ok(self.inner.start(
            dataset_name.to_string(),
            dataset_version,
            prompt_name,
            prompt_version,
            model.to_string(),
            scorer.to_string(),
            m,
        ))
    }

    #[pyo3(signature = (run_id, row_id, score, output_text = String::new(), notes = None))]
    fn score(
        &self,
        py: Python<'_>,
        run_id: &str,
        row_id: &str,
        score: f32,
        output_text: String,
        notes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let n = py_to_json(py, notes)?;
        self.inner
            .score(run_id, row_id.to_string(), score, output_text, n)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn complete(&self, run_id: &str) -> PyResult<EvalSummaryPy> {
        self.inner
            .complete(run_id)
            .map(EvalSummaryPy::from_rust)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn fail(&self, run_id: &str, reason: &str) -> PyResult<()> {
        self.inner
            .fail(run_id, reason.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get(&self, run_id: &str) -> Option<EvalRunPy> {
        self.inner.get(run_id).map(EvalRunPy::from_rust)
    }

    fn scores(&self, run_id: &str) -> Vec<EvalScorePy> {
        self.inner.scores(run_id).into_iter().map(EvalScorePy::from_rust).collect()
    }

    #[pyo3(signature = (dataset_name = None, dataset_version = None))]
    fn list(&self, dataset_name: Option<String>, dataset_version: Option<u64>) -> Vec<EvalRunPy> {
        let runs = match (dataset_name, dataset_version) {
            (Some(n), Some(v)) => self.inner.list_runs_for(&n, v),
            _ => self.inner.list_runs(),
        };
        runs.into_iter().map(EvalRunPy::from_rust).collect()
    }

    /// Returns ``(mean_delta, pass_rate_50_delta, regressed_row_count,
    /// improved_row_count, new_row_count, dropped_row_count)``.
    fn compare(&self, run_a: &str, run_b: &str) -> PyResult<(f32, f32, usize, usize, usize, usize)> {
        let c = self
            .inner
            .compare(run_a, run_b)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            c.mean_delta,
            c.pass_rate_50_delta,
            c.regressed.len(),
            c.improved.len(),
            c.new_rows.len(),
            c.dropped_rows.len(),
        ))
    }

    /// Returns ``(representative_row_id, representative_text, member_count, mean_score)`` per cluster.
    #[pyo3(signature = (run_id, score_threshold = 0.5, sim_threshold = 0.8, max_clusters = 10))]
    fn cluster_failures(
        &self,
        run_id: &str,
        score_threshold: f32,
        sim_threshold: f32,
        max_clusters: usize,
    ) -> PyResult<Vec<(String, String, usize, f32)>> {
        let clusters = self
            .inner
            .cluster_failures(run_id, score_threshold, sim_threshold, max_clusters)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(clusters
            .into_iter()
            .map(|c| {
                (
                    c.representative_row_id,
                    c.representative_text,
                    c.members.len(),
                    c.mean_score,
                )
            })
            .collect())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "<EvalRegistry runs={} scores={} running={} completed={} failed={}>",
            s.runs, s.scores, s.running, s.completed, s.failed
        )
    }
}

// ---------------------------------------------------------------------------
// ReplayRegistry (Phase 7.5, v0.2.1)
// ---------------------------------------------------------------------------

#[pyclass(name = "ReplayInvocation", module = "duxxdb._native")]
#[derive(Clone)]
struct ReplayInvocationPy {
    #[pyo3(get)]
    idx: usize,
    #[pyo3(get)]
    span_id: String,
    /// One of ``"llm_call"`` / ``"tool_call:<name>"`` / ``"other:<label>"``.
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    model: Option<String>,
    #[pyo3(get)]
    prompt_name: Option<String>,
    #[pyo3(get)]
    prompt_version: Option<u64>,
    input_json: String,
    output_json: Option<String>,
    metadata_json: String,
    #[pyo3(get)]
    recorded_at_unix_ns: u128,
}

#[pymethods]
impl ReplayInvocationPy {
    #[getter]
    fn input(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.input_json)
    }
    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.output_json {
            Some(s) => Ok(Some(json_str_to_py(py, s)?)),
            None => Ok(None),
        }
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }
    fn __repr__(&self) -> String {
        format!("<ReplayInvocation idx={} kind={}>", self.idx, self.kind)
    }
}

fn invocation_kind_to_string(k: RustInvocationKind) -> String {
    match k {
        RustInvocationKind::LlmCall => "llm_call".into(),
        RustInvocationKind::ToolCall { tool } => format!("tool_call:{tool}"),
        RustInvocationKind::Other { label } => format!("other:{label}"),
    }
}

fn invocation_kind_from_string(s: &str) -> RustInvocationKind {
    if let Some(tool) = s.strip_prefix("tool_call:") {
        RustInvocationKind::ToolCall { tool: tool.into() }
    } else if let Some(label) = s.strip_prefix("other:") {
        RustInvocationKind::Other { label: label.into() }
    } else {
        RustInvocationKind::LlmCall
    }
}

impl ReplayInvocationPy {
    fn from_rust(i: RustReplayInvocation) -> Self {
        Self {
            idx: i.idx,
            span_id: i.span_id,
            kind: invocation_kind_to_string(i.kind),
            model: i.model,
            prompt_name: i.prompt_name,
            prompt_version: i.prompt_version,
            input_json: serde_json::to_string(&i.input).unwrap_or_else(|_| "null".into()),
            output_json: i
                .output
                .map(|o| serde_json::to_string(&o).unwrap_or_else(|_| "null".into())),
            metadata_json: serde_json::to_string(&i.metadata).unwrap_or_else(|_| "null".into()),
            recorded_at_unix_ns: i.recorded_at_unix_ns,
        }
    }
}

#[pyclass(name = "ReplaySession", module = "duxxdb._native")]
#[derive(Clone)]
struct ReplaySessionPy {
    #[pyo3(get)]
    trace_id: String,
    #[pyo3(get)]
    invocations: Vec<ReplayInvocationPy>,
    #[pyo3(get)]
    fingerprint: String,
    #[pyo3(get)]
    captured_at_unix_ns: u128,
}

#[pymethods]
impl ReplaySessionPy {
    fn __repr__(&self) -> String {
        format!(
            "<ReplaySession trace_id={:?} invocations={}>",
            self.trace_id,
            self.invocations.len(),
        )
    }
}

impl ReplaySessionPy {
    fn from_rust(s: RustReplaySession) -> Self {
        Self {
            trace_id: s.trace_id,
            invocations: s.invocations.into_iter().map(ReplayInvocationPy::from_rust).collect(),
            fingerprint: s.fingerprint,
            captured_at_unix_ns: s.captured_at_unix_ns,
        }
    }
}

#[pyclass(name = "ReplayRun", module = "duxxdb._native")]
#[derive(Clone)]
struct ReplayRunPy {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    source_trace_id: String,
    /// One of ``"cached"`` / ``"live"`` / ``"stepped"``.
    #[pyo3(get)]
    mode: String,
    /// One of ``"pending"`` / ``"running"`` / ``"completed"`` / ``"failed"``.
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    current_idx: usize,
    #[pyo3(get)]
    replay_trace_id: Option<String>,
    #[pyo3(get)]
    created_at_unix_ns: u128,
    #[pyo3(get)]
    completed_at_unix_ns: Option<u128>,
    metadata_json: String,
}

#[pymethods]
impl ReplayRunPy {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.metadata_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<ReplayRun id={:?} mode={} status={} current_idx={}>",
            self.id, self.mode, self.status, self.current_idx
        )
    }
}

fn replay_mode_to_string(m: RustReplayMode) -> String {
    match m {
        RustReplayMode::Cached => "cached",
        RustReplayMode::Live => "live",
        RustReplayMode::Stepped => "stepped",
    }
    .into()
}

fn replay_mode_from_string(s: &str) -> PyResult<RustReplayMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cached" => Ok(RustReplayMode::Cached),
        "live" => Ok(RustReplayMode::Live),
        "stepped" => Ok(RustReplayMode::Stepped),
        other => Err(PyValueError::new_err(format!(
            "unknown replay mode {other:?} (use: cached | live | stepped)"
        ))),
    }
}

fn replay_status_to_string(s: RustReplayStatus) -> String {
    match s {
        RustReplayStatus::Pending => "pending",
        RustReplayStatus::Running => "running",
        RustReplayStatus::Completed => "completed",
        RustReplayStatus::Failed => "failed",
    }
    .into()
}

impl ReplayRunPy {
    fn from_rust(r: RustReplayRun) -> Self {
        Self {
            id: r.id,
            source_trace_id: r.source_trace_id,
            mode: replay_mode_to_string(r.mode),
            status: replay_status_to_string(r.status),
            current_idx: r.current_idx,
            replay_trace_id: r.replay_trace_id,
            created_at_unix_ns: r.created_at_unix_ns,
            completed_at_unix_ns: r.completed_at_unix_ns,
            metadata_json: serde_json::to_string(&r.metadata).unwrap_or_else(|_| "null".into()),
        }
    }
}

/// Deterministic agent replay.
#[pyclass(name = "ReplayRegistry", module = "duxxdb._native")]
struct ReplayRegistry {
    inner: RustReplayRegistry,
}

#[pymethods]
impl ReplayRegistry {
    #[new]
    #[pyo3(signature = (storage = None))]
    fn new(storage: Option<&str>) -> PyResult<Self> {
        let backend = open_backend_arc(storage)?;
        let inner = RustReplayRegistry::open(backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Capture one invocation against ``trace_id``. Returns the
    /// assigned ordinal index.
    #[pyo3(signature = (
        trace_id,
        kind,
        input,
        model = None,
        prompt_name = None,
        prompt_version = None,
        output = None,
        metadata = None,
        span_id = String::new(),
    ))]
    fn capture(
        &self,
        py: Python<'_>,
        trace_id: &str,
        kind: &str,
        input: &Bound<'_, PyAny>,
        model: Option<String>,
        prompt_name: Option<String>,
        prompt_version: Option<u64>,
        output: Option<&Bound<'_, PyAny>>,
        metadata: Option<&Bound<'_, PyAny>>,
        span_id: String,
    ) -> PyResult<usize> {
        let inv = RustReplayInvocation {
            idx: 0, // assigned by capture()
            span_id,
            kind: invocation_kind_from_string(kind),
            model,
            prompt_name,
            prompt_version,
            input: py_to_json(py, Some(input))?,
            output: match output {
                Some(o) if !o.is_none() => Some(py_to_json(py, Some(o))?),
                _ => None,
            },
            metadata: py_to_json(py, metadata)?,
            recorded_at_unix_ns: 0,
        };
        Ok(self.inner.capture(trace_id.to_string(), inv))
    }

    fn get_session(&self, trace_id: &str) -> Option<ReplaySessionPy> {
        self.inner.get_session(trace_id).map(ReplaySessionPy::from_rust)
    }

    fn list_sessions(&self) -> Vec<ReplaySessionPy> {
        self.inner.list_sessions().into_iter().map(ReplaySessionPy::from_rust).collect()
    }

    /// Start a new replay run.
    #[pyo3(signature = (source_trace_id, mode = "live", metadata = None))]
    fn start(
        &self,
        py: Python<'_>,
        source_trace_id: &str,
        mode: &str,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let m = replay_mode_from_string(mode)?;
        let md = py_to_json(py, metadata)?;
        self.inner
            .start(source_trace_id.to_string(), m, Vec::new(), md)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn step(&self, run_id: &str) -> PyResult<Option<ReplayInvocationPy>> {
        self.inner
            .step(run_id)
            .map(|o| o.map(ReplayInvocationPy::from_rust))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn record_output(
        &self,
        py: Python<'_>,
        run_id: &str,
        invocation_idx: usize,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let v = py_to_json(py, Some(output))?;
        self.inner
            .record_output(run_id, invocation_idx, v)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn complete(&self, run_id: &str) -> PyResult<()> {
        self.inner
            .complete(run_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn fail(&self, run_id: &str, reason: &str) -> PyResult<()> {
        self.inner
            .fail(run_id, reason.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn set_replay_trace_id(&self, run_id: &str, replay_trace_id: &str) -> PyResult<()> {
        self.inner
            .set_replay_trace_id(run_id, replay_trace_id.to_string())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get_run(&self, run_id: &str) -> Option<ReplayRunPy> {
        self.inner.get_run(run_id).map(ReplayRunPy::from_rust)
    }

    #[pyo3(signature = (source_trace_id = None))]
    fn list_runs(&self, source_trace_id: Option<String>) -> Vec<ReplayRunPy> {
        match source_trace_id {
            Some(t) => self.inner.list_runs_for(&t),
            None => self.inner.list_runs(),
        }
        .into_iter()
        .map(ReplayRunPy::from_rust)
        .collect()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "<ReplayRegistry sessions={} invocations={} runs={}>",
            s.sessions, s.invocations, s.runs,
        )
    }
}

// ---------------------------------------------------------------------------
// TraceStore (Phase 7.1, v0.2.1)
// ---------------------------------------------------------------------------

#[pyclass(name = "Span", module = "duxxdb._native")]
#[derive(Clone)]
struct SpanPy {
    #[pyo3(get)]
    trace_id: String,
    #[pyo3(get)]
    span_id: String,
    #[pyo3(get)]
    parent_span_id: Option<String>,
    #[pyo3(get)]
    thread_id: Option<String>,
    #[pyo3(get)]
    name: String,
    /// One of ``"internal"`` / ``"server"`` / ``"client"`` /
    /// ``"producer"`` / ``"consumer"``.
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    start_unix_ns: u128,
    #[pyo3(get)]
    end_unix_ns: Option<u128>,
    /// One of ``"unset"`` / ``"ok"`` / ``"error"``.
    #[pyo3(get)]
    status: String,
    attributes_json: String,
}

#[pymethods]
impl SpanPy {
    #[getter]
    fn attributes(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_str_to_py(py, &self.attributes_json)
    }
    fn __repr__(&self) -> String {
        format!(
            "<Span trace={} span={} name={:?} status={}>",
            self.trace_id, self.span_id, self.name, self.status,
        )
    }
}

fn span_kind_to_string(k: RustSpanKind) -> String {
    match k {
        RustSpanKind::Internal => "internal",
        RustSpanKind::Server => "server",
        RustSpanKind::Client => "client",
        RustSpanKind::Producer => "producer",
        RustSpanKind::Consumer => "consumer",
    }
    .into()
}

fn span_kind_from_string(s: &str) -> RustSpanKind {
    match s.trim().to_ascii_lowercase().as_str() {
        "server" => RustSpanKind::Server,
        "client" => RustSpanKind::Client,
        "producer" => RustSpanKind::Producer,
        "consumer" => RustSpanKind::Consumer,
        _ => RustSpanKind::Internal,
    }
}

fn span_status_to_string(s: RustSpanStatus) -> String {
    match s {
        RustSpanStatus::Unset => "unset",
        RustSpanStatus::Ok => "ok",
        RustSpanStatus::Error => "error",
    }
    .into()
}

fn span_status_from_string(s: &str) -> RustSpanStatus {
    match s.trim().to_ascii_lowercase().as_str() {
        "ok" => RustSpanStatus::Ok,
        "error" | "err" => RustSpanStatus::Error,
        _ => RustSpanStatus::Unset,
    }
}

impl SpanPy {
    fn from_rust(s: RustSpan) -> Self {
        Self {
            trace_id: s.trace_id,
            span_id: s.span_id,
            parent_span_id: s.parent_span_id,
            thread_id: s.thread_id,
            name: s.name,
            kind: span_kind_to_string(s.kind),
            start_unix_ns: s.start_unix_ns,
            end_unix_ns: s.end_unix_ns,
            status: span_status_to_string(s.status),
            attributes_json: serde_json::to_string(&s.attributes).unwrap_or_else(|_| "null".into()),
        }
    }
}

/// Distributed-trace storage for agent observability.
#[pyclass(name = "TraceStore", module = "duxxdb._native")]
struct TraceStore {
    inner: RustTraceStore,
}

#[pymethods]
impl TraceStore {
    #[new]
    #[pyo3(signature = (storage = None))]
    fn new(storage: Option<&str>) -> PyResult<Self> {
        let backend = open_backend_arc(storage)?;
        let inner = RustTraceStore::open(backend)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (
        trace_id,
        span_id,
        name,
        parent_span_id = None,
        thread_id = None,
        kind = "internal",
        start_unix_ns = 0,
        end_unix_ns = None,
        status = "unset",
        attributes = None,
    ))]
    fn record_span(
        &self,
        py: Python<'_>,
        trace_id: &str,
        span_id: &str,
        name: &str,
        parent_span_id: Option<String>,
        thread_id: Option<String>,
        kind: &str,
        start_unix_ns: u128,
        end_unix_ns: Option<u128>,
        status: &str,
        attributes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let start = if start_unix_ns == 0 {
            now_unix_ns()
        } else {
            start_unix_ns
        };
        let span = RustSpan {
            trace_id: trace_id.into(),
            span_id: span_id.into(),
            parent_span_id,
            thread_id,
            name: name.into(),
            kind: span_kind_from_string(kind),
            start_unix_ns: start,
            end_unix_ns,
            status: span_status_from_string(status),
            attributes: py_to_json(py, attributes)?,
        };
        self.inner
            .record_span(span)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (span_id, end_unix_ns, status = "ok"))]
    fn close_span(&self, span_id: &str, end_unix_ns: u128, status: &str) -> PyResult<()> {
        self.inner
            .close_span(span_id, end_unix_ns, span_status_from_string(status))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn get_trace(&self, trace_id: &str) -> Vec<SpanPy> {
        self.inner.get_trace(trace_id).into_iter().map(SpanPy::from_rust).collect()
    }

    fn subtree(&self, span_id: &str) -> Vec<SpanPy> {
        self.inner.subtree(span_id).into_iter().map(SpanPy::from_rust).collect()
    }

    fn thread(&self, thread_id: &str) -> Vec<SpanPy> {
        self.inner.thread(thread_id).into_iter().map(SpanPy::from_rust).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<TraceStore spans={} traces={} threads={}>",
            self.inner.span_count(),
            self.inner.trace_count(),
            self.inner.thread_count(),
        )
    }
}

fn now_unix_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Resolve a storage spec into an ``Arc<dyn Backend>``. Shared by
/// every Phase 7 native binding above.
fn open_backend_arc(storage: Option<&str>) -> PyResult<Arc<dyn Backend>> {
    match storage {
        None | Some("") | Some("memory") => Ok(Arc::new(MemoryBackend::new())),
        Some(spec) => open_backend(Some(spec))
            .map(Arc::from)
            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

/// Native module entrypoint. Maturin packages this as `duxxdb._native`.
/// The pure-Python facade in `python/duxxdb/__init__.py` re-exports
/// the public surface.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<MemoryStore>()?;
    m.add_class::<MemoryHit>()?;
    m.add_class::<ToolCache>()?;
    m.add_class::<ToolCacheHit>()?;
    m.add_class::<SessionStore>()?;
    m.add_class::<PromptRegistry>()?;
    m.add_class::<Prompt>()?;
    m.add_class::<PromptHit>()?;
    // v0.2.1: native bindings for the remaining Phase 7 primitives.
    m.add_class::<CostLedger>()?;
    m.add_class::<CostEntryPy>()?;
    m.add_class::<BudgetPy>()?;
    m.add_class::<DatasetRegistry>()?;
    m.add_class::<DatasetPy>()?;
    m.add_class::<DatasetRowPy>()?;
    m.add_class::<EvalRegistry>()?;
    m.add_class::<EvalRunPy>()?;
    m.add_class::<EvalScorePy>()?;
    m.add_class::<EvalSummaryPy>()?;
    m.add_class::<ReplayRegistry>()?;
    m.add_class::<ReplayInvocationPy>()?;
    m.add_class::<ReplaySessionPy>()?;
    m.add_class::<ReplayRunPy>()?;
    m.add_class::<TraceStore>()?;
    m.add_class::<SpanPy>()?;
    Ok(())
}
