//! # duxxdb-py
//!
//! Python bindings for DuxxDB. Exposes the agent-native primitives
//! (`MemoryStore`, `ToolCache`, `SessionStore`) as Python classes via
//! [PyO3]. Compiled into a stable-API extension module (`abi3-py38`)
//! so a single wheel works across Python 3.8 → 3.13.
//!
//! [PyO3]: https://github.com/PyO3/pyo3

use duxx_embed::{Embedder, HashEmbedder};
use duxx_memory::{
    HitKind as RustHitKind, MemoryStore as RustMemoryStore, SessionStore as RustSessionStore,
    ToolCache as RustToolCache,
};
use duxx_prompts::{Prompt as RustPrompt, PromptRegistry as RustPromptRegistry};
use duxx_storage::{open_backend, Backend, MemoryBackend};
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
    Ok(())
}
