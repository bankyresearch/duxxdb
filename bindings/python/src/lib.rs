//! # duxxdb-py
//!
//! Python bindings for DuxxDB. Exposes the agent-native primitives
//! (`MemoryStore`, `ToolCache`, `SessionStore`) as Python classes via
//! [PyO3]. Compiled into a stable-API extension module (`abi3-py38`)
//! so a single wheel works across Python 3.8 → 3.13.
//!
//! [PyO3]: https://github.com/PyO3/pyo3

use duxx_memory::{
    HitKind as RustHitKind, MemoryStore as RustMemoryStore, SessionStore as RustSessionStore,
    ToolCache as RustToolCache,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
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

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<MemoryStore dim={} memories={}>",
            self.inner.dim(),
            self.inner.len()
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
    Ok(())
}
