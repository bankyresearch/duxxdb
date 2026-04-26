//! # duxx-memory
//!
//! High-level primitives for AI-agent use cases: `MemoryStore`,
//! `ToolCache`, `Session`. These are what chatbots and voice bots talk
//! to directly. Internally they compose `duxx-storage`, `duxx-index`,
//! and `duxx-query`.
//!
//! Phase 1: everything in-memory; vector and text indices are the
//! placeholder implementations from `duxx-index`. Replaced transparently
//! in Phase 2.

use duxx_core::Result;
use duxx_index::{TextIndex, VectorIndex};
use duxx_query::{hybrid_recall, RecallHit};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A single stored memory record.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: u64,
    pub key: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub importance: f32,
}

/// A recall result — a `Memory` paired with its fused score.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub memory: Memory,
    pub score: f32,
}

/// An in-memory memory store partitioned by logical key (user / agent / session).
///
/// Writes update both the vector index and the text index atomically,
/// so recalls are always consistent.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    dim: usize,
    by_id: RwLock<HashMap<u64, Memory>>,
    vector_index: RwLock<VectorIndex>,
    text_index: RwLock<TextIndex>,
    next_id: RwLock<u64>,
}

impl MemoryStore {
    /// Create a store with the default vector-index capacity.
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, 100_000)
    }

    /// Create a store with an explicit vector-index capacity. Use a
    /// smaller value when memory is tight (benchmarks, embedded use)
    /// or a larger one when you expect millions of memories per store.
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                dim,
                by_id: RwLock::new(HashMap::new()),
                vector_index: RwLock::new(VectorIndex::with_capacity(dim, capacity)),
                text_index: RwLock::new(TextIndex::new()),
                next_id: RwLock::new(1),
            }),
        }
    }

    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    pub fn len(&self) -> usize {
        self.inner.by_id.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a new memory, updating all indices.
    ///
    /// Returns the assigned id. `embedding.len()` must equal `self.dim()`.
    pub fn remember(
        &self,
        key: impl Into<String>,
        text: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Result<u64> {
        let key = key.into();
        let text = text.into();
        let id = {
            let mut n = self.inner.next_id.write();
            let v = *n;
            *n += 1;
            v
        };
        self.inner.vector_index.write().insert(id, embedding.clone())?;
        self.inner.text_index.write().insert(id, text.clone())?;
        let mem = Memory {
            id,
            key,
            text,
            embedding,
            importance: 1.0,
        };
        self.inner.by_id.write().insert(id, mem);
        tracing::debug!(id, "remembered");
        Ok(id)
    }

    /// Hybrid recall over vector + text indices, fused with RRF.
    ///
    /// `_key` is accepted for API symmetry and to make partition-by-key
    /// filtering easy to wire up once `duxx-query` supports pushdown.
    pub fn recall(
        &self,
        _key: &str,
        query_text: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<MemoryHit>> {
        let v = self.inner.vector_index.read();
        let t = self.inner.text_index.read();
        let hits: Vec<RecallHit> = hybrid_recall(&v, &t, query_vec, query_text, k)?;
        let by_id = self.inner.by_id.read();
        let mut out = Vec::with_capacity(hits.len());
        for h in hits {
            if let Some(m) = by_id.get(&h.id) {
                out.push(MemoryHit {
                    memory: m.clone(),
                    score: h.score,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(text: &str, dim: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut v = vec![0.0f32; dim];
        for w in text.to_lowercase().split_whitespace() {
            let mut h = DefaultHasher::new();
            w.hash(&mut h);
            let b = (h.finish() as usize) % dim;
            v[b] += 1.0;
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= n;
        }
        v
    }

    #[test]
    fn remember_and_recall_roundtrip() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        s.remember("u1", "I want a refund for my order", embed("refund order", DIM))
            .unwrap();
        s.remember("u1", "What's the weather?", embed("weather", DIM))
            .unwrap();
        assert_eq!(s.len(), 2);

        let q = "refund";
        let hits = s.recall("u1", q, &embed(q, DIM), 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].memory.text.contains("refund"));
    }
}
