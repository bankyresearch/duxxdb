//! Vector index — HNSW (Hierarchical Navigable Small World).
//!
//! Phase 2.2 — backed by [`hnsw_rs`] cosine HNSW. Sub-linear search
//! latency that scales to multi-million points.
//!
//! Tunables (compile-time constants for now; per-table config in Phase 3):
//! - `M = 16`            — max graph degree per layer
//! - `EF_CONSTRUCTION`   — build-time candidate-pool size (200)
//! - `EF_SEARCH`         — query-time candidate-pool size (64)
//! - `MAX_LAYERS`        — HNSW layer cap (16)
//! - `MAX_ELEMENTS`      — pre-allocated capacity (1 M)
//!
//! [`hnsw_rs`]: https://crates.io/crates/hnsw_rs

use duxx_core::{Error, Result};
use hnsw_rs::prelude::{DistCosine, Hnsw};
use parking_lot::RwLock;

const M: usize = 16;
const DEFAULT_MAX_ELEMENTS: usize = 100_000;
const MAX_LAYERS: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 64;

/// HNSW vector index over cosine distance.
///
/// `hnsw_rs::Hnsw` uses `usize` internal point ids; we map them onto the
/// caller's `u64` ids via an internal `Vec<u64>`.
pub struct VectorIndex {
    dim: usize,
    inner: RwLock<HnswInner>,
}

struct HnswInner {
    hnsw: Hnsw<'static, f32, DistCosine>,
    next_internal: usize,
    /// internal index → external id
    id_map: Vec<u64>,
}

impl VectorIndex {
    /// Construct with the default capacity ([`DEFAULT_MAX_ELEMENTS`]).
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, DEFAULT_MAX_ELEMENTS)
    }

    /// Construct with an explicit `capacity`. `hnsw_rs` cannot grow past
    /// the cap, so size for the largest workload you expect.
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        let hnsw = Hnsw::new(M, capacity, MAX_LAYERS, EF_CONSTRUCTION, DistCosine);
        Self {
            dim,
            inner: RwLock::new(HnswInner {
                hnsw,
                next_internal: 0,
                id_map: Vec::new(),
            }),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn insert(&mut self, id: u64, vec: Vec<f32>) -> Result<()> {
        if vec.len() != self.dim {
            return Err(Error::Index(format!(
                "vector dim mismatch: expected {}, got {}",
                self.dim,
                vec.len()
            )));
        }
        let mut inner = self.inner.write();
        let internal = inner.next_internal;
        inner.id_map.push(id);
        inner.next_internal += 1;
        // hnsw_rs copies the slice contents internally on insert.
        inner.hnsw.insert((&vec, internal));
        Ok(())
    }

    /// Top-`k` ids by cosine similarity, descending.
    ///
    /// `hnsw_rs` returns *distance* (1 − cos for `DistCosine`); we flip
    /// to similarity so higher = better, matching the prior placeholder.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let inner = self.inner.read();
        let neighbours = inner.hnsw.search(query, k, EF_SEARCH);
        neighbours
            .into_iter()
            .filter_map(|n| {
                let id = inner.id_map.get(n.d_id)?;
                let similarity = 1.0 - n.distance;
                Some((*id, similarity))
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.read().id_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("dim", &self.dim)
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Kept as a free function for callers that want it without going through
/// the index (e.g. unit tests, ad-hoc rerankers).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_vector_with_itself_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hnsw_finds_self_first() {
        let mut idx = VectorIndex::new(3);
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 0.0, 1.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0], 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 1, "self should be top-1");
    }

    #[test]
    fn hnsw_orders_by_similarity() {
        let mut idx = VectorIndex::new(3);
        // Two clusters: near (1,0,0) and near (0,1,0).
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.95, 0.1, 0.0]).unwrap();
        idx.insert(3, vec![0.0, 1.0, 0.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0], 3);
        assert!(hits.len() >= 2);
        // Doc 3 should rank below 1 and 2.
        let doc3 = hits.iter().position(|h| h.0 == 3);
        let doc1 = hits.iter().position(|h| h.0 == 1).unwrap();
        if let Some(p3) = doc3 {
            assert!(doc1 < p3);
        }
    }

    #[test]
    fn vector_dim_mismatch_errors() {
        let mut idx = VectorIndex::new(3);
        let err = idx.insert(1, vec![1.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::Index(_)));
    }

    #[test]
    fn len_tracks_inserts() {
        let mut idx = VectorIndex::new(2);
        assert_eq!(idx.len(), 0);
        idx.insert(1, vec![1.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 2);
    }
}
