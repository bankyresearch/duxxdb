//! # duxx-index
//!
//! Vector and full-text indices for DuxxDB.
//!
//! **Phase 1 (this file):**
//! - `VectorIndex` — exhaustive cosine search over an in-memory list. Fine up to ~10 k vectors.
//! - `TextIndex` — naive term-count scoring. Sufficient for the demo.
//!
//! **Phase 2:** `VectorIndex` switches to [usearch](https://github.com/unum-cloud/usearch)
//! HNSW and `TextIndex` switches to [tantivy](https://github.com/quickwit-oss/tantivy)
//! BM25. The trait-like surface here is preserved.

use duxx_core::{Error, Result};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Vector index
// ---------------------------------------------------------------------------

/// Exhaustive cosine vector index — a Phase-1 placeholder for HNSW.
#[derive(Debug, Clone)]
pub struct VectorIndex {
    dim: usize,
    vectors: Vec<(u64, Vec<f32>)>,
}

impl VectorIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, vectors: Vec::new() }
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
        self.vectors.push((id, vec));
        Ok(())
    }

    /// Return the top-`k` ids by cosine similarity, descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (*id, cosine(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Cosine similarity between two equal-length vectors.
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

// ---------------------------------------------------------------------------
// Text (BM25 placeholder)
// ---------------------------------------------------------------------------

/// Naive term-count text index — a Phase-1 placeholder for tantivy BM25.
#[derive(Debug, Clone, Default)]
pub struct TextIndex {
    docs: Vec<(u64, String)>,
}

impl TextIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: u64, text: String) -> Result<()> {
        self.docs.push((id, text));
        Ok(())
    }

    /// Return top-`k` ids by term-match count, descending.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        let q = query.to_lowercase();
        let terms: Vec<&str> = q.split_whitespace().collect();
        let mut scored: Vec<(u64, f32)> = self
            .docs
            .iter()
            .map(|(id, text)| {
                let lower = text.to_lowercase();
                let score: f32 = terms
                    .iter()
                    .map(|t| lower.matches(t).count() as f32)
                    .sum();
                (*id, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
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
    fn vector_index_finds_self_first() {
        let mut idx = VectorIndex::new(3);
        idx.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn text_index_term_matches() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund my order".into()).unwrap();
        idx.insert(2, "weather today".into()).unwrap();
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn vector_dim_mismatch_errors() {
        let mut idx = VectorIndex::new(3);
        let err = idx.insert(1, vec![1.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::Index(_)));
    }
}
