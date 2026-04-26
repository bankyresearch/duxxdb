//! Vector index — exhaustive cosine for now; HNSW lands in Phase 2.2.

use duxx_core::{Error, Result};

/// Exhaustive cosine vector index. Linear-scan, suitable up to ~10 k vectors.
///
/// Replaced by `hnsw_rs::Hnsw` in Phase 2.2 — same public API, sub-linear
/// search.
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

    /// Top-`k` ids by cosine similarity, descending.
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
    fn vector_dim_mismatch_errors() {
        let mut idx = VectorIndex::new(3);
        let err = idx.insert(1, vec![1.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::Index(_)));
    }
}
