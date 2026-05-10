//! Hash-bucket embedder. Deterministic, dependency-free, useful for
//! tests and self-contained demos. **Not** a real semantic embedder —
//! it has no notion of word meaning, just collision-prone token hashing.

use crate::Embedder;
use duxx_core::Result;

/// Hash-bucket embedder. Tokens (whitespace-split, lowercased) are
/// hashed into `dim`-many buckets; the resulting count vector is
/// L2-normalized.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "HashEmbedder dim must be positive");
        Self { dim }
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut v = vec![0.0f32; self.dim];
        for token in text.to_lowercase().split_whitespace() {
            let mut h = DefaultHasher::new();
            token.hash(&mut h);
            let bucket = (h.finish() as usize) % self.dim;
            v[bucket] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_returns_zero_vector() {
        let e = HashEmbedder::new(8);
        let v = e.embed("").unwrap();
        assert_eq!(v.len(), 8);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn deterministic() {
        let e = HashEmbedder::new(16);
        assert_eq!(e.embed("hello world").unwrap(), e.embed("hello world").unwrap());
    }

    #[test]
    fn case_insensitive() {
        let e = HashEmbedder::new(16);
        assert_eq!(e.embed("Hello World").unwrap(), e.embed("hello world").unwrap());
    }

    #[test]
    fn dim_matches() {
        for dim in [4usize, 16, 32, 128, 1024] {
            let e = HashEmbedder::new(dim);
            assert_eq!(e.dim(), dim);
            assert_eq!(e.embed("anything").unwrap().len(), dim);
        }
    }

    #[test]
    fn l2_normalized_for_nonempty_input() {
        let e = HashEmbedder::new(32);
        let v = e.embed("a quick brown fox").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }
}
