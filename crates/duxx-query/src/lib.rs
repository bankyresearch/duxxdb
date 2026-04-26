//! # duxx-query
//!
//! Hybrid query engine for DuxxDB.
//!
//! The signature operation is [`hybrid_recall`], which runs a query
//! against a `VectorIndex` and a `TextIndex` in parallel and fuses the
//! rankings via **Reciprocal Rank Fusion (RRF)**.
//!
//! RRF is index-agnostic and well-behaved even when the underlying scores
//! are on different scales (cosine similarity vs. BM25). The formula:
//!
//! ```text
//! score(doc) = Σ_i   1 / (k + rank_i(doc))
//! ```
//!
//! where `rank_i(doc)` is the 1-indexed rank of `doc` in retriever `i`'s
//! result list, and `k` is a smoothing constant (we default to 60, which
//! matches the original [Cormack et al. 2009] recommendation).

use duxx_core::Result;
use duxx_index::{TextIndex, VectorIndex};
use std::collections::HashMap;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A single fused recall hit.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallHit {
    pub id: u64,
    pub score: f32,
}

/// Default RRF smoothing constant.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Reciprocal Rank Fusion over multiple ranked lists.
///
/// Input: vector of ranked `(id, _score)` lists, each in descending relevance.
/// Output: top-`top_n` fused `RecallHit`s, descending by fused score.
pub fn rrf_fuse(rankings: Vec<Vec<(u64, f32)>>, k: f32, top_n: usize) -> Vec<RecallHit> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, (id, _score)) in ranking.into_iter().enumerate() {
            *scores.entry(id).or_insert(0.0) += 1.0 / (k + (rank + 1) as f32);
        }
    }
    let mut out: Vec<RecallHit> = scores
        .into_iter()
        .map(|(id, score)| RecallHit { id, score })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(top_n);
    out
}

/// Hybrid recall: query a vector index and a text index and fuse the results.
///
/// `per_retriever` is the candidate cap per retriever before fusion.
/// Returns up to `k` results.
pub fn hybrid_recall(
    vector_index: &VectorIndex,
    text_index: &TextIndex,
    query_vec: &[f32],
    query_text: &str,
    k: usize,
) -> Result<Vec<RecallHit>> {
    let per = (k * 3).max(32);
    let v_hits = vector_index.search(query_vec, per);
    let t_hits = text_index.search(query_text, per);
    tracing::debug!(v = v_hits.len(), t = t_hits.len(), "hybrid_recall pre-fusion");
    Ok(rrf_fuse(vec![v_hits, t_hits], DEFAULT_RRF_K, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_elevates_doc_ranked_high_in_both() {
        // doc 1 is in top-2 of both retrievers; doc 3 is top-1 in one only.
        let a = vec![(1u64, 0.9), (2, 0.5), (3, 0.3)];
        let b = vec![(3u64, 1.0), (1, 0.8), (4, 0.2)];
        let fused = rrf_fuse(vec![a, b], 60.0, 10);
        assert_eq!(fused[0].id, 1);
    }

    #[test]
    fn hybrid_recall_end_to_end() {
        let mut v = VectorIndex::new(3);
        v.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        v.insert(2, vec![0.0, 1.0, 0.0]).unwrap();

        let mut t = TextIndex::new();
        t.insert(1, "refund request".into()).unwrap();
        t.insert(2, "weather forecast".into()).unwrap();

        let hits = hybrid_recall(&v, &t, &[1.0, 0.0, 0.0], "refund", 5).unwrap();
        assert_eq!(hits[0].id, 1);
    }
}
