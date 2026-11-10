//! # duxx-rerank
//!
//! Second-stage rerankers for DuxxDB. First-stage retrieval (HNSW +
//! BM25 fused by `RECALL`) is fast but scores the query and each
//! document separately; a cross-encoder reads the query and a candidate
//! *together* and scores how well the candidate answers it, which is
//! markedly more precise — and far more expensive, so it is applied
//! only to the top few dozen candidates.
//!
//! - [`Reranker`] — the trait the server holds.
//! - [`CrossEncoder`] (feature `local`) — XLM-RoBERTa cross-encoders on
//!   Candle, in-process, CPU. Default model
//!   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual: English,
//!   Hindi and ~100 more).
//! - [`OverlapReranker`] — a dependency-free toy (query-term overlap),
//!   the reranker counterpart of `HashEmbedder`, for tests and demos.
//!
//! ```no_run
//! # #[cfg(feature = "local")] {
//! use duxx_rerank::Reranker;
//! let r = duxx_rerank::from_spec(Some("local:mmarco-minilm-l12")).unwrap().unwrap();
//! let top = r.rerank("capital of India", &["Paris is in France", "नई दिल्ली भारत की राजधानी है"], 1).unwrap();
//! assert_eq!(top[0].0, 1);
//! # }
//! ```

use duxx_core::{Error, Result};

#[cfg(feature = "local")]
mod cross_encoder;
#[cfg(feature = "local")]
pub use cross_encoder::{CrossEncoder, RERANK_MODELS};

/// Scores how relevant each document is to a query.
pub trait Reranker: Send + Sync {
    /// One score per document, in input order. Higher is more relevant;
    /// scores are in `[0, 1]` and comparable within one call.
    fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;

    /// Stable identity, e.g. `local:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd652930`.
    fn id(&self) -> String;

    /// The `top_k` most relevant documents as `(index, score)`, best
    /// first. Ties keep input order.
    fn rerank(&self, query: &str, docs: &[&str], top_k: usize) -> Result<Vec<(usize, f32)>> {
        let scores = self.score(query, docs)?;
        if scores.len() != docs.len() {
            return Err(Error::Internal(format!(
                "reranker returned {} scores for {} docs",
                scores.len(),
                docs.len()
            )));
        }
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        Ok(ranked)
    }
}

impl<T: Reranker + ?Sized> Reranker for std::sync::Arc<T> {
    fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        (**self).score(query, docs)
    }
    fn id(&self) -> String {
        (**self).id()
    }
}

impl<T: Reranker + ?Sized> Reranker for Box<T> {
    fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        (**self).score(query, docs)
    }
    fn id(&self) -> String {
        (**self).id()
    }
}

/// Toy reranker: the fraction of distinct query terms (lowercased,
/// whitespace-split) that occur in the document. No model, no
/// semantics — use it to exercise reranking plumbing, not for quality.
#[derive(Debug, Clone, Default)]
pub struct OverlapReranker;

impl Reranker for OverlapReranker {
    fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        let mut terms: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).collect();
        terms.sort();
        terms.dedup();
        if terms.is_empty() {
            return Ok(vec![0.0; docs.len()]);
        }
        Ok(docs
            .iter()
            .map(|d| {
                let words: std::collections::HashSet<String> =
                    d.split_whitespace().map(|t| t.to_lowercase()).collect();
                terms.iter().filter(|t| words.contains(*t)).count() as f32 / terms.len() as f32
            })
            .collect())
    }

    fn id(&self) -> String {
        "overlap".to_string()
    }
}

/// Build a reranker from a spec string (e.g. `DUXX_RERANKER`).
///
/// - `"local:<model>[,max_tokens=N][,batch=N]"` (feature `local`) —
///   a cross-encoder. `<model>` is an alias from [`RERANK_MODELS`]
///   (`mmarco-minilm-l12`, `mmarco-minilm-l6`, `bge-reranker-base`),
///   a Hugging Face repo id of one of them, or a local model directory.
/// - `"overlap"` — [`OverlapReranker`].
/// - `"none"` or `None` — no reranker (`Ok(None)`).
pub fn from_spec(spec: Option<&str>) -> Result<Option<Box<dyn Reranker>>> {
    let Some(spec) = spec.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if spec.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    if spec.eq_ignore_ascii_case("overlap") {
        return Ok(Some(Box::new(OverlapReranker)));
    }
    let (kind, rest) = spec.split_once(':').unwrap_or((spec, ""));
    match kind {
        #[cfg(feature = "local")]
        "local" => Ok(Some(Box::new(CrossEncoder::from_spec(rest)?))),
        #[cfg(not(feature = "local"))]
        "local" => {
            let _ = rest;
            Err(Error::Internal(
                "reranker spec local:<model> needs DuxxDB built with the `local` \
                 feature of duxx-rerank (the official Docker image has it)"
                    .to_string(),
            ))
        }
        other => Err(Error::Internal(format!(
            "unknown reranker spec {other:?}: expected local:<model>, overlap or none"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_ranks_matching_doc_first() {
        let r = OverlapReranker;
        let docs = [
            "the cat sat",
            "monsoon rains in Kerala",
            "Kerala monsoon arrives early",
        ];
        let top = r.rerank("Kerala monsoon", &docs, 2).unwrap();
        assert_eq!(top.len(), 2);
        // Both full matches score 1.0; ties keep input order.
        assert_eq!(top[0], (1, 1.0));
        assert_eq!(top[1], (2, 1.0));
        let all = r.rerank("cat", &docs, 10).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].0, 0);
    }

    #[test]
    fn spec_parsing() {
        assert!(from_spec(None).unwrap().is_none());
        assert!(from_spec(Some("")).unwrap().is_none());
        assert!(from_spec(Some("none")).unwrap().is_none());
        assert_eq!(from_spec(Some("overlap")).unwrap().unwrap().id(), "overlap");
        assert!(from_spec(Some("cohere:rerank-3")).is_err());
        #[cfg(not(feature = "local"))]
        assert!(from_spec(Some("local:mmarco-minilm-l12")).is_err());
    }
}
