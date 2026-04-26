//! `TOOL_CACHE` — caches the results of expensive tool calls
//! (web search, LLM summarization, slow DB lookups) so an agent doesn't
//! re-run them when arguments are *equivalent*, not just identical.
//!
//! Two-stage lookup:
//! 1. **Exact-hash hit.** Same tool name + same `args_hash` → return.
//! 2. **Semantic near-hit.** If no exact hit, scan all cached args
//!    embeddings for this tool. Return any whose cosine similarity to
//!    the query embedding is `>= threshold` (default `0.95`).
//!
//! Hits are filtered by per-entry TTL.

use duxx_core::Result;
use duxx_index::cosine;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default cosine-similarity threshold for a semantic near-hit.
pub const DEFAULT_NEAR_HIT_THRESHOLD: f32 = 0.95;

/// What kind of hit was returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    /// Exact `(tool, args_hash)` match.
    Exact,
    /// Cosine match on `args_embedding`.
    SemanticNearHit,
}

/// A cache hit.
#[derive(Debug, Clone)]
pub struct ToolCacheHit {
    pub kind: HitKind,
    pub similarity: f32,
    /// Opaque result bytes — caller decides serialization.
    pub result: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Entry {
    result: Vec<u8>,
    stored_at: Instant,
    ttl: Duration,
}

impl Entry {
    fn fresh(&self) -> bool {
        self.stored_at.elapsed() < self.ttl
    }
}

#[derive(Default, Debug)]
struct Inner {
    /// (tool_name, args_hash) -> Entry. Primary index.
    entries: HashMap<(String, u64), Entry>,
    /// tool_name -> list of (hash, embedding). Secondary for vector probe.
    by_tool: HashMap<String, Vec<(u64, Vec<f32>)>>,
}

/// Semantic-aware tool-result cache.
#[derive(Debug, Clone)]
pub struct ToolCache {
    inner: Arc<RwLock<Inner>>,
    threshold: f32,
}

impl ToolCache {
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_NEAR_HIT_THRESHOLD)
    }

    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
            threshold,
        }
    }

    /// Insert (or overwrite) a cache entry.
    pub fn put(
        &self,
        tool_name: impl Into<String>,
        args_hash: u64,
        args_embedding: Vec<f32>,
        result: Vec<u8>,
        ttl: Duration,
    ) -> Result<()> {
        let tool_name = tool_name.into();
        let entry = Entry {
            result,
            stored_at: Instant::now(),
            ttl,
        };
        let mut inner = self.inner.write();
        inner.entries.insert((tool_name.clone(), args_hash), entry);
        let bucket = inner.by_tool.entry(tool_name).or_default();
        // Replace any stale entry for the same hash; otherwise append.
        if let Some(slot) = bucket.iter_mut().find(|(h, _)| *h == args_hash) {
            slot.1 = args_embedding;
        } else {
            bucket.push((args_hash, args_embedding));
        }
        Ok(())
    }

    /// Look up a cache entry. Tries exact-hash first, then semantic.
    pub fn get(
        &self,
        tool_name: &str,
        args_hash: u64,
        args_embedding: &[f32],
    ) -> Option<ToolCacheHit> {
        let inner = self.inner.read();

        // Exact hit?
        if let Some(entry) = inner.entries.get(&(tool_name.to_string(), args_hash)) {
            if entry.fresh() {
                return Some(ToolCacheHit {
                    kind: HitKind::Exact,
                    similarity: 1.0,
                    result: entry.result.clone(),
                });
            }
        }

        // Semantic near-hit.
        let candidates = inner.by_tool.get(tool_name)?;
        let mut best: Option<(u64, f32)> = None;
        for (hash, emb) in candidates {
            let sim = cosine(args_embedding, emb);
            if sim >= self.threshold {
                let take = match best {
                    Some((_, b)) => sim > b,
                    None => true,
                };
                if take {
                    best = Some((*hash, sim));
                }
            }
        }
        let (hash, sim) = best?;
        let entry = inner.entries.get(&(tool_name.to_string(), hash))?;
        if !entry.fresh() {
            return None;
        }
        Some(ToolCacheHit {
            kind: HitKind::SemanticNearHit,
            similarity: sim,
            result: entry.result.clone(),
        })
    }

    /// Purge expired entries; returns how many were removed.
    pub fn purge_expired(&self) -> usize {
        let mut inner = self.inner.write();
        let before = inner.entries.len();
        let to_drop: Vec<(String, u64)> = inner
            .entries
            .iter()
            .filter(|(_, e)| !e.fresh())
            .map(|(k, _)| k.clone())
            .collect();
        for k in &to_drop {
            inner.entries.remove(k);
            if let Some(bucket) = inner.by_tool.get_mut(&k.0) {
                bucket.retain(|(h, _)| *h != k.1);
            }
        }
        before - inner.entries.len()
    }

    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ToolCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ttl() -> Duration {
        Duration::from_secs(60)
    }

    #[test]
    fn exact_hit() {
        let c = ToolCache::new();
        c.put("web", 42, vec![1.0, 0.0], b"answer".to_vec(), ttl()).unwrap();
        let hit = c.get("web", 42, &[1.0, 0.0]).unwrap();
        assert_eq!(hit.kind, HitKind::Exact);
        assert_eq!(hit.result, b"answer");
    }

    #[test]
    fn semantic_near_hit() {
        let c = ToolCache::new();
        // Stored: (web, h=1, embedding ~[1, 0]).
        c.put("web", 1, vec![1.0, 0.0], b"answer".to_vec(), ttl()).unwrap();
        // Query: same tool, different hash, embedding very close.
        let hit = c.get("web", 999, &[0.99, 0.05]).unwrap();
        assert_eq!(hit.kind, HitKind::SemanticNearHit);
        assert!(hit.similarity >= 0.95);
        assert_eq!(hit.result, b"answer");
    }

    #[test]
    fn miss_when_below_threshold() {
        let c = ToolCache::new();
        c.put("web", 1, vec![1.0, 0.0], b"a".to_vec(), ttl()).unwrap();
        // Orthogonal vector should miss (cosine 0).
        assert!(c.get("web", 999, &[0.0, 1.0]).is_none());
    }

    #[test]
    fn miss_when_different_tool() {
        let c = ToolCache::new();
        c.put("web", 1, vec![1.0, 0.0], b"a".to_vec(), ttl()).unwrap();
        assert!(c.get("calc", 1, &[1.0, 0.0]).is_none());
    }

    #[test]
    fn ttl_expires_entries() {
        let c = ToolCache::new();
        c.put("web", 1, vec![1.0, 0.0], b"a".to_vec(), Duration::from_millis(1)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert!(c.get("web", 1, &[1.0, 0.0]).is_none());
    }

    #[test]
    fn purge_removes_only_expired() {
        let c = ToolCache::new();
        c.put("web", 1, vec![1.0, 0.0], b"a".to_vec(), Duration::from_millis(1)).unwrap();
        c.put("web", 2, vec![0.0, 1.0], b"b".to_vec(), Duration::from_secs(60)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let removed = c.purge_expired();
        assert_eq!(removed, 1);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn put_overwrites_same_key() {
        let c = ToolCache::new();
        c.put("web", 1, vec![1.0, 0.0], b"first".to_vec(), ttl()).unwrap();
        c.put("web", 1, vec![1.0, 0.0], b"second".to_vec(), ttl()).unwrap();
        assert_eq!(c.len(), 1);
        let hit = c.get("web", 1, &[1.0, 0.0]).unwrap();
        assert_eq!(hit.result, b"second");
    }
}
