//! # duxx-memory
//!
//! Agent-facing primitives:
//! - [`MemoryStore`] — long-term semantic memory with hybrid recall and
//!   exponential importance decay.
//! - [`ToolCache`] — exact + semantic-near-hit cache for expensive tool
//!   calls.
//! - [`SessionStore`] — hot KV with sliding TTL for per-conversation
//!   working state.

pub mod session;
pub mod tool_cache;

use duxx_core::Result;
use duxx_index::{TextIndex, VectorIndex};
use duxx_query::{hybrid_recall, RecallHit};
use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use duxx_storage::Storage;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

// Re-export so users get the full Storage surface from one place.
pub use duxx_storage::{MemoryStorage, Storage as StorageTrait};
#[cfg(feature = "redb-store")]
pub use duxx_storage::RedbStorage;

pub use session::{SessionStore, DEFAULT_TTL as DEFAULT_SESSION_TTL};
pub use tool_cache::{HitKind, ToolCache, ToolCacheHit, DEFAULT_NEAR_HIT_THRESHOLD};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A single stored memory.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: u64,
    pub key: String,
    pub text: String,
    pub embedding: Vec<f32>,
    /// Base importance, set at insert time. The *effective* importance
    /// returned at recall time is computed via [`Memory::effective_importance`]
    /// and decays exponentially with age.
    pub importance: f32,
    /// Wall-clock insert time as nanoseconds since the Unix epoch.
    /// Wall-clock so it survives process restart (a monotonic `Instant`
    /// would not). Tiny NTP-induced wobble is acceptable for decay use.
    pub created_at_unix_ns: u128,
}

/// Current wall-clock time as Unix-epoch nanoseconds.
fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

impl Memory {
    /// Importance decayed by elapsed time, half-life style.
    ///
    /// `effective = importance * 2^(-age / half_life)`
    ///
    /// Age is measured against `now_unix_ns()` so the value continues
    /// to decay correctly after a process restart.
    pub fn effective_importance(&self, half_life: Duration) -> f32 {
        let now = now_unix_ns();
        let age_ns = now.saturating_sub(self.created_at_unix_ns);
        let age_secs = age_ns as f32 / 1.0e9;
        let half = half_life.as_secs_f32().max(1e-3);
        self.importance * 2.0_f32.powf(-age_secs / half)
    }
}

/// Recall result — a memory paired with its (decayed) score.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub memory: Memory,
    pub score: f32,
}

/// Long-term semantic memory store.
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
    bus: ChangeBus,
    /// Optional durable backing store. When `Some`, every `remember`
    /// writes through to it; on `with_storage` open we replay everything
    /// back into the in-memory caches and indices.
    storage: Option<Arc<dyn Storage>>,
}

/// Wire format for persisted memories. `created_at_unix_ns` is stored
/// so importance decay continues across process restart (Phase 2.4).
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredMemory {
    id: u64,
    key: String,
    text: String,
    embedding: Vec<f32>,
    importance: f32,
    /// Unix epoch nanoseconds. `0` for legacy rows persisted before
    /// Phase 2.4 — restored as "ancient" so they still appear but
    /// with near-zero decayed importance.
    #[serde(default)]
    created_at_unix_ns: u128,
}

impl From<&Memory> for StoredMemory {
    fn from(m: &Memory) -> Self {
        Self {
            id: m.id,
            key: m.key.clone(),
            text: m.text.clone(),
            embedding: m.embedding.clone(),
            importance: m.importance,
            created_at_unix_ns: m.created_at_unix_ns,
        }
    }
}

impl MemoryStore {
    /// Create a store with the default vector-index capacity.
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, 100_000)
    }

    /// Create a store with an explicit vector-index capacity.
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                dim,
                by_id: RwLock::new(HashMap::new()),
                vector_index: RwLock::new(VectorIndex::with_capacity(dim, capacity)),
                text_index: RwLock::new(TextIndex::new()),
                next_id: RwLock::new(1),
                bus: ChangeBus::default(),
                storage: None,
            }),
        }
    }

    /// Build a store backed by a durable [`Storage`].
    ///
    /// On open, every memory currently in `storage` is replayed back
    /// into the in-memory cache and rebuilt into the vector / text
    /// indices. Subsequent `remember` calls write through to `storage`.
    ///
    /// Cold start cost is roughly proportional to the corpus size —
    /// expect ~250 µs per memory on NVMe (HNSW insert + tantivy add).
    /// At 1 M rows that's ~5 minutes; Phase 2.3.5 will persist the
    /// indices alongside the rows to skip rebuild.
    pub fn with_storage(
        dim: usize,
        capacity: usize,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let store = Self::with_capacity(dim, capacity);
        let rows = storage.iter()?;
        let mut max_id = 0u64;
        let n = rows.len();
        for (id, bytes) in rows {
            let stored: StoredMemory = bincode::deserialize(&bytes).map_err(|e| {
                duxx_core::Error::Storage(format!("decode memory id={id}: {e}"))
            })?;
            if stored.embedding.len() != dim {
                return Err(duxx_core::Error::Storage(format!(
                    "memory id={id} has dim {} but store expects {dim}",
                    stored.embedding.len()
                )));
            }
            store
                .inner
                .vector_index
                .write()
                .insert(id, stored.embedding.clone())?;
            store.inner.text_index.write().insert(id, stored.text.clone())?;
            let mem = Memory {
                id,
                key: stored.key,
                text: stored.text,
                embedding: stored.embedding,
                importance: stored.importance,
                created_at_unix_ns: stored.created_at_unix_ns,
            };
            store.inner.by_id.write().insert(id, mem);
            if id > max_id {
                max_id = id;
            }
        }
        *store.inner.next_id.write() = max_id + 1;
        // Now wire the storage into Inner so future writes flow through.
        // We have to re-build Inner because Inner owns storage and we
        // already started constructing without it.
        let inner = Arc::new(Inner {
            dim,
            by_id: RwLock::new(std::mem::take(&mut *store.inner.by_id.write())),
            vector_index: RwLock::new(std::mem::replace(
                &mut *store.inner.vector_index.write(),
                VectorIndex::new(dim),
            )),
            text_index: RwLock::new(std::mem::replace(
                &mut *store.inner.text_index.write(),
                TextIndex::new(),
            )),
            next_id: RwLock::new(*store.inner.next_id.read()),
            bus: ChangeBus::default(),
            storage: Some(storage),
        });
        tracing::info!(memories = n, "loaded memories from storage");
        Ok(MemoryStore { inner })
    }

    /// Subscribe to change events on this store. Each `remember` call
    /// publishes one [`ChangeEvent`] (table = `"memory"`).
    ///
    /// The receiver lives behind a [`tokio::sync::broadcast`] channel —
    /// slow subscribers may miss messages if the buffer (default 1024)
    /// is exceeded.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
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

    /// Insert a new memory. `embedding.len()` must match `self.dim()`.
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
            key: key.clone(),
            text: text.clone(),
            embedding,
            importance: 1.0,
            created_at_unix_ns: now_unix_ns(),
        };
        // Write through to durable storage BEFORE inserting into the
        // in-memory cache. If the persistence write fails, we don't
        // want a phantom row in memory.
        if let Some(s) = &self.inner.storage {
            let stored = StoredMemory::from(&mem);
            let bytes = bincode::serialize(&stored).map_err(|e| {
                duxx_core::Error::Storage(format!("encode memory id={id}: {e}"))
            })?;
            s.put(id, &bytes)?;
        }
        self.inner.by_id.write().insert(id, mem);
        // Publish a change event. Lossy by design — slow subscribers
        // may miss events. Channel = "memory.<key>", so `PSUBSCRIBE
        // memory.*` filters by user/agent key.
        self.inner.bus.publish(ChangeEvent {
            table: "memory".to_string(),
            key: Some(key.clone()),
            row_id: id,
            kind: ChangeKind::Insert,
        });
        tracing::debug!(id, key = %key, "remembered");
        Ok(id)
    }

    /// Whether this store has a durable backing.
    pub fn is_persistent(&self) -> bool {
        self.inner.storage.is_some()
    }

    /// Hybrid recall (vector + BM25, RRF-fused).
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

    /// Recall with importance-decay reranking.
    ///
    /// First runs hybrid recall, then multiplies each hit's RRF score
    /// by the memory's effective importance (exponential half-life).
    pub fn recall_decayed(
        &self,
        key: &str,
        query_text: &str,
        query_vec: &[f32],
        k: usize,
        half_life: std::time::Duration,
    ) -> Result<Vec<MemoryHit>> {
        let mut hits = self.recall(key, query_text, query_vec, k * 2)?;
        for h in &mut hits {
            h.score *= h.memory.effective_importance(half_life);
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        Ok(hits)
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

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_persistence_across_reopen() {
        const DIM: usize = 8;
        let dir = std::env::temp_dir().join(format!(
            "duxx-memory-persist-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memories.redb");

        // First open: write data.
        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage(DIM, 1_000, storage).unwrap();
            store
                .remember("alice", "I lost my wallet", embed("wallet", DIM))
                .unwrap();
            store
                .remember("alice", "Favorite color blue", embed("blue", DIM))
                .unwrap();
            assert_eq!(store.len(), 2);
        }

        // Second open: data should still be there + queryable.
        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage(DIM, 1_000, storage).unwrap();
            assert_eq!(store.len(), 2, "data should survive reopen");
            let hits = store
                .recall("alice", "wallet", &embed("wallet", DIM), 5)
                .unwrap();
            assert!(!hits.is_empty());
            assert!(hits[0].memory.text.contains("wallet"));
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn effective_importance_decays() {
        // Half-life 1ms → after a few ms, importance should be near zero.
        let s = MemoryStore::new(8);
        let id = s
            .remember("u", "test", embed("test", 8))
            .unwrap();
        let m = s.inner.by_id.read().get(&id).cloned().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let eff = m.effective_importance(std::time::Duration::from_millis(1));
        assert!(eff < 0.01, "effective importance after many half-lives: {eff}");
    }

    #[test]
    fn ancient_memory_decays_to_near_zero() {
        // Manually construct a Memory with a created_at far in the past.
        let m = Memory {
            id: 1,
            key: "u".into(),
            text: "ancient".into(),
            embedding: vec![0.0; 4],
            importance: 1.0,
            created_at_unix_ns: 0, // 1970 — extremely old
        };
        let eff = m.effective_importance(std::time::Duration::from_secs(60 * 60));
        assert!(eff < 1e-10, "expected near-zero decay, got {eff}");
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn decay_continues_across_restart() {
        // Insert a memory, persist it. Reopen. Verify the loaded
        // memory's created_at_unix_ns matches the original (i.e. the
        // age grew across the reopen, didn't reset).
        const DIM: usize = 4;
        let dir = std::env::temp_dir().join(format!(
            "duxx-decay-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("decay.redb");

        let original_ts: u128;
        let id: u64;
        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage(DIM, 1_000, storage).unwrap();
            id = store
                .remember("u", "remembered_long_ago", embed("ancient", DIM))
                .unwrap();
            let m = store.inner.by_id.read().get(&id).cloned().unwrap();
            original_ts = m.created_at_unix_ns;
            assert!(original_ts > 0);
        }

        // Wait at least 1 ms so age > 0 across restart.
        std::thread::sleep(std::time::Duration::from_millis(2));

        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage(DIM, 1_000, storage).unwrap();
            let m = store.inner.by_id.read().get(&id).cloned().unwrap();
            assert_eq!(
                m.created_at_unix_ns, original_ts,
                "timestamp must be preserved across restart"
            );
            // Now decay a millisecond-half-life: should be very small.
            let eff = m.effective_importance(std::time::Duration::from_millis(1));
            assert!(eff < 1.0, "should have aged, got eff={eff}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
