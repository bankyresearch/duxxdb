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
#[cfg(feature = "redb-store")]
pub use duxx_storage::RedbStorage;
pub use duxx_storage::{MemoryStorage, Storage as StorageTrait};

pub use session::{SessionStore, DEFAULT_TTL as DEFAULT_SESSION_TTL};
pub use tool_cache::{HitKind, ToolCache, ToolCacheHit, DEFAULT_NEAR_HIT_THRESHOLD};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default tombstone ratio (`tombstones / live`) at which `remember`
/// auto-triggers a [`MemoryStore::compact`]. See [`MemoryStore::set_auto_compact_ratio`].
pub const DEFAULT_AUTO_COMPACT_RATIO: f32 = 0.20;

/// Don't auto-compact below this live-row count. Compaction is O(n); on a
/// tiny store the rebuild cost dwarfs the recall it buys back, and a single
/// eviction would otherwise blow past the ratio and thrash. Explicit
/// [`MemoryStore::compact`] is unaffected by this floor.
const AUTO_COMPACT_MIN_LIVE: usize = 64;

/// Default time-to-live for idempotency keys (see
/// [`MemoryStore::remember_idempotent`]). A retry within this window returns
/// the original id instead of inserting a duplicate.
pub const DEFAULT_IDEMPOTENCY_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Purge expired idempotency entries once the map exceeds this many keys.
const IDEMPOTENCY_SOFT_CAP: usize = 10_000;

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
    /// Optional row cap (Phase 6.2). When set and exceeded, the lowest
    /// effective-importance row gets evicted on every subsequent
    /// `remember`. `None` means unlimited.
    max_rows: RwLock<Option<usize>>,
    /// Half-life used by the eviction policy when reranking by
    /// effective importance. Default 24 h.
    evict_half_life: RwLock<Duration>,
    /// Number of rows evicted by the cap so far. Useful for metrics.
    evictions_total: std::sync::atomic::AtomicU64,
    /// Number of graph compactions performed so far (manual + auto).
    compactions_total: std::sync::atomic::AtomicU64,
    /// Tombstone ratio at which `remember` auto-triggers `compact`.
    /// `None` disables auto-compaction. Default [`DEFAULT_AUTO_COMPACT_RATIO`].
    auto_compact_ratio: RwLock<Option<f32>>,
    /// Idempotency cache: `idempotency_key -> (row_id, recorded_at_unix_ns)`.
    /// A repeat `remember_idempotent` with a live key returns the original id.
    idempotency: RwLock<HashMap<String, (u64, u128)>>,
    /// TTL for idempotency keys. Default [`DEFAULT_IDEMPOTENCY_TTL`].
    idempotency_ttl: RwLock<Duration>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // In persistent mode, the disk-backed tantivy index batches
        // commits via `commit_every` (default 100). On graceful
        // shutdown we MUST flush the pending writes so the on-disk
        // index matches `vector_index` — otherwise the next open will
        // see an empty tantivy and trigger a full rebuild that
        // double-inserts every memory.
        if self.storage.is_some() {
            if let Err(e) = self.text_index.read().flush() {
                tracing::warn!(error = %e, "MemoryStore Drop: tantivy flush failed");
            }
        }
        // VectorIndex's own Drop dumps HNSW to disk.
    }
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
                max_rows: RwLock::new(None),
                evict_half_life: RwLock::new(Duration::from_secs(24 * 60 * 60)),
                evictions_total: std::sync::atomic::AtomicU64::new(0),
                compactions_total: std::sync::atomic::AtomicU64::new(0),
                auto_compact_ratio: RwLock::new(Some(DEFAULT_AUTO_COMPACT_RATIO)),
                idempotency: RwLock::new(HashMap::new()),
                idempotency_ttl: RwLock::new(DEFAULT_IDEMPOTENCY_TTL),
            }),
        }
    }

    /// Build a store backed by a durable [`Storage`].
    ///
    /// On open, every memory currently in `storage` is replayed back
    /// into the in-memory cache and rebuilt into the vector / text
    /// indices. Subsequent `remember` calls write through to `storage`.
    ///
    /// Cold start cost is proportional to the corpus size — full HNSW
    /// + tantivy rebuild from rows.
    ///
    /// For persisted indices that skip the rebuild on graceful-shutdown
    /// reopens, use [`MemoryStore::open_at`] instead.
    pub fn with_storage(dim: usize, capacity: usize, storage: Arc<dyn Storage>) -> Result<Self> {
        let store = Self::with_capacity(dim, capacity);
        let rows = storage.iter()?;
        let mut max_id = 0u64;
        let n = rows.len();
        for (id, bytes) in rows {
            let stored: StoredMemory = bincode::deserialize(&bytes)
                .map_err(|e| duxx_core::Error::Storage(format!("decode memory id={id}: {e}")))?;
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
            store
                .inner
                .text_index
                .write()
                .insert(id, stored.text.clone())?;
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
            max_rows: RwLock::new(None),
            evict_half_life: RwLock::new(Duration::from_secs(24 * 60 * 60)),
            evictions_total: std::sync::atomic::AtomicU64::new(0),
            compactions_total: std::sync::atomic::AtomicU64::new(0),
            auto_compact_ratio: RwLock::new(Some(DEFAULT_AUTO_COMPACT_RATIO)),
            idempotency: RwLock::new(HashMap::new()),
            idempotency_ttl: RwLock::new(DEFAULT_IDEMPOTENCY_TTL),
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

    /// Configure a maximum number of rows. Once exceeded, every
    /// subsequent `remember` evicts the row with the lowest *effective*
    /// (decayed) importance until the row count is back at the cap.
    /// `None` (the default) means unlimited.
    ///
    /// **Phase 6.2.** This is the agent-friendly knob: a long-running
    /// chatbot can keep `--max-memories 1_000_000` and trust the store
    /// to forget the boring stuff before the interesting stuff.
    pub fn set_max_rows(&self, cap: Option<usize>) {
        *self.inner.max_rows.write() = cap;
    }

    /// Read the current row cap (`None` = unlimited).
    pub fn max_rows(&self) -> Option<usize> {
        *self.inner.max_rows.read()
    }

    /// Set the half-life used to weight effective importance during
    /// eviction. Default is 24 h. Shorter = forget recent stuff faster.
    pub fn set_eviction_half_life(&self, hl: Duration) {
        *self.inner.evict_half_life.write() = hl;
    }

    /// Total number of rows evicted by the cap since process start.
    pub fn evictions_total(&self) -> u64 {
        self.inner
            .evictions_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total number of graph compactions performed since process start
    /// (manual [`MemoryStore::compact`] + auto-triggered).
    pub fn compactions_total(&self) -> u64 {
        self.inner
            .compactions_total
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Current tombstone ratio: `(indexed - live) / live`.
    ///
    /// `forget`/eviction drop a row from `by_id` but leave its node in the
    /// HNSW graph (and its postings in tantivy) — those stale entries are
    /// tombstones. The ratio is how many tombstones the vector graph carries
    /// per live row; it climbs with deletions and resets to `0` after
    /// [`MemoryStore::compact`]. Returns `0.0` for an empty store.
    pub fn tombstone_ratio(&self) -> f32 {
        let live = self.inner.by_id.read().len();
        if live == 0 {
            return 0.0;
        }
        let indexed = self.inner.vector_index.read().len();
        indexed.saturating_sub(live) as f32 / live as f32
    }

    /// Set the tombstone ratio at which `remember` auto-triggers a
    /// compaction. `None` disables auto-compaction (explicit
    /// [`MemoryStore::compact`] still works). Default
    /// [`DEFAULT_AUTO_COMPACT_RATIO`] (0.20). Values `<= 0.0` are treated
    /// as "disabled".
    pub fn set_auto_compact_ratio(&self, ratio: Option<f32>) {
        *self.inner.auto_compact_ratio.write() = ratio;
    }

    /// The configured auto-compaction tombstone ratio (`None` = disabled).
    pub fn auto_compact_ratio(&self) -> Option<f32> {
        *self.inner.auto_compact_ratio.read()
    }

    /// Drop one row by id from the row map and durable storage.
    /// Returns true if the id existed.
    ///
    /// **Note:** The underlying HNSW + tantivy indices retain the
    /// vector / token postings until the next process restart. That's
    /// fine for correctness — `recall` filters every hit through
    /// `by_id` (see below), so an evicted row never appears in
    /// results. Reclaiming the index memory itself requires HNSW
    /// tombstones + tantivy deletes, which is Phase 6.3 work.
    pub fn forget(&self, id: u64) -> bool {
        let removed = self.inner.by_id.write().remove(&id).is_some();
        if removed {
            if let Some(s) = &self.inner.storage {
                let _ = s.delete(id);
            }
            self.inner
                .evictions_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.bus.publish(ChangeEvent {
                table: "memory".to_string(),
                key: None,
                row_id: id,
                kind: ChangeKind::Delete,
            });
        }
        removed
    }

    /// If the store is over its `max_rows` cap, evict the lowest
    /// effective-importance rows until back at the cap. No-op when
    /// no cap is set or the store is already under it.
    fn enforce_cap(&self) {
        let cap = match *self.inner.max_rows.read() {
            Some(c) => c,
            None => return,
        };
        let len_now = self.inner.by_id.read().len();
        if len_now <= cap {
            return;
        }
        let half_life = *self.inner.evict_half_life.read();

        // Snapshot (id, effective_importance) under the read lock,
        // then sort ascending. We drop the read lock before forget()
        // takes the write lock.
        let mut scored: Vec<(u64, f32)> = {
            let by_id = self.inner.by_id.read();
            by_id
                .values()
                .map(|m| (m.id, m.effective_importance(half_life)))
                .collect()
        };
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_evict = len_now - cap;
        for (id, _score) in scored.into_iter().take(to_evict) {
            self.forget(id);
        }
    }

    /// Rebuild the vector + text indices from the surviving rows, dropping
    /// every tombstone left behind by `forget`/eviction.
    ///
    /// Why this exists: `hnsw_rs` has no in-place delete, so forgotten rows
    /// stay nodes in the navigable graph. `recall` filters them out via
    /// `by_id` (so results are always *correct*), but the deleted nodes are
    /// still traversed during search and burn the candidate budget — recall
    /// *quality* rots in exactly the regions you've deleted from most. Rebuild
    /// is the supported repair: a fresh graph over the survivors, swapped in
    /// atomically, with tantivy rebuilt to match.
    ///
    /// Returns the number of tombstones reclaimed (`indexed - live` before the
    /// rebuild). Cheap when there are none. Takes write locks on both indices
    /// for the duration; concurrent `recall`/`remember` block until it
    /// finishes. Locks are acquired in `remember`'s order (vector → text) so
    /// the two can't deadlock.
    pub fn compact(&self) -> Result<usize> {
        // Acquire the index write locks first (same order remember() uses),
        // then read by_id. Holding both index locks across the rebuild stops
        // a concurrent remember from inserting a row we'd then lose.
        let mut v = self.inner.vector_index.write();
        let mut t = self.inner.text_index.write();

        let before_indexed = v.len();
        let mut survivors_vec: Vec<(u64, Vec<f32>)> = Vec::new();
        let mut survivors_text: Vec<(u64, String)> = Vec::new();
        {
            let by_id = self.inner.by_id.read();
            survivors_vec.reserve(by_id.len());
            survivors_text.reserve(by_id.len());
            for m in by_id.values() {
                survivors_vec.push((m.id, m.embedding.clone()));
                survivors_text.push((m.id, m.text.clone()));
            }
        }
        let live = survivors_vec.len();
        let reclaimed = before_indexed.saturating_sub(live);

        v.rebuild(&survivors_vec)?;
        t.rebuild(&survivors_text)?;

        // Persist the freshly compacted indices when durable, so a later
        // reopen sees the compacted form rather than re-loading tombstones.
        if self.inner.storage.is_some() {
            if let Err(e) = v.dump() {
                tracing::warn!(error = %e, "compact: hnsw dump failed");
            }
            // tantivy already committed inside rebuild(); flush is a no-op.
        }
        drop(v);
        drop(t);

        self.inner
            .compactions_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(live, reclaimed, "memory compaction complete");
        Ok(reclaimed)
    }

    /// Auto-trigger compaction if the tombstone ratio has crossed the
    /// configured threshold. Called at the end of `remember`. No-op when
    /// auto-compaction is disabled or the store is below the size floor.
    fn maybe_auto_compact(&self) {
        let threshold = match *self.inner.auto_compact_ratio.read() {
            Some(t) if t > 0.0 => t,
            _ => return,
        };
        if self.inner.by_id.read().len() < AUTO_COMPACT_MIN_LIVE {
            return;
        }
        if self.tombstone_ratio() >= threshold {
            if let Err(e) = self.compact() {
                tracing::warn!(error = %e, "auto-compaction failed");
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner.by_id.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot every memory as a Vec. Order is by id ascending.
    /// Used by the cold-tier exporter; the cost is O(n) clones.
    pub fn all_memories(&self) -> Vec<Memory> {
        let by_id = self.inner.by_id.read();
        let mut out: Vec<Memory> = by_id.values().cloned().collect();
        out.sort_by_key(|m| m.id);
        out
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
        self.inner
            .vector_index
            .write()
            .insert(id, embedding.clone())?;
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
            let bytes = bincode::serialize(&stored)
                .map_err(|e| duxx_core::Error::Storage(format!("encode memory id={id}: {e}")))?;
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
        // Phase 6.2: enforce the optional row cap. Cheap when no cap
        // is configured (single read of an Option<usize>).
        self.enforce_cap();
        // P0: if eviction (or prior forgets) left enough tombstones in the
        // graph, rebuild it now so recall quality doesn't rot. No-op unless
        // the tombstone ratio crossed the configured threshold.
        self.maybe_auto_compact();
        Ok(id)
    }

    /// Like [`MemoryStore::remember`], but **idempotent** on `idem_key`: a
    /// repeat call with the same key within the TTL returns the id of the
    /// original insert instead of creating a duplicate. Lets clients retry a
    /// write after a network blip without double-inserting.
    ///
    /// The check-and-insert is serialized under the idempotency lock, so two
    /// concurrent calls with the same key still produce exactly one row.
    pub fn remember_idempotent(
        &self,
        key: impl Into<String>,
        text: impl Into<String>,
        embedding: Vec<f32>,
        idem_key: impl Into<String>,
    ) -> Result<u64> {
        let idem_key = idem_key.into();
        let ttl_ns = self.inner.idempotency_ttl.read().as_nanos();
        // Held across the check + insert so concurrent retries can't race.
        let mut cache = self.inner.idempotency.write();
        if let Some((id, ts)) = cache.get(&idem_key) {
            if now_unix_ns().saturating_sub(*ts) < ttl_ns {
                return Ok(*id);
            }
        }
        let id = self.remember(key, text, embedding)?;
        if cache.len() >= IDEMPOTENCY_SOFT_CAP {
            let now = now_unix_ns();
            cache.retain(|_, (_, ts)| now.saturating_sub(*ts) < ttl_ns);
        }
        cache.insert(idem_key, (id, now_unix_ns()));
        Ok(id)
    }

    /// Set the idempotency-key TTL (default [`DEFAULT_IDEMPOTENCY_TTL`]).
    pub fn set_idempotency_ttl(&self, ttl: Duration) {
        *self.inner.idempotency_ttl.write() = ttl;
    }

    /// Whether this store has a durable backing.
    pub fn is_persistent(&self) -> bool {
        self.inner.storage.is_some()
    }

    /// Open a fully-persistent store rooted at `dir`. Three artifacts
    /// share the directory:
    ///
    /// - `<dir>/store.redb`  — row store (redb)
    /// - `<dir>/tantivy/`    — disk-backed BM25 index
    /// - `<dir>/hnsw/`       — HNSW dump + id-map sidecar
    ///
    /// Reopening an existing dir after a graceful shutdown skips the
    /// expensive HNSW + tantivy rebuild — both indices come back ready
    /// to query in milliseconds. After a hard kill / panic-abort the
    /// HNSW dump may be missing or stale; we detect that by comparing
    /// `vector_index.len()` to the row count and rebuild from the row
    /// store when they disagree.
    #[cfg(feature = "redb-store")]
    pub fn open_at(dim: usize, capacity: usize, dir: impl AsRef<std::path::Path>) -> Result<Self> {
        use duxx_index::{TextIndex, VectorIndex};

        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| duxx_core::Error::Storage(format!("create memory dir {dir:?}: {e}")))?;

        // 1. Open the three on-disk artifacts.
        let storage: Arc<dyn Storage> =
            Arc::new(duxx_storage::RedbStorage::open(dir.join("store.redb"))?);
        let text_index = TextIndex::open(dir.join("tantivy"))?;
        let vector_index = VectorIndex::open(dim, capacity, dir.join("hnsw"))?;

        // 2. Decide: do the indices already match the row store?
        //    If yes, we can skip the rebuild. If not (fresh dir, missing
        //    dump, or stale after a crash), full rebuild is the safe choice.
        let row_count = storage.len()?;
        let indices_intact =
            row_count > 0 && vector_index.len() == row_count && text_index.len() == row_count;

        // 3. Walk the row store. Always rebuild `by_id` (it's not persisted
        //    independently). Only re-insert into the indices if they're
        //    NOT intact.
        let mut by_id = HashMap::with_capacity(row_count);
        let mut max_id = 0u64;
        let rows = storage.iter()?;
        let mut text_index = text_index;
        let mut vector_index = vector_index;
        for (id, bytes) in rows {
            let stored: StoredMemory = bincode::deserialize(&bytes)
                .map_err(|e| duxx_core::Error::Storage(format!("decode memory id={id}: {e}")))?;
            if stored.embedding.len() != dim {
                return Err(duxx_core::Error::Storage(format!(
                    "memory id={id} has dim {} but store expects {dim}",
                    stored.embedding.len()
                )));
            }
            if !indices_intact {
                vector_index.insert(id, stored.embedding.clone())?;
                text_index.insert(id, stored.text.clone())?;
            }
            let mem = Memory {
                id,
                key: stored.key,
                text: stored.text,
                embedding: stored.embedding,
                importance: stored.importance,
                created_at_unix_ns: stored.created_at_unix_ns,
            };
            by_id.insert(id, mem);
            if id > max_id {
                max_id = id;
            }
        }
        if !indices_intact && row_count > 0 {
            tracing::info!(
                rows = row_count,
                "rebuilt indices from row store (cold path)"
            );
        } else if indices_intact {
            tracing::info!(rows = row_count, "skipped index rebuild (loaded from disk)");
        }

        let inner = Arc::new(Inner {
            dim,
            by_id: RwLock::new(by_id),
            vector_index: RwLock::new(vector_index),
            text_index: RwLock::new(text_index),
            next_id: RwLock::new(max_id + 1),
            bus: ChangeBus::default(),
            storage: Some(storage),
            max_rows: RwLock::new(None),
            evict_half_life: RwLock::new(Duration::from_secs(24 * 60 * 60)),
            evictions_total: std::sync::atomic::AtomicU64::new(0),
            compactions_total: std::sync::atomic::AtomicU64::new(0),
            auto_compact_ratio: RwLock::new(Some(DEFAULT_AUTO_COMPACT_RATIO)),
            idempotency: RwLock::new(HashMap::new()),
            idempotency_ttl: RwLock::new(DEFAULT_IDEMPOTENCY_TTL),
        });
        Ok(MemoryStore { inner })
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
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
    fn remember_idempotent_dedupes_by_key() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        let e = embed("refund order", DIM);

        // Same idempotency key -> one row, same id returned.
        let id1 = s
            .remember_idempotent("u", "I want a refund", e.clone(), "req-abc")
            .unwrap();
        let id2 = s
            .remember_idempotent("u", "I want a refund", e.clone(), "req-abc")
            .unwrap();
        assert_eq!(id1, id2, "retry must return the original id");
        assert_eq!(s.len(), 1, "no duplicate row");

        // A different key inserts a new row.
        let id3 = s
            .remember_idempotent("u", "different note", embed("weather", DIM), "req-xyz")
            .unwrap();
        assert_ne!(id3, id1);
        assert_eq!(s.len(), 2);

        // After the key expires, the same key inserts again.
        s.set_idempotency_ttl(std::time::Duration::from_nanos(1));
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id4 = s
            .remember_idempotent("u", "I want a refund", e, "req-abc")
            .unwrap();
        assert_ne!(id4, id1, "expired key must not dedupe");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn remember_and_recall_roundtrip() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        s.remember(
            "u1",
            "I want a refund for my order",
            embed("refund order", DIM),
        )
        .unwrap();
        s.remember("u1", "What's the weather?", embed("weather", DIM))
            .unwrap();
        assert_eq!(s.len(), 2);

        let q = "refund";
        let hits = s.recall("u1", q, &embed(q, DIM), 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].memory.text.contains("refund"));
    }

    // -------- Phase 6.2: row cap + importance-based eviction --------

    #[test]
    fn cap_evicts_lowest_effective_importance() {
        const DIM: usize = 8;
        let s = MemoryStore::new(DIM);
        s.set_max_rows(Some(3));
        // Use a very short half-life so created_at differences dominate
        // and the *oldest* rows look least important.
        s.set_eviction_half_life(std::time::Duration::from_micros(1));

        // Insert 5 rows in chronological order.
        let id_a = s.remember("u1", "alpha", embed("alpha", DIM)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id_b = s.remember("u1", "beta", embed("beta", DIM)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _id_c = s.remember("u1", "gamma", embed("gamma", DIM)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _id_d = s.remember("u1", "delta", embed("delta", DIM)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _id_e = s.remember("u1", "epsilon", embed("epsilon", DIM)).unwrap();

        // Cap should hold the row count at 3.
        assert_eq!(s.len(), 3, "len should equal the cap");
        assert_eq!(
            s.evictions_total(),
            2,
            "two oldest rows should have been evicted"
        );

        // The first two (oldest -> lowest decayed importance) are gone.
        let by_id = s.inner.by_id.read();
        assert!(!by_id.contains_key(&id_a), "alpha should be evicted");
        assert!(!by_id.contains_key(&id_b), "beta should be evicted");
    }

    #[test]
    fn no_cap_means_unlimited() {
        const DIM: usize = 8;
        let s = MemoryStore::new(DIM);
        // Default: no cap.
        assert_eq!(s.max_rows(), None);
        for i in 0..50 {
            s.remember("u", format!("msg-{i}"), embed(&format!("m{i}"), DIM))
                .unwrap();
        }
        assert_eq!(s.len(), 50);
        assert_eq!(s.evictions_total(), 0);
    }

    #[test]
    fn evicted_rows_disappear_from_recall() {
        const DIM: usize = 8;
        let s = MemoryStore::new(DIM);
        s.set_max_rows(Some(2));
        s.set_eviction_half_life(std::time::Duration::from_micros(1));

        s.remember("u", "old wallet message", embed("wallet old", DIM))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        s.remember("u", "newer wallet message", embed("wallet new", DIM))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        s.remember("u", "third unrelated message", embed("third", DIM))
            .unwrap();

        // The oldest (id=1) should be evicted from `by_id`. recall()
        // filters its results through `by_id`, so even though the
        // HNSW + tantivy indices still reference id=1 internally, it
        // must NOT show up in the response.
        let hits = s.recall("u", "wallet", &embed("wallet", DIM), 10).unwrap();
        for h in &hits {
            assert_ne!(h.memory.id, 1, "evicted row leaked into recall");
        }
    }

    // -------- P0: deletion-safe recall (graph compaction) --------

    #[test]
    fn compact_drops_tombstones_and_preserves_recall() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        s.set_auto_compact_ratio(None); // exercise manual compact in isolation

        let mut ids = Vec::new();
        for i in 0..40 {
            let t = format!("memory number {i} about topic {}", i % 7);
            ids.push(s.remember("u", t.clone(), embed(&t, DIM)).unwrap());
        }
        assert_eq!(s.len(), 40);

        // Forget the first 20.
        for id in ids.iter().take(20) {
            assert!(s.forget(*id));
        }
        assert_eq!(s.len(), 20);
        // The graph still carries all 40 nodes — 20 are tombstones.
        assert_eq!(s.inner.vector_index.read().len(), 40);
        assert!(s.tombstone_ratio() > 0.9, "ratio={}", s.tombstone_ratio());

        let reclaimed = s.compact().unwrap();
        assert_eq!(reclaimed, 20, "20 tombstones reclaimed");
        assert_eq!(
            s.inner.vector_index.read().len(),
            20,
            "graph rebuilt to survivors only"
        );
        assert_eq!(
            s.inner.text_index.read().len(),
            20,
            "tantivy rebuilt to survivors only"
        );
        assert_eq!(s.tombstone_ratio(), 0.0);
        assert_eq!(s.compactions_total(), 1);

        // A survivor is still recallable; no forgotten row can surface.
        let survivor = ids[25];
        let m = s.inner.by_id.read().get(&survivor).cloned().unwrap();
        let hits = s.recall("u", &m.text, &m.embedding, 10).unwrap();
        assert!(
            hits.iter().any(|h| h.memory.id == survivor),
            "survivor must be recallable after compaction"
        );
        for h in &hits {
            assert!(
                !ids[..20].contains(&h.memory.id),
                "forgotten row {} leaked into recall",
                h.memory.id
            );
        }
    }

    /// Acceptance for P0: after forgetting half the corpus and compacting,
    /// recall@10 must be within 3% of a freshly-built index over the same
    /// survivors. We measure *self-recall* (can the store retrieve the exact
    /// document you query it with?) by content, so it's comparable across two
    /// stores that assign different internal ids.
    #[test]
    fn recall_after_churn_within_3pct_of_fresh_survivor_index() {
        const DIM: usize = 32;
        const N: usize = 1_000;
        let topics = [
            "refund order delivery shipping",
            "weather forecast tomorrow rain",
            "login password account security",
            "billing invoice payment receipt",
            "product feature request feedback",
            "support agent chat human",
            "tracking number package status",
            "discount promo coupon sale",
        ];
        let doc = |i: usize| format!("doc {i} about {} item code {}", topics[i % topics.len()], i);

        // Self-recall@k by content: fraction of probes whose own text is in
        // the top-k results when queried with that text + embedding.
        fn self_recall(s: &MemoryStore, probes: &[(String, Vec<f32>)], k: usize) -> f32 {
            let mut hit = 0usize;
            for (text, emb) in probes {
                let hits = s.recall("u", text, emb, k).unwrap();
                if hits.iter().any(|h| &h.memory.text == text) {
                    hit += 1;
                }
            }
            hit as f32 / probes.len() as f32
        }

        // Churned store: insert N, then forget every other row (50%).
        let churned = MemoryStore::with_capacity(DIM, N * 2);
        churned.set_auto_compact_ratio(None); // compact explicitly below
        let mut ids = Vec::with_capacity(N);
        for i in 0..N {
            let t = doc(i);
            ids.push((i, churned.remember("u", t.clone(), embed(&t, DIM)).unwrap()));
        }
        for (i, id) in &ids {
            if i % 2 == 0 {
                churned.forget(*id);
            }
        }
        let survivors: Vec<usize> = ids.iter().map(|(i, _)| *i).filter(|i| i % 2 == 1).collect();
        assert_eq!(churned.len(), survivors.len());

        // Fresh store over exactly the survivors.
        let fresh = MemoryStore::with_capacity(DIM, N * 2);
        fresh.set_auto_compact_ratio(None);
        for i in &survivors {
            let t = doc(*i);
            fresh.remember("u", t.clone(), embed(&t, DIM)).unwrap();
        }

        // Probe set: first 50 survivors, queried by their own text+embedding.
        let probes: Vec<(String, Vec<f32>)> = survivors
            .iter()
            .take(50)
            .map(|i| {
                let t = doc(*i);
                let e = embed(&t, DIM);
                (t, e)
            })
            .collect();

        let fresh_recall = self_recall(&fresh, &probes, 10);

        // Compact the churned store, then re-measure.
        let reclaimed = churned.compact().unwrap();
        assert_eq!(reclaimed, N - survivors.len(), "half the corpus reclaimed");
        assert_eq!(
            churned.inner.vector_index.read().len(),
            survivors.len(),
            "graph physically rebuilt to survivors only"
        );
        let compacted_recall = self_recall(&churned, &probes, 10);

        assert!(
            (fresh_recall - compacted_recall).abs() <= 0.03,
            "post-compact recall {compacted_recall} must be within 3% of fresh {fresh_recall}"
        );
    }

    #[test]
    fn auto_compaction_fires_when_tombstones_exceed_threshold() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        s.set_max_rows(Some(100));
        // Near-zero half-life so eviction always drops the oldest row.
        s.set_eviction_half_life(std::time::Duration::from_micros(1));
        s.set_auto_compact_ratio(Some(0.20));

        // 100 fill the cap; the next 30 each evict one, so tombstones build
        // up until the 0.20 ratio trips a rebuild that resets them.
        for i in 0..130 {
            let t = format!("doc {i} topic {}", i % 9);
            s.remember("u", t.clone(), embed(&t, DIM)).unwrap();
        }
        assert_eq!(s.len(), 100, "cap holds the live count");
        assert!(
            s.compactions_total() >= 1,
            "auto-compaction should have fired at the threshold"
        );
        assert!(
            s.tombstone_ratio() < 0.20,
            "ratio should be reset below threshold, got {}",
            s.tombstone_ratio()
        );
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_persistence_across_reopen() {
        const DIM: usize = 8;
        let dir =
            std::env::temp_dir().join(format!("duxx-memory-persist-{}", uuid::Uuid::new_v4()));
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
        let id = s.remember("u", "test", embed("test", 8)).unwrap();
        let m = s.inner.by_id.read().get(&id).cloned().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let eff = m.effective_importance(std::time::Duration::from_millis(1));
        assert!(
            eff < 0.01,
            "effective importance after many half-lives: {eff}"
        );
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
    fn open_at_persists_indices_across_restart() {
        const DIM: usize = 4;
        let dir = std::env::temp_dir().join(format!("duxx-open-at-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        // First open: write 3 memories.
        {
            let store = MemoryStore::open_at(DIM, 1_000, &dir).unwrap();
            store
                .remember(
                    "alice",
                    "I lost my wallet at the cafe",
                    embed("wallet", DIM),
                )
                .unwrap();
            store
                .remember("alice", "Favorite color is blue", embed("blue", DIM))
                .unwrap();
            store
                .remember("bob", "Tracking number DX-002341", embed("tracking", DIM))
                .unwrap();
            assert_eq!(store.len(), 3);
        } // graceful drop -> dumps HNSW + commits tantivy

        // Second open: indices should be ready immediately, no rebuild.
        {
            let store = MemoryStore::open_at(DIM, 1_000, &dir).unwrap();
            assert_eq!(store.len(), 3, "row count should match");
            // Both indices loaded from disk, no rebuild path.
            assert_eq!(
                store.inner.vector_index.read().len(),
                3,
                "hnsw should be loaded"
            );
            assert_eq!(
                store.inner.text_index.read().len(),
                3,
                "tantivy should be loaded"
            );
            // Recall still works.
            let hits = store
                .recall("alice", "wallet", &embed("wallet", DIM), 5)
                .unwrap();
            assert!(!hits.is_empty());
            assert!(hits[0].memory.text.contains("wallet"));
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn decay_continues_across_restart() {
        // Insert a memory, persist it. Reopen. Verify the loaded
        // memory's created_at_unix_ns matches the original (i.e. the
        // age grew across the reopen, didn't reset).
        const DIM: usize = 4;
        let dir = std::env::temp_dir().join(format!("duxx-decay-{}", uuid::Uuid::new_v4()));
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
