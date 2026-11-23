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
use duxx_index::text::env_weight;
use duxx_index::{TextIndex, VectorIndex};
use duxx_query::{rrf_fuse, DEFAULT_RRF_K};
use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use duxx_storage::Storage;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

pub use duxx_index::text::{DocFields, FieldBoosts};

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

/// Inserts between tantivy commits while `open_at` reloads the text index
/// (the normal [`duxx_index::text::DEFAULT_COMMIT_EVERY`] would commit, and
/// write a segment, every hundred rows of a multi-million-row reload).
const BULK_TEXT_COMMIT_EVERY: usize = 50_000;

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
    /// Importance of the row's source in `0..=1` (e.g. a site prior), given
    /// with `REMEMBER ... PRIOR p`. Recall multiplies the fused score by
    /// `1 + prior_weight * prior` (see [`MemoryStore::set_prior_weight`]).
    /// `None` counts as 0.
    pub prior: Option<f32>,
    /// Title and URL given with the row (`REMEMBER ... TITLE t URL u`), indexed
    /// as their own text fields. Fields not given here are read from the
    /// text's layout when it is indexed ([`DocFields::from_layout`]). Empty
    /// in lean rows.
    pub fields: DocFields,
}

/// What a row may carry beside its key, text and embedding.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RememberOptions {
    /// Importance of the row's source, in `0..=1`. See [`Memory::prior`].
    pub prior: Option<f32>,
    /// Title and URL. See [`Memory::fields`].
    pub fields: DocFields,
}

/// How a recall may be limited.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecallOptions {
    /// Stop searching at this instant and fuse whatever each side found by
    /// then. `None` waits for the full search.
    pub deadline: Option<Instant>,
}

/// The hits of [`MemoryStore::recall_with`], and whether the search was cut
/// short by its deadline.
#[derive(Debug, Clone, Default)]
pub struct Recall {
    pub hits: Vec<MemoryHit>,
    /// `true` when the deadline came before the vector search, the text
    /// search or the title/URL search finished; `hits` are then the best of
    /// what was found in time.
    pub timed_out: bool,
}

/// The ranked candidates of each side of a hybrid search.
#[derive(Default)]
struct Sides {
    vector: Vec<(u64, f32)>,
    text: Vec<(u64, f32)>,
    /// Title/URL-only scores (empty unless the field-match weight is on).
    fields: Vec<(u64, f32)>,
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
    /// Lean rows: `by_id` keeps each row's metadata but not its text or
    /// embedding, which stay in `storage` and are read back only for the
    /// rows a caller receives. Only ever `true` together with `storage`.
    /// See [`lean_rows_from_env`].
    lean: bool,
    /// Weight of [`Memory::prior`] in recall (`DUXX_PRIOR_WEIGHT`, default 0).
    prior_weight: RwLock<f32>,
    /// Weight of the title/URL match in recall (`DUXX_FIELD_MATCH_WEIGHT`,
    /// default 0). See [`MemoryStore::set_field_match_weight`].
    field_match_weight: RwLock<f32>,
    /// Number of live rows that carry a prior.
    priors: AtomicU64,
    /// Number of recalls cut short by their deadline.
    recall_timeouts: AtomicU64,
}

/// `DUXX_PRIOR_WEIGHT`: how much [`Memory::prior`] counts in recall. Unset
/// means 0 (priors are stored but do not change the ranking).
pub fn prior_weight_from_env() -> f32 {
    env_weight("DUXX_PRIOR_WEIGHT", 0.0)
}

/// `DUXX_FIELD_MATCH_WEIGHT`: how much the title/URL match counts in recall,
/// on top of the text side's rank. Unset means 0.
pub fn field_match_weight_from_env() -> f32 {
    env_weight("DUXX_FIELD_MATCH_WEIGHT", 0.0)
}

/// Whether the `DUXX_LEAN_ROWS` environment variable asks for lean rows
/// (`1`, `true`, `on` or `yes`). Off when unset.
///
/// With lean rows a persistent store keeps only each row's metadata in
/// memory (id, key, importance, timestamps); the text and the embedding
/// are read from the row store for the few rows a recall returns. The
/// vector and text indices are unchanged, so ranking is identical. For a
/// store of web pages this cuts the row map from several kilobytes per
/// row to about a hundred bytes, at the cost of one row-store read per
/// returned hit.
pub fn lean_rows_from_env() -> bool {
    std::env::var("DUXX_LEAN_ROWS")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// Decode one persisted row (either layout, see [`StoredExt`]).
fn decode_row(id: u64, bytes: &[u8]) -> Result<(StoredMemory, StoredExt)> {
    let mut rest: &[u8] = bytes;
    let stored: StoredMemory = bincode::deserialize_from(&mut rest)
        .map_err(|e| duxx_core::Error::Storage(format!("decode memory id={id}: {e}")))?;
    let ext = if rest.is_empty() {
        StoredExt::default()
    } else {
        bincode::deserialize(rest)
            .map_err(|e| duxx_core::Error::Storage(format!("decode memory id={id} extras: {e}")))?
    };
    Ok((stored, ext))
}

/// Decode one persisted row and check its dimension.
fn decode_stored(id: u64, bytes: &[u8], dim: usize) -> Result<(StoredMemory, StoredExt)> {
    let (stored, ext) = decode_row(id, bytes)?;
    if stored.embedding.len() != dim {
        return Err(duxx_core::Error::Storage(format!(
            "memory id={id} has dim {} but store expects {dim}",
            stored.embedding.len()
        )));
    }
    Ok((stored, ext))
}

/// Encode one row for the row store. A row with no prior, title or URL is
/// written exactly as 0.4.3 wrote it.
fn encode_row(m: &Memory) -> Result<Vec<u8>> {
    let mut bytes = bincode::serialize(&StoredMemory::from(m))
        .map_err(|e| duxx_core::Error::Storage(format!("encode memory id={}: {e}", m.id)))?;
    let ext = StoredExt::of(m);
    if !ext.is_empty() {
        let more = bincode::serialize(&ext).map_err(|e| {
            duxx_core::Error::Storage(format!("encode memory id={} extras: {e}", m.id))
        })?;
        bytes.extend_from_slice(&more);
    }
    Ok(bytes)
}

/// Whether `prior` is a valid [`Memory::prior`]: a number in `0..=1`.
pub fn valid_prior(prior: f32) -> bool {
    prior.is_finite() && (0.0..=1.0).contains(&prior)
}

impl Memory {
    fn from_stored(id: u64, stored: StoredMemory, ext: StoredExt) -> Self {
        Memory {
            id,
            key: stored.key,
            text: stored.text,
            embedding: stored.embedding,
            importance: stored.importance,
            created_at_unix_ns: stored.created_at_unix_ns,
            prior: ext.prior,
            fields: DocFields {
                title: ext.title,
                url: ext.url,
            },
        }
    }

    /// The same row without its text, embedding, title and URL (lean-rows
    /// mode). The prior stays: recall needs it for every candidate.
    fn into_lean(mut self) -> Self {
        self.text = String::new();
        self.embedding = Vec::new();
        self.fields = DocFields::default();
        self
    }
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

/// What a row carries beyond [`StoredMemory`], written after it (0.4.4 and
/// later) only when there is something to write.
///
/// bincode reads a struct field by field and ignores trailing bytes, so:
/// a row written by 0.4.3 or earlier (no extras) reads here with
/// `StoredExt::default()`, and a row written here reads in 0.4.3 as the
/// same row without its prior, title and URL. The row store needs no
/// migration in either direction. New members go at the end of this struct.
#[derive(serde::Serialize, serde::Deserialize, Default, Debug, PartialEq)]
struct StoredExt {
    prior: Option<f32>,
    title: Option<String>,
    url: Option<String>,
}

impl StoredExt {
    fn of(m: &Memory) -> Self {
        StoredExt {
            prior: m.prior,
            title: m.fields.title.clone(),
            url: m.fields.url.clone(),
        }
    }

    fn is_empty(&self) -> bool {
        self.prior.is_none() && self.title.is_none() && self.url.is_none()
    }

    fn fields(&self) -> DocFields {
        DocFields {
            title: self.title.clone(),
            url: self.url.clone(),
        }
    }
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
                lean: false,
                prior_weight: RwLock::new(prior_weight_from_env()),
                field_match_weight: RwLock::new(field_match_weight_from_env()),
                priors: AtomicU64::new(0),
                recall_timeouts: AtomicU64::new(0),
            }),
        }
    }

    /// The `Inner` of a store opened over durable storage.
    fn persistent_inner(
        dim: usize,
        by_id: HashMap<u64, Memory>,
        vector_index: VectorIndex,
        text_index: TextIndex,
        next_id: u64,
        storage: Arc<dyn Storage>,
        lean: bool,
    ) -> Inner {
        let priors = by_id.values().filter(|m| m.prior.is_some()).count() as u64;
        Inner {
            dim,
            by_id: RwLock::new(by_id),
            vector_index: RwLock::new(vector_index),
            text_index: RwLock::new(text_index),
            next_id: RwLock::new(next_id),
            bus: ChangeBus::default(),
            storage: Some(storage),
            max_rows: RwLock::new(None),
            evict_half_life: RwLock::new(Duration::from_secs(24 * 60 * 60)),
            evictions_total: std::sync::atomic::AtomicU64::new(0),
            compactions_total: std::sync::atomic::AtomicU64::new(0),
            auto_compact_ratio: RwLock::new(Some(DEFAULT_AUTO_COMPACT_RATIO)),
            lean,
            prior_weight: RwLock::new(prior_weight_from_env()),
            field_match_weight: RwLock::new(field_match_weight_from_env()),
            priors: AtomicU64::new(priors),
            recall_timeouts: AtomicU64::new(0),
        }
    }

    /// Whether rows are kept lean in memory (see [`lean_rows_from_env`]).
    pub fn is_lean(&self) -> bool {
        self.inner.lean
    }

    /// Name of the tokenizer the text index was built with: `default`, or
    /// `duxx_words` when `DUXX_TEXT_TOKENIZER=words` (see
    /// [`duxx_index::text::TextTokenizer`]).
    pub fn text_tokenizer(&self) -> &'static str {
        self.inner.text_index.read().tokenizer().name()
    }

    /// How title, URL and title-phrase matches weigh in the text side
    /// (`DUXX_TEXT_TITLE_BOOST`, `DUXX_TEXT_URL_BOOST`,
    /// `DUXX_TEXT_TITLE_PHRASE_BOOST`).
    pub fn field_boosts(&self) -> FieldBoosts {
        self.inner.text_index.read().field_boosts()
    }

    /// Change the text side's field boosts. [`FieldBoosts::OFF`] scores
    /// text as 0.4.3 did.
    pub fn set_field_boosts(&self, boosts: FieldBoosts) {
        self.inner.text_index.write().set_field_boosts(boosts);
    }

    /// Weight `w` of [`Memory::prior`]: recall multiplies each fused score by
    /// `1 + w * prior`. 0 (the default) leaves the ranking unchanged.
    pub fn prior_weight(&self) -> f32 {
        *self.inner.prior_weight.read()
    }

    /// Set the prior weight (negative or non-finite values count as 0).
    pub fn set_prior_weight(&self, w: f32) {
        *self.inner.prior_weight.write() = if w.is_finite() { w.max(0.0) } else { 0.0 };
    }

    /// Weight `w` of the title/URL match: recall multiplies each fused score
    /// by `1 + w * m`, where `m` in `0..=1` is the row's title/URL score
    /// relative to the best title/URL match for the query. Rank fusion
    /// otherwise hides how much better a title match is than a body match
    /// (it sees only the order of the text side). 0 (the default) leaves the
    /// ranking to rank fusion alone.
    pub fn field_match_weight(&self) -> f32 {
        *self.inner.field_match_weight.read()
    }

    /// Set the title/URL match weight (negative or non-finite count as 0).
    pub fn set_field_match_weight(&self, w: f32) {
        *self.inner.field_match_weight.write() = if w.is_finite() { w.max(0.0) } else { 0.0 };
    }

    /// Number of live rows that carry a prior.
    pub fn priors(&self) -> u64 {
        self.inner.priors.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of recalls cut short by their deadline since start.
    pub fn recall_timeouts_total(&self) -> u64 {
        self.inner
            .recall_timeouts
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The full row for a lean in-memory row: text and embedding read back
    /// from the row store. A row whose stored copy cannot be read comes back
    /// as it is in memory (with empty text), and the failure is logged.
    fn hydrate(&self, m: &Memory) -> Memory {
        if !self.inner.lean {
            return m.clone();
        }
        let Some(s) = &self.inner.storage else {
            return m.clone();
        };
        match s.get(m.id) {
            Ok(Some(bytes)) => match decode_row(m.id, &bytes) {
                Ok((stored, ext)) => Memory {
                    id: m.id,
                    key: m.key.clone(),
                    text: stored.text,
                    embedding: stored.embedding,
                    importance: m.importance,
                    created_at_unix_ns: m.created_at_unix_ns,
                    prior: m.prior,
                    fields: ext.fields(),
                },
                Err(e) => {
                    tracing::warn!(id = m.id, error = %e, "lean rows: stored row did not decode");
                    m.clone()
                }
            },
            Ok(None) => {
                tracing::warn!(id = m.id, "lean rows: row missing from storage");
                m.clone()
            }
            Err(e) => {
                tracing::warn!(id = m.id, error = %e, "lean rows: storage read failed");
                m.clone()
            }
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
        Self::with_storage_opts(dim, capacity, storage, lean_rows_from_env())
    }

    /// [`MemoryStore::with_storage`] with lean rows chosen explicitly
    /// instead of from `DUXX_LEAN_ROWS`.
    ///
    /// Rows are streamed from storage one at a time, so opening never holds
    /// a second copy of every row in memory.
    pub fn with_storage_opts(
        dim: usize,
        capacity: usize,
        storage: Arc<dyn Storage>,
        lean: bool,
    ) -> Result<Self> {
        let mut vector_index = VectorIndex::with_capacity(dim, capacity);
        let mut text_index = TextIndex::new();
        let mut by_id: HashMap<u64, Memory> = HashMap::with_capacity(storage.len().unwrap_or(0));
        let mut max_id = 0u64;
        storage.for_each(&mut |id, bytes| {
            let (stored, ext) = decode_stored(id, bytes, dim)?;
            vector_index.insert(id, stored.embedding.clone())?;
            text_index.insert_with(id, stored.text.clone(), &ext.fields())?;
            let mem = Memory::from_stored(id, stored, ext);
            by_id.insert(id, if lean { mem.into_lean() } else { mem });
            max_id = max_id.max(id);
            Ok(())
        })?;
        let n = by_id.len();
        let inner = Arc::new(Self::persistent_inner(
            dim,
            by_id,
            vector_index,
            text_index,
            max_id + 1,
            storage,
            lean,
        ));
        tracing::info!(memories = n, lean, "loaded memories from storage");
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
        let removed_row = self.inner.by_id.write().remove(&id);
        let removed = removed_row.is_some();
        if let Some(m) = removed_row {
            if m.prior.is_some() {
                self.inner
                    .priors
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
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
        let mut survivors_text: Vec<(u64, String, DocFields)> = Vec::new();
        {
            let by_id = self.inner.by_id.read();
            survivors_vec.reserve(by_id.len());
            survivors_text.reserve(by_id.len());
            match (&self.inner.storage, self.inner.lean) {
                // Lean rows: the text and embedding live only in storage.
                (Some(s), true) => {
                    s.for_each(&mut |id, bytes| {
                        if by_id.contains_key(&id) {
                            let (stored, ext) = decode_stored(id, bytes, self.inner.dim)?;
                            survivors_vec.push((id, stored.embedding));
                            survivors_text.push((id, stored.text, ext.fields()));
                        }
                        Ok(())
                    })?;
                }
                _ => {
                    for m in by_id.values() {
                        survivors_vec.push((m.id, m.embedding.clone()));
                        survivors_text.push((m.id, m.text.clone(), m.fields.clone()));
                    }
                }
            }
        }
        let live = survivors_vec.len();
        let reclaimed = before_indexed.saturating_sub(live);

        v.rebuild(&survivors_vec)?;
        t.rebuild_with(&survivors_text)?;

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
        let mut out: Vec<Memory> = {
            let by_id = self.inner.by_id.read();
            by_id.values().cloned().collect()
        };
        out.sort_by_key(|m| m.id);
        if self.inner.lean {
            out = out.iter().map(|m| self.hydrate(m)).collect();
        }
        out
    }

    /// Insert a new memory. `embedding.len()` must match `self.dim()`.
    pub fn remember(
        &self,
        key: impl Into<String>,
        text: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Result<u64> {
        self.remember_with(key, text, embedding, RememberOptions::default())
    }

    /// [`MemoryStore::remember`] with a prior and a title and URL. A prior
    /// outside `0..=1` is refused.
    pub fn remember_with(
        &self,
        key: impl Into<String>,
        text: impl Into<String>,
        embedding: Vec<f32>,
        opts: RememberOptions,
    ) -> Result<u64> {
        if let Some(p) = opts.prior {
            if !valid_prior(p) {
                return Err(duxx_core::Error::Schema(format!(
                    "prior must be a number from 0 to 1, got {p}"
                )));
            }
        }
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
        self.inner
            .text_index
            .write()
            .insert_with(id, text.clone(), &opts.fields)?;
        let mem = Memory {
            id,
            key: key.clone(),
            text: text.clone(),
            embedding,
            importance: 1.0,
            created_at_unix_ns: now_unix_ns(),
            prior: opts.prior,
            fields: opts.fields,
        };
        // Write through to durable storage BEFORE inserting into the
        // in-memory cache. If the persistence write fails, we don't
        // want a phantom row in memory.
        if let Some(s) = &self.inner.storage {
            s.put(id, &encode_row(&mem)?)?;
        }
        if mem.prior.is_some() {
            self.inner
                .priors
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mem = if self.inner.lean {
            mem.into_lean()
        } else {
            mem
        };
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
        Self::open_at_opts(dim, capacity, dir, lean_rows_from_env())
    }

    /// [`MemoryStore::open_at`] with lean rows chosen explicitly instead of
    /// from `DUXX_LEAN_ROWS`. Rows are streamed from the row store one at a
    /// time, so opening never holds a second copy of every row in memory.
    #[cfg(feature = "redb-store")]
    pub fn open_at_opts(
        dim: usize,
        capacity: usize,
        dir: impl AsRef<std::path::Path>,
        lean: bool,
    ) -> Result<Self> {
        use duxx_index::{TextIndex, VectorIndex};

        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| duxx_core::Error::Storage(format!("create memory dir {dir:?}: {e}")))?;

        // 1. Open the three on-disk artifacts.
        let storage: Arc<dyn Storage> =
            Arc::new(duxx_storage::RedbStorage::open(dir.join("store.redb"))?);
        let text_index = TextIndex::open(dir.join("tantivy"))?;
        let vector_index = VectorIndex::open(dim, capacity, dir.join("hnsw"))?;

        // 2. Bring each index in line with the row store, never by adding
        //    rows it already holds (that duplicated every row on each
        //    restart after a hard kill).
        //    - Vectors: keep the loaded graph and insert only the rows it is
        //      missing (rows written after the last dump, or every row when
        //      there is no dump). Nodes whose row is gone are harmless:
        //      recall drops ids that are not in `by_id`.
        //    - Text: tantivy cannot be checked row by row cheaply, so when
        //      its document count differs from the row count it is cleared
        //      and reloaded, committing in large batches.
        let row_count = storage.len()?;
        let mut text_index = text_index;
        let mut vector_index = vector_index;
        let have_vector = vector_index.ids();
        let text_count = text_index.len();
        let rebuild_text = text_count != row_count;
        if rebuild_text {
            text_index.rebuild(&[])?;
            text_index.set_commit_every(BULK_TEXT_COMMIT_EVERY);
        }

        // 3. Walk the row store once: rebuild `by_id` (never persisted on its
        //    own) and fill whatever step 2 found missing.
        let mut by_id: HashMap<u64, Memory> = HashMap::with_capacity(row_count);
        let mut max_id = 0u64;
        let mut vectors_added = 0usize;
        storage.for_each(&mut |id, bytes| {
            let (stored, ext) = decode_stored(id, bytes, dim)?;
            if !have_vector.contains(&id) {
                vector_index.insert(id, stored.embedding.clone())?;
                vectors_added += 1;
            }
            if rebuild_text {
                text_index.insert_with(id, stored.text.clone(), &ext.fields())?;
            }
            let mem = Memory::from_stored(id, stored, ext);
            by_id.insert(id, if lean { mem.into_lean() } else { mem });
            max_id = max_id.max(id);
            Ok(())
        })?;
        drop(have_vector);
        if rebuild_text {
            text_index.flush()?;
            text_index.set_commit_every(duxx_index::text::DEFAULT_COMMIT_EVERY);
        }
        tracing::info!(
            rows = row_count,
            vectors_loaded = vector_index.len() - vectors_added,
            vectors_added,
            text_docs_before = text_count,
            text_rebuilt = rebuild_text,
            lean,
            "opened memory store"
        );

        let inner = Arc::new(Self::persistent_inner(
            dim,
            by_id,
            vector_index,
            text_index,
            max_id + 1,
            storage,
            lean,
        ));
        Ok(MemoryStore { inner })
    }

    /// Hybrid recall (vector + BM25, RRF-fused). With a prior weight or a
    /// title/URL match weight set, the fused scores are then multiplied by
    /// them (see [`MemoryStore::recall_with`]).
    pub fn recall(
        &self,
        key: &str,
        query_text: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<MemoryHit>> {
        Ok(self
            .recall_with(key, query_text, query_vec, k, RecallOptions::default())?
            .hits)
    }

    /// Hybrid recall with options.
    ///
    /// Each side (vector and BM25) yields `max(3k, 32)` candidates, fused by
    /// reciprocal rank. When [`MemoryStore::prior_weight`] `wp` or
    /// [`MemoryStore::field_match_weight`] `wf` is above 0, every fused
    /// candidate's score is multiplied by `(1 + wf * field_match) * (1 + wp *
    /// prior)` and the best `k` are returned; with both at 0 this is the
    /// plain fusion of 0.4.3.
    ///
    /// With a deadline, the two sides run in parallel and the call returns
    /// at the deadline with what each side found by then (the text side
    /// stops between index segments); [`Recall::timed_out`] says so.
    pub fn recall_with(
        &self,
        _key: &str,
        query_text: &str,
        query_vec: &[f32],
        k: usize,
        opts: RecallOptions,
    ) -> Result<Recall> {
        let prior_w = self.prior_weight();
        let field_w = self.field_match_weight();
        let rerank = prior_w > 0.0 || field_w > 0.0;
        let per = (k * 3).max(32);
        let (sides, timed_out) = match opts.deadline {
            None => (
                self.search_sides(query_text, query_vec, per, field_w > 0.0),
                false,
            ),
            Some(deadline) => {
                self.search_sides_until(query_text, query_vec, per, field_w > 0.0, deadline)
            }
        };
        if timed_out {
            self.inner
                .recall_timeouts
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        tracing::debug!(
            v = sides.vector.len(),
            t = sides.text.len(),
            timed_out,
            "recall pre-fusion"
        );
        let pool = if rerank { per * 2 } else { k };
        let fused = rrf_fuse(vec![sides.vector, sides.text], DEFAULT_RRF_K, pool);

        // Title/URL match of each candidate, relative to the best one.
        let field_match: HashMap<u64, f32> = {
            let best = sides.fields.iter().map(|h| h.1).fold(0.0f32, f32::max);
            if best > 0.0 {
                sides.fields.iter().map(|(id, s)| (*id, s / best)).collect()
            } else {
                HashMap::new()
            }
        };

        let mut out = Vec::with_capacity(fused.len().min(k.max(1) * 2));
        {
            let by_id = self.inner.by_id.read();
            for h in fused {
                if let Some(m) = by_id.get(&h.id) {
                    let mut score = h.score;
                    if field_w > 0.0 {
                        score *= 1.0 + field_w * field_match.get(&h.id).copied().unwrap_or(0.0);
                    }
                    if prior_w > 0.0 {
                        score *= 1.0 + prior_w * m.prior.unwrap_or(0.0);
                    }
                    out.push(MemoryHit {
                        memory: m.clone(),
                        score,
                    });
                }
            }
        }
        if rerank {
            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.memory.id.cmp(&b.memory.id))
            });
            out.truncate(k);
        }
        if self.inner.lean {
            // Read the returned rows' text and embedding back from storage,
            // after letting go of every lock.
            for hit in &mut out {
                hit.memory = self.hydrate(&hit.memory);
            }
        }
        Ok(Recall {
            hits: out,
            timed_out,
        })
    }

    /// Both sides of a hybrid search, run one after the other.
    fn search_sides(
        &self,
        query_text: &str,
        query_vec: &[f32],
        per: usize,
        with_fields: bool,
    ) -> Sides {
        let v = self.inner.vector_index.read();
        let t = self.inner.text_index.read();
        Sides {
            vector: v.search(query_vec, per),
            text: t.search(query_text, per),
            fields: if with_fields {
                t.field_search(query_text, per)
            } else {
                Vec::new()
            },
        }
    }

    /// Both sides of a hybrid search, run in parallel on their own threads,
    /// taking what each has found at `deadline`. A side still running then
    /// finishes in the background (the text side stops at its next segment)
    /// and its result is dropped. Returns `true` with the sides when any
    /// part was cut short.
    fn search_sides_until(
        &self,
        query_text: &str,
        query_vec: &[f32],
        per: usize,
        with_fields: bool,
        deadline: Instant,
    ) -> (Sides, bool) {
        if Instant::now() >= deadline {
            return (Sides::default(), true);
        }
        let (vector_tx, vector_rx) = std::sync::mpsc::channel();
        let store = self.clone();
        let qvec = query_vec.to_vec();
        let vector_thread = std::thread::Builder::new()
            .name("duxx-recall-vector".into())
            .spawn(move || {
                let hits = store.inner.vector_index.read().search(&qvec, per);
                let _ = vector_tx.send(hits);
            });

        let found: Arc<Mutex<Vec<(u64, f32)>>> = Arc::new(Mutex::new(Vec::new()));
        let (text_tx, text_rx) = std::sync::mpsc::channel();
        let store = self.clone();
        let query = query_text.to_string();
        let text_found = Arc::clone(&found);
        let text_thread = std::thread::Builder::new()
            .name("duxx-recall-text".into())
            .spawn(move || {
                let t = store.inner.text_index.read();
                let complete = t.search_until(&query, per, deadline, &text_found);
                let fields = if complete && with_fields && Instant::now() < deadline {
                    Some(t.field_search(&query, per))
                } else {
                    None
                };
                let _ = text_tx.send((complete, fields));
            });
        if vector_thread.is_err() || text_thread.is_err() {
            // No thread to be had: search in this one, without a deadline.
            tracing::warn!("recall: could not start search threads; searching without a deadline");
            return (
                self.search_sides(query_text, query_vec, per, with_fields),
                false,
            );
        }

        let mut timed_out = false;
        let left = || deadline.saturating_duration_since(Instant::now());
        let vector = match vector_rx.recv_timeout(left()) {
            Ok(hits) => hits,
            Err(_) => {
                timed_out = true;
                Vec::new()
            }
        };
        let fields = match text_rx.recv_timeout(left()) {
            Ok((complete, fields)) => {
                if !complete || (with_fields && fields.is_none()) {
                    timed_out = true;
                }
                fields.unwrap_or_default()
            }
            Err(_) => {
                timed_out = true;
                Vec::new()
            }
        };
        let mut text = std::mem::take(&mut *found.lock());
        text.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        text.truncate(per);
        (
            Sides {
                vector,
                text,
                fields,
            },
            timed_out,
        )
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
            prior: None,
            fields: DocFields::default(),
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

    /// Every row in `by_id` is lean (no text, no embedding).
    fn all_rows_lean(s: &MemoryStore) -> bool {
        s.inner
            .by_id
            .read()
            .values()
            .all(|m| m.text.is_empty() && m.embedding.is_empty())
    }

    const LEAN_TEXTS: [&str; 6] = [
        "I lost my wallet at the cafe",
        "Favorite color is blue",
        "Tracking number DX-002341",
        "The train to Pune leaves at nine",
        "Monsoon rainfall in Kerala this week",
        "Railway recruitment results announced",
    ];

    #[cfg(feature = "redb-store")]
    #[test]
    fn lean_open_at_keeps_text_on_disk_and_recall_returns_full_rows() {
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-lean-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
            assert!(store.is_lean());
            for t in LEAN_TEXTS {
                store.remember("web", t, embed(t, DIM)).unwrap();
            }
            assert_eq!(store.len(), LEAN_TEXTS.len());
            assert!(all_rows_lean(&store), "rows in memory must be lean");
            let hits = store
                .recall("web", "wallet", &embed("lost my wallet", DIM), 3)
                .unwrap();
            assert!(!hits.is_empty());
            assert_eq!(hits[0].memory.text, "I lost my wallet at the cafe");
            assert_eq!(hits[0].memory.embedding.len(), DIM);
            assert!(all_rows_lean(&store), "recall must not fill rows back in");
        }

        // Reopen lean: the same answers, still lean.
        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
            assert_eq!(store.len(), LEAN_TEXTS.len());
            assert!(all_rows_lean(&store));
            let hits = store
                .recall("web", "monsoon", &embed("monsoon rainfall", DIM), 3)
                .unwrap();
            assert_eq!(hits[0].memory.text, "Monsoon rainfall in Kerala this week");
            let all = store.all_memories();
            assert_eq!(all.len(), LEAN_TEXTS.len());
            assert!(all
                .iter()
                .all(|m| !m.text.is_empty() && m.embedding.len() == DIM));
        }

        // The same directory opens normally too: lean is a memory layout, not a file format.
        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
            assert!(!store.is_lean());
            assert!(store
                .inner
                .by_id
                .read()
                .values()
                .all(|m| !m.text.is_empty() && m.embedding.len() == DIM));
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn lean_and_full_rank_identically() {
        // The same directory (same HNSW dump, same tantivy index) opened in full
        // mode and in lean mode must return the same rows with the same scores.
        // Rows with equal scores come back in no fixed order (the fusion step
        // does not break ties), so both sides are ordered by score, then id.
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-lean-rank-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let queries = ["wallet cafe", "train Pune", "railway results", "blue"];
        type Row = (u64, String, usize, f32);
        let summary = |s: &MemoryStore, q: &str| -> Vec<Row> {
            let mut rows: Vec<Row> = s
                .recall("web", q, &embed(q, DIM), 10)
                .unwrap()
                .into_iter()
                .map(|h| {
                    (
                        h.memory.id,
                        h.memory.text,
                        h.memory.embedding.len(),
                        h.score,
                    )
                })
                .collect();
            rows.sort_by(|a, b| {
                b.3.partial_cmp(&a.3)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            rows
        };
        let full_results: Vec<Vec<Row>> = {
            let full = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
            for i in 0..40 {
                let t = format!("{} note {i}", LEAN_TEXTS[i % LEAN_TEXTS.len()]);
                full.remember("web", t.clone(), embed(&t, DIM)).unwrap();
            }
            drop(full); // graceful: dumps HNSW, commits tantivy
            let full = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
            queries.iter().map(|q| summary(&full, q)).collect()
        };
        let lean = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
        for (q, expected) in queries.iter().zip(full_results) {
            assert!(!expected.is_empty());
            assert_eq!(
                summary(&lean, q),
                expected,
                "ranking or rows changed for {q}"
            );
        }
        drop(lean);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn lean_compact_rebuilds_from_storage() {
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-lean-compact-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
        store.set_auto_compact_ratio(None);
        let mut ids = Vec::new();
        for i in 0..80 {
            let t = format!("{} item {i}", LEAN_TEXTS[i % LEAN_TEXTS.len()]);
            ids.push(store.remember("web", t.clone(), embed(&t, DIM)).unwrap());
        }
        for id in ids.iter().take(30) {
            assert!(store.forget(*id));
        }
        let reclaimed = store.compact().unwrap();
        assert_eq!(reclaimed, 30);
        assert_eq!(store.inner.vector_index.read().len(), 50);
        assert!(all_rows_lean(&store));
        let hits = store
            .recall(
                "web",
                "railway",
                &embed("Railway recruitment results announced", DIM),
                5,
            )
            .unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| !h.memory.text.is_empty()));
        assert!(hits.iter().all(|h| !ids[..30].contains(&h.memory.id)));
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn lean_with_storage_roundtrip() {
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-lean-ws-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.redb");
        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage_opts(DIM, 1_000, storage, true).unwrap();
            for t in LEAN_TEXTS {
                store.remember("web", t, embed(t, DIM)).unwrap();
            }
            assert!(all_rows_lean(&store));
        }
        {
            let storage: Arc<dyn Storage> =
                Arc::new(duxx_storage::RedbStorage::open(&path).unwrap());
            let store = MemoryStore::with_storage_opts(DIM, 1_000, storage, true).unwrap();
            assert_eq!(store.len(), LEAN_TEXTS.len());
            assert!(all_rows_lean(&store));
            let hits = store
                .recall(
                    "web",
                    "tracking",
                    &embed("Tracking number DX-002341", DIM),
                    3,
                )
                .unwrap();
            assert_eq!(hits[0].memory.text, "Tracking number DX-002341");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn open_at_clears_a_text_index_that_does_not_match_the_rows() {
        // A hard kill used to leave tantivy with more documents than rows, and
        // every later open re-added every row on top. Now a mismatched text
        // index is cleared and reloaded, and the vector graph is left alone.
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-text-dupes-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
            for t in LEAN_TEXTS {
                store.remember("web", t, embed(t, DIM)).unwrap();
            }
        }
        // Simulate the old duplication: every row indexed twice.
        {
            let mut t = TextIndex::open(dir.join("tantivy")).unwrap();
            for (i, text) in LEAN_TEXTS.iter().enumerate() {
                t.insert(i as u64 + 1, text.to_string()).unwrap();
            }
            t.flush().unwrap();
            assert_eq!(t.len(), LEAN_TEXTS.len() * 2);
        }
        let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
        assert_eq!(store.inner.text_index.read().len(), LEAN_TEXTS.len());
        assert_eq!(store.inner.vector_index.read().len(), LEAN_TEXTS.len());
        let hits = store
            .recall("web", "wallet", &embed("lost my wallet", DIM), 3)
            .unwrap();
        assert_eq!(hits[0].memory.text, "I lost my wallet at the cafe");
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn open_at_adds_only_the_vectors_a_dump_is_missing() {
        // Rows written after the last HNSW dump (a hard kill) are added on
        // open; rows the dump already holds are not inserted again.
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-vec-missing-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
            for t in LEAN_TEXTS {
                store.remember("web", t, embed(t, DIM)).unwrap();
            }
        } // graceful: the dump holds every row so far
        let late = "Late row written after the dump";
        {
            let s = duxx_storage::RedbStorage::open(dir.join("store.redb")).unwrap();
            let row = StoredMemory {
                id: 100,
                key: "web".into(),
                text: late.into(),
                embedding: embed(late, DIM),
                importance: 1.0,
                created_at_unix_ns: 1,
            };
            s.put(100, &bincode::serialize(&row).unwrap()).unwrap();
        }
        let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, false).unwrap();
        assert_eq!(store.len(), LEAN_TEXTS.len() + 1);
        assert_eq!(
            store.inner.vector_index.read().len(),
            LEAN_TEXTS.len() + 1,
            "one vector added, none duplicated"
        );
        assert_eq!(store.inner.text_index.read().len(), LEAN_TEXTS.len() + 1);
        let hits = store
            .recall("web", "late row", &embed(late, DIM), 3)
            .unwrap();
        assert_eq!(hits[0].memory.id, 100);
        drop(store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn in_memory_store_is_never_lean() {
        let s = MemoryStore::new(8);
        assert!(!s.is_lean());
        s.remember("u", "kept in full", embed("kept", 8)).unwrap();
        assert!(!all_rows_lean(&s));
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

    // -------- 0.4.4: title/URL fields, priors, recall deadline --------

    /// A page as Gaelxe writes it: URL, title, description, marker, body.
    fn page(url: &str, title: &str, description: &str, body: &str) -> String {
        format!("{url}\n{title}\n{description}\ngxm20261109Len\n{body}")
    }

    /// One site: the home page titled "Wipro", and two deep pages that repeat
    /// the name in their body (so both the text side and the bag-of-words
    /// vector side prefer them) but not in their title.
    fn wipro_pages() -> [String; 3] {
        [
            page(
                "https://www.wipro.com/",
                "Wipro",
                "Technology services and consulting company helping clients build better businesses",
                "Explore our services, industries, insights and careers across the world.",
            ),
            page(
                "https://www.wipro.com/investors/annual-report/",
                "Annual report 2025",
                "Annual report",
                "wipro revenue grew. wipro margins held. wipro board met. wipro paid a dividend.",
            ),
            page(
                "https://www.wipro.com/careers/india/",
                "Careers in India",
                "Jobs",
                "join wipro. wipro hires freshers. wipro campus drives. wipro offices.",
            ),
        ]
    }

    #[test]
    fn page_titled_with_the_query_comes_first() {
        const DIM: usize = 64;
        let s = MemoryStore::new(DIM);
        let mut ids = Vec::new();
        for t in wipro_pages() {
            ids.push(s.remember("web", t.clone(), embed(&t, DIM)).unwrap());
        }
        let home = ids[0];
        let q = "wipro";

        // 0.4.3 ranking: no field boosts, no field-match weight.
        s.set_field_boosts(FieldBoosts::OFF);
        s.set_field_match_weight(0.0);
        let before = s.recall("web", q, &embed(q, DIM), 3).unwrap();
        assert_eq!(before.len(), 3);
        assert_ne!(
            before[0].memory.id, home,
            "body mentions win without fields"
        );

        // Title and URL fields on, and their match weighed into the fusion.
        s.set_field_boosts(FieldBoosts::default());
        s.set_field_match_weight(1.0);
        let after = s.recall("web", q, &embed(q, DIM), 3).unwrap();
        assert_eq!(after.len(), 3);
        assert_eq!(after[0].memory.id, home, "{after:?}");
        assert!(after[0].score > after[1].score);
    }

    #[test]
    fn prior_lifts_an_otherwise_equal_row() {
        const DIM: usize = 32;
        let s = MemoryStore::new(DIM);
        let text = "india post tracking consignment status";
        let low = s
            .remember_with(
                "web",
                text,
                embed(text, DIM),
                RememberOptions {
                    prior: Some(0.1),
                    ..Default::default()
                },
            )
            .unwrap();
        let high = s
            .remember_with(
                "web",
                text,
                embed(text, DIM),
                RememberOptions {
                    prior: Some(0.9),
                    ..Default::default()
                },
            )
            .unwrap();
        let plain = s.remember("web", text, embed(text, DIM)).unwrap();
        assert_eq!(s.priors(), 2);
        let q = "india post tracking";

        // Weight 0 (the default): priors are stored but change nothing.
        assert_eq!(s.prior_weight(), 0.0);
        let unweighted = s.recall("web", q, &embed(q, DIM), 3).unwrap();
        assert_eq!(unweighted.len(), 3);

        s.set_prior_weight(0.6);
        let hits = s.recall("web", q, &embed(q, DIM), 3).unwrap();
        let order: Vec<u64> = hits.iter().map(|h| h.memory.id).collect();
        assert_eq!(order, vec![high, low, plain], "{hits:?}");
        assert_eq!(hits[0].memory.prior, Some(0.9));
        // The three rows fuse to nearly the same score, so the prior decides.
        assert!(hits[0].score > hits[1].score * 1.3, "{hits:?}");

        // Forgetting a row with a prior updates the count.
        assert!(s.forget(high));
        assert_eq!(s.priors(), 1);
    }

    #[test]
    fn prior_outside_zero_to_one_is_refused() {
        let s = MemoryStore::new(8);
        for bad in [-0.1f32, 1.5, f32::NAN, f32::INFINITY] {
            let r = s.remember_with(
                "u",
                "text",
                embed("text", 8),
                RememberOptions {
                    prior: Some(bad),
                    ..Default::default()
                },
            );
            assert!(r.is_err(), "{bad} accepted");
        }
        assert_eq!(s.len(), 0);
        assert!(valid_prior(0.0) && valid_prior(1.0));
    }

    #[test]
    fn rows_without_extras_are_written_as_before() {
        let m = Memory {
            id: 7,
            key: "k".into(),
            text: "hello".into(),
            embedding: vec![0.5; 4],
            importance: 1.0,
            created_at_unix_ns: 42,
            prior: None,
            fields: DocFields::default(),
        };
        // Byte for byte the 0.4.3 row.
        let old = bincode::serialize(&StoredMemory::from(&m)).unwrap();
        assert_eq!(encode_row(&m).unwrap(), old);
        let (stored, ext) = decode_row(7, &old).unwrap();
        assert_eq!(stored.text, "hello");
        assert_eq!(ext, StoredExt::default());

        // A row with extras: 0.4.3's decoder (plain bincode, trailing bytes
        // allowed) still reads it, and this version reads the extras back.
        let rich = Memory {
            prior: Some(0.75),
            fields: DocFields {
                title: Some("Hello".into()),
                url: Some("https://hello.example/".into()),
            },
            ..m
        };
        let bytes = encode_row(&rich).unwrap();
        let as_043: StoredMemory = bincode::deserialize(&bytes).unwrap();
        assert_eq!(as_043.text, "hello");
        assert_eq!(as_043.created_at_unix_ns, 42);
        let (_, ext) = decode_row(7, &bytes).unwrap();
        assert_eq!(ext.prior, Some(0.75));
        assert_eq!(ext.title.as_deref(), Some("Hello"));
        assert_eq!(ext.url.as_deref(), Some("https://hello.example/"));
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn prior_and_fields_survive_reopen_and_compaction() {
        const DIM: usize = 16;
        let dir = std::env::temp_dir().join(format!("duxx-prior-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let fields = DocFields {
            title: Some("Digilocker login".into()),
            url: Some("https://www.digilocker.gov.in/".into()),
        };
        let rich_id;
        {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, true).unwrap();
            store.set_auto_compact_ratio(None);
            for t in LEAN_TEXTS {
                store.remember("web", t, embed(t, DIM)).unwrap();
            }
            rich_id = store
                .remember_with(
                    "web",
                    "Sign in to your documents",
                    embed("sign in documents", DIM),
                    RememberOptions {
                        prior: Some(0.8),
                        fields: fields.clone(),
                    },
                )
                .unwrap();
            assert_eq!(store.priors(), 1);
            assert!(store.forget(1));
            assert_eq!(store.compact().unwrap(), 1);
            let t = store.inner.text_index.read();
            assert_eq!(
                t.field_search("login", 5).first().map(|h| h.0),
                Some(rich_id)
            );
        }
        for lean in [true, false] {
            let store = MemoryStore::open_at_opts(DIM, 1_000, &dir, lean).unwrap();
            assert_eq!(store.priors(), 1, "lean={lean}");
            let m = store.inner.by_id.read().get(&rich_id).cloned().unwrap();
            assert_eq!(m.prior, Some(0.8), "lean={lean}");
            let all = store.all_memories();
            let full = all.iter().find(|m| m.id == rich_id).unwrap();
            assert_eq!(full.fields, fields, "lean={lean}");
            assert_eq!(full.prior, Some(0.8));
            let t = store.inner.text_index.read();
            assert_eq!(
                t.field_search("digilocker", 5).first().map(|h| h.0),
                Some(rich_id),
                "lean={lean}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recall_deadline_returns_what_was_found_in_time() {
        const DIM: usize = 16;
        let s = MemoryStore::new(DIM);
        for i in 0..200 {
            let t = format!("{} entry {i}", LEAN_TEXTS[i % LEAN_TEXTS.len()]);
            s.remember("web", t.clone(), embed(&t, DIM)).unwrap();
        }
        // The query is one row's own text, so that row is first either way
        // (many other rows tie with each other, and ties have no fixed order).
        let q = format!("{} entry 4", LEAN_TEXTS[4]);
        let q = q.as_str();
        let qv = embed(q, DIM);
        let plain = s.recall("web", q, &qv, 10).unwrap();
        assert_eq!(plain.len(), 10);
        assert_eq!(plain[0].memory.text, q);

        // Plenty of time: the same rows, not cut short.
        let roomy = s
            .recall_with(
                "web",
                q,
                &qv,
                10,
                RecallOptions {
                    deadline: Some(Instant::now() + Duration::from_secs(30)),
                },
            )
            .unwrap();
        assert!(!roomy.timed_out);
        assert_eq!(roomy.hits.len(), 10);
        assert_eq!(roomy.hits[0].memory.id, plain[0].memory.id);
        assert_eq!(s.recall_timeouts_total(), 0);

        // A deadline already passed: no error, nothing found, counted.
        let late = s
            .recall_with(
                "web",
                q,
                &qv,
                10,
                RecallOptions {
                    deadline: Some(Instant::now()),
                },
            )
            .unwrap();
        assert!(late.timed_out);
        assert!(late.hits.is_empty());
        assert_eq!(s.recall_timeouts_total(), 1);

        // A short deadline returns promptly, with at most k rows, whatever
        // each side managed in time.
        let start = Instant::now();
        let short = s
            .recall_with(
                "web",
                q,
                &qv,
                10,
                RecallOptions {
                    deadline: Some(start + Duration::from_millis(5)),
                },
            )
            .unwrap();
        assert!(short.hits.len() <= 10);
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
