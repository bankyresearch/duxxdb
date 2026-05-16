//! # `duxx-datasets` — versioned eval datasets (Phase 7.3)
//!
//! Datasets in agent stacks are usually flat JSONL files at rest and
//! ad-hoc Python lists at run time. That makes them awkward to
//! version, share across machines, query, and feed into eval runs.
//! `duxx-datasets` makes them a first-class primitive next to
//! `MEMORY` / `TOOL_CACHE` / `SESSION` / `TRACE` / `PROMPT`.
//!
//! ## Capabilities
//!
//! - **Versioned snapshots** — every `add` produces a new immutable
//!   version. Older versions stay queryable forever. Versions are
//!   never reused on delete (audit-safe).
//! - **Per-row splits** — `train` / `eval` / `test` (or any custom
//!   name). Sample uniformly within a split.
//! - **Semantic search across rows** — uses the same embedder the
//!   rest of DuxxDB does. "Find me eval rows that look like this
//!   failing trace" works because failures, datasets, and memories
//!   compete in one shared vector space. Nobody else ships this.
//! - **Tag aliases** — `golden` / `staging` / `experimental` to swap
//!   eval baselines without changing eval-runner code.
//! - **Reactive change feed** — `PSUBSCRIBE dataset.*` so eval
//!   workers re-run automatically when a tagged dataset moves.
//!
//! ## What's NOT in this crate (yet)
//!
//! - **CSV / JSONL / Parquet import on the wire** — caller decodes
//!   the file and pushes rows in. Wire-level import lands in 7.3b
//!   once the disk-backed store ships.
//! - **Dedupe-on-add via cosine ≥ 0.95** — designed, not built.
//!   Phase 7.3b.
//! - **Persistence** — same redb + tantivy pattern as `duxx-memory`'s
//!   `dir:` backend. Phase 7.3b.

use duxx_embed::Embedder;
use duxx_index::vector::VectorIndex;
use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use duxx_storage::{Backend, BatchOp, MemoryBackend, key};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::broadcast;

/// Backend table names.
mod tables {
    /// `(name, version_be)` -> JSON(Dataset)
    pub const DATASETS: &str = "dataset.versions";
    /// `(name, tag)` -> version_be
    pub const TAGS: &str = "dataset.tags";
    /// `name` -> next_version_be
    pub const NEXT_VERSION: &str = "dataset.next_version";
    /// `name` -> JSON(schema)
    pub const SCHEMAS: &str = "dataset.schemas";
    /// `internal_id_be8` -> JSON((name, version, row_id))
    ///
    /// Lets ``open_with_index_dir`` resurrect the catalog-wide
    /// id-to-key map without re-embedding every row. New in v0.2.2.
    pub const ID_TO_KEY: &str = "dataset.id_to_key";
}

#[inline]
fn u64_to_be(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

#[inline]
fn be_to_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() != 8 {
        return None;
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(bytes);
    Some(u64::from_be_bytes(arr))
}

/// Errors surfaced by the dataset registry.
#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("dataset {name:?} version {version} not found")]
    VersionNotFound { name: String, version: u64 },
    #[error("dataset {name:?} not found")]
    NameNotFound { name: String },
    #[error("tag {tag:?} not found on dataset {name:?}")]
    TagNotFound { name: String, tag: String },
    #[error("embedder error: {0}")]
    Embed(String),
}

pub type Result<T> = std::result::Result<T, DatasetError>;

// ---------------------------------------------------------------- Types

/// Identifier for a dataset.
pub type DatasetName = String;
/// Monotonic version number assigned by the registry.
pub type DatasetVersion = u64;
/// Symbolic alias for a version (e.g. `"golden"`, `"staging"`).
pub type DatasetTag = String;

/// A canonical split name. Strings let callers invent custom splits.
pub const SPLIT_TRAIN: &str = "train";
pub const SPLIT_EVAL: &str = "eval";
pub const SPLIT_TEST: &str = "test";

/// One row in a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRow {
    /// Stable id within this dataset version. Caller-provided or
    /// auto-assigned at `add` time. When deserializing rows from JSON
    /// (e.g. via `DATASET.ADD rows_json`), omitting `id` triggers a
    /// fresh UUID assignment.
    #[serde(default = "new_row_id")]
    pub id: String,
    /// Canonical text representation. This is what gets embedded for
    /// semantic search. Typically the input field (`question`,
    /// `prompt`, `query`, …). For multimodal datasets, the caller
    /// can serialize a description here.
    pub text: String,
    /// Full structured payload. Free-form JSON — the schema is set
    /// at dataset-creation time but not enforced here.
    #[serde(default)]
    pub data: serde_json::Value,
    /// Which split this row belongs to. Empty string means "no split"
    /// (use the whole dataset).
    #[serde(default)]
    pub split: String,
    /// Labels, scores, reviewer notes, etc. Free-form JSON.
    #[serde(default)]
    pub annotations: serde_json::Value,
}

/// Default factory for [`DatasetRow::id`] when deserializing rows that
/// omit the field.
fn new_row_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

impl DatasetRow {
    /// Build a row with an auto-assigned UUID and no annotations.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: new_row_id(),
            text: text.into(),
            data: serde_json::Value::Null,
            split: String::new(),
            annotations: serde_json::Value::Null,
        }
    }

    /// Builder helper: attach the structured payload.
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = data;
        self
    }

    /// Builder helper: place this row in a split.
    pub fn with_split(mut self, split: impl Into<String>) -> Self {
        self.split = split.into();
        self
    }

    /// Builder helper: attach annotations.
    pub fn with_annotations(mut self, annotations: serde_json::Value) -> Self {
        self.annotations = annotations;
        self
    }
}

/// One versioned dataset. Cheaply clonable (Vec of rows + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: DatasetName,
    pub version: DatasetVersion,
    /// Optional schema hint (column names + types). Free-form JSON.
    #[serde(default)]
    pub schema: serde_json::Value,
    pub rows: Vec<DatasetRow>,
    /// Tags currently pointing at this exact version.
    #[serde(default)]
    pub tags: Vec<DatasetTag>,
    /// Free-form metadata. Useful keys: `description`, `source`,
    /// `created_by`, `num_train`, `num_eval`, `num_test`.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_unix_ns: u128,
}

/// One result returned by [`DatasetRegistry::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetHit {
    pub dataset: DatasetName,
    pub version: DatasetVersion,
    pub row: DatasetRow,
    /// Cosine similarity in `[0, 1]`.
    pub score: f32,
}

/// Counters returned by [`DatasetRegistry::stats`].
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct DatasetStats {
    pub names: usize,
    pub versions: usize,
    pub rows: usize,
    pub tags: usize,
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- Registry

/// In-process dataset registry. Cheaply clonable (Arc internals).
#[derive(Clone)]
pub struct DatasetRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    /// (name, version) -> Dataset
    by_id: RwLock<HashMap<(DatasetName, DatasetVersion), Dataset>>,
    /// name -> ordered live versions.
    versions: RwLock<HashMap<DatasetName, Vec<DatasetVersion>>>,
    /// Monotonic per-name counter. Never decreases on delete.
    next_version: RwLock<HashMap<DatasetName, DatasetVersion>>,
    /// (name, tag) -> version.
    tags: RwLock<HashMap<(DatasetName, DatasetTag), DatasetVersion>>,
    /// name -> optional schema hint, persists across versions.
    schemas: RwLock<HashMap<DatasetName, serde_json::Value>>,
    /// Shared embedder.
    embedder: Arc<dyn Embedder>,
    /// Catalog-wide vector index over every row in every version.
    vector_index: RwLock<VectorIndex>,
    /// Internal HNSW id -> (name, version, row_id) for hit resolution.
    id_to_key: RwLock<HashMap<u64, (DatasetName, DatasetVersion, String)>>,
    next_internal_id: RwLock<u64>,
    bus: ChangeBus,
    /// Durable backend. Always present; v0.1.x semantics preserved
    /// by plugging in a fresh ``MemoryBackend`` from ``new``.
    backend: Arc<dyn Backend>,
}

impl std::fmt::Debug for DatasetRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("DatasetRegistry")
            .field("names", &s.names)
            .field("versions", &s.versions)
            .field("rows", &s.rows)
            .field("tags", &s.tags)
            .finish()
    }
}

impl DatasetRegistry {
    /// Build a non-persistent registry. Equivalent to
    /// ``open(embedder, MemoryBackend::new())``.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        Self::open(embedder, backend).expect("MemoryBackend cannot fail open")
    }

    /// Build a registry backed by the given persistence layer.
    /// Re-embeds every row across every version to rebuild the
    /// catalog-wide HNSW on open.
    ///
    /// For million-row workloads or when restart latency matters,
    /// use [`DatasetRegistry::open_with_index_dir`] to plug in a
    /// directory that holds the persisted HNSW dump.
    pub fn open(embedder: Arc<dyn Embedder>, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::open_with_index_dir(embedder, backend, None)
    }

    /// Like [`DatasetRegistry::open`] but persists the HNSW dump
    /// under ``vector_index_dir``. On graceful drop the dump is
    /// written back; on the next open it's loaded directly and the
    /// re-embedding rebuild is skipped. New in v0.2.2.
    pub fn open_with_index_dir(
        embedder: Arc<dyn Embedder>,
        backend: Arc<dyn Backend>,
        vector_index_dir: Option<&Path>,
    ) -> Result<Self> {
        let dim = embedder.dim();
        let vector_index = match vector_index_dir {
            Some(dir) => VectorIndex::open(dim, 100_000, dir)
                .map_err(|e| DatasetError::Embed(format!("vector index open: {e}")))?,
            None => VectorIndex::with_capacity(dim, 100_000),
        };
        let me = Self {
            inner: Arc::new(Inner {
                by_id: RwLock::new(HashMap::new()),
                versions: RwLock::new(HashMap::new()),
                next_version: RwLock::new(HashMap::new()),
                tags: RwLock::new(HashMap::new()),
                schemas: RwLock::new(HashMap::new()),
                embedder,
                vector_index: RwLock::new(vector_index),
                id_to_key: RwLock::new(HashMap::new()),
                next_internal_id: RwLock::new(1),
                bus: ChangeBus::default(),
                backend,
            }),
        };
        me.rehydrate()?;
        Ok(me)
    }

    /// Flush the HNSW vector index to disk if a persist directory
    /// was configured. No-op otherwise.
    pub fn flush_indices(&self) -> Result<()> {
        self.inner
            .vector_index
            .read()
            .dump()
            .map(|_| ())
            .map_err(|e| DatasetError::Embed(format!("vector index dump: {e}")))
    }

    fn persist_id_to_key(
        &self,
        internal_id: u64,
        name: &str,
        version: u64,
        row_id: &str,
    ) -> Result<()> {
        let payload = serde_json::to_vec(&(name, version, row_id))
            .map_err(|e| DatasetError::Embed(format!("id_to_key encode: {e}")))?;
        self.inner
            .backend
            .put(tables::ID_TO_KEY, &internal_id.to_be_bytes(), &payload)
            .map_err(|e| DatasetError::Embed(format!("backend put id_to_key: {e}")))
    }

    fn rehydrate(&self) -> Result<()> {
        // Fast path: HNSW dump was loaded from disk by VectorIndex::open.
        let skip_reembed = self.inner.vector_index.read().was_loaded_from_disk();

        // datasets
        let rows = self
            .inner
            .backend
            .scan(tables::DATASETS)
            .map_err(|e| DatasetError::Embed(format!("backend scan datasets: {e}")))?;
        for (_k, value_bytes) in rows {
            let dataset: Dataset = serde_json::from_slice(&value_bytes)
                .map_err(|e| DatasetError::Embed(format!("dataset decode: {e}")))?;
            let key_tuple = (dataset.name.clone(), dataset.version);
            // SLOW PATH ONLY: re-embed every row to rebuild the HNSW.
            // The fast-load path restores id_to_key directly from the
            // backend below.
            if !skip_reembed {
                for row in &dataset.rows {
                    let emb = self
                        .inner
                        .embedder
                        .embed(&row.text)
                        .map_err(|e| DatasetError::Embed(e.to_string()))?;
                    let internal_id = self.next_internal_id_bump();
                    self.inner
                        .vector_index
                        .write()
                        .insert(internal_id, emb)
                        .map_err(|e| DatasetError::Embed(format!("vector index: {e}")))?;
                    self.inner.id_to_key.write().insert(
                        internal_id,
                        (dataset.name.clone(), dataset.version, row.id.clone()),
                    );
                    self.persist_id_to_key(internal_id, &dataset.name, dataset.version, &row.id)
                        .ok();
                }
            }
            self.inner.by_id.write().insert(
                key_tuple.clone(),
                Dataset {
                    tags: Vec::new(),
                    ..dataset.clone()
                },
            );
            self.inner
                .versions
                .write()
                .entry(dataset.name.clone())
                .or_default()
                .push(dataset.version);
        }
        for list in self.inner.versions.write().values_mut() {
            list.sort_unstable();
        }

        // FAST PATH: rebuild id_to_key from its dedicated table.
        if skip_reembed {
            let id_rows = self
                .inner
                .backend
                .scan(tables::ID_TO_KEY)
                .map_err(|e| DatasetError::Embed(format!("backend scan id_to_key: {e}")))?;
            let mut max_internal = 0u64;
            for (key_bytes, value_bytes) in id_rows {
                let internal_id = match be_to_u64(&key_bytes) {
                    Some(v) => v,
                    None => continue,
                };
                let parsed: (String, u64, String) =
                    match serde_json::from_slice(&value_bytes) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                if internal_id > max_internal {
                    max_internal = internal_id;
                }
                self.inner.id_to_key.write().insert(internal_id, parsed);
            }
            if max_internal > 0 {
                let mut n = self.inner.next_internal_id.write();
                *n = max_internal + 1;
            }
        }
        // tags
        let tags = self
            .inner
            .backend
            .scan(tables::TAGS)
            .map_err(|e| DatasetError::Embed(format!("backend scan tags: {e}")))?;
        for (key_bytes, value_bytes) in tags {
            let mut parts = key_bytes.splitn(2, |b| *b == 0);
            let name = match parts.next().map(|b| std::str::from_utf8(b).ok()) {
                Some(Some(s)) => s.to_string(),
                _ => continue,
            };
            let tag = match parts.next().map(|b| std::str::from_utf8(b).ok()) {
                Some(Some(s)) => s.to_string(),
                _ => continue,
            };
            let version = match be_to_u64(&value_bytes) {
                Some(v) => v,
                None => continue,
            };
            self.inner.tags.write().insert((name, tag), version);
        }
        // next_version
        let counters = self
            .inner
            .backend
            .scan(tables::NEXT_VERSION)
            .map_err(|e| DatasetError::Embed(format!("backend scan counters: {e}")))?;
        for (key_bytes, value_bytes) in counters {
            let name = match std::str::from_utf8(&key_bytes) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            };
            let next = match be_to_u64(&value_bytes) {
                Some(v) => v,
                None => continue,
            };
            self.inner.next_version.write().insert(name, next);
        }
        // schemas
        let schemas = self
            .inner
            .backend
            .scan(tables::SCHEMAS)
            .map_err(|e| DatasetError::Embed(format!("backend scan schemas: {e}")))?;
        for (key_bytes, value_bytes) in schemas {
            let name = match std::str::from_utf8(&key_bytes) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            };
            let schema: serde_json::Value = serde_json::from_slice(&value_bytes)
                .unwrap_or(serde_json::Value::Null);
            self.inner.schemas.write().insert(name, schema);
        }
        Ok(())
    }

    fn next_internal_id_bump(&self) -> u64 {
        let mut n = self.inner.next_internal_id.write();
        let v = *n;
        *n += 1;
        v
    }

    /// Subscribe to change events. Each `create` / `add` / `tag` /
    /// `untag` / `delete` publishes a [`ChangeEvent`] with
    /// `table = "dataset"` and `key = Some(name)`, so
    /// `PSUBSCRIBE dataset.*` filters by dataset name.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Register a dataset name with an optional schema hint. Idempotent
    /// — re-creating an existing name replaces the schema and keeps
    /// every existing version.
    pub fn create(&self, name: impl Into<String>, schema: serde_json::Value) -> Result<()> {
        let name = name.into();
        let schema_bytes = serde_json::to_vec(&schema)
            .map_err(|e| DatasetError::Embed(format!("schema encode: {e}")))?;
        self.inner
            .backend
            .put(tables::SCHEMAS, name.as_bytes(), &schema_bytes)
            .map_err(|e| DatasetError::Embed(format!("backend put schema: {e}")))?;
        self.inner.schemas.write().insert(name.clone(), schema);
        // Make sure the version list at least exists (empty for now).
        self.inner.versions.write().entry(name.clone()).or_default();
        self.inner.bus.publish(ChangeEvent {
            table: "dataset".to_string(),
            key: Some(name),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Append a new immutable version containing the supplied rows.
    /// Embeds every row's `text` for semantic search. Returns the
    /// assigned monotonic version.
    pub fn add(
        &self,
        name: impl Into<String>,
        rows: Vec<DatasetRow>,
        metadata: serde_json::Value,
    ) -> Result<DatasetVersion> {
        let name = name.into();
        let version = self.bump_version(&name);
        // Embed and index each row.
        for row in &rows {
            let emb = self
                .inner
                .embedder
                .embed(&row.text)
                .map_err(|e| DatasetError::Embed(e.to_string()))?;
            let internal_id = {
                let mut n = self.inner.next_internal_id.write();
                let v = *n;
                *n += 1;
                v
            };
            self.inner
                .vector_index
                .write()
                .insert(internal_id, emb)
                .map_err(|e| DatasetError::Embed(format!("vector index: {e}")))?;
            self.inner
                .id_to_key
                .write()
                .insert(internal_id, (name.clone(), version, row.id.clone()));
            self.persist_id_to_key(internal_id, &name, version, &row.id).ok();
        }
        let schema = self
            .inner
            .schemas
            .read()
            .get(&name)
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let dataset = Dataset {
            name: name.clone(),
            version,
            schema,
            rows,
            tags: Vec::new(),
            metadata,
            created_at_unix_ns: now_unix_ns(),
        };
        // Persist the dataset row + bumped counter atomically.
        let bytes = serde_json::to_vec(&dataset)
            .map_err(|e| DatasetError::Embed(format!("dataset encode: {e}")))?;
        self.inner
            .backend
            .batch(&[
                BatchOp::Put {
                    table: tables::DATASETS.into(),
                    key: key::two(name.as_bytes(), &u64_to_be(version)),
                    value: bytes,
                },
                BatchOp::Put {
                    table: tables::NEXT_VERSION.into(),
                    key: name.as_bytes().to_vec(),
                    value: u64_to_be(version).to_vec(),
                },
            ])
            .map_err(|e| DatasetError::Embed(format!("backend add batch: {e}")))?;
        self.inner
            .by_id
            .write()
            .insert((name.clone(), version), dataset);
        self.inner.bus.publish(ChangeEvent {
            table: "dataset".to_string(),
            key: Some(name),
            row_id: version,
            kind: ChangeKind::Insert,
        });
        Ok(version)
    }

    /// Bump and return the next version number for `name`. Monotonic;
    /// survives delete.
    fn bump_version(&self, name: &str) -> DatasetVersion {
        let mut nv = self.inner.next_version.write();
        let v = *nv.entry(name.to_string()).or_insert(0) + 1;
        nv.insert(name.to_string(), v);
        self.inner
            .versions
            .write()
            .entry(name.to_string())
            .or_default()
            .push(v);
        v
    }

    /// Get a specific version of a dataset.
    pub fn get(&self, name: &str, version: DatasetVersion) -> Option<Dataset> {
        self.inner
            .by_id
            .read()
            .get(&(name.to_string(), version))
            .map(|d| {
                let tags = self.tags_for_version(name, version);
                Dataset {
                    tags,
                    ..d.clone()
                }
            })
    }

    /// Resolve a tag to its version, then return that dataset.
    pub fn get_by_tag(&self, name: &str, tag: &str) -> Option<Dataset> {
        let version = self
            .inner
            .tags
            .read()
            .get(&(name.to_string(), tag.to_string()))
            .copied()?;
        self.get(name, version)
    }

    /// Get the highest version of a dataset.
    pub fn get_latest(&self, name: &str) -> Option<Dataset> {
        let latest = self
            .inner
            .versions
            .read()
            .get(name)
            .and_then(|v| v.last().copied())?;
        self.get(name, latest)
    }

    /// List every version of a dataset, ascending.
    pub fn list(&self, name: &str) -> Vec<Dataset> {
        let versions = self
            .inner
            .versions
            .read()
            .get(name)
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(versions.len());
        for v in versions {
            if let Some(d) = self.get(name, v) {
                out.push(d);
            }
        }
        out
    }

    /// Names known to the registry. Lexicographic order.
    pub fn names(&self) -> Vec<DatasetName> {
        let mut ns: Vec<_> = self
            .inner
            .versions
            .read()
            .keys()
            .chain(self.inner.schemas.read().keys())
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ns.sort();
        ns
    }

    /// Point `tag` at `version` of `name`. Moves the tag atomically
    /// if it already exists.
    pub fn tag(&self, name: &str, version: DatasetVersion, tag: &str) -> Result<()> {
        if !self
            .inner
            .by_id
            .read()
            .contains_key(&(name.to_string(), version))
        {
            return Err(DatasetError::VersionNotFound {
                name: name.to_string(),
                version,
            });
        }
        if let Err(e) = self.inner.backend.put(
            tables::TAGS,
            &key::two(name.as_bytes(), tag.as_bytes()),
            &u64_to_be(version),
        ) {
            return Err(DatasetError::Embed(format!("backend put tag: {e}")));
        }
        self.inner
            .tags
            .write()
            .insert((name.to_string(), tag.to_string()), version);
        self.inner.bus.publish(ChangeEvent {
            table: "dataset".to_string(),
            key: Some(name.to_string()),
            row_id: version,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Remove a tag. No-op if it doesn't exist.
    pub fn untag(&self, name: &str, tag: &str) -> bool {
        let removed = self
            .inner
            .tags
            .write()
            .remove(&(name.to_string(), tag.to_string()))
            .is_some();
        if removed {
            if let Err(e) = self
                .inner
                .backend
                .delete(tables::TAGS, &key::two(name.as_bytes(), tag.as_bytes()))
            {
                tracing::warn!(error = %e, "backend untag delete failed");
            }
            self.inner.bus.publish(ChangeEvent {
                table: "dataset".to_string(),
                key: Some(name.to_string()),
                row_id: 0,
                kind: ChangeKind::Update,
            });
        }
        removed
    }

    /// Delete one version. The version number is NOT reused.
    pub fn delete(&self, name: &str, version: DatasetVersion) -> bool {
        let key_tuple = (name.to_string(), version);
        let removed = self.inner.by_id.write().remove(&key_tuple).is_some();
        if removed {
            if let Some(list) = self.inner.versions.write().get_mut(name) {
                list.retain(|v| *v != version);
            }
            let dropped_tags: Vec<String> = {
                let mut tags = self.inner.tags.write();
                let mut victims = Vec::new();
                tags.retain(|(n, t), v| {
                    let drop = n == name && *v == version;
                    if drop {
                        victims.push(t.clone());
                    }
                    !drop
                });
                victims
            };
            // Drop catalog index entries for this version, and
            // collect their internal ids so we can sweep the on-disk
            // id_to_key rows too.
            let dropped_internal_ids: Vec<u64> = {
                let mut m = self.inner.id_to_key.write();
                let victims: Vec<u64> = m
                    .iter()
                    .filter_map(|(iid, (n, v, _r))| {
                        if n == name && *v == version {
                            Some(*iid)
                        } else {
                            None
                        }
                    })
                    .collect();
                for iid in &victims {
                    m.remove(iid);
                }
                victims
            };
            // Mirror to backend in one batch.
            let mut ops: Vec<BatchOp> =
                Vec::with_capacity(1 + dropped_tags.len() + dropped_internal_ids.len());
            ops.push(BatchOp::Delete {
                table: tables::DATASETS.into(),
                key: key::two(name.as_bytes(), &u64_to_be(version)),
            });
            for t in dropped_tags {
                ops.push(BatchOp::Delete {
                    table: tables::TAGS.into(),
                    key: key::two(name.as_bytes(), t.as_bytes()),
                });
            }
            for iid in dropped_internal_ids {
                ops.push(BatchOp::Delete {
                    table: tables::ID_TO_KEY.into(),
                    key: iid.to_be_bytes().to_vec(),
                });
            }
            if let Err(e) = self.inner.backend.batch(&ops) {
                tracing::warn!(error = %e, "backend delete batch failed");
            }
            self.inner.bus.publish(ChangeEvent {
                table: "dataset".to_string(),
                key: Some(name.to_string()),
                row_id: version,
                kind: ChangeKind::Delete,
            });
        }
        removed
    }

    /// Sample up to `n` rows from a specific version. If `split` is
    /// `Some`, only rows in that split are sampled. Returns rows in
    /// insertion order (first n match); a real random sampler can
    /// live on top of this in 7.3b.
    pub fn sample(
        &self,
        name: &str,
        version: DatasetVersion,
        n: usize,
        split: Option<&str>,
    ) -> Vec<DatasetRow> {
        let ds = match self.get(name, version) {
            Some(d) => d,
            None => return Vec::new(),
        };
        ds.rows
            .into_iter()
            .filter(|r| split.map_or(true, |s| r.split == s))
            .take(n)
            .collect()
    }

    /// Total rows in a specific version, optionally restricted to a split.
    pub fn size(
        &self,
        name: &str,
        version: DatasetVersion,
        split: Option<&str>,
    ) -> usize {
        match self.get(name, version) {
            Some(d) => d
                .rows
                .iter()
                .filter(|r| split.map_or(true, |s| r.split == s))
                .count(),
            None => 0,
        }
    }

    /// Every distinct split name present in a specific version.
    pub fn splits(&self, name: &str, version: DatasetVersion) -> Vec<String> {
        let ds = match self.get(name, version) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let mut set: HashSet<String> = HashSet::new();
        for r in &ds.rows {
            set.insert(r.split.clone());
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort();
        v
    }

    /// Semantic search across every row of every version. Optionally
    /// restricted to one dataset. Returns up to `k` hits.
    pub fn search(&self, query: &str, k: usize, name_filter: Option<&str>) -> Result<Vec<DatasetHit>> {
        let query_vec = self
            .inner
            .embedder
            .embed(query)
            .map_err(|e| DatasetError::Embed(e.to_string()))?;
        // Over-fetch when filtering.
        let fetch_k = if name_filter.is_some() { k * 5 } else { k };
        let raw = self.inner.vector_index.read().search(&query_vec, fetch_k);
        let id_to_key = self.inner.id_to_key.read();
        let mut out = Vec::with_capacity(raw.len());
        for (id, dist) in raw {
            let (dname, dversion, row_id) = match id_to_key.get(&id) {
                Some(k) => k.clone(),
                None => continue,
            };
            if let Some(f) = name_filter {
                if dname != f {
                    continue;
                }
            }
            let row = match self.get(&dname, dversion) {
                Some(d) => d.rows.into_iter().find(|r| r.id == row_id),
                None => None,
            };
            let row = match row {
                Some(r) => r,
                None => continue,
            };
            let sim = (1.0 - dist).clamp(0.0, 1.0);
            out.push(DatasetHit {
                dataset: dname,
                version: dversion,
                row,
                score: sim,
            });
            if out.len() >= k {
                break;
            }
        }
        Ok(out)
    }

    /// Convenience: build a dataset version directly from raw
    /// `(text, split)` pairs. Auto-IDs each row.
    pub fn add_from_texts(
        &self,
        name: impl Into<String>,
        items: impl IntoIterator<Item = (String, String)>,
    ) -> Result<DatasetVersion> {
        let rows: Vec<DatasetRow> = items
            .into_iter()
            .map(|(t, split)| DatasetRow::new(t).with_split(split))
            .collect();
        self.add(name, rows, serde_json::Value::Null)
    }

    /// Counters for ops dashboards / metrics.
    pub fn stats(&self) -> DatasetStats {
        let versions_total = self.inner.by_id.read().len();
        let rows_total: usize = self.inner.by_id.read().values().map(|d| d.rows.len()).sum();
        DatasetStats {
            names: self.inner.versions.read().len(),
            versions: versions_total,
            rows: rows_total,
            tags: self.inner.tags.read().len(),
        }
    }

    // -- helpers --

    fn tags_for_version(&self, name: &str, version: DatasetVersion) -> Vec<DatasetTag> {
        self.inner
            .tags
            .read()
            .iter()
            .filter(|((n, _t), v)| n == name && **v == version)
            .map(|((_n, t), _v)| t.clone())
            .collect()
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn reg() -> DatasetRegistry {
        DatasetRegistry::new(Arc::new(HashEmbedder::new(16)))
    }

    fn row(text: &str, split: &str) -> DatasetRow {
        DatasetRow::new(text).with_split(split)
    }

    #[test]
    fn add_returns_monotonic_version() {
        let r = reg();
        let v1 = r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        let v2 = r.add("d", vec![row("b", "train")], serde_json::Value::Null).unwrap();
        assert_eq!((v1, v2), (1, 2));
    }

    #[test]
    fn create_then_add_uses_schema() {
        let r = reg();
        let schema = serde_json::json!({"fields": ["q", "a"]});
        r.create("d", schema.clone()).unwrap();
        r.add("d", vec![row("hello", "train")], serde_json::Value::Null).unwrap();
        let ds = r.get_latest("d").unwrap();
        assert_eq!(ds.schema, schema);
    }

    #[test]
    fn get_returns_named_version_with_tags() {
        let r = reg();
        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        r.add("d", vec![row("b", "train")], serde_json::Value::Null).unwrap();
        r.tag("d", 2, "golden").unwrap();
        let ds = r.get("d", 2).unwrap();
        assert_eq!(ds.tags, vec!["golden"]);
        assert_eq!(ds.rows.len(), 1);
    }

    #[test]
    fn get_by_tag_resolves_alias_and_re_targets() {
        let r = reg();
        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        r.add("d", vec![row("b", "train")], serde_json::Value::Null).unwrap();
        r.tag("d", 1, "golden").unwrap();
        assert_eq!(r.get_by_tag("d", "golden").unwrap().version, 1);
        r.tag("d", 2, "golden").unwrap();
        assert_eq!(r.get_by_tag("d", "golden").unwrap().version, 2);
    }

    #[test]
    fn tag_on_missing_version_errors() {
        let r = reg();
        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        let err = r.tag("d", 99, "golden").unwrap_err();
        matches!(err, DatasetError::VersionNotFound { .. });
    }

    #[test]
    fn untag_removes_alias_idempotently() {
        let r = reg();
        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        r.tag("d", 1, "golden").unwrap();
        assert!(r.untag("d", "golden"));
        assert!(r.get_by_tag("d", "golden").is_none());
        assert!(!r.untag("d", "golden"));
    }

    #[test]
    fn delete_preserves_monotonic_counter() {
        let r = reg();
        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        r.add("d", vec![row("b", "train")], serde_json::Value::Null).unwrap();
        assert!(r.delete("d", 2));
        let v3 = r.add("d", vec![row("c", "train")], serde_json::Value::Null).unwrap();
        // delete v2 then add gives v3, never v2 reused.
        assert_eq!(v3, 3);
        assert!(r.get("d", 2).is_none());
    }

    #[test]
    fn list_returns_versions_ascending() {
        let r = reg();
        for _ in 0..3 {
            r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        }
        let vs = r.list("d");
        assert_eq!(vs.len(), 3);
        assert_eq!(vs[0].version, 1);
        assert_eq!(vs[2].version, 3);
    }

    #[test]
    fn names_returns_known_datasets_lexicographic() {
        let r = reg();
        r.add("beta", vec![row("a", "")], serde_json::Value::Null).unwrap();
        r.add("alpha", vec![row("b", "")], serde_json::Value::Null).unwrap();
        assert_eq!(r.names(), vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn sample_filters_by_split() {
        let r = reg();
        let rows = vec![
            row("t1", "train"),
            row("t2", "train"),
            row("e1", "eval"),
            row("e2", "eval"),
            row("e3", "eval"),
        ];
        let v = r.add("d", rows, serde_json::Value::Null).unwrap();
        let eval_only = r.sample("d", v, 10, Some("eval"));
        assert_eq!(eval_only.len(), 3);
        let train_only = r.sample("d", v, 10, Some("train"));
        assert_eq!(train_only.len(), 2);
        let any = r.sample("d", v, 10, None);
        assert_eq!(any.len(), 5);
    }

    #[test]
    fn size_and_splits_reflect_per_split_counts() {
        let r = reg();
        let rows = vec![
            row("a", "train"),
            row("b", "train"),
            row("c", "eval"),
        ];
        let v = r.add("d", rows, serde_json::Value::Null).unwrap();
        assert_eq!(r.size("d", v, None), 3);
        assert_eq!(r.size("d", v, Some("train")), 2);
        assert_eq!(r.size("d", v, Some("eval")), 1);
        let splits = r.splits("d", v);
        assert_eq!(splits, vec!["eval".to_string(), "train".to_string()]);
    }

    #[test]
    fn search_finds_semantically_close_rows() {
        let r = reg();
        let rows = vec![
            row("hello world how are you", "eval"),
            row("goodbye see you later", "eval"),
            row("i can help with your refund issue", "eval"),
        ];
        r.add("d", rows, serde_json::Value::Null).unwrap();
        let hits = r.search("hello world", 1, None).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].row.text.contains("hello"));
    }

    #[test]
    fn search_filters_by_dataset_name() {
        let r = reg();
        r.add("a", vec![row("hello world", "eval")], serde_json::Value::Null).unwrap();
        r.add("b", vec![row("hello world", "eval")], serde_json::Value::Null).unwrap();
        let hits = r.search("hello", 10, Some("a")).unwrap();
        assert!(hits.iter().all(|h| h.dataset == "a"));
    }

    #[test]
    fn change_bus_publishes_create_add_tag_delete() {
        let r = reg();
        let mut rx = r.subscribe();
        r.create("d", serde_json::Value::Null).unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Update));
        assert_eq!(e.key.as_deref(), Some("d"));

        r.add("d", vec![row("a", "train")], serde_json::Value::Null).unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Insert));

        r.tag("d", 1, "golden").unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Update));

        r.delete("d", 1);
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Delete));
    }

    #[test]
    fn add_from_texts_auto_ids_and_splits() {
        let r = reg();
        let v = r
            .add_from_texts(
                "d",
                vec![
                    ("alpha".to_string(), "train".to_string()),
                    ("beta".to_string(), "eval".to_string()),
                ],
            )
            .unwrap();
        let ds = r.get("d", v).unwrap();
        assert_eq!(ds.rows.len(), 2);
        assert!(!ds.rows[0].id.is_empty());
        assert_eq!(ds.rows[0].split, "train");
        assert_eq!(ds.rows[1].split, "eval");
    }

    #[test]
    fn stats_counts_everything() {
        let r = reg();
        r.add("a", vec![row("x", "train"), row("y", "train")], serde_json::Value::Null).unwrap();
        r.add("a", vec![row("z", "eval")], serde_json::Value::Null).unwrap();
        r.add("b", vec![row("p", "train")], serde_json::Value::Null).unwrap();
        r.tag("a", 1, "golden").unwrap();
        let s = r.stats();
        assert_eq!(s.names, 2);
        assert_eq!(s.versions, 3);
        assert_eq!(s.rows, 4);
        assert_eq!(s.tags, 1);
    }

    // ---------------------------------------------------------------- persistence

    #[test]
    fn open_rehydrates_versions_rows_tags_counter() {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        let embedder = Arc::new(HashEmbedder::new(16));
        {
            let r = DatasetRegistry::open(embedder.clone(), backend.clone()).unwrap();
            r.create("refunds", serde_json::json!({"version": 1})).unwrap();
            r.add(
                "refunds",
                vec![row("r1", "train"), row("r2", "eval")],
                serde_json::json!({"author": "alice"}),
            )
            .unwrap();
            let v2 = r
                .add("refunds", vec![row("r3", "train")], serde_json::Value::Null)
                .unwrap();
            r.tag("refunds", v2, "golden").unwrap();
            assert!(r.delete("refunds", 1));
        }
        let r = DatasetRegistry::open(embedder.clone(), backend.clone()).unwrap();
        // v1 was deleted; v2 survives.
        assert!(r.get("refunds", 1).is_none());
        let v2 = r.get("refunds", 2).unwrap();
        assert_eq!(v2.rows.len(), 1);
        assert_eq!(v2.tags, vec!["golden"]);
        // Counter must NOT reuse 1.
        let v3 = r.add("refunds", vec![row("r4", "train")], serde_json::Value::Null).unwrap();
        assert_eq!(v3, 3);
        // Schema survives.
        assert_eq!(r.names(), vec!["refunds".to_string()]);
    }

    #[test]
    fn open_rehydrates_search_index() {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        let embedder = Arc::new(HashEmbedder::new(16));
        {
            let r = DatasetRegistry::open(embedder.clone(), backend.clone()).unwrap();
            r.add(
                "qa",
                vec![row("hello world how are you", "train"), row("goodbye see you soon", "train")],
                serde_json::Value::Null,
            )
            .unwrap();
        }
        let r = DatasetRegistry::open(embedder.clone(), backend.clone()).unwrap();
        let hits = r.search("hello", 2, None).unwrap();
        assert!(!hits.is_empty(), "search returned nothing after reopen");
    }
}
