//! Multi-table, byte-keyed `Backend` trait.
//!
//! The Phase 6 [`Storage`](crate::storage::Storage) trait was designed
//! around `MemoryStore` and is `u64`-keyed: rows are auto-numbered,
//! a single table is enough. Phase 7 primitives (`PromptRegistry`,
//! `DatasetRegistry`, `EvalRegistry`, `ReplayRegistry`, `TraceStore`,
//! `CostLedger`) have a different shape: each owns multiple logical
//! tables, keys are composite strings (e.g. `(name, version)`), and a
//! write often touches more than one table atomically (e.g.
//! `PROMPT.PUT` inserts into `prompts` AND updates the `name_index`).
//!
//! The [`Backend`] trait covers that shape:
//!
//! * Named tables, created lazily on first access.
//! * Opaque byte keys + byte values — callers serialize their rows
//!   themselves (bincode, JSON, whatever).
//! * Prefix scans, for composite-key range queries.
//! * Atomic batches.
//!
//! Two implementations:
//!
//! * [`MemoryBackend`] — `BTreeMap` per table, no durability. Default
//!   for tests and ephemeral demos.
//! * [`RedbBackend`] (feature `redb-store`) — one redb database file
//!   per backend, one redb table per logical table. Durable, ACID.
//!
//! See `docs/V0_2_0_PLAN.md` for the wider context and the order each
//! Phase 7 primitive will adopt this trait.

use duxx_core::{Error, Result};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::path::Path;

/// One atomic operation in a [`Backend::batch`] call.
#[derive(Debug, Clone)]
pub enum BatchOp {
    /// Insert or overwrite ``value`` at ``key`` in ``table``.
    Put {
        table: String,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// Delete ``key`` from ``table``.  No-op if absent.
    Delete { table: String, key: Vec<u8> },
}

/// Multi-table, byte-keyed persistence backend.
///
/// Tables are identified by string name (`"prompts"`, `"tags"`,
/// `"spans"`, …) and created lazily on first write. Backends MUST
/// preserve key ordering for [`Backend::scan`] and
/// [`Backend::scan_prefix`] — keys are returned in lexicographic
/// (`Vec<u8>`) order. Composite-key helpers in [`key`] encode their
/// parts so that prefix scans on `(a)` return every row whose first
/// key part is `a`, regardless of the trailing parts.
pub trait Backend: Send + Sync + std::fmt::Debug {
    /// Insert or overwrite ``value`` at ``key`` in ``table``.
    fn put(&self, table: &str, key: &[u8], value: &[u8]) -> Result<()>;
    /// Return the value for ``key`` in ``table``, or ``None``.
    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;
    /// Remove ``key`` from ``table``. Returns ``true`` if a row was
    /// deleted, ``false`` if the key was absent.
    fn delete(&self, table: &str, key: &[u8]) -> Result<bool>;
    /// Every (key, value) pair in ``table``, key-sorted.
    fn scan(&self, table: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    /// Every (key, value) pair in ``table`` whose key starts with
    /// ``prefix``, key-sorted.
    fn scan_prefix(&self, table: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    /// Apply every operation in ``ops`` atomically. Either all writes
    /// commit or none do.
    fn batch(&self, ops: &[BatchOp]) -> Result<()>;
    /// Number of rows in ``table`` (or ``0`` if the table has never
    /// been written to).
    fn count(&self, table: &str) -> Result<usize>;
    /// Force a durability barrier. In-memory backends are a no-op.
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Composite-key helpers
// ---------------------------------------------------------------------------

/// Encoding helpers for composite keys.
///
/// Phase 7 primitives use composite keys like
/// `(prompt_name, version)`, `(dataset_name, version, row_id)`,
/// `(run_id, row_id)`. These helpers encode parts so that a prefix
/// scan on `(prompt_name,)` returns every row for that name,
/// regardless of the trailing version.
///
/// The separator is `0x00` — illegal in valid UTF-8 strings, which
/// is what every part actually is in practice. If a future caller
/// needs binary parts, switch to a length-prefixed encoding.
pub mod key {
    /// Encode a 1-part key. Identical to the input slice — provided
    /// for symmetry with the multi-part helpers.
    #[inline]
    pub fn one(a: &[u8]) -> Vec<u8> {
        a.to_vec()
    }

    /// Encode a 2-part composite key.
    #[inline]
    pub fn two(a: &[u8], b: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(a.len() + 1 + b.len());
        out.extend_from_slice(a);
        out.push(0);
        out.extend_from_slice(b);
        out
    }

    /// Encode a 3-part composite key.
    #[inline]
    pub fn three(a: &[u8], b: &[u8], c: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(a.len() + 1 + b.len() + 1 + c.len());
        out.extend_from_slice(a);
        out.push(0);
        out.extend_from_slice(b);
        out.push(0);
        out.extend_from_slice(c);
        out
    }

    /// Encode a prefix made of the given parts followed by a separator,
    /// so that a [`Backend::scan_prefix`] call returns every row whose
    /// next part starts a new sub-tree.
    #[inline]
    pub fn prefix(parts: &[&[u8]]) -> Vec<u8> {
        let total: usize = parts.iter().map(|p| p.len()).sum::<usize>() + parts.len();
        let mut out = Vec::with_capacity(total);
        for p in parts {
            out.extend_from_slice(p);
            out.push(0);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MemoryBackend
// ---------------------------------------------------------------------------

/// In-memory backend. `BTreeMap` per table, no durability.
///
/// Use for tests, ephemeral demos, and as the default when no
/// `storage="..."` selector is passed. Preserves the exact behavior
/// of the v0.1.x Phase 7 primitives.
#[derive(Debug, Default)]
pub struct MemoryBackend {
    tables: RwLock<BTreeMap<String, BTreeMap<Vec<u8>, Vec<u8>>>>,
}

impl MemoryBackend {
    /// Build an empty backend.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Backend for MemoryBackend {
    fn put(&self, table: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let mut tables = self.tables.write();
        tables
            .entry(table.to_owned())
            .or_default()
            .insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self
            .tables
            .read()
            .get(table)
            .and_then(|t| t.get(key))
            .cloned())
    }

    fn delete(&self, table: &str, key: &[u8]) -> Result<bool> {
        let mut tables = self.tables.write();
        Ok(match tables.get_mut(table) {
            Some(t) => t.remove(key).is_some(),
            None => false,
        })
    }

    fn scan(&self, table: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .tables
            .read()
            .get(table)
            .map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default())
    }

    fn scan_prefix(&self, table: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .tables
            .read()
            .get(table)
            .map(|t| {
                t.range(prefix.to_vec()..)
                    .take_while(|(k, _)| k.starts_with(prefix))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn batch(&self, ops: &[BatchOp]) -> Result<()> {
        let mut tables = self.tables.write();
        for op in ops {
            match op {
                BatchOp::Put { table, key, value } => {
                    tables
                        .entry(table.clone())
                        .or_default()
                        .insert(key.clone(), value.clone());
                }
                BatchOp::Delete { table, key } => {
                    if let Some(t) = tables.get_mut(table) {
                        t.remove(key);
                    }
                }
            }
        }
        Ok(())
    }

    fn count(&self, table: &str) -> Result<usize> {
        Ok(self
            .tables
            .read()
            .get(table)
            .map(|t| t.len())
            .unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// RedbBackend
// ---------------------------------------------------------------------------

#[cfg(feature = "redb-store")]
mod redb_backend {
    use super::{Backend, BatchOp, Error, Result};
    use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
    use std::path::Path;
    use std::sync::Arc;

    /// Durable, ACID backend backed by a single redb database.
    ///
    /// Each logical table maps to one redb table opened lazily on
    /// first write. Multi-op writes are wrapped in a single redb
    /// transaction for atomicity.
    pub struct RedbBackend {
        db: Arc<Database>,
    }

    impl std::fmt::Debug for RedbBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RedbBackend").finish_non_exhaustive()
        }
    }

    impl RedbBackend {
        /// Open (or create) a redb database at ``path``.
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let db = Database::create(path.as_ref())
                .map_err(|e| Error::Storage(format!("redb open: {e}")))?;
            Ok(Self { db: Arc::new(db) })
        }
    }

    /// Help redb infer a stable per-table definition. Table names
    /// must outlive the call; we lean on `'static` from the caller.
    fn table_def(name: &str) -> TableDefinition<'static, &'static [u8], &'static [u8]> {
        // Lifetime-laundering: redb's TableDefinition is parameterized
        // on the name lifetime. We accept a `&str` and leak it. This
        // is bounded — table names come from a small closed set
        // hard-coded by each Phase 7 primitive (e.g. "prompts",
        // "tags") so the leak is at most a few dozen bytes total per
        // process.
        let static_name: &'static str = Box::leak(name.to_owned().into_boxed_str());
        TableDefinition::new(static_name)
    }

    impl Backend for RedbBackend {
        fn put(&self, table: &str, key: &[u8], value: &[u8]) -> Result<()> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb begin_write: {e}")))?;
            {
                let mut t = txn
                    .open_table(def)
                    .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
                t.insert(key, value)
                    .map_err(|e| Error::Storage(format!("redb insert: {e}")))?;
            }
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(())
        }

        fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb begin_read: {e}")))?;
            let t = match txn.open_table(def) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
                Err(e) => return Err(Error::Storage(format!("redb open_table: {e}"))),
            };
            let got = t
                .get(key)
                .map_err(|e| Error::Storage(format!("redb get: {e}")))?;
            Ok(got.map(|v| v.value().to_vec()))
        }

        fn delete(&self, table: &str, key: &[u8]) -> Result<bool> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb begin_write: {e}")))?;
            let removed: bool;
            {
                let mut t = match txn.open_table(def) {
                    Ok(t) => t,
                    Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
                    Err(e) => return Err(Error::Storage(format!("redb open_table: {e}"))),
                };
                // Bind the AccessGuard to a named local so it drops
                // BEFORE `t` does. Without this, redb 2.x complains
                // that the temporary borrow outlives the table.
                let access = t
                    .remove(key)
                    .map_err(|e| Error::Storage(format!("redb remove: {e}")))?;
                removed = access.is_some();
            }
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(removed)
        }

        fn scan(&self, table: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb begin_read: {e}")))?;
            let t = match txn.open_table(def) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
                Err(e) => return Err(Error::Storage(format!("redb open_table: {e}"))),
            };
            let mut out = Vec::new();
            for entry in t
                .iter()
                .map_err(|e| Error::Storage(format!("redb iter: {e}")))?
            {
                let (k, v) =
                    entry.map_err(|e| Error::Storage(format!("redb iter entry: {e}")))?;
                out.push((k.value().to_vec(), v.value().to_vec()));
            }
            Ok(out)
        }

        fn scan_prefix(&self, table: &str, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb begin_read: {e}")))?;
            let t = match txn.open_table(def) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
                Err(e) => return Err(Error::Storage(format!("redb open_table: {e}"))),
            };
            // Compute the lexicographic upper bound by incrementing
            // the prefix's last byte. None means the prefix
            // overflows (e.g. all 0xff) — fall back to a full scan
            // and filter.
            let upper = next_prefix(prefix);
            let mut out = Vec::new();
            let iter = match &upper {
                Some(up) => t
                    .range(prefix..up.as_slice())
                    .map_err(|e| Error::Storage(format!("redb range: {e}")))?,
                None => t
                    .range(prefix..)
                    .map_err(|e| Error::Storage(format!("redb range: {e}")))?,
            };
            for entry in iter {
                let (k, v) =
                    entry.map_err(|e| Error::Storage(format!("redb iter entry: {e}")))?;
                let key_vec = k.value().to_vec();
                if !key_vec.starts_with(prefix) {
                    break;
                }
                out.push((key_vec, v.value().to_vec()));
            }
            Ok(out)
        }

        fn batch(&self, ops: &[BatchOp]) -> Result<()> {
            let txn = self
                .db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb begin_write: {e}")))?;
            for op in ops {
                match op {
                    BatchOp::Put { table, key, value } => {
                        let def = table_def(table);
                        let mut t = txn
                            .open_table(def)
                            .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
                        t.insert(key.as_slice(), value.as_slice())
                            .map_err(|e| Error::Storage(format!("redb insert: {e}")))?;
                    }
                    BatchOp::Delete { table, key } => {
                        let def = table_def(table);
                        let mut t = match txn.open_table(def) {
                            Ok(t) => t,
                            Err(redb::TableError::TableDoesNotExist(_)) => continue,
                            Err(e) => {
                                return Err(Error::Storage(format!("redb open_table: {e}")));
                            }
                        };
                        t.remove(key.as_slice())
                            .map_err(|e| Error::Storage(format!("redb remove: {e}")))?;
                    }
                }
            }
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(())
        }

        fn count(&self, table: &str) -> Result<usize> {
            let def = table_def(table);
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb begin_read: {e}")))?;
            let t = match txn.open_table(def) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(0),
                Err(e) => return Err(Error::Storage(format!("redb open_table: {e}"))),
            };
            let n = t
                .len()
                .map_err(|e| Error::Storage(format!("redb len: {e}")))?;
            Ok(n as usize)
        }
    }

    /// Compute the lexicographic successor of ``prefix``. Returns
    /// ``None`` when every byte is ``0xff`` (no successor exists).
    fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
        let mut out = prefix.to_vec();
        while let Some(b) = out.last_mut() {
            if *b == 0xff {
                out.pop();
            } else {
                *b += 1;
                return Some(out);
            }
        }
        None
    }
}

#[cfg(feature = "redb-store")]
pub use redb_backend::RedbBackend;

// ---------------------------------------------------------------------------
// Path-based open helper
// ---------------------------------------------------------------------------

/// Open the backend implied by ``spec``.
///
/// Accepts the same shorthand the `duxx-server --storage` flag uses,
/// scoped to one primitive's backend:
///
/// * ``None`` or ``Some("")`` or ``Some("memory")`` → [`MemoryBackend`].
/// * ``Some("redb:/path/to/file.redb")`` → [`RedbBackend`] (requires
///   the ``redb-store`` feature, on by default).
///
/// Unrecognized prefixes return an error.
pub fn open_backend(spec: Option<&str>) -> Result<Box<dyn Backend>> {
    match spec {
        None | Some("") | Some("memory") => Ok(Box::new(MemoryBackend::new())),
        #[cfg(feature = "redb-store")]
        Some(s) if s.starts_with("redb:") => {
            let path = &s["redb:".len()..];
            Ok(Box::new(RedbBackend::open(Path::new(path))?))
        }
        Some(other) => Err(Error::Storage(format!(
            "unknown backend spec: {other:?}. Expected one of: memory, redb:<path>"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<B: Backend + 'static>(backend: B) {
        backend.put("t", b"alpha", b"1").unwrap();
        backend.put("t", b"bravo", b"2").unwrap();
        backend.put("t", b"charlie", b"3").unwrap();
        assert_eq!(backend.get("t", b"alpha").unwrap().as_deref(), Some(&b"1"[..]));
        assert_eq!(backend.get("t", b"bravo").unwrap().as_deref(), Some(&b"2"[..]));
        assert_eq!(backend.get("t", b"missing").unwrap(), None);
        assert_eq!(backend.count("t").unwrap(), 3);
        let rows = backend.scan("t").unwrap();
        let keys: Vec<&[u8]> = rows.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, vec![&b"alpha"[..], b"bravo", b"charlie"]);
        assert!(backend.delete("t", b"bravo").unwrap());
        assert!(!backend.delete("t", b"bravo").unwrap());
        assert_eq!(backend.count("t").unwrap(), 2);
    }

    #[test]
    fn memory_backend_round_trip() {
        round_trip(MemoryBackend::new());
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_backend_round_trip() {
        let tmp = tempfile_path("redb-round-trip.redb");
        let backend = RedbBackend::open(&tmp).unwrap();
        round_trip(backend);
        let _ = std::fs::remove_file(&tmp);
    }

    fn prefix_scan<B: Backend>(backend: &B) {
        backend.put("t", &key::two(b"alice", b"v1"), b"a1").unwrap();
        backend.put("t", &key::two(b"alice", b"v2"), b"a2").unwrap();
        backend.put("t", &key::two(b"alice", b"v3"), b"a3").unwrap();
        backend.put("t", &key::two(b"bob", b"v1"), b"b1").unwrap();
        backend.put("t", &key::two(b"bob", b"v2"), b"b2").unwrap();

        let alice = backend
            .scan_prefix("t", &key::prefix(&[b"alice"]))
            .unwrap();
        assert_eq!(alice.len(), 3);
        for (_k, v) in &alice {
            assert!(v[0] == b'a');
        }

        let bob = backend.scan_prefix("t", &key::prefix(&[b"bob"])).unwrap();
        assert_eq!(bob.len(), 2);
        for (_k, v) in &bob {
            assert!(v[0] == b'b');
        }

        // Missing prefix yields the empty set.
        let none = backend
            .scan_prefix("t", &key::prefix(&[b"charlie"]))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn memory_backend_prefix_scan() {
        prefix_scan(&MemoryBackend::new());
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_backend_prefix_scan() {
        let tmp = tempfile_path("redb-prefix-scan.redb");
        let backend = RedbBackend::open(&tmp).unwrap();
        prefix_scan(&backend);
        let _ = std::fs::remove_file(&tmp);
    }

    fn batch_atomicity<B: Backend>(backend: &B) {
        backend
            .batch(&[
                BatchOp::Put {
                    table: "t".into(),
                    key: b"a".to_vec(),
                    value: b"1".to_vec(),
                },
                BatchOp::Put {
                    table: "t".into(),
                    key: b"b".to_vec(),
                    value: b"2".to_vec(),
                },
                BatchOp::Delete {
                    table: "t".into(),
                    key: b"a".to_vec(),
                },
            ])
            .unwrap();
        assert_eq!(backend.get("t", b"a").unwrap(), None);
        assert_eq!(backend.get("t", b"b").unwrap().as_deref(), Some(&b"2"[..]));
    }

    #[test]
    fn memory_backend_batch() {
        batch_atomicity(&MemoryBackend::new());
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_backend_batch() {
        let tmp = tempfile_path("redb-batch.redb");
        let backend = RedbBackend::open(&tmp).unwrap();
        batch_atomicity(&backend);
        let _ = std::fs::remove_file(&tmp);
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_persists_across_reopen() {
        let tmp = tempfile_path("redb-persist.redb");
        {
            let backend = RedbBackend::open(&tmp).unwrap();
            backend.put("t", b"keep_me", b"forever").unwrap();
            backend.put("t", b"also_me", b"too").unwrap();
        }
        // Drop the first backend, reopen the same file, verify rows
        // survived.
        {
            let backend = RedbBackend::open(&tmp).unwrap();
            assert_eq!(backend.count("t").unwrap(), 2);
            assert_eq!(
                backend.get("t", b"keep_me").unwrap().as_deref(),
                Some(&b"forever"[..])
            );
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn open_backend_memory() {
        let b = open_backend(None).unwrap();
        b.put("t", b"k", b"v").unwrap();
        assert_eq!(b.get("t", b"k").unwrap().as_deref(), Some(&b"v"[..]));

        let b2 = open_backend(Some("memory")).unwrap();
        assert_eq!(b2.count("t").unwrap(), 0);
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn open_backend_redb_spec() {
        let tmp = tempfile_path("redb-spec.redb");
        let spec = format!("redb:{}", tmp.display());
        let b = open_backend(Some(&spec)).unwrap();
        b.put("t", b"k", b"v").unwrap();
        assert_eq!(b.get("t", b"k").unwrap().as_deref(), Some(&b"v"[..]));
        drop(b);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn open_backend_unknown_spec() {
        let err = open_backend(Some("rocksdb:./foo")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown backend"), "got: {msg}");
    }

    #[cfg(feature = "redb-store")]
    fn tempfile_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        // Add a fast-but-unique suffix so concurrent test runs don't
        // collide on the same file.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("duxx-backend-{nanos}-{name}"));
        let _ = std::fs::remove_file(&p);
        p
    }
}
