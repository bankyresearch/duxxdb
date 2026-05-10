//! Byte-keyed key/value storage trait + impls.
//!
//! `MemoryStore` plugs in a `Storage` to make memories durable across
//! restarts. The trait is intentionally tiny — `put` / `get` / `iter` —
//! so we can ship multiple backends behind feature flags without API
//! churn:
//!
//! - [`MemoryStorage`] (always available) — in-process `HashMap`. No
//!   durability; matches the pre-Phase-2.3 behavior.
//! - [`RedbStorage`] (feature `redb-store`, on by default) — durable
//!   ACID storage via the [redb] embedded crate. Pure Rust, no FFI.
//!
//! [redb]: https://github.com/cberner/redb

use duxx_core::{Error, Result};
use parking_lot::RwLock;
use std::collections::HashMap;

/// Minimum surface a `MemoryStore` needs to persist row payloads
/// keyed by `u64` id.
pub trait Storage: Send + Sync + std::fmt::Debug {
    /// Insert (or overwrite) the value for `id`.
    fn put(&self, id: u64, value: &[u8]) -> Result<()>;
    /// Get the value for `id`, or `None` if absent.
    fn get(&self, id: u64) -> Result<Option<Vec<u8>>>;
    /// Remove the value for `id`. Returns true if a row was deleted.
    fn delete(&self, id: u64) -> Result<bool>;
    /// Snapshot every (id, bytes) pair. Order is unspecified.
    fn iter(&self) -> Result<Vec<(u64, Vec<u8>)>>;
    /// Number of stored rows.
    fn len(&self) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// MemoryStorage
// ---------------------------------------------------------------------------

/// In-process `HashMap`-backed `Storage`. No durability — process exit
/// loses everything. Useful for tests, CI, and ephemeral demos.
#[derive(Debug, Default)]
pub struct MemoryStorage {
    inner: RwLock<HashMap<u64, Vec<u8>>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemoryStorage {
    fn put(&self, id: u64, value: &[u8]) -> Result<()> {
        self.inner.write().insert(id, value.to_vec());
        Ok(())
    }

    fn get(&self, id: u64) -> Result<Option<Vec<u8>>> {
        Ok(self.inner.read().get(&id).cloned())
    }

    fn delete(&self, id: u64) -> Result<bool> {
        Ok(self.inner.write().remove(&id).is_some())
    }

    fn iter(&self) -> Result<Vec<(u64, Vec<u8>)>> {
        Ok(self
            .inner
            .read()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect())
    }

    fn len(&self) -> Result<usize> {
        Ok(self.inner.read().len())
    }
}

// ---------------------------------------------------------------------------
// RedbStorage
// ---------------------------------------------------------------------------

#[cfg(feature = "redb-store")]
mod redb_impl {
    use super::*;
    use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
    use std::path::Path;

    const TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("memories");

    /// Durable redb-backed storage. ACID, MVCC, copy-on-write — survives
    /// process exit and crashes.
    ///
    /// Each `put` opens a write transaction, inserts, and commits. That's
    /// safe but ~1 ms per call on NVMe. For high-throughput bulk loads
    /// the future `RedbStorage::with_batch_writer()` API will keep a
    /// long-running transaction open. Closed-UAT scope sticks with the
    /// safe per-write commit.
    pub struct RedbStorage {
        db: Database,
    }

    impl std::fmt::Debug for RedbStorage {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RedbStorage").finish_non_exhaustive()
        }
    }

    impl RedbStorage {
        /// Open or create a redb database at `path`. The parent directory
        /// must exist.
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let db = Database::create(path.as_ref())
                .map_err(|e| Error::Storage(format!("redb open: {e}")))?;
            // Make sure the table exists — empty write txn that opens it.
            let txn = db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb txn: {e}")))?;
            {
                let _t = txn
                    .open_table(TABLE)
                    .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
            }
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(Self { db })
        }
    }

    impl Storage for RedbStorage {
        fn put(&self, id: u64, value: &[u8]) -> Result<()> {
            let txn = self
                .db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb txn: {e}")))?;
            {
                let mut t = txn
                    .open_table(TABLE)
                    .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
                t.insert(id, value)
                    .map_err(|e| Error::Storage(format!("redb insert: {e}")))?;
            }
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(())
        }

        fn get(&self, id: u64) -> Result<Option<Vec<u8>>> {
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb read txn: {e}")))?;
            let t = txn
                .open_table(TABLE)
                .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
            let v = t
                .get(id)
                .map_err(|e| Error::Storage(format!("redb get: {e}")))?;
            Ok(v.map(|access| access.value().to_vec()))
        }

        fn delete(&self, id: u64) -> Result<bool> {
            let txn = self
                .db
                .begin_write()
                .map_err(|e| Error::Storage(format!("redb txn: {e}")))?;
            let removed = {
                let mut t = txn
                    .open_table(TABLE)
                    .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
                let prior = t
                    .remove(id)
                    .map_err(|e| Error::Storage(format!("redb remove: {e}")))?;
                let was_present = prior.is_some();
                drop(prior); // release borrow on `t` before block end
                was_present
            };
            txn.commit()
                .map_err(|e| Error::Storage(format!("redb commit: {e}")))?;
            Ok(removed)
        }

        fn iter(&self) -> Result<Vec<(u64, Vec<u8>)>> {
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb read txn: {e}")))?;
            let t = txn
                .open_table(TABLE)
                .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
            let mut out = Vec::new();
            for entry in t.iter().map_err(|e| Error::Storage(format!("redb iter: {e}")))? {
                let (k, v) = entry.map_err(|e| Error::Storage(format!("redb iter: {e}")))?;
                out.push((k.value(), v.value().to_vec()));
            }
            Ok(out)
        }

        fn len(&self) -> Result<usize> {
            let txn = self
                .db
                .begin_read()
                .map_err(|e| Error::Storage(format!("redb read txn: {e}")))?;
            let t = txn
                .open_table(TABLE)
                .map_err(|e| Error::Storage(format!("redb open_table: {e}")))?;
            t.len()
                .map(|n| n as usize)
                .map_err(|e| Error::Storage(format!("redb len: {e}")))
        }
    }
}

#[cfg(feature = "redb-store")]
pub use redb_impl::RedbStorage;

#[cfg(test)]
mod tests {
    use super::*;

    /// Generic "any Storage impl" smoke suite.
    fn run_basic_suite(s: &dyn Storage) {
        s.put(1, b"alpha").unwrap();
        s.put(2, b"beta").unwrap();
        assert_eq!(s.get(1).unwrap(), Some(b"alpha".to_vec()));
        assert_eq!(s.get(2).unwrap(), Some(b"beta".to_vec()));
        assert_eq!(s.get(99).unwrap(), None);
        assert_eq!(s.len().unwrap(), 2);

        let mut all = s.iter().unwrap();
        all.sort_by_key(|(id, _)| *id);
        assert_eq!(all, vec![(1, b"alpha".to_vec()), (2, b"beta".to_vec())]);

        assert!(s.delete(1).unwrap());
        assert!(!s.delete(1).unwrap());
        assert_eq!(s.len().unwrap(), 1);

        // Overwrite.
        s.put(2, b"BETA").unwrap();
        assert_eq!(s.get(2).unwrap(), Some(b"BETA".to_vec()));
    }

    #[test]
    fn memory_storage_basic() {
        let s = MemoryStorage::new();
        run_basic_suite(&s);
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_storage_basic() {
        let dir = std::env::temp_dir().join(format!("duxx-storage-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.redb");
        {
            let s = RedbStorage::open(&path).unwrap();
            run_basic_suite(&s);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "redb-store")]
    #[test]
    fn redb_storage_persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!("duxx-storage-persist-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("persist.redb");
        {
            let s = RedbStorage::open(&path).unwrap();
            s.put(7, b"durable").unwrap();
        } // drop closes the db
        {
            let s2 = RedbStorage::open(&path).unwrap();
            assert_eq!(s2.get(7).unwrap(), Some(b"durable".to_vec()));
            assert_eq!(s2.len().unwrap(), 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
