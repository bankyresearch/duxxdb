//! The in-memory row map of a [`crate::MemoryStore`].
//!
//! Full rows live in a `HashMap<u64, Memory>`. Lean rows (`DUXX_LEAN_ROWS`)
//! need only each row's key, importance and creation time, because the text
//! and the embedding are read from the row store for the rows a recall
//! returns. A `HashMap<u64, Memory>` still spends about 250 bytes on such a
//! row (the 112-byte `Memory` with three empty `String`/`Vec` headers, the
//! key's own allocation, and the table's spare buckets), so lean rows are
//! kept here as 16 bytes in a vector indexed by row id (ids are handed out in
//! order, so the vector is dense), with keys interned once.

use crate::Memory;
use std::collections::HashMap;
use std::sync::Arc;

/// Marks a vacant slot (never written, or forgotten).
const VACANT: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct LeanRow {
    key: u32,
    importance: f32,
    /// Unix nanoseconds; `u64` reaches the year 2554.
    created_at_unix_ns: u64,
}

const EMPTY: LeanRow = LeanRow {
    key: VACANT,
    importance: 0.0,
    created_at_unix_ns: 0,
};

#[derive(Default)]
pub(crate) struct LeanRows {
    rows: Vec<LeanRow>,
    keys: Vec<Arc<str>>,
    key_ids: HashMap<Arc<str>, u32>,
    live: usize,
}

impl LeanRows {
    fn key_id(&mut self, key: &str) -> u32 {
        if let Some(id) = self.key_ids.get(key) {
            return *id;
        }
        let id = self.keys.len() as u32;
        let k: Arc<str> = Arc::from(key);
        self.keys.push(k.clone());
        self.key_ids.insert(k, id);
        id
    }

    fn get(&self, id: u64) -> Option<Memory> {
        let r = self.rows.get(usize::try_from(id).ok()?)?;
        if r.key == VACANT {
            return None;
        }
        Some(Memory {
            id,
            key: self.keys[r.key as usize].to_string(),
            text: String::new(),
            embedding: Vec::new(),
            importance: r.importance,
            created_at_unix_ns: r.created_at_unix_ns as u128,
        })
    }

    fn insert(&mut self, m: &Memory) {
        let idx = usize::try_from(m.id).expect("row id fits in usize");
        if idx >= self.rows.len() {
            self.rows.resize(idx + 1, EMPTY);
        }
        let key = self.key_id(&m.key);
        if self.rows[idx].key == VACANT {
            self.live += 1;
        }
        self.rows[idx] = LeanRow {
            key,
            importance: m.importance,
            created_at_unix_ns: u64::try_from(m.created_at_unix_ns).unwrap_or(u64::MAX),
        };
    }

    fn remove(&mut self, id: u64) -> bool {
        let Some(r) = usize::try_from(id).ok().and_then(|i| self.rows.get_mut(i)) else {
            return false;
        };
        if r.key == VACANT {
            return false;
        }
        *r = EMPTY;
        self.live -= 1;
        true
    }
}

/// Row map: full rows in a hash map, or lean rows packed (see module docs).
pub(crate) enum RowMap {
    Full(HashMap<u64, Memory>),
    Lean(LeanRows),
}

impl RowMap {
    pub(crate) fn new(lean: bool, capacity: usize) -> Self {
        if lean {
            RowMap::Lean(LeanRows {
                rows: Vec::with_capacity(capacity),
                ..Default::default()
            })
        } else {
            RowMap::Full(HashMap::with_capacity(capacity))
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            RowMap::Full(m) => m.len(),
            RowMap::Lean(l) => l.live,
        }
    }

    pub(crate) fn contains(&self, id: u64) -> bool {
        match self {
            RowMap::Full(m) => m.contains_key(&id),
            RowMap::Lean(l) => usize::try_from(id)
                .ok()
                .and_then(|i| l.rows.get(i))
                .is_some_and(|r| r.key != VACANT),
        }
    }

    /// The row; for lean rows text and embedding are empty.
    pub(crate) fn get(&self, id: u64) -> Option<Memory> {
        match self {
            RowMap::Full(m) => m.get(&id).cloned(),
            RowMap::Lean(l) => l.get(id),
        }
    }

    /// Insert a row. A lean map keeps only key, importance and time.
    pub(crate) fn insert(&mut self, m: Memory) {
        match self {
            RowMap::Full(map) => {
                map.insert(m.id, m);
            }
            RowMap::Lean(l) => l.insert(&m),
        }
    }

    pub(crate) fn remove(&mut self, id: u64) -> bool {
        match self {
            RowMap::Full(m) => m.remove(&id).is_some(),
            RowMap::Lean(l) => l.remove(id),
        }
    }

    /// Every live row (lean rows without text or embedding).
    pub(crate) fn all(&self) -> Vec<Memory> {
        match self {
            RowMap::Full(m) => m.values().cloned().collect(),
            RowMap::Lean(l) => (0..l.rows.len() as u64)
                .filter_map(|id| l.get(id))
                .collect(),
        }
    }

    /// (id, effective importance) of every live row, for eviction.
    pub(crate) fn importances(&self, half_life: std::time::Duration) -> Vec<(u64, f32)> {
        match self {
            RowMap::Full(m) => m
                .values()
                .map(|m| (m.id, m.effective_importance(half_life)))
                .collect(),
            RowMap::Lean(l) => (0..l.rows.len() as u64)
                .filter_map(|id| l.get(id))
                .map(|m| (m.id, m.effective_importance(half_life)))
                .collect(),
        }
    }

    /// Bytes held by a lean map (`None` for full rows).
    pub(crate) fn lean_bytes(&self) -> Option<usize> {
        match self {
            RowMap::Full(_) => None,
            RowMap::Lean(l) => Some(
                l.rows.capacity() * std::mem::size_of::<LeanRow>()
                    + l.keys.iter().map(|k| k.len() + 40).sum::<usize>(),
            ),
        }
    }
}

impl std::fmt::Debug for RowMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RowMap::Full(m) => f
                .debug_struct("RowMap::Full")
                .field("len", &m.len())
                .finish(),
            RowMap::Lean(l) => f
                .debug_struct("RowMap::Lean")
                .field("len", &l.live)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: u64, key: &str) -> Memory {
        Memory {
            id,
            key: key.into(),
            text: "t".into(),
            embedding: vec![1.0],
            importance: 0.5,
            created_at_unix_ns: 1_700_000_000_000_000_000,
        }
    }

    #[test]
    fn lean_rows_keep_metadata_only() {
        let mut r = RowMap::new(true, 0);
        r.insert(mem(3, "web"));
        r.insert(mem(5, "web"));
        r.insert(mem(6, "agent"));
        assert_eq!(r.len(), 3);
        assert!(r.contains(5) && !r.contains(4) && !r.contains(99));
        let m = r.get(6).unwrap();
        assert_eq!((m.key.as_str(), m.importance), ("agent", 0.5));
        assert_eq!(m.created_at_unix_ns, 1_700_000_000_000_000_000);
        assert!(m.text.is_empty() && m.embedding.is_empty());
        assert!(r.remove(5) && !r.remove(5));
        assert_eq!(r.len(), 2);
        assert_eq!(r.all().iter().map(|m| m.id).collect::<Vec<_>>(), vec![3, 6]);
        // re-inserting an id does not double count
        r.insert(mem(3, "web"));
        assert_eq!(r.len(), 2);
        assert_eq!(r.importances(std::time::Duration::from_secs(1)).len(), 2);
        if let RowMap::Lean(l) = &r {
            assert_eq!(l.keys.len(), 2, "keys are interned");
        }
    }

    #[test]
    fn lean_row_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<LeanRow>(), 16);
    }

    #[test]
    fn full_rows_behave_like_a_map() {
        let mut r = RowMap::new(false, 0);
        r.insert(mem(1, "k"));
        assert_eq!(r.get(1).unwrap().text, "t");
        assert!(r.remove(1));
        assert_eq!(r.len(), 0);
    }
}
