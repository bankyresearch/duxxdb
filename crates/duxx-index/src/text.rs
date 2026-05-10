//! Tantivy-backed BM25 text index with batched commits.
//!
//! Phase 2.5: instead of committing tantivy on every `insert` (slow:
//! ~4 ms each), we commit every `commit_every` inserts. Reads (`search`,
//! `len`) auto-flush any pending writes so the "insert-then-search"
//! contract still holds.
//!
//! Defaults to `commit_every = 100`. Override at construction with
//! [`TextIndex::with_commit_every`].

use duxx_core::{Error, Result};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    doc,
    query::QueryParser,
    schema::{Field, Schema, Value, FAST, INDEXED, STORED, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};

/// Default insert count between forced commits.
pub const DEFAULT_COMMIT_EVERY: usize = 100;

/// BM25 full-text index, in-memory, Send + Sync.
pub struct TextIndex {
    id_field: Field,
    text_field: Field,
    index: Index,
    writer: Arc<Mutex<IndexWriter>>,
    reader: IndexReader,
    pending: AtomicUsize,
    commit_every: usize,
}

impl TextIndex {
    /// Build with the default commit threshold ([`DEFAULT_COMMIT_EVERY`]).
    pub fn new() -> Self {
        Self::with_commit_every(DEFAULT_COMMIT_EVERY)
    }

    /// Build with a custom commit threshold. Use a low value (1) when
    /// every insert must be immediately durable; higher values amortize
    /// commit overhead across batches.
    pub fn with_commit_every(commit_every: usize) -> Self {
        Self::try_with(commit_every).expect("tantivy in-RAM index init must not fail")
    }

    fn try_with(commit_every: usize) -> Result<Self> {
        let (id_field, text_field, schema) = build_schema();
        let index = Index::create_in_ram(schema);
        Self::from_index(index, id_field, text_field, commit_every)
    }

    /// Open a disk-backed tantivy index at `dir`. The directory is
    /// created if missing. Reopening an existing dir restores the
    /// previously written state — the writer queue is empty, so the
    /// caller can immediately commit new docs without rebuild.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_commit_every(dir, DEFAULT_COMMIT_EVERY)
    }

    pub fn open_with_commit_every(
        dir: impl AsRef<Path>,
        commit_every: usize,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir.as_ref())
            .map_err(|e| Error::Index(format!("create tantivy dir: {e}")))?;
        let (id_field, text_field, schema) = build_schema();
        let mmap = MmapDirectory::open(dir.as_ref())
            .map_err(|e| Error::Index(format!("tantivy mmap dir: {e}")))?;
        let index = Index::open_or_create(mmap, schema)
            .map_err(|e| Error::Index(format!("tantivy open_or_create: {e}")))?;
        Self::from_index(index, id_field, text_field, commit_every)
    }

    fn from_index(
        index: Index,
        id_field: Field,
        text_field: Field,
        commit_every: usize,
    ) -> Result<Self> {
        let writer: IndexWriter = index
            .writer(50_000_000)
            .map_err(|e| Error::Index(format!("tantivy writer: {e}")))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::Index(format!("tantivy reader: {e}")))?;

        Ok(Self {
            id_field,
            text_field,
            index,
            writer: Arc::new(Mutex::new(writer)),
            reader,
            pending: AtomicUsize::new(0),
            commit_every: commit_every.max(1),
        })
    }

    /// Insert a document. Commits if the threshold is reached.
    pub fn insert(&mut self, id: u64, text: String) -> Result<()> {
        let mut w = self.writer.lock();
        w.add_document(doc!(self.id_field => id, self.text_field => text))
            .map_err(|e| Error::Index(format!("add_document: {e}")))?;
        let n = self.pending.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= self.commit_every {
            w.commit()
                .map_err(|e| Error::Index(format!("commit: {e}")))?;
            self.pending.store(0, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Force-commit any pending writes. Idempotent.
    pub fn flush(&self) -> Result<()> {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return Ok(());
        }
        let mut w = self.writer.lock();
        w.commit()
            .map_err(|e| Error::Index(format!("commit: {e}")))?;
        self.pending.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// BM25 search; auto-flushes pending writes first.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        if let Err(e) = self.flush() {
            tracing::warn!("flush before search failed: {e}");
            return Vec::new();
        }
        if let Err(e) = self.reader.reload() {
            tracing::warn!("tantivy reader reload failed: {e}");
            return Vec::new();
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let parsed = match parser.parse_query(query) {
            Ok(q) => q,
            Err(e) => {
                tracing::debug!("query parse failed: {e}");
                return Vec::new();
            }
        };
        let docs = match searcher.search(&parsed, &TopDocs::with_limit(k)) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("tantivy search failed: {e}");
                return Vec::new();
            }
        };
        docs.into_iter()
            .filter_map(|(score, addr)| {
                let d: TantivyDocument = searcher.doc(addr).ok()?;
                let v = d.get_first(self.id_field)?;
                let id = v.as_u64()?;
                Some((id, score))
            })
            .collect()
    }

    /// Number of documents — auto-flushes pending writes first.
    pub fn len(&self) -> usize {
        if self.flush().is_err() {
            return 0;
        }
        if self.reader.reload().is_err() {
            return 0;
        }
        self.reader.searcher().num_docs() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build the canonical (id_field, text_field, schema) triple. Matches
/// the on-disk schema so re-opens succeed.
fn build_schema() -> (Field, Field, Schema) {
    let mut sb = Schema::builder();
    let id_field = sb.add_u64_field("id", STORED | INDEXED | FAST);
    let text_field = sb.add_text_field("text", TEXT);
    let schema = sb.build();
    (id_field, text_field, schema)
}

impl Default for TextIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TextIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextIndex")
            .field("len", &self.len())
            .field("commit_every", &self.commit_every)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_match_finds_only_relevant_doc() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund my order".into()).unwrap();
        idx.insert(2, "weather today".into()).unwrap();
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn bm25_ranks_higher_term_frequency_first() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund refund refund my order".into()).unwrap();
        idx.insert(2, "refund please".into()).unwrap();
        idx.insert(3, "weather forecast".into()).unwrap();
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 2, "doc 3 should be filtered out");
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let mut idx = TextIndex::new();
        idx.insert(1, "anything".into()).unwrap();
        assert!(idx.search("", 10).is_empty());
    }

    #[test]
    fn len_reflects_committed_count() {
        let mut idx = TextIndex::new();
        assert_eq!(idx.len(), 0);
        idx.insert(1, "one".into()).unwrap();
        idx.insert(2, "two".into()).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn batched_inserts_searchable_via_auto_flush() {
        // commit_every=10 means doc 1 isn't auto-committed by insert,
        // but search must still find it via auto-flush.
        let mut idx = TextIndex::with_commit_every(10);
        idx.insert(1, "single doc".into()).unwrap();
        let hits = idx.search("single", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn explicit_flush_makes_writes_durable() {
        let mut idx = TextIndex::with_commit_every(1000);
        for i in 0..50u64 {
            idx.insert(i, format!("doc {i}")).unwrap();
        }
        idx.flush().unwrap();
        assert_eq!(idx.len(), 50);
    }

    #[test]
    fn disk_backed_index_persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "duxx-tantivy-persist-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut idx = TextIndex::open(&dir).unwrap();
            idx.insert(1, "refund my order".into()).unwrap();
            idx.insert(2, "weather forecast".into()).unwrap();
            idx.flush().unwrap();
            assert_eq!(idx.len(), 2);
        } // drop closes index
        {
            let idx = TextIndex::open(&dir).unwrap();
            assert_eq!(idx.len(), 2, "data should survive reopen");
            let hits = idx.search("refund", 5);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
