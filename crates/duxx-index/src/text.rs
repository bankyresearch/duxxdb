//! Tantivy-backed BM25 text index.
//!
//! Phase 2: an in-RAM tantivy index. Auto-commits on every insert —
//! correct but slow under high write rates. Phase 2.5 batches commits.
//!
//! Phase 3+: optional disk-backed mode with WAL and incremental indexing.

use duxx_core::{Error, Result};
use parking_lot::Mutex;
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Field, Schema, Value, FAST, INDEXED, STORED, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};

/// BM25 full-text index, in-memory, Send + Sync.
///
/// `insert` is `&mut self` for API symmetry with `VectorIndex`; internally
/// the writer is wrapped in `Arc<Mutex<_>>` so calls serialize even if the
/// caller cheats and goes through `&self` via `RwLock::read()`.
pub struct TextIndex {
    id_field: Field,
    text_field: Field,
    index: Index,
    writer: Arc<Mutex<IndexWriter>>,
    reader: IndexReader,
}

impl TextIndex {
    /// Construct a fresh in-RAM index. Panics on init failure (which would
    /// indicate a tantivy bug, not a user error).
    pub fn new() -> Self {
        Self::try_new().expect("tantivy in-RAM index init must not fail")
    }

    fn try_new() -> Result<Self> {
        let mut sb = Schema::builder();
        let id_field = sb.add_u64_field("id", STORED | INDEXED | FAST);
        let text_field = sb.add_text_field("text", TEXT);
        let schema = sb.build();

        let index = Index::create_in_ram(schema);
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
        })
    }

    /// Insert a document and commit so it's immediately searchable.
    pub fn insert(&mut self, id: u64, text: String) -> Result<()> {
        let mut w = self.writer.lock();
        w.add_document(doc!(self.id_field => id, self.text_field => text))
            .map_err(|e| Error::Index(format!("add_document: {e}")))?;
        w.commit()
            .map_err(|e| Error::Index(format!("commit: {e}")))?;
        Ok(())
    }

    /// BM25 search; returns `(id, score)` pairs descending by score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
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

    /// Number of committed documents.
    pub fn len(&self) -> usize {
        if self.reader.reload().is_err() {
            return 0;
        }
        self.reader.searcher().num_docs() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
        assert_eq!(hits[0].0, 1, "doc 1 has higher term frequency");
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
}
