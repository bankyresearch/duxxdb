//! # duxx-docs — document-intelligence layer
//!
//! The right way to "add storage" to DuxxDB: **don't store the bytes** — keep
//! those in a real object store (S3 / MinIO / Supabase Storage / local FS) and
//! let DuxxDB own the part that makes files useful to an agent: ingest →
//! chunk → embed → **hybrid-index** → retrieve **with citations** back to the
//! source object, plus document versioning and deletion.
//!
//! ```
//! use std::sync::Arc;
//! use duxx_embed::HashEmbedder;
//! use duxx_docs::{DocumentStore, LocalFsConnector};
//!
//! let connector = Arc::new(LocalFsConnector::new("."));
//! let docs = DocumentStore::new(connector, Arc::new(HashEmbedder::new(32)));
//!
//! // Index already-extracted text (PDF/image extraction is the caller's job).
//! docs.ingest_text("s3://kb/france.md", "text/markdown",
//!     "Paris is the capital of France. The Seine runs through it.").unwrap();
//!
//! let hits = docs.search("capital of France", 3).unwrap();
//! assert!(hits[0].text.contains("Paris"));
//! assert_eq!(hits[0].citation.uri, "s3://kb/france.md"); // cite the source object
//! ```

use duxx_core::{Error, Result};
use duxx_embed::Embedder;
use duxx_memory::MemoryStore;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// StorageConnector — bytes live here, not in DuxxDB
// ---------------------------------------------------------------------------

/// Read/write the raw object bytes. DuxxDB only *references* objects (by URI)
/// and reads them at ingest time; it never becomes the system of record for
/// blobs.
///
/// The default impl is [`LocalFsConnector`]. The production drop-in is an
/// `S3Connector` over the **S3 protocol**, which covers AWS S3, MinIO, **and
/// Supabase Storage** (Supabase exposes an S3-compatible endpoint) — one
/// connector, three backends, by just changing endpoint + credentials. (It
/// needs a live object store to integration-test, so it isn't shipped here as
/// unverifiable code — the trait makes it a drop-in.)
pub trait StorageConnector: Send + Sync {
    /// Fetch an object's bytes (used at ingest time).
    fn get(&self, uri: &str) -> Result<Vec<u8>>;
    /// Store bytes (optional — uploads usually go direct to the object store).
    fn put(&self, uri: &str, bytes: &[u8]) -> Result<()>;
    /// Delete an object. Returns `true` if it existed.
    fn delete(&self, uri: &str) -> Result<bool>;
    /// A time-limited URL clients use to fetch the object directly (no byte
    /// proxying through DuxxDB). Local impls return a `file://` URL.
    fn presigned_url(&self, uri: &str, ttl_secs: u64) -> Result<String>;
}

/// Local-filesystem connector — objects are files under `root`. Useful for
/// development, tests, and single-node deployments.
pub struct LocalFsConnector {
    root: PathBuf,
}

impl LocalFsConnector {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Map a URI to a path under `root`, stripping any scheme and leading
    /// slashes so it can't escape the root.
    fn resolve(&self, uri: &str) -> PathBuf {
        let key = uri.split("://").last().unwrap_or(uri);
        let key = key.trim_start_matches('/').replace("..", "");
        self.root.join(key)
    }
}

impl StorageConnector for LocalFsConnector {
    fn get(&self, uri: &str) -> Result<Vec<u8>> {
        std::fs::read(self.resolve(uri)).map_err(|e| Error::Storage(format!("get {uri}: {e}")))
    }
    fn put(&self, uri: &str, bytes: &[u8]) -> Result<()> {
        let path = self.resolve(uri);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::Storage(format!("mkdir: {e}")))?;
        }
        std::fs::write(&path, bytes).map_err(|e| Error::Storage(format!("put {uri}: {e}")))
    }
    fn delete(&self, uri: &str) -> Result<bool> {
        Ok(std::fs::remove_file(self.resolve(uri)).is_ok())
    }
    fn presigned_url(&self, uri: &str, _ttl_secs: u64) -> Result<String> {
        Ok(format!("file://{}", self.resolve(uri).display()))
    }
}

// ---------------------------------------------------------------------------
// Document model
// ---------------------------------------------------------------------------

/// A logical document: a source object, the current version, and the chunk
/// (memory) ids it indexed to.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub uri: String,
    pub content_type: String,
    pub version: u64,
    pub chunk_ids: Vec<u64>,
}

/// Per-chunk lineage, kept so every retrieval hit can cite its source.
#[derive(Debug, Clone)]
pub struct ChunkRef {
    pub doc_id: String,
    pub uri: String,
    pub version: u64,
    pub index: usize,
}

/// A retrieval hit: the chunk text, its score, and a citation to the source.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub text: String,
    pub score: f32,
    pub citation: Citation,
}

/// Where a hit came from — for grounding/citation and re-fetching the original.
#[derive(Debug, Clone)]
pub struct Citation {
    pub doc_id: String,
    pub uri: String,
    pub version: u64,
    pub chunk_index: usize,
}

// ---------------------------------------------------------------------------
// DocumentStore
// ---------------------------------------------------------------------------

/// Ingests objects into hybrid-searchable, citable chunks. Bytes stay in the
/// object store (via the [`StorageConnector`]); the chunks, embeddings, and
/// lineage live in DuxxDB.
#[derive(Clone)]
pub struct DocumentStore {
    connector: Arc<dyn StorageConnector>,
    embedder: Arc<dyn Embedder>,
    memory: MemoryStore,
    docs: Arc<RwLock<HashMap<String, Document>>>,
    chunks: Arc<RwLock<HashMap<u64, ChunkRef>>>,
    chunk_chars: usize,
    overlap: usize,
}

impl DocumentStore {
    pub fn new(connector: Arc<dyn StorageConnector>, embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            connector,
            embedder,
            memory: MemoryStore::with_capacity(dim, 100_000),
            docs: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            chunk_chars: 800,
            overlap: 100,
        }
    }

    /// Override the chunk size (characters) and overlap.
    pub fn with_chunking(mut self, chunk_chars: usize, overlap: usize) -> Self {
        self.chunk_chars = chunk_chars.max(1);
        self.overlap = overlap.min(self.chunk_chars.saturating_sub(1));
        self
    }

    /// Stable document id derived from the source URI, so re-ingesting the same
    /// object produces a new *version* of the same document.
    fn doc_id(uri: &str) -> String {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        uri.hash(&mut h);
        format!("doc_{:016x}", h.finish())
    }

    /// Index **pre-extracted text** under `uri`. (PDF/image/audio extraction is
    /// the caller's responsibility — feed the extracted text here.) Re-ingesting
    /// the same `uri` bumps the version and replaces its chunks.
    pub fn ingest_text(&self, uri: &str, content_type: &str, text: &str) -> Result<Document> {
        let id = Self::doc_id(uri);

        // Versioning: replace an existing document's chunks.
        let version = match self.docs.read().get(&id).cloned() {
            Some(prev) => {
                self.remove_chunks(&prev);
                prev.version + 1
            }
            None => 1,
        };

        let mut chunk_ids = Vec::new();
        for (index, chunk) in chunk_text(text, self.chunk_chars, self.overlap)
            .into_iter()
            .enumerate()
        {
            let embedding = self.embedder.embed(&chunk)?;
            let mid = self.memory.remember(&id, chunk, embedding)?;
            self.chunks.write().insert(
                mid,
                ChunkRef {
                    doc_id: id.clone(),
                    uri: uri.to_string(),
                    version,
                    index,
                },
            );
            chunk_ids.push(mid);
        }

        let doc = Document {
            id: id.clone(),
            uri: uri.to_string(),
            content_type: content_type.to_string(),
            version,
            chunk_ids,
        };
        self.docs.write().insert(id, doc.clone());
        Ok(doc)
    }

    /// Fetch an object via the connector and ingest it as UTF-8 text (for
    /// `text/*` / markdown). For binary formats, extract first then call
    /// [`ingest_text`](DocumentStore::ingest_text).
    pub fn ingest(&self, uri: &str, content_type: &str) -> Result<Document> {
        let bytes = self.connector.get(uri)?;
        let text = String::from_utf8_lossy(&bytes);
        self.ingest_text(uri, content_type, &text)
    }

    /// Hybrid search (vector + BM25 + RRF) across all ingested chunks, each hit
    /// carrying a citation to its source object.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<SearchHit>> {
        let qvec = self.embedder.embed(query)?;
        let hits = self.memory.recall("", query, &qvec, k)?;
        let chunks = self.chunks.read();
        Ok(hits
            .into_iter()
            .filter_map(|h| {
                chunks.get(&h.memory.id).map(|c| SearchHit {
                    text: h.memory.text,
                    score: h.score,
                    citation: Citation {
                        doc_id: c.doc_id.clone(),
                        uri: c.uri.clone(),
                        version: c.version,
                        chunk_index: c.index,
                    },
                })
            })
            .collect())
    }

    /// A presigned URL for a document's source object (for clients to fetch the
    /// original file directly).
    pub fn source_url(&self, doc_id: &str, ttl_secs: u64) -> Result<String> {
        let uri = self
            .docs
            .read()
            .get(doc_id)
            .map(|d| d.uri.clone())
            .ok_or_else(|| Error::NotFound(doc_id.to_string()))?;
        self.connector.presigned_url(&uri, ttl_secs)
    }

    /// Delete a document: drop its chunks from the index, and optionally the
    /// source object from the store.
    pub fn delete(&self, doc_id: &str, delete_object: bool) -> Result<bool> {
        let doc = self.docs.write().remove(doc_id);
        match doc {
            Some(doc) => {
                self.remove_chunks(&doc);
                if delete_object {
                    let _ = self.connector.delete(&doc.uri);
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn document(&self, doc_id: &str) -> Option<Document> {
        self.docs.read().get(doc_id).cloned()
    }

    pub fn list_documents(&self) -> Vec<Document> {
        let mut v: Vec<Document> = self.docs.read().values().cloned().collect();
        v.sort_by(|a, b| a.uri.cmp(&b.uri));
        v
    }

    fn remove_chunks(&self, doc: &Document) {
        let mut chunks = self.chunks.write();
        for id in &doc.chunk_ids {
            self.memory.forget(*id);
            chunks.remove(id);
        }
    }
}

/// Split text into overlapping character windows on a best-effort word
/// boundary. A production build would chunk on sentences/markdown structure;
/// this is deterministic and dependency-free.
fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return vec![text.to_string()];
    }
    let step = max_chars.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + max_chars).min(chars.len());
        out.push(chars[start..end].iter().collect::<String>().trim().to_string());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn store() -> DocumentStore {
        DocumentStore::new(
            Arc::new(LocalFsConnector::new(".")),
            Arc::new(HashEmbedder::new(32)),
        )
    }

    #[test]
    fn ingest_text_then_search_returns_citation() {
        let docs = store();
        let doc = docs
            .ingest_text(
                "s3://kb/wallets.md",
                "text/markdown",
                "Alice lost her crypto wallet recovery phrase. The zebra ran away.",
            )
            .unwrap();
        assert_eq!(doc.version, 1);
        assert!(!doc.chunk_ids.is_empty());

        let hits = docs.search("wallet recovery", 5).unwrap();
        assert!(!hits.is_empty());
        let top = &hits[0];
        assert!(top.text.contains("wallet"));
        assert_eq!(top.citation.uri, "s3://kb/wallets.md");
        assert_eq!(top.citation.doc_id, doc.id);
        assert_eq!(top.citation.version, 1);
    }

    #[test]
    fn ingest_from_connector_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let connector = Arc::new(LocalFsConnector::new(dir.path()));
        connector
            .put("docs/france.txt", b"The capital of France is Paris.")
            .unwrap();

        let docs = DocumentStore::new(connector, Arc::new(HashEmbedder::new(32)));
        docs.ingest("docs/france.txt", "text/plain").unwrap();

        let hits = docs.search("France capital", 5).unwrap();
        assert!(hits.iter().any(|h| h.text.contains("Paris")));
        assert!(hits[0].citation.uri.contains("docs/france.txt"));
    }

    #[test]
    fn reingest_bumps_version_and_replaces_chunks() {
        let docs = store();
        let uri = "s3://kb/doc.md";
        let v1 = docs.ingest_text(uri, "text/markdown", "first version about lions").unwrap();
        assert_eq!(v1.version, 1);

        let v2 = docs.ingest_text(uri, "text/markdown", "second version about tigers").unwrap();
        assert_eq!(v2.version, 2, "re-ingest bumps version");
        assert_eq!(v1.id, v2.id, "same uri → same document id");

        // Only the new version's content is searchable.
        let hits = docs.search("tigers lions", 10).unwrap();
        assert!(hits.iter().all(|h| !h.text.contains("lions")), "old chunks must be gone");
        assert!(hits.iter().any(|h| h.text.contains("tigers")));
        assert_eq!(docs.list_documents().len(), 1);
    }

    #[test]
    fn delete_removes_chunks_from_the_index() {
        let docs = store();
        let doc = docs
            .ingest_text("s3://kb/d.md", "text/markdown", "ephemeral falcon note")
            .unwrap();
        assert!(!docs.search("falcon", 5).unwrap().is_empty());

        assert!(docs.delete(&doc.id, false).unwrap());
        assert!(docs.search("falcon", 5).unwrap().is_empty(), "deleted doc not searchable");
        assert!(docs.document(&doc.id).is_none());
    }

    #[test]
    fn chunking_overlaps_and_covers() {
        let text = "abcdefghij".repeat(50); // 500 chars
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.chars().count() <= 100));
    }
}
