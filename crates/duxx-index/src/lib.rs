//! # duxx-index
//!
//! Vector and full-text indices for DuxxDB.
//!
//! Phase 2:
//! - [`TextIndex`] — backed by [tantivy] (BM25 over an in-RAM index).
//! - [`VectorIndex`] — exhaustive cosine. Upgraded to HNSW in Phase 2.2.
//!
//! [tantivy]: https://github.com/quickwit-oss/tantivy

pub mod text;
pub mod vector;

pub use text::TextIndex;
pub use vector::{cosine, VectorIndex};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
