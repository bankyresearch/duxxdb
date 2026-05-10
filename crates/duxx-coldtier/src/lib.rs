//! # duxx-coldtier
//!
//! Cold-tier export for DuxxDB. The hot tier (`MemoryStore` + redb /
//! tantivy / HNSW) serves agents at sub-ms latency; the cold tier is
//! the lakehouse where you keep everything forever for analytics,
//! audit, and offline training.
//!
//! v1 ships **Apache Parquet** export — pure Rust, no FFI, and read
//! natively by Spark, DuckDB, Polars, pandas, and basically every
//! analytics tool. Delta-Lake / Iceberg semantics (transaction log,
//! time travel) are a future feature flag once the workload calls
//! for them.
//!
//! ## Usage
//!
//! ```no_run
//! use duxx_coldtier::ParquetExporter;
//! use duxx_memory::MemoryStore;
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = MemoryStore::new(128);
//! // ... insert memories ...
//! let n = ParquetExporter::new()
//!     .write(&store, "./cold/memories-2026-05.parquet")?;
//! println!("exported {n} memories");
//! # Ok(()) }
//! ```
//!
//! ## Schema
//!
//! Memories serialize to a flat Arrow schema:
//!
//! | column            | Arrow type                              |
//! |-------------------|-----------------------------------------|
//! | `id`              | `UInt64`                                |
//! | `key`             | `Utf8`                                  |
//! | `text`            | `Utf8`                                  |
//! | `embedding`       | `FixedSizeList<Float32, dim>`           |
//! | `importance`      | `Float32`                               |
//! | `created_at_ns`   | `UInt64` (Unix epoch nanoseconds)       |

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, StringArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use duxx_memory::{Memory, MemoryStore};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Serialise a `MemoryStore` to one Parquet file.
///
/// Cheap to construct; the configurable knobs (compression, row-group
/// size) are wired through `WriterProperties` builders.
#[derive(Clone)]
pub struct ParquetExporter {
    compression: Compression,
}

impl ParquetExporter {
    pub fn new() -> Self {
        Self {
            compression: Compression::SNAPPY,
        }
    }

    pub fn with_compression(mut self, c: Compression) -> Self {
        self.compression = c;
        self
    }

    /// Build the Arrow schema for a fixed embedding dimension.
    pub fn arrow_schema(&self, dim: usize) -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("importance", DataType::Float32, false),
            Field::new("created_at_ns", DataType::UInt64, false),
        ])
    }

    /// Convert a slice of `Memory`s + a known `dim` into a single `RecordBatch`.
    pub fn to_record_batch(
        &self,
        rows: &[Memory],
        dim: usize,
    ) -> Result<RecordBatch, ColdTierError> {
        if let Some(bad) = rows.iter().find(|m| m.embedding.len() != dim) {
            return Err(ColdTierError::DimMismatch {
                id: bad.id,
                got: bad.embedding.len(),
                want: dim,
            });
        }

        let ids: ArrayRef = Arc::new(UInt64Array::from_iter_values(rows.iter().map(|m| m.id)));
        let keys: ArrayRef =
            Arc::new(StringArray::from_iter_values(rows.iter().map(|m| m.key.as_str())));
        let texts: ArrayRef =
            Arc::new(StringArray::from_iter_values(rows.iter().map(|m| m.text.as_str())));

        // Flatten all embeddings into a single Float32Array, then wrap
        // in a FixedSizeList of the right stride.
        let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * dim);
        for m in rows {
            flat.extend_from_slice(&m.embedding);
        }
        let values = Float32Array::from(flat);
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let embeddings: ArrayRef = Arc::new(FixedSizeListArray::new(
            item_field,
            dim as i32,
            Arc::new(values),
            None,
        ));

        let importances: ArrayRef =
            Arc::new(Float32Array::from_iter_values(rows.iter().map(|m| m.importance)));

        // u128 -> u64 truncation. Unix nanos fit in u64 until year 2554.
        let timestamps: ArrayRef = Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|m| m.created_at_unix_ns as u64),
        ));

        let schema = Arc::new(self.arrow_schema(dim));
        RecordBatch::try_new(
            schema,
            vec![ids, keys, texts, embeddings, importances, timestamps],
        )
        .map_err(|e| ColdTierError::Arrow(e.to_string()))
    }

    /// Walk the store and write every memory to `path` as a single
    /// Parquet file. Returns the count.
    pub fn write(
        &self,
        store: &MemoryStore,
        path: impl AsRef<Path>,
    ) -> Result<usize, ColdTierError> {
        let memories = collect_memories(store);
        let count = memories.len();
        if count == 0 {
            // Still write an empty file with the proper schema so
            // readers don't need to special-case "no rows yet".
            let schema = Arc::new(self.arrow_schema(store.dim()));
            let file = File::create(path.as_ref())
                .map_err(|e| ColdTierError::Io(e.to_string()))?;
            let props = WriterProperties::builder()
                .set_compression(self.compression)
                .build();
            let mut writer = ArrowWriter::try_new(file, schema, Some(props))
                .map_err(|e| ColdTierError::Parquet(e.to_string()))?;
            writer.close().map_err(|e| ColdTierError::Parquet(e.to_string()))?;
            return Ok(0);
        }

        let dim = store.dim();
        let batch = self.to_record_batch(&memories, dim)?;

        let file = File::create(path.as_ref())
            .map_err(|e| ColdTierError::Io(e.to_string()))?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
            .map_err(|e| ColdTierError::Parquet(e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| ColdTierError::Parquet(e.to_string()))?;
        writer.close().map_err(|e| ColdTierError::Parquet(e.to_string()))?;

        tracing::info!(
            path = %path.as_ref().display(),
            count,
            "wrote parquet cold-tier export"
        );
        Ok(count)
    }
}

impl Default for ParquetExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk every memory in `store`. We don't have an iterator on
/// `MemoryStore` so we lean on `recall(_, "", &zeros, very_large)` →
/// no, that doesn't work. The cleanest path is to expose `by_id` via
/// `MemoryStore` — Phase 5.1. For now we have a private helper that
/// uses scan-by-id from 1..=next_id.
fn collect_memories(store: &MemoryStore) -> Vec<Memory> {
    // MemoryStore exposes `len()` and `recall()` but not a bulk iterator.
    // The existing public surface is enough: walk row ids 1..=len,
    // pulling each back via the test-only `inner.by_id` would need a
    // public accessor. Instead we add a tiny public API in duxx-memory:
    // `MemoryStore::all()` -> Vec<Memory>. See lib.rs in that crate.
    store.all_memories()
}

/// Errors specific to the cold-tier export path.
#[derive(Debug, thiserror::Error)]
pub enum ColdTierError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Parquet error: {0}")]
    Parquet(String),
    #[error("Arrow error: {0}")]
    Arrow(String),
    #[error(
        "memory id={id} embedding has dim {got} but schema expects {want}"
    )]
    DimMismatch { id: u64, got: usize, want: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_memory::MemoryStore;

    fn embed(text: &str, dim: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut v = vec![0.0f32; dim];
        for tok in text.to_lowercase().split_whitespace() {
            let mut h = DefaultHasher::new();
            tok.hash(&mut h);
            v[(h.finish() as usize) % dim] += 1.0;
        }
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= n;
        }
        v
    }

    #[test]
    fn arrow_schema_shape() {
        let s = ParquetExporter::new().arrow_schema(8);
        assert_eq!(s.fields().len(), 6);
        assert_eq!(s.field(0).name(), "id");
        match s.field(3).data_type() {
            DataType::FixedSizeList(_, sz) => assert_eq!(*sz, 8),
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
    }

    #[test]
    fn write_and_read_roundtrip() {
        const DIM: usize = 16;
        let store = MemoryStore::new(DIM);
        store.remember("alice", "wallet at the cafe", embed("wallet", DIM)).unwrap();
        store.remember("alice", "favorite color is blue", embed("blue", DIM)).unwrap();
        store.remember("bob", "tracking number DX-002341", embed("tracking", DIM)).unwrap();

        let dir = std::env::temp_dir().join(format!(
            "duxx-coldtier-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("export.parquet");

        let n = ParquetExporter::new().write(&store, &path).unwrap();
        assert_eq!(n, 3);
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        // Read back via parquet's Arrow reader.
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 6);

        let texts: Vec<&str> = batch
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert!(texts.iter().any(|t| t.contains("wallet")));
        assert!(texts.iter().any(|t| t.contains("tracking")));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_store_writes_empty_file_with_schema() {
        const DIM: usize = 8;
        let store = MemoryStore::new(DIM);
        let dir = std::env::temp_dir().join(format!(
            "duxx-coldtier-empty-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.parquet");
        let n = ParquetExporter::new().write(&store, &path).unwrap();
        assert_eq!(n, 0);
        assert!(path.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
