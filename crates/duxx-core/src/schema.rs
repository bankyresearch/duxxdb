//! Schema definition for DuxxDB tables.
//!
//! A `Schema` is an ordered list of `Column`s. Each column has a `ColumnKind`
//! that determines its storage representation and which indices (if any)
//! are built for it.

use serde::{Deserialize, Serialize};

/// Vector column indexing / storage parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSpec {
    pub dim: usize,
    /// Whether to build an HNSW index. Defaults to true.
    #[serde(default = "default_true")]
    pub hnsw: bool,
    /// HNSW `M` parameter (graph degree).
    #[serde(default = "default_m")]
    pub m: u32,
    /// HNSW `ef_construction`.
    #[serde(default = "default_ef")]
    pub ef_construction: u32,
}

fn default_true() -> bool { true }
fn default_m() -> u32 { 16 }
fn default_ef() -> u32 { 64 }

impl VectorSpec {
    pub fn new(dim: usize) -> Self {
        Self { dim, hnsw: true, m: 16, ef_construction: 64 }
    }
}

/// The kind of a column — scalar, text, vector, or nested.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColumnKind {
    I64,
    F64,
    Bool,
    Text {
        #[serde(default)]
        bm25: bool,
    },
    Timestamp,
    Json,
    Vector(VectorSpec),
}

/// A single column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    #[serde(flatten)]
    pub kind: ColumnKind,
    #[serde(default)]
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, kind: ColumnKind) -> Self {
        Self { name: name.into(), kind, nullable: false }
    }

    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// A complete table schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_builder() {
        let c = Column::new("x", ColumnKind::I64).nullable(true);
        assert!(c.nullable);
        assert_eq!(c.name, "x");
    }

    #[test]
    fn schema_column_lookup() {
        let s = Schema::new(vec![
            Column::new("id", ColumnKind::I64),
            Column::new("msg", ColumnKind::Text { bm25: true }),
            Column::new("emb", ColumnKind::Vector(VectorSpec::new(1024))),
        ]);
        assert!(s.column("id").is_some());
        assert!(s.column("missing").is_none());
        assert_eq!(s.len(), 3);
    }
}
