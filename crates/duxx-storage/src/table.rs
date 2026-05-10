//! Row-oriented schema-aware table — placeholder for the future Lance
//! / Arrow columnar backend (see PHASE_2_3_PLAN.md).
//!
//! Not currently used in the hot path; `MemoryStore` persists via the
//! byte-keyed [`crate::Storage`] trait instead.

use duxx_core::{Column, ColumnKind, Error, Result, Schema, Value, VectorSpec};
use parking_lot::RwLock;
use std::sync::Arc;

/// Monotonic row id.
pub type RowId = u64;

/// A row is an ordered list of values, one per schema column.
pub type Row = Vec<Value>;

/// In-memory schema-validated table. Cheap-cloned (`Arc` internals).
#[derive(Debug, Clone)]
pub struct Table {
    inner: Arc<TableInner>,
}

#[derive(Debug)]
struct TableInner {
    schema: Schema,
    rows: RwLock<Vec<(RowId, Row)>>,
    next_id: RwLock<RowId>,
}

impl Table {
    pub fn new(schema: Schema) -> Self {
        Self {
            inner: Arc::new(TableInner {
                schema,
                rows: RwLock::new(Vec::new()),
                next_id: RwLock::new(1),
            }),
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.inner.schema
    }

    pub fn insert(&self, row: Row) -> Result<RowId> {
        let cols = &self.inner.schema.columns;
        if row.len() != cols.len() {
            return Err(Error::Schema(format!(
                "row has {} values; schema has {} columns",
                row.len(),
                cols.len()
            )));
        }
        for (value, col) in row.iter().zip(cols.iter()) {
            check_value(value, col)?;
        }
        let id = {
            let mut n = self.inner.next_id.write();
            let id = *n;
            *n += 1;
            id
        };
        self.inner.rows.write().push((id, row));
        tracing::trace!(id, "inserted row");
        Ok(id)
    }

    pub fn len(&self) -> usize {
        self.inner.rows.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scan(&self) -> Vec<(RowId, Row)> {
        self.inner.rows.read().clone()
    }

    pub fn get(&self, id: RowId) -> Option<Row> {
        self.inner
            .rows
            .read()
            .iter()
            .find(|(rid, _)| *rid == id)
            .map(|(_, r)| r.clone())
    }
}

fn check_value(value: &Value, col: &Column) -> Result<()> {
    if value.is_null() {
        return if col.nullable {
            Ok(())
        } else {
            Err(Error::Schema(format!("column '{}' is not nullable", col.name)))
        };
    }
    let ok = matches!(
        (&col.kind, value),
        (ColumnKind::I64, Value::I64(_))
            | (ColumnKind::F64, Value::F64(_))
            | (ColumnKind::Bool, Value::Bool(_))
            | (ColumnKind::Text { .. }, Value::Text(_))
            | (ColumnKind::Timestamp, Value::Timestamp(_))
            | (ColumnKind::Json, Value::Json(_))
    );
    if ok {
        return Ok(());
    }
    if let (ColumnKind::Vector(VectorSpec { dim, .. }), Value::Vector(v)) = (&col.kind, value) {
        if v.len() == *dim {
            return Ok(());
        }
        return Err(Error::Schema(format!(
            "column '{}' expects vector dim {}, got {}",
            col.name,
            dim,
            v.len()
        )));
    }
    Err(Error::Schema(format!(
        "column '{}' type mismatch: col={:?} value={:?}",
        col.name, col.kind, value
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_core::VectorSpec;

    fn test_schema() -> Schema {
        Schema::new(vec![
            Column::new("id", ColumnKind::I64),
            Column::new("msg", ColumnKind::Text { bm25: true }),
            Column::new("emb", ColumnKind::Vector(VectorSpec::new(3))),
        ])
    }

    #[test]
    fn insert_and_get() {
        let t = Table::new(test_schema());
        let id = t
            .insert(vec![
                Value::I64(1),
                Value::Text("hi".into()),
                Value::Vector(vec![0.1, 0.2, 0.3]),
            ])
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(t.len(), 1);
        assert!(t.get(id).is_some());
    }

    #[test]
    fn arity_mismatch_rejected() {
        let t = Table::new(test_schema());
        let err = t.insert(vec![Value::I64(1)]).unwrap_err();
        assert!(matches!(err, Error::Schema(_)));
    }

    #[test]
    fn vector_dim_mismatch_rejected() {
        let t = Table::new(test_schema());
        let err = t
            .insert(vec![
                Value::I64(1),
                Value::Text("hi".into()),
                Value::Vector(vec![0.1]),
            ])
            .unwrap_err();
        assert!(matches!(err, Error::Schema(_)));
    }
}
