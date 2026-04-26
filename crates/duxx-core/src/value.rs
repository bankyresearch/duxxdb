//! Dynamically-typed cell values used for inserts and query results.

use serde::{Deserialize, Serialize};

/// A single cell value. When writing Rust code you generally use concrete
/// types — `Value` is the lingua franca for dynamic APIs (bindings, CLI).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    I64(i64),
    F64(f64),
    Bool(bool),
    Text(String),
    Bytes(Vec<u8>),
    Vector(Vec<f32>),
    Timestamp(i64),
    Json(serde_json::Value),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl From<i64> for Value { fn from(v: i64) -> Self { Value::I64(v) } }
impl From<f64> for Value { fn from(v: f64) -> Self { Value::F64(v) } }
impl From<bool> for Value { fn from(v: bool) -> Self { Value::Bool(v) } }
impl From<String> for Value { fn from(v: String) -> Self { Value::Text(v) } }
impl From<&str> for Value { fn from(v: &str) -> Self { Value::Text(v.to_string()) } }
impl From<Vec<f32>> for Value { fn from(v: Vec<f32>) -> Self { Value::Vector(v) } }
