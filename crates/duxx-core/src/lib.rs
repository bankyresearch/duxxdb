//! # duxx-core
//!
//! Shared types and errors used across every DuxxDB crate. Keep this
//! crate dependency-light — everything here is pulled into every other
//! crate in the workspace.

pub mod error;
pub mod schema;
pub mod value;

pub use error::{Error, Result};
pub use schema::{Column, ColumnKind, Schema, VectorSpec};
pub use value::Value;

/// DuxxDB version — matches the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!VERSION.is_empty());
    }
}
