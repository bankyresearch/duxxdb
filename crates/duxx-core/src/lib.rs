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

/// DuxxDB **wire-protocol / command-schema** version — the single source of
/// truth surfaced on RESP `HELLO`, gRPC `Stats`, and MCP `initialize`.
///
/// This is independent of [`VERSION`] (the release version). Compatibility
/// policy:
/// * Stable commands keep their request/response shape within a protocol
///   version. Additive changes (new optional fields, new commands) do **not**
///   bump it.
/// * A breaking change to an existing command's wire shape bumps
///   `PROTOCOL_VERSION` and is announced with a deprecation window.
///
/// Clients can gate on this to negotiate behavior across releases.
pub const PROTOCOL_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!VERSION.is_empty());
    }
}
