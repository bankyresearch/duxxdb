//! Error + Result types shared across every DuxxDB crate.

/// Canonical DuxxDB error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Filesystem or network I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A value did not match the declared schema.
    #[error("schema error: {0}")]
    Schema(String),

    /// An index build or lookup failed.
    #[error("index error: {0}")]
    Index(String),

    /// Storage-layer failure (Lance, redb, etc).
    #[error("storage error: {0}")]
    Storage(String),

    /// A key or row was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Serialization / deserialization failure.
    #[error("serde error: {0}")]
    Serde(String),

    /// Catch-all for unclassified failures.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serde(e.to_string())
    }
}

/// Workspace-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
