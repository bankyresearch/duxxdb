//! # duxx-embed
//!
//! Embedding providers for DuxxDB.
//!
//! Until now we've shipped a hash-bucket "toy" embedder duplicated in
//! every binary. This crate centralizes that, and adds first-class
//! OpenAI and Cohere providers behind feature flags.
//!
//! ## Quickstart
//!
//! ```no_run
//! use duxx_embed::{Embedder, HashEmbedder};
//!
//! let e = HashEmbedder::new(32);
//! let v = e.embed("hello world").unwrap();
//! assert_eq!(v.len(), 32);
//! ```
//!
//! With the `openai` feature:
//!
//! ```no_run
//! # #[cfg(feature = "openai")] {
//! use duxx_embed::{Embedder, OpenAIEmbedder};
//! let e = OpenAIEmbedder::small(std::env::var("OPENAI_API_KEY").unwrap());
//! let v = e.embed("hello").unwrap();   // 1536-d
//! # }
//! ```

use duxx_core::Result;

mod hash;
pub use hash::HashEmbedder;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAIEmbedder;

#[cfg(feature = "cohere")]
mod cohere;
#[cfg(feature = "cohere")]
pub use cohere::CohereEmbedder;

/// An embedding provider — turns a string into a fixed-length vector.
///
/// Implementations must be cheap to clone (typically `Arc` internally)
/// and `Send + Sync` so they can be shared across server threads.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
}

impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed(text)
    }
    fn dim(&self) -> usize {
        (**self).dim()
    }
}

/// Build an embedder from a string spec — convenient for env-var config.
///
/// Recognized specs:
/// - `"hash:<dim>"`                     — `HashEmbedder` of given dim
/// - `"openai:<model>"` (with `openai` feature) — uses `OPENAI_API_KEY` env
/// - `"cohere:<model>"` (with `cohere` feature) — uses `COHERE_API_KEY` env
///
/// Returns `Ok(None)` if no spec is given (caller picks a default).
pub fn from_spec(spec: Option<&str>) -> Result<Option<Box<dyn Embedder>>> {
    let Some(spec) = spec else { return Ok(None) };
    let (kind, rest) = spec
        .split_once(':')
        .ok_or_else(|| duxx_core::Error::Internal(format!("bad embedder spec: {spec}")))?;
    match kind {
        "hash" => {
            let dim: usize = rest.parse().map_err(|_| {
                duxx_core::Error::Internal(format!("hash dim must be int, got {rest}"))
            })?;
            Ok(Some(Box::new(HashEmbedder::new(dim))))
        }
        #[cfg(feature = "openai")]
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                duxx_core::Error::Internal("OPENAI_API_KEY not set".to_string())
            })?;
            let dim = openai::default_dim_for_model(rest);
            Ok(Some(Box::new(OpenAIEmbedder::new(key, rest, dim))))
        }
        #[cfg(feature = "cohere")]
        "cohere" => {
            let key = std::env::var("COHERE_API_KEY").map_err(|_| {
                duxx_core::Error::Internal("COHERE_API_KEY not set".to_string())
            })?;
            let dim = cohere::default_dim_for_model(rest);
            Ok(Some(Box::new(CohereEmbedder::new(key, rest, dim))))
        }
        other => Err(duxx_core::Error::Internal(format!(
            "unknown embedder kind: {other} (built-in features: hash{}{})",
            if cfg!(feature = "openai") { ", openai" } else { "" },
            if cfg!(feature = "cohere") { ", cohere" } else { "" },
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_spec_hash() {
        let e = from_spec(Some("hash:32")).unwrap().unwrap();
        assert_eq!(e.dim(), 32);
        let v = e.embed("hello").unwrap();
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn from_spec_unknown_errors() {
        assert!(from_spec(Some("nope:foo")).is_err());
    }

    #[test]
    fn from_spec_none() {
        assert!(from_spec(None).unwrap().is_none());
    }
}
