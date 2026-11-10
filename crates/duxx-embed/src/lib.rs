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
//! With the `local` feature, a real multilingual model runs in-process
//! on Candle (pure Rust, CPU) — no Python, no external service:
//!
//! ```no_run
//! # #[cfg(feature = "local")] {
//! use duxx_embed::{Embedder, LocalEmbedder};
//! let e = LocalEmbedder::multilingual_e5_small().unwrap(); // downloads once
//! let doc = e.embed("नई दिल्ली भारत की राजधानी है").unwrap();   // passage
//! let q = e.embed_query("capital of India").unwrap();        // query
//! assert_eq!(doc.len(), 384);
//! # }
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

#[cfg(feature = "local")]
pub mod local;
#[cfg(feature = "local")]
pub use local::LocalEmbedder;

/// An embedding provider — turns a string into a fixed-length vector.
///
/// Implementations must be cheap to clone (typically `Arc` internally)
/// and `Send + Sync` so they can be shared across server threads.
///
/// Only `embed` and `dim` are required. Models trained with asymmetric
/// query/document inputs (e5: `query: ` vs `passage: `) override
/// [`Embedder::embed_query`]; stored text always goes through `embed`.
pub trait Embedder: Send + Sync {
    /// Embed text that will be stored and searched over (a document,
    /// a memory, a passage).
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Dimension of every vector this embedder returns.
    fn dim(&self) -> usize;

    /// Embed a search query. Defaults to [`Embedder::embed`].
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    /// Embed many stored texts at once. Defaults to one `embed` per
    /// text; local models override it to batch through the network.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Stable identity of the vector space, e.g. `hash:32` or
    /// `local:intfloat/multilingual-e5-small@614241f622f5`. Persistent
    /// stores record it and refuse to reopen under a different one,
    /// because vectors from two embedders are not comparable even when
    /// their dimensions happen to match.
    fn id(&self) -> String {
        format!("custom:{}", self.dim())
    }
}

impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed(text)
    }
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text)
    }
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }
    fn id(&self) -> String {
        (**self).id()
    }
}

impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed(text)
    }
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        (**self).embed_query(text)
    }
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_batch(texts)
    }
    fn id(&self) -> String {
        (**self).id()
    }
}

/// Build an embedder from a string spec — convenient for env-var config.
///
/// Recognized specs:
/// - `"hash:<dim>"`                     — `HashEmbedder` of given dim
/// - `"openai:<model>"` (with `openai` feature) — uses `OPENAI_API_KEY` env
/// - `"cohere:<model>"` (with `cohere` feature) — uses `COHERE_API_KEY` env
/// - `"local:<model>[,opt=val]*"` (with `local` feature) — in-process
///   Candle model, e.g. `local:multilingual-e5-small` (384-d). `<model>`
///   is a built-in alias (downloaded once into `DUXX_MODEL_DIR`) or a
///   model directory. Options: `max_tokens`, `batch`, `prefix=e5|none`.
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
            let key = std::env::var("OPENAI_API_KEY")
                .map_err(|_| duxx_core::Error::Internal("OPENAI_API_KEY not set".to_string()))?;
            let dim = openai::default_dim_for_model(rest);
            Ok(Some(Box::new(OpenAIEmbedder::new(key, rest, dim))))
        }
        #[cfg(feature = "cohere")]
        "cohere" => {
            let key = std::env::var("COHERE_API_KEY")
                .map_err(|_| duxx_core::Error::Internal("COHERE_API_KEY not set".to_string()))?;
            let dim = cohere::default_dim_for_model(rest);
            Ok(Some(Box::new(CohereEmbedder::new(key, rest, dim))))
        }
        #[cfg(feature = "local")]
        "local" => Ok(Some(Box::new(LocalEmbedder::from_spec(rest)?))),
        other => Err(duxx_core::Error::Internal(format!(
            "unknown embedder kind: {other} (built-in features: hash{}{}{})",
            if cfg!(feature = "openai") {
                ", openai"
            } else {
                ""
            },
            if cfg!(feature = "cohere") {
                ", cohere"
            } else {
                ""
            },
            if cfg!(feature = "local") {
                ", local"
            } else {
                " — rebuild with the `local` feature for local:<model>"
            },
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
    fn ids_name_the_vector_space() {
        assert_eq!(HashEmbedder::new(32).id(), "hash:32");
        let boxed: Box<dyn Embedder> = from_spec(Some("hash:16")).unwrap().unwrap();
        assert_eq!(boxed.id(), "hash:16");
        let arc: std::sync::Arc<dyn Embedder> = std::sync::Arc::from(boxed);
        assert_eq!(arc.id(), "hash:16");
        // Defaults: query and batch paths agree with `embed`.
        assert_eq!(arc.embed_query("a b").unwrap(), arc.embed("a b").unwrap());
        let b = arc.embed_batch(&["a", "b c"]).unwrap();
        assert_eq!(b[1], arc.embed("b c").unwrap());
    }

    #[cfg(not(feature = "local"))]
    #[test]
    fn local_without_feature_says_how_to_enable() {
        let err = from_spec(Some("local:multilingual-e5-small"))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("local"), "{err}");
    }

    #[test]
    fn from_spec_none() {
        assert!(from_spec(None).unwrap().is_none());
    }
}
