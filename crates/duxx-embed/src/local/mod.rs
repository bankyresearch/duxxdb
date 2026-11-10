//! In-process transformer models on Candle (feature `local`).
//!
//! - [`LocalEmbedder`] — `multilingual-e5-small` sentence embeddings
//!   (384-d, ~100 languages including Hindi), selected with the spec
//!   `local:multilingual-e5-small`.
//! - [`hub`] — pinned model downloads, weight/tokenizer loading and the
//!   batching planner; `duxx-rerank` reuses it for its cross-encoders.

pub mod hub;

mod e5;
pub use e5::{LocalEmbedder, EMBEDDING_MODELS};

/// Turn a Candle error into a DuxxDB error.
pub fn candle_err(e: candle_core::Error) -> duxx_core::Error {
    duxx_core::Error::Internal(format!("candle: {e}"))
}

/// Turn a tokenizers error into a DuxxDB error.
pub fn tok_err(e: tokenizers::Error) -> duxx_core::Error {
    duxx_core::Error::Internal(format!("tokenizer: {e}"))
}
