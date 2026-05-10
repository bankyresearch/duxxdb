//! OpenAI embeddings provider.
//!
//! Uses the synchronous `reqwest::blocking` client + rustls (no OpenSSL
//! native build). Endpoint: `POST https://api.openai.com/v1/embeddings`.
//!
//! ## Models
//!
//! | Model                          | Dim   |
//! |--------------------------------|-------|
//! | `text-embedding-3-small`       | 1536  |
//! | `text-embedding-3-large`       | 3072  |
//! | `text-embedding-ada-002`       | 1536  |
//!
//! Use [`OpenAIEmbedder::small`] / [`OpenAIEmbedder::large`] as
//! convenience constructors with the right `dim` baked in.

use crate::Embedder;
use duxx_core::{Error, Result};

const ENDPOINT: &str = "https://api.openai.com/v1/embeddings";

/// OpenAI HTTP embedder.
#[derive(Debug, Clone)]
pub struct OpenAIEmbedder {
    api_key: String,
    model: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl OpenAIEmbedder {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            dim,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// `text-embedding-3-small` — 1536 dim. Fast and cheap.
    pub fn small(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-small", 1536)
    }

    /// `text-embedding-3-large` — 3072 dim. Higher quality.
    pub fn large(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-large", 3072)
    }
}

/// Best-effort default dim for an OpenAI model name.
pub(crate) fn default_dim_for_model(model: &str) -> usize {
    match model {
        "text-embedding-3-large" => 3072,
        _ => 1536, // small / ada-002 / unknown -> conservative default
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(serde::Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl Embedder for OpenAIEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp = self
            .client
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&EmbedRequest {
                model: &self.model,
                input: text,
            })
            .send()
            .map_err(|e| Error::Internal(format!("openai request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(Error::Internal(format!(
                "openai HTTP {status}: {body}"
            )));
        }

        let parsed: EmbedResponse = resp
            .json()
            .map_err(|e| Error::Internal(format!("openai parse: {e}")))?;
        let item = parsed
            .data
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("openai: empty data".to_string()))?;
        if item.embedding.len() != self.dim {
            return Err(Error::Internal(format!(
                "openai returned dim {} but expected {}",
                item.embedding.len(),
                self.dim
            )));
        }
        Ok(item.embedding)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
