//! Cohere embeddings provider.
//!
//! Endpoint: `POST https://api.cohere.ai/v1/embed`. Same blocking
//! reqwest+rustls pattern as the OpenAI backend.
//!
//! ## Models
//!
//! | Model                       | Dim  |
//! |-----------------------------|------|
//! | `embed-english-v3.0`        | 1024 |
//! | `embed-multilingual-v3.0`   | 1024 |
//! | `embed-english-light-v3.0`  | 384  |

use crate::Embedder;
use duxx_core::{Error, Result};

const ENDPOINT: &str = "https://api.cohere.ai/v1/embed";

#[derive(Debug, Clone)]
pub struct CohereEmbedder {
    api_key: String,
    model: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl CohereEmbedder {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            dim,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// `embed-english-v3.0` — 1024 dim, most common choice.
    pub fn english_v3(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "embed-english-v3.0", 1024)
    }
}

pub(crate) fn default_dim_for_model(model: &str) -> usize {
    match model {
        "embed-english-light-v3.0" => 384,
        _ => 1024,
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    texts: [&'a str; 1],
    model: &'a str,
    input_type: &'static str,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl Embedder for CohereEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp = self
            .client
            .post(ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&EmbedRequest {
                texts: [text],
                model: &self.model,
                input_type: "search_document",
            })
            .send()
            .map_err(|e| Error::Internal(format!("cohere request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(Error::Internal(format!("cohere HTTP {status}: {body}")));
        }

        let parsed: EmbedResponse = resp
            .json()
            .map_err(|e| Error::Internal(format!("cohere parse: {e}")))?;
        let v = parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("cohere: empty embeddings".to_string()))?;
        if v.len() != self.dim {
            return Err(Error::Internal(format!(
                "cohere returned dim {} but expected {}",
                v.len(),
                self.dim
            )));
        }
        Ok(v)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
