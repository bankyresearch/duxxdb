//! XLM-RoBERTa cross-encoders on Candle.
//!
//! Each `(query, document)` pair is tokenized together as
//! `<s> query </s></s> document </s>` (truncated to `max_tokens`, cutting
//! the longer side first — in practice the document's tail), run through
//! the encoder, and the classification head's single logit on `<s>` is
//! squashed with a sigmoid into a `[0, 1]` relevance score.
//!
//! Cost grows with pairs × tokens (and attention with tokens²), so the
//! two levers are how many candidates to rerank and `max_tokens`; see
//! the crate README for measured latencies.

use std::path::Path;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use candle_transformers::models::xlm_roberta::{
    Config as XlmrConfig, XLMRobertaForSequenceClassification,
};
use duxx_core::{Error, Result};
use duxx_embed::local::hub::{self, LocalSpec, PinnedFile, PinnedModel};
use duxx_embed::local::{candle_err, tok_err};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::Reranker;

const TOKENIZER_XLMR: PinnedFile = PinnedFile {
    name: "tokenizer.json",
    size: 17_082_660,
    sha256: "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
};

/// Cross-encoders DuxxDB can download and run by alias. The first is
/// the default.
pub const RERANK_MODELS: &[PinnedModel] = &[
    // 12 layers, hidden 384, multilingual (mMARCO: 14 languages incl.
    // Hindi, on an XLM-R vocabulary covering ~100). ~118 M params, most
    // of them in the 250k-token embedding table.
    PinnedModel {
        alias: "mmarco-minilm-l12",
        repo: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision: "1427fd652930e4ba29e8149678df786c240d8825",
        files: &[
            PinnedFile {
                name: "config.json",
                size: 891,
                sha256: "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77",
            },
            TOKENIZER_XLMR,
            PinnedFile {
                name: "model.safetensors",
                size: 470_592_698,
                sha256: "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4",
            },
        ],
    },
    // Same family, 6 layers: about half the compute of L12.
    PinnedModel {
        alias: "mmarco-minilm-l6",
        repo: "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
        revision: "4ceabf2d1e212e16da0d1fb94d5dea66a9a1cca0",
        files: &[
            PinnedFile {
                name: "config.json",
                size: 811,
                sha256: "04fc264314c4b1d6f878adff70961c92134437439a4a02dc884f9b4f409af802",
            },
            TOKENIZER_XLMR,
            PinnedFile {
                name: "pytorch_model.bin",
                size: 428_014_765,
                sha256: "795f7056ac9d129f5a7fe4bb77cda35cbefa5e5e98b507cf8ee0b2c6db122796",
            },
        ],
    },
    // XLM-R base (278 M params, 12 layers, hidden 768): best quality of
    // the three, roughly 4× the compute of L12 — for GPU-less hosts only
    // with small candidate counts.
    PinnedModel {
        alias: "bge-reranker-base",
        repo: "BAAI/bge-reranker-base",
        revision: "2cfc18c9415c912f9d8155881c133215df768a70",
        files: &[
            PinnedFile {
                name: "config.json",
                size: 799,
                sha256: "289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708",
            },
            PinnedFile {
                name: "tokenizer.json",
                size: 17_098_107,
                sha256: "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e",
            },
            PinnedFile {
                name: "model.safetensors",
                size: 1_112_206_140,
                sha256: "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd",
            },
        ],
    },
];

/// Default `max_tokens` per (query, document) pair. Cost grows faster
/// than linearly with length (12-layer model, 50 pairs, 2 contended
/// Graviton3 threads: 5.5 s at 128, 14.3 s at 256, 38 s at 512), and a
/// title plus lead paragraph fits in 128, so that is the default.
pub const DEFAULT_MAX_TOKENS: usize = 128;

/// Padded-batch budget (rows × longest²): 4 × 512² keeps one layer's
/// attention scores near 50 MB.
const MAX_CELLS: usize = 4 * 512 * 512;

/// A cross-encoder reranker. Cheap to clone, `Send + Sync`.
#[derive(Clone)]
pub struct CrossEncoder {
    inner: Arc<Inner>,
}

struct Inner {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    pad_id: u32,
    max_tokens: usize,
    max_batch: usize,
    id: String,
}

impl std::fmt::Debug for CrossEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoder")
            .field("id", &self.inner.id)
            .field("max_tokens", &self.inner.max_tokens)
            .finish()
    }
}

#[derive(serde::Deserialize)]
struct ModelType {
    model_type: Option<String>,
}

impl CrossEncoder {
    /// Build from the part of a spec after `local:`, e.g.
    /// `mmarco-minilm-l12,max_tokens=128`. Options: `max_tokens`
    /// (default [`DEFAULT_MAX_TOKENS`], capped at the model's 512),
    /// `batch` (max pairs per forward pass, default 16).
    pub fn from_spec(rest: &str) -> Result<Self> {
        let rest = if rest.trim().is_empty() {
            RERANK_MODELS[0].alias
        } else {
            rest
        };
        let spec = LocalSpec::parse(rest)?;
        spec.check_known(&["max_tokens", "batch"])?;
        let max_tokens = spec.usize_opt("max_tokens", DEFAULT_MAX_TOKENS)?;
        let max_batch = spec.usize_opt("batch", 16)?;
        let (dir, model_id) = hub::resolve(&spec.model, RERANK_MODELS)?;
        Self::load(&dir, format!("local:{model_id}"), max_tokens, max_batch)
    }

    /// Load an XLM-RoBERTa sequence-classification model (one output
    /// logit) from a directory with `config.json`, `tokenizer.json` and
    /// weights.
    pub fn load(dir: &Path, id: String, max_tokens: usize, max_batch: usize) -> Result<Self> {
        let started = std::time::Instant::now();
        let kind: ModelType = hub::read_config(dir)?;
        match kind.model_type.as_deref() {
            Some("xlm-roberta") | Some("roberta") => {}
            other => {
                return Err(Error::Internal(format!(
                    "reranker {id}: model_type {other:?} is not supported \
                     (need an XLM-RoBERTa cross-encoder)"
                )))
            }
        }
        let config: XlmrConfig = hub::read_config(dir)?;
        // RoBERTa positions start at pad_token_id + 1, so two slots of
        // max_position_embeddings are never usable by tokens.
        let max_tokens = max_tokens
            .min(
                config
                    .max_position_embeddings
                    .saturating_sub(config.pad_token_id as usize + 1),
            )
            .max(16);
        let mut tokenizer = hub::load_tokenizer(dir)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: max_tokens,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(tok_err)?;
        tokenizer.with_padding(None);
        let vb = hub::load_weights(dir)?;
        let model = XLMRobertaForSequenceClassification::new(1, &config, vb).map_err(candle_err)?;
        tracing::info!(
            %id,
            layers = config.num_hidden_layers,
            hidden = config.hidden_size,
            max_tokens,
            load_ms = started.elapsed().as_millis() as u64,
            "local reranker loaded"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                model,
                tokenizer,
                // Must be the model's pad id: XLM-R derives position ids
                // from `input_ids != pad_token_id`.
                pad_id: config.pad_token_id,
                max_tokens,
                max_batch: max_batch.max(1),
                id,
            }),
        })
    }

    /// Max tokens per (query, document) pair.
    pub fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    /// Token ids for a pair exactly as the model sees them.
    pub fn tokenize_pair(&self, query: &str, doc: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .tokenizer
            .encode((query, doc), true)
            .map_err(tok_err)?;
        Ok(enc.get_ids().to_vec())
    }

    /// Raw logits (before the sigmoid), one per document.
    pub fn logits(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let inner = &*self.inner;
        let pairs: Vec<(String, String)> = docs
            .iter()
            .map(|d| (query.to_string(), d.to_string()))
            .collect();
        let encodings = inner.tokenizer.encode_batch(pairs, true).map_err(tok_err)?;
        let lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
        let mut out = vec![0f32; docs.len()];
        let device = Device::Cpu;
        for batch in hub::plan_batches(&lens, MAX_CELLS, inner.max_batch) {
            let rows = batch.len();
            let width = batch.iter().map(|&i| lens[i]).max().unwrap_or(1).max(1);
            let mut ids = Vec::with_capacity(rows * width);
            let mut mask = Vec::with_capacity(rows * width);
            for &i in &batch {
                let e = &encodings[i];
                ids.extend_from_slice(e.get_ids());
                mask.extend_from_slice(e.get_attention_mask());
                let pad = width - e.get_ids().len();
                ids.extend(std::iter::repeat(inner.pad_id).take(pad));
                mask.extend(std::iter::repeat(0u32).take(pad));
            }
            let logits = (|| -> candle_core::Result<Vec<f32>> {
                let ids = Tensor::from_vec(ids, (rows, width), &device)?;
                let mask = Tensor::from_vec(mask, (rows, width), &device)?;
                // XLM-R has a single token type.
                let type_ids = ids.zeros_like()?;
                inner
                    .model
                    .forward(&ids, &mask, &type_ids)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            })()
            .map_err(candle_err)?;
            for (l, &i) in logits.into_iter().zip(batch.iter()) {
                out[i] = l;
            }
        }
        Ok(out)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Reranker for CrossEncoder {
    fn score(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        Ok(self.logits(query, docs)?.into_iter().map(sigmoid).collect())
    }

    fn id(&self) -> String {
        self.inner.id.clone()
    }
}

/// These download `mmarco-mMiniLMv2-L12-H384-v1` (~490 MB) on first run
/// into the model directory, so they are `#[ignore]`d by default:
///
/// ```text
/// DUXX_MODEL_DIR=/models cargo test -p duxx-rerank --features local --release -- --ignored
/// ```
#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CrossEncoder {
        use std::sync::OnceLock;
        static M: OnceLock<CrossEncoder> = OnceLock::new();
        M.get_or_init(|| CrossEncoder::from_spec("mmarco-minilm-l12").expect("load reranker"))
            .clone()
    }

    #[test]
    #[ignore = "downloads mmarco-mMiniLMv2-L12-H384-v1"]
    fn pair_tokenization_and_truncation() {
        let m = model();
        assert!(m
            .id()
            .starts_with("local:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@"));
        let ids = m.tokenize_pair("monsoon", "rain in Kerala").unwrap();
        // <s> q </s></s> d </s>
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&2));
        assert!(ids.windows(2).any(|w| w == [2, 2]), "{ids:?}");
        let long = "बारिश ".repeat(4000);
        let ids = m.tokenize_pair("मानसून कब आएगा", &long).unwrap();
        assert_eq!(ids.len(), m.max_tokens());
        // The query survives truncation intact.
        let q = m.tokenize_pair("मानसून कब आएगा", "x").unwrap();
        let q_part = &q[..q.iter().position(|&t| t == 2).unwrap()];
        assert_eq!(&ids[..q_part.len()], q_part);
    }

    #[test]
    #[ignore = "downloads mmarco-mMiniLMv2-L12-H384-v1"]
    fn relevant_document_scores_highest() {
        check_ranking(&model());
    }

    #[test]
    #[ignore = "downloads mmarco-mMiniLMv2-L6-H384-v1 (pytorch_model.bin)"]
    fn l6_relevant_document_scores_highest() {
        let m = CrossEncoder::from_spec("mmarco-minilm-l6").expect("load reranker");
        assert!(m
            .id()
            .starts_with("local:nreimers/mmarco-mMiniLMv2-L6-H384-v1@"));
        check_ranking(&m);
    }

    fn check_ranking(m: &CrossEncoder) {
        let docs = [
            "Basmati rice is grown in the foothills of the Himalayas.",
            "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent.",
            "Virat Kohli scored a century against Australia in Perth.",
            "The Taj Mahal was commissioned by Shah Jahan in 1632.",
        ];
        let scores = m.score("What is the RBI repo rate?", &docs).unwrap();
        assert_eq!(scores.len(), 4);
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)));
        let top = m.rerank("What is the RBI repo rate?", &docs, 1).unwrap();
        assert_eq!(top[0].0, 1, "{scores:?}");
        assert_eq!(
            m.rerank("who built the Taj Mahal", &docs, 1).unwrap()[0].0,
            3
        );

        let hi_docs = [
            "भारत में मानसून आमतौर पर जून में केरल पहुंचता है।",
            "शेयर बाजार में आज गिरावट दर्ज की गई।",
            "दिल्ली में प्रदूषण का स्तर बढ़ गया है।",
        ];
        let top = m.rerank("मानसून केरल कब पहुंचता है", &hi_docs, 3).unwrap();
        assert_eq!(top[0].0, 0, "{top:?}");
        // Batching and padding do not change a document's score.
        let alone = m.score("मानसून केरल कब पहुंचता है", &hi_docs[..1]).unwrap()[0];
        assert!((alone - top[0].1).abs() < 1e-4, "{alone} vs {}", top[0].1);
    }
}
