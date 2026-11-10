//! `multilingual-e5-small` sentence embeddings on Candle.
//!
//! Pipeline per text: optional e5 prefix (`query: ` for searches,
//! `passage: ` for stored text) → the model's own XLM-R SentencePiece
//! tokenizer (truncated to `max_tokens`, at most 512) → 12-layer BERT
//! encoder → mean pooling over real (non-padding) tokens → L2
//! normalisation. Inputs are batched by length so padding stays small.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use duxx_core::{Error, Result};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use super::hub::{self, LocalSpec, PinnedFile, PinnedModel};
use super::{candle_err, tok_err};
use crate::Embedder;

/// Embedding models DuxxDB can download and run by alias.
pub const EMBEDDING_MODELS: &[PinnedModel] = &[PinnedModel {
    alias: "multilingual-e5-small",
    repo: "intfloat/multilingual-e5-small",
    revision: "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    files: &[
        PinnedFile {
            name: "config.json",
            size: 655,
            sha256: "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
        },
        PinnedFile {
            name: "tokenizer.json",
            size: 17_082_730,
            sha256: "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        },
        PinnedFile {
            name: "model.safetensors",
            size: 470_641_600,
            sha256: "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
        },
    ],
}];

/// Padded-batch budget: rows × longest² ≤ this. 4 × 512² keeps one
/// layer's attention scores near 50 MB at the 512-token cap.
const MAX_CELLS: usize = 4 * 512 * 512;

/// A sentence embedder running an e5-family BERT model in-process.
///
/// Cheap to clone (shared `Arc`), `Send + Sync`; concurrent calls run
/// concurrently. Matrix multiplies use Candle's CPU threads (set
/// `RAYON_NUM_THREADS` to cap them).
#[derive(Clone)]
pub struct LocalEmbedder {
    inner: Arc<Inner>,
}

struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
    pad_id: u32,
    dim: usize,
    max_tokens: usize,
    max_batch: usize,
    id: String,
    /// `(query, passage)` prefixes; `None` for models trained without.
    prefixes: Option<(&'static str, &'static str)>,
}

impl std::fmt::Debug for LocalEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEmbedder")
            .field("id", &self.inner.id)
            .field("dim", &self.inner.dim)
            .field("max_tokens", &self.inner.max_tokens)
            .finish()
    }
}

impl LocalEmbedder {
    /// Build from the part of a spec after `local:`, e.g.
    /// `multilingual-e5-small` or `/models/e5,max_tokens=256`.
    ///
    /// Options: `max_tokens` (default 512, capped by the model),
    /// `batch` (max rows per forward pass, default 32), `prefix`
    /// (`e5` = add `query: `/`passage: `, the default; `none` = raw text).
    pub fn from_spec(rest: &str) -> Result<Self> {
        let spec = LocalSpec::parse(rest)?;
        spec.check_known(&["max_tokens", "batch", "prefix"])?;
        let max_tokens = spec.usize_opt("max_tokens", 512)?;
        let max_batch = spec.usize_opt("batch", 32)?;
        let prefixes = match spec.options.get("prefix").map(|s| s.as_str()) {
            None | Some("e5") => Some(("query: ", "passage: ")),
            Some("none") => None,
            Some(other) => {
                return Err(Error::Internal(format!(
                    "prefix must be e5 or none, got {other:?}"
                )))
            }
        };
        let (dir, model_id) = hub::resolve(&spec.model, EMBEDDING_MODELS)?;
        Self::load(
            &dir,
            format!("local:{model_id}"),
            max_tokens,
            max_batch,
            prefixes,
        )
    }

    /// The default model (`multilingual-e5-small`), downloaded if needed.
    pub fn multilingual_e5_small() -> Result<Self> {
        Self::from_spec("multilingual-e5-small")
    }

    /// Load from a model directory (`config.json`, `tokenizer.json`,
    /// weights). `id` is what [`Embedder::id`] reports.
    pub fn load(
        dir: &Path,
        id: String,
        max_tokens: usize,
        max_batch: usize,
        prefixes: Option<(&'static str, &'static str)>,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        let config: BertConfig = hub::read_config(dir)?;
        let max_tokens = max_tokens.min(config.max_position_embeddings).max(8);
        let mut tokenizer = hub::load_tokenizer(dir)?;
        // Padding positions are masked out of attention and pooling, so
        // the id only has to be a valid row of the embedding table.
        let pad_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(config.pad_token_id as u32);
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: max_tokens,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(tok_err)?;
        // Batches are padded by hand to their own longest member.
        tokenizer.with_padding(None);
        let vb = hub::load_weights(dir)?;
        let model = BertModel::load(vb, &config).map_err(candle_err)?;
        tracing::info!(
            %id,
            dim = config.hidden_size,
            max_tokens,
            load_ms = started.elapsed().as_millis() as u64,
            "local embedder loaded"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                model,
                tokenizer,
                pad_id,
                dim: config.hidden_size,
                max_tokens,
                max_batch: max_batch.max(1),
                id,
                prefixes,
            }),
        })
    }

    /// Max tokens per input (longer inputs are truncated).
    pub fn max_tokens(&self) -> usize {
        self.inner.max_tokens
    }

    /// Token ids for `text` exactly as the model will see it (prefix,
    /// special tokens, truncation). Mostly for tests and debugging.
    pub fn tokenize(&self, text: &str, is_query: bool) -> Result<Vec<u32>> {
        let t = self.with_prefix(text, is_query);
        let enc = self.inner.tokenizer.encode(t, true).map_err(tok_err)?;
        Ok(enc.get_ids().to_vec())
    }

    fn with_prefix(&self, text: &str, is_query: bool) -> String {
        match self.inner.prefixes {
            Some((q, p)) => format!("{}{text}", if is_query { q } else { p }),
            None => text.to_string(),
        }
    }

    fn run(&self, texts: &[&str], is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inner = &*self.inner;
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| self.with_prefix(t, is_query))
            .collect();
        let encodings = inner
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(tok_err)?;
        let lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
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
            let ids = Tensor::from_vec(ids, (rows, width), &device).map_err(candle_err)?;
            let mask = Tensor::from_vec(mask, (rows, width), &device).map_err(candle_err)?;
            let type_ids = ids.zeros_like().map_err(candle_err)?;
            let pooled = (|| -> candle_core::Result<Tensor> {
                let hidden = inner.model.forward(&ids, &type_ids, Some(&mask))?; // (b, t, h)
                let m = mask.to_dtype(DType::F32)?.unsqueeze(2)?; // (b, t, 1)
                let summed = hidden.broadcast_mul(&m)?.sum(1)?; // (b, h)
                let counts = m.sum(1)?.clamp(1f32, f32::MAX)?; // (b, 1)
                let mean = summed.broadcast_div(&counts)?;
                let norm = mean
                    .sqr()?
                    .sum_keepdim(D::Minus1)?
                    .sqrt()?
                    .clamp(1e-12f32, f32::MAX)?;
                mean.broadcast_div(&norm)
            })()
            .map_err(candle_err)?;
            let rows_out: Vec<Vec<f32>> = pooled.to_vec2().map_err(candle_err)?;
            for (row, &i) in rows_out.into_iter().zip(batch.iter()) {
                out[i] = row;
            }
        }
        Ok(out)
    }
}

impl Embedder for LocalEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.run(&[text], false)?.pop().unwrap_or_default())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.run(&[text], true)?.pop().unwrap_or_default())
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.run(texts, false)
    }

    fn dim(&self) -> usize {
        self.inner.dim
    }

    fn id(&self) -> String {
        self.inner.id.clone()
    }
}

/// These tests download `multilingual-e5-small` (~490 MB) on first run
/// into the model directory, so they are `#[ignore]`d by default:
///
/// ```text
/// DUXX_MODEL_DIR=/models cargo test -p duxx-embed --features local --release -- --ignored
/// ```
#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LocalEmbedder {
        use std::sync::OnceLock;
        static M: OnceLock<LocalEmbedder> = OnceLock::new();
        M.get_or_init(|| LocalEmbedder::multilingual_e5_small().expect("load e5"))
            .clone()
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    #[ignore = "downloads multilingual-e5-small"]
    fn tokenizes_with_prefix_and_special_tokens() {
        let m = model();
        let q = m.tokenize("hello world", true).unwrap();
        let p = m.tokenize("hello world", false).unwrap();
        // XLM-R specials: <s> = 0 first, </s> = 2 last.
        assert_eq!(q.first(), Some(&0));
        assert_eq!(q.last(), Some(&2));
        assert_ne!(q, p, "query: and passage: prefixes must differ");
        // Devanagari goes through SentencePiece, not to <unk> (= 3).
        let hi = m.tokenize("भारत की राजधानी नई दिल्ली है", false).unwrap();
        assert!(hi.len() > 4);
        assert!(hi.iter().filter(|&&t| t == 3).count() == 0, "{hi:?}");
        // Truncation caps at max_tokens including specials.
        let long = "word ".repeat(5000);
        assert_eq!(m.tokenize(&long, false).unwrap().len(), m.max_tokens());
    }

    #[test]
    #[ignore = "downloads multilingual-e5-small"]
    fn embeddings_have_dim_384_and_unit_norm() {
        let m = model();
        assert_eq!(m.dim(), 384);
        assert!(m.id().starts_with("local:intfloat/multilingual-e5-small@"));
        let texts = ["short", "a somewhat longer sentence about the monsoon", ""];
        let batch = m.embed_batch(&texts).unwrap();
        assert_eq!(batch.len(), 3);
        for v in &batch {
            assert_eq!(v.len(), 384);
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-3, "norm {n}");
        }
        // Batched and single calls agree (padding must not leak in).
        let single = m.embed(texts[1]).unwrap();
        assert!(cos(&single, &batch[1]) > 0.9999);
        let q = m.embed_query("monsoon").unwrap();
        assert_eq!(q.len(), 384);
    }

    #[test]
    #[ignore = "downloads multilingual-e5-small"]
    fn relevant_passage_ranks_first_english_and_hindi() {
        let m = model();
        let docs = [
            "The Reserve Bank of India sets the repo rate to control inflation.",
            "Virat Kohli scored a century in the test match at Eden Gardens.",
            "Basmati rice is grown in the foothills of the Himalayas.",
            "भारतीय रिज़र्व बैंक ने रेपो दर में बदलाव नहीं किया।",
        ];
        let dv = m.embed_batch(&docs).unwrap();
        let rank = |q: &str| -> usize {
            let qv = m.embed_query(q).unwrap();
            let scores: Vec<f32> = dv.iter().map(|d| cos(&qv, d)).collect();
            (0..scores.len())
                .max_by(|a, b| scores[*a].total_cmp(&scores[*b]))
                .unwrap()
        };
        assert_eq!(rank("who won the cricket match"), 1);
        assert_eq!(rank("where is basmati rice grown"), 2);
        let top = rank("RBI interest rate decision");
        assert!(top == 0 || top == 3, "got {top}");
        // Cross-lingual: a Hindi query finds the English cricket passage
        // over the unrelated ones.
        let qv = m.embed_query("क्रिकेट मैच में शतक").unwrap();
        assert!(cos(&qv, &dv[1]) > cos(&qv, &dv[2]));
    }
}
