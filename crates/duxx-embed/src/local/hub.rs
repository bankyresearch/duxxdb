//! Model files: where they live, how they get there, how they load.
//!
//! Built-in models are pinned to an exact Hugging Face commit and the
//! weights / tokenizer / config are checked against pinned SHA-256
//! digests after download, so a model can never silently change under
//! a running database. Files land in the model directory once and are
//! reused on every later start:
//!
//! 1. `DUXX_MODEL_DIR` if set (the Docker image sets
//!    `/var/lib/duxxdb/models`, on the data volume);
//! 2. otherwise `$XDG_CACHE_HOME/duxxdb/models`;
//! 3. otherwise `$HOME/.cache/duxxdb/models`;
//! 4. otherwise `./duxxdb-models`.
//!
//! A model can also be loaded from any local directory holding
//! `config.json`, `tokenizer.json` and `model.safetensors` (or
//! `pytorch_model.bin`) — pass the directory path instead of an alias.
//! Set `DUXX_MODEL_OFFLINE=1` to forbid downloads entirely (air-gapped
//! hosts: copy the directory in by hand). `DUXX_HF_ENDPOINT` points at a
//! Hugging Face mirror (default `https://huggingface.co`).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use duxx_core::{Error, Result};
use sha2::{Digest, Sha256};

/// One file of a pinned model.
#[derive(Debug, Clone, Copy)]
pub struct PinnedFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// A built-in model: alias → exact Hugging Face repo + commit + files.
#[derive(Debug, Clone, Copy)]
pub struct PinnedModel {
    /// Short name accepted in specs, e.g. `multilingual-e5-small`.
    pub alias: &'static str,
    /// Hugging Face repo id, e.g. `intfloat/multilingual-e5-small`.
    pub repo: &'static str,
    /// Full commit SHA the files are pinned to.
    pub revision: &'static str,
    pub files: &'static [PinnedFile],
}

impl PinnedModel {
    /// True if `name` refers to this model (alias or repo id, any case).
    pub fn matches(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case(self.alias) || name.eq_ignore_ascii_case(self.repo)
    }
}

/// Directory model files are cached in (see module docs).
pub fn model_root() -> PathBuf {
    if let Some(d) = non_empty_env("DUXX_MODEL_DIR") {
        return PathBuf::from(d);
    }
    if let Some(d) = non_empty_env("XDG_CACHE_HOME") {
        return Path::new(&d).join("duxxdb").join("models");
    }
    if let Some(d) = non_empty_env("HOME") {
        return Path::new(&d).join(".cache").join("duxxdb").join("models");
    }
    PathBuf::from("duxxdb-models")
}

fn non_empty_env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

fn offline() -> bool {
    matches!(
        non_empty_env("DUXX_MODEL_OFFLINE").as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Resolve a model name to a local directory with its files present.
///
/// `name` is either a path to an existing directory (used as-is) or the
/// alias / repo id of one of `registry`'s pinned models (downloaded on
/// first use into [`model_root`]). Returns the directory and a stable
/// identifier for the model (`repo@rev12`, or `path:<dir name>`).
pub fn resolve(name: &str, registry: &[PinnedModel]) -> Result<(PathBuf, String)> {
    let as_path = Path::new(name);
    if as_path.is_dir() {
        let label = as_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| name.to_string());
        return Ok((as_path.to_path_buf(), format!("path:{label}")));
    }
    let model = registry.iter().find(|m| m.matches(name)).ok_or_else(|| {
        let known: Vec<&str> = registry.iter().map(|m| m.alias).collect();
        Error::Internal(format!(
            "unknown local model {name:?}: use one of {known:?} or a directory path"
        ))
    })?;
    let dir = ensure_downloaded(model, &model_root())?;
    Ok((dir, format!("{}@{}", model.repo, &model.revision[..12])))
}

/// Make sure every file of `model` is present (and was verified) under
/// `root`, downloading what is missing. Returns the model's directory.
pub fn ensure_downloaded(model: &PinnedModel, root: &Path) -> Result<PathBuf> {
    let dir = root.join(format!(
        "{}@{}",
        model.repo.replace('/', "--"),
        &model.revision[..12]
    ));
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::Internal(format!("create model dir {dir:?}: {e}")))?;
    for f in model.files {
        let path = dir.join(f.name);
        // Files are verified by digest when downloaded and renamed into
        // place atomically, so a present file with the pinned size is a
        // complete, verified file.
        if std::fs::metadata(&path)
            .map(|m| m.len() == f.size)
            .unwrap_or(false)
        {
            continue;
        }
        if offline() {
            return Err(Error::Internal(format!(
                "model file {path:?} is missing and DUXX_MODEL_OFFLINE is set; \
                 copy {}@{} into {dir:?} or unset DUXX_MODEL_OFFLINE",
                model.repo, model.revision
            )));
        }
        download(model, f, &path)?;
    }
    Ok(dir)
}

fn download(model: &PinnedModel, f: &PinnedFile, dest: &Path) -> Result<()> {
    let endpoint =
        non_empty_env("DUXX_HF_ENDPOINT").unwrap_or_else(|| "https://huggingface.co".to_string());
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        model.repo,
        model.revision,
        f.name
    );
    tracing::info!(%url, bytes = f.size, "downloading model file");
    let part = dest.with_extension("part");
    let dest = dest.to_path_buf();
    let expect = *f;
    // reqwest's blocking client owns a small runtime of its own and may
    // not be created or dropped on a tokio worker thread; a plain OS
    // thread keeps this callable from anywhere (CLI start-up, tests,
    // inside the server's runtime).
    let handle = std::thread::spawn(move || -> Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60 * 30))
            .build()
            .map_err(|e| Error::Internal(format!("http client: {e}")))?;
        let mut resp = client
            .get(&url)
            .send()
            .map_err(|e| Error::Internal(format!("GET {url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Internal(format!(
                "GET {url}: HTTP {}",
                resp.status()
            )));
        }
        let mut out = std::fs::File::create(&part)
            .map_err(|e| Error::Internal(format!("create {part:?}: {e}")))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut total = 0u64;
        loop {
            let n = resp
                .read(&mut buf)
                .map_err(|e| Error::Internal(format!("read {url}: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])
                .map_err(|e| Error::Internal(format!("write {part:?}: {e}")))?;
            total += n as u64;
        }
        out.sync_all().ok();
        drop(out);
        let digest = hex(&hasher.finalize());
        if total != expect.size || digest != expect.sha256 {
            let _ = std::fs::remove_file(&part);
            return Err(Error::Internal(format!(
                "{url}: got {total} bytes sha256 {digest}, expected {} bytes sha256 {}",
                expect.size, expect.sha256
            )));
        }
        std::fs::rename(&part, &dest)
            .map_err(|e| Error::Internal(format!("rename {part:?} -> {dest:?}: {e}")))?;
        Ok(())
    });
    handle
        .join()
        .map_err(|_| Error::Internal("model download thread panicked".to_string()))?
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read and parse `config.json` from a model directory.
pub fn read_config<T: serde::de::DeserializeOwned>(dir: &Path) -> Result<T> {
    let path = dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::Internal(format!("read {path:?}: {e}")))?;
    serde_json::from_str(&raw).map_err(|e| Error::Internal(format!("parse {path:?}: {e}")))
}

/// Load `tokenizer.json` from a model directory.
pub fn load_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer> {
    let path = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| Error::Internal(format!("load tokenizer {path:?}: {e}")))
}

/// Open the model weights as an f32 [`VarBuilder`] on CPU.
///
/// Prefers `model.safetensors` (memory-mapped: tensors are copied out
/// of the map as the model is built, so the map itself only costs page
/// cache); falls back to `pytorch_model.bin`, read with Candle's
/// pickle reader (it parses tensor data only; nothing is executed).
pub fn load_weights(dir: &Path) -> Result<VarBuilder<'static>> {
    let device = Device::Cpu;
    let st = dir.join("model.safetensors");
    if st.is_file() {
        // SAFETY: memmap2 requires the file not be modified while
        // mapped. Model files are written once (atomic rename after a
        // digest check) and never touched again by DuxxDB.
        return unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&st), DType::F32, &device)
        }
        .map_err(|e| Error::Internal(format!("load {st:?}: {e}")));
    }
    let pth = dir.join("pytorch_model.bin");
    if pth.is_file() {
        return VarBuilder::from_pth(&pth, DType::F32, &device)
            .map_err(|e| Error::Internal(format!("load {pth:?}: {e}")));
    }
    Err(Error::Internal(format!(
        "no model.safetensors or pytorch_model.bin in {dir:?}"
    )))
}

/// Parsed `local:` spec: `<model>[,key=value]*`.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSpec {
    pub model: String,
    pub options: BTreeMap<String, String>,
}

impl LocalSpec {
    /// Parse the part after `local:`. Options are comma-separated
    /// `key=value` pairs after the model name, e.g.
    /// `multilingual-e5-small,max_tokens=256`.
    pub fn parse(rest: &str) -> Result<Self> {
        let mut parts = rest.split(',');
        let model = parts.next().unwrap_or("").trim().to_string();
        if model.is_empty() {
            return Err(Error::Internal(
                "local spec needs a model name or directory".to_string(),
            ));
        }
        let mut options = BTreeMap::new();
        for p in parts {
            let p = p.trim();
            if p.is_empty() {
                continue;
            }
            let (k, v) = p.split_once('=').ok_or_else(|| {
                Error::Internal(format!("bad local option {p:?}: want key=value"))
            })?;
            options.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
        Ok(Self { model, options })
    }

    /// A positive integer option, or `default` when absent.
    pub fn usize_opt(&self, key: &str, default: usize) -> Result<usize> {
        match self.options.get(key) {
            None => Ok(default),
            Some(v) => v.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| {
                Error::Internal(format!(
                    "option {key} must be a positive integer, got {v:?}"
                ))
            }),
        }
    }

    /// Fail on options not in `allowed` (catches typos like `max_token`).
    pub fn check_known(&self, allowed: &[&str]) -> Result<()> {
        for k in self.options.keys() {
            if !allowed.contains(&k.as_str()) {
                return Err(Error::Internal(format!(
                    "unknown local option {k:?} (known: {allowed:?})"
                )));
            }
        }
        Ok(())
    }
}

/// Group sequences into padded batches.
///
/// `lens[i]` is the token length of sequence `i`. Sequences are sorted
/// longest-first and packed greedily so that each batch has at most
/// `max_batch` rows and `rows × longest²` stays within `max_cells` — the
/// attention matrices grow with the square of the padded length, so this
/// bounds peak memory per forward pass while still batching short
/// inputs widely. Each returned batch lists original indices.
pub fn plan_batches(lens: &[usize], max_cells: usize, max_batch: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by(|a, b| lens[*b].cmp(&lens[*a]));
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_len = 0usize;
    for i in order {
        if cur.is_empty() {
            cur_len = lens[i].max(1);
            cur.push(i);
            continue;
        }
        let rows = cur.len() + 1;
        if rows > max_batch || rows * cur_len * cur_len > max_cells {
            out.push(std::mem::take(&mut cur));
            cur_len = lens[i].max(1);
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_parses_model_and_options() {
        let s = LocalSpec::parse("multilingual-e5-small, max_tokens=256 ,batch=8").unwrap();
        assert_eq!(s.model, "multilingual-e5-small");
        assert_eq!(s.usize_opt("max_tokens", 512).unwrap(), 256);
        assert_eq!(s.usize_opt("batch", 32).unwrap(), 8);
        assert_eq!(s.usize_opt("missing", 7).unwrap(), 7);
        assert!(s.check_known(&["max_tokens", "batch"]).is_ok());
        assert!(s.check_known(&["max_tokens"]).is_err());
        assert!(LocalSpec::parse("").is_err());
        assert!(LocalSpec::parse("m,oops").is_err());
        assert!(LocalSpec::parse("m,max_tokens=0")
            .unwrap()
            .usize_opt("max_tokens", 1)
            .is_err());
    }

    #[test]
    fn batches_respect_limits_and_cover_everything() {
        let lens = vec![512, 10, 300, 12, 512, 40, 7, 512, 512, 512, 512, 512];
        let batches = plan_batches(&lens, 4 * 512 * 512, 32);
        let mut seen: Vec<usize> = batches.iter().flatten().copied().collect();
        seen.sort();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>());
        for b in &batches {
            let longest = b.iter().map(|&i| lens[i]).max().unwrap();
            assert!(b.len() == 1 || b.len() * longest * longest <= 4 * 512 * 512);
            assert!(b.len() <= 32);
        }
        // Seven 512-token inputs at 4 per batch need two batches; the
        // short tail packs into one more.
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn unknown_alias_lists_known_models() {
        let reg = [PinnedModel {
            alias: "tiny",
            repo: "org/tiny",
            revision: "0123456789abcdef0123456789abcdef01234567",
            files: &[],
        }];
        let err = resolve("nope", &reg).unwrap_err().to_string();
        assert!(err.contains("tiny"), "{err}");
    }
}
