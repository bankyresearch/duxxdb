//! Compare the two vector graphs on a real text corpus.
//!
//! ```text
//! cargo run --release -p duxx-index --example vector_compare -- <corpus> <hnsw|compact> [dim]
//! ```
//!
//! `<corpus>.docs` and `<corpus>.queries` hold length-prefixed UTF-8 strings
//! (u32 little-endian length, then bytes). Every text is embedded with the
//! same hash embedder `duxx-server --embedder hash:<dim>` uses, inserted
//! through [`VectorIndex`] exactly as the server does, and every query is
//! searched at the `k` values Gaelxe's recall produces (`RECALL k` asks the
//! vector side for `3k`). Reported: build speed, resident memory added by the
//! index (Linux, from `/proc/self/statm`), recall against exact `f32` cosine,
//! and search latency. Run once per kind: each run is its own process, so
//! the memory figures do not mix.

use duxx_index::{VectorIndex, VectorKind};
use std::time::Instant;

fn read_strings(path: &str) -> Vec<String> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= b.len() {
        let n = u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
        i += 4;
        out.push(String::from_utf8_lossy(&b[i..i + n]).into_owned());
        i += n;
    }
    out
}

/// The hash embedder of `duxx-embed` (`hash:<dim>`).
fn embed(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = vec![0.0f32; dim];
    for token in text.to_lowercase().split_whitespace() {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        v[(h.finish() as usize) % dim] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

fn exact(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u64> {
    let mut s: Vec<(f32, u64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>(), i as u64))
        .collect();
    s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    s.into_iter().take(k).map(|x| x.1).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base = args.get(1).expect("corpus path");
    let kind = match args.get(2).map(String::as_str) {
        Some("compact") => VectorKind::Compact,
        _ => VectorKind::Hnsw,
    };
    let dim: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(128);
    let docs = read_strings(&format!("{base}.docs"));
    let queries = read_strings(&format!("{base}.queries"));
    let data: Vec<Vec<f32>> = docs.iter().map(|d| embed(d, dim)).collect();
    drop(docs);
    let qv: Vec<Vec<f32>> = queries.iter().map(|q| embed(q, dim)).collect();

    let before = rss_bytes();
    let t = Instant::now();
    let mut idx = VectorIndex::with_capacity_kind(dim, data.len() + 1, kind);
    for (i, v) in data.iter().enumerate() {
        idx.insert(i as u64, v.clone()).unwrap();
    }
    let build = t.elapsed().as_secs_f64();
    let after = rss_bytes();
    let n = data.len();
    println!("kind={} points={n} dim={dim}", kind.name());
    println!("build: {build:.1} s ({:.0} inserts/s)", n as f64 / build);
    if let (Some(b), Some(a)) = (before, after) {
        println!(
            "resident memory added by the index: {:.1} MB, {} bytes/point",
            (a.saturating_sub(b)) as f64 / 1e6,
            a.saturating_sub(b) / n as u64
        );
    }
    if let Some(m) = idx.memory_bytes() {
        println!("index arrays: {} bytes/point", m / n);
    }
    for k in [30usize, 120, 360] {
        let mut hit = 0usize;
        let mut lat = Vec::with_capacity(qv.len());
        for q in &qv {
            let truth = exact(&data, q, k);
            let t = Instant::now();
            let got: Vec<u64> = idx.search(q, k).into_iter().map(|h| h.0).collect();
            lat.push(t.elapsed().as_secs_f64() * 1e3);
            hit += truth.iter().filter(|x| got.contains(x)).count();
        }
        lat.sort_by(|a, b| a.total_cmp(b));
        println!(
            "k={k}: recall vs exact {:.4}, search median {:.2} ms, p90 {:.2} ms",
            hit as f64 / (qv.len() * k) as f64,
            lat[lat.len() / 2],
            lat[lat.len() * 9 / 10]
        );
    }
}
