//! Latency / throughput / memory of the local embedder and rerankers.
//!
//! ```text
//! DUXX_MODEL_DIR=/models RAYON_NUM_THREADS=4 \
//!   cargo run --release -p duxx-rerank --features local --example bench_models
//! ```
//!
//! Env knobs:
//! - `BENCH_RERANKERS`  comma list of reranker aliases (default `mmarco-minilm-l12`)
//! - `BENCH_MAX_TOKENS` comma list of reranker `max_tokens` (default `512,256,128`)
//! - `BENCH_DOCS`       candidates per rerank call (default 50)
//! - `BENCH_SKIP_EMBED` set to skip the embedder section
//!
//! Each model is loaded, measured and dropped before the next one, so
//! the reported RSS is per model (plus the process baseline).

use std::time::Instant;

use duxx_embed::{Embedder, LocalEmbedder};
use duxx_rerank::{CrossEncoder, Reranker};

const EN: &[&str] = &[
    "The monsoon usually reaches the Kerala coast in the first week of June and then spreads north across the country over the following weeks.",
    "Farmers in Punjab and Haryana depend on canal irrigation and groundwater for the rabi wheat crop sown after the kharif harvest.",
    "The Reserve Bank of India reviews the repo rate every two months, weighing retail inflation against the pace of economic growth.",
    "Indian Railways operates one of the largest networks in the world, carrying more than twenty million passengers every day.",
    "Bengaluru has grown into a major technology hub, home to thousands of start-ups and the research centres of global companies.",
    "The Supreme Court heard arguments on the constitutional validity of the amendment and reserved its judgement for a later date.",
    "Air quality in Delhi deteriorates every winter as stubble burning, vehicle exhaust and still weather trap pollutants near the ground.",
    "The national cricket team named an unchanged squad for the test series, with the selectors backing the young opening pair.",
];

const HI: &[&str] = &[
    "भारत में मानसून आमतौर पर जून के पहले सप्ताह में केरल तट पर पहुंचता है और फिर धीरे-धीरे पूरे देश में फैल जाता है।",
    "भारतीय रिज़र्व बैंक हर दो महीने में रेपो दर की समीक्षा करता है और महंगाई तथा आर्थिक विकास के बीच संतुलन बनाता है।",
    "दिल्ली में सर्दियों के दौरान वायु गुणवत्ता खराब हो जाती है क्योंकि पराली जलाने और वाहनों के धुएं से प्रदूषण बढ़ता है।",
    "भारतीय रेलवे दुनिया के सबसे बड़े रेल नेटवर्कों में से एक है जो हर दिन करोड़ों यात्रियों को उनकी मंज़िल तक पहुंचाता है।",
];

/// A ~`target_words`-word document mixing the sentence pools; `seed`
/// rotates the order so documents differ.
fn make_doc(seed: usize, target_words: usize) -> String {
    let mut out = String::new();
    let mut i = seed;
    while out.split_whitespace().count() < target_words {
        let s = if i % 5 == 4 {
            HI[i % HI.len()]
        } else {
            EN[i % EN.len()]
        };
        out.push_str(s);
        out.push(' ');
        i += 3;
    }
    out
}

fn rss_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |k: &str| -> f64 {
        status
            .lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|kb| kb / 1024.0)
            .unwrap_or(0.0)
    };
    (get("VmRSS:"), get("VmHWM:"))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    println!("threads (RAYON_NUM_THREADS) = {threads}");
    let (rss0, _) = rss_mb();
    println!("baseline RSS {rss0:.0} MB");

    let n_docs: usize = std::env::var("BENCH_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    // ~380 words of this mixed text is ~500 XLM-R tokens.
    let docs: Vec<String> = (0..n_docs).map(|i| make_doc(i, 380)).collect();
    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

    if std::env::var("BENCH_SKIP_EMBED").is_err() {
        let t = Instant::now();
        let e = LocalEmbedder::multilingual_e5_small().expect("load e5");
        let load = t.elapsed().as_secs_f64();
        let (rss, _) = rss_mb();
        let toks = e.tokenize(&docs[0], false).unwrap().len();
        println!(
            "\n== embedder {} (dim {}) load {load:.1}s, RSS {rss:.0} MB (+{:.0})",
            e.id(),
            e.dim(),
            rss - rss0
        );
        e.embed_query("warm up").unwrap();
        let q: Vec<f64> = (0..10)
            .map(|_| {
                let t = Instant::now();
                e.embed_query("when does the monsoon reach Kerala").unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        println!("query embed (8 tokens):       median {:.1} ms", median(q));
        let short: Vec<String> = (0..64).map(|i| make_doc(i, 60)).collect();
        let short_refs: Vec<&str> = short.iter().map(|s| s.as_str()).collect();
        let t = Instant::now();
        e.embed_batch(&short_refs).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "passages ~80 tokens  x64:      {:.0} ms total, {:.1} passages/s",
            s * 1e3,
            64.0 / s
        );
        let t = Instant::now();
        e.embed_batch(&doc_refs[..16.min(doc_refs.len())]).unwrap();
        let s = t.elapsed().as_secs_f64();
        let n = 16.min(doc_refs.len()) as f64;
        println!(
            "passages {toks} tokens x{n:.0}:      {:.0} ms total, {:.1} passages/s",
            s * 1e3,
            n / s
        );
        let (rss, hwm) = rss_mb();
        println!("RSS after embedding {rss:.0} MB, peak {hwm:.0} MB");
    }

    let rerankers = std::env::var("BENCH_RERANKERS").unwrap_or_else(|_| "mmarco-minilm-l12".into());
    let lens = std::env::var("BENCH_MAX_TOKENS").unwrap_or_else(|_| "512,256,128".into());
    for alias in rerankers
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        for max_tokens in lens
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
        {
            let (before, _) = rss_mb();
            let t = Instant::now();
            let r = CrossEncoder::from_spec(&format!("{alias},max_tokens={max_tokens}"))
                .expect("load reranker");
            let load = t.elapsed().as_secs_f64();
            let (loaded, _) = rss_mb();
            let toks = r
                .tokenize_pair("when does the monsoon reach Kerala", &docs[0])
                .unwrap()
                .len();
            let query = "when does the monsoon reach Kerala";
            r.score(query, &doc_refs[..2]).unwrap(); // warm-up
            let runs: Vec<f64> = (0..3)
                .map(|_| {
                    let t = Instant::now();
                    let top = r.rerank(query, &doc_refs, 10).unwrap();
                    assert_eq!(top.len(), 10.min(n_docs));
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            let (rss, hwm) = rss_mb();
            println!(
                "\n== reranker {} max_tokens={max_tokens} (pair = {toks} tokens)\n\
                 load {load:.1}s, RSS {loaded:.0} MB (+{:.0} for the model), {rss:.0} MB after scoring, peak {hwm:.0} MB\n\
                 rerank {n_docs} docs: median {:.0} ms  (runs {:?})",
                r.id(),
                loaded - before,
                median(runs.clone()),
                runs.iter().map(|x| x.round() as i64).collect::<Vec<_>>()
            );
        }
    }
}
