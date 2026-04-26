//! Hybrid recall latency benchmarks.
//!
//! Measures `MemoryStore::recall` latency at three corpus sizes:
//!   100, 1 000, and 10 000 documents.
//!
//! The corpus is built once outside the timed loop; only `recall` calls
//! are timed. The toy embedder is the same one the chatbot example uses
//! (hash-bucket tokens into a 32-d vector) so we don't pay an embedding
//! provider's network cost during measurement.
//!
//! Run with:
//!   scripts/build.sh bench -p duxx-bench
//!
//! Results land under `target/criterion/`. The HTML report is at
//! `target/criterion/report/index.html`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use duxx_memory::MemoryStore;

const DIM: usize = 32;

/// Toy embedder. DO NOT ship; benchmarks only.
fn embed(text: &str) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = vec![0.0f32; DIM];
    for token in text.to_lowercase().split_whitespace() {
        let mut h = DefaultHasher::new();
        token.hash(&mut h);
        let bucket = (h.finish() as usize) % DIM;
        v[bucket] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Build a `MemoryStore` populated with `n` synthetic documents.
fn build_store(n: usize) -> MemoryStore {
    let store = MemoryStore::new(DIM);
    let topics = [
        "refund order delivery shipping",
        "weather forecast tomorrow rain",
        "login password account security",
        "billing invoice payment receipt",
        "product feature request feedback",
        "support agent chat human",
        "tracking number package status",
        "discount promo coupon sale",
    ];
    for i in 0..n {
        let topic = topics[i % topics.len()];
        let text = format!("doc {i} about {topic} item code {}", i % 1000);
        let emb = embed(&text);
        store.remember("user_42", text, emb).unwrap();
    }
    store
}

fn bench_recall(c: &mut Criterion) {
    let query_text = "refund delivery issue";
    let query_vec = embed(query_text);

    let mut group = c.benchmark_group("recall");
    group.sample_size(20); // criterion default 100 is overkill for ms-scale ops

    for &n in &[100usize, 1_000, 10_000] {
        let store = build_store(n);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("hybrid_k=10", n), &n, |b, _| {
            b.iter(|| {
                let hits = store
                    .recall(
                        "user_42",
                        black_box(query_text),
                        black_box(&query_vec),
                        black_box(10),
                    )
                    .unwrap();
                black_box(hits);
            });
        });
    }
    group.finish();
}

/// Insert latency on an empty store — measures the marginal cost of
/// `remember` (tantivy auto-commit + HNSW insert).
///
/// We deliberately don't sweep warmup sizes here: with auto-commit,
/// each setup of a 1k or 10k store takes seconds, blowing past
/// criterion's measurement budget. A dedicated bulk-insert bench will
/// land alongside Phase 2.5 (batched tantivy commits).
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.sample_size(20);
    group.throughput(Throughput::Elements(1));

    // Use a small capacity so iter_batched doesn't allocate huge HNSW
    // structures per batch (each `Hnsw::new` reserves space proportional
    // to capacity).
    const BENCH_CAPACITY: usize = 2_000;

    group.bench_function(BenchmarkId::new("remember_into_empty", 0), |b| {
        b.iter_batched(
            || {
                let store = MemoryStore::with_capacity(DIM, BENCH_CAPACITY);
                let text = String::from("user wants a refund for broken order");
                let emb = embed(&text);
                (store, text, emb)
            },
            |(store, text, emb)| {
                store.remember("user_42", text, emb).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Bulk insert: time the cost of populating an empty store with N memories.
///
/// Phase 2.5 (batched tantivy commits) shows up most clearly here —
/// pre-2.5 this scaled at ~4 ms × N; post-2.5 it should scale at
/// ~50 µs × N + one commit overhead.
fn bench_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_insert");
    group.sample_size(10);

    for &n in &[100usize, 1_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("remember_n", n), &n, |b, &n| {
            b.iter_batched(
                || MemoryStore::with_capacity(DIM, n.max(2_000)),
                |store| {
                    for i in 0..n {
                        let text = format!("doc {i} about topic {}", i % 16);
                        let emb = embed(&text);
                        store.remember("user", text, emb).unwrap();
                    }
                    black_box(&store);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_recall, bench_insert, bench_bulk_insert);
criterion_main!(benches);
