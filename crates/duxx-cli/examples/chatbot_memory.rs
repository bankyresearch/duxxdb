//! End-to-end demo: store and recall agent memories.
//!
//! Run from the workspace root:
//! ```bash
//! cargo run -p duxx-cli --example chatbot_memory
//! ```
//!
//! The example uses a **toy embedder** (hashing tokens into a 32-d vector)
//! so we don't pull in a model at build time. In real use you plug in an
//! actual embedding provider (OpenAI, Cohere, local BGE, etc).

use duxx_memory::MemoryStore;

const DIM: usize = 32;

/// Toy embedder — DO NOT ship this in production.
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let store = MemoryStore::new(DIM);

    let corpus = [
        ("user_42", "I want a refund for order #9910 — it arrived broken."),
        ("user_42", "What's the status of my delivery?"),
        ("user_42", "I love your customer service team, very helpful."),
        ("user_42", "Can you recommend a good winter coat?"),
        ("user_42", "The tracking number is DX-002341."),
        ("user_42", "Order arrived damaged, need replacement or money back."),
        ("user_42", "My favorite colour is blue."),
    ];

    for (key, text) in corpus {
        let emb = embed(text);
        store.remember(key, text, emb)?;
    }
    println!("stored {} memories", store.len());

    let query = "refund for broken order";
    let qvec = embed(query);

    let start = std::time::Instant::now();
    let hits = store.recall("user_42", query, &qvec, 3)?;
    let elapsed = start.elapsed();

    println!("\nTop-3 recall for {:?}  ({} μs):", query, elapsed.as_micros());
    for (i, h) in hits.iter().enumerate() {
        println!(
            "  {}. id={}  score={:.4}  — {}",
            i + 1,
            h.memory.id,
            h.score,
            h.memory.text
        );
    }

    Ok(())
}
