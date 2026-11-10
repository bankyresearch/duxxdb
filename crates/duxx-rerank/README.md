# duxx-rerank

Second-stage reranking for DuxxDB. `RECALL` (HNSW + BM25 fused) finds
candidates fast but scores the query and each document separately; a
**cross-encoder** reads query and candidate together and judges how well
the candidate answers the query — much more precise, much more
expensive, so it runs on the top few dozen candidates only.

| Spec | Feature | Model |
|---|---|---|
| `local:mmarco-minilm-l12` (default for `local:`) | `local` | [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) — 12 layers × 384, multilingual (trained on mMARCO's 14 languages incl. Hindi, XLM-R vocabulary) |
| `local:mmarco-minilm-l6` | `local` | [`nreimers/mmarco-mMiniLMv2-L6-H384-v1`](https://huggingface.co/nreimers/mmarco-mMiniLMv2-L6-H384-v1) — same family, 6 layers, ~2× faster |
| `local:bge-reranker-base` | `local` | [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) — XLM-R base (278 M params), best quality, ~4× the compute of L12 |
| `overlap` | — | toy query-term overlap, for tests and demos |

All `local:` models run in-process on [Candle](https://github.com/huggingface/candle)
(pure Rust, CPU), pinned to an exact Hugging Face commit and SHA-256
checked, downloaded once into `DUXX_MODEL_DIR` — the same model store
and env vars as the local embedder (see `crates/duxx-embed/README.md`).
A local model directory (XLM-RoBERTa sequence classifier with one
logit) can be given instead of an alias.

```
local:<model>[,max_tokens=N][,batch=N]
```

- `max_tokens` — tokens per (query, document) pair, default **256**,
  max 512. The query is kept whole; the document's tail is cut. This is
  the main latency lever (see below).
- `batch` — max pairs per forward pass (default 16); batches are also
  capped at `rows × longest² ≤ 4 × 512²` to bound memory.

Scores are `sigmoid(logit)` in `[0, 1]`, comparable within one call.

## In duxx-server

Start the server with `--reranker local:mmarco-minilm-l12` (or
`DUXX_RERANKER`); the Docker image has the `local-rerank` feature built
in. `INFO` shows it as `reranker:`.

```
RERANK <query> <k> <doc1> [doc2 ...]
  → up to k [index, score] pairs, best first (index = 0-based position
    of the document in the call; score a string in [0, 1])

RECALL <key> <query> <k> RERANK <n>
RECALL.FILTER <key> <json_filter> <query> <k> RERANK <n>
  → hybrid-recall n candidates (n ≥ k, ≤ 1000), rerank them, reply with
    the best k in the usual [id, score, text] shape; score is the
    reranker's
```

```
> RERANK "RBI repo rate" 2 "Kohli scored a century" "RBI kept the repo rate at 6.5%" "Taj Mahal history"
1) 1) (integer) 1
   2) "0.9…"
2) …
```

Without a configured reranker both forms reply
`ERR no reranker configured`. Inference runs on the calling connection's
thread inside `tokio::task::block_in_place`, so a long rerank does not
stall other connections.

## Performance

A cross-encoder costs a full transformer pass **per candidate**, so
latency is set by `candidates × max_tokens × layers`, not by the store.

Measured on an AWS **c7g.xlarge** (4 × Graviton3, aarch64), release
build, 50 candidates of ~500 tokens (English + Hindi news text), median of
3 runs. The host was also serving a live site and crawler (load average
7–9 on 4 vCPUs) and the benchmark ran at `nice 15`, so it got ~1.5
effective cores even when given 4 threads — an idle host is faster.

| Model | `max_tokens` | 50 docs, 2 threads | 50 docs, 4 threads (≈1.5 cores) |
|---|---|---|---|
| `mmarco-minilm-l12` | 512 | 38.3 s | — |
| `mmarco-minilm-l12` | 256 | 14.3 s | 11.8 s |
| `mmarco-minilm-l12` | 128 | 5.5 s | 4.0 s |
| `mmarco-minilm-l6` | 512 | 18.6 s | — |
| `mmarco-minilm-l6` | 256 | 7.1 s | 5.8 s |
| `mmarco-minilm-l6` | 128 | 2.6 s | 1.8 s |

- **CPU cost**: `mmarco-minilm-l6` at 128 tokens is ~75 ms of CPU time
  per candidate (3.7 CPU-seconds for 50; ~37 GFLOP/s per core, about
  half of Graviton3's f32 peak — Candle's GEMM is not the bottleneck).
  L12 costs 2×, and doubling `max_tokens` costs ~2.5–3×.
- **Memory**: a `duxx-server` with only the L12 reranker (hash
  embedder) sits at **~770 MB RSS**, with a **~1.5 GB peak while the
  model loads** (f32 weights: 118 M parameters, most of them the 250 k
  XLM-R vocabulary). With the e5 embedder as well, budget ~1.6 GB
  resident and give the container ≥ 2.5 GB so start-up peaks fit — under
  a 1.5 GB cgroup limit the combined server was OOM-killed while loading
  the second model.
- **Load**: 3–10 s from the local model store (first start also
  downloads ~430–470 MB).

**50 × 500-token candidates in 400 ms is not reachable on 4 CPU cores**
with any of these models: at 512 tokens L12 is ~1.3 TFLOP of work, ~9 s
even on an idle 4-core host. What fits a few-hundred-millisecond budget
on that hardware is ~20 candidates of title + lead (≈128 tokens) through
`mmarco-minilm-l6` — e.g.
`--reranker local:mmarco-minilm-l6` and `RECALL key q 10 RERANK 20` —
or more candidates at a lower `max_tokens`. Send the reranker short
text (title + snippet), not whole pages.

## Testing

```text
cargo test -p duxx-rerank                                   # trait, overlap, specs
DUXX_MODEL_DIR=/models cargo test -p duxx-rerank --features local --release -- --include-ignored
DUXX_MODEL_DIR=/models RAYON_NUM_THREADS=4 \
  cargo run --release -p duxx-rerank --features local --example bench_models
```

The model tests (pair tokenisation `<s> q </s></s> d </s>`, query kept
whole under truncation, scores in `[0, 1]`, the relevant document ranked
first in English and Hindi, batched = single scores) download the model
on first run and are `#[ignore]`d in the default test run.
