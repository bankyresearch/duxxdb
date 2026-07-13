# DuxxDB Benchmarks & Methodology

This document describes how DuxxDB is benchmarked, what the numbers mean, and
how to reproduce them. The goal is a benchmark an enterprise buyer can trust:
**retrieval quality**, **latency**, and **throughput under load**, with a fair
apples-to-apples configuration against the systems people actually compare us to.

We deliberately do **not** lead with an unqualified "Nx faster" headline. Any
speed claim in DuxxDB docs must cite the workload, dimensions, hardware, and the
competing system's configuration — the numbers below are the source of truth.

---

## 1. What is measured

| Axis | Metric | How |
|------|--------|-----|
| **Quality** | `recall@k`, `nDCG@k` | vs an **exact cosine** ground truth (brute force) on a clustered corpus |
| **Latency (single client)** | insert p50/p95/p99, recall p50/p95/p99 | per-operation wall clock, µs |
| **Throughput under load** | achieved QPS, recall p95/p99 | N concurrent clients for a fixed duration (network backends) |
| **Load** | inserts/s, total load time | full-corpus ingest |

### Why a clustered corpus

Uniform-random vectors have no meaningful nearest neighbours — every ANN index
scores either ~perfect or ~random, which measures nothing. The harness generates
a **Gaussian mixture**: `max(8, N/500)` centroids with points scattered around
them (noise σ=0.35), then L2-normalised. Queries are existing corpus points
perturbed by σ=0.15 — a realistic "find similar" load with a real answer.

### Ground truth

For each query we compute the **exact** cosine top-100 by brute force (NumPy,
unit-norm dot product). `recall@k` is the fraction of the exact top-k a system
returns; `nDCG@k` uses binary relevance (in / out of the exact top-k) with the
standard `1/log2(rank+1)` discount. A built-in `reference` backend re-runs the
same brute force and **must** score `recall@k == 1.0` — the harness's self-check.

---

## 2. Fairness rules

These are the rules that keep the comparison honest. Read them before quoting
any result.

1. **Disk-backed vs disk-backed.** DuxxDB is benchmarked via `open_at` (durable
   redb + persisted HNSW/tantivy), *not* in-memory, so it is compared like-for-
   like against disk-backed Redis Stack, Qdrant, and pgvector. The old
   microbenchmark compared in-memory DuxxDB to disk-backed peers — that number
   is retired.
2. **Native query mode.** Every system is queried the way it ships. DuxxDB uses
   **hybrid** recall (vector ANN + BM25, RRF-fused) — its actual product
   behaviour. Because ground truth is *pure cosine*, hybrid recall can only
   **understate** DuxxDB's vector recall here, never inflate it. This is a
   conservative choice, stated plainly rather than tuned away.
3. **Network vs network for throughput.** In-process backends (embedded DuxxDB,
   `reference`) hold the Python GIL for the duration of a call, so their
   concurrency numbers would not reflect real parallelism. They are reported
   **single-client** (latency only). The concurrency sweep runs against the
   **network** backends — `duxx-grpc` vs redis/qdrant/pgvector — which is the
   fair parallel comparison.
4. **Same corpus, same queries, same seed.** All backends see identical data and
   queries (`seed=42`), and doc ids are the corpus index threaded through every
   backend so returned ids map back to ground truth exactly.
5. **Report the machine.** Every result JSON embeds host CPU, core count, OS, and
   Python version. A latency number without its hardware is not a result.

---

## 3. Reproducing

### Quick (no services)

Embedded DuxxDB + the exact reference. Needs only `numpy` and the `duxxdb` wheel:

```bash
cd bench/comparative
./run.sh --quick                 # or: ./run.ps1 -Quick  on Windows
```

### Full suite

Brings up Redis Stack, Qdrant, and pgvector via Docker Compose, builds and
launches the disk-backed DuxxDB gRPC daemon, runs the sweep, and tears
everything down:

```bash
cd bench/comparative
./run.sh                         # ./run.ps1  on Windows
./run.sh --dims 128,768,1536 --n 50000 --queries 500 --concurrency 1,8,32 --duration 10
```

Requirements: Docker + Compose, a Rust toolchain (for the gRPC daemon), and the
Python clients (`redis`, `qdrant-client`, `psycopg2-binary`, `grpcio`) — the
runner installs these if missing.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--dims` | `128` | comma list of embedding dimensions (use `768,1536` for real models) |
| `-n` | `5000` | corpus size |
| `-q/--queries` | `200` | number of queries |
| `--k` | `1,10` | recall/nDCG cutoffs |
| `--concurrency` | `1,8,32` | client counts for the throughput sweep |
| `--duration` | `5` | seconds per concurrency level |
| `--backend` | `all` | comma list, or `all` / `quick` |
| `--out` | — | write the full JSON report |

---

## 4. Rust micro-benchmarks

Component-level Criterion benches (index build, recall, churn, batch ingest,
filtered recall) live in `crates/duxx-bench`:

```bash
cargo bench -p duxx-bench
```

These isolate a single subsystem on a fixed machine — use them for regression
tracking, not for cross-system claims.

---

## 5. Interpreting the output

- **`reference` recall@k must be 1.0.** If it isn't, the harness or environment
  is broken — stop and fix before trusting anything else.
- **DuxxDB recall@1 vs recall@10.** recall@1 reflects the top hit; recall@10 is
  lower partly because hybrid fusion trades some pure-vector recall for text
  relevance. For a pure-vector-recall comparison, note this caveat explicitly.
- **Latency is µs, single-client** unless read from the concurrency table.
- **QPS is achieved throughput**, not a theoretical cap — it is `ops / elapsed`
  over the measured window at that client count.

Publish results with the JSON attached, or they are not reproducible.
