# Comparative benchmark

A market-grade comparison of DuxxDB against the systems it is usually weighed
against — **Redis Stack**, **Qdrant**, and **pgvector** — plus an exact
brute-force `reference`. It measures **retrieval quality** (recall@k / nDCG@k vs
an exact cosine ground truth), **latency**, and **throughput under concurrent
load**, with a fair disk-backed-vs-disk-backed configuration.

The full methodology, fairness rules, and how to read the numbers live in
**[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md)**. This file is the quick start.

> We do not publish an unqualified "Nx faster" headline. Any speed claim must
> carry its workload, dimensions, hardware, and the peer's configuration.

## One command

```bash
cd bench/comparative

# Quick: embedded DuxxDB + exact reference. Needs only numpy + the duxxdb wheel.
./run.sh --quick                 # ./run.ps1 -Quick   on Windows

# Full: stands up redis/qdrant/pgvector + the disk-backed DuxxDB gRPC daemon,
# runs the sweep, tears everything down. Needs Docker + a Rust toolchain.
./run.sh                         # ./run.ps1          on Windows
./run.sh --dims 128,768,1536 --n 50000 --queries 500 --concurrency 1,8,32
```

Results print as markdown and are written to `results-<timestamp>.json`
(hardware + config embedded, so the run is reproducible).

## What you get

- **Quality:** `recall@k`, `nDCG@k` for every backend vs the exact cosine top-k
  on a clustered corpus (real nearest neighbours, not uniform noise). The
  `reference` backend must score `recall@k == 1.0` — the built-in self-check.
- **Latency:** insert and single-client recall p50/p95/p99 (µs).
- **Throughput:** achieved QPS and recall p95/p99 under 1 / 8 / 32 concurrent
  clients, for the network backends (`duxx-grpc` vs redis / qdrant / pgvector).

## Fairness (the short version — see BENCHMARKS.md for the rules)

1. **DuxxDB runs disk-backed** (`open_at`), compared like-for-like against the
   disk-backed peers — not the old in-memory-vs-disk comparison.
2. **DuxxDB queries in hybrid mode** (vector + BM25 + RRF), its real behaviour.
   Ground truth is pure cosine, so this *understates* DuxxDB's vector recall
   rather than flattering it.
3. **Concurrency is measured network-vs-network.** In-process backends hold the
   GIL, so they are reported single-client (latency only); the sweep runs over
   the served systems.
4. **Same corpus, same queries, same seed (42)** for every backend; doc ids are
   the corpus index so returned ids map back to ground truth exactly.

## Manual driving

```bash
# Regenerate the gRPC Python stubs into _proto/ (gitignored) if needed:
python -m grpc_tools.protoc -Icrates/duxx-grpc/proto \
    --python_out=bench/comparative/_proto \
    --grpc_python_out=bench/comparative/_proto \
    crates/duxx-grpc/proto/duxx.proto

# Peers up, then pick backends explicitly:
docker compose up -d
python bench.py --backend duxxdb,redis,qdrant,pgvector -n 10000 -q 200 --dims 128
```

## Rust micro-benchmarks

Component-level Criterion benches (index build, recall, churn, batch ingest,
filtered recall) are in [`crates/duxx-bench`](../../crates/duxx-bench):

```bash
cargo bench -p duxx-bench
```

Use these for single-subsystem regression tracking on a fixed machine — not for
cross-system claims.

## Caveats

- Synthetic corpus (clustered Gaussian mixture) — good for ANN recall and load,
  but **not** a domain-relevance bench. For text-retrieval quality use BEIR /
  MS MARCO with real embeddings.
- Python client overhead taxes every backend equally; the Rust benches isolate
  DuxxDB without it.
- Generated proto stubs in `_proto/` are gitignored — regenerate as above.
