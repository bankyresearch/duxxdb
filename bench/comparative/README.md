# Comparative bench — DuxxDB vs LanceDB (Phase 4.6)

This directory holds an **embedded** comparative benchmark: DuxxDB
(via the Python wheel) and LanceDB on identical workloads, in the
same Python process, on the same machine. No network — keeps the
numbers honest.

## Run it

```bash
pip install duxxdb-0.1.0-cp38-abi3-*.whl   # build via maturin first
pip install lancedb

python bench.py --n 1000  --queries 100  # both backends
python bench.py --n 10000 --queries 200  # bigger workload
python bench.py --backend duxxdb           # single backend
```

## Measured numbers

> **Workload:** L2-normalized random Gaussian vectors, dim 128.
> Insert one document at a time, then 100–200 vector recalls (k=10).
> Single-thread Python on a Windows-mingw build of the DuxxDB wheel.

### N = 1,000 documents, 100 queries

| backend  | ins p50 (µs) | ins p99 (µs) | ins/s    | rec p50 (µs) | rec p99 (µs) |
|----------|-------------:|-------------:|---------:|-------------:|-------------:|
| duxxdb   |        354.9 |       2470.0 |    1,959 |        322.3 |        755.0 |
| lancedb  |     35,155.7 |    805,239.4 |       17 |     73,014.8 |    122,649.2 |

### N = 10,000 documents, 200 queries (DuxxDB only — see caveats below)

| backend  | ins p50 (µs) | ins p99 (µs) | ins/s    | rec p50 (µs) | rec p99 (µs) |
|----------|-------------:|-------------:|---------:|-------------:|-------------:|
| duxxdb   |       2,505.9 |       8,831.1 |      335 |       2,587.9 |       4,185.4 |

DuxxDB at 10× scale: insert latency 7× higher (HNSW build cost grows
with graph size), recall latency 8× higher. **Both still well under
the 10 ms p99 we set as the target in [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).**

## Honest caveats

**These numbers are NOT a fair head-to-head.** A few asymmetries:

1. **DuxxDB Python wheel runs in-memory only;** LanceDB always writes
   to disk. LanceDB is paying I/O DuxxDB isn't. For an apples-to-apples
   durable comparison, run DuxxDB through the gRPC daemon with
   `--storage dir:./path` — that adds the disk write cost back. (See
   the network bench TODO below.)

2. **Single-row inserts** are LanceDB's worst case. Real LanceDB
   workloads batch (`tbl.add([row1, row2, ...])`); per-row inserts
   force a separate columnar fragment per row. DuxxDB has no such
   penalty because its in-memory tantivy / HNSW append cheaply.

3. **No vector index on LanceDB.** The default LanceDB table does
   linear-scan vector search. DuxxDB always uses HNSW. Adding
   `tbl.create_index()` to LanceDB would close part of the recall gap.
   It's not enabled here because LanceDB recommends ≥ 256 points per
   IVF partition and the small-N tests fall below that threshold.

4. **Hybrid vs vector-only.** DuxxDB's `recall` is hybrid (vector +
   BM25 fused via RRF). LanceDB's `tbl.search(vec)` is vector-only.
   Hybrid is harder; DuxxDB is doing more work and still winning.

5. **Embedding dim 128.** Larger dims (e.g. 1536 from OpenAI) shift
   relative numbers — both systems do more work per insert/recall but
   the network overhead becomes a smaller share. Run with `--dim 1536`
   to see.

6. **Python overhead.** Both backends are called from Python. PyO3
   marshaling (DuxxDB) and PyArrow round-trips (LanceDB) add ~10–50 µs
   per call. Pure-Rust criterion benches in
   [`crates/duxx-bench`](../../crates/duxx-bench) measure DuxxDB at
   ~187 µs recall on 10 k docs (dim 32) — much faster than the
   Python-perceived 2587 µs at dim 128.

## Reproducibility

- Random seed 42 in `bench.py` so corpus + queries are deterministic.
- Toy text + topic mix; not a semantic-retrieval quality bench.
- For accuracy / recall-quality measurement, use a real corpus
  (MS MARCO, NQ, BEIR) — out of scope for this latency micro-bench.

## Network targets — TODO

[`docker-compose.yml`](./docker-compose.yml) spins up Redis Stack,
Qdrant, and pgvector. The corresponding `Backend` classes in
`bench.py` are stubs with TODO markers — drop in the appropriate
client library calls and re-run. The honest network-vs-network
comparison is **DuxxDB-via-gRPC** against each peer; uncomment the
`duxxdb` service in the compose file once we publish a Docker image.

When this lands, the reportable numbers will be:

```
| backend            | ins p50 | rec p50 | rec p99 |
|--------------------|--------:|--------:|--------:|
| duxxdb (embedded)  |   ...   |   ...   |   ...   |
| duxxdb (gRPC)      |   ...   |   ...   |   ...   |
| redis-stack        |   ...   |   ...   |   ...   |
| qdrant             |   ...   |   ...   |   ...   |
| pgvector           |   ...   |   ...   |   ...   |
| lancedb (embedded) |   ...   |   ...   |   ...   |
```

The framework's there; the work is straightforward client wiring.
Tracked as Phase 4.7 (network comparative bench).
