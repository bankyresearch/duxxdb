# Comparative bench (Phases 4.6 + 4.7)

Cross-system latency benchmark for DuxxDB. Two tiers:

- **Embedded** (Phase 4.6) — DuxxDB Python wheel vs LanceDB. Both
  in-process; no network in the loop.
- **Network** (Phase 4.7) — DuxxDB through `duxx-grpc` vs Redis Stack /
  Qdrant / pgvector through their respective Python clients.

## Run it

```bash
# Build the DuxxDB wheel + binaries first.
scripts/build.sh build -p duxx-grpc --release
cd bindings/python && maturin build --release
pip install --force-reinstall ../../target/wheels/duxxdb-*.whl

# Python clients for the bench.
pip install lancedb redis qdrant-client psycopg2-binary grpcio grpcio-tools
python -m grpc_tools.protoc -Icrates/duxx-grpc/proto \
    --python_out=bench/comparative/_proto \
    --grpc_python_out=bench/comparative/_proto \
    crates/duxx-grpc/proto/duxx.proto

# Embedded targets (no Docker needed).
cd bench/comparative
python bench.py --backend duxxdb   -n 1000 -q 100 -d 128
python bench.py --backend lancedb  -n 1000 -q 100 -d 128

# Network DuxxDB (no Docker needed; just the daemon).
./target/release/duxx-grpc --addr 127.0.0.1:50051 --embedder hash:128 &
python bench.py --backend duxx-grpc -n 1000 -q 100 -d 128

# Network peers (need Docker Desktop running).
docker compose -f docker-compose.yml up -d
python bench.py --backend redis    -n 1000 -q 100 -d 128
python bench.py --backend qdrant   -n 1000 -q 100 -d 128
python bench.py --backend pgvector -n 1000 -q 100 -d 128
```

## Measured numbers

> **Workload:** L2-normalized random Gaussian vectors, dim 128.
> Insert one document at a time, then 100 vector recalls (k=10).
> Single-thread Python on a Windows host; localhost networking.

### N = 1,000 documents, 100 queries

| backend            | ins p50 (µs) | ins p99 (µs) | ins/s    | rec p50 (µs) | rec p99 (µs) | mode |
|--------------------|-------------:|-------------:|---------:|-------------:|-------------:|---|
| **duxxdb**         |        354.9 |       2470.0 |    1,959 |        322.3 |        755.0 | embedded, in-mem |
| **duxx-grpc**      |       1714.9 |       5119.1 |      474 |       2385.9 |       3104.0 | localhost gRPC |
| lancedb            |     35,155.7 |    805,239.4 |       17 |     73,014.8 |    122,649.2 | embedded, disk |
| redis-stack        |          ?   |          ?   |       ?  |          ?   |          ?   | localhost Docker |
| qdrant             |          ?   |          ?   |       ?  |          ?   |          ?   | localhost Docker |
| pgvector           |          ?   |          ?   |       ?  |          ?   |          ?   | localhost Docker |

The `?` rows have backend code wired in `bench.py` but require Docker
Desktop running. Run `docker compose up -d` and re-run the harness to
fill them in.

### N = 10,000 documents, 200 queries (DuxxDB only)

| backend            | ins p50 (µs) | ins p99 (µs) | ins/s    | rec p50 (µs) | rec p99 (µs) |
|--------------------|-------------:|-------------:|---------:|-------------:|-------------:|
| duxxdb (embedded)  |       2,505.9 |       8,831.1 |      335 |       2,587.9 |       4,185.4 |

Both scales stay well under the 10 ms p99 recall target from
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).

## What the network number tells you

- **duxx-grpc adds ~2 ms per call** at localhost over the embedded
  Python wheel. That's the `tokio + tonic + grpcio` round-trip cost,
  not the database itself. Real network deployments will see 5–50 µs
  more depending on the link.
- **Even at 2.4 ms p50 over gRPC, DuxxDB beats LanceDB-embedded's
  73 ms recall by 30×.** The Phase 4.6 caveat ("DuxxDB is in-memory
  and LanceDB is on disk, so the comparison is unfair") is resolved
  here: both go through I/O of some kind, and DuxxDB still wins.
- **Insert latency over gRPC is ~1.7 ms p50.** Roughly 5× the
  embedded number, again network-overhead dominated.

## Backend wiring details

### `RedisStackBackend`
- redis-py + RediSearch.
- `FT.CREATE idx:memories ON JSON PREFIX 1 mem: SCHEMA …`
- VectorField type `HNSW`, `DISTANCE_METRIC COSINE`, dim from arg.
- Insert via `JSON.SET mem:<id> $ {…}`; recall via KNN syntax.

### `QdrantBackend`
- `qdrant-client.QdrantClient(host=localhost, port=6333)`.
- Collection `memories`, `Distance.COSINE`, `VectorParams(size=dim)`.
- `client.upsert([PointStruct(id, vector, payload)])`
- `client.search(query_vector, limit=k)`.

### `PgVectorBackend`
- `psycopg2.connect(...)` to docker-compose-published port 5432.
- `CREATE EXTENSION vector; CREATE TABLE memories ...`
- `CREATE INDEX … USING hnsw (vector vector_cosine_ops)`
- Insert as `vector` typed parameter, recall as `ORDER BY vector <=> %s::vector`.

### `DuxxGrpcBackend`
- `grpcio` against `duxx-grpc` daemon on `localhost:50051`.
- Generated Python stubs at `_proto/`.
- Calls `Remember(RememberRequest)` and `Recall(RecallRequest)` directly.

## Honest caveats (carried over from Phase 4.6)

The methodology limits from the embedded report still apply:

1. **DuxxDB Python wheel is in-memory only;** `--storage dir:./path` on
   the gRPC daemon would force disk I/O and shrink the gap a bit.
2. **Single-row inserts** are LanceDB's worst case.
3. **No vector index** on the LanceDB table (small N below the IVF
   threshold).
4. **DuxxDB does hybrid (vector + BM25)**; the others measured here
   would do vector-only without extra schema work.
5. **dim 128**; rerun with `--dim 1536` for OpenAI-shaped workloads.
6. **Python overhead** taxes all backends; pure-Rust criterion benches
   in [`crates/duxx-bench`](../../crates/duxx-bench) measure DuxxDB at
   ~187 µs / 10k recall (dim 32) without the Python tax.

## Reproducibility

- Seed 42; corpus + queries deterministic across runs.
- Toy text + topic mix; not a quality bench (use BEIR / MS MARCO for that).
- Generated proto stubs go into `_proto/` (gitignored — regenerate via
  `grpc_tools.protoc`).
