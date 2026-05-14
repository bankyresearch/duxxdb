# DuxxDB — Roadmap

Phased implementation plan. Each phase ships something runnable.

---

## Phase 0 — Foundation (this session)

- [x] Architecture locked (see [ARCHITECTURE.md](./ARCHITECTURE.md))
- [ ] Rust toolchain installed
- [ ] Git repo init (local)
- [ ] Cargo workspace with 10 crate stubs
- [ ] Apache 2.0 LICENSE + NOTICE
- [ ] CI config (GitHub Actions) — not pushed yet, ready to go
- [ ] First `cargo build --workspace` green

**Exit criterion:** `cargo build` passes on all crates, even if they only contain `pub fn version() -> &'static str`.

---

## Phase 1 — Embedded core (Week 1–2)

Goal: working in-process database with hybrid search on 1M rows.

- [ ] `duxx-core`: `Error`, `Config`, `Schema`, `Value` types
- [ ] `duxx-storage`:
  - [ ] `Table::open(path, schema)`
  - [ ] `Table::insert(batch: RecordBatch)`
  - [ ] `Table::scan(filter) -> Stream<RecordBatch>`
  - [ ] redb-backed `SessionStore`
- [ ] `duxx-index`:
  - [ ] `VectorIndex` (usearch) build / search / incremental insert
  - [ ] `TextIndex` (tantivy) build / search
- [ ] `duxx-query`:
  - [ ] `recall()` with RRF fusion
  - [ ] Filter pushdown via DataFusion
- [ ] Example: `examples/chatbot_memory.rs` — insert 10 k messages, recall, print top-10

**Exit criterion:** `cargo run --example chatbot_memory` runs end-to-end, recall p99 < 20 ms on 100 k rows.

---

## Phase 2 — Production indices (in progress)

- [x] **2.1** — Replace placeholder `TextIndex` with [tantivy] BM25
- [x] **2.2** — Replace placeholder `VectorIndex` with [hnsw_rs] HNSW
- [x] **2.5** — Batched tantivy commits + auto-flush on read
      (≈30× bulk-insert speedup)
- [x] **2.3** — Durable storage via [redb] behind a pluggable
      `Storage` trait. Server flag `--storage redb:./path` enables
      row durability; indices rebuild from rows on open.
- [x] **2.3.5** — Index persistence: disk-backed tantivy
      (`MmapDirectory`) + HNSW dump/load (`hnsw_rs::file_dump`).
      Server flag `--storage dir:./path` enables full persistence
      (rows + indices). Graceful reopens skip the rebuild (fast path);
      hard kills fall back to row-rebuild (cold path). Ctrl+C handler
      added so `docker stop` / SIGTERM reach the fast path.

[redb]: https://github.com/cberner/redb

[tantivy]: https://github.com/quickwit-oss/tantivy
[hnsw_rs]: https://crates.io/crates/hnsw_rs

**Measured (Phase 2.5, debug release, single thread, 100 k vector capacity):**
- recall  @ 100 docs:   123 µs median
- recall  @ 1 k docs:   166 µs median
- recall  @ 10 k docs:  373 µs median   (target: < 10 ms — beating 25×)
- bulk insert 100 docs: 14 ms total
- bulk insert 1 k  docs: 342 ms total

---

## Phase 2.6 — Agent primitives ✅ Done

- [x] `duxx-memory::ToolCache` — exact + semantic-near-hit (cosine ≥ 0.95)
- [x] `duxx-memory::SessionStore` — sliding-TTL KV with lazy eviction
- [x] `Memory::effective_importance(half_life)` — exponential decay
- [x] `MemoryStore::recall_decayed()` — recall reranked by decayed importance
- [ ] Pluggable embedding providers (OpenAI, Cohere, local BGE) — Phase 3
- [ ] Background decay worker (currently lazy on read) — future

---

## Phase 3 — Server + bindings (Phase 3.1 done; rest planned)

- [x] **3.1** — `duxx-mcp` JSON-RPC 2.0 stdio server with `remember` /
      `recall` / `stats` tools. Connect any MCP agent (Claude Desktop,
      Cline) by pointing its config at the `duxx-mcp` binary.
- [x] **3.2** — `duxx-server` RESP2/3 daemon (Valkey/Redis-compatible).
      `valkey-cli`, `redis-cli`, redis-rs / node-redis / go-redis all
      work unchanged.
- [x] **3.3** — Python bindings (`bindings/python`, PyO3 + maturin
      abi3-py38). Single wheel works on Python 3.8 through 3.13.
- [x] **3.4** — Node.js / TypeScript bindings (`bindings/node`,
      napi-rs v2). `npm run build` produces a `.node` native module +
      `index.js` + `index.d.ts`. Builds on Linux / macOS / Windows-MSVC.
- [x] **3.5** — gRPC daemon (tonic + prost) with streaming Subscribe.
      `duxx-grpc` binary at `127.0.0.1:50051` by default; proto schema
      at `crates/duxx-grpc/proto/duxx.proto`. Verified Python grpcio
      end-to-end including streaming Subscribe.

---

## Phase 3 — Server + bindings (Week 4–5)

- [ ] `duxx-server`: gRPC + RESP3 daemon
- [ ] `duxx-mcp`: MCP server (stdio + SSE transport)
- [ ] Python bindings (`bindings/python`, PyO3 + maturin)
- [ ] TypeScript bindings (`bindings/node`, napi-rs)
- [ ] `pip install duxxdb` (TestPyPI)
- [ ] `npm install duxxdb` (local pack)

**Exit criterion:** a Claude / GPT agent can `pip install duxxdb` and store + recall memories via MCP.

---

## Phase 4 — Reactive subscriptions (mostly done)

- [x] `duxx-reactive::ChangeBus` over `tokio::sync::broadcast`
- [x] `MemoryStore::subscribe()` + `remember()` publishes `ChangeEvent`
- [x] `SUBSCRIBE` / `UNSUBSCRIBE` over RESP — verified live with
      Python clients across two connections
- [x] Redis pub/sub `message`-format pushes carrying JSON ChangeEvents
- [x] Phase 4 micro-benchmarks: hybrid recall, single insert, bulk insert
- [ ] WAL tailer (durable resume tokens) — depends on Phase 2.3 (Lance)
- [ ] Comparative benchmarks vs Redis/Qdrant/pgvector/LanceDB on
      identical workloads
- [x] **Phase 4.5** — `PSUBSCRIBE` patterns + per-key channels.
      `MemoryStore.remember` publishes to `memory.<key>`; supervisors
      `PSUBSCRIBE memory.*` to filter by user/agent.
- [ ] **v0.1.0 release tag** — after Phase 2.3 lands

**Status:** the in-process reactive path works end to end. Network
durability (resume after crash) waits on Lance.

---

## Phase 5 — Cold-tier export ✅ Shipped (Parquet)

- [x] **Apache Parquet** export via [`duxx-coldtier`](../crates/duxx-coldtier/).
  - `ParquetExporter::write(&store, path)` library API.
  - `duxx-export --storage dir:./path --out file.parquet` binary.
  - Schema: `id` / `key` / `text` / `embedding` (FixedSizeList<Float32, dim>) /
    `importance` / `created_at_ns`.
  - SNAPPY-compressed by default; Spark / DuckDB / Polars / pandas /
    pyarrow read it natively.
- [ ] Delta Lake wrapping (`deltalake` crate) — defer until a
      production user actually needs the transaction log + time travel.
      Parquet covers the common "export to lakehouse" use case.
- [ ] Iceberg / Unity Catalog — same posture as Delta; defer.
- [ ] Scheduled / streaming export — currently one-shot; periodic
      re-export via cron / systemd timer is the standard pattern.

**Exit criterion (met):** a DuxxDB store can be exported to a
Parquet file readable by every major analytics tool. End-to-end
verified via pyarrow.

---

## Phase 6.1 — Production hardening ✅ Shipped

UAT-ready prod-level capabilities for the core daemons.

- [x] **Authentication.** `--token TOKEN` / `DUXX_TOKEN` env on both
      `duxx-server` (RESP) and `duxx-grpc`. Constant-time compare.
      RESP returns Redis-compatible `NOAUTH` / `WRONGPASS`; gRPC uses
      a tonic `Interceptor` checking `x-duxx-token` metadata on every
      RPC. Pre-AUTH RESP allows only `PING` / `AUTH` / `QUIT` / `HELLO`.
- [x] **Health.** gRPC server registers
      `grpc.health.v1.Health` via [tonic-health]; standard
      `grpc_health_probe` / Kubernetes probes work unchanged. RESP
      `/health` exposed via the metrics endpoint (returns 200 OK).
- [x] **Prometheus metrics.** `--metrics-addr HOST:PORT` /
      `DUXX_METRICS_ADDR` binds a separate hyper listener serving
      `/metrics` (Prometheus text format) + `/health`. Counters for
      connections / commands / errors / remembers / recalls; gauges
      for active connections / memory count / session count;
      histogram for per-command latency.
- [x] **Graceful shutdown.** `serve_with_shutdown(addr, signal, drain)`
      stops accepting on Ctrl+C / SIGTERM, drains in-flight
      connections up to the budget (`--drain-secs N`, default 30),
      then triggers `MemoryStore::Drop` (tantivy commit + HNSW dump)
      so a `dir:` backend reopen takes the fast path.
- [x] **Backup & restore documented.** `USER_GUIDE.md` § 6 covers
      Parquet snapshot via cron + a Rust restore stub + a
      disaster-recovery posture table.

[tonic-health]: https://crates.io/crates/tonic-health

**Exit criterion (met):** a single binary can be deployed behind a
TLS-terminating load balancer with auth, scraped by Prometheus,
probed for health, and gracefully drained on rolling restarts.

---

## Phase 6.2 — Production hardening II ✅ Shipped

Public-internet readiness — DuxxDB no longer needs a TLS-terminating
sidecar to be exposed safely.

- [x] **TLS-native on RESP** via [rustls] + [tokio-rustls].
      `--tls-cert PATH --tls-key PATH` (or `DUXX_TLS_CERT` /
      `DUXX_TLS_KEY` env). `redis-cli --tls` and any rustls / OpenSSL
      RESP client connect directly. Connection upgrade happens in the
      accept loop; auth + drain + metrics paths all run unchanged.
- [x] **TLS-native on gRPC** via tonic's `tls` feature (rustls under
      the hood). Same flags / env on `duxx-grpc`. `grpcurl --insecure`
      / Python `grpc.secure_channel(creds)` work. Health protocol
      stays unencrypted-on-the-internal-port-friendly via separate
      builder.
- [x] **Row cap + importance-based eviction.**
      `--max-memories N` / `DUXX_MAX_MEMORIES` on `duxx-server`. Once
      exceeded, every `REMEMBER` evicts the row with the lowest
      *effective* (decayed) importance until the count is back at the
      cap. Agent-friendly: a chatbot can keep one big cap and trust
      the store to forget the boring stuff first. `set_eviction_half_life`
      tunes how aggressively old rows look "forgotten".
- [x] **Tests:** TLS handshake end-to-end (rustls client → server →
      RESP PING) + cap eviction policy + cap-with-recall integration.
      Workspace: 112 tests pass.

[rustls]: https://github.com/rustls/rustls
[tokio-rustls]: https://github.com/rustls/tokio-rustls

**Exit criterion (met):** a single binary can be exposed on a public
IP with TLS, an auth token, Prometheus metrics, a memory cap, and a
graceful drain on rolling restarts — no sidecars required.

---

## Phase 7.1 — `duxx-trace` ✅ Shipped (v0.1.1)

Agent observability — Span / Trace / Thread primitives that match
the OTel + OpenInference shape. Six RESP commands surface the API:
`TRACE.RECORD`, `TRACE.CLOSE`, `TRACE.GET`, `TRACE.SUBTREE`,
`TRACE.THREAD`, `TRACE.SEARCH`. `PSUBSCRIBE trace.*` for live tail.

In-memory store; persistence + tantivy-backed JSON-attribute search
land in 7.1b.

duxx-ai side: `DuxxExporter` in `duxx_ai/observability/duxx_exporter.py`
(merged as PR #2). Plugs into the existing `Tracer` and flushes every
finished trace via RESP.

## Phase 7.3 — `duxx-datasets` ✅ Shipped

Versioned eval datasets with **per-row semantic search** and a
killer move competitors can't ship: `DATASET.FROM_RECALL` — turn any
memory-recall result into a new dataset version in one RESP command.

Capabilities:
- Versioned snapshots (monotonic; deletes don't reuse numbers).
- Per-row splits (`train` / `eval` / `test` or custom strings).
- Tag aliases (`golden` / `staging` / `experimental`).
- Semantic row search across the whole catalog or filtered by name.
- Reactive change feed (`PSUBSCRIBE dataset.*`).
- `add_from_texts` convenience for the simple case.

Thirteen RESP commands: `DATASET.CREATE`, `DATASET.ADD`, `DATASET.GET`
(by version or tag), `DATASET.LIST`, `DATASET.NAMES`, `DATASET.TAG`,
`DATASET.UNTAG`, `DATASET.DELETE`, `DATASET.SAMPLE`, `DATASET.SIZE`,
`DATASET.SPLITS`, `DATASET.SEARCH`, `DATASET.FROM_RECALL`.

Tests: 16 crate-level + 8 RESP-level. Workspace at 176 tests.

In-memory store; persistence lands in 7.x.b alongside trace +
prompt persistence.

---

## Phase 7.2 — `duxx-prompts` ✅ Shipped

Versioned prompt registry with **semantic search across the catalog**
(uses DuxxDB's existing embedder — competitors don't ship this).

Capabilities:
- Monotonic versioning per name (1, 2, 3, … — never reused on delete)
- Tag aliases (`prod` / `staging` / `experimental`) — operators move
  a tag to ship a new prompt without touching agent code
- Semantic search via HNSW + the shared embedder
- Line-diff between any two versions
- `PSUBSCRIBE prompt.*` so running agents hot-reload on every put/tag

Nine RESP commands: `PROMPT.PUT`, `PROMPT.GET` (by version or tag),
`PROMPT.LIST`, `PROMPT.NAMES`, `PROMPT.TAG`, `PROMPT.UNTAG`,
`PROMPT.DELETE`, `PROMPT.SEARCH`, `PROMPT.DIFF`.

In-memory store; persistence lands in 7.2b alongside trace persistence.

---

## Phase 6.3+ — Future hardening

Tracked, not yet scheduled.

- mTLS (client cert auth).
- Index-side eviction (HNSW tombstones, tantivy deletes) so cap
  reclaims index memory too — Phase 6.2 only reclaims row + storage
  bytes; the indices retain entries until restart.
- Distributed / sharded mode.
- Row-level security / RBAC.
- Query cache.
- OpenTelemetry tracing export.
- SIMD tuning per architecture (AVX-512, ARM NEON).

---

## How to pick up mid-phase

Each phase has its own issue label (`phase-1`, `phase-2`, …). Pick an unchecked box, open a PR titled `<crate>: <what>`. Every PR must:

1. Compile clean (`cargo clippy --workspace -- -D warnings`).
2. Add / update at least one test.
3. Not regress any benchmark by > 5 %.
