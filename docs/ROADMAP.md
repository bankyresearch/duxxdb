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

## Phase 5 — Cold-tier bridge (Week 7–8)

- [ ] Arrow Flight exporter
- [ ] Iceberg writer (via `iceberg-rust` or bundled catalog)
- [ ] Delta writer (via `delta-rs`)
- [ ] Unity Catalog ACL stub (read-only)
- [ ] Scheduled compaction / re-embedding from cold → hot

**Exit criterion:** a DuxxDB table can sync to an Iceberg table on S3 / local file store.

---

## Phase 6 — Production hardening (later)

- Distributed / sharded mode.
- Row-level security.
- Query cache.
- Observability: OpenTelemetry, Prometheus metrics.
- SIMD tuning per architecture (AVX-512, ARM NEON).

---

## How to pick up mid-phase

Each phase has its own issue label (`phase-1`, `phase-2`, …). Pick an unchecked box, open a PR titled `<crate>: <what>`. Every PR must:

1. Compile clean (`cargo clippy --workspace -- -D warnings`).
2. Add / update at least one test.
3. Not regress any benchmark by > 5 %.
