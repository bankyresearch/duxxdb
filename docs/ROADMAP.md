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
- [ ] **2.3** — Lance-backed `Table` for durable storage. See
      [PHASE_2_3_PLAN.md](./PHASE_2_3_PLAN.md). Deferred to its own
      session because of the dependency footprint and FFI risk on
      Windows + mingw.

[tantivy]: https://github.com/quickwit-oss/tantivy
[hnsw_rs]: https://crates.io/crates/hnsw_rs

**Measured (Phase 2.5, debug release, single thread, 100 k vector capacity):**
- recall  @ 100 docs:   123 µs median
- recall  @ 1 k docs:   166 µs median
- recall  @ 10 k docs:  373 µs median   (target: < 10 ms — beating 25×)
- bulk insert 100 docs: 14 ms total
- bulk insert 1 k  docs: 342 ms total

---

## Phase 2.6 — Agent primitives (next)

- [ ] `duxx-memory::Memory` — auto-embedding on insert
- [ ] `duxx-memory::TOOL_CACHE` — exact + semantic-near-hit lookup
- [ ] `duxx-memory::SESSION` — hot KV with auto-flush
- [ ] Pluggable embedding providers: OpenAI, Cohere, local BGE
- [ ] Importance decay worker

**Exit criterion:** `TOOL_CACHE` demo showing a cache hit on a paraphrased query.

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

## Phase 4 — Reactive + benchmark (Week 6)

- [ ] `duxx-reactive`: WAL tailer + subscriptions
- [ ] `SUBSCRIBE` over gRPC streaming + MCP SSE
- [ ] `duxx-bench`:
  - [ ] Micro: insert, point-read, vector-search, hybrid recall
  - [ ] Comparison: Redis, Qdrant, pgvector, LanceDB on identical workload
- [ ] Public benchmark report (Markdown + charts)
- [ ] **v0.1.0 release tag**

**Exit criterion:** published benchmark showing we meet or beat all targets in ARCHITECTURE §1.

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
