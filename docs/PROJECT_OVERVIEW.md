# DuxxDB — Project Overview

> The database built for AI agents. Embedded + server, hybrid (vector +
> BM25 + structured), durable, with reactive subscriptions and Apache
> Parquet cold-tier. **Closed UAT — feature-complete through Phase 5.**

[![CI](https://github.com/bankyresearch/duxxdb/actions/workflows/ci.yml/badge.svg)](https://github.com/bankyresearch/duxxdb/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

---

## Table of Contents

1. [30-second pitch](#1-30-second-pitch)
2. [The problem we solve](#2-the-problem-we-solve)
3. [What DuxxDB is](#3-what-duxxdb-is)
4. [What makes it unique](#4-what-makes-it-unique)
5. [Capabilities today](#5-capabilities-today)
6. [Performance — measured](#6-performance--measured)
7. [Architecture at a glance](#7-architecture-at-a-glance)
8. [Competitive landscape](#8-competitive-landscape)
9. [Install & quickstart](#9-install--quickstart)
10. [Project structure](#10-project-structure)
11. [Roadmap](#11-roadmap)
12. [Limitations & caveats](#12-limitations--caveats)
13. [Contributing](#13-contributing)
14. [License & acknowledgements](#14-license--acknowledgements)

---

## 1. 30-second pitch

AI agents (chatbots, voice bots, autonomous agents) need three databases
glued together: a KV store for session state, a vector DB for semantic
recall, and a relational store for structured facts. Every glue layer
adds latency. Voice bots blow their 200 ms response budget; chatbots
feel sluggish; autonomous agents lose context.

**DuxxDB merges all three behind one low-latency Rust engine** with
agent-native primitives (`MEMORY`, `TOOL_CACHE`, `SESSION`) baked in.
Same data store. Same query plan. Sub-millisecond hybrid recall. Pure
Rust, no GC pauses. Embedded, RESP server, gRPC server, and MCP server
all from the same workspace.

---

## 2. The problem we solve

### The "three databases" tax

A typical production agent stack today:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Redis   │    │ Qdrant   │    │  Postgres    │
│  session │    │  vectors │    │  facts/users │
│  ~0.5ms  │    │  ~10ms   │    │  ~1-5ms      │
└──────────┘    └──────────┘    └──────────────┘
      ▲              ▲                ▲
      └──────────────┼────────────────┘
                     │
              every turn:
              3 RTTs, 3 client libs,
              3 sets of credentials,
              eventual consistency
```

The tax shows up in three places:

| Tax | Cost |
|---|---|
| **Latency** | Each network hop is 100 µs–1 ms; 3 hops = 300 µs–3 ms before any work. |
| **Consistency** | A user message, its embedding, and its session metadata land in three stores at three times. |
| **Code complexity** | Glue code (transactions across stores, retries, drift) is the silent killer of agent reliability. |

### Latency budgets that matter

| Use case | Total turn budget | DB share |
|---|---|---|
| Voice bot (real-time) | ~200 ms | < 20 ms |
| Chat (snappy) | ~1 s | < 100 ms |
| Autonomous agent | ~10 s | < 500 ms |

For voice in particular, every millisecond cut from the DB path goes
straight back to the user as lower latency.

---

## 3. What DuxxDB is

### One sentence

A pure-Rust, embedded-or-server hybrid database that natively
understands **agent memory**, fuses **vector + full-text + structured**
retrieval in a single query plan, and runs in-process or as a
RESP/gRPC/MCP server from the same codebase.

### Six integration surfaces, one core

```
                 ┌─────────────────────────┐
                 │       your agent        │
                 └────────────┬────────────┘
                              │
        ┌─────────┬───────────┼───────────┬─────────┐
        ▼         ▼           ▼           ▼         ▼
   embedded   Python      Node /TS      RESP    gRPC    MCP stdio
   (cargo)    wheel       (.node)      TCP    server   (JSON-RPC)
        │         │           │           │         │
        └─────────┴────┬──────┴────┬─────┴────┬─────┘
                       ▼           ▼          ▼
                 ┌─────────────────────────────────┐
                 │  duxx-memory + duxx-query +     │
                 │  duxx-index + duxx-storage      │
                 │  + duxx-reactive + duxx-embed   │
                 └─────────────────────────────────┘
                                │
                                ▼
                 ┌─────────────────────────────────┐
                 │   redb / tantivy / HNSW          │
                 │   on disk under <data>/          │
                 └─────────────────────────────────┘
                                │
                                ▼ (cold-tier export)
                 ┌─────────────────────────────────┐
                 │   Apache Parquet → Spark /      │
                 │   DuckDB / Polars / pandas      │
                 └─────────────────────────────────┘
```

Each surface is a thin adapter over the same hot-path code. There are
no rewrites between modes — the embedded `MemoryStore` *is* what the
RESP / gRPC / MCP servers wrap.

### Three pillars

1. **Hybrid retrieval.** Vector ANN (HNSW) + BM25 full-text + structured
   filters fused via Reciprocal Rank Fusion (RRF) in one round-trip.
2. **Agent-native primitives.** First-class `MEMORY`, `TOOL_CACHE`
   (with semantic-near-hit), `SESSION` (sliding TTL), and importance
   decay — not generic tables with "examples for chatbots".
3. **MCP-native.** First-class Model Context Protocol stdio server so
   any Claude / GPT / Cline agent plugs in with zero glue code.

---

## 4. What makes it unique

| Differentiator | Status | Who else has it? |
|---|---|---|
| Hybrid (vector + BM25 + structured) in **one query plan** with **RRF** | ✅ | Weaviate has hybrid but not RRF; pgvector has both but no RRF planner |
| **Embedded *and* server** from one Rust codebase | ✅ | LanceDB has embedded; Qdrant is server-only |
| Agent primitives (`MEMORY`, `TOOL_CACHE`, `SESSION`) as **first-class types** | ✅ | Nobody — agent frameworks (LangChain, LlamaIndex) bolt this on top of generic stores |
| `TOOL_CACHE` with **semantic-near-hit** lookup (cosine ≥ 0.95 of args embedding) | ✅ | Nobody — saves real money on expensive tool calls |
| **MCP-native** wire protocol (stdio JSON-RPC 2.0) | ✅ | Nobody (MCP is from late 2024; everyone else writes adapter code) |
| **Reactive subscriptions** with glob pattern + per-key channels | ✅ | Redis/Valkey have it on KV (`PSUBSCRIBE`); nobody has reactive *on hybrid memory* |
| **gRPC streaming Subscribe** for typed cross-language consumers | ✅ | Nobody |
| **Importance decay** (cross-restart Unix-epoch timestamps) | ✅ | Nobody — agent frameworks decay client-side |
| Pure Rust core — no GC pauses, no FFI bottlenecks | ✅ | Qdrant ✅, LanceDB ✅; Pinecone/Weaviate/Milvus all have GC |
| Apache 2.0, no proprietary "open core" tax | ✅ | Pinecone is proprietary; Milvus has Zilliz cloud lock-in pressure |
| **Apache Parquet cold-tier export** | ✅ | Nobody bundles it for agent memory |

**The single biggest differentiator:** every other tool forces you to
*choose* between vector / KV / SQL, or to *compose* multiple stores
yourself. DuxxDB collapses that decision and ships across six client
surfaces from one binary.

---

## 5. Capabilities today

All of Phase 0 → 5 ships. **99 tests passing** workspace-wide; CI
matrix on Linux + macOS + Windows.

### Storage layer

- **`duxx-core`** — `Schema`, `Column`, `ColumnKind` (`I64`, `F64`,
  `Bool`, `Text { bm25 }`, `Vector(VectorSpec)`, `Timestamp`, `Json`),
  `Value`, `Error`, `Config`
- **`duxx-storage`** — `Storage` trait + `MemoryStorage` (HashMap-backed)
  + `RedbStorage` (durable, ACID, MVCC). Schema-aware `Table` placeholder
  reserved for future Lance-shape integration.
- **`duxx-index`**:
  - `TextIndex` — production [tantivy](https://github.com/quickwit-oss/tantivy)
    BM25, in-memory or disk-backed (`MmapDirectory`), batched commits,
    auto-flush on read.
  - `VectorIndex` — production [hnsw_rs](https://crates.io/crates/hnsw_rs)
    HNSW (cosine), in-memory or disk-backed (file_dump on graceful drop,
    cold-rebuild from row store on hard kill).
- **`duxx-query`** — Reciprocal Rank Fusion, `hybrid_recall`.
- **`duxx-memory`** — `MemoryStore` (`remember` / `recall` /
  `recall_decayed`); `ToolCache` (exact + semantic-near-hit, TTL,
  purge); `SessionStore` (sliding TTL, lazy eviction).
- **`duxx-reactive`** — `ChangeBus` over `tokio::sync::broadcast`.
- **`duxx-embed`** — `Embedder` trait + `HashEmbedder` /
  `OpenAIEmbedder` / `CohereEmbedder` (HTTP via `reqwest`+`rustls`).
- **`duxx-coldtier`** — `ParquetExporter`. Apache Parquet output:
  Spark / DuckDB / Polars / pandas / pyarrow read it natively.

### Servers + bindings

- **`duxx-server`** — RESP2/3 TCP server. `valkey-cli` /
  `redis-cli` compatible. Standard commands plus DuxxDB extensions
  (`REMEMBER`, `RECALL`, `SUBSCRIBE`, `PSUBSCRIBE`).
- **`duxx-mcp`** — Model Context Protocol stdio server. Claude
  Desktop / Cline / any MCP agent plugs in via config.
- **`duxx-grpc`** — Tonic gRPC server with streaming `Subscribe`.
  Schema in [`crates/duxx-grpc/proto/duxx.proto`](../crates/duxx-grpc/proto/duxx.proto).
- **Python wheel** — PyO3 + maturin abi3-py38; one wheel works on
  Python 3.8 → 3.13.
- **Node bindings** — napi-rs v2; `.node` native module per platform.

### Storage modes

```bash
duxx-server                                        # in-memory only
duxx-server --storage redb:./data/duxx.redb        # rows persist; indices rebuild on open
duxx-server --storage dir:./data/duxx              # FULLY persistent (rows + tantivy + HNSW)
```

`dir:` graceful reopens skip the rebuild (sub-second cold start);
hard kills auto-fall-back to row-rebuild.

### Reactive

```bash
redis-cli -p 6379 PSUBSCRIBE memory.alice
# any other client's `REMEMBER alice ...` pushes here as
# pmessage memory.alice memory.alice {"row_id":N,"kind":"insert",...}
```

Same primitive available over gRPC streaming.

### Cold-tier

```bash
duxx-export --storage dir:./data/duxx --out hourly.parquet
# rows: id, key, text, embedding (FixedSizeList<f32, dim>),
#       importance, created_at_ns
```

→ Full surface tour with copy-paste examples per language:
**[USER_GUIDE.md](USER_GUIDE.md)**.

---

## 6. Performance — measured

Median of 20 samples, criterion 0.5, single thread, debug release
build, 100 k vector capacity, toy embedder (32-dim, hash-bucket
tokens). Reproduce via
[`crates/duxx-bench/benches/recall.rs`](../crates/duxx-bench/benches/recall.rs).

### Hybrid recall (`MemoryStore::recall`, k = 10)

| Corpus size | Median | Throughput | vs target (10 ms) |
|---:|---:|---:|---:|
| 100   | **123 µs** | 8.1 k QPS | 81× headroom |
| 1 000 | **166 µs** | 6.0 k QPS | 60× headroom |
| 10 000 | **373 µs** | 2.7 k QPS | 27× headroom |

**Reading:** going from 100 to 10 000 documents (100× more data),
recall latency rose only ~3× — sub-linear scaling, exactly what HNSW
is supposed to deliver.

### Insert / bulk insert

| Operation | Time | Notes |
|---|---:|---|
| Single insert (cold store) | 4 ms | HNSW lazy-init dominates |
| Bulk insert 100 docs | 14 ms total | 140 µs / doc amortized |
| Bulk insert 1 000 docs | 342 ms total | 342 µs / doc amortized |

Phase 2.5's batched tantivy commits gave **~30× speedup** on the 100-
doc bulk case (was ~400 ms → 14 ms) and **~12×** on the 1 k case
(was ~4 s → 342 ms).

### Comparative bench (Python clients, dim 128, N=1k)

| backend | ins p50 | rec p50 | rec p99 | mode |
|---|---:|---:|---:|---|
| **duxxdb** (Python wheel, embedded) | 354.9 µs | 322.3 µs | 755.0 µs | in-memory |
| **duxxdb** (`duxx-grpc` over localhost) | 1714.9 µs | 2385.9 µs | 3104.0 µs | network |
| lancedb | 35,155.7 µs | 73,014.8 µs | 122,649.2 µs | embedded, disk |

Even over the gRPC round-trip, DuxxDB recall (2.4 ms p50) beats
LanceDB-embedded recall (73 ms) by **30×**. Full numbers + honest
caveats: [`bench/comparative/README.md`](../bench/comparative/README.md).

Redis Stack / Qdrant / pgvector are wired in the same harness;
unblock by running `docker compose up -d` and re-running the bench.

---

## 7. Architecture at a glance

Full spec: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
              AGENT (chatbot / voice bot / autonomous)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
    embedded                                ┌───────────────┐
    duxx-memory (lib)                       │ duxx-server   │ RESP TCP
        │                                   ├───────────────┤
        ▼                                   │ duxx-grpc     │ gRPC + streaming
    Python wheel                            ├───────────────┤
    (PyO3, abi3)                            │ duxx-mcp      │ stdio JSON-RPC
        │                                   └───────┬───────┘
    Node .node                                      │
    (napi-rs)                                       │
                                                    ▼
                            ┌───────────────────────────────────┐
                            │  duxx-memory (MEMORY / TOOL_CACHE  │
                            │              / SESSION + decay)    │
                            └───────┬────────────┬──────────────┘
                                    │            │
                                    ▼            ▼
                            ┌───────────┐   ┌───────────┐
                            │ duxx-     │   │ duxx-     │
                            │ query     │   │ reactive  │
                            │ (RRF +    │   │ (Change   │
                            │ hybrid)   │   │  Bus)     │
                            └─────┬─────┘   └───────────┘
                                  │
                            ┌─────┼──────────────┐
                            ▼     ▼              ▼
                    ┌─────────┐ ┌─────────┐ ┌──────────────┐
                    │ duxx-   │ │ duxx-   │ │  duxx-embed  │
                    │ index   │ │ storage │ │  (HTTP +     │
                    │         │ │         │ │   hash)      │
                    │ tantivy │ │ redb /  │ └──────────────┘
                    │ hnsw_rs │ │ memory  │
                    └─────────┘ └─────────┘
                          │           │
                          └─────┬─────┘
                                ▼
                        ┌──────────────┐
                        │  duxx-core   │  Error / Schema / Value
                        └──────────────┘

                        ┌──────────────┐
                        │ duxx-coldtier│  ──→  Parquet (Spark / DuckDB / pandas)
                        └──────────────┘
```

---

## 8. Competitive landscape

✅ = first-class • ⚠ = partial / extension • ✗ = not supported

|                       | Vector ANN | BM25 | Structured | Embedded | Server | Reactive | Agent prims | MCP | License | Lang |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| **Redis** (post-2024) | ✗      | ✗    | ⚠         | ✗       | ✅      | ✅       | ✗          | ✗  | RSALv2/SSPL (non-OSS) | C |
| **Valkey**            | ⚠ (module) | ⚠ (module) | ⚠ | ✗       | ✅      | ✅       | ✗          | ✗  | BSD-3 | C |
| **DiceDB**            | ✗      | ✗    | ⚠         | ✗       | ✅      | ✅       | ✗          | ✗  | AGPL | Go |
| **Qdrant**            | ✅      | ⚠    | ⚠         | ✗       | ✅      | ⚠       | ✗          | ✗  | Apache 2 | Rust |
| **Pinecone**          | ✅      | ✗    | ⚠         | ✗       | ✅      | ✗       | ✗          | ✗  | Proprietary | (closed) |
| **Milvus**            | ✅      | ⚠    | ✅         | ✗       | ✅      | ✗       | ✗          | ✗  | Apache 2 | Go / C++ |
| **Weaviate**          | ✅      | ✅    | ✅         | ✗       | ✅      | ✗       | ✗          | ✗  | BSD-3 | Go |
| **pgvector** (Postgres) | ✅    | ⚠    | ✅         | ✗       | ✅      | ⚠       | ✗          | ✗  | PostgreSQL | C |
| **LanceDB**           | ✅      | ⚠    | ✅         | ✅       | ⚠       | ✗       | ✗          | ✗  | Apache 2 | Rust |
| **ChromaDB**          | ✅      | ✗    | ⚠         | ✅       | ✅      | ✗       | ⚠          | ✗  | Apache 2 | Python |
| **DuckDB**            | ⚠ (ext) | ⚠    | ✅         | ✅       | ✗      | ✗       | ✗          | ✗  | MIT | C++ |
| **DuxxDB**            | ✅      | ✅    | ✅         | ✅       | ✅      | ✅       | ✅          | ✅  | Apache 2 | Rust |

### When each one is the right choice

- **Pure KV with reactive (sessions only):** **Valkey** is the
  default OSS answer (BSD-3, Linux Foundation, AWS/Google/Snap
  backed). DiceDB if you want reactive-first design. We don't try to
  replace either at the raw KV job; we coexist (see below).
- **Pure vector search at scale:** Qdrant if you're already in Rust,
  Pinecone if you want hosted, Milvus if you need distributed.
- **You're already on Postgres:** pgvector. Don't add another store.
- **Embedded analytics + occasional vector:** LanceDB or DuckDB +
  vector extension.
- **Pre-built RAG with batteries:** ChromaDB, LlamaIndex+Postgres.
- **Building an agent and starting fresh:** **DuxxDB.** Single dep,
  hybrid retrieval, no glue, agent primitives, six integration paths.

### Coexistence with Valkey / Redis

DuxxDB doesn't replace Valkey at the raw KV job. The clean
architecture:

```
┌──────────────────────────────────┐
│ Valkey / Redis (RESP3)           │  ← session blobs, rate limits,
│ ~50 µs reads, distributed        │     pubsub, dedup keys
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ DuxxDB                            │  ← agent memory, tool cache,
│ ~200 µs hybrid recall             │     hybrid recall, MCP, gRPC
└──────────────────────────────────┘
```

**Wire-protocol bonus:** [`duxx-server`](../crates/duxx-server) speaks
RESP2/3 — `valkey-cli`, `redis-cli`, and any RESP client speak DuxxDB
out of the box. Custom commands (`REMEMBER`, `RECALL`) live alongside
standard `PING` / `SET` / `GET` so existing Redis muscle-memory
transfers.

---

## 9. Install & quickstart

The full guides:

- **[INSTALLATION.md](INSTALLATION.md)** — per-platform install,
  every surface, troubleshooting.
- **[USER_GUIDE.md](USER_GUIDE.md)** — quickstart per integration
  surface, common workflows, configuration.

The shortest path:

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
docker build -t duxxdb:0.1.0 .
docker run --rm -p 6379:6379 \
  -e DUXX_STORAGE=redb:/data/duxx.redb \
  -v "$PWD/data:/data" \
  duxxdb:0.1.0

# In another shell:
redis-cli -p 6379
> REMEMBER alice "I lost my wallet at the cafe"
(integer) 1
> RECALL alice "wallet" 3
1) 1) (integer) 1
   2) "0.032787"
   3) "I lost my wallet at the cafe"
```

---

## 10. Project structure

```
duxxdb/
├── Cargo.toml                       workspace root
├── Dockerfile                       multi-stage build
├── README.md                        landing page
├── LICENSE                          Apache 2.0
├── NOTICE                           upstream attributions
├── .github/workflows/ci.yml         Linux + macOS + Windows CI
├── scripts/
│   ├── build.sh                     Windows + Git Bash cargo wrapper
│   └── build.bat                    MSVC vcvars64 wrapper
├── docs/
│   ├── INSTALLATION.md              ◄── start here for setup
│   ├── USER_GUIDE.md                ◄── start here for usage
│   ├── PROJECT_OVERVIEW.md          ◄── this file
│   ├── ARCHITECTURE.md              system design
│   ├── ROADMAP.md                   phased plan + status
│   ├── SETUP.md                     toolchain log + troubleshooting
│   ├── UAT_GUIDE.md                 Closed-UAT specific
│   └── PHASE_2_3_PLAN.md            Lance-as-alternate-Storage plan
├── crates/
│   ├── duxx-core/                   Schema, Value, Error
│   ├── duxx-storage/                Storage trait + MemoryStorage / RedbStorage
│   ├── duxx-index/                  tantivy BM25 + hnsw_rs HNSW
│   ├── duxx-query/                  RRF, hybrid recall
│   ├── duxx-memory/                 MEMORY / TOOL_CACHE / SESSION + decay
│   ├── duxx-reactive/               ChangeBus pub/sub
│   ├── duxx-embed/                  hash + OpenAI + Cohere embedders
│   ├── duxx-server/                 RESP2/3 daemon + glob (PSUBSCRIBE)
│   ├── duxx-mcp/                    MCP stdio JSON-RPC server
│   ├── duxx-grpc/                   tonic gRPC daemon (streaming Subscribe)
│   ├── duxx-coldtier/               Apache Parquet exporter
│   ├── duxx-cli/                    `duxx` shell + chatbot_memory example
│   └── duxx-bench/                  Criterion benchmarks
├── bindings/
│   ├── python/                      PyO3 + maturin abi3 wheel
│   └── node/                        napi-rs v2 .node module (workspace-excluded)
└── bench/comparative/               DuxxDB vs LanceDB / Redis / Qdrant / pgvector
```

---

## 11. Roadmap

Full version: [ROADMAP.md](ROADMAP.md). Snapshot:

| Phase | Title | State |
|---|---|---|
| 0 | Foundation (workspace, license, docs) | ✅ |
| 1 | Embedded core (KV, vector, text, hybrid) | ✅ |
| 2.1–2.2 | Production tantivy + hnsw_rs | ✅ |
| 2.3 | Durable storage (redb) | ✅ |
| 2.3.5 | Index persistence (tantivy disk + HNSW dump) | ✅ |
| 2.4 | Cross-restart importance decay | ✅ |
| 2.5 | Batched commits + benchmarks | ✅ |
| 2.6 | `TOOL_CACHE` + `SESSION` + decay | ✅ |
| 3.1 | MCP stdio server | ✅ |
| 3.2 | RESP2/3 TCP server | ✅ |
| 3.3 | Python bindings (PyO3 + maturin abi3) | ✅ |
| 3.4 | Node / TypeScript bindings (napi-rs) | ✅ |
| 3.5 | gRPC daemon (tonic, streaming Subscribe) | ✅ |
| 4 | Reactive subscriptions (RESP) | ✅ |
| 4.5 | `PSUBSCRIBE` patterns + per-key channels | ✅ |
| 4.6 | Comparative bench (embedded vs LanceDB) | ✅ |
| 4.7 | Network comparative bench (vs Redis / Qdrant / pgvector) | ✅ wired; need Docker for full numbers |
| 5 | Cold-tier export (Apache Parquet) | ✅ |
| **6** | **Distributed mode, RBAC, observability** | **Future** |
| Lance | As alternate `Storage` impl | Designed; not blocking — redb covers durability |

---

## 12. Limitations & caveats

We're **Closed UAT** (v0.1). Honest list of what's still missing:

1. **No auth on the wire.** `duxx-server` and `duxx-grpc` accept any
   connection. Bind to `127.0.0.1` only; reverse-proxy with auth
   (nginx + basic auth, mTLS, or a service mesh) for now. RBAC + mTLS
   land in Phase 6.
2. **No multi-tenant isolation** in a single process. Run one
   `--storage dir:./tenant-X` daemon per tenant for now.
3. **No SIMD-tuned distance kernels yet.** `hnsw_rs` has decent SIMD
   internally; future work may swap to AVX-512 hand-tuned routines.
4. **Single node only.** No sharding / replication. Phase 6.
5. **No schema migrations.** Tables are immutable in shape; the row
   store is just bytes-keyed today.
6. **Public API may shift** before v1.0. Pin a git SHA, not `master`,
   if stability matters. The `duxx-memory` surface has been stable
   since Phase 2.6.
7. **Comparative bench has 3 wired-but-unrun targets** (Redis Stack,
   Qdrant, pgvector). Numbers fill in once Docker daemon is up on a
   bench host.

For Closed UAT use cases (single-tenant agent prototypes, internal
demos, PoCs), this is enough. Open UAT to public traffic waits on
Phase 6.

---

## 13. Contributing

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md).
2. Pick an unchecked box from the roadmap (Phase 6 hardening, Lance
   `Storage` impl, comparative bench follow-ups, etc.).
3. Open a PR titled `<crate>: <what>` (e.g. `duxx-storage: Lance
   backend feature flag`).
4. Every PR must:
   - Pass CI (Linux + macOS + Windows matrix on `cargo test`).
   - Add or update at least one test.
   - Not regress any benchmark by > 5 %.
5. We follow Conventional Commits in commit messages.

---

## 14. License & acknowledgements

**License:** Apache 2.0. See [LICENSE](../LICENSE) and
[NOTICE](../NOTICE).

**Standing on the shoulders of:**

- [tantivy](https://github.com/quickwit-oss/tantivy) — BM25 full-text
- [hnsw_rs](https://crates.io/crates/hnsw_rs) — HNSW vectors
- [redb](https://github.com/cberner/redb) — embedded ACID KV
- [tonic](https://github.com/hyperium/tonic) + [prost](https://github.com/tokio-rs/prost) — gRPC + protobuf
- [PyO3](https://github.com/PyO3/pyo3) + [maturin](https://github.com/PyO3/maturin) — Python bindings
- [napi-rs](https://github.com/napi-rs/napi-rs) — Node bindings
- [Apache Arrow](https://github.com/apache/arrow-rs) + [parquet](https://github.com/apache/arrow-rs) — cold-tier
- [reqwest](https://github.com/seanmonstar/reqwest) + [rustls](https://github.com/rustls/rustls) — HTTP embedders
- [DuckDB](https://duckdb.org/) — embedded-first philosophy
- [LanceDB](https://github.com/lancedb/lance) — comparative bench peer
- [Model Context Protocol](https://modelcontextprotocol.io/) — agent integration standard
- [criterion.rs](https://github.com/bheisler/criterion.rs) — benchmark harness

This project would not exist without the OSS work above. If DuxxDB is
useful to you, please support its upstream dependencies first.

---

*Last updated: Phase 5 shipped, CI green. See `git log` for changes.*
