# DuxxDB — Project Overview

> The database built for AI agents. Embedded + server, hybrid (vector +
> BM25 + structured), durable, with reactive subscriptions, Apache
> Parquet cold-tier, and prod-grade hardening (auth, native TLS, health,
> Prometheus metrics, graceful shutdown, importance-based eviction).
> **Public-ready — feature-complete through Phase 6.2.**

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
14. [License](#14-license)

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

DuxxDB is built **for agents specifically**, not retrofitted from a
general-purpose store. That shapes every design choice:

| Capability | What it gives you |
|---|---|
| Hybrid (vector + BM25 + structured) in **one query plan** with **RRF** | One recall call returns semantic + keyword + filtered results, fused. No second roundtrip, no client-side merging. |
| **Embedded *and* server** from one Rust codebase | Same engine, same query plan in-process or behind RESP / gRPC / MCP. Move from notebook to production without changing schema. |
| Agent primitives (`MEMORY`, `TOOL_CACHE`, `SESSION`) as **first-class types** | The data model knows what an agent stores. You do not assemble these out of a generic KV + a generic vector index. |
| `TOOL_CACHE` with **semantic-near-hit** lookup (cosine ≥ 0.95 on the args embedding) | Skip the expensive tool call when the new args mean the same thing as a cached call. Real dollars saved on every near-duplicate. |
| **MCP-native** wire protocol (stdio JSON-RPC 2.0) | Any MCP-aware client — Claude Desktop, Cline, Cursor — plugs in with zero adapter code. |
| **Reactive subscriptions** on hybrid memory (glob patterns + per-key channels) | Push semantics on agent state, not just on plain keys. Build dashboards and live-updating crews against memory changes. |
| **gRPC streaming Subscribe** for typed cross-language consumers | Strongly typed change feed without writing a protocol adapter. |
| **Importance decay** with cross-restart Unix-epoch timestamps | Forgetting is part of the storage model, not application logic. |
| Pure Rust core — no GC pauses, no FFI bottlenecks | p99 latency stays flat under load. |
| Apache 2.0, no proprietary "open core" tier | Production features (auth, TLS, metrics, gRPC health) ship in the same binary. |
| **Apache Parquet cold-tier export** | Move old memories to object storage with one command; query them later with DuckDB / Spark. |
| **Auth + Prometheus /metrics + gRPC health + graceful shutdown** | Operationally complete on day one — not a Phase 4 chore. |

**The single biggest differentiator:** you don't *compose* DuxxDB out
of three databases. It ships agent-native primitives, six wire
surfaces, and production hardening as a single binary you `cargo
install` or `docker pull`.

---

## 5. Capabilities today

**Phase 7 complete.** All of Phase 0 → 6.2 + 7.1 through 7.6 ships.
**248 tests passing** workspace-wide; CI matrix on Linux + macOS +
Windows.

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

### Production hardening (Phase 6.1 + 6.2)

- **Auth.** `--token TOKEN` / `DUXX_TOKEN` on `duxx-server` (RESP) and
  `duxx-grpc`; constant-time compare; RESP `NOAUTH` / `WRONGPASS` flow,
  gRPC `x-duxx-token` metadata via tonic Interceptor.
- **RESP RBAC + audit.** `DUXX_AUTH_KEYS` adds read/write/admin API
  keys with optional tenant scopes for tenant-safe core commands.
  `DUXX_AUDIT_LOG` writes JSON-lines security events.
- **Native TLS** (Phase 6.2). `--tls-cert PATH --tls-key PATH` /
  `DUXX_TLS_CERT` / `DUXX_TLS_KEY` on both daemons. Pure-Rust rustls;
  no OpenSSL dependency. RESP via tokio-rustls in the accept loop;
  gRPC via tonic's `tls` feature. `redis-cli --tls`, `grpcurl --tls`,
  any rustls / OpenSSL client connect directly.
- **mTLS client verification.** Add `--tls-client-ca PATH` /
  `DUXX_TLS_CLIENT_CA` to require client certificates on RESP or gRPC.
- **Health.** gRPC standard `grpc.health.v1.Health` via [tonic-health];
  RESP exposes `/health` on the metrics listener.
- **Prometheus metrics.** `--metrics-addr HOST:PORT` /
  `DUXX_METRICS_ADDR` binds a separate hyper listener serving
  `/metrics` (counters / gauges / histograms) + `/health`.
- **Graceful shutdown.** Ctrl+C / SIGTERM stops accepting and drains
  in-flight connections up to `--drain-secs N` (default 30) before
  triggering tantivy commit + HNSW dump (so `dir:` reopens fast).
- **Memory cap + eviction** (Phase 6.2). `--max-memories N` /
  `DUXX_MAX_MEMORIES` enforces a soft row cap. On overflow, the
  lowest *effective* (decayed) importance row is evicted first —
  agent-friendly forgetting, not naive LRU.
- **Backup.** Offline `duxx-snapshot create/verify/restore` with a
  SHA-256 manifest, plus Parquet cold-tier export, documented in
  [USER_GUIDE.md § 6](USER_GUIDE.md#6-backup--restore).

[tonic-health]: https://crates.io/crates/tonic-health

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

### Comparative bench

The comparative harness measures **retrieval quality** (recall@k / nDCG@k vs an
exact cosine ground truth), latency, and **throughput under concurrent load**
against Redis Stack / Qdrant / pgvector, with DuxxDB run **disk-backed** for a
like-for-like comparison and queried in its native hybrid mode. One command
reproduces the suite and writes a JSON report with the hardware embedded:

```bash
cd bench/comparative && ./run.sh          # ./run.sh --quick for the no-services path
```

Methodology, fairness rules, and how to read the output:
[`docs/BENCHMARKS.md`](BENCHMARKS.md). We publish no unqualified "Nx faster"
headline — every speed claim carries its workload, dimensions, and hardware.

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

## 8. Where DuxxDB fits

DuxxDB is purpose-built for **the storage tier of an AI agent** —
the place that holds memories, tool-call results, sessions, traces,
prompts, datasets, eval runs, replay sessions, and a cost ledger,
and answers hybrid (vector + BM25 + structured) recall in
sub-millisecond p99.

**Reach for DuxxDB when the workload is agent-shaped:**

- You need **hybrid recall** (vector + keyword + filters) in one
  query plan, not three round trips.
- You want **agent primitives as first-class types** —
  `MemoryStore`, `ToolCache`, `SessionStore`, `TraceStore`,
  `PromptRegistry`, `DatasetRegistry`, `EvalRegistry`,
  `ReplayRegistry`, `CostLedger` — not generic KV + vector you
  assemble yourself.
- You want **one binary** that gives you embedded mode, RESP, gRPC,
  MCP, REST, and Parquet export — same query plan everywhere.
- You want **reactive subscriptions on hybrid memory**, not just on
  plain keys.
- You want **production hardening on day one**: auth, TLS,
  Prometheus metrics, gRPC health, graceful shutdown.

**It is fine to coexist with what you already run.** DuxxDB does not
try to be your OLTP database, your distributed cache, or your data
warehouse. Postgres still owns users / billing / SQL analytics;
Valkey or Redis still own raw KV and rate-limit counters; DuckDB or
Spark still own offline analytics over Parquet exports. DuxxDB owns
the **agent recall + agent-ops path**.

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
| **6.1** | **Prod hardening (auth, health, Prometheus, drain)** | ✅ |
| **6.2** | **Native TLS + memory cap + importance-based eviction** | ✅ |
| **7.1** | **`duxx-trace` — agent observability (Span / Trace / Thread + 6 RESP commands)** | ✅ |
| **7.2** | **`duxx-prompts` — versioned prompt registry with semantic search (9 RESP commands)** | ✅ |
| **7.3** | **`duxx-datasets` — versioned eval datasets + DATASET.FROM_RECALL (13 RESP commands)** | ✅ |
| **7.4** | **`duxx-eval` — runs, scores, regressions, semantic failure clustering (9 RESP commands)** | ✅ |
| **7.5** | **`duxx-replay` — deterministic agent replay with per-invocation overrides (12 RESP commands)** | ✅ |
| **7.6** | **`duxx-cost` — token + cost ledger, budgets, semantic clustering of spend (10 RESP commands)** | ✅ |
| 7.4 | `duxx-eval` — eval runs + scorers + regression detection | Planned |
| 7.5 | `duxx-replay` — deterministic agent replay | Planned |
| 7.6 | `duxx-cost` — token + cost ledger with budgets | Planned |
| 6.3+ | distributed mode, full row-level security, OpenTelemetry, SIMD tuning | Future |
| Lance | As alternate `Storage` impl | Designed; not blocking — redb covers durability |

---

## 12. Limitations & caveats

We're **public-ready** (v0.1). Honest list of what's still missing:

1. **No full row-level security across every Phase 7 primitive.**
   RESP role API keys protect command classes, and tenant-scoped keys
   namespace memory/session/cost commands, but shared-daemon
   multi-tenant use should stay on tenant-safe commands for now.
2. **No complete multi-tenant isolation** in a single process. Run one
   `--storage dir:./tenant-X` daemon per tenant for strict isolation.
3. **Eviction reclaims rows but not index memory.** When the
   `--max-memories` cap evicts a row, the row store + duxx-memory
   row map drop it (so `recall` never returns it again), but the
   HNSW + tantivy entries stay until the next process restart.
   Index-side tombstones are Phase 6.3.
4. **No SIMD-tuned distance kernels yet.** `hnsw_rs` has decent SIMD
   internally; future work may swap to AVX-512 hand-tuned routines.
5. **Single node only.** No sharding / replication. Phase 6.3+.
6. **No schema migrations.** Tables are immutable in shape; the row
   store is just bytes-keyed today.
7. **Public API may shift** before v1.0. Pin a git SHA / tag, not
   `master`, if stability matters. The `duxx-memory` surface has been
   stable since Phase 2.6.
8. **Comparative bench has 3 wired-but-unrun targets** (Redis Stack,
   Qdrant, pgvector). Numbers fill in once Docker daemon is up on a
   bench host.

With native TLS, mTLS, auth, resource limits, and eviction, DuxxDB is
suitable for single-tenant network exposure behind an operator-managed
certificate. Multi-tenant SaaS deployment still wants Phase 6.3+
(full row-level security, tenant isolation, sharding).

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

## 14. License

**License:** Apache 2.0. See [LICENSE](../LICENSE) and
[NOTICE](../NOTICE).

Third-party dependency notices required for distribution are maintained
in [NOTICE](../NOTICE).

---

*Last updated: Phase 6.2 shipped (native TLS on RESP + gRPC, memory
cap + importance-based eviction). See `git log` for changes.*
