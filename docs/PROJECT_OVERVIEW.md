# DuxxDB — Project Overview

> The database built for AI agents. Embedded, hybrid (vector + BM25 +
> structured), with agent-native primitives. Pre-alpha; Phase 2 of 5
> complete.

---

## Table of Contents

1. [30-second pitch](#1-30-second-pitch)
2. [The problem](#2-the-problem-we-solve)
3. [What DuxxDB is](#3-what-duxxdb-is)
4. [What makes it unique](#4-what-makes-it-unique)
5. [Capabilities today](#5-capabilities-today)
6. [Performance — measured](#6-performance--measured)
7. [Architecture at a glance](#7-architecture-at-a-glance)
8. [Competitive landscape](#8-competitive-landscape)
9. [Installation & setup](#9-installation--setup)
10. [Quick start](#10-quick-start)
11. [Project structure](#11-project-structure)
12. [Roadmap](#12-roadmap)
13. [Limitations & caveats](#13-limitations--caveats)
14. [Contributing](#14-contributing)
15. [License & acknowledgements](#15-license--acknowledgements)

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
Rust, no GC pauses, embedded or server.

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

### The latency budgets that matter

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

A pure-Rust, embedded-first hybrid database that natively understands
**agent memory**, fuses **vector + full-text + structured** retrieval
in a single query plan, and runs in-process or as a server from the
same codebase.

### Two modes

```
EMBEDDED MODE                          SERVER MODE
                                       (Phase 3 — planned)
┌──────────────────┐                   ┌──────────────────┐
│   your agent     │                   │   your agent     │
│  ┌────────────┐  │                   │ ┌──────────────┐ │
│  │  duxxdb    │  │                   │ │  duxxdb      │ │
│  │  library   │  │                   │ │  client      │ │
│  └────────────┘  │                   │ └──────┬───────┘ │
└──────────────────┘                   └────────┼─────────┘
   in-process                             gRPC / RESP3
   zero RTT                              │ MCP / SSE
   <1ms recall                            ▼
                                  ┌──────────────────┐
                                  │  duxx-server     │
                                  │  daemon          │
                                  └──────────────────┘
```

Same `duxx-core` and storage. Embedded ships today; server is Phase 3.

### Three pillars

1. **Hybrid retrieval.** Vector ANN (HNSW) + BM25 full-text + structured
   filters fused via Reciprocal Rank Fusion (RRF) in one round-trip.
2. **Agent-native primitives.** Not a generic table store with
   "examples for chatbots" — actual first-class `MEMORY`, `TOOL_CACHE`,
   `SESSION` types with semantic-near-hit and importance decay.
3. **MCP-native.** First-class Model Context Protocol support so any
   Claude / GPT / open-source agent plugs in with zero glue code.
   (Phase 3.)

---

## 4. What makes it unique

The market has lots of vector DBs, lots of KV stores, lots of
SQL engines. The thing nobody has put together:

| Differentiator | Status | Who else has it? |
|---|---|---|
| Hybrid (vector + BM25 + structured) in **one query plan** with **RRF** | ✅ today | Weaviate has hybrid but not RRF; pgvector has both but no RRF planner |
| Embedded **and** server from one Rust codebase | embedded ✅, server in Phase 3 | LanceDB has embedded; Qdrant is server-only |
| Agent primitives (`MEMORY`, `TOOL_CACHE`, `SESSION`) as **first-class types** | `MEMORY` ✅; others Phase 2.6 | Nobody — agent frameworks (LangChain, LlamaIndex) bolt this on top of generic stores |
| `TOOL_CACHE` with **semantic-near-hit** lookup (cosine ≥ 0.95 of args embedding) | Phase 2.6 | Nobody — saves real money on expensive tool calls |
| **MCP-native** wire protocol | Phase 3 | Nobody (MCP is new — late 2024) |
| **Reactive subscriptions** (`SUBSCRIBE memory WHERE …`) tied to vector + structured filters | Phase 4 | DiceDB has reactive on KV; nobody has reactive on hybrid |
| Pure Rust core — no GC pauses, no FFI bottlenecks | ✅ | Qdrant ✅, LanceDB ✅; Pinecone/Weaviate/Milvus all have GC |
| Apache 2.0, no proprietary "open core" tax | ✅ | Pinecone is proprietary; Milvus has Zilliz cloud lock-in pressure |

**The single biggest differentiator:** every other tool forces you to
*choose* between vector / KV / SQL, or to *compose* multiple stores.
DuxxDB collapses that decision.

---

## 5. Capabilities today

What ships in Phase 2 (current state, all green, all tested):

### ✅ Done

- **Workspace** — 10 Rust crates, Apache 2.0, all `cargo test` green (21/21)
- **`duxx-core`** — `Schema`, `Column`, `ColumnKind` (`I64`, `F64`,
  `Bool`, `Text { bm25 }`, `Vector(VectorSpec)`, `Timestamp`, `Json`),
  `Value`, `Error`, `Config`
- **`duxx-storage`** — schema-aware in-memory `Table` (Lance-backed in
  Phase 2.3)
- **`duxx-index`**:
  - `TextIndex` — production [tantivy] BM25, batched commits,
    auto-flush on read
  - `VectorIndex` — production [hnsw_rs] HNSW (cosine), `with_capacity`
    for memory tuning
- **`duxx-query`** — Reciprocal Rank Fusion (`rrf_fuse`),
  `hybrid_recall(vector_index, text_index, query_vec, query_text, k)`
- **`duxx-memory`** — `MemoryStore` with `remember()` / `recall()`
- **`duxx-reactive`** — `ChangeBus` over `tokio::sync::broadcast`
  (publish/subscribe scaffolding)
- **`duxx-bench`** — Criterion benchmarks: hybrid recall + insert + bulk
- **Working example** — `chatbot_memory.rs` shows insert + recall on
  a 7-doc corpus, top result correct
- **Cross-platform build** — Windows + Git Bash + WinLibs MinGW path
  documented end-to-end (the journey is in
  [SETUP.md](SETUP.md))

### ✅ Phase 2.6 — Agent primitives (added after the comparison matrix)

- **`ToolCache`** — exact-hash and semantic-near-hit (cosine ≥ 0.95)
  lookup, per-entry TTL, purge.
- **`SessionStore`** — sliding-TTL KV. Reads bump last-access; lazy
  eviction.
- **Importance decay** — `Memory::effective_importance(half_life)` and
  `MemoryStore::recall_decayed()` reranks recall hits by exponentially
  decayed importance.

### ✅ Phase 3.1 — MCP stdio server

[`duxx-mcp`](../crates/duxx-mcp) ships a working JSON-RPC 2.0 server
over stdio. Build the binary, point any MCP agent (Claude Desktop,
Cline, …) at it, and you get three tools: `remember`, `recall`, `stats`.

```jsonc
// Claude Desktop / Cline config
{
  "mcpServers": {
    "duxxdb": {
      "command": "/path/to/duxx-mcp",
      "args": []
    }
  }
}
```

### ✅ Phase 3.2 — RESP2/3 TCP server

[`duxx-server`](../crates/duxx-server) ships a `valkey-cli` /
`redis-cli` compatible TCP server. Standard commands (`PING`,
`HELLO`, `COMMAND`, `INFO`, `SET`, `GET`, `DEL`, `QUIT`) plus DuxxDB
extensions (`REMEMBER`, `RECALL`, `SUBSCRIBE`, `UNSUBSCRIBE`).

```bash
duxx-server --addr 127.0.0.1:6379
redis-cli -p 6379
> REMEMBER alice "I lost my wallet"
(integer) 1
> RECALL alice "wallet" 2
1) 1) (integer) 1
   2) "0.032787"
   3) "I lost my wallet"
```

### ✅ Phase 4 (in-process) — Reactive subscriptions

`MemoryStore::remember()` publishes a `ChangeEvent` to a
`tokio::sync::broadcast` bus. RESP clients can `SUBSCRIBE memory` and
receive Redis-format push messages whenever any other connection
writes a memory.

```bash
# terminal 1
redis-cli -p 6379
> SUBSCRIBE memory
1) "subscribe"
2) "memory"
3) (integer) 1
# now waiting for events…

# terminal 2
redis-cli -p 6379
> REMEMBER alice "fresh memory"
(integer) 7

# terminal 1 immediately receives:
1) "message"
2) "memory"
3) "{\"table\":\"memory\",\"row_id\":7,\"kind\":\"insert\"}"
```

### ✅ Phase 3.3 — Python bindings

[`bindings/python`](../bindings/python) ships a PyO3 + maturin
extension module compiled against the stable Python ABI
(`abi3-py38`). One wheel works on **Python 3.8 through 3.13**.

```bash
cd bindings/python
maturin build --release
pip install ../../target/wheels/duxxdb-0.1.0-cp38-abi3-*.whl
```

```python
import duxxdb
store = duxxdb.MemoryStore(dim=4)
store.remember(key="alice", text="hello", embedding=[1.0, 0.0, 0.0, 0.0])
hits = store.recall(key="alice", query="hello",
                    embedding=[1.0, 0.0, 0.0, 0.0], k=5)
print(hits[0])
# <MemoryHit id=1 score=0.0328 text="hello">
```

Also available: `duxxdb.ToolCache` (semantic-near-hit) and
`duxxdb.SessionStore` (sliding-TTL KV).

### ✅ Phase 3.4 — Node.js / TypeScript bindings

[`bindings/node`](../bindings/node) ships napi-rs v2 wrappers exposing
the same public surface as the Python binding (`MemoryStore`,
`ToolCache`, `SessionStore`). Builds on Linux / macOS / Windows-MSVC
via `npm run build`; Bun loads the resulting `.node` module too.

```ts
import { MemoryStore } from "duxxdb";
const store = new MemoryStore(4);
store.remember("alice", "hello", [1, 0, 0, 0]);
```

### ✅ Phase 4.5 — Pattern subscribe + per-key channels

Subscribers can now express "watch only Alice's memories" server-side:

```bash
redis-cli -p 6379
> PSUBSCRIBE memory.alice*
1) "psubscribe"
2) "memory.alice*"
3) (integer) 1
# any other connection's REMEMBER alice "..." now pushes here
```

Glob patterns: `*` matches any sequence, `?` matches one char,
`\\X` is a literal escape.

### 🚧 Still to do

| Phase | Component | Status |
|---|---|---|
| 2.3 | Lance-backed `Table` (durable storage) | Plan written ([PHASE_2_3_PLAN.md](PHASE_2_3_PLAN.md)); next session |
| 3.5 | gRPC daemon for typed cross-language streaming | Designed |
| 4.6 | Comparative bench vs Redis / Qdrant / pgvector / LanceDB | Designed |
| 5 | Lakehouse cold-tier export (Iceberg / Delta) | Designed |
| 6 | Distributed mode, RBAC, observability | Future |

[tantivy]: https://github.com/quickwit-oss/tantivy
[hnsw_rs]: https://crates.io/crates/hnsw_rs

---

## 6. Performance — measured

Median of 20 samples, criterion 0.5, single thread, debug release
build, 100 k vector capacity, toy embedder (32-dim, hash-bucket
tokens). See [`crates/duxx-bench/benches/recall.rs`](../crates/duxx-bench/benches/recall.rs).

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

### Reproduce on your machine

```bash
scripts/build.sh bench -p duxx-bench
# HTML report: target/criterion/report/index.html
```

---

## 7. Architecture at a glance

```
              AGENT (chatbot / voice bot / autonomous)
                              │
                              ▼
                  ┌─────────────────────┐
                  │   duxxdb (façade)   │
                  └──────────┬──────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ duxx-memory  │    │ duxx-server  │    │  duxx-mcp    │
│ MEMORY       │    │ gRPC + RESP3 │    │ MCP          │
│ TOOL_CACHE   │    │  (Phase 3)   │    │ stdio + SSE  │
│ SESSION      │    │              │    │  (Phase 3)   │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐
│ duxx-query   │ ◄── RRF fusion, hybrid recall
└──────┬───────┘
       │
       ├───────────┐
       ▼           ▼
┌──────────────┐ ┌──────────────┐
│  duxx-index  │ │ duxx-storage │
│              │ │              │
│ tantivy BM25 │ │ in-memory    │
│ hnsw_rs HNSW │ │ Lance (2.3)  │
└──────────────┘ └──────────────┘
       │           │
       └─────┬─────┘
             ▼
       ┌──────────────┐
       │  duxx-core   │ ◄── Error, Schema, Value, Config
       └──────────────┘

           ┌──────────────┐
           │ duxx-reactive│ ◄── ChangeBus (Phase 4)
           └──────────────┘
```

Full architecture spec: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 8. Competitive landscape

### Capability matrix

✅ = first-class • ⚠ = partial / extension • ✗ = not supported •
*\** = planned in DuxxDB roadmap

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
| **SQLite** + ext      | ⚠       | ✅    | ✅         | ✅       | ✗      | ✗       | ✗          | ✗  | Public | C |
| **DuxxDB**            | ✅      | ✅    | ⚠ → ✅\*    | ✅       | ⚠ → ✅\* | ⚠ → ✅\*  | ⚠ → ✅\*    | ✗ → ✅\* | Apache 2 | Rust |

### When each one is the right choice

- **Pure KV with reactive (sessions only):** **Valkey** is now the
  default open-source answer (BSD-3, Linux Foundation, AWS/Google/Snap
  backed) — Redis 7.2.4 fork after Redis Labs went non-OSS in 2024.
  DiceDB if you want reactive-first design. We don't try to replace
  any of them at the raw KV job; we coexist (see below).
- **Pure vector search at scale:** Qdrant if you're already in Rust,
  Pinecone if you want hosted, Milvus if you need distributed.
- **You're already on Postgres:** pgvector. Don't add another store.
- **Embedded analytics + occasional vector:** LanceDB or DuckDB +
  vector extension.
- **Pre-built RAG with batteries:** ChromaDB, LlamaIndex+Postgres.
- **Building an agent and starting fresh:** **DuxxDB.** Single dep,
  hybrid retrieval, no glue, agent primitives.

### Coexistence with Valkey / Redis / DiceDB

DuxxDB is **not** trying to be the new KV cache. Valkey already wins
at that job — battle-tested at hyperscale, decade of operational
experience, every cloud has managed offerings.

The clean architecture:

```
┌──────────────────────────────────┐
│ Valkey / Redis (RESP3)           │  ← session blobs, rate limits,
│ ~50 µs reads, distributed        │     pubsub, dedup keys
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ DuxxDB                            │  ← agent memory, tool cache,
│ ~200 µs hybrid recall             │     hybrid recall, MCP
└──────────────────────────────────┘
```

**Wire-protocol bonus:** [Phase 3.2](ROADMAP.md) ships a RESP3
TCP server. That means **`valkey-cli`, `redis-cli`, and any RESP
client speak DuxxDB out of the box**. Custom commands (`REMEMBER`,
`RECALL`) live alongside standard `PING` / `SET` / `GET` so existing
Redis muscle-memory transfers. Engineers don't learn a new client.

### Honest weaknesses today

We are **pre-alpha**. As of Phase 2:

- No on-disk persistence yet (Phase 2.3 — Lance integration deferred).
- No server mode yet (Phase 3) — embedded only.
- No bindings outside Rust yet (Phase 3 — Python + TS planned).
- No distributed mode (single node only — Phase 6).
- No production users; no LTS guarantees.

Use it for prototyping and to inform Phase 3+ roadmap. Don't put it in
production yet.

---

## 9. Installation & setup

### Prerequisites (all platforms)

- Git ≥ 2.30
- A C++ build chain (varies by OS — see below)
- ≥ 4 GB RAM, ≥ 4 GB free disk for build artifacts
- Rust ≥ 1.75 (we use 1.95.0 in CI)

### Windows (Git Bash + GNU toolchain — the path we live on)

This is the path verified in this repo's [SETUP.md](SETUP.md).
**No admin privileges required.**

```bash
# 1. Install Rust (rustup)
winget install Rustlang.Rustup --silent

# 2. Switch to GNU toolchain (avoids MSVC + Windows SDK requirement)
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

# 3. Install WinLibs MinGW (gcc, ld, dlltool — needed by GNU toolchain)
winget install BrechtSanders.WinLibs.POSIX.MSVCRT --silent \
  --accept-package-agreements --accept-source-agreements

# 4. Clone and build
git clone <repo-url> DuxxDB
cd DuxxDB
scripts/build.sh test --workspace
```

`scripts/build.sh` prepends the WinLibs `bin/` directory to PATH so
cargo finds the linker. Use it for every cargo invocation in Git Bash.

If you prefer MSVC instead (because you'll need it eventually for
Phase 2.3 / Lance):

```powershell
# Elevated PowerShell:
winget install Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --add Microsoft.VisualStudio.Workload.VCTools `
              --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
rustup default stable-x86_64-pc-windows-msvc
```

Then build via `scripts/build.bat` which sources `vcvars64.bat`.

### Linux (Debian / Ubuntu)

```bash
# Toolchain
sudo apt update
sudo apt install -y build-essential pkg-config

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Clone and build
git clone <repo-url> DuxxDB
cd DuxxDB
cargo test --workspace
```

No build wrapper needed on Linux — the system linker is already on PATH.

### macOS

```bash
# Toolchain (Xcode CLT)
xcode-select --install   # if not already installed

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Clone and build
git clone <repo-url> DuxxDB
cd DuxxDB
cargo test --workspace
```

### Verify the install

```bash
# Windows (Git Bash)
scripts/build.sh test --workspace
scripts/build.sh run -p duxx-cli --example chatbot_memory

# Linux / macOS
cargo test --workspace
cargo run -p duxx-cli --example chatbot_memory
```

Expected:
- `21 passed` across the workspace
- The example prints "stored 7 memories" and a 3-line top-K recall
  with **id=1** as the top hit

### Run the benchmarks

```bash
# Windows
scripts/build.sh bench -p duxx-bench

# Linux / macOS
cargo bench -p duxx-bench
```

Open `target/criterion/report/index.html` for the interactive HTML
report.

---

## 10. Quick start

### Embedded usage in Rust (today)

```rust
use duxx_memory::MemoryStore;

// 1. Open a store. 32-dim embeddings, default capacity (100k vectors).
let store = MemoryStore::new(32);

// 2. Bring your own embedder. Anything Vec<f32>-shaped works.
fn embed(text: &str) -> Vec<f32> { /* OpenAI / Cohere / local BGE */ }

// 3. Remember.
store.remember(
    "user_42",
    "I want a refund for order #9910 — it arrived broken.",
    embed("I want a refund for order #9910 — it arrived broken."),
)?;

// 4. Recall — hybrid (vector + BM25 + RRF) under the hood.
let q = "refund for broken order";
let hits = store.recall("user_42", q, &embed(q), /* k = */ 3)?;

for h in hits {
    println!("{:.4}  {}", h.score, h.memory.text);
}
```

### Tuning capacity

```rust
// For tight memory environments (benchmarks, embedded devices):
let store = MemoryStore::with_capacity(32, /* max vectors = */ 2_000);

// For production at scale:
let store = MemoryStore::with_capacity(1024, /* max vectors = */ 10_000_000);
```

### Coming in Phase 3

```python
# Python (Phase 3)
from duxxdb import Duxx
db = Duxx.open("./mydb")
db.remember("user_42", "...")
hits = db.recall("user_42", "...", k=10)
```

```typescript
// TypeScript (Phase 3)
import { Duxx } from "duxxdb";
const db = await Duxx.open("./mydb");
await db.remember("user_42", "…");
const hits = await db.recall("user_42", "…", 10);
```

```bash
# MCP — Claude / GPT plugs in (Phase 3)
duxx-mcp --data ./mydb
# Now any MCP-enabled agent can list_memories / remember / recall as tools.
```

---

## 11. Project structure

```
DuxxDB/
├── Cargo.toml                  workspace root
├── LICENSE                     Apache 2.0
├── NOTICE                      attributions for upstream OSS
├── README.md                   short intro (this doc is the long one)
├── .cargo/
│   └── config.toml             toolchain hints
├── scripts/
│   ├── build.sh                Windows + Git Bash cargo wrapper
│   └── build.bat               MSVC vcvars64 wrapper
├── docs/
│   ├── ARCHITECTURE.md         full system design
│   ├── SETUP.md                install log + troubleshooting
│   ├── ROADMAP.md              phased plan
│   ├── PROJECT_OVERVIEW.md     ← this file
│   └── PHASE_2_3_PLAN.md       Lance integration plan
└── crates/
    ├── duxx-core/              Schema, Value, Error, Config
    ├── duxx-storage/           Table (in-memory; Lance in Phase 2.3)
    ├── duxx-index/             tantivy BM25 + hnsw_rs HNSW
    ├── duxx-query/             RRF fusion, hybrid recall
    ├── duxx-memory/            MemoryStore — agent-facing API
    ├── duxx-reactive/          ChangeBus pub/sub (subscriptions: Phase 4)
    ├── duxx-server/            gRPC + RESP3 daemon (Phase 3)
    ├── duxx-mcp/               Model Context Protocol server (Phase 3)
    ├── duxx-cli/               `duxx` shell + `chatbot_memory` example
    └── duxx-bench/             Criterion benchmarks
```

---

## 12. Roadmap

Full version: [ROADMAP.md](ROADMAP.md). Snapshot:

| Phase | Title | State |
|---|---|---|
| 0 | Foundation (workspace, license, docs) | ✅ Done |
| 1 | Embedded core (KV, vector, text, hybrid) | ✅ Done |
| 2.1 | Tantivy BM25 | ✅ Done |
| 2.2 | hnsw_rs HNSW | ✅ Done |
| 2.5 | Batched commits + benchmarks | ✅ Done |
| 2.3 | Lance-backed `Table` | 📋 Plan written, next session |
| 2.6 | `TOOL_CACHE` + `SESSION` + decay | 📋 Designed |
| 3 | Server (gRPC + RESP3 + MCP) + bindings | 📋 Designed |
| 4 | Reactive subscriptions, comparative bench | 📋 Designed |
| 5 | Cold-tier (Iceberg / Delta) | 📋 Designed |
| 6 | Distribution, RBAC, observability | 🌅 Future |

---

## 13. Limitations & caveats

We're being honest about pre-alpha state:

1. **No durability.** Restart the process and your data is gone (Phase
   2.3).
2. **No transactions.** Inserts are atomic per index; cross-index
   atomicity isn't enforced.
3. **No auth, no RBAC, no rate limiting.** Don't expose `duxx-server`
   to untrusted networks.
4. **No SIMD-tuned distance kernels yet.** `hnsw_rs` has decent SIMD;
   future work may swap to AVX-512 hand-tuned routines.
5. **Single node only.** No sharding / replication.
6. **No schema migrations.** Tables are immutable in shape.
7. **API may break** between phases. Pin a git SHA, not a branch.

If you build something on it now, expect to update when Phase 3 lands.
We'll keep the public `duxx-memory` surface stable from Phase 3 onward.

---

## 14. Contributing

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md).
2. Pick an unchecked box from [ROADMAP.md](ROADMAP.md).
3. Open a PR titled `<crate>: <what>` (e.g. `duxx-storage: Lance backend
   skeleton`).
4. Every PR must:
   - Pass `cargo clippy --workspace -- -D warnings`
   - Add or update at least one test
   - Not regress any benchmark by > 5 %
5. We follow Conventional Commits in commit messages.

This is small enough to build through end-to-end. If you want to grow
with it, now is a great time.

---

## 15. License & acknowledgements

**License:** Apache 2.0. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).

**Standing on the shoulders of:**

- [tantivy](https://github.com/quickwit-oss/tantivy) — BM25 full-text
- [hnsw_rs](https://crates.io/crates/hnsw_rs) — HNSW vectors
- [Lance](https://github.com/lancedb/lance) — columnar + vector storage (Phase 2.3)
- [Apache DataFusion](https://github.com/apache/datafusion) — SQL engine (future)
- [Apache Arrow](https://github.com/apache/arrow-rs) — columnar in-memory format
- [redb](https://github.com/cberner/redb) — embedded KV (future SESSION)
- [DiceDB](https://github.com/DiceDB/dice) — reactive subscription model
- [DuckDB](https://duckdb.org/) — embedded-first philosophy
- [Model Context Protocol](https://modelcontextprotocol.io/) — agent integration standard
- [criterion.rs](https://github.com/bheisler/criterion.rs) — benchmark harness

This project would not exist without the OSS work above. If DuxxDB is
useful to you, please support its upstream dependencies first.

---

*Last updated: Phase 2.5. See `git log` for changes since then.*
