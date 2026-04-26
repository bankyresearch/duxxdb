# DuxxDB — Architecture

**Version:** 0.1-draft
**Status:** Design locked, implementation starting
**Audience:** Contributors, integrators, system designers

---

## 1. Mission

Build the **fastest database purpose-built for AI agents** — chatbots, voice bots, and autonomous agents — unifying structured data, unstructured text, and vector embeddings behind a single low-latency API.

**Target latency budgets (p99):**

| Operation | Budget | Why |
|---|---|---|
| Session KV read | < 0.5 ms | Voice bots have a ~200 ms end-to-end budget |
| Hybrid RECALL (vector + BM25) | < 10 ms | RAG must feel instant |
| Structured filter + ANN | < 15 ms | "Find my top-10 refund emails this week" |
| Write + index | < 5 ms | Agents write memory on every turn |

---

## 2. Design Principles

1. **Agent-native, not agent-friendly.** First-class primitives for MEMORY, TOOL_CACHE, SESSION — not generic tables that agents bolt onto.
2. **Embedded-first, server-optional.** Same binary runs in-process (like SQLite / DuckDB) or as a daemon. Zero network hop when you don't need one.
3. **Compose, don't reinvent.** Storage, vector index, BM25, SQL — all from battle-tested Rust crates. The innovation is in the **agent layer and the hybrid query planner**, not in yet another LSM tree.
4. **One table, mixed columns.** Structured, unstructured, and vector columns coexist in a single row. One insert, one query plan.
5. **Realtime reactive.** `SUBSCRIBE` for push updates — agents shouldn't poll for new memories.
6. **MCP-native.** The Model Context Protocol is the default wire format. Any Claude/GPT/Llama agent plugs in with zero glue code.

---

## 3. System Tiering

DuxxDB is the **hot tier** only. It is explicitly **not** a lakehouse.

```
┌─────────────────────────────────────────────────────────────┐
│   AGENT (chatbot / voice bot / MCP client)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │  <5 ms RECALL / WRITE
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   HOT TIER — DuxxDB                                         │
│   Last 30–90 days, live sessions, tool caches, hot vectors  │
└─────────────────────────┬───────────────────────────────────┘
                          │  async CDC / batch export (minutes)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│   COLD TIER — Lakehouse (Iceberg / Delta on S3)             │
│   Full history, training corpora, audit logs, analytics     │
└─────────────────────────────────────────────────────────────┘
```

Cold-tier integration is a **v0.3 feature** (Arrow Flight → Iceberg/Delta exporter). Out of scope for v0.1.

---

## 4. Component Architecture

```
                    ┌──────────────────────────────────────┐
                    │         AGENT BINDINGS               │
                    │   Rust | Python | TypeScript | Go    │
                    └──────────────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐             ┌───────────────┐             ┌───────────────┐
│  duxx-server  │             │   duxx-mcp    │             │   duxx-cli    │
│  gRPC + RESP3 │             │  MCP protocol │             │  duxx shell   │
└───────┬───────┘             └───────┬───────┘             └───────┬───────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      ▼
                    ┌──────────────────────────────────────┐
                    │         duxx-memory                  │
                    │  MEMORY | TOOL_CACHE | SESSION types │
                    │  Auto-embed, TTL, importance decay   │
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         duxx-query                   │
                    │  DataFusion SQL + hybrid fusion (RRF)│
                    │  Cost-based planner, filter pushdown │
                    └──────────────────┬───────────────────┘
                                       ▼
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐             ┌───────────────┐             ┌───────────────┐
│  duxx-index   │             │ duxx-reactive │             │  duxx-core    │
│  usearch HNSW │             │  subscriptions│             │  types/errors │
│  tantivy BM25 │             │  change feed  │             │  config       │
└───────┬───────┘             └───────────────┘             └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    duxx-storage                           │
│  Lance columnar format (Arrow-native, vector-aware)       │
│  redb for session/KV hot path                             │
└───────────────────────────────────────────────────────────┘
```

### 4.1 Crate responsibilities

| Crate | Role |
|---|---|
| `duxx-core` | Shared types, errors, config, feature flags |
| `duxx-storage` | Lance table wrapper, schema, open/scan/insert/compact. redb for session KV |
| `duxx-index` | HNSW (usearch) + BM25 (tantivy) builders and searchers |
| `duxx-query` | DataFusion plan integration, hybrid fusion via Reciprocal Rank Fusion |
| `duxx-memory` | High-level agent primitives (MEMORY, TOOL_CACHE, SESSION) |
| `duxx-reactive` | Subscription manager, change feed, WAL tailer |
| `duxx-server` | gRPC + RESP3 daemon |
| `duxx-mcp` | MCP server (stdio + SSE transports) |
| `duxx-cli` | Interactive shell + admin commands |
| `duxx-bench` | Criterion benchmarks; runs against Redis/Qdrant/pgvector for comparison |

---

## 5. Data Model

### 5.1 Unified schema

A DuxxDB table can mix any of these column kinds in a single row:

| Kind | Rust type | Index | Use |
|---|---|---|---|
| Scalar | `i64`, `f64`, `bool`, `Utf8`, `Timestamp` | B-tree (optional) | Filter, sort |
| JSON / Struct | `Struct<...>` | — | Metadata |
| Text | `Utf8` with `@bm25` tag | Tantivy inverted index | Keyword search |
| Vector | `FixedSizeList<Float32, D>` with `@hnsw` tag | usearch HNSW | ANN search |
| Array | `List<T>` | — | Nested collections |

### 5.2 Example DDL

```sql
CREATE TABLE conversations (
  id         UUID PRIMARY KEY,
  user_id    TEXT,
  agent_id   TEXT,
  turn_at    TIMESTAMP,
  role       TEXT,                            -- 'user' | 'assistant' | 'tool'
  message    TEXT @bm25,                      -- BM25 index
  embedding  VECTOR(1024) @hnsw(m=16, ef=64), -- HNSW index
  tool_calls JSON,
  importance FLOAT DEFAULT 1.0,
  ttl        INTERVAL DEFAULT '90 days'
);
```

### 5.3 Agent primitive: MEMORY

`MEMORY` is a built-in managed type. On insert:
1. Auto-embed `message` with the configured provider (OpenAI / Cohere / local BGE).
2. Set `turn_at = now()`.
3. Compute `importance` from caller + optional LLM scoring.
4. Start TTL countdown with exponential decay on `importance`.

### 5.4 Agent primitive: TOOL_CACHE

Caches tool-call results keyed by `(tool_name, args_hash)` **and** args embedding.
Lookup returns a hit if:
- exact `args_hash` match, **or**
- cosine similarity of args embedding ≥ 0.95 **and** result hasn't expired.

Saves real money on expensive tool calls (web search, LLM summarization, DB queries).

### 5.5 Agent primitive: SESSION

Hot KV under redb, keyed by `session_id`. Holds conversation buffer, open tool invocations, flags. Auto-flushed to main table on session end.

---

## 6. Query Flow — the RECALL operation

```
RECALL "user's refund issue last month"
  WHERE user_id = '42'
  USING vector_col = embedding, text_col = message
  LIMIT 10
```

**Pipeline:**

1. **Parse & plan** (`duxx-query`)
   - Embed the query string → `q_vec` (same provider as inserted vectors).
   - Tokenize → `q_terms` for BM25.
2. **Filter pushdown** (DataFusion) — apply `user_id = '42'` first. Narrows candidate set.
3. **Parallel index probes** (tokio, join!):
   - HNSW search on `embedding` → top-50 by cosine, filtered by candidate set.
   - BM25 search on `message` → top-50 by score, filtered.
4. **Fusion via Reciprocal Rank Fusion (RRF)**:
   ```
   score(doc) = Σ  1 / (k + rank_i(doc))   for k ≈ 60
   ```
   Tunable weight per index; recency boost optional.
5. **Hydrate** — fetch full rows for top-N from Lance (columnar projection).
6. **Return** as Arrow RecordBatch (zero-copy to Python/JS bindings).

**Expected p99:** < 10 ms on 1M rows, single node, NVMe.

---

## 7. Storage Layer Details

### 7.1 Why Lance

- Columnar (Parquet-like) but **random-access friendly** — critical for ANN hydration.
- Vector indices are **first-class** and stored in the same file.
- Arrow-native → zero-copy to every language binding.
- Versioning built in — we get time-travel for free.
- Apache 2.0.

### 7.2 Why redb for hot KV

- Pure Rust, no FFI.
- mmap-backed, MVCC, ACID.
- Sub-microsecond reads.
- Used for `SESSION`, the reactive change-feed WAL, and small metadata.

### 7.3 Directory layout on disk

```
<data_dir>/
├── duxx.config            # TOML config
├── sessions.redb          # redb file — session/KV hot path
├── wal/                   # reactive change feed
│   └── 000001.log
└── tables/
    └── conversations/
        ├── _versions/     # Lance dataset versions
        ├── _indices/      # HNSW, BM25 index files
        └── data/          # Arrow/Lance fragments
```

---

## 8. Concurrency Model

- **Tokio** async runtime; one reactor per process.
- **Read path:** shared `Arc<LanceDataset>` + `Arc<IndexSet>`, lock-free.
- **Write path:** single writer per table (MVCC via Lance snapshots). Appends are O(1); index updates amortized.
- **Compaction + reindex** runs on a background tokio task, triggered by write-volume threshold.

---

## 9. Wire Protocols

| Protocol | When | Why |
|---|---|---|
| **Embedded (in-process)** | Rust agent code | Zero overhead |
| **gRPC + protobuf** | Services, multi-lang | Typed, streaming, HTTP/2 |
| **RESP3** | Existing Redis/DiceDB clients | Drop-in compat path |
| **MCP (stdio + SSE)** | Claude / GPT / any LLM agent | Zero-glue integration |

---

## 10. Reactive Subscriptions

```
SUBSCRIBE conversations WHERE agent_id = 'sales-bot-01'
```

- WAL-tailing model (Postgres logical replication inspiration).
- Subscribers receive Arrow RecordBatches on change.
- Backpressure via bounded `tokio::mpsc`.
- Filter predicates evaluated at publish-time to avoid fan-out explosion.

Use case: a supervisor agent watching every memory created by a fleet of worker agents.

---

## 11. Security & Multi-tenancy (post v0.1)

- Row-level isolation via `tenant_id` column + DataFusion filter injection.
- mTLS between server and clients.
- At-rest encryption via Lance's optional AES-GCM.
- Unity-Catalog-compatible ACL layer in v0.3.

---

## 12. Non-Goals (v0.1)

- Distributed / sharded cluster mode — single node only. Distribution comes after we saturate single-node perf.
- OLAP dashboards — that's the cold tier's job.
- Full SQL compliance — DataFusion covers what we need; we don't add custom SQL surface in v0.1.
- Write-optimized graph traversal — future.

---

## 13. Open Questions (tracked for resolution)

1. **Default embedding provider** — bundle a local BGE model or require explicit config?
2. **Vector quantization default** — full float32 at 1M scale, switch to PQ at ≥ 10M?
3. **Compaction trigger** — write count or wall clock?
4. **RRF `k` tuning** — expose or hardcode at 60?

These are implementation-time decisions and don't block scaffolding.

---

## 14. References & Credits

Standing on the shoulders of:

- [Lance](https://github.com/lancedb/lance) — columnar + vector storage
- [Apache DataFusion](https://github.com/apache/datafusion) — SQL engine
- [usearch](https://github.com/unum-cloud/usearch) — HNSW
- [tantivy](https://github.com/quickwit-oss/tantivy) — BM25
- [redb](https://github.com/cberner/redb) — embedded KV
- [DiceDB](https://github.com/DiceDB/dice) — reactive subscription model inspiration
- [DuckDB](https://duckdb.org/) — embedded-first philosophy
- [Model Context Protocol](https://modelcontextprotocol.io/) — agent integration standard
