# DuxxDB — Architecture

**Version:** 0.4.0
**Status:** Shipping (self-hosted beta) — Apache-2.0 single-node engine
**Audience:** Contributors, integrators, system designers

> This document describes the **current** engine. Multi-tenancy, the control
> plane, replication/HA, and the Cloud Console live in the separate
> managed-services product (DuxxDB Cloud) — see [LICENSING.md](../LICENSING.md)
> and the "Future architecture" section below.

---

## 1. What DuxxDB is

DuxxDB is a database built for AI agents. It unifies three retrieval modes —
**vector ANN**, **BM25 full-text**, and **structured filters** — behind one
hybrid query plan, and exposes first-class **agent primitives** (`MEMORY`,
`TOOL_CACHE`, `SESSION`) plus an observability/evaluation stack (traces,
prompts, datasets, evals, replay, cost) as the operational state layer an agent
uses while it runs.

The same Rust engine runs **embedded** (in-process, like SQLite) or as a
**network server** (RESP / gRPC / MCP). Storage is durable and pure-Rust; there
is no GC and no external service dependency.

**Design latency targets (single node, NVMe):**

| Operation | Target |
|---|---|
| Session KV read | sub-millisecond |
| Hybrid recall (vector + BM25, k=10) | low single-digit ms |
| Write + index | a few ms (batched tantivy commit) |

See `bench/comparative/` for measured numbers and methodology.

---

## 2. Design principles

1. **Agent-native, not agent-friendly.** `MEMORY`, `TOOL_CACHE`, and `SESSION`
   are first-class managed types, not generic tables agents bolt onto.
2. **Embedded-first, server-optional.** One codebase; run in-process or as a
   daemon. Zero network hop when you don't need one.
3. **Compose, don't reinvent.** Durable storage, the vector index, and BM25 are
   battle-tested Rust crates. The value is the **agent layer**, the **hybrid
   query plan**, and the **Phase-7 operational stack** — not another LSM tree.
4. **Correctness under deletion.** Agent memory is deletion-heavy; recall must
   stay correct *and* stay fast as rows are forgotten (see §7, Compaction).
5. **Many surfaces, one engine.** RESP, gRPC, MCP, Python, Node, and embedded
   Rust all sit on the same core.
6. **Open-core, honestly.** The engine is Apache-2.0 and fully useful on its
   own; the commercial value is operating, scaling, and governing it.

---

## 3. System tiering

DuxxDB is the **hot operational tier**. It is not a lakehouse.

```
┌─────────────────────────────────────────────────────────────┐
│  AGENT  (chatbot / voice bot / autonomous / MCP client)      │
└────────────────────────────┬────────────────────────────────┘
                             │  low-latency recall / write
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  HOT TIER — DuxxDB (this repo)                               │
│  live memory / sessions / tool cache / hot vectors +         │
│  traces · prompts · datasets · evals · replay · cost · docs  │
└────────────────────────────┬────────────────────────────────┘
                             │  batch export (duxx-coldtier)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  COLD TIER — Apache Parquet on object storage                │
│  full history · training corpora · analytics (Spark/DuckDB)  │
└─────────────────────────────────────────────────────────────┘
```

Cold tiering is **Parquet export** via `duxx-coldtier` (`duxx-export`), read
natively by Spark / DuckDB / Polars / pandas. Offline `duxx-snapshot`
create/verify/restore provides consistent filesystem snapshots.

---

## 4. Crate map (21 crates, all Apache-2.0)

```
                 access surfaces
   ┌───────────────┬───────────────┬───────────────┐
   │ duxx-server   │  duxx-grpc    │  duxx-mcp      │  + embedded Rust
   │ RESP2/3 TCP   │  tonic gRPC   │  MCP stdio     │  + bindings/python (PyO3)
   │ auth·TLS·     │  streaming    │  JSON-RPC      │  + bindings/node   (napi)
   │ metrics       │  Subscribe    │                │
   └───────┬───────┴───────┬───────┴───────┬────────┘
           └───────────────┼───────────────┘
                           ▼
           ┌───────────────────────────────┐   agent primitives + Phase 7
           │  duxx-memory                  │   MEMORY · TOOL_CACHE · SESSION
           │  decay · cap-eviction · COMPACT│   + duxx-{trace,prompts,datasets,
           └───────────────┬───────────────┘        eval,replay,cost} · duxx-docs
                           ▼
           ┌───────────────────────────────┐
           │  duxx-query — hybrid recall    │   RRF fusion (k=60)
           └───────────────┬───────────────┘
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  duxx-index   │  │ duxx-reactive │  │  duxx-embed   │
│  hnsw_rs HNSW │  │  ChangeBus    │  │ hash/OpenAI/  │
│  tantivy BM25 │  │  pub/sub      │  │ Cohere        │
└───────┬───────┘  └───────────────┘  └───────────────┘
        ▼
┌───────────────────────────────────────────────────────┐
│  duxx-storage — Backend trait                          │
│  redb (durable, ACID, MVCC) · in-memory                │
└───────────────────────────────────────────────────────┘
        ▲            duxx-coldtier (Parquet export + snapshot)
        │            duxx-token   (signed JWT credentials)
        │            duxx-core    (types · errors · Value)
        └── duxx-cli (shell + examples) · duxx-bench (Criterion)
```

| Crate | Role |
|---|---|
| `duxx-core` | Shared `Value`, error types, config |
| `duxx-storage` | `Backend` trait; **redb** (durable ACID/MVCC) + in-memory backends |
| `duxx-index` | **hnsw_rs** HNSW (cosine) + **tantivy** BM25; `rebuild` for compaction |
| `duxx-query` | Hybrid recall — Reciprocal Rank Fusion of vector + BM25 rankings |
| `duxx-embed` | Embedders: deterministic `hash`, OpenAI, Cohere (HTTP) |
| `duxx-memory` | `MEMORY` / `TOOL_CACHE` / `SESSION`, importance decay, cap-eviction, `compact` |
| `duxx-reactive` | `ChangeBus` pub/sub change feed |
| `duxx-trace` | Agent trace/thread store (Phase 7.1) |
| `duxx-prompts` | Versioned prompt registry with semantic search (7.2) |
| `duxx-datasets` | Versioned eval datasets (7.3) |
| `duxx-eval` | Eval runs, scores, regressions, failure clustering (7.4) |
| `duxx-replay` | Deterministic agent replay (7.5) |
| `duxx-cost` | Token + cost ledger, budgets, expensive-query clustering (7.6) |
| `duxx-docs` | Document intelligence: ingest → chunk → embed → index → cite |
| `duxx-token` | Signed short-lived credentials (HS256 + Ed25519 JWT) |
| `duxx-coldtier` | Apache Parquet exporter (`duxx-export`) + `duxx-snapshot` |
| `duxx-server` | RESP2/3 TCP daemon: auth/RBAC, TLS/mTLS, Prometheus, graceful shutdown |
| `duxx-grpc` | tonic gRPC daemon with streaming `Subscribe` + gRPC health |
| `duxx-mcp` | MCP stdio JSON-RPC server (Claude / Cline / any MCP agent) |
| `duxx-cli` | Interactive shell + worked examples |
| `duxx-bench` | Criterion benchmarks |

`bindings/python` (PyO3, abi3 wheel) and `bindings/node` (napi-rs) wrap the
embedded engine.

---

## 5. Agent primitives (`duxx-memory`)

### 5.1 `MEMORY`
Long-term semantic memory. `remember(key, text, embedding)` stores a row and
indexes it in both HNSW and tantivy. `recall(key, query, embedding, k)` runs
hybrid retrieval (§6). Each memory carries an `importance` and a wall-clock
`created_at`; the value used for ranking/eviction is the **effective**
importance, which decays with age:

```
effective_importance = importance × 2^(−age / half_life)
```

An optional row **cap** evicts the lowest effective-importance rows on overflow
(agent-friendly: keep `--max-memories 1_000_000` and trust the store to forget
the boring stuff first).

### 5.2 `TOOL_CACHE`
Two-stage cache for expensive tool/LLM results, keyed by `(tool, args_hash)`:
an **exact** hash hit, or a **semantic near-hit** (cosine ≥ 0.95) against the
args embedding, with per-entry TTL. Turns repeated tool calls into cache hits.

### 5.3 `SESSION`
Hot KV with **sliding TTL** for per-conversation working state (turn buffer,
in-flight tool invocations, scratch flags). In-process, lazily evicted.

---

## 6. Hybrid recall pipeline (`duxx-query`)

```
recall(key, "user's refund issue", q_embedding, k=10)
```

1. **Vector probe** — HNSW (`hnsw_rs`, cosine) returns the top `per = max(k×3, 32)` ids by similarity.
2. **BM25 probe** — tantivy returns the top `per` ids by BM25 score.
3. **Reciprocal Rank Fusion** — the two ranked lists are fused:
   ```
   score(id) = Σ_i  1 / (k_rrf + rank_i(id))     k_rrf = 60
   ```
   Ids ranked highly by *both* retrievers rise to the top; the fusion is
   scale-free (it doesn't care that cosine and BM25 use different scales).
4. **Hydrate + filter** — hits are resolved against the live row map (`by_id`),
   which also guarantees a forgotten row can never surface even before the
   index is compacted.
5. **Optional decay rerank** — `recall_decayed` multiplies each hit by its
   effective importance for recency-aware ordering.

---

## 7. Deletion-safe recall (compaction)

`hnsw_rs` has no in-place delete: a forgotten vector stays a node in the
navigable graph. Recall stays **correct** (results are filtered through the live
row map), but those tombstone nodes are still traversed during search, so recall
**quality** degrades in exactly the regions you've deleted from most.

`MEMORY` tracks a **tombstone ratio** `(indexed − live) / live`. `COMPACT`
rebuilds the HNSW graph and the BM25 index from the surviving rows and swaps
them in atomically, reclaiming all tombstones. When the ratio crosses a
threshold (default `0.20`), the next write auto-compacts; it is also exposed
explicitly on every surface (`COMPACT` in RESP, `compact()` in Python/MCP/gRPC)
and observable via `duxx_resp_memory_compactions` / `duxx_resp_memory_tombstone_ratio`.

---

## 8. Storage layer (`duxx-storage`)

The `Backend` trait abstracts durable byte-keyed storage. The default backend is
**redb** — pure Rust, ACID, MVCC, mmap-backed, no FFI — used for the memory row
store and the Phase-7 registries. An in-memory backend is available for tests
and ephemeral use.

Fully-persistent memory (`MemoryStore::open_at`) keeps three artifacts under one
directory so a graceful restart skips the expensive rebuild:

```
<data_dir>/
├── store.redb      # memory rows (redb)
├── tantivy/        # BM25 index (mmap)
└── hnsw/           # HNSW graph dump + id-map sidecar
```

On reopen, if the on-disk indices match the row count they are loaded directly
(sub-second cold start); after a hard kill they are rebuilt from the rows.

---

## 9. Concurrency model

- **Tokio** async runtime for the servers; the engine core is synchronous and
  cheaply `Clone` (Arc internals).
- **Reads** take shared locks over the indices and the row map.
- **Writes** take short-lived write locks; tantivy commits are **batched**
  (default every 100 inserts) and auto-flushed on read so "insert-then-recall"
  holds.
- **Compaction** takes the index write locks for the rebuild; it acquires them
  in the same order as writes, so the two can't deadlock.

---

## 10. Access surfaces

| Surface | Crate / binding | Notes |
|---|---|---|
| Embedded Rust | `duxx-memory` etc. | Zero overhead, in-process |
| Python | `bindings/python` (PyO3, abi3) | One wheel for 3.8–3.13 |
| Node / TypeScript | `bindings/node` (napi-rs) | Native module |
| RESP2/3 | `duxx-server` | Valkey/Redis-client compatible |
| gRPC | `duxx-grpc` (tonic) | Streaming `Subscribe`, gRPC health |
| MCP stdio | `duxx-mcp` | Claude / Cline / any MCP agent |

---

## 11. Reactive subscriptions (`duxx-reactive`)

Every mutation publishes a `ChangeEvent` on a `ChangeBus`. Clients `SUBSCRIBE` /
`PSUBSCRIBE` (RESP) or stream the gRPC `Subscribe` RPC to watch memory changes —
e.g. a supervisor agent watching a fleet of workers. The bus is a bounded
broadcast channel (lossy by design under slow consumers).

---

## 12. Security (single-node)

- **Auth**: shared token (`--token` / `DUXX_TOKEN`) or a role-based API-key
  catalog (`--auth-key principal:token:role`, `DUXX_AUTH_KEYS`) with
  read/write/admin roles and finer capabilities.
- **Transport**: native TLS on RESP + gRPC (rustls); mTLS client-cert
  verification via `--tls-client-ca`.
- **Audit**: JSON-lines security audit log (`--audit-log`).
- **Limits**: configurable RESP resource limits, connection caps, and
  per-connection command-rate limits.
- **Observability**: Prometheus `/metrics` + `/health`; gRPC health protocol.
- **Credentials**: `duxx-token` issues/verifies short-lived HS256 or Ed25519
  JWTs (the data plane can verify with only a public key).

---

## 13. Future architecture (DuxxDB Cloud)

The following are **not** in this repo — they are the managed-services layer
(source-available, BUSL-1.1) that builds on these Apache crates:

- **Multi-tenancy** — physical per-workspace isolation of every primitive.
- **Control plane** — orgs / projects / environments / API keys / placement /
  usage / billing, with durable persistence and SSO.
- **Replication / HA** — leader/follower, failover, cross-AZ, PITR.
- **Cloud Console + Studio** — operator UI and cross-primitive debugging.

Their engineering status and roadmap are tracked as issues in the
`duxxdb-cloud` repository.

---

## 14. Non-goals (this repo)

- Distributed / sharded cluster mode and multi-tenancy — single node here; those
  are DuxxDB Cloud.
- OLAP dashboards — that's the cold tier's job (Parquet → Spark/DuckDB).
- A full SQL surface — retrieval is via the agent primitives and hybrid recall,
  not a SQL parser.

---

## 15. Implementation notes

Third-party dependency notices required for distribution are maintained in
[`../NOTICE`](../NOTICE).
