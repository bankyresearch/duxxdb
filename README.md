# DuxxDB

**The database built for AI agents.**

Hybrid structured + vector + full-text search in one embedded engine, with realtime reactive subscriptions and native support for the Model Context Protocol.

> **Status:** 🚧 pre-alpha. Architecture locked, implementation in progress. See [docs/ROADMAP.md](./docs/ROADMAP.md).

---

## Why

Existing options force AI agents to duct-tape three databases together:

- **Redis / DiceDB** for session state (fast, but no vectors, no SQL).
- **Qdrant / Milvus / pgvector** for embeddings (vectors only, awkward with structured filters).
- **Postgres / DuckDB** for structured data (no vector-first query planner).

Every round-trip costs latency. Every sync costs consistency. Voice bots with 200 ms end-to-end budgets cannot afford this.

DuxxDB fuses all three into a **single low-latency engine** with **agent-native primitives** (`MEMORY`, `TOOL_CACHE`, `SESSION`) built in.

---

## Targets

| Operation | p99 latency |
|---|---|
| Session KV read | < 0.5 ms |
| Hybrid recall (vector + BM25) | < 10 ms |
| Structured filter + ANN | < 15 ms |
| Write + index | < 5 ms |

---

## Quickstart (coming after Phase 1)

```rust
use duxxdb::{Duxx, Memory};

let db = Duxx::open("./mydb")?;
db.remember("user_42", "I want a refund for order #9910").await?;
let hits = db.recall("user_42", "refund issue", 10).await?;
```

```python
from duxxdb import Duxx

db = Duxx.open("./mydb")
db.remember(user="user_42", text="I want a refund for order #9910")
hits = db.recall(user="user_42", query="refund issue", k=10)
```

---

## Architecture

See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the full design.

Short version: Rust workspace, embedded-first, Lance for storage, usearch for vectors, tantivy for BM25, DataFusion for SQL, all glued together with a DuxxDB-native hybrid query planner. Server mode (gRPC + RESP3 + MCP) ships in Phase 3.

---

## Repository layout

```
duxxdb/
├── Cargo.toml            workspace root
├── crates/               10 Rust crates (core, storage, index, query, …)
├── bindings/             Python + TypeScript + Go bindings
├── examples/             end-to-end demos
└── docs/                 architecture, setup, roadmap
```

---

## Contributing

This is pre-alpha; expect shape to change. Read [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) and [docs/ROADMAP.md](./docs/ROADMAP.md) before opening issues. Pick any unchecked box in the roadmap.

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
