# DuxxDB v0.2.0 — Persistence + Native Python Bindings for Phase 7

**Status:** plan
**Target ship:** ~3 weeks of focused work
**Owner:** core team

This document scopes v0.2.0 and breaks it into discrete, ship-able
increments. v0.1.x is feature-complete for Phase 7 (59 RESP commands
across six primitives) but the storage and Python stories lag:

* Phase 7 primitives (`TraceStore`, `PromptRegistry`,
  `DatasetRegistry`, `EvalRegistry`, `ReplayRegistry`, `CostLedger`)
  are **in-memory only**. They live in `RwLock<HashMap<...>>`. Process
  exit loses everything.
* Phase 7 is not in the `duxxdb` Python wheel. Python users today
  reach Phase 7 over the RESP facade (`duxxdb.server.ServerClient`,
  shipped in v0.1.3). That works but pays per-call serialization;
  embedded use is faster and is a hard requirement for some
  notebook / batch workflows.

v0.2.0 closes both gaps.

---

## What v0.2.0 ships

1. **Persistent storage** for every Phase 7 primitive, behind the
   same `storage="dir:..."` and `storage="redb:..."` selectors used
   today for memory / tool-cache / session.
2. **Native PyO3 bindings** for every Phase 7 primitive:
   * `duxxdb.TraceStore`
   * `duxxdb.PromptRegistry`
   * `duxxdb.DatasetRegistry`
   * `duxxdb.EvalRegistry`
   * `duxxdb.ReplayRegistry`
   * `duxxdb.CostLedger`
3. **Migration path** from v0.1.x to v0.2.0:
   * Existing v0.1.x clients keep working unchanged — the wire
     protocol does not break.
   * Existing on-disk directories created by `duxx-memory` keep
     loading.
4. **Documentation:**
   * Each new primitive gets the same recipe coverage in
     `INTEGRATION_GUIDE.md` that memory/tool-cache/session got in
     Phase 6.
   * Migration notes from in-memory v0.1.x to persistent v0.2.0.

What v0.2.0 does **not** ship: distributed mode, replication,
horizontal sharding. Those are v0.3.x.

---

## Persistence design

### Today

`duxx-memory` uses a layered abstraction:

```
MemoryStore
  ├── RwLock<HashMap<u64, MemoryRow>>      ← in-memory rows
  ├── VectorIndex                          ← HNSW (always in RAM)
  ├── BM25Index (tantivy)                  ← optional disk
  └── Storage trait (u64-keyed K/V)
        ├── MemoryStorage  (HashMap)
        └── RedbStorage    (redb)
```

The trait is `u64`-keyed because memory rows are auto-numbered.

### The Phase 7 mismatch

Every Phase 7 primitive needs **multiple logical tables** with
**string composite keys**:

| Primitive | Tables |
|---|---|
| TraceStore | `spans` (span_id → Span), `trace_index` (trace_id → \[span_id\]), `thread_index` (thread_id → \[span_id\]) |
| PromptRegistry | `prompts` ((name, version) → Prompt), `tags` ((name, tag) → version), `name_index` |
| DatasetRegistry | `datasets` ((name, version) → Dataset), `rows` ((name, version, row_id) → DatasetRow), `tags` |
| EvalRegistry | `runs` (run_id → EvalRun), `scores` ((run_id, row_id) → EvalScore) |
| ReplayRegistry | `sessions` (trace_id → ReplaySession), `runs` (run_id → ReplayRun) |
| CostLedger | `entries` (id → CostEntry), `budgets` (tenant → CostBudget) |

The existing `Storage` trait is too narrow. We need a multi-table,
string-keyed abstraction.

### Proposed: the `Backend` trait

```rust
/// Multi-table, string-keyed persistence backend.
///
/// Each Phase 7 primitive owns one `Backend`. Tables are named
/// (e.g. `"prompts"`, `"tags"`) and are created lazily on first
/// access. All values are opaque bytes — primitives serialize their
/// own rows with serde + bincode.
pub trait Backend: Send + Sync + std::fmt::Debug {
    /// Insert or overwrite a value.
    fn put(&self, table: &str, key: &[u8], value: &[u8]) -> Result<()>;
    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn delete(&self, table: &str, key: &[u8]) -> Result<bool>;
    /// Iterate every (key, value) in a table. Order is sorted by key.
    fn scan(&self, table: &str) -> Result<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + '_>>;
    /// Iterate every key with the given prefix. Cheap for composite keys.
    fn scan_prefix(&self, table: &str, prefix: &[u8]) -> Result<Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + '_>>;
    /// Atomic batch — either all writes apply or none do.
    fn batch(&self, ops: &[BatchOp]) -> Result<()>;
}

pub enum BatchOp<'a> {
    Put { table: &'a str, key: &'a [u8], value: &'a [u8] },
    Delete { table: &'a str, key: &'a [u8] },
}
```

Two implementations:

* `MemoryBackend` — `RwLock<HashMap<String, BTreeMap<Vec<u8>, Vec<u8>>>>`. No durability. Default.
* `RedbBackend` — one redb database, one table per primitive table. Durable.

Composite keys serialize as `b"<part1>\0<part2>\0..."`. `scan_prefix`
returns the right rows by encoding the prefix the same way.

### Per-primitive port plan

Each primitive port is a self-contained PR:

1. Add `backend: Arc<dyn Backend>` to the registry struct.
2. Every `put / get / delete` writes through the backend in addition
   to the in-memory map.
3. Add `open(backend: Arc<dyn Backend>) -> Result<Self>` that
   rebuilds the in-memory state by `scan`ning every table on startup.
4. The HNSW vector index gets rebuilt on open (same pattern as
   `MemoryStore`).
5. Tests:
   * Round-trip: put → open new instance → get returns the same value.
   * Crash safety: put → drop instance mid-flight → reopen → recover.
   * Backend swap: in-memory tests still pass; new redb tests added.

**Recommended order** (smallest to largest, ship one per week):

1. `PromptRegistry` — three tables, no vector index churn beyond
   what's already there. Establishes the pattern.
2. `CostLedger` — two tables.
3. `DatasetRegistry` — three tables, vector rebuild for row search.
4. `EvalRegistry` — two tables + vector rebuild for failure
   clustering.
5. `ReplayRegistry` — two tables.
6. `TraceStore` — three tables, indices on parent / thread / kind.

---

## Native Python bindings design

### Today

`bindings/python/src/lib.rs` exposes three PyO3 classes:
`MemoryStore`, `ToolCache`, `SessionStore`. Each is a thin wrapper
around the Rust struct with `#[pymethods]` for the public API. The
single-wheel abi3 build covers Python 3.8 → 3.13.

### v0.2.0 additions

Add six new `#[pyclass]` types, one per Phase 7 primitive. Each
follows the established pattern:

```rust
#[pyclass]
pub struct PromptRegistry {
    inner: Arc<duxx_prompts::PromptRegistry>,
}

#[pymethods]
impl PromptRegistry {
    #[new]
    #[pyo3(signature = (storage = None))]
    fn new(storage: Option<&str>) -> PyResult<Self> {
        let backend = resolve_backend(storage)?;
        let inner = Arc::new(duxx_prompts::PromptRegistry::open(backend)?);
        Ok(Self { inner })
    }

    fn put(&self, name: &str, content: &str, metadata: Option<&Bound<'_, PyAny>>) -> PyResult<u64> { ... }
    fn get(&self, name: &str, version_or_tag: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PyPrompt>> { ... }
    // … one method per public function on the Rust registry.
}
```

The `storage` constructor argument mirrors what `duxx-server`
accepts on the command line:

* `None` — in-memory (`MemoryBackend`)
* `"redb:./path/to/file.redb"` — single-file durable
* `"dir:./path/to/dir"` — directory containing redb + tantivy indices

### Python-side return types

Each PyO3 class has companion `#[pyclass]` types for its returns:

```python
prompt = registry.get("classifier")
# Prompt(name='classifier', version=3, content='...', metadata={...}, tags=['prod'])

hits = registry.search("refund", k=5)
# [PromptHit(name='refund_classifier', version=2, score=0.84, content='...'), ...]
```

These mirror the dataclasses already shipping in
`duxxdb.server.ServerClient` (Phase 7 RESP facade), so users moving
from RESP to embedded get the same shape.

### Optional dependency posture

The new classes are part of the base wheel — no extra is needed for
embedded use. The optional `[server]` extra (introduced with the
RESP facade) keeps `redis-py` opt-in, since it is only needed for
the server client.

---

## Migration

### From v0.1.x to v0.2.0

**No wire-protocol breakage.** Every RESP command keeps the same
signature and same response shape. v0.1.x clients keep working
against a v0.2.0 server.

**New CLI flags on `duxx-server`:**

```
--phase7-storage SPEC    Where Phase 7 primitives persist
                         (default: in-memory, matches v0.1.x behavior)
                         Use `redb:./phase7.redb` or `dir:./phase7/`
                         for durable mode.
```

If `--phase7-storage` is omitted, the server behaves exactly like
v0.1.x — Phase 7 stays in memory and is lost on restart. This makes
the upgrade a no-op for callers who are not ready for persistence.

### On-disk format

A new directory layout, separate from memory's `dir:./data/`:

```
phase7-data/
├── prompts.redb       # PromptRegistry backend
├── datasets.redb      # DatasetRegistry backend (+ rows.tantivy/)
├── evals.redb         # EvalRegistry backend (+ failures.hnsw)
├── replays.redb       # ReplayRegistry backend
├── traces.redb        # TraceStore backend
├── costs.redb         # CostLedger backend
└── version.json       # {"schema": 1, "duxxdb": "0.2.0"}
```

The `version.json` is the upgrade contract — v0.3.x reads it,
checks the `schema` int, and refuses (with a clear message) if it
is newer than the binary supports.

---

## Test strategy

Each persistence PR adds:

1. **Property-style round-trip** — put N rows, drop, reopen, scan,
   assert every row is preserved.
2. **Crash safety** — same as round-trip but with the writer
   dropped before its `flush()` is called.
3. **Backend equivalence** — every existing in-memory test
   parameterized over both `MemoryBackend` and `RedbBackend`. New
   `tests/persistence_*.rs` directories per crate.
4. **Migration regression** — open a fixture directory produced by
   the v0.1.x test suite, assert the v0.2.0 binary loads it.

Each PyO3 PR adds:

1. **Unit tests** in `bindings/python/tests/` parameterized over
   in-memory and durable storage modes.
2. **Cross-mode equivalence** — same workload through embedded
   `duxxdb.PromptRegistry` vs `duxxdb.server.ServerClient`, assert
   identical results.

---

## Ship sequence

| Week | PR | What |
|---|---|---|
| 1 | Backend trait + `MemoryBackend` + `RedbBackend` + composite-key helpers | foundations |
| 1 | PromptRegistry persistence + Python `duxxdb.PromptRegistry` | first primitive |
| 2 | CostLedger + DatasetRegistry persistence + Python bindings | two more primitives |
| 2 | EvalRegistry + ReplayRegistry persistence + Python bindings | two more primitives |
| 3 | TraceStore persistence + Python `duxxdb.TraceStore` | last primitive |
| 3 | `--phase7-storage` CLI flag + `version.json` + migration docs | server integration |
| 3 | v0.2.0 release | tag, build wheels, publish PyPI + GHCR |

Each row is one PR. Each PR ships independently behind the
`MemoryBackend` default, so master stays releasable throughout.

---

## Out of scope for v0.2.0

* **Tantivy persistence for Phase 7 search indices.** v0.2.0 keeps
  search indices in RAM (rebuilt on open). Persistent search comes
  in v0.2.1 after we measure how big real-world Phase 7 indices
  actually get.
* **Replication / multi-node.** v0.3.x.
* **Snapshot / restore CLI.** v0.2.x point release after we see
  what users want.
* **Schema migrations.** v0.2.0 ships schema v1 and refuses to load
  anything else. Real migrations come with the first breaking
  change.

---

## Status

- [ ] Backend trait merged
- [ ] PromptRegistry persistence
- [ ] CostLedger persistence
- [ ] DatasetRegistry persistence
- [ ] EvalRegistry persistence
- [ ] ReplayRegistry persistence
- [ ] TraceStore persistence
- [ ] `duxxdb.PromptRegistry` (Python)
- [ ] `duxxdb.CostLedger` (Python)
- [ ] `duxxdb.DatasetRegistry` (Python)
- [ ] `duxxdb.EvalRegistry` (Python)
- [ ] `duxxdb.ReplayRegistry` (Python)
- [ ] `duxxdb.TraceStore` (Python)
- [ ] `--phase7-storage` CLI flag + `version.json`
- [ ] Migration docs
- [ ] v0.2.0 tagged + published
