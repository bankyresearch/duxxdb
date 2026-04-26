# Phase 2.3 — Lance-backed `Table`

**Status:** Planned. Deferred to its own session.
**Why deferred:** Lance pulls in ~300 transitive crates and several FFI/native
build steps; first compile is ~10–15 min and known-risky on Windows + mingw.
The current Phase 1/2 code is healthy and all green — adding Lance to the
same change would risk the whole stack on a toolchain issue.

---

## Goals

1. Persist a `Table` to disk in [Lance format](https://github.com/lancedb/lance) —
   columnar, Arrow-native, with versioning.
2. Keep the public `Table` API stable. Callers in `duxx-memory` should
   not change.
3. Vector and BM25 indices remain in `duxx-index` (HNSW + tantivy).
   Lance's own vector indices stay unused for now — we may revisit
   in Phase 3 once we benchmark them against `hnsw_rs`.
4. Build behind a feature flag (`lance`, off by default) so cargo
   defaults stay fast and the workspace stays portable.

## Non-goals (this phase)

- Distributed object-store backing (S3, GCS) — local file system only.
- Time-travel / version pinning queries.
- Schema evolution.
- Tantivy / HNSW persistence — those stay in-memory and rebuild on open.

---

## Crate plan

```
duxx-storage/
├── Cargo.toml
│     [features]
│     default = []
│     lance = ["dep:lance", "dep:arrow-array", "dep:arrow-schema",
│              "dep:tokio"]
│     [dependencies]
│     lance        = { workspace = true, optional = true }
│     arrow-array  = { workspace = true, optional = true }
│     arrow-schema = { workspace = true, optional = true }
│     tokio        = { workspace = true, optional = true }
└── src/
    ├── lib.rs           # re-exports; selects backend by feature
    ├── memory.rs        # current `Table` (renamed)
    └── lance.rs         # new — `LanceTable`, only with `lance` feature
```

`Table` becomes a trait:

```rust
pub trait Table: Send + Sync {
    fn schema(&self) -> &Schema;
    fn insert(&self, row: Row) -> Result<RowId>;
    fn scan(&self) -> Vec<(RowId, Row)>;
    fn get(&self, id: RowId) -> Option<Row>;
    fn len(&self) -> usize;
}
```

`MemoryTable` (in-memory, current code) and `LanceTable` (disk-backed)
both implement it. Existing tests that take `Table` keep working without
touching tantivy/HNSW changes.

---

## Phase 2.3 work breakdown

1. **Convert `Table` to a trait** (no Lance code; just refactor).
   - 1 commit, no behavior change. Existing tests + chatbot demo keep
     passing on the in-memory backend.
2. **Add the feature flag and stub `LanceTable`** with all methods
   `todo!()`. Verify `cargo check` passes on default features.
3. **Run `cargo check --features lance`** for the first time.
   Expected pain points:
   - Native deps may need MSVC + Windows SDK (we're on mingw + WinLibs).
     If they fail, install the SDK or use the GitHub Actions CI as the
     "always-works" environment for `--features lance`.
   - `protoc` may be required for protobuf code generation. Install via
     `winget install protobuf` or use a vendored binary.
4. **Implement `LanceTable::open` / `insert` / `scan`** using
   `lance::Dataset`. Each insert appends a small RecordBatch; periodic
   compaction (Phase 3) merges small fragments.
5. **Wire `MemoryStore::with_storage(Box<dyn Table>)`** so users opt in.
6. **Add a Lance-specific bench** that measures insert + recall over a
   persisted dataset on cold start.

---

## Open questions

- **Schema-to-Arrow mapping.** Our `ColumnKind::Vector(VectorSpec)`
  becomes `FixedSizeList<Float32, dim>`. Decide handling for `Json` —
  serialize to UTF-8 string, or use Arrow's struct type? Probably
  string for simplicity; revisit if perf matters.
- **Insert batching.** Lance prefers larger RecordBatches per write.
  Mirror Phase 2.5's batching strategy: queue rows, commit every N or
  on read-time flush.
- **Cold start.** On `open(path)`, we'd need to scan all rows back into
  the HNSW + tantivy indices. For 1 M rows that's minutes. Phase 3 adds
  index persistence so we don't pay that.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lance won't compile on Windows + mingw | medium | Switch to MSVC + Windows SDK (we know how — see SETUP.md) or build only in CI |
| `protoc` missing | medium | Install via winget; document |
| Major API change in newer Lance versions | low | Pin version in Cargo.toml |
| Cold-start reindex too slow on 1M rows | medium | Phase 3 — persist HNSW + tantivy alongside Lance fragments |

---

## When to start

Start fresh in a new session. Budget 2–4 focused hours for the full
Phase 2.3. Don't multi-task with other features — one toolchain incident
swallows that budget.
