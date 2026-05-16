# Migrating from DuxxDB v0.1.x to v0.2.0

This guide covers the upgrade from the v0.1.x line (Phase 7 primitives
in-memory only) to v0.2.0 (Phase 7 primitives persistent via a new
`Backend` trait, plus native Python bindings for the prompt registry).

## TL;DR

* **Wire-protocol stability is total.** Every `TRACE.*`, `PROMPT.*`,
  `DATASET.*`, `EVAL.*`, `REPLAY.*`, and `COST.*` command keeps the
  same arguments and same response shape. v0.1.x clients keep working
  against a v0.2.0 server unchanged.
* **In-memory mode is unchanged.** If you do not pass
  `--phase7-storage`, the server behaves exactly as v0.1.x did —
  Phase 7 state lives in RAM and is lost on restart.
* **Opt in to persistence with one flag.** Pass
  `--phase7-storage dir:./phase7-data` to make traces, prompts,
  datasets, evals, replays, and the cost ledger durable.
* **Python users get a native `duxxdb.PromptRegistry`.** Pure-Python
  facade users (`duxxdb.server.ServerClient`) continue to work
  unchanged; the native class is an extra option, not a replacement.

## What's new

### Persistent storage for every Phase 7 primitive

v0.1.x kept these in `RwLock<HashMap<...>>` — handy for unit tests
and notebooks, but every restart erased the state. v0.2.0 wires every
primitive to a new `duxx_storage::Backend` trait:

| Primitive | Backend tables |
|---|---|
| `TraceStore` | `trace.spans` |
| `PromptRegistry` | `prompts`, `tags`, `next_version` |
| `DatasetRegistry` | `dataset.versions`, `dataset.tags`, `dataset.next_version`, `dataset.schemas` |
| `EvalRegistry` | `eval.runs`, `eval.scores` |
| `ReplayRegistry` | `replay.sessions`, `replay.runs` |
| `CostLedger` | `cost.entries`, `cost.budgets` |

Two `Backend` implementations ship in `duxx-storage`:

* `MemoryBackend` — `BTreeMap` per table, no durability. Default;
  preserves v0.1.x behavior bit-identically.
* `RedbBackend` — ACID storage backed by [redb]. Pure Rust, no FFI.

[redb]: https://github.com/cberner/redb

### `--phase7-storage` CLI flag

`duxx-server` accepts a new flag:

```
--phase7-storage SPEC   Persist Phase 7 primitives.
                        Default: memory (matches v0.1.x).
                        Use 'dir:./path/to/dir' for one redb file
                        per primitive under the directory.
```

Equivalent env var: `DUXX_PHASE7_STORAGE`.

### Native Python class for the prompt registry

`bindings/python` now exposes `duxxdb.PromptRegistry` directly,
backed by the same Rust crate the server uses. Useful for notebook
workflows that don't want to spin up a daemon:

```python
import duxxdb

# In-memory — preserves v0.1.x in-memory feel
r = duxxdb.PromptRegistry(dim=16)
v1 = r.put("classifier", "You are a refund agent.")
r.tag("classifier", v1, "prod")

# Or open with durable storage
r = duxxdb.PromptRegistry(dim=16, storage="redb:./prompts.redb")
```

The Python facade for the other five primitives via
`duxxdb.server.ServerClient` (introduced in v0.1.3) is unchanged.

## Upgrade procedure

### Existing in-memory workloads

If you're not yet asking for persistence, **there is nothing to do**.
Drop the v0.2.0 binary in place of v0.1.x and the server starts
exactly the way it did before. The wire protocol is unchanged.

### Adopting Phase 7 persistence

1. Pick a directory for Phase 7 state. It can sit next to the
   existing `--storage dir:./data/` directory or anywhere else
   writable by the daemon.
2. Start the daemon with the new flag:

   ```bash
   duxx-server \
     --addr 0.0.0.0:6379 \
     --storage dir:./data/ \
     --phase7-storage dir:./phase7-data/
   ```

3. On first boot the directory is populated with six redb files
   (`prompts.redb`, `datasets.redb`, `evals.redb`, `replays.redb`,
   `traces.redb`, `costs.redb`) and a `version.json` schema-pin file.
4. Existing in-memory state at the moment of cut-over is lost. If
   you have v0.1.x in-memory prompts you want to keep, dump them via
   `PROMPT.LIST` / `PROMPT.GET` before stopping the daemon and replay
   them into the v0.2.0 daemon via `PROMPT.PUT` once it's up. Most
   v0.1.x deployments treated Phase 7 state as ephemeral anyway, so
   this is rarely needed.

### `version.json`

The first boot writes a small JSON file at the root of the storage
directory:

```json
{"schema": 1, "duxxdb": "0.2.0"}
```

The `schema` integer is the upgrade contract. v0.3.x and later will
check this on startup and refuse (with a clear message) to load a
directory written by a binary newer than the running one. Do not
edit this file by hand.

## Behavior notes

### Crash safety

Mutations go through the backend **before** any in-memory state is
updated. If the process is killed mid-write, the on-disk row is the
source of truth; rehydrate on next open will pick it up. The
exception is the failure-clustering HNSW (used by
`EVAL.CLUSTER_FAILURES` and `COST.CLUSTER_EXPENSIVE`) — it stays in
RAM and is rebuilt on `open` by re-embedding every stored row.

### Vector indices rebuild on open

The HNSW that backs `PROMPT.SEARCH`, `DATASET.SEARCH`,
`EVAL.CLUSTER_FAILURES`, and `COST.CLUSTER_EXPENSIVE` is **not**
persisted in v0.2.0 — it's rebuilt by re-embedding every persisted
row on `open`. That keeps the storage layout small and avoids the
complexity of persisting a stable HNSW dump for six different
schemas at once. For prompt-catalog volumes (~hundreds of rows) the
rebuild is essentially instant; for million-row dataset registries
plan for a few seconds of startup latency in v0.2.0. Persistent
indices land in v0.2.1.

### Performance posture

The write-through path adds one redb transaction per mutation. On
the SSD-backed workloads we measured this is a low single-digit
millisecond overhead per `PROMPT.PUT` / `EVAL.SCORE` / etc. — fine
for agent workloads, well under any LLM call this would be feeding.

If you need higher write throughput than that, switch the affected
primitive back to in-memory by passing
`--phase7-storage memory` and treating Phase 7 state as a process
warm cache.

### Single-file redb is not supported (yet)

We considered `--phase7-storage redb:./phase7.redb` (one file shared
by every primitive). It works conceptually — each crate already
prefixes its tables — but redb takes an exclusive file lock per
`Database` handle, and we currently open six handles. A clean fix
(sharing one `Arc<Database>` across all backends) lands in v0.2.1.
For now, use `dir:` — it's the recommended path anyway because per-
primitive isolation is easier to operate.

## Rolling back

There is no breaking change to roll back. Restart on v0.1.x and the
new directory is simply ignored. Phase 7 state already written to
disk stays there; v0.1.x just won't know about it.

## Python `duxx-ai` users

Nothing changes on the application side. The `DuxxBackend` (memory)
and `DuxxExporter` (trace) integrations introduced in `duxx-ai`
v0.31.0 keep working unchanged against a v0.2.0 server, and they
benefit from the new persistence transparently.

If you want native (embedded, no server) access to the prompt
registry, the new `duxxdb.PromptRegistry` class works alongside the
existing facade:

```python
# Embedded mode — no server required
from duxxdb import PromptRegistry

reg = PromptRegistry(dim=16, storage="redb:./prompts.redb")
reg.put("classifier", "You are a refund agent.")
```

## Help, support, issues

Open an issue at <https://github.com/bankyresearch/duxxdb/issues>.
Include the `--phase7-storage` value you used and the contents of
`version.json` if you hit an upgrade error.
