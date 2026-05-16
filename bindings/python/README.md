# duxxdb — Python bindings

Native Python bindings for [DuxxDB](https://github.com/duxxdb/duxxdb), the
agent-native hybrid database. Built with [PyO3] + [maturin] against the
stable Python ABI (`abi3-py38`) so a single wheel supports
**Python 3.8 through 3.13**.

## Install

### From a wheel (recommended)

```bash
pip install duxxdb-0.1.0-cp38-abi3-<platform>.whl
```

(For now, build the wheel locally — see the next section. Public
PyPI release lands with v0.1.0.)

### Build from source

```bash
# Prerequisites: Rust toolchain, Python ≥ 3.8.
pip install --user maturin
cd bindings/python
maturin build --release
pip install --force-reinstall ../../target/wheels/duxxdb-0.1.0-cp38-abi3-*.whl
```

On Windows + Git Bash the workspace uses the GNU toolchain — see
[../../docs/SETUP.md](../../docs/SETUP.md) for the WinLibs MinGW
prerequisite.

## Quickstart

```python
import duxxdb

store = duxxdb.MemoryStore(dim=4)

def embed(text):
    """Replace this with a real embedder (OpenAI / Cohere / local BGE)."""
    import hashlib
    h = int(hashlib.sha1(text.lower().encode()).hexdigest()[:16], 16)
    v = [(h >> (i*4)) & 0xff for i in range(4)]
    norm = sum(x*x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]

store.remember(key="alice", text="I lost my wallet at the cafe", embedding=embed("wallet"))
store.remember(key="alice", text="My favorite color is blue",     embedding=embed("blue"))

hits = store.recall(key="alice", query="wallet",
                    embedding=embed("wallet"), k=3)
for hit in hits:
    print(f"{hit.score:.4f}  {hit.text}")
```

Output:

```
0.0328  I lost my wallet at the cafe
0.0161  My favorite color is blue
```

## API surface

### Embedded (native, no server required)

| Class | Constructor | Methods |
|---|---|---|
| `MemoryStore` | `dim`, `capacity=100_000` | `remember(key, text, embedding) -> id`, `recall(key, query, embedding, k=10) -> [MemoryHit]`, `len()`, `dim` |
| `MemoryHit` | (returned by `recall`) | `id`, `key`, `text`, `score` |
| `ToolCache` | `threshold=0.95` | `put(tool, args_hash, args_embedding, result, ttl_secs=3600)`, `get(tool, args_hash, args_embedding) -> ToolCacheHit \| None`, `purge_expired()`, `len()` |
| `ToolCacheHit` | (returned by `get`) | `kind` (`"exact"` or `"semantic_near_hit"`), `similarity`, `result` |
| `SessionStore` | `ttl_secs=1800` | `put(session_id, data)`, `get(session_id) -> bytes \| None`, `delete(session_id) -> bool`, `purge_expired()`, `len()` |
| `PromptRegistry` | `dim=32`, `storage=None` | `put(name, content, metadata=None) -> version`, `get(name, version_or_tag=None) -> Prompt \| None`, `list(name) -> [Prompt]`, `names() -> [str]`, `tag(name, version, tag)`, `untag(name, tag) -> bool`, `delete(name, version) -> bool`, `search(query, k=10) -> [(name, version, score, content)]`, `diff(name, v_a, v_b) -> str` |
| `Prompt` | (returned by `get` / `list`) | `name`, `version`, `content`, `tags`, `metadata` (decoded JSON), `created_at_unix_ns` |
| `PromptHit` | (returned by `search`) | `prompt`, `score` |

### Phase 7 (Phase-7 agent ops, native bindings new in v0.2.1)

| Class | Constructor | Methods |
|---|---|---|
| `CostLedger` | `dim=32`, `storage=None` | `record(tenant, model, tokens_in, tokens_out, cost_usd, ...)`, `query(...)`, `total(tenant)`, `aggregate(group_by, ...)`, `set_budget(tenant, period, amount_usd, warn_pct=0.8, ...)`, `get_budget(tenant)`, `delete_budget(tenant)`, `status(tenant)` |
| `CostEntry` / `Budget` | (returned by ledger methods) | typed fields incl. `metadata` (decoded JSON) |
| `DatasetRegistry` | `dim=32`, `storage=None` | `create(name, schema=None)`, `add(name, rows, metadata=None) -> version`, `get(name, version_or_tag=None)`, `list(name)`, `names()`, `tag/untag/delete`, `sample(name, version, n, split=None)`, `size`, `splits`, `search(query, k=10, name_filter=None)` |
| `Dataset` / `DatasetRow` | (returned by registry) | typed fields incl. `metadata`, `data`, `annotations`, `schema` (all decoded JSON) |
| `EvalRegistry` | `dim=32`, `storage=None` | `start(...)`, `score(run_id, row_id, score, output_text="", notes=None)`, `complete(run_id) -> EvalSummary`, `fail(run_id, reason)`, `get(run_id)`, `scores(run_id)`, `list(dataset_name=None, dataset_version=None)`, `compare(run_a, run_b) -> tuple`, `cluster_failures(run_id, ...) -> list[tuple]` |
| `EvalRun` / `EvalScore` / `EvalSummary` | (returned by registry) | typed fields incl. `metadata`, `notes` (decoded JSON) |
| `ReplayRegistry` | `storage=None` | `capture(trace_id, kind, input, ...)`, `get_session(trace_id)`, `list_sessions()`, `start(source_trace_id, mode="live", ...)`, `step(run_id)`, `record_output(run_id, idx, output)`, `complete/fail`, `set_replay_trace_id`, `get_run`, `list_runs(source_trace_id=None)` |
| `ReplaySession` / `ReplayInvocation` / `ReplayRun` | (returned by registry) | typed fields incl. `input`, `output`, `metadata` (decoded JSON) |
| `TraceStore` | `storage=None` | `record_span(trace_id, span_id, name, ...)`, `close_span(span_id, end_unix_ns, status="ok")`, `get_trace(trace_id)`, `subtree(span_id)`, `thread(thread_id)` |
| `Span` | (returned by store) | typed fields incl. `attributes` (decoded JSON) |

Module-level: `duxxdb.__version__`.

### Server (typed Python facade over RESP)

When you're running `duxx-server` as a daemon and want a typed Python
surface over **every** Phase 7 agent-ops primitive (traces, prompts,
datasets, evals, replay, cost ledger), install the `server` extra:

```bash
pip install 'duxxdb[server]'
```

```python
from duxxdb.server import ServerClient

client = ServerClient(url="redis://:<token>@localhost:6379")

# Phase 7.2 — prompt registry
v1 = client.prompts.put("classifier", "you are a refund agent")
prompt = client.prompts.get("classifier", v1)

# Phase 7.3 — dataset registry
client.datasets.create("refunds")
ds_v = client.datasets.add("refunds", [
    {"id": "r1", "text": "I want a refund", "split": "train"},
    {"id": "r2", "text": "Where is my package?", "split": "test"},
])

# Phase 7.4 — evals with summary stats + failure clustering
run_id = client.evals.start(
    dataset_name="refunds",
    dataset_version=ds_v,
    model="gpt-4o-mini",
    scorer="llm_judge",
    prompt_name="classifier",
    prompt_version=v1,
)
client.evals.score(run_id, row_id="r1", score=0.9, output_text="REFUND")
client.evals.score(run_id, row_id="r2", score=0.1, output_text="REFUND")
summary = client.evals.complete(run_id)
print(summary.mean, summary.pass_rate_50)

# Phase 7.6 — cost ledger
client.cost.record(tenant="acme", model="gpt-4o-mini",
                   tokens_in=120, tokens_out=80, cost_usd=0.0023)
print(client.cost.total("acme"))
```

| Namespace | Wraps | Methods (highlights) |
|---|---|---|
| `client.trace` | `TRACE.*` (6 cmds) | `record`, `close`, `get`, `subtree`, `thread`, `search` |
| `client.prompts` | `PROMPT.*` (9 cmds) | `put`, `get`, `list`, `names`, `tag`, `untag`, `delete`, `search`, `diff` |
| `client.datasets` | `DATASET.*` (13 cmds) | `create`, `add`, `get`, `sample`, `size`, `splits`, `search`, `from_recall`, …  |
| `client.evals` | `EVAL.*` (9 cmds) | `start`, `score`, `complete`, `get`, `scores`, `list`, `compare`, `cluster_failures` |
| `client.replay` | `REPLAY.*` (12 cmds) | `capture`, `start`, `step`, `record`, `complete`, `diff`, `list_runs`, … |
| `client.cost` | `COST.*` (10 cmds) | `record`, `query`, `aggregate`, `total`, `set_budget`, `status`, `alerts`, `cluster_expensive` |

All return types are plain `dataclasses` decoded from the server's
JSON responses. The raw `redis.Redis` client is exposed as
`client.raw` for anything not yet wrapped.

## PromptRegistry (embedded, native bindings): versioned prompts with semantic search

New in v0.2.0: native PyO3 bindings for the prompt registry, so you
can use it embedded — no server, no `redis-py` — straight from a
notebook or batch script. The same Rust crate that powers
`duxx-server`'s `PROMPT.*` commands is exposed directly.

```python
import duxxdb

# In-memory (default; matches v0.1.x in-memory feel).
r = duxxdb.PromptRegistry(dim=16)
v1 = r.put("classifier", "You are a refund agent.", metadata={"author": "alice"})
v2 = r.put("classifier", "You are a friendly refund agent.")
r.tag("classifier", v2, "prod")

p = r.get("classifier", "prod")     # resolves the tag
print(p.version, p.content, p.metadata)

# Durable (rows + tags + monotonic counter survive process exit).
r = duxxdb.PromptRegistry(dim=16, storage="redb:./prompts.redb")
r.put("greeting", "Hello! How can I help today?")
# ... process dies ...
r = duxxdb.PromptRegistry(dim=16, storage="redb:./prompts.redb")
assert r.get("greeting").content == "Hello! How can I help today?"

# Semantic search across the catalog.
for hit in r.search("hello", k=3):
    print(hit.score, hit.prompt.name, hit.prompt.content)
```

The HNSW vector index is rebuilt on `open` by re-embedding every
persisted prompt — fine for the typical prompt-catalog scale
(<1000 rows). Larger Phase 7 primitives (datasets, evals) keep
shipping through `duxxdb.server.ServerClient` for now; native
PyO3 bindings for them land progressively through v0.2.x.

## Phase 7 (embedded, native) — v0.2.1 worked example

The five Phase 7 primitives that shipped through the RESP facade in
v0.1.3 are now available natively too — no `duxx-server`, no
`redis-py`. Same Rust crates the server uses.

```python
import duxxdb

# Cost ledger
cost = duxxdb.CostLedger(dim=16, storage="redb:./cost.redb")
cost.record(tenant="acme", model="gpt-4o-mini",
            tokens_in=120, tokens_out=80, cost_usd=0.0023,
            input_text="please refund order #9910")
cost.set_budget("acme", "monthly", 100.0, warn_pct=0.8)
print(cost.total("acme"), cost.status("acme"))

# Dataset registry
ds = duxxdb.DatasetRegistry(dim=16, storage="redb:./ds.redb")
ds.create("refunds", schema={"columns": ["text", "label"]})
v = ds.add("refunds", [
    {"id": "r1", "text": "I want a refund", "split": "train"},
    {"id": "r2", "text": "Where is my package?", "split": "test"},
])

# Eval registry
evals = duxxdb.EvalRegistry(dim=16, storage="redb:./evals.redb")
rid = evals.start(dataset_name="refunds", dataset_version=v,
                  model="gpt-4o-mini", scorer="llm_judge")
evals.score(rid, row_id="r1", score=0.9, output_text="REFUND")
evals.score(rid, row_id="r2", score=0.1, output_text="REFUND")
summary = evals.complete(rid)
print(summary.mean, summary.p99, summary.pass_rate_50)

# Replay
replay = duxxdb.ReplayRegistry(storage="redb:./replay.redb")
replay.capture(trace_id="t1", kind="llm_call",
               input={"messages": [{"role": "user", "content": "hi"}]},
               output={"role": "assistant", "content": "hello"})
run_id = replay.start("t1", mode="live")

# Trace store
trace = duxxdb.TraceStore(storage="redb:./trace.redb")
trace.record_span(trace_id="t1", span_id="root", name="agent.turn",
                  attributes={"user": "alice"}, status="ok",
                  start_unix_ns=1_000_000_000)
spans = trace.get_trace("t1")
print(spans[0].name, spans[0].attributes)
```

Every JSON-shaped field (`metadata`, `attributes`, `data`,
`annotations`, `notes`, `input`, `output`, `schema`) round-trips
transparently to native Python dicts/lists.

## ToolCache: semantic-near-hit demo

```python
cache = duxxdb.ToolCache(threshold=0.95)

# Cache the result of an expensive web_search call.
cache.put(tool="web_search", args_hash=hash("what is rust?"),
          args_embedding=embed("what is rust?"),
          result=b"A systems programming language ...",
          ttl_secs=600)

# Later, a paraphrased query — different hash, similar embedding.
hit = cache.get("web_search",
                args_hash=hash("describe rust"),
                args_embedding=embed("describe rust"))

if hit and hit.kind == "semantic_near_hit":
    print(f"cache hit by paraphrase, similarity={hit.similarity:.3f}")
    answer = hit.result.decode() if isinstance(hit.result, bytes) else bytes(hit.result).decode()
```

## What's missing today

- **Single-file `redb:` mode** for `--phase7-storage`. Today only
  `dir:<directory>` is supported (one redb file per primitive).
  Single-file mode requires sharing one `Arc<Database>` across all
  six backends — queued for a future v0.2.x.
- Subscriptions (`MemoryStore.subscribe()`) — Phase 4.5; the Rust /
  RESP servers already support this, the Python wrapper just needs to
  bridge `tokio::broadcast::Receiver` into a Python iterator.
- Async API — currently sync only. `async def` wrappers are planned.
- Numpy / typed-array fast path — embeddings are currently
  `list[float]`. Phase 3.5 will accept `np.ndarray[float32]` directly
  via `numpy::PyArrayLike`.

See [../../docs/ROADMAP.md](../../docs/ROADMAP.md) for the broader plan.

## License

Apache 2.0. See [../../LICENSE](../../LICENSE).

[PyO3]: https://github.com/PyO3/pyo3
[maturin]: https://github.com/PyO3/maturin
