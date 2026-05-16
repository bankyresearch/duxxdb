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
| `PromptRegistry` | `dim=32`, `storage=None` | `put(name, content, metadata=None) -> version`, `get(name, version_or_tag=None) -> Prompt \| None`, `list(name) -> [Prompt]`, `names() -> [str]`, `tag(name, version, tag)`, `untag(name, tag) -> bool`, `delete(name, version) -> bool`, `search(query, k=10) -> [PromptHit]`, `diff(name, v_a, v_b) -> str` |
| `Prompt` | (returned by `get` / `list`) | `name`, `version`, `content`, `tags`, `metadata` (decoded JSON), `created_at_unix_ns` |
| `PromptHit` | (returned by `search`) | `prompt`, `score` |

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

- The other five Phase 7 primitives (`TraceStore`, `DatasetRegistry`,
  `EvalRegistry`, `ReplayRegistry`, `CostLedger`) — sequenced in
  [`docs/V0_2_0_PLAN.md`](../../docs/V0_2_0_PLAN.md). Today they're
  reachable via the RESP facade (`duxxdb.server.ServerClient`); native
  bindings land progressively through v0.2.x.
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
