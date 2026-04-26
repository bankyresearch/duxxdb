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

| Class | Constructor | Methods |
|---|---|---|
| `MemoryStore` | `dim`, `capacity=100_000` | `remember(key, text, embedding) -> id`, `recall(key, query, embedding, k=10) -> [MemoryHit]`, `len()`, `dim` |
| `MemoryHit` | (returned by `recall`) | `id`, `key`, `text`, `score` |
| `ToolCache` | `threshold=0.95` | `put(tool, args_hash, args_embedding, result, ttl_secs=3600)`, `get(tool, args_hash, args_embedding) -> ToolCacheHit \| None`, `purge_expired()`, `len()` |
| `ToolCacheHit` | (returned by `get`) | `kind` (`"exact"` or `"semantic_near_hit"`), `similarity`, `result` |
| `SessionStore` | `ttl_secs=1800` | `put(session_id, data)`, `get(session_id) -> bytes \| None`, `delete(session_id) -> bool`, `purge_expired()`, `len()` |

Module-level: `duxxdb.__version__`.

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
