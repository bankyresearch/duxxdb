# DuxxDB — Closed UAT Guide

Welcome. You're an early DuxxDB user. This guide tells you:

1. **What works today.** ✅
2. **What's missing and how to plan around it.** ⚠
3. **How to install, configure, and integrate.** 🚀
4. **What signals we want from your testing.** 📡

> **Status:** pre-alpha (v0.1). Closed UAT means: **don't put production
> data in this database**, and **don't expect data to survive a
> restart** until Phase 2.3 (Lance disk persistence) lands.

---

## 1. What works today ✅

| Feature | How to drive it |
|---|---|
| Embedded Rust library | `duxx-memory` crate; see `crates/duxx-cli/examples/chatbot_memory.rs` |
| MCP stdio server | `duxx-mcp` binary — Claude Desktop / Cline plug-in |
| RESP TCP server | `duxx-server` — `valkey-cli` / `redis-cli` compatible |
| RESP pub/sub | `SUBSCRIBE memory` — pushed live on every `REMEMBER` |
| Python wheel | `pip install duxxdb-0.1.0-cp38-abi3-*.whl` |
| Real embedders | OpenAI / Cohere via `--embedder` flag or `DUXX_EMBEDDER` env |
| Hybrid recall | tantivy BM25 + hnsw_rs HNSW, RRF-fused |
| Importance decay | `MemoryStore::recall_decayed(half_life)` |
| Tool cache | exact + semantic-near-hit (cosine ≥ 0.95) |
| Session KV | sliding-TTL store, RESP `SET`/`GET`/`DEL` |

**Measured (Phase 2.5, single thread, debug release):**
- recall @ 100 / 1 k / 10 k docs: **123 / 166 / 373 µs** (target was < 10 ms)
- bulk insert @ 1 k docs: **342 ms** (~342 µs/insert)

---

## 2. Known limitations ⚠

### ✅ Memories ARE durable when `--storage` is set

`duxx-server --storage redb:./data/duxx.redb` makes every `REMEMBER`
write through to a [redb] file on disk. Restarting the process
reloads everything from that file before accepting new connections.

Without `--storage` (the default), state is in-memory only and lost on
exit — fine for ephemeral demos or stateless agents.

**Still in-memory regardless of `--storage`:** `SessionStore` and
`ToolCache`. Their TTLs are usually shorter than process uptime so
this is fine; durability lands in a follow-up.

[redb]: https://github.com/cberner/redb

### Other current gaps

| Gap | Workaround |
|---|---|
| No authentication on RESP / MCP | Bind to `127.0.0.1` only. Don't expose to the public internet. |
| No multi-tenant isolation | One process per tenant for now. |
| No `PSUBSCRIBE` (pattern subscribe) | Subscribe to the literal `memory` channel and filter client-side. |
| No async Python API | Sync only. Wrap in `asyncio.to_thread` if needed. |
| No numpy fast path | Python embeddings are `list[float]`. Will accept `np.ndarray[float32]` in 3.5. |
| No TypeScript bindings | Use the RESP TCP server from a Node redis client; PR 3.4 in flight. |
| Single-node | No sharding or replication yet. |

---

## 3. Install, configure, integrate 🚀

### Option A — Docker (one command)

```bash
docker run --rm -p 6379:6379 duxxdb:0.1.0
# Then:
redis-cli -p 6379
> REMEMBER alice "I lost my wallet"
(integer) 1
```

With OpenAI embeddings + persistent storage:

```bash
mkdir -p ./duxx-data
docker run --rm -p 6379:6379 \
  -v "$PWD/duxx-data:/data" \
  -e DUXX_EMBEDDER=openai:text-embedding-3-small \
  -e DUXX_STORAGE=redb:/data/duxx.redb \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  duxxdb:0.1.0
```

Now memories survive container restart.

### Option B — Python wheel

```bash
cd bindings/python
maturin build --release
pip install --user --force-reinstall ../../target/wheels/duxxdb-*.whl
```

```python
import duxxdb
store = duxxdb.MemoryStore(dim=4)
store.remember(key="alice", text="hi", embedding=[1.0, 0.0, 0.0, 0.0])
hits = store.recall(key="alice", query="hi",
                    embedding=[1.0, 0.0, 0.0, 0.0], k=5)
```

### Option C — Source build

See [SETUP.md](SETUP.md) for the full toolchain (Rust + WinLibs MinGW
on Windows). Then:

```bash
scripts/build.sh test --workspace        # 60+ tests
scripts/build.sh run -p duxx-cli --example chatbot_memory
```

---

## Embedder options

| Spec | Provider | Notes |
|---|---|---|
| `hash:<dim>` | built-in | Deterministic toy. Default `hash:32`. Use only for demos / tests. |
| `openai:text-embedding-3-small` | OpenAI HTTP | 1536-d. Needs `OPENAI_API_KEY`. |
| `openai:text-embedding-3-large` | OpenAI HTTP | 3072-d. Higher quality, costlier. |
| `cohere:embed-english-v3.0` | Cohere HTTP | 1024-d. Needs `COHERE_API_KEY`. |
| `cohere:embed-english-light-v3.0` | Cohere HTTP | 384-d. Smaller / cheaper. |

Set via:
- CLI flag: `duxx-server --embedder openai:text-embedding-3-small`
- Env var: `DUXX_EMBEDDER=openai:text-embedding-3-small`

---

## Integration recipes

### Claude Desktop / Cline (MCP)

Add to your MCP server config:

```jsonc
{
  "mcpServers": {
    "duxxdb": {
      "command": "/path/to/duxx-mcp",
      "env": {
        "DUXX_EMBEDDER": "openai:text-embedding-3-small",
        "OPENAI_API_KEY": "sk-…"
      }
    }
  }
}
```

The agent now has `remember`, `recall`, and `stats` tools.

### Wiring DuxxDB into a Python agent framework

Most Python agent frameworks let you supply a custom memory class.
The shape is the same regardless of framework: hold a
`duxxdb.MemoryStore`, embed text on save, hand back recall results
on load.

```python
import duxxdb

class DuxxMemory:
    def __init__(self, key: str, dim: int = 1536):
        self.key = key
        self.store = duxxdb.MemoryStore(dim=dim)
        # ... wire up your embedder of choice ...
    # implement your framework's save / load hooks against self.store
```

The simpler route — when the framework already speaks Redis — is
to point its built-in memory adapter at `duxx-server` over RESP.
See INTEGRATION_GUIDE.md for the recipe.

### Reactive supervisor agent

```python
import socket
s = socket.socket(); s.connect(('127.0.0.1', 6379))
s.sendall(b'SUBSCRIBE memory\r\n')
# now stream every memory write into your supervisor
```

---

## 4. What we want from you 📡

**Please report:**

- Crashes / panics — copy the stderr line, please.
- Recall hits that feel obviously wrong (top-k contains a clearly
  unrelated memory). Include the query, the corpus, and the embedder
  spec.
- Performance regressions — `target/criterion/report/index.html` after
  `scripts/build.sh bench -p duxx-bench` shows the latest numbers.
- Any API friction — function signatures, naming, error messages —
  this is the cheapest time to fix it.

**File reports as:**

- GitHub issue with `[UAT]` prefix.
- Or, for high-frequency stuff, ping the maintainers on the
  early-tester channel.

**You don't need to report:**

- Anything in the [Known limitations](#2-known-limitations-) table.
- Lost data after a restart (expected — Phase 2.3 will fix).
- Things mentioned in [ROADMAP.md](ROADMAP.md) as future phases.

---

## License

Apache 2.0. See [../LICENSE](../LICENSE).
