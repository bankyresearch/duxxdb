# DuxxDB — User Guide

Practical recipes for using DuxxDB through each of its 6 integration
surfaces. If you haven't installed yet, start at
[INSTALLATION.md](INSTALLATION.md).

## Contents

1. [Mental model](#1-mental-model)
2. [Quickstart per integration surface](#2-quickstart-per-integration-surface)
   - [Embedded Rust](#21-embedded-rust)
   - [RESP TCP — redis-cli compatible](#22-resp-tcp--redis-cli-compatible)
   - [MCP stdio — Claude Desktop / Cline](#23-mcp-stdio--claude-desktop--cline)
   - [gRPC — typed cross-language streaming](#24-grpc--typed-cross-language-streaming)
   - [Python wheel](#25-python-wheel)
   - [Node / TypeScript](#26-node--typescript)
3. [Common workflows](#3-common-workflows)
   - [Store + recall agent memory](#31-store--recall-agent-memory)
   - [Reactive subscriptions](#32-reactive-subscriptions-pubsub)
   - [Tool cache with semantic-near-hit](#33-tool-cache-with-semantic-near-hit)
   - [Session KV](#34-session-kv)
   - [Cold-tier export to Parquet](#35-cold-tier-export-to-parquet)
4. [Configuration](#4-configuration)
   - [Embedders](#41-embedders)
   - [Storage modes](#42-storage-modes)
   - [Server flags](#43-server-flags)
5. [Going to production](#5-going-to-production)
6. [Backup & restore](#6-backup--restore)

---

## 1. Mental model

DuxxDB stores **memories** for AI agents. Each memory is:

| field | type | notes |
|---|---|---|
| `id` | u64 | auto-assigned, monotonic |
| `key` | string | partition / user / agent identifier |
| `text` | string | the memory content |
| `embedding` | `Vec<f32>` | client-supplied or server-embedded |
| `importance` | f32 | base 1.0; decays with age via half-life |
| `created_at_unix_ns` | u64 | wall-clock; survives restart |

**Recall** runs hybrid search (vector ANN + BM25 full-text) fused
via Reciprocal Rank Fusion, optionally re-ranked by decayed
importance.

**Reactive**: every `remember` publishes a `ChangeEvent` on a
broadcast bus, routable by glob pattern (`PSUBSCRIBE memory.*`).

**Storage**: in-memory (default), `redb:` (rows persist, indices
rebuild on open), or `dir:` (rows + tantivy + HNSW dump all
persisted, fast cold start on graceful shutdown).

**Embedder**: pluggable. Built-in `hash:<dim>` for tests/demos;
`openai:<model>` and `cohere:<model>` HTTP providers ship in the
default build.

---

## 2. Quickstart per integration surface

### 2.1 Embedded Rust

```rust
use duxx_memory::MemoryStore;
use duxx_embed::{Embedder, OpenAIEmbedder};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Pluggable embedder. HashEmbedder for tests, OpenAIEmbedder for prod.
    let embedder: Arc<dyn Embedder> =
        Arc::new(OpenAIEmbedder::small(std::env::var("OPENAI_API_KEY")?));
    let dim = embedder.dim();

    // Pick a store: in-memory for ephemeral, open_at(dir) for durable.
    let store = MemoryStore::open_at(dim, 100_000, "./data/duxx")?;

    // Remember.
    let id = store.remember(
        "alice",
        "I lost my wallet at the cafe",
        embedder.embed("I lost my wallet at the cafe")?,
    )?;
    println!("stored id={id}");

    // Recall (hybrid: vector + BM25 + RRF).
    let q = "wallet";
    let hits = store.recall("alice", q, &embedder.embed(q)?, 5)?;
    for h in hits {
        println!("{:.4}  {}", h.score, h.memory.text);
    }

    // Recall with decay reranking.
    use std::time::Duration;
    let hits = store.recall_decayed(
        "alice", q, &embedder.embed(q)?, 5,
        Duration::from_secs(7 * 24 * 3600),
    )?;
    Ok(())
}
```

### 2.2 RESP TCP — redis-cli compatible

```bash
duxx-server --addr 127.0.0.1:6379 \
            --embedder openai:text-embedding-3-small \
            --storage dir:./data/duxx
```

```bash
redis-cli -p 6379
> PING
+PONG
> REMEMBER alice I lost my wallet at the cafe
(integer) 1
> RECALL alice wallet 3
1) 1) (integer) 1
   2) "0.032787"
   3) "I lost my wallet at the cafe"
> SET sid-42 some-blob
+OK
> GET sid-42
"some-blob"
> SUBSCRIBE memory
1) "subscribe"
2) "memory"
3) (integer) 1
# (now waiting; another client's REMEMBER pushes here)
```

Any `redis-py` / `node-redis` / `go-redis` / `redis-rs` client also
just works — DuxxDB speaks RESP2/3.

### 2.3 MCP stdio — Claude Desktop / Cline

Add to your MCP server config (Claude Desktop:
`%APPDATA%\Claude\claude_desktop_config.json`, or Cline's
extension settings):

```jsonc
{
  "mcpServers": {
    "duxxdb": {
      "command": "/path/to/duxx-mcp",
      "args": [],
      "env": {
        "DUXX_EMBEDDER": "openai:text-embedding-3-small",
        "DUXX_STORAGE":  "dir:/Users/me/duxx-data",
        "OPENAI_API_KEY": "sk-…"
      }
    }
  }
}
```

Restart the agent. It now sees three tools: **`remember`**,
**`recall`**, **`stats`**. Prompt it with "remember that I prefer
short answers" and it'll call `duxxdb.remember`. Future turns
auto-recall via the agent's tool-use loop.

### 2.4 gRPC — typed cross-language streaming

```bash
duxx-grpc --addr 127.0.0.1:50051 \
          --embedder openai:text-embedding-3-small \
          --storage dir:./data/duxx
```

Schema in [`crates/duxx-grpc/proto/duxx.proto`](../crates/duxx-grpc/proto/duxx.proto).
Generate clients per language:

```bash
# Python
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I crates/duxx-grpc/proto \
    --python_out=. --grpc_python_out=. \
    crates/duxx-grpc/proto/duxx.proto
```

```python
import grpc
import duxx_pb2 as pb
import duxx_pb2_grpc as svc

ch = grpc.insecure_channel("localhost:50051")
client = svc.DuxxStub(ch)

client.Remember(pb.RememberRequest(
    key="alice", text="I lost my wallet"))
hits = client.Recall(pb.RecallRequest(
    key="alice", query="wallet", k=5))
for h in hits.hits:
    print(h.score, h.text)

# Streaming subscribe — blocks, yields ChangeEvent for every
# matching write from any other client.
for ev in client.Subscribe(pb.SubscribeRequest(pattern="memory.alice*")):
    print(ev.kind, ev.channel, ev.row_id)
```

Same generated client pattern works for Go, Java, Swift, C#, …

### 2.5 Python wheel

```python
import duxxdb

store = duxxdb.MemoryStore(dim=1536)   # match your embedder

# Bring your own embedding (e.g. OpenAI / Cohere / sentence-transformers)
def embed(text):
    # ... your provider ...
    return [...]                       # 1536-d list of floats

store.remember(key="alice", text="hello", embedding=embed("hello"))

hits = store.recall(key="alice", query="hello",
                    embedding=embed("hello"), k=5)
for h in hits:
    print(h.id, h.score, h.text)

# Tool cache with semantic-near-hit
cache = duxxdb.ToolCache(threshold=0.95)
cache.put(tool="web_search", args_hash=42,
          args_embedding=embed("what is rust?"),
          result=b"...the cached answer...",
          ttl_secs=600)

hit = cache.get("web_search",
                args_hash=999,                       # different hash
                args_embedding=embed("describe rust"))  # similar embedding
if hit and hit.kind == "semantic_near_hit":
    print(f"paraphrase hit, sim={hit.similarity:.3f}")

# Session KV
sessions = duxxdb.SessionStore(ttl_secs=1800)
sessions.put("sid-42", b"<conversation buffer>")
data = sessions.get("sid-42")        # bytes or None
```

### 2.6 Node / TypeScript

```typescript
import { MemoryStore, ToolCache, SessionStore, version } from "duxxdb";

console.log(`duxxdb v${version()}`);

const store = new MemoryStore(1536);   // match your embedder

async function embed(text: string): Promise<number[]> {
  // … your embedding provider …
  return [/* 1536 floats */];
}

await store.remember("alice", "hello", await embed("hello"));
const hits = store.recall("alice", "hello", await embed("hello"), 5);
for (const h of hits) console.log(h.id, h.score, h.text);
```

Buffers (Node `Buffer`) for `ToolCache` results and `SessionStore`
data — see [`bindings/node/README.md`](../bindings/node/README.md).

---

## 3. Common workflows

### 3.1 Store + recall agent memory

End-to-end agent turn through DuxxDB-gRPC (Python client):

```python
def agent_turn(user_id: str, user_msg: str) -> str:
    # 1. Remember the new turn.
    client.Remember(pb.RememberRequest(
        key=user_id, text=user_msg, embedding=embed(user_msg)))

    # 2. Recall context for the LLM prompt.
    ctx = client.Recall(pb.RecallRequest(
        key=user_id, query=user_msg, embedding=embed(user_msg), k=10))
    prompt_context = "\n".join(h.text for h in ctx.hits)

    # 3. Call the LLM.
    response = llm.chat(prompt_context, user_msg)

    # 4. Remember the assistant turn too.
    client.Remember(pb.RememberRequest(
        key=user_id, text=response, embedding=embed(response)))

    return response
```

### 3.2 Reactive subscriptions (pub/sub)

Supervisor agent watching every memory created by user `alice`:

```bash
redis-cli -p 6379
> PSUBSCRIBE memory.alice
1) "psubscribe"
2) "memory.alice"
3) (integer) 1

# Now another connection writes:
#   REMEMBER alice "I'm thinking about cancelling"
# Supervisor instantly receives:
1) "pmessage"
2) "memory.alice"
3) "memory.alice"
4) "{\"table\":\"memory\",\"key\":\"alice\",\"row_id\":42,\"kind\":\"insert\"}"
```

Glob patterns: `*` matches any sequence, `?` matches one char,
`\X` is a literal escape.

In Python over gRPC:

```python
for event in client.Subscribe(pb.SubscribeRequest(pattern="memory.*")):
    if event.kind == pb.ChangeEvent.Kind.INSERT:
        on_new_memory(event.key, event.row_id)
```

### 3.3 Tool cache with semantic-near-hit

Saves real money on expensive tool calls (web search, LLM
summarization, slow DB queries) by recognizing paraphrased args:

```python
cache = duxxdb.ToolCache(threshold=0.95)

def web_search_cached(query: str) -> str:
    h = hash(query)
    e = embed(query)
    cached = cache.get("web_search", h, e)
    if cached is not None:
        return cached.result.decode()       # exact or near-hit; short-circuit

    # Cache miss → real call.
    answer = expensive_web_search(query)
    cache.put("web_search", h, e,
              answer.encode(), ttl_secs=3600)
    return answer
```

A query like "what is Rust?" then "describe Rust" hits the cache
on the second call (cosine ≥ 0.95 → `kind == "semantic_near_hit"`).

### 3.4 Session KV

Per-conversation working state with sliding TTL. Reads bump the
expiry; idle sessions auto-expire.

```python
sessions = duxxdb.SessionStore(ttl_secs=1800)   # 30 min idle

def on_message(session_id: str, msg: str):
    blob = sessions.get(session_id) or b""    # bumps last_access
    buf = (blob + b"\n" + msg.encode())[-4096:]   # tail
    sessions.put(session_id, buf)
```

Or via RESP:

```
> SET sid-42 <blob>
> GET sid-42
> DEL sid-42
```

### 3.5 Cold-tier export to Parquet

Periodic dump of the row store to Apache Parquet for analytics,
audit, or training-set generation. Spark / DuckDB / Polars / pandas
read it natively.

```bash
duxx-export --storage dir:./data/duxx \
            --out ./cold/memories-2026-05.parquet \
            --dim 1536
```

```python
import pyarrow.parquet as pq
tbl = pq.read_table("./cold/memories-2026-05.parquet")
print(tbl.schema)
# id              UInt64
# key             Utf8
# text            Utf8
# embedding       FixedSizeList<Float32, 1536>
# importance      Float32
# created_at_ns   UInt64
```

For periodic export, drive `duxx-export` from cron / systemd timer:

```cron
0 * * * *  /usr/local/bin/duxx-export --storage dir:/var/lib/duxx --out /var/cold/$(date +\%Y\%m\%d-\%H).parquet --dim 1536
```

---

## 4. Configuration

### 4.1 Embedders

| Spec | Provider | Dim | Needs |
|---|---|---|---|
| `hash:32` | built-in | 32 | — |
| `hash:128` etc. | built-in | any | — |
| `openai:text-embedding-3-small` | OpenAI HTTP | 1536 | `OPENAI_API_KEY` |
| `openai:text-embedding-3-large` | OpenAI HTTP | 3072 | `OPENAI_API_KEY` |
| `openai:text-embedding-ada-002` | OpenAI HTTP | 1536 | `OPENAI_API_KEY` |
| `cohere:embed-english-v3.0` | Cohere HTTP | 1024 | `COHERE_API_KEY` |
| `cohere:embed-english-light-v3.0` | Cohere HTTP | 384 | `COHERE_API_KEY` |
| `cohere:embed-multilingual-v3.0` | Cohere HTTP | 1024 | `COHERE_API_KEY` |

Set via:

- CLI flag: `--embedder openai:text-embedding-3-small`
- Env var: `DUXX_EMBEDDER=openai:text-embedding-3-small`

Plug your own in Rust:

```rust
struct MyEmbedder;
impl duxx_embed::Embedder for MyEmbedder {
    fn embed(&self, text: &str) -> duxx_core::Result<Vec<f32>> { /* … */ }
    fn dim(&self) -> usize { 768 }
}
let server = Server::with_provider(Arc::new(MyEmbedder));
```

### 4.2 Storage modes

| Spec | Durability | Indices on disk? | Cold-start cost | When to pick |
|---|---|---|---|---|
| _(none)_ | None — in-memory | No | 0 | Tests, demos, ephemeral agents |
| `redb:./path/db.redb` | Rows ACID | No (rebuilt on open) | ~250 µs / row | Small corpus, simple deploys |
| `dir:./path` | Full (rows + indices) | Yes | ~0 (graceful) / ~250 µs/row (hard kill) | Production |

Layout under `dir:`:

```
./path/
├── store.redb       row store (redb)
├── tantivy/         BM25 disk index
└── hnsw/            HNSW dump + id_map sidecar
```

Set via:

- CLI flag: `--storage dir:./data`
- Env var: `DUXX_STORAGE=dir:/var/lib/duxx`

### 4.3 Server flags

```
duxx-server [OPTIONS]
  --addr HOST:PORT          default 127.0.0.1:6379
  --embedder SPEC           default hash:32
  --storage SPEC            default in-memory only
  --token TOKEN             require AUTH <TOKEN> (default: no auth)
  --drain-secs N            graceful-shutdown drain window (default 30)
  --metrics-addr HOST:PORT  Prometheus + /health endpoint (default: disabled)

duxx-grpc [OPTIONS]
  --addr HOST:PORT          default 127.0.0.1:50051
  --embedder SPEC           default hash:32
  --storage SPEC            default in-memory only
  --token TOKEN             require Bearer TOKEN (default: no auth)
  # grpc.health.v1.Health is always served (no auth required)

duxx-mcp                    # stdio JSON-RPC; uses DUXX_EMBEDDER + DUXX_STORAGE env

duxx-export
  --storage SPEC            required
  --out PATH                required Parquet path
  --dim N                   default 32
```

**Auth semantics:**
- RESP: `AUTH <token>` (Redis-compatible). Pre-auth, only `PING` /
  `AUTH` / `QUIT` / `HELLO` work; everything else returns `NOAUTH`.
  Wrong tokens return `WRONGPASS`.
- gRPC: send `authorization: Bearer <token>` metadata on every
  request. Missing or wrong token returns `UNAUTHENTICATED`.
  Health protocol bypasses auth so probes work.
- Tokens are compared in constant time to defeat timing
  side-channel attacks.

Env vars override defaults but lose to explicit CLI flags:
`DUXX_EMBEDDER` / `DUXX_STORAGE` / `DUXX_TOKEN` / `DUXX_METRICS_ADDR`.

---

## 5. Going to production

| Concern | Status | Recommendation |
|---|---|---|
| Durability | ✅ via `--storage dir:` | Pin `--storage dir:...`; hard-kill safe via cold-path rebuild |
| Cross-restart decay | ✅ Phase 2.4 | Set per-store half-life via `recall_decayed` in code |
| **Auth** | ✅ Phase 6.1 — token-based | Set `--token` / `DUXX_TOKEN` (≥ 16 chars). Clients issue `AUTH <token>` (RESP) or `authorization: Bearer <token>` (gRPC). |
| **Health probes** | ✅ Phase 6.1 | gRPC: `grpc.health.v1.Health/Check`. HTTP: `GET /health` on the metrics port. |
| **Prometheus metrics** | ✅ Phase 6.1 | `--metrics-addr 0.0.0.0:9091` exposes `/metrics` (text format) + `/health` |
| **Graceful shutdown** | ✅ Phase 6.1 | SIGINT/SIGTERM stops accepting new connections, drains for `--drain-secs` (default 30) before exiting. Rolling-deploy safe. |
| TLS | ⚠ not yet — terminate at LB | Put nginx / Envoy / a service mesh in front; bind DuxxDB to `127.0.0.1`. |
| Multi-tenancy | ⚠ no isolation in process | One `--storage dir:./tenant-X` daemon per tenant for now |
| RBAC | ✗ | Deferred to Phase 6.2 |
| Sharding / replication | ✗ | Single node; replicate at the orchestration layer (Kubernetes) for now |
| Backups | ✅ Parquet export | See [§ 3.5](#35-cold-tier-export-to-parquet) and [§ 6](#6-backup--restore) below |

### Production startup recipe

```bash
duxx-server \
  --addr 0.0.0.0:6379 \
  --storage dir:/var/lib/duxx \
  --embedder openai:text-embedding-3-small \
  --token "$(cat /etc/duxx/token)" \
  --metrics-addr 127.0.0.1:9091 \
  --drain-secs 60
```

```bash
duxx-grpc \
  --addr 0.0.0.0:50051 \
  --storage dir:/var/lib/duxx \
  --embedder openai:text-embedding-3-small \
  --token "$(cat /etc/duxx/token)"
```

K8s liveness probe:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 9091
  initialDelaySeconds: 5
  periodSeconds: 10

# OR for gRPC:
readinessProbe:
  grpc:
    port: 50051
  periodSeconds: 5
```

---

## 6. Backup & restore

### Backup — periodic Parquet dump

```cron
# /etc/cron.hourly/duxx-backup
0 * * * *  /usr/local/bin/duxx-export \
  --storage dir:/var/lib/duxx \
  --out /var/cold/$(date +\%Y\%m\%d-\%H).parquet \
  --dim 1536
```

Uploads to S3 / GCS / Azure are a follow-on `aws s3 cp …` — treat
the parquet file as any other artifact.

### Restore — from Parquet back into DuxxDB

There's no auto-import yet (Phase 6.2). Until then, write a small
Rust program against `MemoryStore` that reads the Parquet file via
the `parquet` crate and calls `remember()` for each row:

```rust
use duxx_memory::{MemoryStore, Memory};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let store = MemoryStore::open_at(1536, 100_000, "/var/lib/duxx")?;
    let file = File::open("/var/cold/snapshot.parquet")?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    for batch in reader {
        let batch = batch?;
        // … decode columns, call store.remember(...) for each row …
    }
    Ok(())
}
```

The `bench/comparative/bench.py` harness has a similar pattern in
Python via `pyarrow` — copy that loop and point it at your file.

### Disaster recovery posture

| Failure | Recovery |
|---|---|
| Server crashed (panic / SIGKILL) | Restart — auto-rebuilds tantivy + HNSW from `redb` rows. Sub-minute for 100k memories; minutes for 1M. |
| Disk corruption on `redb` | Restore from latest Parquet dump (hourly). |
| Disk lost entirely | Restore from latest Parquet dump in object storage. |
| Single bad memory | Manually `DEL` via the row store, or `--remove` flag (Phase 6.2). |

For Closed UAT: bind to localhost, use `dir:` storage, set a token,
configure your embedder. For Open UAT: add the `--metrics-addr`
endpoint, put nginx/mTLS in front, and you're set. Production
deployments need Phase 6.2 (RBAC, multi-tenant isolation,
distributed mode) — see [ROADMAP.md](ROADMAP.md).

---

## Next

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design, data model, query plan
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — capabilities matrix + competitive landscape
- [ROADMAP.md](ROADMAP.md) — what's shipped, what's next
