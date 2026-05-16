<div align="center">

# DuxxDB

**The database built for AI agents.**

Hybrid retrieval (vector + BM25 + structured), agent-native primitives
(`MEMORY` / `TOOL_CACHE` / `SESSION`), and embedded-or-server deployment
from one Rust codebase. Speaks RESP2/3, gRPC, and MCP out of the box.

[![CI](https://github.com/bankyresearch/duxxdb/actions/workflows/ci.yml/badge.svg)](https://github.com/bankyresearch/duxxdb/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Made with Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-112%20passing-success)](https://github.com/bankyresearch/duxxdb/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-Public%20Ready-brightgreen)](docs/PROJECT_OVERVIEW.md)
[![Docs](https://img.shields.io/badge/docs-7%20guides-blue)](docs/)

[**Install**](#install) · [**Quickstart**](#quickstart) · [**Why**](#why-duxxdb) · [**Features**](#features) · [**Integrate**](docs/INTEGRATION_GUIDE.md) · [**FAQ**](docs/FAQ.md) · [**Architecture**](#architecture) · [**Benchmarks**](#benchmarks) · [**Roadmap**](docs/ROADMAP.md) · [**Contribute**](CONTRIBUTING.md)

</div>

---

## TL;DR

```bash
docker run -d --name duxxdb -p 6379:6379 ghcr.io/bankyresearch/duxxdb:latest

redis-cli -p 6379 REMEMBER alice "I lost my wallet at the cafe"
redis-cli -p 6379 RECALL    alice "wallet" 3
# 1) 1) (integer) 1
#    2) "0.032787"
#    3) "I lost my wallet at the cafe"
```

DuxxDB just spoke RESP — `redis-cli`, `valkey-cli`, `redis-rs`,
`node-redis`, `go-redis`, all work unchanged. Now switch to the
[Quickstart](#quickstart) for Python, gRPC, or MCP.

> **Status:** Public-ready — feature-complete through Phase 6.2
> (auth · native TLS · health · Prometheus · graceful shutdown ·
> importance-based eviction). 112 tests green on Linux + macOS +
> Windows. See the [Roadmap](docs/ROADMAP.md).

---

## The Duxx Stack

DuxxDB is the **storage engine** half. The **framework** half is
[**duxx-ai**](https://github.com/bankyresearch/duxx-ai) — an
Apache 2.0 Python SDK for building, fine-tuning, orchestrating, and
governing AI agents:

```
   ┌─ duxx-ai (Python) ────────────────────────────────────┐
   │  agents · graph · crew · tools · memory · guardrails  │
   │  RBAC · observability · fine-tune · adaptive routing  │
   └────────────────────┬──────────────────────────────────┘
                        │ uses (memory / traces / prompts / eval / cost)
   ┌────────────────────▼──────────────────────────────────┐
   │  DuxxDB (Rust) — storage + retrieval + observability  │
   │  hybrid recall · TLS · MCP · gRPC · RESP · Parquet    │
   └───────────────────────────────────────────────────────┘
```

| You're building... | Reach for |
|---|---|
| The agent itself (Python) | [`duxx-ai`](https://pypi.org/project/duxx-ai/) — agents, crews, tools, governance |
| The storage / retrieval layer | **`duxxdb`** (this repo) — single binary, six client surfaces |
| Both together | `pip install duxx-ai duxxdb` — the full Apache 2.0 agent stack |

**Why one stack, two projects?** Rust gets you sub-ms hybrid recall +
TLS + multi-platform binaries. Python gets you the agent-developer
ergonomics — typed primitives, LLM SDKs, tools, governance — without
dragging C++ deployment headaches into the storage tier. Each side
plays to its strengths; they ship independently with semver.

---

## Why DuxxDB?

A typical agent stack today glues three databases together:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Redis   │    │ Qdrant   │    │  Postgres    │
│  session │    │  vectors │    │  facts/users │
│  ~0.5ms  │    │  ~10ms   │    │  ~1-5ms      │
└────┬─────┘    └────┬─────┘    └──────┬───────┘
     ▼               ▼                 ▼
            3 round-trips per turn
            3 client libs · 3 sets of credentials
            eventual consistency between them
```

The tax shows up in three places:

| Tax | Cost |
|---|---|
| **Latency** | 3 hops = 300 µs–3 ms before any work. Voice bots blow their 200 ms budget. |
| **Consistency** | A user message, its embedding, and its session metadata land in three stores at three times. |
| **Glue code** | Cross-store transactions, retries, drift — the silent killer of agent reliability. |

**DuxxDB collapses the three into one engine** with first-class agent
primitives, a single hybrid query plan (vector ANN + BM25 + structured
filters fused via Reciprocal Rank Fusion), and six integration surfaces
out of one codebase. Pure Rust, no GC pauses.

---

## Features

<table>
<tr><td valign="top" width="50%">

#### Hybrid retrieval
- HNSW vector ANN ([`hnsw_rs`](https://crates.io/crates/hnsw_rs))
- BM25 full-text ([`tantivy`](https://github.com/quickwit-oss/tantivy))
- Structured filters
- **All three fused in one query plan** (RRF, k=60)

#### Agent-native primitives
- `MEMORY` — sticky long-term recall with **importance decay**
- `TOOL_CACHE` — exact + **semantic-near-hit** lookup (cosine ≥ 0.95)
- `SESSION` — sliding-TTL KV with lazy eviction
- All first-class types — not generic tables

</td><td valign="top">

#### Six integration surfaces
- **Embedded** Rust crate
- **Python** wheel (PyO3, abi3-py38, one wheel for 3.8–3.13)
- **Node / TypeScript** native module (napi-rs v2)
- **RESP2/3** TCP server (Valkey/Redis-compatible)
- **gRPC** with streaming `Subscribe` (tonic)
- **MCP stdio** for Claude / GPT / Cline agents

#### Production hardening (Phase 6.1 + 6.2)
- Token auth (`--token` / `DUXX_TOKEN`)
- **Native TLS** on RESP + gRPC (rustls)
- Prometheus `/metrics` + `/health`
- gRPC `grpc.health.v1.Health` protocol
- Graceful shutdown drain
- **Memory cap + importance-based eviction**
- Apache Parquet cold-tier export

</td></tr>
</table>

---

## Install

Postgres-style — pick your platform.

| Platform | Command | Details |
|---|---|---|
| 🐳 **Docker** (any OS) | `docker run -p 6379:6379 ghcr.io/bankyresearch/duxxdb:latest` | [Docker guide](docs/INSTALLATION.md#docker-single-command) |
| 🐳 **Docker Compose** (production) | `cd packaging/docker && cp .env.example .env && docker compose up -d` | [Compose guide](docs/INSTALLATION.md#docker-compose-production-grade) |
| 🐧 **Debian / Ubuntu** | `sudo apt install ./duxxdb_*.deb` | [.deb install](docs/INSTALLATION.md#linux--debian--ubuntu-apt-deb) |
| 🐧 **RHEL / Fedora / Rocky** | `curl -fsSL …/install.sh \| sudo sh` + systemd unit | [.rpm install](docs/INSTALLATION.md#linux--rhel--fedora--rocky--alma-dnf-rpm) |
| 🍏 **macOS** | `brew install bankyresearch/duxxdb/duxxdb` | [Homebrew](docs/INSTALLATION.md#macos--homebrew) |
| 🐚 **One-line installer** | `curl -fsSL …/install.sh \| sh` | [Installer](docs/INSTALLATION.md#linux--macos--one-line-installer) |
| 🪟 **Windows** | zip from [Releases](https://github.com/bankyresearch/duxxdb/releases) | [Windows guide](docs/INSTALLATION.md#windows--zip-download-manual) |
| ☸ **Kubernetes** | `kubectl apply -f packaging/k8s/duxxdb.yaml` | [k8s guide](docs/INSTALLATION.md#kubernetes) |
| 🐍 **Python** | `pip install …` (locally built wheel) | [Python](docs/INSTALLATION.md#embed--python-wheel) |
| 📦 **Node / TS** | `npm install …` (locally built `.node`) | [Node](docs/INSTALLATION.md#embed--node--typescript) |
| 🦀 **Rust crate** | `cargo add duxx-memory --git …` | [Cargo](docs/INSTALLATION.md#embed--rust-crate) |
| 📐 **From source** | `cargo build --release --workspace` | [Source](docs/INSTALLATION.md#from-source-any-os) |

→ Full per-platform install + uninstall + service-control cheatsheet:
**[docs/INSTALLATION.md](docs/INSTALLATION.md)**.

---

## Quickstart

#### From a redis client (any language)

```bash
redis-cli -p 6379

127.0.0.1:6379> REMEMBER alice "I want a refund for order #9910"
(integer) 1
127.0.0.1:6379> RECALL alice "refund" 5
1) 1) (integer) 1
   2) "0.032787"
   3) "I want a refund for order #9910"
127.0.0.1:6379> PSUBSCRIBE memory.*
```

#### From Python

```python
import duxxdb

store = duxxdb.MemoryStore(dim=32)
store.remember(key="alice", text="I lost my wallet")
hits = store.recall(key="alice", query="wallet", k=5)
for hit in hits:
    print(hit.id, hit.score, hit.text)
```

#### From Node / TypeScript

```ts
import { MemoryStore } from "duxxdb";

const store = new MemoryStore({ dim: 32 });
store.remember("alice", "I lost my wallet");
console.log(store.recall("alice", "wallet", 5));
```

#### Embedded in a Rust app

```rust
use duxx_memory::MemoryStore;
use duxx_embed::HashEmbedder;
use std::sync::Arc;

let store = MemoryStore::new(Arc::new(HashEmbedder::new(32)));
store.remember("alice", "I lost my wallet", None)?;
let hits = store.recall("alice", "wallet", 5)?;
```

#### Over gRPC (any tonic / grpcio / grpc-go client)

```bash
grpcurl -plaintext -d '{"key":"alice","text":"I lost my wallet"}' \
  localhost:50051 duxx.v1.Duxx/Remember

grpcurl -plaintext -d '{"key":"alice","query":"wallet","k":5}' \
  localhost:50051 duxx.v1.Duxx/Recall
```

#### Over MCP (Claude Desktop / Cline / any MCP agent)

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "duxxdb": { "command": "/usr/local/bin/duxx-mcp" }
  }
}
```

→ Full per-surface walkthrough: **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.
→ Wiring into a real agent (chatbot / voice bot / migration): **[docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)**.

---

## Architecture

```
              AGENT (chatbot / voice bot / autonomous)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
    embedded                                ┌───────────────┐
    duxx-memory (lib)                       │ duxx-server   │ RESP TCP
        │                                   ├───────────────┤
        ▼                                   │ duxx-grpc     │ gRPC + streaming
    Python / Node bindings                  ├───────────────┤
                                            │ duxx-mcp      │ stdio JSON-RPC
                                            └───────┬───────┘
                                                    │
                            ┌───────────────────────┼──────────────────┐
                            │                       │                  │
                            ▼                       ▼                  ▼
                      duxx-memory            duxx-reactive       duxx-coldtier
                      MEMORY / TOOL_CACHE     (ChangeBus)         (Parquet)
                      / SESSION + decay
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         duxx-query   duxx-index    duxx-storage   duxx-embed
         (RRF +       tantivy +     redb /         hash / OpenAI /
          hybrid)     hnsw_rs       memory         Cohere
```

12 Rust crates, all Apache 2.0. Full design: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Benchmarks

Hybrid recall, k = 10, single thread, debug release, toy 32-dim
embedder. Reproduce: `cargo bench -p duxx-bench`.

| Corpus | Median recall | Throughput | vs 10 ms target |
|---:|---:|---:|---:|
| 100 docs | **123 µs** | 8.1 k QPS | 81× headroom |
| 1 000 docs | **166 µs** | 6.0 k QPS | 60× headroom |
| 10 000 docs | **373 µs** | 2.7 k QPS | 27× headroom |

Even over a localhost gRPC round-trip (Python client, dim 128, N=1k):
recall p50 = **2.4 ms** — **30× faster** than embedded LanceDB on the
same workload. Full numbers + caveats:
[`bench/comparative/README.md`](bench/comparative/README.md).

---

## How DuxxDB compares

|                       | Vector | BM25 | Structured | Embedded | Server | Reactive | Agent prims | MCP |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Redis**             | ✗    | ✗    | ⚠         | ✗       | ✅      | ✅       | ✗          | ✗  |
| **Valkey**            | ⚠ (mod) | ⚠ (mod) | ⚠ | ✗      | ✅      | ✅       | ✗          | ✗  |
| **Qdrant**            | ✅    | ⚠    | ⚠         | ✗       | ✅      | ⚠       | ✗          | ✗  |
| **Pinecone**          | ✅    | ✗    | ⚠         | ✗       | ✅      | ✗       | ✗          | ✗  |
| **Milvus**            | ✅    | ⚠    | ✅         | ✗       | ✅      | ✗       | ✗          | ✗  |
| **Weaviate**          | ✅    | ✅    | ✅         | ✗       | ✅      | ✗       | ✗          | ✗  |
| **pgvector**          | ✅    | ⚠    | ✅         | ✗       | ✅      | ⚠       | ✗          | ✗  |
| **LanceDB**           | ✅    | ⚠    | ✅         | ✅       | ⚠       | ✗       | ✗          | ✗  |
| **ChromaDB**          | ✅    | ✗    | ⚠         | ✅       | ✅      | ✗       | ⚠          | ✗  |
| **DuxxDB**            | ✅    | ✅    | ✅         | ✅       | ✅      | ✅       | ✅          | ✅  |

Full take with "when each one is the right choice" + coexistence
patterns: [docs/PROJECT_OVERVIEW.md § 8](docs/PROJECT_OVERVIEW.md#8-competitive-landscape).

---

## Documentation

| Doc | Use it when |
|---|---|
| [INSTALLATION.md](docs/INSTALLATION.md) | Picking install method per OS |
| [USER_GUIDE.md](docs/USER_GUIDE.md) | Writing client code in any of 6 languages |
| [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) | Wiring DuxxDB into a chatbot, voice bot, or autonomous agent (with diagrams) |
| [DUXX_STACK_INTEGRATION.md](docs/DUXX_STACK_INTEGRATION.md) | How **duxx-ai** (Python framework) plugs into DuxxDB — backend protocol, API contracts, deployment shapes |
| [FAQ.md](docs/FAQ.md) | "Is DuxxDB right for me?" — when to use, when not to, framework support, migration |
| [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Pitching DuxxDB to a teammate / making a build-vs-buy call |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Understanding the internals before contributing |
| [ROADMAP.md](docs/ROADMAP.md) | Knowing what's shipped vs. planned |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Opening your first PR |

---

## Repository layout

```
duxxdb/
├── Cargo.toml                 workspace root (12 crates)
├── Dockerfile                 multi-stage, multi-arch image
├── packaging/                 Postgres-style installers
│   ├── docker/                docker-compose.yml + .env.example + prometheus.yml
│   ├── systemd/               duxxdb.service + duxx.env
│   ├── debian/                .deb postinst / prerm / postrm
│   ├── homebrew/              duxxdb.rb formula
│   ├── k8s/                   StatefulSet + Service + ConfigMap
│   └── scripts/install.sh     curl-pipe installer
├── crates/
│   ├── duxx-core/             Schema · Value · Error
│   ├── duxx-storage/          Storage trait + redb / memory backends
│   ├── duxx-index/            tantivy BM25 + hnsw_rs HNSW
│   ├── duxx-query/            RRF, hybrid recall
│   ├── duxx-memory/           MEMORY / TOOL_CACHE / SESSION + decay
│   ├── duxx-reactive/         ChangeBus pub/sub
│   ├── duxx-embed/            hash + OpenAI + Cohere embedders
│   ├── duxx-server/           RESP2/3 daemon + auth + Prometheus
│   ├── duxx-mcp/              MCP stdio JSON-RPC server
│   ├── duxx-grpc/             tonic gRPC daemon (streaming Subscribe)
│   ├── duxx-coldtier/         Apache Parquet exporter
│   ├── duxx-cli/              shell + chatbot_memory example
│   └── duxx-bench/            Criterion benchmarks
├── bindings/
│   ├── python/                PyO3 + maturin abi3 wheel
│   └── node/                  napi-rs v2 (workspace-excluded)
├── bench/comparative/         DuxxDB vs LanceDB / Redis / Qdrant / pgvector
└── docs/                      installation · user guide · architecture · …
```

---

## Contributing

We love PRs. Three principles:

1. **Pick an open box** in [ROADMAP.md](docs/ROADMAP.md) — or anything
   in the [issue tracker](https://github.com/bankyresearch/duxxdb/issues)
   labelled `good first issue` / `help wanted`.
2. **Tests + clippy stay green** (`cargo test --workspace` + `cargo clippy --workspace -- -D warnings`).
3. **Don't regress benchmarks by > 5 %** (`cargo bench -p duxx-bench`).

Full guide, including how to run the test matrix locally, the
commit-message style, and the PR review process:
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

---

## Security

Please report vulnerabilities **privately** via the process in
[SECURITY.md](SECURITY.md) — not through public issues.

---

## License

[Apache License 2.0](LICENSE). No proprietary "open core" tax.

Standing on the shoulders of:
[tantivy](https://github.com/quickwit-oss/tantivy),
[hnsw_rs](https://crates.io/crates/hnsw_rs),
[redb](https://github.com/cberner/redb),
[tonic](https://github.com/hyperium/tonic),
[PyO3](https://github.com/PyO3/pyo3),
[napi-rs](https://github.com/napi-rs/napi-rs),
[Apache Arrow](https://github.com/apache/arrow-rs),
[rustls](https://github.com/rustls/rustls),
the [Model Context Protocol](https://modelcontextprotocol.io/),
and many more — see [NOTICE](NOTICE).

If DuxxDB is useful to you, please support its upstream dependencies first.

---

<div align="center">

**[⬆ Back to top](#duxxdb)**

Built by [Bankatesh Choudhary](https://github.com/bankyresearch)
</div>
