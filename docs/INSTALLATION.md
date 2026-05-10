# DuxxDB — Installation

Pick the install path that matches how you'll use DuxxDB.

| You want to … | Install via |
|---|---|
| Drop in a `redis-cli`-compatible server | [Docker](#install-docker) or [Build from source](#install-source) |
| Use it from Python | [Python wheel](#install-python) |
| Use it from Node / TypeScript | [Node bindings](#install-node) |
| Embed it in a Rust app | [Cargo dependency](#install-rust) |
| Connect from Claude Desktop / Cline (MCP) | [Build from source](#install-source) → [User Guide § MCP](USER_GUIDE.md#claude-desktop--cline-mcp) |

---

## Prerequisites

| Platform | Required |
|---|---|
| Linux (Debian/Ubuntu) | `build-essential pkg-config` |
| macOS | Xcode Command Line Tools (`xcode-select --install`) |
| Windows (preferred — GNU toolchain) | Git for Windows + WinLibs MinGW (no admin) |
| Windows (alt — MSVC) | Visual Studio 2022 Build Tools with C++ workload + Windows 11 SDK |

Plus, for source builds:
- Rust ≥ 1.75 (CI runs on stable; we tested 1.95)
- ≥ 4 GB free disk for build artifacts
- `protoc` (only if building `duxx-grpc`; auto-skipped otherwise)

---

## Install — Docker

Fastest path to a running server. Uses the multi-stage Dockerfile in
the repo root.

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb

# Build the image (~5 min on first build)
docker build -t duxxdb:0.1.0 .

# Run with persistent storage mounted in
mkdir -p ./data
docker run --rm -p 6379:6379 \
  -v "$PWD/data:/data" \
  -e DUXX_STORAGE=redb:/data/duxx.redb \
  duxxdb:0.1.0
```

With OpenAI embeddings:

```bash
docker run --rm -p 6379:6379 \
  -v "$PWD/data:/data" \
  -e DUXX_EMBEDDER=openai:text-embedding-3-small \
  -e DUXX_STORAGE=redb:/data/duxx.redb \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  duxxdb:0.1.0
```

Verify:

```bash
redis-cli -p 6379 PING       # +PONG
redis-cli -p 6379 INFO       # server stats
```

---

## Install — Source (Linux / macOS)

```bash
# 1. Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 2. protoc (only needed for duxx-grpc)
sudo apt install -y protobuf-compiler   # Debian/Ubuntu
brew install protobuf                   # macOS

# 3. Clone + build
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
cargo build --release --workspace

# 4. Run the daemons you care about
./target/release/duxx-server --addr 127.0.0.1:6379
./target/release/duxx-grpc   --addr 127.0.0.1:50051
./target/release/duxx-mcp     # speaks JSON-RPC on stdio
```

Verify:

```bash
cargo test --workspace        # 99 tests should pass
cargo run -p duxx-cli --example chatbot_memory
# stored 7 memories
# Top-3 recall for "refund for broken order":
#   1. id=1 score=0.0328 ...
```

---

## Install — Source (Windows, GNU toolchain — no admin)

The path verified end-to-end on this repo's primary dev machine.

```bash
# 1. Install Rust + GNU toolchain
winget install Rustlang.Rustup --silent
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

# 2. Install WinLibs MinGW (gcc / ld / dlltool — needed by GNU toolchain)
winget install BrechtSanders.WinLibs.POSIX.MSVCRT --silent \
  --accept-package-agreements --accept-source-agreements

# 3. (optional) Install protoc — only needed for duxx-grpc
winget install Google.Protobuf --silent \
  --accept-package-agreements --accept-source-agreements

# 4. Clone + build
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
scripts/build.sh build --release --workspace
```

`scripts/build.sh` prepends WinLibs MinGW (and protoc, if installed)
to PATH so cargo finds the linker. Use it for every cargo invocation
in Git Bash. Bare `cargo` calls without it will fail with
linker errors.

Verify:

```bash
scripts/build.sh test --workspace          # 99 passing
scripts/build.sh run -p duxx-cli --example chatbot_memory
```

---

## Install — Source (Windows, MSVC alternative)

Use this if you'll eventually need Lance / native deps that prefer
MSVC.

```powershell
# Elevated PowerShell:
winget install Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --add Microsoft.VisualStudio.Workload.VCTools `
              --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
rustup default stable-x86_64-pc-windows-msvc
```

Then build via `scripts\build.bat` (sources `vcvars64.bat` so MSVC
env is set):

```cmd
scripts\build.bat test --workspace
```

---

## Install — Python wheel

PyO3 + maturin abi3 — single wheel works on Python 3.8–3.13.

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb/bindings/python

pip install --user maturin
maturin build --release
pip install --force-reinstall ../../target/wheels/duxxdb-*.whl
```

Verify:

```python
import duxxdb
print(duxxdb.__version__)            # 0.1.0
store = duxxdb.MemoryStore(dim=4)
store.remember(key="alice", text="hello", embedding=[1, 0, 0, 0])
print(store.recall(key="alice", query="hello",
                   embedding=[1, 0, 0, 0], k=1))
```

PyPI publish lands with v0.1.0 — see [ROADMAP.md](ROADMAP.md).

---

## Install — Node / TypeScript

napi-rs v2; one `.node` per platform.

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb/bindings/node

npm install                  # pulls @napi-rs/cli
npm run build                # produces *.node + index.js + index.d.ts
npm test
```

> **Windows note:** `napi-build` needs `libnode.dll.a`, which the
> rustup-bundled MinGW doesn't include. On Windows, use the MSVC
> Rust toolchain (not GNU) for this binding. Linux + macOS work
> with either toolchain.

npm publish lands with v0.1.0.

---

## Install — Rust crate

```toml
# Cargo.toml
[dependencies]
duxx-memory  = { git = "https://github.com/bankyresearch/duxxdb" }
duxx-embed   = { git = "https://github.com/bankyresearch/duxxdb", features = ["http"] }
```

`crates.io` publish is on the to-do; pin to a SHA in the meantime if
you want stable behavior.

---

## After install — sanity check

| Surface | Command | Expected |
|---|---|---|
| Embedded Rust example | `cargo run -p duxx-cli --example chatbot_memory` | `stored 7 memories` + 3-line recall |
| RESP TCP server | `duxx-server --addr 127.0.0.1:6379` then `redis-cli -p 6379 PING` | `+PONG` |
| gRPC daemon | `duxx-grpc --addr 127.0.0.1:50051` then `grpcurl -plaintext :50051 duxx.v1.Duxx/Ping` | `nonce: ""` |
| MCP stdio | `duxx-mcp` then send `{"jsonrpc":"2.0","id":1,"method":"initialize"}` | `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"duxxdb",…}}}` |
| Parquet export | `duxx-export --storage dir:./data --out cold.parquet` | `exported N memories -> cold.parquet` |
| Python | `python -c "import duxxdb; print(duxxdb.__version__)"` | `0.1.0` |
| Node | `node -e "console.log(require('duxxdb').version())"` | `0.1.0` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `link: extra operand` on Windows | MSYS2 `/usr/bin/link` shadowing MSVC linker | Use `scripts/build.sh` (forces WinLibs MinGW first on PATH) |
| `cannot open input file 'kernel32.lib'` | MSVC toolchain selected but no Windows SDK | Install Windows 11 SDK component (see MSVC section), or switch to GNU toolchain |
| `dlltool: CreateProcess` errors | Rustup's "self-contained" GNU toolchain is incomplete on Windows | Install WinLibs MinGW (see GNU section) |
| `protoc: command not found` when building | `duxx-grpc` build.rs needs protoc | `winget install Google.Protobuf` (Win) / `apt install protobuf-compiler` (Linux) / `brew install protobuf` (macOS) |
| `napi-build: libnode.dll.a not found` | Rustup MinGW lacks Node import lib on Windows | Use MSVC toolchain for `bindings/node`, or build the binding in CI on Linux |
| Slow first build (~5–10 min) | Heavy deps (tantivy, hnsw_rs, tonic, etc.) | Normal. Caches subsequent builds. Use `sccache` for faster CI/CD. |

Full toolchain history (with the actual journey we took on Windows) is
in [SETUP.md](SETUP.md).

---

## Next

→ [USER_GUIDE.md](USER_GUIDE.md) — quickstart for each integration
surface (Rust, Python, Node, RESP, MCP, gRPC), common workflows,
and configuration knobs.
