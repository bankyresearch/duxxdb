# DuxxDB — Installation

This is the operator's reference for installing DuxxDB. The format
mirrors the Postgres docs: pick your OS, pick your install method,
follow the steps.

---

## Table of contents

- [Pick a path](#pick-a-path)
- [Quickstart (60 seconds, Docker)](#quickstart-60-seconds-docker)
- **Server install**
  - [Docker (single command)](#docker-single-command)
  - [Docker Compose (production-grade)](#docker-compose-production-grade)
  - [Linux — Debian / Ubuntu (`apt`, .deb)](#linux--debian--ubuntu-apt-deb)
  - [Linux — RHEL / Fedora / Rocky / Alma (`dnf`, .rpm)](#linux--rhel--fedora--rocky--alma-dnf-rpm)
  - [Linux / macOS — one-line installer](#linux--macos--one-line-installer)
  - [macOS — Homebrew](#macos--homebrew)
  - [Windows — zip download (manual)](#windows--zip-download-manual)
  - [Kubernetes](#kubernetes)
  - [From source (any OS)](#from-source-any-os)
- **Embed in an app (no server)**
  - [Rust crate](#embed--rust-crate)
  - [Python wheel](#embed--python-wheel)
  - [Node / TypeScript](#embed--node--typescript)
- **After install**
  - [First connection](#first-connection)
  - [Configuration tour](#configuration-tour)
  - [Service control cheatsheet](#service-control-cheatsheet)
  - [Uninstall](#uninstall)
  - [Troubleshooting](#troubleshooting)

---

## Pick a path

| You want to … | Use |
|---|---|
| Run on a laptop right now | [Docker single-command](#docker-single-command) |
| Run on a Linux VM, managed by systemd | [.deb](#linux--debian--ubuntu-apt-deb) / [.rpm](#linux--rhel--fedora--rocky--alma-dnf-rpm) |
| Run on macOS, started by `brew services` | [Homebrew](#macos--homebrew) |
| Run on Windows | [zip download](#windows--zip-download-manual) (or Docker Desktop) |
| Run a production stack with persistent volumes + Prometheus | [Docker Compose](#docker-compose-production-grade) |
| Run on Kubernetes | [k8s StatefulSet](#kubernetes) |
| Embed inside a Rust / Python / Node app | [Rust](#embed--rust-crate) / [Python](#embed--python-wheel) / [Node](#embed--node--typescript) |
| Build from a git checkout | [From source](#from-source-any-os) |

---

## Quickstart (60 seconds, Docker)

Smoke-test on a single machine. **No persistence, no auth.**

```bash
docker run --rm -p 6379:6379 ghcr.io/bankyresearch/duxxdb:latest
```

In another shell:

```bash
redis-cli -p 6379 PING
# +PONG

redis-cli -p 6379 REMEMBER alice "I lost my wallet at the cafe"
# (integer) 1

redis-cli -p 6379 RECALL alice "wallet" 3
# 1) 1) (integer) 1
#    2) "0.032787"
#    3) "I lost my wallet at the cafe"
```

That's it — DuxxDB speaks RESP, so any `redis-cli` / `valkey-cli` /
`redis-rs` / `node-redis` / `go-redis` works unchanged. For anything
real, use one of the production paths below.

---

## Docker (single command)

The image is multi-arch (linux/amd64 + linux/arm64). It bundles
`duxx-server`, `duxx-grpc`, `duxx-mcp`, and `duxx-export` and runs as a
non-root user with a `redis-cli PING` healthcheck baked in.

### Persistent run (host bind-mount)

```bash
mkdir -p ./duxxdb-data

docker run -d --name duxxdb \
  -p 6379:6379 \
  -v "$PWD/duxxdb-data:/var/lib/duxxdb" \
  -e DUXX_STORAGE=dir:/var/lib/duxxdb \
  ghcr.io/bankyresearch/duxxdb:latest
```

Restarts skip the index rebuild (Phase 2.3.5 fast path) because rows +
tantivy + HNSW all live under the bind-mount.

### Production-style run (auth + Prometheus)

```bash
TOKEN=$(openssl rand -hex 32)
echo "DUXX_TOKEN: $TOKEN"   # save somewhere safe

docker run -d --name duxxdb \
  -p 6379:6379 -p 9100:9100 \
  -v duxxdb-data:/var/lib/duxxdb \
  -e DUXX_STORAGE=dir:/var/lib/duxxdb \
  -e DUXX_TOKEN="$TOKEN" \
  -e DUXX_METRICS_ADDR=0.0.0.0:9100 \
  ghcr.io/bankyresearch/duxxdb:latest

curl -sf http://localhost:9100/health      # ok
curl -s  http://localhost:9100/metrics | head
redis-cli -p 6379 -a "$TOKEN" PING         # +PONG
```

### Build the image yourself

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
docker build -t duxxdb:local .
```

---

## Docker Compose (production-grade)

For anything beyond a smoke test, use the compose file in this repo. It
ships:

- A named volume for `dir:` storage (rows + indices survive restarts).
- Required `DUXX_TOKEN` (compose refuses to start without one).
- Prometheus `/metrics` exposed on `:9100` and a Prometheus scraper
  behind the optional `metrics` profile.
- `stop_grace_period: 35s` so SIGTERM gets the full Phase 6.1 drain.
- Healthcheck via `redis-cli PING` (with the auth token).

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb/packaging/docker

cp .env.example .env
$EDITOR .env                        # set DUXX_TOKEN at minimum

docker compose up -d                # core daemon
# OR include the bundled Prometheus scraper:
docker compose --profile metrics up -d
```

Verify:

```bash
docker compose ps                  # duxxdb -> healthy
docker compose logs -f duxxdb
redis-cli -p 6379 -a "$(grep ^DUXX_TOKEN .env | cut -d= -f2)" PING
curl -s http://localhost:9100/health
```

Update to a newer image:

```bash
docker compose pull
docker compose up -d
```

---

## Linux — Debian / Ubuntu (`apt`, .deb)

Tested on Debian 12+ and Ubuntu 22.04+.

### Install from the GitHub Release

```bash
TAG=v0.1.0
curl -fL -o duxxdb.deb \
  "https://github.com/bankyresearch/duxxdb/releases/download/${TAG}/duxxdb_${TAG#v}-1_amd64.deb"
sudo apt install -y ./duxxdb.deb
```

The post-install script:

- Creates the `duxxdb` system user + group.
- Creates `/var/lib/duxxdb` (mode `0750`, owned by `duxxdb`).
- Installs the systemd unit at `/lib/systemd/system/duxxdb.service`.
- Drops `/etc/duxxdb/duxx.env` (mode `0640`).
- Auto-generates a strong `DUXX_TOKEN` if you didn't pre-seed one.

### Start the service

```bash
sudo systemctl start duxxdb
sudo systemctl enable duxxdb         # start on boot

sudo systemctl status duxxdb
journalctl -u duxxdb -f              # tail logs
```

### Verify

```bash
TOKEN=$(sudo grep ^DUXX_TOKEN /etc/duxxdb/duxx.env | cut -d= -f2)
redis-cli -p 6379 -a "$TOKEN" PING       # +PONG
redis-cli -p 6379 -a "$TOKEN" REMEMBER alice "hello"
redis-cli -p 6379 -a "$TOKEN" RECALL alice "hello" 1
```

### Uninstall

```bash
sudo apt remove duxxdb         # keeps /var/lib/duxxdb
sudo apt purge  duxxdb         # destroys data + user (irreversible)
```

---

## Linux — RHEL / Fedora / Rocky / Alma (`dnf`, .rpm)

> .rpm packaging lands in v0.1.1. For now use the
> [one-line installer](#linux--macos--one-line-installer) plus the
> systemd unit from `packaging/systemd/`. The two-step manual install:

```bash
# Binaries (latest release, x86_64).
curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh | sudo sh

# systemd unit + env file.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin duxxdb
sudo mkdir -p /etc/duxxdb /var/lib/duxxdb
sudo chown duxxdb:duxxdb /var/lib/duxxdb
sudo chmod 0750 /var/lib/duxxdb

sudo curl -fsSL -o /etc/systemd/system/duxxdb.service \
  https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/systemd/duxxdb.service
sudo curl -fsSL -o /etc/duxxdb/duxx.env \
  https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/systemd/duxx.env

# Set a token then start.
sudo sh -c 'sed -i "s|^DUXX_TOKEN=$|DUXX_TOKEN=$(openssl rand -hex 32)|" /etc/duxxdb/duxx.env'
sudo chown root:duxxdb /etc/duxxdb/duxx.env && sudo chmod 0640 /etc/duxxdb/duxx.env

sudo systemctl daemon-reload
sudo systemctl enable --now duxxdb
```

---

## Linux / macOS — one-line installer

For machines without `apt` / `brew` / Docker. Drops binaries in
`/usr/local/bin` (or `$DUXX_INSTALL_DIR`).

```bash
curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh | sh
```

Pin a version:

```bash
curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh \
  | DUXX_VERSION=v0.1.0 sh
```

User-only install (no sudo, no system service):

```bash
curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh \
  | DUXX_INSTALL_DIR="$HOME/bin" sh
```

The script verifies the SHA-256 of the downloaded archive against the
`SHA256SUMS` file in the GitHub Release before unpacking.

---

## macOS — Homebrew

> Tap publication is part of the v0.1.0 release cut. Until it's live,
> use the [one-line installer](#linux--macos--one-line-installer) or
> Docker. After the tap goes up:

```bash
brew tap bankyresearch/duxxdb
brew install duxxdb

# Start as a service (relaunch on reboot, log to /var/log/duxxdb.log).
brew services start duxxdb

# Or run interactively in this shell:
duxx-server --addr 127.0.0.1:6379
```

The formula creates `$(brew --prefix)/etc/duxxdb/duxx.env`. Open it,
fill in `DUXX_TOKEN`, then `brew services restart duxxdb`.

```bash
# Connect.
TOKEN=$(grep ^DUXX_TOKEN "$(brew --prefix)/etc/duxxdb/duxx.env" | cut -d= -f2)
redis-cli -p 6379 -a "$TOKEN" PING
```

---

## Windows — zip download (manual)

> A WinGet manifest is on the v0.1.0 to-do. For now:

1. Download `duxxdb-vX.Y.Z-x86_64-windows.zip` from the
   [Releases](https://github.com/bankyresearch/duxxdb/releases) page.
2. Extract somewhere on `%PATH%`, e.g. `C:\Program Files\DuxxDB\`.
3. Open *PowerShell* (Admin) and verify:
   ```powershell
   duxx-server --version
   duxx-server --addr 127.0.0.1:6379 --storage dir:C:\Users\$env:USERNAME\.duxxdb
   ```

For long-running deployment on Windows, prefer **Docker Desktop**
(then follow the [Docker section](#docker-single-command)) or run
DuxxDB inside **WSL 2** with the [.deb](#linux--debian--ubuntu-apt-deb).
Native Windows Service registration ships in v0.2.

---

## Kubernetes

Manifest: `packaging/k8s/duxxdb.yaml` (StatefulSet + headless Service +
ClusterIP Service + ConfigMap, with PVC, healthchecks, and Prometheus
scrape annotations).

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb

kubectl create namespace duxxdb
kubectl -n duxxdb create secret generic duxxdb-token \
  --from-literal=token=$(openssl rand -hex 32)
kubectl -n duxxdb apply -f packaging/k8s/duxxdb.yaml

# Verify.
kubectl -n duxxdb rollout status statefulset/duxxdb
kubectl -n duxxdb port-forward svc/duxxdb 6379:6379 &
TOKEN=$(kubectl -n duxxdb get secret duxxdb-token -o jsonpath='{.data.token}' | base64 -d)
redis-cli -p 6379 -a "$TOKEN" PING
```

The pod runs as UID 10001 with `readOnlyRootFilesystem: true` and the
Phase 6.1 graceful shutdown gets a 45-second termination grace period.

---

## From source (any OS)

The build is fast on Linux/macOS (~5 min cold). Windows-MinGW has its
own tooling section below.

### Prerequisites

| Platform | Required |
|---|---|
| Linux (Debian/Ubuntu) | `build-essential pkg-config protobuf-compiler` |
| Linux (Fedora/RHEL) | `gcc make pkgconf protobuf-compiler` |
| macOS | Xcode CLT (`xcode-select --install`); `brew install protobuf` |
| Windows (GNU, no admin) | Git for Windows + WinLibs MinGW + (optional) `protoc` |
| Windows (MSVC) | VS 2022 Build Tools (C++ workload) + Windows 11 SDK |

Plus Rust ≥ 1.75. Install via [rustup](https://rustup.rs/) if you don't
already have it.

### Linux / macOS

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
cargo build --release --workspace

# Install the four daemons globally (optional).
sudo install -m 0755 target/release/duxx-server /usr/local/bin/
sudo install -m 0755 target/release/duxx-grpc   /usr/local/bin/
sudo install -m 0755 target/release/duxx-mcp    /usr/local/bin/
sudo install -m 0755 target/release/duxx-export /usr/local/bin/
```

Verify:

```bash
cargo test --workspace            # 106 tests should pass
duxx-server --help
```

### Windows — GNU toolchain (preferred, no admin)

```bash
# 1. Rust + GNU toolchain.
winget install Rustlang.Rustup --silent
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

# 2. WinLibs MinGW (gcc / ld / dlltool).
winget install BrechtSanders.WinLibs.POSIX.MSVCRT --silent \
  --accept-package-agreements --accept-source-agreements

# 3. (optional) protoc — only needed for duxx-grpc.
winget install Google.Protobuf --silent \
  --accept-package-agreements --accept-source-agreements

# 4. Clone + build via the wrapper that fixes PATH for MinGW.
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
scripts/build.sh build --release --workspace
```

Use `scripts/build.sh` for every cargo invocation in Git Bash. Bare
`cargo` calls without it will fail with linker errors.

### Windows — MSVC toolchain

```powershell
# Elevated PowerShell:
winget install Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --add Microsoft.VisualStudio.Workload.VCTools `
              --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
rustup default stable-x86_64-pc-windows-msvc
scripts\build.bat test --workspace
```

Full toolchain history (with the actual journey we took on Windows) is
in [SETUP.md](SETUP.md).

---

## Embed — Rust crate

```toml
# Cargo.toml
[dependencies]
duxx-memory = { git = "https://github.com/bankyresearch/duxxdb", tag = "v0.1.0" }
duxx-embed  = { git = "https://github.com/bankyresearch/duxxdb", tag = "v0.1.0", features = ["http"] }
```

```rust
use duxx_memory::MemoryStore;
use duxx_embed::HashEmbedder;
use std::sync::Arc;

let embedder = Arc::new(HashEmbedder::new(32));
let store    = MemoryStore::new(embedder);
let id       = store.remember("alice", "I lost my wallet", None)?;
let hits     = store.recall("alice", "wallet", 5)?;
```

`crates.io` publish is on the v0.1.1 to-do; pin to a tag in the
meantime.

---

## Embed — Python wheel

PyO3 + maturin abi3 — single wheel works on Python 3.8 → 3.13.

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb/bindings/python

pip install --user maturin
maturin build --release
pip install --force-reinstall ../../target/wheels/duxxdb-*.whl
```

```python
import duxxdb

store = duxxdb.MemoryStore(dim=4)
store.remember(key="alice", text="hello", embedding=[1, 0, 0, 0])
print(store.recall(key="alice", query="hello", embedding=[1, 0, 0, 0], k=1))
```

PyPI publish lands with v0.1.0 — see [ROADMAP.md](ROADMAP.md).

---

## Embed — Node / TypeScript

napi-rs v2; one `.node` per platform.

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb/bindings/node
npm install                  # pulls @napi-rs/cli
npm run build                # produces *.node + index.js + index.d.ts
npm test
```

> **Windows note:** `napi-build` needs `libnode.dll.a`, which the
> rustup-bundled MinGW doesn't include. On Windows, use the MSVC Rust
> toolchain (not GNU) for this binding. Linux + macOS work with either.

`npm publish` lands with v0.1.0.

---

## After install

### First connection

| Surface | Command | Expected |
|---|---|---|
| RESP TCP | `redis-cli -p 6379 -a $TOKEN PING` | `+PONG` |
| RESP TCP | `redis-cli -p 6379 -a $TOKEN REMEMBER alice "hello"` | `(integer) 1` |
| Health | `curl -sf http://localhost:9100/health` | `ok` |
| Metrics | `curl -s http://localhost:9100/metrics \| head` | Prometheus text |
| gRPC | `grpcurl -plaintext -H 'x-duxx-token: '$TOKEN :50051 duxx.v1.Duxx/Ping` | `nonce: ""` |
| MCP stdio | `duxx-mcp` (echo `{"jsonrpc":"2.0","id":1,"method":"initialize"}`) | `serverInfo: {name: "duxxdb", …}` |
| Parquet export | `duxx-export --storage dir:./data --out cold.parquet` | `exported N memories -> cold.parquet` |

Full surface tour with copy-paste examples per language:
**[USER_GUIDE.md](USER_GUIDE.md)**.

### Configuration tour

Configuration lives in two places:

- `/etc/duxxdb/duxx.env` (or `$(brew --prefix)/etc/duxxdb/duxx.env` on
  Homebrew, or `.env` next to docker-compose.yml). Loaded by systemd
  via `EnvironmentFile=`.
- CLI flags on `duxx-server` itself (override env every time).

The most-used knobs:

| Variable | CLI flag | Default | Notes |
|---|---|---|---|
| `DUXX_TOKEN` | `--token` | (none) | Required in production. Empty = no auth. |
| `DUXX_STORAGE` | `--storage` | `memory:_` | Use `dir:./path` for full persistence. |
| `DUXX_EMBEDDER` | `--embedder` | `hash:32` | OpenAI / Cohere need an API key env. |
| `DUXX_METRICS_ADDR` | `--metrics-addr` | (off) | `127.0.0.1:9100` is the convention. |
| `DUXX_TLS_CERT` | `--tls-cert` | (off) | PEM cert chain. Both --tls-cert AND --tls-key required. |
| `DUXX_TLS_KEY` | `--tls-key` | (off) | PEM private key. Phase 6.2. |
| `DUXX_MAX_MEMORIES` | `--max-memories` | unlimited | Soft row cap. Lowest decayed-importance rows evicted on overflow. |
| — | `--addr` | `127.0.0.1:6379` | RESP listen address. |
| — | `--drain-secs` | `30` | SIGTERM drain budget. |
| `RUST_LOG` | — | `info` | Use `debug` for troubleshooting. |
| `OPENAI_API_KEY` | — | — | Required iff `DUXX_EMBEDDER=openai:*`. |
| `COHERE_API_KEY` | — | — | Required iff `DUXX_EMBEDDER=cohere:*`. |

### Enabling TLS (Phase 6.2)

```bash
# 1. Get a cert + key. With Let's Encrypt + certbot:
sudo certbot certonly --standalone -d duxxdb.example.com
# /etc/letsencrypt/live/duxxdb.example.com/fullchain.pem
# /etc/letsencrypt/live/duxxdb.example.com/privkey.pem

# 2. Point duxx-server at them:
duxx-server --addr 0.0.0.0:6379 \
  --token "$DUXX_TOKEN" \
  --tls-cert /etc/letsencrypt/live/duxxdb.example.com/fullchain.pem \
  --tls-key  /etc/letsencrypt/live/duxxdb.example.com/privkey.pem \
  --storage  dir:/var/lib/duxxdb

# 3. Connect with redis-cli:
redis-cli --tls -h duxxdb.example.com -p 6379 -a "$DUXX_TOKEN" PING
# +PONG
```

The same `--tls-cert` / `--tls-key` flags work on `duxx-grpc`. For
`grpcurl`:

```bash
grpcurl -H "x-duxx-token: $DUXX_TOKEN" \
  duxxdb.example.com:50051 duxx.v1.Duxx/Ping
```

Restart after edits:

```bash
sudo systemctl restart duxxdb         # systemd
brew services restart duxxdb          # Homebrew
docker compose restart duxxdb         # compose
```

### Service control cheatsheet

| Action | systemd (.deb) | Homebrew | Docker Compose |
|---|---|---|---|
| Start | `sudo systemctl start duxxdb` | `brew services start duxxdb` | `docker compose up -d` |
| Stop | `sudo systemctl stop duxxdb` | `brew services stop duxxdb` | `docker compose stop` |
| Restart | `sudo systemctl restart duxxdb` | `brew services restart duxxdb` | `docker compose restart` |
| Status | `systemctl status duxxdb` | `brew services info duxxdb` | `docker compose ps` |
| Logs | `journalctl -u duxxdb -f` | `tail -f $(brew --prefix)/var/log/duxxdb.log` | `docker compose logs -f` |
| Enable on boot | `sudo systemctl enable duxxdb` | (default) | (handled by `restart: unless-stopped`) |

### Uninstall

| Method | Stop & remove | Also destroy data |
|---|---|---|
| `.deb` | `sudo apt remove duxxdb` | `sudo apt purge duxxdb` |
| Homebrew | `brew services stop duxxdb && brew uninstall duxxdb` | `rm -rf $(brew --prefix)/var/lib/duxxdb` |
| Docker single | `docker rm -f duxxdb` | `docker volume rm duxxdb-data` |
| Docker Compose | `docker compose down` | `docker compose down -v` |
| One-line installer | `sudo rm /usr/local/bin/duxx-{server,grpc,mcp,export}` | (no managed data dir) |
| From source | `cargo clean` and remove your binaries from `$PATH` | (you chose where to write data) |

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NOAUTH Authentication required.` from `redis-cli` | Server has `--token` set | `redis-cli -a "$TOKEN" …` or `AUTH <token>` after connecting |
| `Connection refused :6379` | Server bound to `127.0.0.1` only | Set `--addr 0.0.0.0:6379` (and add auth!) |
| Service starts then dies on first restart with `dir:` storage | tantivy lockfile from previous hard kill | Phase 2.3.5 row-rebuild path takes over automatically — let it finish; subsequent restarts are fast |
| Container won't start, "DUXX_TOKEN is required" | Compose `.env` missing | `cp packaging/docker/.env.example packaging/docker/.env` and set the token |
| `Permission denied` on `/var/lib/duxxdb` | Wrong owner | `sudo chown -R duxxdb:duxxdb /var/lib/duxxdb` |
| `link: extra operand` on Windows | MSYS2 `/usr/bin/link` shadowing MSVC linker | Use `scripts/build.sh` (forces WinLibs MinGW first on PATH) |
| `cannot open input file 'kernel32.lib'` | MSVC selected but no Windows SDK | Install Windows 11 SDK component, or switch to GNU toolchain |
| `protoc: command not found` when building | `duxx-grpc` build.rs needs protoc | `apt install protobuf-compiler` / `brew install protobuf` / `winget install Google.Protobuf` |
| Slow first build (~5–10 min) | Heavy deps (tantivy, hnsw_rs, tonic, etc.) | Normal. Caches subsequent builds. Use `sccache` for CI/CD. |

Full toolchain history is in [SETUP.md](SETUP.md). For runtime
troubleshooting (slow recall, growing memory, lost subscriptions),
see [USER_GUIDE.md](USER_GUIDE.md).

---

## Next

→ [USER_GUIDE.md](USER_GUIDE.md) — quickstart for each integration
surface (Rust, Python, Node, RESP, MCP, gRPC), common workflows, and
configuration knobs.
