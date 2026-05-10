# Contributing to DuxxDB

Thanks for thinking about contributing — DuxxDB is small enough that
your first PR can ship the same week.

This guide covers the operational stuff: how to run the build, what
quality gates a PR has to clear, and how the review cycle works. For
the design rationale behind any specific component, read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the planned-vs-shipped
view, read [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Picking what to work on](#picking-what-to-work-on)
- [Development setup](#development-setup)
- [Build & test](#build--test)
- [Lint, format, benchmarks](#lint-format-benchmarks)
- [Pull request process](#pull-request-process)
- [Commit message style](#commit-message-style)
- [Project structure cheatsheet](#project-structure-cheatsheet)
- [Adding a new crate](#adding-a-new-crate)
- [Adding a new RESP / gRPC / MCP command](#adding-a-new-resp--grpc--mcp-command)
- [Releasing (maintainers)](#releasing-maintainers)
- [Where to ask questions](#where-to-ask-questions)

---

## Code of conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
Be kind, assume good faith, give specific and actionable feedback.

---

## Picking what to work on

In order of "most likely to land cleanly":

1. **Issues labelled `good first issue` or `help wanted`** —
   pre-scoped, with acceptance criteria.
2. **An unchecked box in [`docs/ROADMAP.md`](docs/ROADMAP.md)** —
   bigger pieces; please open a tracking issue first so we can sketch
   the shape together.
3. **A bug you hit in real use** — open an issue using the
   [bug-report template](.github/ISSUE_TEMPLATE/bug_report.yml) before
   the PR; the issue is where we'll discuss the fix's scope.
4. **A new feature** — open an issue using the
   [feature-request template](.github/ISSUE_TEMPLATE/feature_request.yml)
   and wait for an `accepted` label before opening a PR. We sometimes
   say no to features that don't fit the scope; that's why we gate.

Things we generally **decline**:

- New storage backends without a use case that redb or `dir:` can't serve.
- Adapter shims for proprietary vector DBs.
- Cosmetic refactors without a measurable improvement.

---

## Development setup

### Prerequisites

| Platform | Required |
|---|---|
| Linux (Debian/Ubuntu) | `build-essential pkg-config protobuf-compiler` |
| Linux (Fedora/RHEL) | `gcc make pkgconf protobuf-compiler` |
| macOS | Xcode CLT (`xcode-select --install`); `brew install protobuf` |
| Windows (GNU, no admin) | Git for Windows + WinLibs MinGW |
| Windows (MSVC) | VS 2022 Build Tools (C++ workload) + Windows 11 SDK |

Plus Rust **≥ 1.75**. Use [rustup](https://rustup.rs/).

### Clone

```bash
git clone https://github.com/bankyresearch/duxxdb.git
cd duxxdb
```

### Optional but recommended

```bash
rustup component add clippy rustfmt
cargo install cargo-deny    # supply-chain audit
cargo install cargo-deb     # only if you want to build .deb locally
```

---

## Build & test

```bash
# Linux / macOS
cargo build --workspace
cargo test  --workspace      # all 106 tests, < 60s on a modern laptop

# Windows (GNU toolchain)
scripts/build.sh build --workspace
scripts/build.sh test  --workspace

# Windows (MSVC toolchain)
scripts\build.bat test --workspace
```

Run a single crate's tests:

```bash
cargo test -p duxx-server
cargo test -p duxx-memory
```

Run a single test by name:

```bash
cargo test -p duxx-server -- auth_required_when_token_set --nocapture
```

The full integration matrix runs in CI on every push (Linux + macOS +
Windows). Don't merge anything red.

---

## Lint, format, benchmarks

Three checks gate every PR.

### Clippy

```bash
cargo clippy --workspace -- -D warnings
```

Treat warnings as errors. If a lint flags a real concern, fix the
code; if it's wrong, `#[allow(clippy::specific_lint)]` with a comment
explaining why.

### Format

```bash
cargo fmt --all
cargo fmt --all --check       # CI runs this
```

We use stock `rustfmt`. No `rustfmt.toml` overrides — keep it boring.

### Benchmarks

```bash
cargo bench -p duxx-bench
```

PRs that touch `duxx-memory`, `duxx-index`, or `duxx-query` should
post `cargo bench` numbers in the PR description. **Don't regress any
bench by more than 5 %.** If you must, justify it.

---

## Pull request process

1. **Fork** and create a branch off `master`. Name it
   `<crate>/<short-description>` — e.g.
   `duxx-server/streaming-recall`.
2. **Write tests first** when you can. New behaviour without a test
   gets a "needs test" label.
3. **Make the change small.** Multiple small PRs > one big one.
4. **Run the full local check list:**
   ```bash
   cargo fmt --all
   cargo clippy --workspace -- -D warnings
   cargo test  --workspace
   cargo bench -p duxx-bench       # if you touched a hot path
   ```
5. **Open the PR** using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
   The template's checklist is the same one a reviewer will use.
6. **CI** runs the matrix on Linux + macOS + Windows + clippy + fmt +
   the Python bindings cargo-check job. All must pass.
7. **Review.** A maintainer will respond within ~3 business days. If
   you don't hear back, ping the PR thread.
8. **Squash on merge.** Keep the merge commit message clean — see
   [Commit message style](#commit-message-style).

---

## Commit message style

Conventional Commits, lightly enforced:

```
<type>(<scope>): <imperative summary, ≤ 70 chars>

<body — what & why; not how, the diff is the how>

Refs: #<issue>
```

Common types:

| Type | When |
|---|---|
| `feat` | New user-visible capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Internal restructure, no behavior change |
| `docs` | Documentation only |
| `test` | Test-only changes |
| `chore` | Tooling, deps, CI |
| `Phase 6.1:` | Roadmap-aligned milestone (no scope; e.g. `Phase 6.2: native TLS`) |

Real example from the log:

```
Phase 6.1: production hardening (auth, health, metrics, graceful shutdown)

Table-stakes "prod-level DB" capabilities for UAT and beyond.
…
```

---

## Project structure cheatsheet

```
crates/duxx-core/         primitive types — Schema, Value, Error
crates/duxx-storage/      Storage trait + redb / memory backends
crates/duxx-index/        tantivy BM25 + hnsw_rs HNSW
crates/duxx-query/        Reciprocal Rank Fusion, hybrid_recall
crates/duxx-memory/       MEMORY / TOOL_CACHE / SESSION + decay
crates/duxx-reactive/     ChangeBus over tokio broadcast
crates/duxx-embed/        Embedder trait + hash / OpenAI / Cohere
crates/duxx-server/       RESP2/3 daemon (Valkey-compat)
crates/duxx-mcp/          MCP stdio JSON-RPC server
crates/duxx-grpc/         tonic gRPC daemon
crates/duxx-coldtier/     Apache Parquet exporter
crates/duxx-cli/          duxx CLI + examples
crates/duxx-bench/        criterion benches
bindings/python/          PyO3 + maturin abi3 wheel
bindings/node/            napi-rs v2 .node module (workspace-excluded)
```

Most user-visible features live in **`duxx-memory`**. Network surfaces
(`duxx-server`, `duxx-grpc`, `duxx-mcp`) are thin adapters — they
should never re-implement logic that belongs in `duxx-memory`.

---

## Adding a new crate

1. Add the directory under `crates/`.
2. Add it to the `[workspace] members =` list in the root `Cargo.toml`.
3. Use `version.workspace = true` etc. for shared metadata.
4. Add a `description` and `license = "Apache-2.0"` in your
   `Cargo.toml` so it's `crates.io`-publishable.
5. Drop a `lib.rs` with at least one `#[test]` so CI exercises it.

---

## Adding a new RESP / gRPC / MCP command

Three surfaces, one source of truth:

1. **Implement** the logic in `duxx-memory` (or whichever core crate
   it belongs in). Add unit tests there.
2. **Wire RESP** in `crates/duxx-server/src/lib.rs` — add a `cmd_xxx`
   helper, hook it into `dispatch_with_auth`, and update
   `crates/duxx-server/src/lib.rs` tests.
3. **Wire gRPC** in `crates/duxx-grpc/proto/duxx.proto`, regenerate,
   then implement on the `DuxxService` impl. Add an integration test.
4. **Wire MCP** in `crates/duxx-mcp/src/lib.rs` as a tool with
   `name`, `description`, JSON-Schema `inputSchema`. Update the MCP
   golden tests.
5. **Document** in `docs/USER_GUIDE.md` (per-surface examples).

The auth path applies to RESP + gRPC automatically — don't try to
shortcut it.

---

## Releasing (maintainers)

We use semver and tag-driven releases. To cut `vX.Y.Z`:

```bash
# 1. Update CHANGELOG.md with the new section.
# 2. Bump workspace version in Cargo.toml.
# 3. Commit on master; CI green.

git tag -a vX.Y.Z -m "DuxxDB vX.Y.Z"
git push origin vX.Y.Z
```

The [release workflow](.github/workflows/release.yml) handles the rest:

- Builds binaries for 5 targets (linux/macos × x86_64/aarch64 + windows-x86_64)
- Builds a `.deb` on amd64
- Computes `SHA256SUMS`
- Uploads everything to the GitHub Release
- Pushes a multi-arch image to `ghcr.io/<owner>/duxxdb:vX.Y.Z` + `:latest`

After the release lands:

1. Open a follow-up PR bumping the Homebrew formula's `version` +
   `sha256` lines (the workflow leaves a comment with the values).
2. PyPI / npm / crates.io publishes are manual for now — see the
   per-binding notes in `bindings/*/README.md`.

---

## Where to ask questions

- **Bugs & features** → [GitHub Issues](https://github.com/bankyresearch/duxxdb/issues)
  (use the templates).
- **Open-ended design discussion** → [GitHub Discussions](https://github.com/bankyresearch/duxxdb/discussions).
- **Security** → see [SECURITY.md](SECURITY.md). **Do not** open a
  public issue for vulnerabilities.

Thank you. ❤️
