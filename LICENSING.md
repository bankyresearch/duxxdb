# DuxxDB Licensing (Open-Core)

DuxxDB is **open-core**: a permissively-licensed engine that anyone can use and
self-host, plus a source-available enterprise layer that funds the project.

| Tier | License | What it is |
|---|---|---|
| **Core** | **Apache-2.0** ([`LICENSE`](LICENSE)) | The single-node agent database — engine, primitives, server, bindings, integrations, document layer |
| **Enterprise** | **BUSL-1.1** ([`LICENSE-BSL`](LICENSE-BSL)) | Multi-tenancy, control plane, replication/HA, and the managed-cloud frontend |

## Core — Apache-2.0 (OSS)

Free for any use, including commercial and as-a-service. Crates:

```
duxx-core  duxx-storage  duxx-index  duxx-query  duxx-memory  duxx-reactive
duxx-embed  duxx-trace  duxx-prompts  duxx-datasets  duxx-eval  duxx-replay
duxx-cost  duxx-docs  duxx-token  duxx-coldtier  duxx-grpc  duxx-mcp
bindings/*  integrations/*
```

These crates set `license = "Apache-2.0"` and are crates.io-publishable.

## Enterprise — BUSL-1.1 (source-available, NOT OSS)

Source is in this repo (read it, modify it, self-host it), but you may **not**
offer it as a competing hosted/managed service. Each version's terms convert to
**Apache-2.0 four years** after publication (the BSL "Change Date"). Crates/dirs:

```
crates/duxx-tenant     multi-tenancy & isolation
crates/duxx-control    control plane (orgs/projects/keys/members, JWT, billing)
crates/duxx-cluster    replication / HA / clustering
frontend/              Cloud Console + Studio
```

These set `license = "BUSL-1.1"` and **`publish = false`** (never pushed to
crates.io).

## Why some crates that look "core" are not crates.io-publishable

`duxx-server` is Apache-2.0 **source**, but its *default build* links the
Enterprise crates (`duxx-tenant`, `duxx-cluster`), so the produced binary is
governed by BUSL-1.1 and the crate is marked `publish = false`. It ships as a
binary / Docker image.

**Roadmap:** gate the multi-tenant / replication / Studio code behind an
`enterprise` cargo feature so the *default* `duxx-server` build is a pure
single-node Apache-2.0 server that **is** crates.io-publishable, and the
enterprise build opts in. Until then, anything depending on an Enterprise crate
also carries `publish = false`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md#licensing-open-core). Core contributions are
Apache-2.0; Enterprise contributions are BUSL-1.1.

## Commercial licensing

For a commercial license (to remove the BSL "as-a-service" restriction, or for
support/SLA), contact the maintainers.
