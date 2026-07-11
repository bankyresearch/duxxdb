# DuxxDB Licensing

DuxxDB (this repository) is **open source under [Apache-2.0](LICENSE)** — free for any
use, including commercial and as-a-service. Every crate here is Apache-2.0 and
crates.io-publishable:

```
duxx-core  duxx-storage  duxx-index  duxx-query  duxx-memory  duxx-reactive
duxx-embed  duxx-trace  duxx-prompts  duxx-datasets  duxx-eval  duxx-replay
duxx-cost  duxx-docs  duxx-token  duxx-coldtier  duxx-grpc  duxx-mcp
duxx-server  duxx-cli  bindings/*  integrations/*
```

`duxx-server` is the single-node agent database: hybrid retrieval, the agent
primitives (MEMORY / TOOL_CACHE / SESSION), the Phase-7 registries, auth + RBAC,
TLS, metrics, and graph compaction — all Apache-2.0.

## Open-core

The **managed-services / enterprise layer** — multi-tenancy, the control plane,
replication / HA, and the Cloud Console + Studio — lives in a separate,
source-available repository (**DuxxDB Cloud**, BUSL-1.1) that builds on these
Apache crates. It is not part of this repo. For a commercial license or the
hosted service, contact the maintainers.

## Contributing

All contributions to this repository are under Apache-2.0. See
[CONTRIBUTING.md](CONTRIBUTING.md).
