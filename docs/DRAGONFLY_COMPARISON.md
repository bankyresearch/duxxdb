# DuxxDB vs DragonflyDB: Enterprise AI Database Roadmap

This document compares DuxxDB with DragonflyDB from the perspective of
enterprise AI-agent workloads. DragonflyDB is a mature in-memory
Redis/Memcached-compatible datastore. DuxxDB is an agent-native
memory, retrieval, trace, prompt, dataset, eval, replay, and cost
engine. The goal is not to copy DragonflyDB; it is to identify the
reliability and operational patterns DuxxDB should implement for
enterprise deployments.

Sources reviewed:

- https://www.dragonflydb.io/features
- https://www.dragonflydb.io/docs
- https://www.dragonflydb.io/kubernetes
- https://www.dragonflydb.io/blog/dragonfly-as-a-multi-purpose-data-store-for-ai-applications
- https://www.dragonflydb.io/blog/replication-for-high-availability

## Positioning

| Area | DragonflyDB | DuxxDB Today | DuxxDB Enterprise Direction |
|---|---|---|---|
| Primary purpose | General in-memory cache/store | AI-agent memory and retrieval engine | Stay AI-native, add database-grade reliability |
| Wire compatibility | Redis and Memcached APIs | RESP subset, gRPC, MCP, Python, Node | Expand RESP compatibility only where it helps agent stacks |
| AI primitives | Redis-compatible data structures plus vector search | Memory, tool cache, sessions, traces, prompts, datasets, evals, replay, costs | Add lifecycle, governance, and isolation around these primitives |
| Scale model | Vertical multi-core first, HA replicas/operator/cloud | Single node with persistence | Add shard-per-core execution, replicas, and operator-managed failover |
| Enterprise ops | Snapshots, replication, K8s operator, Prometheus, OTel | Prometheus, health, TLS/mTLS, token/RBAC auth, audit log, configurable RESP/rate limits, offline snapshot CLI, packaging | Add OTel, live snapshots, HA controller |

## Features To Add

### P0: Enterprise Gatekeepers

1. **Protocol resource limits**
   - Status: RESP frame, line, bulk, array, nesting-depth, and
     per-connection input-buffer limits are configurable via CLI flags
     and `DUXX_MAX_*` environment variables.
   - Next: add per-IP connection caps and tenant-aware request budgets.

2. **mTLS and service identity**
   - Status: RESP and gRPC listeners can require client certificates
     with `--tls-client-ca` / `DUXX_TLS_CLIENT_CA`.
   - Next: map cert subjects/SANs to service principals.
   - Keep token auth as a simple bootstrap mode.

3. **RBAC and tenant isolation**
   - Status: RESP supports role API keys with read/write/admin roles,
     audit events, and tenant-scoped key namespacing for tenant-safe
     memory/session/cost commands.
   - Next: enforce tenant filters inside every Phase 7 primitive, not
     only the tenant-safe RESP command set.
   - Add tests proving tenant A cannot read tenant B data through RESP,
     gRPC, Python bindings, subscriptions, export, or recall.

4. **Backup and restore CLI**
   - Status: `duxx-snapshot create`, `duxx-snapshot verify`, and
     `duxx-snapshot restore` support offline local filesystem snapshots
     with a SHA-256 manifest.
   - Next: add live/nonblocking snapshots and S3-compatible object
     storage.
   - Restore must stay documented and tested in CI.

5. **Supply-chain release controls**
   - Keep hard `fmt`, `clippy -D warnings`, tests, and `cargo audit`.
   - Add SBOM/provenance for container releases.
   - Add binary signing and checksum verification to release assets.

### P1: Dragonfly-Inspired Reliability

1. **Shard-per-core execution**
   - Partition keys and vector index work across worker shards.
   - Route commands to the owning shard and avoid global locks.
   - Preserve agent-level consistency for multi-primitive workflows.

2. **Low-overhead snapshotting**
   - Status: offline manifest snapshotting covers row stores, HNSW
     metadata, Tantivy segments, and Phase 7 primitive backends when
     they live under the same storage directory.
   - Next: build an async snapshot writer that does not block
     recall/write traffic.
   - Record snapshot checkpoints so restore can reject partial or mixed
     versions.

3. **Replica protocol**
   - Start with async primary-to-replica log shipping.
   - Use per-shard streams once shard-per-core exists.
   - Expose `ROLE`/`REPLICAOF`-style admin commands only if compatible
     with DuxxDB semantics.

4. **Kubernetes operator**
   - Manage a primary plus replicas.
   - Automate failover, PVC attachment, readiness gates, and snapshot
     restore.
   - Add `DuxxDB` CRD fields for storage, TLS secrets, token/mTLS
     policy, memory caps, and backup schedules.

5. **OpenTelemetry export**
   - Export internal server spans, request latency, recall/index phases,
     embedder calls, snapshot operations, and replication lag.
   - Keep `duxx-trace` as the AI-agent trace store; OTel export is for
     DuxxDB's own operational telemetry.

### P2: AI-Specific Differentiators

1. **Semantic eviction policy**
   - Keep importance decay, but add frequency, recency, tenant quota,
     semantic diversity, and cost impact.
   - Avoid evicting the only memory in a semantic cluster.

2. **RAG consistency checkpoints**
   - Snapshot memories, prompts, datasets, eval baselines, and traces
     under one versioned checkpoint.
   - Let agents say: "recall against checkpoint X" for deterministic
     eval and replay.

3. **Prompt/dataset/eval governance**
   - Require approvals for tag moves like `prod`.
   - Add audit records for prompt changes, dataset deletes, replay
     overrides, and budget changes.
   - Support retention windows per tenant and primitive.

4. **Vector and text index compaction**
   - Implement HNSW tombstones and Tantivy deletes for real memory
     reclamation after eviction/delete.
   - Add background compaction with Prometheus and OTel visibility.

5. **Workload protection**
   - Add tenant budgets for rows, memory bytes, QPS, recall fanout,
     embedder calls, and export volume.
   - Return predictable errors instead of letting one tenant exhaust the
     process.

## What Not To Copy

- DuxxDB should not become a generic Redis clone. RESP compatibility is
  valuable for adoption, but the product advantage is agent-native
  semantics.
- DuxxDB should not hide dependency attribution. Legal notices must
  remain accurate while the project uses external crates.
- DuxxDB should not claim distributed reliability until failover,
  restore, and tenant-isolation tests run in CI.

## Enterprise Release Definition

DuxxDB can be called enterprise-ready when:

- Single-tenant production mode passes hard CI and has tested restore.
- Multi-tenant mode has enforced tenant isolation and RBAC.
- Network-exposed mode has mTLS, rate limits, resource limits, and
  audit logs.
- HA mode has a documented primary/replica failover path.
- Every release ships checksums, SBOM/provenance, and a security
  advisory process.
