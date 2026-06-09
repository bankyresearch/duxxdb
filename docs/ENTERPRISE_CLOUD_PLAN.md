# DuxxDB Enterprise & Managed Cloud Plan

> **Goal:** Evolve DuxxDB from a feature-complete self-hosted single-node engine
> into a **managed, multi-tenant enterprise cloud** — "Supabase for AI agents" —
> while keeping its agent-native moat (memory, tool cache, hybrid retrieval,
> trace/replay/eval/cost) as the differentiator.
>
> Status: **Planning** · Target baseline: v0.2.2 · Owner: core team

---

## 0. TL;DR

DuxxDB today (v0.2.2) is **further along than external analyses assume**: all
nine agent primitives are implemented and persistent (redb-backed), the server
speaks RESP/gRPC/MCP, and it already has token auth, RBAC roles, TLS/mTLS, an
audit log, Prometheus metrics, and graceful drain. **248 tests pass.**

The journey to a managed cloud is therefore a **platform problem, not an engine
problem.** Five things block it, in strict dependency order:

| # | Blocker | Today | Phase |
|---|---------|-------|-------|
| 1 | **Tenant data isolation** (physical, not just key-prefix) | logical prefix only; shared indices | **A — the gate** |
| 2 | **RBAC enforcement per-operation** with a real permission model | roles defined, coarse command-level checks | **A — the gate** |
| 3 | **Control plane** (orgs/projects/keys/provisioning/metering→billing) | none | **B** |
| 4 | **Studio** (admin UI) + governance (SOC2, encryption-at-rest, KMS, SSO) | partial | **C** |
| 5 | **Clustering / HA / replication** | single-node redb | **D — defer** |

**Do not** try to replace Postgres/Supabase/S3. Position DuxxDB as the **hot
operational database for AI agents** and *integrate* with Postgres (app data),
S3/MinIO (objects), and a lakehouse (analytics). See §2.

---

## 1. Corrected baseline (what actually exists)

| Capability | State | Evidence |
|------------|-------|----------|
| Memory / ToolCache / Session | Solid; ToolCache+Session in-memory only | `crates/duxx-memory` |
| Prompts / Datasets / Eval / Replay / Cost / Trace | Solid, persistent (redb) | `crates/duxx-{prompts,datasets,eval,replay,cost,trace}` |
| Hybrid retrieval (HNSW + BM25 + RRF) | Solid | `duxx-index`, `duxx-query` |
| Server: RESP2/3, gRPC, MCP | Solid | `duxx-server`, `duxx-grpc`, `duxx-mcp` |
| Token auth, RBAC roles, TLS, mTLS, audit log | Solid (coarse RBAC) | `duxx-server/src/security.rs` |
| Tenant scoping | **Logical prefix only**, Phase-7 blocked for tenants | `apply_tenant_scope`, `tenant_safe_command` in `duxx-server/src/lib.rs` |
| Multi-tenant **isolation**, projects, quotas | **None** | — |
| Control plane / dashboard / billing | **None** | — |
| Clustering / replication | **None** | — |

### The v0.2.2 tenant scheme and why it's not enough

Today a tenant-scoped principal (`--auth-key name:token:role:tenant`) gets:
- `SET/GET/DEL/REMEMBER/RECALL` keys silently prefixed with `tenant:`.
- pub/sub channels prefixed.
- `COST.*` filtered/injected with the tenant.
- **All other Phase 7 commands rejected** (`tenant_safe_command` whitelist).

Limitations that the enterprise build must fix:
1. **Shared indices.** One HNSW + one tantivy index hold *all* tenants' vectors.
   Isolation depends entirely on key-prefix string matching at the row layer —
   fragile, and a recall that searches the shared ANN graph can waste `k` slots
   on (or, with a bug, leak) other tenants' data.
2. **No physical separation** → noisy-neighbor blast radius, no per-tenant
   backup/restore/delete, no per-tenant encryption key.
3. **Tenants can't use prompts/evals/traces/datasets/replay at all.**
4. **Flat namespace.** Enterprises need `org → project → environment (dev/staging/prod)`.
5. **String-munging args** is a brittle enforcement point; isolation should be a
   structural property of the data plane, not a per-command rewrite.

---

## 2. Product positioning

> **DuxxDB is the operational database for AI agents** — memory, retrieval, tool
> cache, traces, evals, replay, prompts, and cost ledger in one low-latency
> engine, offered as a managed multi-tenant cloud.

**Integrate, don't replace:**

```
                 ┌─────────────────────────────────────────┐
   App backend   │  Postgres / Supabase  (users, billing,   │
   (you keep)    │                        app tables, RLS)  │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
   Object store  │  S3 / MinIO / Supabase Storage           │
                 │  (PDFs, images, audio, docs)             │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
   ►  DuxxDB ◄   │  MEMORY · TOOL_CACHE · SESSION · TRACE   │
   (the moat)    │  PROMPTS · DATASETS · EVALS · REPLAY     │
                 │  COST · HYBRID RETRIEVAL                  │
                 └─────────────────────────────────────────┘
                 ┌─────────────────────────────────────────┐
   Analytics     │  DuckDB / Spark / Iceberg (cold tier,    │
                 │  via Parquet export)                     │
                 └─────────────────────────────────────────┘
```

Build **connectors** to these, not clones of them.

---

## 3. The multi-tenant data model (foundation for everything)

### 3.1 Hierarchy

```
Organization              billing entity, SSO domain, owner
└── Project               isolation boundary = one logical DuxxDB namespace
    └── Environment       dev | staging | prod  (separate keyspaces/quotas)
        └── Agent         optional sub-namespace for per-agent memory
            └── Session / Memory / Trace / ...
```

The **isolation boundary is the Project+Environment** (a "workspace"). Every
primitive operation is scoped to exactly one workspace, resolved from the
caller's credential — never from a client-supplied argument.

### 3.2 Namespace identity

```rust
/// Resolved from the API key / principal, NEVER from request args.
pub struct Namespace {
    pub org_id: Arc<str>,      // org_<ulid>
    pub project_id: Arc<str>,  // proj_<ulid>
    pub env: Env,              // Dev | Staging | Prod
}
```

A `Namespace` produces a stable byte prefix used everywhere:

```
ns_prefix = org_id \0 project_id \0 env \0      // reuses duxx-storage::backend::key::prefix
```

### 3.3 Isolation strategy per primitive

There are **two** kinds of primitive, each with a different correct isolation
mechanism. **Do not** rely on key-prefixing alone for the index-backed ones.

| Primitive | Storage shape | Isolation mechanism |
|-----------|---------------|---------------------|
| Session, ToolCache | KV (`Backend` byte-keyed) | **Namespace-prefixed keys** — cheap, no index. |
| Prompts, Datasets, Eval, Replay, Cost, Trace | multi-table `Backend` | **Namespace-prefixed composite keys** (prepend `ns_prefix` as the first key part; every `scan_prefix` becomes tenant-scoped automatically). |
| **MemoryStore** | rows + **HNSW + tantivy** | **Per-workspace store routing** — one `MemoryStore` (and thus one HNSW + one tantivy index) *per workspace*, lazily created and cached. True physical isolation of vectors and full-text. This is the only way to keep ANN recall correct **and** leak-proof. |

> **Why per-workspace stores for Memory, not shared+filter:** HNSW returns the
> global top-k by vector distance. Post-filtering by tenant means (a) `k` can be
> silently under-filled by neighbors from other tenants, degrading recall, and
> (b) any filter bug is a cross-tenant data leak. Per-workspace indices make
> isolation a structural guarantee and also enable per-tenant snapshot, restore,
> delete, and (later) per-tenant encryption keys. The cost is memory/handle
> overhead per active workspace — addressed by an LRU of warm stores plus
> cold-load-from-disk, and by Project-per-node placement at scale (§7).

### 3.4 The `Workspace` / `TenantRouter` abstraction (new crate: `duxx-tenant`)

```rust
pub struct TenantRouter {
    embedder: Arc<dyn Embedder>,
    root: PathBuf,                       // dir:/var/lib/duxx/workspaces
    warm: Mutex<LruCache<Namespace, Arc<Workspace>>>,   // bounded; cold ones drop to disk
    quotas: QuotaTable,
}

pub struct Workspace {
    pub ns: Namespace,
    pub memory: MemoryStore,             // isolated HNSW + tantivy + rows
    pub sessions: SessionStore,
    pub tool_cache: ToolCache,
    // Phase-7 registries share a redb file per workspace, namespaced tables:
    pub prompts: PromptRegistry,
    pub datasets: DatasetRegistry,
    pub evals: EvalRegistry,
    pub replays: ReplayRegistry,
    pub costs: CostLedger,
    pub traces: TraceStore,
}

impl TenantRouter {
    /// Resolve (and lazily open/restore) the workspace for a namespace.
    pub fn workspace(&self, ns: &Namespace) -> Result<Arc<Workspace>>;
}
```

On-disk layout:

```
/var/lib/duxx/workspaces/<org_id>/<project_id>/<env>/
    memory.redb  memory.hnsw/  memory.tantivy/
    phase7.redb  phase7.hnsw/
    meta.json    version.json
```

The server's dispatch path changes from *"mutate args to add a tenant prefix"*
to *"resolve the caller's `Namespace`, fetch its `Workspace`, and route the
command to that workspace's primitives."* The isolation is then **impossible to
bypass from the wire** — there is no tenant argument to forge.

---

## 4. RBAC & permission model

Replace the coarse `Role::{ReadOnly,ReadWrite,Admin}` command gate with a
**capability set** evaluated per operation, scoped to a namespace.

### 4.1 Roles → default capabilities

| Role | Capabilities |
|------|--------------|
| Owner | full project control + member/key management + billing |
| Admin | manage agents, keys, datasets, prompts; all data ops |
| Developer | read/write memory, prompts, evals, datasets, sessions |
| Evaluator | run evals, read traces/datasets/results |
| Observer | read traces, costs, eval results only |
| Service account | agent-runtime data ops (the token an agent uses in prod) |
| Tenant user | only own memory/session within an agent sub-namespace |

### 4.2 Capabilities (checked at each handler)

```
read_memory  write_memory  delete_memory  search_memory
read_session write_session
read_trace   write_trace
read_prompt  write_prompt  manage_prompt
read_dataset write_dataset
run_eval     read_eval
read_cost    write_cost
manage_keys  manage_members  manage_billing  manage_project
```

### 4.3 Enforcement points (defense in depth)

1. **Credential → (Namespace, Role, Capabilities)** resolved once at AUTH /
   gRPC interceptor / control-plane-issued JWT validation.
2. **Per-command capability check** (replaces `required_role`): map each RESP /
   gRPC / MCP command to a required capability.
3. **Namespace routing** (§3.4): the command can only ever touch the resolved
   workspace — no client-supplied tenant arg exists anymore.
4. **Audit** every mutating + denied op (already implemented; extend with
   `org_id/project_id/env`).

---

## 5. Phase A — make the data plane multi-tenant-safe (THE GATE)

Pure Rust inside crates you already own. **Nothing in Phase B/C is safe to ship
until A lands.**

### A1. `duxx-tenant` crate — `Namespace`, `Workspace`, `TenantRouter`
- LRU of warm workspaces; lazy open/restore from `dir:` layout.
- Quota table (rows, bytes, QPS) per workspace (enforced; not just cost budget).
- *Prototype already delivered* — see `crates/duxx-tenant` + §8.

### A2. Namespace-prefix every `Backend`-backed primitive
- Add an internal `ns_prefix` to the composite key in Prompts/Datasets/Eval/
  Replay/Cost/Trace, OR open a per-workspace `phase7.redb` (preferred — simpler
  isolation, per-tenant backup). Keep the public command surface unchanged.

### A3. Capability-based RBAC
- New `Capability` enum + `Role::capabilities()`; `command → Capability` map.
- Replace the `auth.allow_command` + `tenant_safe_command` pair with a single
  `auth.allow(capability)` against the resolved namespace.
- Tenants can now **use** prompts/evals/traces/etc. — isolated, not blocked.

### A4. Durable ToolCache + Session
- Back both with the `Backend` trait (they're in-memory today). Required before
  they hold customer data across restarts.

### A5. Per-workspace lifecycle ops
- `snapshot(ns)`, `restore(ns)`, `drop(ns)` (GDPR delete), `export(ns)` (Parquet
  cold tier, already exists — scope it per workspace).

### A6. Async API surface
- Wrap Python/Node bindings + a new HTTP/gRPC data API in async. Agents are
  built around `await`; sync-only is a production blocker.

**Exit criteria for Phase A:** two workspaces on one node, provably unable to
read each other's memory/sessions/traces/prompts/costs (negative tests), with
per-workspace snapshot/restore/drop and capability-enforced RBAC.

---

## 6. Phase B — the control plane (turns software into a cloud)

A **separate service** (`duxx-control`, likely its own repo) — DuxxDB nodes stay
dumb data planes; the control plane owns identity, provisioning, and metering.

### 6.1 Responsibilities

```
Org / Project / Environment CRUD
Member & role management, invitations
API key issuance & rotation (scoped to project+env+role → a signed token/JWT)
Provisioning: place a workspace on a data-plane node (or spin a dedicated node)
Usage metering: pull CostLedger + quota counters → billing
Billing: Stripe (metered + seats), plan/quota enforcement
SSO: SAML / OIDC for org login
Audit aggregation across nodes
```

### 6.2 Architecture

```
        ┌──────────────────────────────────────────────┐
        │              Duxx Cloud Console (web)         │
        └──────────────────────────────────────────────┘
                              │ REST/gRPC
        ┌──────────────────────────────────────────────┐
        │         duxx-control  (control plane)         │
        │  Postgres: orgs, projects, members, keys,     │
        │            placements, usage, invoices        │
        │  Stripe · SSO (OIDC/SAML) · provisioner       │
        └──────────────────────────────────────────────┘
              │ issues scoped JWT          │ provisions / places
              ▼                            ▼
        ┌──────────────┐           ┌──────────────────────────┐
        │   Agent SDK  │──────────▶│  DuxxDB data-plane nodes  │
        │ (project key)│  RESP/    │  TenantRouter → Workspace │
        └──────────────┘  gRPC/MCP │  (per-project isolation)  │
                                   └──────────────────────────┘
```

### 6.3 Provisioning model (staged)

- **B1 — shared nodes, namespace-per-project.** Many projects per node via
  `TenantRouter`. Cheapest path to first revenue. (Most projects live here.)
- **B2 — dedicated node per project** for large/enterprise customers (noisy-
  neighbor isolation, dedicated resources). Same binary, single-tenant config.

### 6.4 Metering → billing

`CostLedger` already records per-tenant token/cost; quota counters record rows/
bytes/QPS. The control plane scrapes these per billing period and pushes to
Stripe (metered usage + seat counts). Plan limits become quota ceilings enforced
in the data plane (`429`/`OVERQUOTA`).

### 6.5 Key issuance

Control plane signs a short-lived JWT carrying `{org_id, project_id, env, role,
capabilities, exp}`. Data-plane nodes validate the signature (control-plane
public key) and resolve the `Namespace` from claims — **no shared static
tokens** for cloud customers. Self-hosted keeps the `--auth-key` flow.

---

## 7. Phase C / D — Studio, governance, scale

### C — Duxx Studio (admin UI) + enterprise readiness
- Memory browser, trace viewer, prompt diff, eval dashboard, cost dashboard,
  dataset browser, access-policy editor, namespace/tenant explorer.
- Your data is already structured for all of these (Phase 7 primitives).
- Governance: encryption at rest, KMS/BYOK (per-workspace keys — enabled by §3.3
  physical isolation), SOC2 controls, immutable audit trail, PII redaction,
  retention policies, SAML/OIDC SSO, VPC/private deploy, Helm + Terraform.

### D — Scale (defer until a customer's load demands it)
- Replication (leader/follower), read replicas, automatic failover, sharding,
  consistent hashing, cross-AZ, rolling upgrades.
- Until then: **Project-per-node placement** (B2) is the scaling unit. A single
  node already does sub-ms recall at 10k docs; vertical + placement gets you far.

---

## 8. Prioritized roadmap

| Priority | Milestone | Contents |
|----------|-----------|----------|
| **P0** | Production-safe data plane (Phase A) | `duxx-tenant`, namespace isolation across all primitives, capability RBAC, durable ToolCache/Session, per-workspace snapshot/restore/drop, async API |
| **P1** | Minimal cloud (Phase B1) | control plane (orgs/projects/keys), JWT issuance + validation, namespace-per-project provisioning, metering→Stripe, Cloud Console MVP |
| **P2** | Studio + dedicated nodes (B2/C) | Duxx Studio dashboards, dedicated-node provisioning, quota enforcement UI |
| **P3** | Enterprise readiness (C) | encryption-at-rest, KMS/BYOK, SSO, audit aggregation, SOC2 controls, Helm/Terraform |
| **P4** | Scale (D) | replication, read replicas, failover, sharding |

**Critical path:** P0 is a hard gate and is not parallelizable with the rest —
everything in P1+ assumes the data plane can isolate tenants.

---

## 9. The reference prototype

`crates/duxx-tenant` implements the **A1 reference pattern** end-to-end on
MemoryStore: `Namespace` → per-workspace isolated store routing →
capability-checked, leak-proof access, with negative tests proving Tenant A
cannot read Tenant B's memory. Roll this same pattern across every primitive in
A2–A5.

See `crates/duxx-tenant/src/lib.rs` and its tests.

---

## 10. Implementation status

### ✅ Landed

- **`duxx-tenant` crate (A1).** `Namespace` (org/project/env), `Capability`/`Role`
  model, `Workspace` (isolated `MemoryStore` + `SessionStore` + **all six Phase 7
  registries**), `TenantRouter` with lazy per-namespace creation and configurable
  per-workspace capacity. 6 unit tests + doctest prove memory/session/env
  isolation and capability enforcement.
- **Server routing via per-request workspace view (A1 + A2).** `duxx-server`'s
  `Server::workspace_view` resolves the caller's `Workspace` from the
  **authenticated** principal's tenant (not a wire argument) and dispatches the
  command on a shallow `Server` clone whose eight agent-primitive fields point
  at the workspace's isolated stores. So **every** handler — memory, session,
  AND Phase 7 (`TRACE/PROMPT/DATASET/EVAL/REPLAY/COST`) — is isolated by one
  mechanism with **zero per-handler routing code**. The leaky
  `apply_tenant_scope` key-prefixing for memory/session was removed.
- **A2 — Phase 7 usable by tenants, isolated.** `tenant_safe_command` was widened
  (prefix match on the registry families) so tenant credentials can now *use*
  Prompts/Datasets/Eval/Replay/Trace/Cost instead of being blocked — each routed
  to its own workspace registry. Role checks still apply on top.
- **A4 — durability.** `TenantRouter::with_root` opens each workspace under
  `root/<org>/<project>/<env>/`: `MemoryStore::open_at` (rows + HNSW + tantivy)
  plus one `redb` file per Phase 7 registry. Wired into the server via
  `Server::with_tenant_root` and the `--tenants-dir` / `DUXX_TENANTS_DIR` flag.
  A durable-open failure falls back to in-memory for that run (isolation
  preserved) rather than crashing.
- **A5 (drop) — GDPR delete / deprovision.** `drop_workspace` evicts the
  workspace and removes its on-disk directory (idle-safe: the `Arc` drops and
  releases `redb`/HNSW locks before the delete).
- **Bounded warm cache + per-tenant quota.** `with_max_warm` evicts *idle*
  workspaces (strong-count 1, so no lock conflict) past a bound, reopening on
  next access; the per-workspace `capacity` doubles as the per-tenant memory
  quota (`hnsw_rs` can't grow past it).
- **A3 (partial) — richer role names.** `--auth-key` now accepts
  `owner|developer|evaluator|service|observer` (mapped onto read/write/admin
  enforcement).
- Tests: `tenant_recall_is_isolated_across_workspaces`,
  `tenant_scoped_principal_routes_to_isolated_workspace`,
  `tenant_can_use_phase7_prompts_and_is_isolated`,
  `tenant_data_is_durable_across_server_restart`, plus tenant-crate
  `durable_workspace_persists_across_reopen`, `drop_workspace_removes_directory`,
  `bounded_warm_cache_evicts_idle_but_persists`. Full suite: **99 server tests
  + 9 tenant tests + doctest** green.

> **Security note:** this closes a real v0.2.2 leak — `MemoryStore::recall`
> ignores its `key` and searches the whole shared index, so the old
> `--auth-key …:tenant` scheme returned cross-tenant hits on `RECALL`.

**Phase A is functionally complete.** What remains is refinement, not a gate:

- **A3 (finish) — fine-grained capability enforcement.** Per-operation
  capabilities so e.g. an Evaluator can write evals but not delete memory
  (today the five new role names collapse onto read/write/admin).
- **A5 (finish) — snapshot/restore.** Programmatic per-workspace snapshot/restore
  (the workspace dir is self-contained, so a cold dir copy already works); a
  per-workspace reactive bus so tenant pub/sub is isolated (still prefix-scoped).
- **Durable ToolCache + Session** (session is deliberately ephemeral; ToolCache
  could be persisted).

### Phase B — control plane (kickoff landed)

**`duxx-control` crate** — the lightweight system of record for the managed
cloud (no data-plane engine deps; metadata only). Implemented + tested:

- **Org / Project CRUD**, with validation and lookups.
- **API keys**: issue / rotate / revoke / `authenticate` → `ResolvedPrincipal`
  (`org/project/env` + role).
- **Provisioning**: `place_project` (shared vs dedicated node) and
  `data_plane_auth_entries(node)` — the literal `--auth-key` lines a node serves.
- **Usage metering** + a `BillingSink` trait (Stripe slots in behind it) and
  `flush_billing`.
- A `duxx-control` **demo binary** that runs the whole flow and prints the node's
  `--auth-key` args.

**Two proven integration paths between control plane and data plane:**

1. **Static catalog** (string contract): the control plane emits
   `key:secret:role:org/project/env`; the data-plane test
   `control_plane_org_project_env_credentials_isolate` shows two such
   credentials route to distinct isolated workspaces.
2. **Signed short-lived JWTs** (the modern path) — landed this increment:
   - **`duxx-token`** crate: HS256 sign/verify over `Claims {sub, org, project,
     env, role, exp, iat}` via the maintained `jsonwebtoken` crate (no
     hand-rolled crypto). Shared by both planes so the claim shape can't drift.
   - **`ControlPlane::with_signing_key` + `mint_jwt`**: exchange a long-lived
     API-key secret for a short-lived signed JWT scoped to its workspace.
   - **`duxx-server --jwt-secret` / `Server::with_jwt_secret`**: `AUTH` verifies
     the signature and resolves the tenant straight from the claims; a non-JWT
     token falls through to the static catalog. Expired/tampered/wrong-secret
     tokens are rejected.
   - Server test `jwt_auth_resolves_tenant_from_claims_and_isolates` proves
     end-to-end auth + isolation + rejection. The `duxx-control` demo binary
     mints a real JWT.

**HTTP API — the control plane is now a runnable service.** `duxx_control::api`
exposes the lifecycle over HTTP (`hyper`): `POST /v1/orgs`, `/v1/projects`,
`/v1/keys`, `/v1/keys/{revoke,rotate}`, `/v1/tokens` (mint a JWT),
`/v1/placements`, `/v1/auth-entries`, `/v1/usage[/query]`, `GET /healthz`.
Routing is a **pure, synchronous `route()`** function (request→response) with a
thin async wrapper, so the whole surface is unit-tested without sockets. Run it
with `duxx-control serve [addr]`; verified live end-to-end (org→project→key→JWT→
placement→auth-entries over real HTTP).

Tests this increment: **duxx-token 4 · duxx-control 10 · duxx-server 101** (all
green).

Still deferred in B (each needs an external service or a frontend — not faked):
**durable Postgres persistence** (the `ControlPlane` API is storage-agnostic — a
`Store` trait slots a DB behind it), **SSO (OIDC/SAML)**, invitations, a **real
Stripe** `BillingSink`, asymmetric **RS256/ES256** tokens (nodes hold only the
public key — a drop-in `EncodingKey`/`DecodingKey` swap), and the **Cloud
Console** web UI.

### Phase C — Studio (kickoff landed)

**Studio read-API** (`duxx_server::studio`) — the read-only backend a future
Studio web UI renders. Every request is authenticated by a control-plane
**workspace JWT** (`Authorization: Bearer`, verified with the same
`--jwt-secret`) and scoped to exactly that token's `(org, project, env)` — the
namespace comes from the verified claims, never the request, so no tenant can
read another's data.

- **All eight read views** over the Phase 7 data, each scoped to the JWT's
  workspace: `healthz` (open), `overview`, `memory?q=&k=` (hybrid recall),
  `cost` (`{total_usd, by_model}`), `evals` (`{runs}`), `datasets`, `replay`
  (`{sessions}`), `traces?trace_id=` (`{spans}`). Routing is the same
  pure-`route()` + thin-`hyper`-wrapper pattern as the control-plane API, so the
  whole surface is unit-tested without sockets.
- Wired via `--studio-addr HOST:PORT` (separate listener, like metrics).
- Tests: `studio_reads_only_the_callers_workspace` (all views + cross-tenant
  isolation), `studio_auth_is_enforced`, `studio_without_jwt_secret_is_unavailable`;
  live-verified (health open, protected routes 401 without a token).

**Web UIs (kickoff landed).** Two zero-build single-file SPAs (vanilla HTML/JS),
embedded with `include_str!` and served directly by the Rust services — no
frontend toolchain:

- **Control Console** (`duxx-control` `GET /`): create/list orgs → projects →
  keys, issue keys (secret shown once), and mint a workspace JWT to copy into
  Studio. Backed by new list endpoints `GET /v1/orgs`, `POST /v1/projects/list`,
  `POST /v1/keys/list` (secrets redacted).
- **Studio** (`duxx-server` `GET /` on `--studio-addr`): paste a JWT → tabs for
  overview, memory search, cost, evals, datasets, replay, traces — each calling
  the `/studio/*` JSON API with `Authorization: Bearer`.

Live-verified end to end: the Control Console mints a JWT, Studio verifies it and
renders the scoped workspace; both pages serve (~7.4 KB / ~7.5 KB).

Still to build for C: a production frontend (the kickoff SPAs are functional, not
polished) and **enterprise governance** (encryption-at-rest, KMS/BYOK, SOC2,
audit aggregation, Helm/Terraform).

### Capstone: the full A→B→C vertical, proven end to end

`end_to_end_control_plane_to_data_plane_to_studio` (in `duxx-server`, with
`duxx-control` as a dev-dependency) exercises the entire system as one:

1. **(B)** `ControlPlane` creates an org → project → key and `mint_jwt`s a
   short-lived workspace token.
2. **(A)** `Server::with_jwt_secret` verifies that JWT at `AUTH` and routes a
   `REMEMBER` to the isolated, durable workspace.
3. **(C)** `studio::route` reads that exact workspace back via the same JWT —
   correct tenant, the memory recalled.

One shared secret, one set of claims, three layers — all connected through
verified contracts, no faking.

### Hardening round (A3 / B / C / D)

A pass across all four phases:

- **A3 finish — fine-grained capabilities.** `security::Capabilities`: a
  per-family permission set checked *on top of* the coarse read/write/admin
  gate. Base roles get the full set for their level (unchanged behavior); richer
  role names are narrowed — an `evaluator` may run evals but not write or delete
  memory, an `observer` is read-only. Applied to both `--auth-key` and JWT-claim
  principals. Test: `fine_grained_capabilities_restrict_rich_roles`.
- **B — RS256/EdDSA asymmetric JWTs.** `duxx-token` adds Ed25519
  `generate_ed25519` / `sign_ed25519` / `verify_ed25519` (via `ring` + the
  `jsonwebtoken` EdDSA path — DER keys, no PEM, no hand-rolled crypto).
  `ControlPlane::with_ed25519` signs with the private key; `Server`/Studio
  verify with **only the public key** (`with_jwt_public_key`) — a compromised
  node cannot mint tokens. Tokens from a different keypair are rejected. Tests in
  all three crates incl. `ed25519_jwt_auth_works_with_public_key_only`. **Wired
  through the binaries**: `duxx-control keygen DIR` writes the keypair,
  `DUXX_CONTROL_ED25519_KEY` selects EdDSA signing for `serve`, and
  `duxx-server --jwt-public-key PATH` loads the public key — verified live.
- **B — Postgres `Store` trait.** `ControlPlane` persistence is now behind a
  `Store` trait (`InMemoryStore` is the default; `ControlPlane::with_store` takes
  any impl). Compound mutations (`insert_key`/`revoke_key`/`rotate_key`/
  `accept_invite`) are single trait methods so a SQL backend maps each to one
  transaction. All control-plane behavior is unchanged through the seam (13
  tests, incl. `control_plane_runs_over_a_pluggable_store`). A real Postgres
  impl is now a drop-in, not a refactor.
- **B — members & invitations.** `ControlPlane::invite_member` / `accept_invite`
  (single-use token) / `list_members` / `remove_member`, with HTTP endpoints
  `POST /v1/members/{invite,accept,list,remove}`. Real multi-user orgs.
- **C — audit trail.** `security::AuditTrail`: a bounded, always-on,
  per-tenant-queryable ring buffer (retention = its bound), recorded alongside
  the optional file log. Surfaced as Studio `GET /studio/audit` + an **Audit
  tab** in the UI — scoped to the caller's tenant only.
- **D — replication scaffold** (`duxx-cluster`, **scaffold not production**): the
  core primitive — a sequenced [`ChangeLog`] a leader appends to and a
  [`Follower`] tails to converge, plus a round-robin `ReadRouter`. Convergence +
  resume + idempotency proven by tests. Real transport/failover/quorum is the
  remaining (weeks-of-work) part.

### Still open

All remaining work needs an **external service**, a **frontend**, or a
**multi-week distributed-systems** effort — none is in-repo Rust:

- 🟡 a real **Stripe** `BillingSink`, **OIDC/SAML SSO**, **encryption-at-rest /
  KMS / BYOK**, and a **`PostgresStore`** impl behind the now-existing `Store`
  trait (needs a running DB).
- 🔵 a **production frontend** (the Console + Studio SPAs are functional kickoffs).
- 🔴 **production-grade Phase D** — real replication transport, failover/leader
  election, sharding, read-quorum (the `duxx-cluster` scaffold pins the
  primitive; the hard part remains).

**Every in-repo Rust item across A–D is now complete and tested.**

---

## Test totals (this whole effort)

`duxx-token 5 · duxx-tenant 9 + doctest · duxx-control 13 · duxx-cluster 5 +
doctest · duxx-server 107` (**139 total**) — all green (`duxx-server` with
`-- --test-threads=3` to avoid the CI box's parallel-HNSW allocation ceiling).
