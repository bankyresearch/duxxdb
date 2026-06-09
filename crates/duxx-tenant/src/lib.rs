//! # duxx-tenant — multi-tenant workspace routing (Phase A reference prototype)
//!
//! This crate is the **reference pattern** for making DuxxDB's data plane
//! safe to host multiple customers on one process. It demonstrates the
//! approach end-to-end on [`MemoryStore`]; rolling it across the other
//! primitives (Session, ToolCache, Prompts, Datasets, Eval, Replay, Cost,
//! Trace) is mechanical once the shape here is agreed.
//!
//! ## The problem it solves
//!
//! In v0.2.2, tenant isolation is *logical*: the RESP server rewrites a
//! tenant-scoped principal's key argument to `"<tenant>:<key>"`. But
//! [`MemoryStore::recall`] **ignores the key** and runs a hybrid search over
//! the single shared HNSW + tantivy index — so a tenant-scoped `RECALL`
//! returns hits from *every* tenant's memory. Key-prefixing cannot fix this
//! because the leak is at the index layer, not the row layer.
//!
//! ## The fix: per-workspace physical isolation + namespace routing
//!
//! 1. The isolation boundary is a [`Namespace`] = `(org, project, env)`.
//! 2. Each namespace gets its **own** [`Workspace`] with its **own**
//!    `MemoryStore` (and thus its own HNSW + tantivy index). Recall over one
//!    workspace's index physically cannot see another's rows.
//! 3. Every operation is routed by the **authenticated** [`Principal`]'s
//!    namespace — there is *no* tenant/namespace argument on the wire to
//!    forge.
//! 4. Every operation is gated by a [`Capability`] derived from the
//!    principal's [`Role`].
//!
//! ```
//! use std::sync::Arc;
//! use duxx_embed::HashEmbedder;
//! use duxx_tenant::{TenantRouter, Principal, Namespace, Env, Role};
//!
//! let router = TenantRouter::new(Arc::new(HashEmbedder::new(32)));
//!
//! let alice = Principal::new("svc-a", Namespace::new("org_1", "proj_a", Env::Prod), Role::Developer);
//! let bob   = Principal::new("svc-b", Namespace::new("org_1", "proj_b", Env::Prod), Role::Developer);
//!
//! router.remember(&alice, "u1", "alice's secret wallet recovery phrase").unwrap();
//! router.remember(&bob,   "u1", "bob unrelated note about the weather").unwrap();
//!
//! // Bob recalls within his own workspace — Alice's data is unreachable.
//! let hits = router.recall(&bob, "wallet recovery", 5).unwrap();
//! assert!(hits.iter().all(|h| !h.memory.text.contains("alice")));
//! ```

use duxx_cost::CostLedger;
use duxx_datasets::DatasetRegistry;
use duxx_docs::{DocumentStore, LocalFsConnector};
use duxx_embed::Embedder;
use duxx_eval::EvalRegistry;
use duxx_memory::{MemoryHit, MemoryStore, SessionStore};
use duxx_prompts::PromptRegistry;
use duxx_replay::ReplayRegistry;
use duxx_storage::{open_backend, Backend};
use duxx_trace::TraceStore;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default per-workspace vector-index capacity.
///
/// `hnsw_rs` pre-allocates this and CANNOT grow past it, so with many
/// resident workspaces a large default would both waste memory and cap each
/// tenant too high to be meaningful. Production sets this per workspace from
/// the control plane's per-project quota via [`TenantRouter::with_capacity`];
/// this default is a sane starting point for a single warm workspace.
const DEFAULT_WORKSPACE_CAPACITY: usize = 16_384;

// ---------------------------------------------------------------------------
// Namespace — the isolation boundary
// ---------------------------------------------------------------------------

/// Deployment environment within a project. Separate environments get
/// separate keyspaces and quotas so `dev` traffic can never read `prod`
/// memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Env {
    Dev,
    Staging,
    Prod,
}

impl Env {
    pub fn as_str(self) -> &'static str {
        match self {
            Env::Dev => "dev",
            Env::Staging => "staging",
            Env::Prod => "prod",
        }
    }

    /// Parse an environment label, defaulting to [`Env::Prod`] for
    /// anything unrecognized (fail safe — prod is the most restricted).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "dev" | "development" => Env::Dev,
            "staging" | "stage" => Env::Staging,
            _ => Env::Prod,
        }
    }
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The isolation boundary: `(org, project, env)`. Resolved from the
/// authenticated credential — NEVER from a client-supplied request argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace {
    pub org_id: Arc<str>,
    pub project_id: Arc<str>,
    pub env: Env,
}

impl Namespace {
    pub fn new(org_id: impl Into<Arc<str>>, project_id: impl Into<Arc<str>>, env: Env) -> Self {
        Self {
            org_id: org_id.into(),
            project_id: project_id.into(),
            env,
        }
    }

    /// Stable byte prefix for this namespace. Used to scope every
    /// `Backend`-backed primitive (Prompts/Datasets/…); for `MemoryStore`
    /// the prefix is also the on-disk directory key. NUL-separated to match
    /// `duxx_storage::backend::key`.
    pub fn prefix(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(self.org_id.as_bytes());
        out.push(0);
        out.extend_from_slice(self.project_id.as_bytes());
        out.push(0);
        out.extend_from_slice(self.env.as_str().as_bytes());
        out.push(0);
        out
    }

    /// Human-readable path form, e.g. `org_1/proj_a/prod`.
    pub fn path(&self) -> String {
        format!("{}/{}/{}", self.org_id, self.project_id, self.env)
    }

    /// On-disk directory for this workspace under `root`, built component by
    /// component so it is correct on every platform (never relies on `/`
    /// inside a single path component).
    pub fn dir(&self, root: &Path) -> PathBuf {
        root.join(&*self.org_id)
            .join(&*self.project_id)
            .join(self.env.as_str())
    }

    /// Bridge a credential's flat tenant string into a namespace, so the
    /// existing `--auth-key principal:token:role:tenant` flow keeps working
    /// while the control plane (which will issue full `org/project/env`
    /// claims) is built. Accepted forms:
    ///
    /// * `org/project/env` → exactly that
    /// * `org/project`     → env defaults to `prod`
    /// * `project`         → org defaults to `default`, env to `prod`
    pub fn parse(s: &str) -> Self {
        let parts: Vec<&str> = s
            .split('/')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        match parts.as_slice() {
            [org, project, env] => Namespace::new(*org, *project, Env::parse(env)),
            [org, project] => Namespace::new(*org, *project, Env::Prod),
            [project] => Namespace::new("default", *project, Env::Prod),
            // Empty or malformed: treat the whole string as a project under
            // the default org. Never panics — isolation must not depend on
            // well-formed input.
            _ => Namespace::new("default", s, Env::Prod),
        }
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path())
    }
}

// ---------------------------------------------------------------------------
// Capabilities & Roles
// ---------------------------------------------------------------------------

/// A single permission checked at each operation. Replaces the coarse
/// `ReadOnly/ReadWrite/Admin` command gate with something a real RBAC
/// policy engine can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    ReadMemory,
    WriteMemory,
    DeleteMemory,
    SearchMemory,
    /// Destroy an entire workspace (GDPR delete, deprovision).
    ManageProject,
}

/// Coarse role that expands into a fixed [`Capability`] set. Production
/// would let an Owner define custom roles; this is the built-in baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Owner,
    Admin,
    Developer,
    Evaluator,
    Observer,
    /// The credential an agent runtime uses in production.
    ServiceAccount,
    /// End user scoped to their own data within an agent sub-namespace.
    TenantUser,
}

impl Role {
    /// Whether this role grants `cap`.
    pub fn allows(self, cap: Capability) -> bool {
        use Capability::*;
        use Role::*;
        match self {
            // Full control.
            Owner | Admin => true,
            // Read/write data, no project destruction.
            Developer | ServiceAccount => {
                matches!(cap, ReadMemory | WriteMemory | DeleteMemory | SearchMemory)
            }
            // Can write/search (to run evals over memory) but not delete.
            Evaluator => matches!(cap, ReadMemory | SearchMemory | WriteMemory),
            // Read-only.
            Observer => matches!(cap, ReadMemory | SearchMemory),
            // End user: read/write/search own data, never delete projects.
            TenantUser => matches!(cap, ReadMemory | WriteMemory | SearchMemory),
        }
    }
}

/// An authenticated caller. Its [`Namespace`] is the routing key for every
/// operation, so there is no way to address another tenant's data from the
/// request body.
#[derive(Debug, Clone)]
pub struct Principal {
    pub name: Arc<str>,
    pub ns: Namespace,
    pub role: Role,
}

impl Principal {
    pub fn new(name: impl Into<Arc<str>>, ns: Namespace, role: Role) -> Self {
        Self {
            name: name.into(),
            ns,
            role,
        }
    }

    fn require(&self, cap: Capability) -> Result<(), TenantError> {
        if self.role.allows(cap) {
            Ok(())
        } else {
            Err(TenantError::Denied {
                principal: self.name.to_string(),
                capability: cap,
                ns: self.ns.to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TenantError {
    #[error("permission denied: principal '{principal}' lacks {capability:?} in namespace '{ns}'")]
    Denied {
        principal: String,
        capability: Capability,
        ns: String,
    },
    #[error(transparent)]
    Core(#[from] duxx_core::Error),
}

// ---------------------------------------------------------------------------
// Workspace — one isolated set of primitives per namespace
// ---------------------------------------------------------------------------

/// One tenant's isolated data plane: its own copy of every agent primitive.
///
/// The crucial property is per-workspace **physical** isolation. Each
/// `Workspace` owns its own `MemoryStore` (own HNSW + tantivy index), its own
/// session store, and its own Phase 7 registries. A query over this
/// workspace's stores cannot return another workspace's rows, because they
/// were never inserted into these stores.
///
/// Primitives are in-memory in this Phase-A increment, matching the rest of
/// the prototype. Durable per-workspace backends (one `phase7.redb` +
/// `MemoryStore::open_at` under `root/<org>/<project>/<env>/`) are Phase A4.
pub struct Workspace {
    pub ns: Namespace,
    memory: MemoryStore,
    sessions: SessionStore,
    traces: TraceStore,
    prompts: PromptRegistry,
    datasets: DatasetRegistry,
    evals: EvalRegistry,
    replays: ReplayRegistry,
    costs: CostLedger,
    docs: DocumentStore,
}

impl Workspace {
    fn new(ns: Namespace, embedder: Arc<dyn Embedder>, capacity: usize) -> Self {
        let dim = embedder.dim();
        Self {
            ns,
            // In production: MemoryStore::open_at(embedder, dim, cap,
            // root.join(ns.path())) so the workspace is durable and can be
            // snapshotted/restored/dropped independently.
            memory: MemoryStore::with_capacity(dim, capacity),
            sessions: SessionStore::new(),
            traces: TraceStore::new(),
            prompts: PromptRegistry::new(embedder.clone()),
            datasets: DatasetRegistry::new(embedder.clone()),
            evals: EvalRegistry::new(embedder.clone()),
            replays: ReplayRegistry::new(),
            costs: CostLedger::new(embedder.clone()),
            // Document layer: object bytes live in the connector's store;
            // chunks/embeddings/citations live here. In-memory connector default.
            docs: DocumentStore::new(Arc::new(LocalFsConnector::new(".")), embedder),
        }
    }

    /// Durable workspace under `dir`. Every primitive that supports
    /// persistence is opened on disk so the tenant's data survives restart:
    ///
    /// * `MemoryStore::open_at` → `dir/memory/` (rows + HNSW + tantivy)
    /// * one `redb` file per Phase 7 registry (`prompts.redb`, …) + a HNSW
    ///   dump dir for the four registries that carry a vector index
    ///
    /// `SessionStore` stays in-memory by design — it is a sliding-TTL hot
    /// cache of per-conversation scratch state, not a system of record.
    fn open(
        ns: Namespace,
        embedder: Arc<dyn Embedder>,
        capacity: usize,
        dir: &Path,
    ) -> duxx_core::Result<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| duxx_core::Error::Storage(format!("create workspace dir {dir:?}: {e}")))?;
        let dim = embedder.dim();

        // One redb file per registry: redb takes an exclusive lock per
        // database handle, so the six registries cannot share one file.
        let backend = |name: &str| -> duxx_core::Result<Arc<dyn Backend>> {
            let spec = format!("redb:{}", dir.join(name).display());
            Ok(Arc::from(open_backend(Some(&spec))?))
        };
        let hnsw = |name: &str| Some(dir.join(name));

        // Each registry has its own error type; normalize to a storage error.
        // A nested generic fn (closures can't be generic over the error type).
        fn storage_err<E: std::fmt::Display>(
            what: &'static str,
        ) -> impl FnOnce(E) -> duxx_core::Error {
            move |e| duxx_core::Error::Storage(format!("open {what}: {e}"))
        }

        Ok(Self {
            memory: MemoryStore::open_at(dim, capacity, dir.join("memory"))?,
            sessions: SessionStore::new(),
            traces: TraceStore::open(backend("traces.redb")?).map_err(storage_err("traces"))?,
            prompts: PromptRegistry::open_with_index_dir(
                embedder.clone(),
                backend("prompts.redb")?,
                hnsw("prompts.hnsw").as_deref(),
            )
            .map_err(storage_err("prompts"))?,
            datasets: DatasetRegistry::open_with_index_dir(
                embedder.clone(),
                backend("datasets.redb")?,
                hnsw("datasets.hnsw").as_deref(),
            )
            .map_err(storage_err("datasets"))?,
            evals: EvalRegistry::open_with_index_dir(
                embedder.clone(),
                backend("evals.redb")?,
                hnsw("evals.hnsw").as_deref(),
            )
            .map_err(storage_err("evals"))?,
            replays: ReplayRegistry::open(backend("replays.redb")?)
                .map_err(storage_err("replays"))?,
            costs: CostLedger::open_with_index_dir(
                embedder.clone(),
                backend("costs.redb")?,
                hnsw("costs.hnsw").as_deref(),
            )
            .map_err(storage_err("costs"))?,
            // Source object bytes are served from `dir/objects`; chunks/index
            // are in-process. (The chunk index itself is not yet persisted —
            // a follow-up like the Phase 7 registries.)
            docs: DocumentStore::new(
                Arc::new(LocalFsConnector::new(dir.join("objects"))),
                embedder,
            ),
            ns,
        })
    }

    /// Read handle to this workspace's memory store. Callers must have
    /// already passed a capability check (the router does this).
    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    /// Read handle to this workspace's session store.
    pub fn sessions(&self) -> &SessionStore {
        &self.sessions
    }

    /// Read handle to this workspace's trace store.
    pub fn traces(&self) -> &TraceStore {
        &self.traces
    }

    /// Read handle to this workspace's prompt registry.
    pub fn prompts(&self) -> &PromptRegistry {
        &self.prompts
    }

    /// Read handle to this workspace's dataset registry.
    pub fn datasets(&self) -> &DatasetRegistry {
        &self.datasets
    }

    /// Read handle to this workspace's eval registry.
    pub fn evals(&self) -> &EvalRegistry {
        &self.evals
    }

    /// Read handle to this workspace's replay registry.
    pub fn replays(&self) -> &ReplayRegistry {
        &self.replays
    }

    /// Read handle to this workspace's document store.
    pub fn docs(&self) -> &DocumentStore {
        &self.docs
    }

    /// Read handle to this workspace's cost ledger.
    pub fn costs(&self) -> &CostLedger {
        &self.costs
    }
}

// ---------------------------------------------------------------------------
// TenantRouter — resolves & caches a Workspace per Namespace
// ---------------------------------------------------------------------------

/// Routes every operation to the caller's [`Workspace`], creating it lazily
/// on first use. This is the single choke point through which all data-plane
/// traffic flows, which is exactly why isolation is enforceable here.
///
/// Production differences (documented, not implemented in this prototype):
/// * `workspaces` becomes a bounded LRU of *warm* stores; cold workspaces
///   drop their in-memory indices and reload from disk on next access.
/// * Construction takes a `root: PathBuf` and opens each workspace durably.
/// * A `QuotaTable` enforces per-workspace row/byte/QPS ceilings (returns
///   an `OverQuota` error) using limits pushed down by the control plane.
pub struct TenantRouter {
    embedder: Arc<dyn Embedder>,
    /// Per-workspace vector-index capacity (a hard `hnsw_rs` cap). This also
    /// serves as the per-tenant memory **quota**: `hnsw_rs` cannot grow past
    /// it, so a workspace holds at most `capacity` memories. Set from the
    /// per-project quota in production.
    capacity: usize,
    /// When `Some`, workspaces are **durable** under this root
    /// (`root/<org>/<project>/<env>/`); when `None`, in-memory.
    root: Option<PathBuf>,
    /// Optional bound on resident ("warm") workspaces. Only honored in
    /// durable mode — an idle workspace can be evicted and transparently
    /// reopened on next access. Never evicts in-memory mode (would lose
    /// data).
    max_warm: Option<usize>,
    workspaces: RwLock<HashMap<Namespace, Arc<Workspace>>>,
}

impl TenantRouter {
    /// In-memory router with the default per-workspace capacity.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self::with_capacity(embedder, DEFAULT_WORKSPACE_CAPACITY)
    }

    /// In-memory router with an explicit per-workspace vector-index capacity.
    pub fn with_capacity(embedder: Arc<dyn Embedder>, capacity: usize) -> Self {
        Self {
            embedder,
            capacity,
            root: None,
            max_warm: None,
            workspaces: RwLock::new(HashMap::new()),
        }
    }

    /// **Durable** router: every workspace persists under
    /// `root/<org>/<project>/<env>/` and survives restart. This is the
    /// production constructor.
    pub fn with_root(
        embedder: Arc<dyn Embedder>,
        capacity: usize,
        root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            embedder,
            capacity,
            root: Some(root.into()),
            max_warm: None,
            workspaces: RwLock::new(HashMap::new()),
        }
    }

    /// Bound the number of resident workspaces. Idle ones (no in-flight
    /// request holding a handle) are evicted past this bound and reopened on
    /// next access. No-op in in-memory mode. Builder style.
    pub fn with_max_warm(mut self, max_warm: usize) -> Self {
        self.max_warm = Some(max_warm);
        self
    }

    /// Whether workspaces are persisted to disk.
    pub fn is_durable(&self) -> bool {
        self.root.is_some()
    }

    /// Build the workspace for `ns` — durable when a root is configured,
    /// else in-memory. A durable-open failure is logged and falls back to an
    /// in-memory workspace for that run (isolation is preserved; durability
    /// is not), so a single bad disk path can't crash the whole server.
    fn build_workspace(&self, ns: &Namespace) -> Workspace {
        match &self.root {
            Some(root) => Workspace::open(
                ns.clone(),
                self.embedder.clone(),
                self.capacity,
                &ns.dir(root),
            )
            .unwrap_or_else(|e| {
                tracing::error!(
                    namespace = %ns,
                    error = %e,
                    "durable workspace open failed; serving in-memory (NOT durable) this run"
                );
                Workspace::new(ns.clone(), self.embedder.clone(), self.capacity)
            }),
            None => Workspace::new(ns.clone(), self.embedder.clone(), self.capacity),
        }
    }

    /// Evict idle workspaces past `max_warm`. Only an entry whose `Arc`
    /// strong-count is 1 (nobody mid-request holds it) is removed, so the
    /// `redb`/HNSW file locks are released by `Drop` before any reopen — no
    /// lock conflict. In-memory mode is never evicted.
    fn maybe_evict(
        &self,
        guard: &mut std::collections::HashMap<Namespace, Arc<Workspace>>,
        keep: &Namespace,
    ) {
        let Some(max) = self.max_warm else { return };
        if self.root.is_none() || guard.len() <= max {
            return;
        }
        let evictable: Vec<Namespace> = guard
            .iter()
            .filter(|(ns, ws)| *ns != keep && Arc::strong_count(ws) == 1)
            .map(|(ns, _)| ns.clone())
            .collect();
        for ns in evictable {
            if guard.len() <= max {
                break;
            }
            guard.remove(&ns); // Drop flushes + releases the file lock.
        }
    }

    /// Number of distinct workspaces currently resident.
    pub fn workspace_count(&self) -> usize {
        self.workspaces.read().len()
    }

    /// Resolve (lazily creating) the workspace for `ns`. Two principals in
    /// the same namespace share the same `Arc<Workspace>`.
    ///
    /// Public so the server can route raw store handles after running its
    /// own authorization gate; the capability-checked helpers below are the
    /// preferred API for library callers.
    pub fn workspace(&self, ns: &Namespace) -> Arc<Workspace> {
        if let Some(ws) = self.workspaces.read().get(ns) {
            return ws.clone();
        }
        let mut guard = self.workspaces.write();
        // Double-checked: another thread may have created it between the
        // read-unlock and the write-lock.
        let ws = guard
            .entry(ns.clone())
            .or_insert_with(|| Arc::new(self.build_workspace(ns)))
            .clone();
        self.maybe_evict(&mut guard, ns);
        ws
    }

    // -- Capability-checked, namespace-routed operations --------------------

    /// Store a memory in the caller's workspace.
    pub fn remember(&self, p: &Principal, key: &str, text: &str) -> Result<u64, TenantError> {
        p.require(Capability::WriteMemory)?;
        let embedding = self.embedder.embed(text)?;
        let ws = self.workspace(&p.ns);
        let id = ws.memory.remember(key, text, embedding)?;
        Ok(id)
    }

    /// Hybrid recall **within the caller's workspace only**. Because the
    /// workspace owns its own index, this cannot leak across tenants — the
    /// exact failure mode of the v0.2.2 shared-index scheme.
    pub fn recall(
        &self,
        p: &Principal,
        query: &str,
        k: usize,
    ) -> Result<Vec<MemoryHit>, TenantError> {
        p.require(Capability::SearchMemory)?;
        let qvec = self.embedder.embed(query)?;
        let ws = self.workspace(&p.ns);
        let hits = ws.memory.recall("", query, &qvec, k)?;
        Ok(hits)
    }

    /// Forget a memory by id in the caller's workspace.
    pub fn forget(&self, p: &Principal, id: u64) -> Result<bool, TenantError> {
        p.require(Capability::DeleteMemory)?;
        let ws = self.workspace(&p.ns);
        Ok(ws.memory.forget(id))
    }

    /// Number of memories in the caller's workspace.
    pub fn memory_len(&self, p: &Principal) -> Result<usize, TenantError> {
        p.require(Capability::ReadMemory)?;
        let ws = self.workspace(&p.ns);
        Ok(ws.memory.len())
    }

    /// Destroy a workspace entirely (GDPR delete / deprovision). Requires
    /// `ManageProject`. Evicts it from the warm cache and, in durable mode,
    /// removes its on-disk directory.
    ///
    /// The removed `Arc` is dropped before the directory is deleted, so when
    /// the workspace is idle the `redb`/HNSW file locks are released first
    /// and the delete succeeds. If a request is mid-flight against it, the
    /// directory delete may fail (locked files) — deprovision when idle.
    pub fn drop_workspace(&self, p: &Principal) -> Result<bool, TenantError> {
        p.require(Capability::ManageProject)?;
        let existed = {
            // Scope the guard + removed Arc so both drop before we touch the
            // filesystem.
            self.workspaces.write().remove(&p.ns).is_some()
        };
        if let Some(root) = &self.root {
            let dir = p.ns.dir(root);
            if dir.exists() {
                std::fs::remove_dir_all(&dir).map_err(|e| {
                    duxx_core::Error::Storage(format!("drop workspace dir {dir:?}: {e}"))
                })?;
            }
        }
        Ok(existed)
    }
}

// ---------------------------------------------------------------------------
// Tests — prove isolation and capability enforcement
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn router() -> TenantRouter {
        TenantRouter::new(Arc::new(HashEmbedder::new(32)))
    }

    fn dev(name: &str, org: &str, proj: &str, env: Env) -> Principal {
        Principal::new(name, Namespace::new(org, proj, env), Role::Developer)
    }

    /// THE core test: the v0.2.2 cross-tenant recall leak is gone. Two
    /// projects in the same org each store a memory; neither can recall the
    /// other's, even though `recall` runs a hybrid search.
    #[test]
    fn recall_cannot_cross_tenant_boundary() {
        let r = router();
        let alice = dev("svc-a", "org_1", "proj_a", Env::Prod);
        let bob = dev("svc-b", "org_1", "proj_b", Env::Prod);

        r.remember(&alice, "u1", "alice secret wallet recovery phrase zebra")
            .unwrap();
        r.remember(&bob, "u1", "bob weather note about sunshine zebra")
            .unwrap();

        // Bob queries a term present in BOTH (zebra) plus Alice-only words.
        let bob_hits = r.recall(&bob, "wallet recovery zebra", 10).unwrap();
        assert!(
            bob_hits.iter().all(|h| !h.memory.text.contains("alice")),
            "LEAK: bob recalled alice's memory: {:?}",
            bob_hits.iter().map(|h| &h.memory.text).collect::<Vec<_>>()
        );
        assert!(
            bob_hits.iter().all(|h| !h.memory.text.contains("secret")),
            "LEAK: bob recalled alice's secret"
        );

        // And Alice cannot see Bob's.
        let alice_hits = r.recall(&alice, "weather sunshine zebra", 10).unwrap();
        assert!(alice_hits.iter().all(|h| !h.memory.text.contains("bob")));

        // Each workspace sees exactly its own single row.
        assert_eq!(r.memory_len(&alice).unwrap(), 1);
        assert_eq!(r.memory_len(&bob).unwrap(), 1);
    }

    /// Same project, different environment → still isolated.
    #[test]
    fn environments_are_isolated() {
        let r = router();
        let dev_p = dev("svc", "org_1", "proj_a", Env::Dev);
        let prod_p = dev("svc", "org_1", "proj_a", Env::Prod);

        r.remember(&dev_p, "k", "dev-only scratch data lion")
            .unwrap();
        let prod_hits = r.recall(&prod_p, "scratch lion", 10).unwrap();
        assert!(prod_hits.is_empty(), "dev data leaked into prod");
        assert_eq!(r.memory_len(&prod_p).unwrap(), 0);
        assert_eq!(r.memory_len(&dev_p).unwrap(), 1);
    }

    /// Two principals in the SAME namespace share one workspace.
    #[test]
    fn same_namespace_shares_workspace() {
        let r = router();
        let a = dev("svc-a", "org_1", "proj_a", Env::Prod);
        let b = dev("svc-b", "org_1", "proj_a", Env::Prod);

        r.remember(&a, "k1", "shared project memory tiger").unwrap();
        // b is a different principal but same (org, proj, env): sees a's write.
        let hits = r.recall(&b, "tiger", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(r.workspace_count(), 1);
    }

    /// Capability enforcement: an Observer cannot write or delete.
    #[test]
    fn observer_cannot_mutate() {
        let r = router();
        let ns = Namespace::new("org_1", "proj_a", Env::Prod);
        let observer = Principal::new("dash", ns.clone(), Role::Observer);

        let err = r.remember(&observer, "k", "nope").unwrap_err();
        assert!(matches!(err, TenantError::Denied { .. }), "got {err:?}");

        // Seed a row via an admin, then confirm observer can read but not delete.
        let admin = Principal::new("root", ns, Role::Admin);
        let id = r.remember(&admin, "k", "admin seeded note").unwrap();
        assert!(r.recall(&observer, "note", 5).is_ok());
        assert!(matches!(
            r.forget(&observer, id).unwrap_err(),
            TenantError::Denied { .. }
        ));
    }

    /// Only project-managers can destroy a workspace.
    #[test]
    fn drop_workspace_requires_manage_project() {
        let r = router();
        let ns = Namespace::new("org_1", "proj_a", Env::Prod);
        let developer = Principal::new("dev", ns.clone(), Role::Developer);
        let owner = Principal::new("owner", ns.clone(), Role::Owner);

        r.remember(&developer, "k", "data").unwrap();
        assert!(matches!(
            r.drop_workspace(&developer).unwrap_err(),
            TenantError::Denied { .. }
        ));
        assert!(r.drop_workspace(&owner).unwrap());
        // After drop, the namespace is gone (count back to zero).
        assert_eq!(r.workspace_count(), 0);
    }

    #[test]
    fn namespace_prefix_is_nul_separated_and_distinct() {
        let a = Namespace::new("org_1", "proj_a", Env::Prod);
        let b = Namespace::new("org_1", "proj_b", Env::Prod);
        assert_ne!(a.prefix(), b.prefix());
        assert!(a.prefix().ends_with(&[0]));
        assert_eq!(a.path(), "org_1/proj_a/prod");
    }

    /// A4: a durable workspace survives router restart.
    #[test]
    fn durable_workspace_persists_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let p = Principal::new(
            "svc",
            Namespace::new("org_1", "proj_a", Env::Prod),
            Role::Developer,
        );

        {
            let r = TenantRouter::with_root(Arc::new(HashEmbedder::new(32)), 1024, root);
            assert!(r.is_durable());
            r.remember(&p, "u1", "alice durable wallet zebra").unwrap();
            assert_eq!(r.memory_len(&p).unwrap(), 1);
        } // router dropped → indices flushed to disk

        // Reopen a fresh router on the same root.
        let r2 = TenantRouter::with_root(Arc::new(HashEmbedder::new(32)), 1024, root);
        assert_eq!(
            r2.memory_len(&p).unwrap(),
            1,
            "memory should survive restart"
        );
        let hits = r2.recall(&p, "wallet zebra", 5).unwrap();
        assert!(
            hits.iter().any(|h| h.memory.text.contains("durable")),
            "recall should find the persisted memory"
        );
    }

    /// A5: dropping a workspace deletes its on-disk directory (GDPR delete).
    #[test]
    fn drop_workspace_removes_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let r = TenantRouter::with_root(Arc::new(HashEmbedder::new(32)), 256, root);
        let owner = Principal::new(
            "owner",
            Namespace::new("org_1", "proj_a", Env::Prod),
            Role::Owner,
        );

        r.remember(&owner, "k", "data lion").unwrap();
        let dir = owner.ns.dir(root);
        assert!(dir.exists(), "workspace dir should exist after a write");

        assert!(r.drop_workspace(&owner).unwrap());
        assert!(!dir.exists(), "workspace dir should be deleted after drop");
    }

    /// The bounded warm cache evicts idle workspaces but keeps data durable.
    #[test]
    fn bounded_warm_cache_evicts_idle_but_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let r = TenantRouter::with_root(Arc::new(HashEmbedder::new(32)), 256, tmp.path())
            .with_max_warm(1);
        let a = Principal::new("a", Namespace::new("org", "pa", Env::Prod), Role::Developer);
        let b = Principal::new("b", Namespace::new("org", "pb", Env::Prod), Role::Developer);

        r.remember(&a, "k", "a data lion").unwrap();
        r.remember(&b, "k", "b data tiger").unwrap(); // inserting b evicts idle a

        assert!(
            r.workspace_count() <= 1,
            "warm cache should be bounded to 1, got {}",
            r.workspace_count()
        );

        // a was evicted from the warm cache but its data persisted — a
        // re-access reopens it from disk.
        assert_eq!(
            r.memory_len(&a).unwrap(),
            1,
            "a's data must survive eviction"
        );
    }
}
