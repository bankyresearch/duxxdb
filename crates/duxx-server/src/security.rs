use serde_json::json;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    ReadOnly,
    ReadWrite,
    Admin,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::ReadOnly => "read",
            Role::ReadWrite => "write",
            Role::Admin => "admin",
        }
    }

    fn can(self, required: Role) -> bool {
        matches!(
            (self, required),
            (Role::Admin, _)
                | (Role::ReadWrite, Role::ReadWrite)
                | (Role::ReadWrite, Role::ReadOnly)
                | (Role::ReadOnly, Role::ReadOnly)
        )
    }
}

impl std::str::FromStr for Role {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            // Canonical levels.
            "read" | "readonly" | "read-only" | "ro" => Ok(Role::ReadOnly),
            "write" | "readwrite" | "read-write" | "rw" => Ok(Role::ReadWrite),
            "admin" | "root" => Ok(Role::Admin),
            // Capability-model role names (see duxx-tenant::Role), mapped onto
            // the enforced read/write/admin levels. They let operators express
            // intent today; finer per-capability enforcement (e.g. Evaluator
            // may write evals but not delete memory) is a follow-up.
            "owner" => Ok(Role::Admin),
            "developer" | "dev" | "evaluator" | "service" | "service-account" => {
                Ok(Role::ReadWrite)
            }
            "observer" | "viewer" => Ok(Role::ReadOnly),
            other => anyhow::bail!(
                "unknown role {other:?}; expected read|write|admin \
                 (or owner|developer|evaluator|service|observer)"
            ),
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fine-grained, per-family permissions checked *in addition to* the coarse
/// [`Role`] gate. The base roles (`read`/`write`/`admin`) get the full
/// capability set for their level, so they behave exactly as before; the
/// richer role names ([`Capabilities::for_role_str`]) get narrower sets — e.g.
/// an `evaluator` may run evals but not delete or overwrite memory.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub read: bool,
    pub write_memory: bool,
    pub delete_memory: bool,
    pub write_session: bool,
    pub write_prompt: bool,
    pub write_dataset: bool,
    pub run_eval: bool,
    pub write_trace: bool,
    pub write_replay: bool,
    pub write_cost: bool,
    pub admin: bool,
}

impl Capabilities {
    pub fn all() -> Self {
        Self {
            read: true,
            write_memory: true,
            delete_memory: true,
            write_session: true,
            write_prompt: true,
            write_dataset: true,
            run_eval: true,
            write_trace: true,
            write_replay: true,
            write_cost: true,
            admin: true,
        }
    }

    pub fn none() -> Self {
        Self {
            read: false,
            write_memory: false,
            delete_memory: false,
            write_session: false,
            write_prompt: false,
            write_dataset: false,
            run_eval: false,
            write_trace: false,
            write_replay: false,
            write_cost: false,
            admin: false,
        }
    }

    /// Full data read+write (no admin) — the default for `ReadWrite`.
    fn data_full() -> Self {
        Self {
            read: true,
            write_memory: true,
            delete_memory: true,
            write_session: true,
            write_prompt: true,
            write_dataset: true,
            run_eval: true,
            write_trace: true,
            write_replay: true,
            write_cost: true,
            admin: false,
        }
    }

    /// Read-only — the default for `ReadOnly`.
    fn read_only() -> Self {
        Self {
            read: true,
            ..Self::none()
        }
    }

    /// Default capabilities for a coarse [`Role`] (used when no richer role
    /// name is supplied, so base roles are unchanged).
    pub fn for_level(role: Role) -> Self {
        match role {
            Role::Admin => Self::all(),
            Role::ReadWrite => Self::data_full(),
            Role::ReadOnly => Self::read_only(),
        }
    }

    /// Capabilities for a role *name* (the richer set accepted by
    /// `--auth-key` and JWT claims).
    pub fn for_role_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "admin" | "root" | "owner" => Self::all(),
            "write" | "readwrite" | "read-write" | "rw" | "developer" | "dev" | "service"
            | "service-account" => Self::data_full(),
            // Evaluator: run evals + build eval datasets; cannot write/delete
            // memory or edit prompts.
            "evaluator" => Self {
                read: true,
                run_eval: true,
                write_dataset: true,
                ..Self::none()
            },
            // Observer / read-only: reads only.
            "observer" | "viewer" | "read" | "readonly" | "read-only" | "ro" => Self::read_only(),
            // Unknown → safest (read-only).
            _ => Self::read_only(),
        }
    }

    /// Whether these capabilities permit `command`. Connection/meta commands
    /// are always allowed; read commands need `read`; each write family needs
    /// its specific flag; anything else needs `admin`.
    pub fn allows(&self, command: &str) -> bool {
        if self.admin {
            return true;
        }
        match command {
            "PING" | "HELLO" | "AUTH" | "COMMAND" | "INFO" | "QUIT" | "SUBSCRIBE"
            | "UNSUBSCRIBE" | "PSUBSCRIBE" | "PUNSUBSCRIBE" => true,
            "SET" => self.write_session,
            "REMEMBER" | "REMEMBER.IDEM" | "REMEMBER.BATCH" | "REMEMBER.META" => self.write_memory,
            "DEL" => self.delete_memory,
            // COMPACT physically purges already-deleted rows from the index —
            // the completion of deletion, so gate it on the same capability.
            "COMPACT" => self.delete_memory,
            "FORGET" | "FORGET.KEY" | "FORGET.OLDER" => self.delete_memory,
            "PROMPT.PUT" | "PROMPT.TAG" | "PROMPT.UNTAG" => self.write_prompt,
            "DATASET.CREATE"
            | "DATASET.ADD"
            | "DATASET.TAG"
            | "DATASET.UNTAG"
            | "DATASET.FROM_RECALL" => self.write_dataset,
            "EVAL.START" | "EVAL.SCORE" | "EVAL.COMPLETE" | "EVAL.FAIL" => self.run_eval,
            "TRACE.RECORD" | "TRACE.CLOSE" => self.write_trace,
            "REPLAY.CAPTURE" | "REPLAY.START" | "REPLAY.STEP" | "REPLAY.RECORD"
            | "REPLAY.COMPLETE" | "REPLAY.FAIL" | "REPLAY.SET_TRACE" => self.write_replay,
            "COST.RECORD" | "COST.SET_BUDGET" | "COST.DELETE_BUDGET" => self.write_cost,
            // Documents are knowledge artifacts; gate writes on dataset cap.
            "DOC.INGEST" | "DOC.DELETE" => self.write_dataset,
            // Any remaining read command needs `read`; non-read → admin-only
            // (already returned false above).
            other => required_role(other) == Role::ReadOnly && self.read,
        }
    }
}

#[derive(Clone)]
pub struct Principal {
    pub name: Arc<str>,
    pub token: Arc<str>,
    pub role: Role,
    pub tenant: Option<Arc<str>>,
    pub caps: Capabilities,
}

impl fmt::Debug for Principal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Principal")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("tenant", &self.tenant)
            .finish_non_exhaustive()
    }
}

impl Principal {
    pub fn new(
        name: impl Into<Arc<str>>,
        token: impl Into<Arc<str>>,
        role: Role,
        tenant: Option<impl Into<Arc<str>>>,
    ) -> anyhow::Result<Self> {
        let name = name.into();
        let token = token.into();
        if name.trim().is_empty() {
            anyhow::bail!("auth principal name must not be empty");
        }
        if token.trim().is_empty() {
            anyhow::bail!("auth token for principal {name:?} must not be empty");
        }
        Ok(Self {
            name,
            token,
            role,
            tenant: tenant.map(Into::into),
            caps: Capabilities::for_level(role),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct AuthCatalog {
    principals: Arc<Vec<Principal>>,
}

impl AuthCatalog {
    pub fn from_principals(principals: Vec<Principal>) -> anyhow::Result<Self> {
        if principals.iter().any(|p| p.token.is_empty()) {
            anyhow::bail!("auth catalog contains an empty token");
        }
        Ok(Self {
            principals: Arc::new(principals),
        })
    }

    pub fn from_shared_admin_token(token: impl Into<Arc<str>>) -> anyhow::Result<Self> {
        Self::from_principals(vec![Principal::new(
            "admin",
            token,
            Role::Admin,
            None::<Arc<str>>,
        )?])
    }

    pub fn is_required(&self) -> bool {
        !self.principals.is_empty()
    }

    pub fn authenticate(&self, token: &str) -> Option<Principal> {
        self.principals
            .iter()
            .find(|p| constant_time_eq(token.as_bytes(), p.token.as_bytes()))
            .cloned()
    }

    pub fn parse_entry(entry: &str) -> anyhow::Result<Principal> {
        let parts: Vec<_> = entry.split(':').collect();
        if !(3..=4).contains(&parts.len()) {
            anyhow::bail!("auth key entry must be principal:token:role[:tenant], got {entry:?}");
        }
        let tenant = parts.get(3).and_then(|v| {
            let trimmed = v.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        let mut principal = Principal::new(
            parts[0].trim().to_string(),
            parts[1].trim().to_string(),
            parts[2].parse()?,
            tenant,
        )?;
        // Refine capabilities from the role *name* (e.g. evaluator/observer),
        // which carries more than the coarse read/write/admin level.
        principal.caps = Capabilities::for_role_str(parts[2].trim());
        Ok(principal)
    }

    pub fn parse_entries(
        entries: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Self> {
        let principals = entries
            .into_iter()
            .map(|entry| Self::parse_entry(entry.as_ref()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Self::from_principals(principals)
    }

    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Ok(raw) = std::env::var("DUXX_AUTH_KEYS") else {
            return Ok(None);
        };
        let entries = raw
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty());
        Ok(Some(Self::parse_entries(entries)?))
    }
}

#[derive(Debug, Clone)]
pub struct AuthState {
    principal: Option<Principal>,
}

impl AuthState {
    pub fn anonymous() -> Self {
        Self { principal: None }
    }

    pub fn unauthenticated() -> Self {
        Self { principal: None }
    }

    pub fn disabled() -> Self {
        Self {
            principal: Some(Principal {
                name: Arc::from("anonymous"),
                token: Arc::from(""),
                role: Role::Admin,
                tenant: None,
                caps: Capabilities::all(),
            }),
        }
    }

    pub fn is_authed(&self) -> bool {
        self.principal.is_some()
    }

    pub fn set_principal(&mut self, principal: Principal) {
        self.principal = Some(principal);
    }

    pub fn role(&self) -> Option<Role> {
        self.principal.as_ref().map(|p| p.role)
    }

    pub fn principal_name(&self) -> &str {
        self.principal
            .as_ref()
            .map(|p| p.name.as_ref())
            .unwrap_or("unauthenticated")
    }

    pub fn tenant(&self) -> Option<&str> {
        self.principal.as_ref().and_then(|p| p.tenant.as_deref())
    }

    pub fn allow_command(&self, command: &str) -> bool {
        let Some(p) = self.principal.as_ref() else {
            return false;
        };
        // Coarse level gate AND fine-grained capability gate.
        p.role.can(required_role(command)) && p.caps.allows(command)
    }
}

pub fn required_role(command: &str) -> Role {
    match command {
        "PING"
        | "HELLO"
        | "COMMAND"
        | "INFO"
        | "GET"
        | "RECALL"
        | "RECALL.FILTER"
        | "MEMORY.SCAN"
        | "SUBSCRIBE"
        | "UNSUBSCRIBE"
        | "PSUBSCRIBE"
        | "PUNSUBSCRIBE"
        | "TRACE.GET"
        | "TRACE.SUBTREE"
        | "TRACE.THREAD"
        | "TRACE.SEARCH"
        | "PROMPT.GET"
        | "PROMPT.LIST"
        | "PROMPT.NAMES"
        | "PROMPT.SEARCH"
        | "PROMPT.DIFF"
        | "DATASET.GET"
        | "DATASET.LIST"
        | "DATASET.NAMES"
        | "DATASET.SAMPLE"
        | "DATASET.SIZE"
        | "DATASET.SPLITS"
        | "DATASET.SEARCH"
        | "EVAL.GET"
        | "EVAL.SCORES"
        | "EVAL.LIST"
        | "EVAL.COMPARE"
        | "EVAL.CLUSTER_FAILURES"
        | "REPLAY.GET_SESSION"
        | "REPLAY.GET_RUN"
        | "REPLAY.LIST_SESSIONS"
        | "REPLAY.LIST_RUNS"
        | "REPLAY.DIFF"
        | "COST.QUERY"
        | "COST.AGGREGATE"
        | "COST.TOTAL"
        | "COST.GET_BUDGET"
        | "COST.STATUS"
        | "COST.ALERTS"
        | "COST.CLUSTER_EXPENSIVE"
        | "DOC.SEARCH"
        | "DOC.LIST" => Role::ReadOnly,
        "SET"
        | "DEL"
        | "REMEMBER"
        | "REMEMBER.IDEM"
        | "REMEMBER.BATCH"
        | "REMEMBER.META"
        | "COMPACT"
        | "FORGET"
        | "FORGET.KEY"
        | "FORGET.OLDER"
        | "TRACE.RECORD"
        | "TRACE.CLOSE"
        | "PROMPT.PUT"
        | "PROMPT.TAG"
        | "PROMPT.UNTAG"
        | "DATASET.CREATE"
        | "DATASET.ADD"
        | "DATASET.TAG"
        | "DATASET.UNTAG"
        | "DATASET.FROM_RECALL"
        | "EVAL.START"
        | "EVAL.SCORE"
        | "EVAL.COMPLETE"
        | "EVAL.FAIL"
        | "REPLAY.CAPTURE"
        | "REPLAY.START"
        | "REPLAY.STEP"
        | "REPLAY.RECORD"
        | "REPLAY.COMPLETE"
        | "REPLAY.FAIL"
        | "REPLAY.SET_TRACE"
        | "COST.RECORD"
        | "DOC.INGEST"
        | "DOC.DELETE" => Role::ReadWrite,
        _ => Role::Admin,
    }
}

pub fn is_audited_command(command: &str) -> bool {
    command == "AUTH" || required_role(command) != Role::ReadOnly
}

#[derive(Clone)]
pub struct AuditLogger {
    file: Arc<Mutex<File>>,
}

impl AuditLogger {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn record(&self, event: AuditEvent<'_>) {
        let payload = json!({
            "ts_unix_ns": unix_ns(),
            "surface": event.surface,
            "principal": event.principal,
            "tenant": event.tenant,
            "role": event.role.map(Role::as_str),
            "command": event.command,
            "outcome": event.outcome,
            "detail": event.detail,
        });
        match self.file.lock() {
            Ok(mut f) => {
                let _ = writeln!(f, "{payload}");
            }
            Err(e) => {
                tracing::warn!(error = %e, "audit log lock poisoned");
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AuditEvent<'a> {
    pub surface: &'a str,
    pub principal: &'a str,
    pub tenant: Option<&'a str>,
    pub role: Option<Role>,
    pub command: &'a str,
    pub outcome: &'a str,
    pub detail: Option<&'a str>,
}

/// One queryable audit record (in addition to the JSON-lines file log).
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditRecord {
    pub ts_unix_ns: u128,
    pub principal: String,
    pub tenant: Option<String>,
    pub command: String,
    pub outcome: String,
}

/// A bounded, in-memory, per-tenant-queryable audit trail. Always on (cheap),
/// so the Studio audit view has recent activity even without `--audit-log`.
/// The bound is the retention window; oldest records are dropped past it.
#[derive(Clone)]
pub struct AuditTrail {
    inner: Arc<Mutex<std::collections::VecDeque<AuditRecord>>>,
    cap: usize,
}

impl AuditTrail {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(std::collections::VecDeque::with_capacity(
                cap.min(1024),
            ))),
            cap: cap.max(1),
        }
    }

    pub fn record(&self, rec: AuditRecord) {
        if let Ok(mut q) = self.inner.lock() {
            if q.len() >= self.cap {
                q.pop_front();
            }
            q.push_back(rec);
        }
    }

    /// Most-recent-first records for `tenant`, capped at `limit`.
    pub fn for_tenant(&self, tenant: &str, limit: usize) -> Vec<AuditRecord> {
        let q = match self.inner.lock() {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        q.iter()
            .rev()
            .filter(|r| r.tenant.as_deref() == Some(tenant))
            .take(limit)
            .cloned()
            .collect()
    }
}

pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}
