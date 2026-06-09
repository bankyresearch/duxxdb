//! # duxx-control — DuxxDB managed-cloud control plane (Phase B kickoff)
//!
//! The control plane is the **system of record** for the managed cloud:
//! organizations, projects, environments, API keys, where each project is
//! placed, and how much it has used. It is deliberately **lightweight** — it
//! holds metadata, not agent data — so it does not depend on the data-plane
//! engine crates (no HNSW/tantivy/redb here).
//!
//! ## How it drives the data plane
//!
//! The control plane and the data plane (`duxx-server`) are connected by a
//! **credential string contract**, not a shared library. The control plane
//! issues an API key and can materialize it as a data-plane auth entry:
//!
//! ```text
//!   <key_id>:<secret>:<role>:<org_id>/<project_id>/<env>
//! ```
//!
//! which `duxx-server` already understands: `Namespace::parse` turns the
//! `org/project/env` tenant field into an isolated, durable workspace, and the
//! A3 role aliases (`owner|developer|evaluator|service|observer`) map onto its
//! enforcement. So today the control plane provisions a node by handing it the
//! set of `--auth-key` entries for the projects placed on it
//! ([`ControlPlane::data_plane_auth_entries`]).
//!
//! ## Scope of this kickoff
//!
//! Implemented: org/project CRUD, API-key issue / rotate / revoke /
//! authenticate, project placement (shared vs dedicated node), usage metering
//! with a pluggable [`BillingSink`], and data-plane catalog materialization.
//!
//! Deliberately deferred (each is a real follow-up, not faked here):
//! * **Signed short-lived tokens (JWT).** The forward path is to issue
//!   `jsonwebtoken`-signed JWTs carrying the claims above and have the data
//!   plane validate the signature, instead of materializing static catalogs.
//!   Not hand-rolled here — that would mean shipping unreviewed crypto.
//! * **Durable persistence** (Postgres) — state is in-memory; the API is
//!   storage-agnostic so a backing store slots in behind it.
//! * **SSO (OIDC/SAML)**, invitations, and a real Stripe integration behind
//!   [`BillingSink`].

pub mod api;

/// Re-export so operators can generate the signing keypair via one crate:
/// private DER → control plane, public DER → data-plane nodes.
pub use duxx_token::generate_ed25519;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Value types (mirror duxx-tenant's, kept local so the control plane stays
// dependency-light; the wire contract is the string format, not the types).
// ---------------------------------------------------------------------------

/// Deployment environment within a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A role granted to an API key. The string form is what the data plane's
/// `--auth-key` parser accepts (see the A3 role aliases in
/// `duxx-server::security`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    Owner,
    Admin,
    Developer,
    Evaluator,
    Observer,
    /// The credential an agent runtime uses in production.
    ServiceAccount,
}

impl Role {
    /// The token the data-plane `--auth-key` role field expects.
    pub fn as_dataplane_str(self) -> &'static str {
        match self {
            Role::Owner => "owner",
            Role::Admin => "admin",
            Role::Developer => "developer",
            Role::Evaluator => "evaluator",
            Role::Observer => "observer",
            Role::ServiceAccount => "service",
        }
    }
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Org {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub id: String,
    pub org_id: String,
    pub name: String,
}

/// An issued API key. `secret` is retained so the control plane can
/// materialize data-plane auth catalogs (see the module docs); a production
/// build would instead store only a hash and issue signed short-lived tokens.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub secret: String,
    pub org_id: String,
    pub project_id: String,
    pub env: Env,
    pub role: Role,
    pub name: String,
    pub revoked: bool,
}

impl ApiKey {
    /// The `principal:token:role:org/project/env` line that `duxx-server`
    /// consumes via `--auth-key`. Returns `None` for a revoked key.
    pub fn data_plane_entry(&self) -> Option<String> {
        if self.revoked {
            return None;
        }
        Some(format!(
            "{}:{}:{}:{}/{}/{}",
            self.id,
            self.secret,
            self.role.as_dataplane_str(),
            self.org_id,
            self.project_id,
            self.env.as_str(),
        ))
    }
}

/// Lifecycle of an org membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemberStatus {
    Invited,
    Active,
}

/// A person's membership in an org, with their role.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Member {
    pub id: String,
    pub org_id: String,
    pub email: String,
    pub role: Role,
    pub status: MemberStatus,
}

/// Where a project's workspaces run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlacementMode {
    /// Many projects share one node (namespace-per-project). Phase B1.
    Shared,
    /// One node dedicated to this project. Phase B2.
    Dedicated,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Placement {
    pub project_id: String,
    /// `host:port` of the data-plane node.
    pub node: String,
    pub mode: PlacementMode,
}

/// Accumulated usage for a project, the basis for metered billing.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub requests: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd: f64,
}

/// What `authenticate` returns: enough to scope a request on the data plane.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPrincipal {
    pub key_id: String,
    pub org_id: String,
    pub project_id: String,
    pub env: Env,
    pub role: Role,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ControlError {
    #[error("organization {0:?} not found")]
    OrgNotFound(String),
    #[error("project {0:?} not found")]
    ProjectNotFound(String),
    #[error("api key {0:?} not found")]
    KeyNotFound(String),
    #[error("name must not be empty")]
    EmptyName,
    #[error("secret does not match any active key")]
    BadSecret,
    #[error("control plane has no JWT signing key configured")]
    NoSigningKey,
    #[error("token error: {0}")]
    Token(String),
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

/// Where metered usage is reported (Stripe, an invoice ledger, etc.). The real
/// Stripe integration slots in behind this trait.
pub trait BillingSink {
    fn report(&self, project_id: &str, usage: &Usage) -> Result<(), String>;
}

/// Captures every report in memory — handy for tests and dry runs.
#[derive(Default)]
pub struct RecordingBillingSink {
    pub reports: RwLock<Vec<(String, Usage)>>,
}

impl BillingSink for RecordingBillingSink {
    fn report(&self, project_id: &str, usage: &Usage) -> Result<(), String> {
        self.reports
            .write()
            .push((project_id.to_string(), usage.clone()));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ControlPlane
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Store — the persistence seam
// ---------------------------------------------------------------------------

/// Persistence backend for the control plane. The default is [`InMemoryStore`];
/// a production deployment implements this over Postgres without touching
/// `ControlPlane`'s business logic. The compound mutations (`insert_key`,
/// `revoke_key`, `rotate_key`, `insert_member`, `accept_invite`) are single
/// methods precisely so a SQL backend can run each inside one transaction.
pub trait Store: Send + Sync {
    fn insert_org(&self, org: Org);
    fn get_org(&self, id: &str) -> Option<Org>;
    fn list_orgs(&self) -> Vec<Org>;

    fn insert_project(&self, project: Project);
    fn get_project(&self, id: &str) -> Option<Project>;
    fn projects_in_org(&self, org_id: &str) -> Vec<Project>;

    /// Insert a key and index its secret for authentication.
    fn insert_key(&self, key: ApiKey);
    fn get_key(&self, id: &str) -> Option<ApiKey>;
    fn keys_for_project(&self, project_id: &str) -> Vec<ApiKey>;
    fn all_keys(&self) -> Vec<ApiKey>;
    fn key_id_by_secret(&self, secret: &str) -> Option<String>;
    /// `None` = not found · `Some(false)` = already revoked · `Some(true)` = revoked now.
    fn revoke_key(&self, key_id: &str) -> Option<bool>;
    /// Swap a key's secret (and its secret index). `None` = not found.
    fn rotate_key(&self, key_id: &str, new_secret: &str) -> Option<ApiKey>;

    fn insert_member(&self, member: Member, invite_token: Option<String>);
    fn accept_invite(&self, token: &str) -> Option<Member>;
    fn list_members(&self, org_id: &str) -> Vec<Member>;
    fn remove_member(&self, member_id: &str) -> bool;

    fn insert_placement(&self, placement: Placement);
    fn get_placement(&self, project_id: &str) -> Option<Placement>;
    fn all_placements(&self) -> Vec<Placement>;

    fn add_usage(&self, project_id: &str, tokens_in: u64, tokens_out: u64, cost_usd: f64);
    fn get_usage(&self, project_id: &str) -> Usage;
    fn all_usage(&self) -> Vec<(String, Usage)>;
}

#[derive(Default)]
struct State {
    orgs: HashMap<String, Org>,
    projects: HashMap<String, Project>,
    keys: HashMap<String, ApiKey>,
    /// secret → key_id, for O(1) authentication.
    by_secret: HashMap<String, String>,
    placements: HashMap<String, Placement>,
    usage: HashMap<String, Usage>,
    /// member_id → Member.
    members: HashMap<String, Member>,
    /// invite token → member_id.
    invites: HashMap<String, String>,
}

/// In-memory [`Store`] — the default backend (thread-safe behind one lock).
#[derive(Default)]
pub struct InMemoryStore {
    state: RwLock<State>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Store for InMemoryStore {
    fn insert_org(&self, org: Org) {
        self.state.write().orgs.insert(org.id.clone(), org);
    }
    fn get_org(&self, id: &str) -> Option<Org> {
        self.state.read().orgs.get(id).cloned()
    }
    fn list_orgs(&self) -> Vec<Org> {
        let mut v: Vec<Org> = self.state.read().orgs.values().cloned().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    fn insert_project(&self, project: Project) {
        self.state.write().projects.insert(project.id.clone(), project);
    }
    fn get_project(&self, id: &str) -> Option<Project> {
        self.state.read().projects.get(id).cloned()
    }
    fn projects_in_org(&self, org_id: &str) -> Vec<Project> {
        let mut v: Vec<Project> = self
            .state
            .read()
            .projects
            .values()
            .filter(|p| p.org_id == org_id)
            .cloned()
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    fn insert_key(&self, key: ApiKey) {
        let mut st = self.state.write();
        st.by_secret.insert(key.secret.clone(), key.id.clone());
        st.keys.insert(key.id.clone(), key);
    }
    fn get_key(&self, id: &str) -> Option<ApiKey> {
        self.state.read().keys.get(id).cloned()
    }
    fn keys_for_project(&self, project_id: &str) -> Vec<ApiKey> {
        let mut v: Vec<ApiKey> = self
            .state
            .read()
            .keys
            .values()
            .filter(|k| k.project_id == project_id)
            .cloned()
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }
    fn all_keys(&self) -> Vec<ApiKey> {
        self.state.read().keys.values().cloned().collect()
    }
    fn key_id_by_secret(&self, secret: &str) -> Option<String> {
        self.state.read().by_secret.get(secret).cloned()
    }
    fn revoke_key(&self, key_id: &str) -> Option<bool> {
        let mut st = self.state.write();
        let key = st.keys.get_mut(key_id)?;
        if key.revoked {
            return Some(false);
        }
        key.revoked = true;
        let secret = key.secret.clone();
        st.by_secret.remove(&secret);
        Some(true)
    }
    fn rotate_key(&self, key_id: &str, new_secret: &str) -> Option<ApiKey> {
        let mut st = self.state.write();
        let key = st.keys.get_mut(key_id)?;
        let old = key.secret.clone();
        key.secret = new_secret.to_string();
        key.revoked = false;
        let updated = key.clone();
        st.by_secret.remove(&old);
        st.by_secret.insert(new_secret.to_string(), key_id.to_string());
        Some(updated)
    }

    fn insert_member(&self, member: Member, invite_token: Option<String>) {
        let mut st = self.state.write();
        if let Some(t) = invite_token {
            st.invites.insert(t, member.id.clone());
        }
        st.members.insert(member.id.clone(), member);
    }
    fn accept_invite(&self, token: &str) -> Option<Member> {
        let mut st = self.state.write();
        let member_id = st.invites.remove(token)?;
        let m = st.members.get_mut(&member_id)?;
        m.status = MemberStatus::Active;
        Some(m.clone())
    }
    fn list_members(&self, org_id: &str) -> Vec<Member> {
        let mut v: Vec<Member> = self
            .state
            .read()
            .members
            .values()
            .filter(|m| m.org_id == org_id)
            .cloned()
            .collect();
        v.sort_by(|a, b| a.email.cmp(&b.email));
        v
    }
    fn remove_member(&self, member_id: &str) -> bool {
        let mut st = self.state.write();
        st.invites.retain(|_, mid| mid != member_id);
        st.members.remove(member_id).is_some()
    }

    fn insert_placement(&self, placement: Placement) {
        self.state
            .write()
            .placements
            .insert(placement.project_id.clone(), placement);
    }
    fn get_placement(&self, project_id: &str) -> Option<Placement> {
        self.state.read().placements.get(project_id).cloned()
    }
    fn all_placements(&self) -> Vec<Placement> {
        self.state.read().placements.values().cloned().collect()
    }

    fn add_usage(&self, project_id: &str, tokens_in: u64, tokens_out: u64, cost_usd: f64) {
        let mut st = self.state.write();
        let u = st.usage.entry(project_id.to_string()).or_default();
        u.requests += 1;
        u.tokens_in += tokens_in;
        u.tokens_out += tokens_out;
        u.cost_usd += cost_usd;
    }
    fn get_usage(&self, project_id: &str) -> Usage {
        self.state.read().usage.get(project_id).cloned().unwrap_or_default()
    }
    fn all_usage(&self) -> Vec<(String, Usage)> {
        self.state
            .read()
            .usage
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ControlPlane — business logic over a pluggable Store
// ---------------------------------------------------------------------------

/// The control plane. Persistence is delegated to a [`Store`] (in-memory by
/// default; Postgres-ready via [`ControlPlane::with_store`]).
pub struct ControlPlane {
    store: Box<dyn Store>,
    /// HS256 secret for signing workspace JWTs, shared with data-plane nodes.
    /// `None` → no HS256 minting.
    signing_key: Option<Vec<u8>>,
    /// Ed25519 private key (PKCS#8 DER) for **asymmetric** signing. Preferred
    /// over HS256 when set: nodes verify with only the public key, so a
    /// compromised node cannot mint tokens. `None` → no EdDSA minting.
    ed_private: Option<Vec<u8>>,
}

impl Default for ControlPlane {
    fn default() -> Self {
        Self {
            store: Box::new(InMemoryStore::default()),
            signing_key: None,
            ed_private: None,
        }
    }
}

impl ControlPlane {
    pub fn new() -> Self {
        Self::default()
    }

    /// Control plane backed by a custom [`Store`] (e.g. Postgres). Compose with
    /// signing by setting the key fields afterward, or extend with a builder.
    pub fn with_store(store: Box<dyn Store>) -> Self {
        Self {
            store,
            signing_key: None,
            ed_private: None,
        }
    }

    /// Control plane that mints **HS256** workspace JWTs ([`mint_jwt`]) using
    /// `signing_key`, the secret shared with data-plane nodes
    /// (`duxx-server --jwt-secret`).
    ///
    /// [`mint_jwt`]: ControlPlane::mint_jwt
    pub fn with_signing_key(signing_key: impl Into<Vec<u8>>) -> Self {
        Self {
            signing_key: Some(signing_key.into()),
            ..Self::default()
        }
    }

    /// Control plane that mints **EdDSA (Ed25519)** workspace JWTs. Pass the
    /// private PKCS#8 DER from [`duxx_token::generate_ed25519`]; give the
    /// matching public key to each node (`duxx-server --jwt-public-key`).
    pub fn with_ed25519(private_pkcs8_der: impl Into<Vec<u8>>) -> Self {
        Self {
            ed_private: Some(private_pkcs8_der.into()),
            ..Self::default()
        }
    }

    // -- Orgs & projects ----------------------------------------------------

    pub fn create_org(&self, name: &str) -> Result<Org, ControlError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ControlError::EmptyName);
        }
        let org = Org {
            id: format!("org_{}", Uuid::new_v4().simple()),
            name: name.to_string(),
        };
        self.store.insert_org(org.clone());
        Ok(org)
    }

    pub fn create_project(&self, org_id: &str, name: &str) -> Result<Project, ControlError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ControlError::EmptyName);
        }
        if self.store.get_org(org_id).is_none() {
            return Err(ControlError::OrgNotFound(org_id.to_string()));
        }
        let project = Project {
            id: format!("proj_{}", Uuid::new_v4().simple()),
            org_id: org_id.to_string(),
            name: name.to_string(),
        };
        self.store.insert_project(project.clone());
        Ok(project)
    }

    pub fn org(&self, org_id: &str) -> Option<Org> {
        self.store.get_org(org_id)
    }

    /// Every organization, name-sorted (stable order for a UI listing).
    pub fn list_orgs(&self) -> Vec<Org> {
        self.store.list_orgs()
    }

    pub fn project(&self, project_id: &str) -> Option<Project> {
        self.store.get_project(project_id)
    }

    pub fn projects_in_org(&self, org_id: &str) -> Vec<Project> {
        self.store.projects_in_org(org_id)
    }

    /// Keys issued for a project. Secrets are intentionally retained on the
    /// `ApiKey` for catalog materialization; the HTTP API redacts them when
    /// listing (a secret is shown only once, at issue time).
    pub fn keys_for_project(&self, project_id: &str) -> Vec<ApiKey> {
        self.store.keys_for_project(project_id)
    }

    // -- API keys -----------------------------------------------------------

    /// Issue a key for `(project, env)` with `role`. The returned `ApiKey`
    /// includes the one-time secret.
    pub fn issue_key(
        &self,
        project_id: &str,
        env: Env,
        role: Role,
        name: &str,
    ) -> Result<ApiKey, ControlError> {
        let project = self
            .store
            .get_project(project_id)
            .ok_or_else(|| ControlError::ProjectNotFound(project_id.to_string()))?;
        let key = ApiKey {
            id: format!("key_{}", Uuid::new_v4().simple()),
            secret: format!("sk_{}", Uuid::new_v4().simple()),
            org_id: project.org_id,
            project_id: project_id.to_string(),
            env,
            role,
            name: name.trim().to_string(),
            revoked: false,
        };
        self.store.insert_key(key.clone());
        Ok(key)
    }

    /// Revoke a key. Returns `true` if it existed and was active.
    pub fn revoke_key(&self, key_id: &str) -> Result<bool, ControlError> {
        self.store
            .revoke_key(key_id)
            .ok_or_else(|| ControlError::KeyNotFound(key_id.to_string()))
    }

    /// Rotate a key's secret in place. The old secret stops authenticating
    /// immediately. Returns the new `ApiKey`.
    pub fn rotate_key(&self, key_id: &str) -> Result<ApiKey, ControlError> {
        let new_secret = format!("sk_{}", Uuid::new_v4().simple());
        self.store
            .rotate_key(key_id, &new_secret)
            .ok_or_else(|| ControlError::KeyNotFound(key_id.to_string()))
    }

    /// Authenticate a presented secret → the principal it resolves to, or
    /// `None` if unknown or revoked. This is the function a JWT-validation or
    /// API-gateway layer would build on.
    pub fn authenticate(&self, secret: &str) -> Option<ResolvedPrincipal> {
        let key_id = self.store.key_id_by_secret(secret)?;
        let key = self.store.get_key(&key_id)?;
        if key.revoked {
            return None;
        }
        Some(ResolvedPrincipal {
            key_id: key.id.clone(),
            org_id: key.org_id.clone(),
            project_id: key.project_id.clone(),
            env: key.env,
            role: key.role,
        })
    }

    /// Exchange a long-lived API-key secret for a **short-lived signed JWT**
    /// scoped to the key's workspace. This is the credential a client presents
    /// to a data-plane node, which verifies the signature and resolves the
    /// namespace from the claims — no static catalog needed.
    ///
    /// `now_unix` is the current time in seconds (passed in, not read here);
    /// `ttl_secs` is how long the token stays valid. Requires a signing key
    /// (see [`ControlPlane::with_signing_key`]).
    pub fn mint_jwt(
        &self,
        api_key_secret: &str,
        now_unix: u64,
        ttl_secs: u64,
    ) -> Result<String, ControlError> {
        // Must have *some* signing key before doing anything else.
        if self.ed_private.is_none() && self.signing_key.is_none() {
            return Err(ControlError::NoSigningKey);
        }
        let p = self
            .authenticate(api_key_secret)
            .ok_or(ControlError::BadSecret)?;
        let claims = duxx_token::Claims::new(
            p.key_id,
            p.org_id,
            p.project_id,
            p.env.as_str(),
            p.role.as_dataplane_str(),
            now_unix,
            ttl_secs,
        );
        // Prefer asymmetric (EdDSA) when configured; else HS256.
        if let Some(ed) = &self.ed_private {
            return duxx_token::sign_ed25519(&claims, ed).map_err(|e| ControlError::Token(e.to_string()));
        }
        let signing_key = self.signing_key.as_ref().ok_or(ControlError::NoSigningKey)?;
        duxx_token::sign(&claims, signing_key).map_err(|e| ControlError::Token(e.to_string()))
    }

    // -- Members & invitations ---------------------------------------------

    /// Invite someone to an org with a role. Returns `(Member, invite_token)`;
    /// the token is what they redeem via [`accept_invite`]. The member starts
    /// `Invited` and becomes `Active` on acceptance.
    ///
    /// [`accept_invite`]: ControlPlane::accept_invite
    pub fn invite_member(
        &self,
        org_id: &str,
        email: &str,
        role: Role,
    ) -> Result<(Member, String), ControlError> {
        let email = email.trim();
        if email.is_empty() {
            return Err(ControlError::EmptyName);
        }
        if self.store.get_org(org_id).is_none() {
            return Err(ControlError::OrgNotFound(org_id.to_string()));
        }
        let member = Member {
            id: format!("mem_{}", Uuid::new_v4().simple()),
            org_id: org_id.to_string(),
            email: email.to_string(),
            role,
            status: MemberStatus::Invited,
        };
        let token = format!("inv_{}", Uuid::new_v4().simple());
        self.store.insert_member(member.clone(), Some(token.clone()));
        Ok((member, token))
    }

    /// Redeem an invite token → the now-`Active` member.
    pub fn accept_invite(&self, invite_token: &str) -> Result<Member, ControlError> {
        self.store
            .accept_invite(invite_token)
            .ok_or_else(|| ControlError::KeyNotFound(invite_token.to_string()))
    }

    /// Every member of an org, email-sorted.
    pub fn list_members(&self, org_id: &str) -> Vec<Member> {
        self.store.list_members(org_id)
    }

    /// Remove a member. Returns `true` if they existed.
    pub fn remove_member(&self, member_id: &str) -> bool {
        self.store.remove_member(member_id)
    }

    // -- Provisioning -------------------------------------------------------

    /// Record where a project is placed (which data-plane node, shared or
    /// dedicated). Overwrites any prior placement.
    pub fn place_project(
        &self,
        project_id: &str,
        node: &str,
        mode: PlacementMode,
    ) -> Result<Placement, ControlError> {
        if self.store.get_project(project_id).is_none() {
            return Err(ControlError::ProjectNotFound(project_id.to_string()));
        }
        let placement = Placement {
            project_id: project_id.to_string(),
            node: node.to_string(),
            mode,
        };
        self.store.insert_placement(placement.clone());
        Ok(placement)
    }

    pub fn placement(&self, project_id: &str) -> Option<Placement> {
        self.store.get_placement(project_id)
    }

    /// The data-plane `--auth-key` entries a given node must serve: every
    /// active key whose project is placed on `node`. This is what the
    /// provisioner pushes to a data-plane node.
    pub fn data_plane_auth_entries(&self, node: &str) -> Vec<String> {
        let projects_on_node: std::collections::HashSet<String> = self
            .store
            .all_placements()
            .into_iter()
            .filter(|p| p.node == node)
            .map(|p| p.project_id)
            .collect();
        self.store
            .all_keys()
            .into_iter()
            .filter(|k| projects_on_node.contains(&k.project_id))
            .filter_map(|k| k.data_plane_entry())
            .collect()
    }

    // -- Usage metering -----------------------------------------------------

    /// Record usage for a project (typically aggregated from the data plane's
    /// `CostLedger` per billing period).
    pub fn record_usage(&self, project_id: &str, tokens_in: u64, tokens_out: u64, cost_usd: f64) {
        self.store.add_usage(project_id, tokens_in, tokens_out, cost_usd);
    }

    pub fn usage(&self, project_id: &str) -> Usage {
        self.store.get_usage(project_id)
    }

    /// Push every project's accumulated usage to a billing sink (Stripe, an
    /// invoice ledger, …). Returns the number of projects reported.
    pub fn flush_billing(&self, sink: &dyn BillingSink) -> Result<usize, String> {
        let snapshot = self.store.all_usage();
        let n = snapshot.len();
        for (project_id, usage) in snapshot {
            sink.report(&project_id, &usage)?;
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn org_project_key_lifecycle() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme Inc").unwrap();
        let proj = cp.create_project(&org.id, "support-bot").unwrap();
        assert_eq!(cp.projects_in_org(&org.id).len(), 1);

        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Developer, "ci")
            .unwrap();

        // The secret authenticates to the right namespace + role.
        let p = cp.authenticate(&key.secret).expect("key should authenticate");
        assert_eq!(p.org_id, org.id);
        assert_eq!(p.project_id, proj.id);
        assert_eq!(p.env, Env::Prod);
        assert_eq!(p.role, Role::Developer);
    }

    #[test]
    fn control_plane_runs_over_a_pluggable_store() {
        // Constructing over an explicit Store proves the persistence seam — a
        // Postgres-backed `Store` would slot in exactly here.
        let cp = ControlPlane::with_store(Box::new(InMemoryStore::new()));
        let org = cp.create_org("Acme").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Developer, "k")
            .unwrap();
        // Full round-trip through the trait: list, authenticate, revoke.
        assert_eq!(cp.list_orgs().len(), 1);
        assert_eq!(cp.keys_for_project(&proj.id).len(), 1);
        assert!(cp.authenticate(&key.secret).is_some());
        assert!(cp.revoke_key(&key.id).unwrap());
        assert!(cp.authenticate(&key.secret).is_none());
    }

    #[test]
    fn create_project_under_missing_org_errors() {
        let cp = ControlPlane::new();
        let err = cp.create_project("org_nope", "x").unwrap_err();
        assert_eq!(err, ControlError::OrgNotFound("org_nope".into()));
    }

    #[test]
    fn revoke_and_rotate_invalidate_old_secret() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::ServiceAccount, "agent")
            .unwrap();

        // Rotate → old secret dead, new secret works.
        let rotated = cp.rotate_key(&key.id).unwrap();
        assert_ne!(rotated.secret, key.secret);
        assert!(cp.authenticate(&key.secret).is_none(), "old secret must die");
        assert!(cp.authenticate(&rotated.secret).is_some());

        // Revoke → new secret dead too.
        assert!(cp.revoke_key(&key.id).unwrap());
        assert!(cp.authenticate(&rotated.secret).is_none());
        assert!(!cp.revoke_key(&key.id).unwrap(), "second revoke is a no-op");
    }

    /// The integration contract with the data plane: a materialized entry is
    /// exactly `principal:token:role:org/project/env` (4 colon-fields, the
    /// tenant carrying the org/project/env that `Namespace::parse` expects).
    #[test]
    fn data_plane_entry_matches_auth_key_contract() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Observer, "dash")
            .unwrap();

        let entry = key.data_plane_entry().unwrap();
        let fields: Vec<&str> = entry.split(':').collect();
        assert_eq!(fields.len(), 4, "must be principal:token:role:tenant: {entry}");
        assert_eq!(fields[0], key.id);
        assert_eq!(fields[1], key.secret);
        assert_eq!(fields[2], "observer");
        assert_eq!(fields[3], format!("{}/{}/prod", org.id, proj.id));

        // Revoked keys produce no entry.
        cp.revoke_key(&key.id).unwrap();
        let reloaded = cp
            .keys_for_project(&proj.id)
            .into_iter()
            .find(|k| k.id == key.id)
            .unwrap();
        assert!(reloaded.data_plane_entry().is_none());
    }

    #[test]
    fn placement_scopes_node_auth_entries() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme").unwrap();
        let a = cp.create_project(&org.id, "a").unwrap();
        let b = cp.create_project(&org.id, "b").unwrap();
        let ka = cp.issue_key(&a.id, Env::Prod, Role::Developer, "ka").unwrap();
        let _kb = cp.issue_key(&b.id, Env::Prod, Role::Developer, "kb").unwrap();

        cp.place_project(&a.id, "node-1:6380", PlacementMode::Shared)
            .unwrap();
        cp.place_project(&b.id, "node-2:6380", PlacementMode::Dedicated)
            .unwrap();

        let node1 = cp.data_plane_auth_entries("node-1:6380");
        assert_eq!(node1.len(), 1, "only project a is on node-1");
        assert!(node1[0].contains(&ka.secret));
        assert!(node1[0].ends_with(&format!("{}/{}/prod", org.id, a.id)));

        assert_eq!(cp.data_plane_auth_entries("node-2:6380").len(), 1);
        assert!(cp.data_plane_auth_entries("node-3:6380").is_empty());
    }

    #[test]
    fn member_invite_accept_remove_lifecycle() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme").unwrap();

        let (m, token) = cp
            .invite_member(&org.id, "dev@acme.com", Role::Developer)
            .unwrap();
        assert_eq!(m.status, MemberStatus::Invited);
        assert_eq!(cp.list_members(&org.id).len(), 1);

        let active = cp.accept_invite(&token).unwrap();
        assert_eq!(active.id, m.id);
        assert_eq!(active.status, MemberStatus::Active);
        assert!(cp.accept_invite(&token).is_err(), "invite token is single-use");

        assert!(cp.remove_member(&m.id));
        assert!(cp.list_members(&org.id).is_empty());

        assert_eq!(
            cp.invite_member("org_nope", "x@y.com", Role::Observer)
                .unwrap_err(),
            ControlError::OrgNotFound("org_nope".into())
        );
    }

    #[test]
    fn mint_jwt_signs_claims_a_node_can_verify() {
        let secret = b"shared-hs256-secret-between-cp-and-nodes".to_vec();
        let cp = ControlPlane::with_signing_key(secret.clone());
        let org = cp.create_org("Acme").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Developer, "agent")
            .unwrap();

        // Exchange the API key secret for a short-lived signed JWT. `exp` is
        // validated against the real clock, so mint with real `now`.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let jwt = cp.mint_jwt(&key.secret, now, 900).unwrap();

        // A node holding the same shared secret verifies it and reads the
        // workspace scope straight from the claims.
        let claims = duxx_token::verify(&jwt, &secret).unwrap();
        assert_eq!(claims.sub, key.id);
        assert_eq!(claims.role, "developer");
        assert_eq!(claims.tenant(), format!("{}/{}/prod", org.id, proj.id));

        // A bad secret can't mint; a control plane with no signing key can't either.
        assert_eq!(
            cp.mint_jwt("sk_nope", 1_700_000_000, 900).unwrap_err(),
            ControlError::BadSecret
        );
        let cp2 = ControlPlane::new();
        assert_eq!(
            cp2.mint_jwt(&key.secret, 1_700_000_000, 900).unwrap_err(),
            ControlError::NoSigningKey
        );
    }

    #[test]
    fn mint_jwt_ed25519_verifies_with_public_key_only() {
        let (priv_der, pub_der) = duxx_token::generate_ed25519().unwrap();
        let cp = ControlPlane::with_ed25519(priv_der);
        let org = cp.create_org("A").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();
        let key = cp
            .issue_key(&proj.id, Env::Prod, Role::Developer, "k")
            .unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let jwt = cp.mint_jwt(&key.secret, now, 300).unwrap();

        // The public key verifies it; an HS256 verify cannot.
        let claims = duxx_token::verify_ed25519(&jwt, &pub_der).unwrap();
        assert_eq!(claims.role, "developer");
        assert!(duxx_token::verify(&jwt, &pub_der).is_err());
    }

    #[test]
    fn usage_meters_and_flushes_to_billing() {
        let cp = ControlPlane::new();
        let org = cp.create_org("Acme").unwrap();
        let proj = cp.create_project(&org.id, "p").unwrap();

        cp.record_usage(&proj.id, 100, 50, 0.01);
        cp.record_usage(&proj.id, 200, 80, 0.02);
        let u = cp.usage(&proj.id);
        assert_eq!(u.requests, 2);
        assert_eq!(u.tokens_in, 300);
        assert_eq!(u.tokens_out, 130);
        assert!((u.cost_usd - 0.03).abs() < 1e-9);

        let sink = RecordingBillingSink::default();
        assert_eq!(cp.flush_billing(&sink).unwrap(), 1);
        let reports = sink.reports.read();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].0, proj.id);
        assert_eq!(reports[0].1.requests, 2);
    }
}
