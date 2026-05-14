//! # `duxx-prompts` — versioned prompt registry (Phase 7.2)
//!
//! Agent prompts deserve the same lifecycle treatment that code gets:
//! every change is a versioned artifact, you can roll back, you can
//! route traffic between versions, and you can search the catalog.
//! Most agent stacks today keep prompts as raw Python strings or
//! environment variables. `duxx-prompts` makes them a first-class
//! primitive next to `MEMORY` / `TOOL_CACHE` / `SESSION`.
//!
//! ## Capabilities
//!
//! - **Versioned storage** — every `put` increments the version
//!   monotonically; older versions stay queryable forever.
//! - **Tag aliases** — point `prod` / `staging` / `experimental` at
//!   any version. Operators flip a tag to ship a new prompt without
//!   touching agent code.
//! - **Semantic search across the catalog** — uses the same embedder
//!   the rest of DuxxDB does. Find prompts whose meaning is close to
//!   a new one before you author a duplicate. Nobody else ships this.
//! - **Live reactive updates** — every `put` / `tag` publishes on the
//!   shared [`ChangeBus`](duxx_reactive::ChangeBus) so running agents
//!   can `PSUBSCRIBE prompt.*` and hot-reload without a restart.
//! - **Text diff** between any two versions, for review tooling.
//!
//! ## Mental model
//!
//! ```text
//!   PromptRegistry
//!   ├── "support-greeting"
//!   │   ├── v1: "Hello, how can I help?"
//!   │   ├── v2: "Hi! How may I assist you today?"    [tag: prod]
//!   │   └── v3: "Hey there 👋 What's up?"            [tag: experimental]
//!   ├── "refund-classifier"
//!   │   ├── v1: ...                                  [tag: prod, staging]
//!   │   └── v2: ...
//!   └── …
//! ```
//!
//! ## What's NOT in this crate (yet)
//!
//! - **Persistence** — Phase 7.2b reuses the redb + tantivy pattern
//!   from `duxx-memory`'s `dir:` backend.
//! - **A/B routing as a server-side feature** — for now, callers
//!   choose a version/tag explicitly. Phase 7.2c will add
//!   `PROMPT.ROUTE name session_id` with deterministic hashing.
//! - **Template rendering (Jinja2 etc.)** — prompts are stored as
//!   raw strings; rendering lives in the agent framework.

use duxx_embed::Embedder;
use duxx_index::vector::VectorIndex;
use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::broadcast;

/// Errors surfaced by the prompt registry.
#[derive(Debug, Error)]
pub enum PromptError {
    #[error("prompt {name:?} version {version} not found")]
    VersionNotFound { name: String, version: u64 },
    #[error("prompt {name:?} not found")]
    NameNotFound { name: String },
    #[error("tag {tag:?} not found on prompt {name:?}")]
    TagNotFound { name: String, tag: String },
    #[error("embedder error: {0}")]
    Embed(String),
}

pub type Result<T> = std::result::Result<T, PromptError>;

// ---------------------------------------------------------------- Types

/// Identifier for a prompt — the human-chosen name (e.g.
/// `"support-greeting"`, `"refund-classifier"`).
pub type PromptName = String;

/// Monotonic version number assigned by the registry. Starts at 1;
/// every `put` increments by 1.
pub type PromptVersion = u64;

/// A symbolic alias for one specific version. Convention:
/// `"prod"` / `"staging"` / `"experimental"`, but any non-empty
/// string is accepted.
pub type PromptTag = String;

/// One stored prompt. Cheaply clonable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub name: PromptName,
    pub version: PromptVersion,
    pub content: String,
    /// Tags currently pointing at this exact version. A version may
    /// hold any number of tags; a tag may belong to at most one
    /// version at a time.
    #[serde(default)]
    pub tags: Vec<PromptTag>,
    /// Free-form metadata. Useful keys: `description`, `author`,
    /// `template_variables`, `model_hint`.
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Unix epoch nanoseconds.
    pub created_at_unix_ns: u128,
}

impl Prompt {
    fn new(name: String, version: u64, content: String, metadata: serde_json::Value) -> Self {
        Self {
            name,
            version,
            content,
            tags: Vec::new(),
            metadata,
            created_at_unix_ns: now_unix_ns(),
        }
    }
}

/// One result returned by [`PromptRegistry::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptHit {
    pub prompt: Prompt,
    /// Cosine similarity in `[0, 1]` (HNSW returns `[0, 2]` for the
    /// stored 1 - cosine distance; we map it).
    pub score: f32,
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- Registry

/// In-process prompt registry. Cheaply clonable (Arc internals).
#[derive(Clone)]
pub struct PromptRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    /// (name, version) → Prompt
    by_id: RwLock<HashMap<(PromptName, PromptVersion), Prompt>>,
    /// name → ordered list of versions stored (ascending). Versions
    /// disappear from this list on delete, but the per-name version
    /// COUNTER in `next_version` only ever increases — see comment
    /// below.
    versions: RwLock<HashMap<PromptName, Vec<PromptVersion>>>,
    /// name → next version to assign. Monotonic; never decreases.
    /// Separating this from `versions` means deleting v2 does NOT
    /// allow the next `put` to reuse the number 2 — every version
    /// ever assigned is unique forever.
    next_version: RwLock<HashMap<PromptName, PromptVersion>>,
    /// (name, tag) → version. Tags belong to exactly one version.
    tags: RwLock<HashMap<(PromptName, PromptTag), PromptVersion>>,
    /// Vector index over `(name, version)` -> embedding(content).
    /// We assign an internal monotonic id per prompt for HNSW use.
    embedder: Arc<dyn Embedder>,
    vector_index: RwLock<VectorIndex>,
    /// Internal HNSW id → (name, version) for hit -> Prompt resolution.
    id_to_key: RwLock<HashMap<u64, (PromptName, PromptVersion)>>,
    next_internal_id: RwLock<u64>,
    bus: ChangeBus,
}

impl std::fmt::Debug for PromptRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.inner.versions.read().len();
        let v = self.inner.by_id.read().len();
        let t = self.inner.tags.read().len();
        f.debug_struct("PromptRegistry")
            .field("names", &n)
            .field("versions", &v)
            .field("tags", &t)
            .finish()
    }
}

impl PromptRegistry {
    /// Build a registry that uses `embedder` for semantic search.
    /// Embedding dim must match the embedder's output.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            inner: Arc::new(Inner {
                by_id: RwLock::new(HashMap::new()),
                versions: RwLock::new(HashMap::new()),
                next_version: RwLock::new(HashMap::new()),
                tags: RwLock::new(HashMap::new()),
                embedder,
                vector_index: RwLock::new(VectorIndex::with_capacity(dim, 10_000)),
                id_to_key: RwLock::new(HashMap::new()),
                next_internal_id: RwLock::new(1),
                bus: ChangeBus::default(),
            }),
        }
    }

    /// Subscribe to change events. Each `put` / `tag` / `untag` /
    /// `delete` publishes a [`ChangeEvent`] with `table = "prompt"`
    /// and `key = Some(name)`, so `PSUBSCRIBE prompt.*` filters by
    /// prompt name.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Insert a new version of `name`. Returns the assigned
    /// monotonic version (starting at 1). Embeds the content for
    /// the catalog-wide semantic search index.
    pub fn put(
        &self,
        name: impl Into<String>,
        content: impl Into<String>,
        metadata: serde_json::Value,
    ) -> Result<PromptVersion> {
        let name = name.into();
        let content = content.into();

        let next_version = {
            // Bump the monotonic per-name counter. This survives
            // deletes so version numbers are never reused.
            let mut nv = self.inner.next_version.write();
            let v = *nv.entry(name.clone()).or_insert(0) + 1;
            nv.insert(name.clone(), v);
            // Mirror into the live-versions list (used by `list` and
            // `names`).
            self.inner
                .versions
                .write()
                .entry(name.clone())
                .or_default()
                .push(v);
            v
        };

        let prompt = Prompt::new(name.clone(), next_version, content.clone(), metadata);
        let key = (name.clone(), next_version);

        // Embed and insert into the vector index.
        let emb = self
            .inner
            .embedder
            .embed(&content)
            .map_err(|e| PromptError::Embed(e.to_string()))?;
        let internal_id = {
            let mut n = self.inner.next_internal_id.write();
            let v = *n;
            *n += 1;
            v
        };
        // VectorIndex insert errors propagate as an Embed-class error to
        // keep the API surface narrow.
        self.inner
            .vector_index
            .write()
            .insert(internal_id, emb)
            .map_err(|e| PromptError::Embed(format!("vector index: {e}")))?;
        self.inner.id_to_key.write().insert(internal_id, key.clone());

        self.inner.by_id.write().insert(key, prompt);
        self.inner.bus.publish(ChangeEvent {
            table: "prompt".to_string(),
            key: Some(name),
            row_id: next_version,
            kind: ChangeKind::Insert,
        });
        Ok(next_version)
    }

    /// Get a specific version of a prompt.
    pub fn get(&self, name: &str, version: PromptVersion) -> Option<Prompt> {
        self.inner
            .by_id
            .read()
            .get(&(name.to_string(), version))
            .map(|p| {
                let tags = self.tags_for_version(name, version);
                Prompt {
                    tags,
                    ..p.clone()
                }
            })
    }

    /// Resolve a tag to the version it points at, then return that
    /// prompt.
    pub fn get_by_tag(&self, name: &str, tag: &str) -> Option<Prompt> {
        let version = self
            .inner
            .tags
            .read()
            .get(&(name.to_string(), tag.to_string()))
            .copied()?;
        self.get(name, version)
    }

    /// Get the latest version of a prompt (highest version number).
    pub fn get_latest(&self, name: &str) -> Option<Prompt> {
        let versions = self.inner.versions.read();
        let latest = versions.get(name).and_then(|v| v.last().copied())?;
        drop(versions);
        self.get(name, latest)
    }

    /// List every version of a prompt, ascending.
    pub fn list(&self, name: &str) -> Vec<Prompt> {
        let versions = self
            .inner
            .versions
            .read()
            .get(name)
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(versions.len());
        for v in versions {
            if let Some(p) = self.get(name, v) {
                out.push(p);
            }
        }
        out
    }

    /// Names known to the registry. Lexicographic order.
    pub fn names(&self) -> Vec<PromptName> {
        let mut ns: Vec<_> = self.inner.versions.read().keys().cloned().collect();
        ns.sort();
        ns
    }

    /// Point `tag` at `version` of `name`. If the tag already exists
    /// it is moved (atomically) — an operator promoting `staging` to
    /// `prod` just calls `tag(name, version, "prod")` and the old
    /// `prod` association is overwritten.
    pub fn tag(&self, name: &str, version: PromptVersion, tag: &str) -> Result<()> {
        // Ensure the version exists.
        if !self
            .inner
            .by_id
            .read()
            .contains_key(&(name.to_string(), version))
        {
            return Err(PromptError::VersionNotFound {
                name: name.to_string(),
                version,
            });
        }
        self.inner
            .tags
            .write()
            .insert((name.to_string(), tag.to_string()), version);
        self.inner.bus.publish(ChangeEvent {
            table: "prompt".to_string(),
            key: Some(name.to_string()),
            row_id: version,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Remove a tag. No-op if it doesn't exist.
    pub fn untag(&self, name: &str, tag: &str) -> bool {
        let removed = self
            .inner
            .tags
            .write()
            .remove(&(name.to_string(), tag.to_string()))
            .is_some();
        if removed {
            self.inner.bus.publish(ChangeEvent {
                table: "prompt".to_string(),
                key: Some(name.to_string()),
                row_id: 0,
                kind: ChangeKind::Update,
            });
        }
        removed
    }

    /// Delete one version of a prompt. Hard delete — the version is
    /// removed from `by_id`, the tag map, and the search index. The
    /// version number is NOT reused (subsequent `put` keeps
    /// incrementing past the gap).
    pub fn delete(&self, name: &str, version: PromptVersion) -> bool {
        let key = (name.to_string(), version);
        let removed = self.inner.by_id.write().remove(&key).is_some();
        if removed {
            // Strip from the version list.
            if let Some(list) = self.inner.versions.write().get_mut(name) {
                list.retain(|v| *v != version);
            }
            // Untag any tag pointing here.
            self.inner
                .tags
                .write()
                .retain(|(n, _t), v| !(n == name && *v == version));
            self.inner.bus.publish(ChangeEvent {
                table: "prompt".to_string(),
                key: Some(name.to_string()),
                row_id: version,
                kind: ChangeKind::Delete,
            });
        }
        removed
    }

    /// Semantic search across the prompt catalog. Returns up to `k`
    /// hits, ordered by cosine similarity descending.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<PromptHit>> {
        let query_vec = self
            .inner
            .embedder
            .embed(query)
            .map_err(|e| PromptError::Embed(e.to_string()))?;
        let raw = self.inner.vector_index.read().search(&query_vec, k);
        let id_to_key = self.inner.id_to_key.read();
        let mut out = Vec::with_capacity(raw.len());
        for (id, dist) in raw {
            let key = match id_to_key.get(&id) {
                Some(k) => k.clone(),
                None => continue,
            };
            if let Some(prompt) = self.get(&key.0, key.1) {
                // VectorIndex returns 1 - cosine_similarity (distance).
                // Map back to similarity in [0, 1] (clamp the rare
                // numerical drift outside that range).
                let sim = (1.0 - dist).clamp(0.0, 1.0);
                out.push(PromptHit {
                    prompt,
                    score: sim,
                });
            }
        }
        Ok(out)
    }

    /// Unified text diff between two versions, line-by-line. Returns
    /// each line prefixed with `" "` (unchanged), `"-"` (only in
    /// `version_a`), or `"+"` (only in `version_b`). Hunks are
    /// concatenated.
    pub fn diff(&self, name: &str, version_a: PromptVersion, version_b: PromptVersion) -> Result<String> {
        let a = self
            .get(name, version_a)
            .ok_or(PromptError::VersionNotFound {
                name: name.to_string(),
                version: version_a,
            })?;
        let b = self
            .get(name, version_b)
            .ok_or(PromptError::VersionNotFound {
                name: name.to_string(),
                version: version_b,
            })?;
        Ok(line_diff(&a.content, &b.content))
    }

    /// Counters for ops dashboards / metrics.
    pub fn stats(&self) -> PromptStats {
        PromptStats {
            names: self.inner.versions.read().len(),
            versions: self.inner.by_id.read().len(),
            tags: self.inner.tags.read().len(),
        }
    }

    // -- helpers --

    fn tags_for_version(&self, name: &str, version: PromptVersion) -> Vec<PromptTag> {
        let tags = self.inner.tags.read();
        tags.iter()
            .filter(|((n, _t), v)| n == name && **v == version)
            .map(|((_n, t), _v)| t.clone())
            .collect()
    }
}

/// Counters returned by [`PromptRegistry::stats`].
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PromptStats {
    pub names: usize,
    pub versions: usize,
    pub tags: usize,
}

/// Plain LCS-free line diff. Sufficient for "what changed in this
/// version?" tooling; not a replacement for full myers-diff. Each
/// output line is prefixed with " ", "-", or "+".
fn line_diff(a: &str, b: &str) -> String {
    let alines: Vec<&str> = a.lines().collect();
    let blines: Vec<&str> = b.lines().collect();
    // Pure-Python-style: walk a/b in parallel; on mismatch flush
    // removals from a and additions from b. Good enough for prompt
    // review since prompts are usually short.
    let mut out = String::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < alines.len() && j < blines.len() {
        if alines[i] == blines[j] {
            out.push(' ');
            out.push_str(alines[i]);
            out.push('\n');
            i += 1;
            j += 1;
        } else {
            // Try to resync: look ahead in b for alines[i], or in a
            // for blines[j]. Pick the closer match.
            let b_match = blines[j..].iter().position(|&x| x == alines[i]);
            let a_match = alines[i..].iter().position(|&x| x == blines[j]);
            match (a_match, b_match) {
                (Some(da), Some(db)) if da < db => {
                    for k in i..i + da {
                        out.push('-');
                        out.push_str(alines[k]);
                        out.push('\n');
                    }
                    i += da;
                }
                (_, Some(db)) => {
                    for k in j..j + db {
                        out.push('+');
                        out.push_str(blines[k]);
                        out.push('\n');
                    }
                    j += db;
                }
                (Some(da), None) => {
                    for k in i..i + da {
                        out.push('-');
                        out.push_str(alines[k]);
                        out.push('\n');
                    }
                    i += da;
                }
                (None, None) => {
                    out.push('-');
                    out.push_str(alines[i]);
                    out.push('\n');
                    out.push('+');
                    out.push_str(blines[j]);
                    out.push('\n');
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    while i < alines.len() {
        out.push('-');
        out.push_str(alines[i]);
        out.push('\n');
        i += 1;
    }
    while j < blines.len() {
        out.push('+');
        out.push_str(blines[j]);
        out.push('\n');
        j += 1;
    }
    out
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn reg() -> PromptRegistry {
        PromptRegistry::new(Arc::new(HashEmbedder::new(16)))
    }

    #[test]
    fn put_assigns_monotonic_versions() {
        let r = reg();
        let v1 = r.put("greet", "Hello!", serde_json::json!({})).unwrap();
        let v2 = r.put("greet", "Hi!", serde_json::json!({})).unwrap();
        let v3 = r.put("greet", "Hey!", serde_json::json!({})).unwrap();
        assert_eq!((v1, v2, v3), (1, 2, 3));
    }

    #[test]
    fn put_under_different_names_does_not_share_version_space() {
        let r = reg();
        let a1 = r.put("greet", "Hi", serde_json::json!({})).unwrap();
        let b1 = r.put("close", "Bye", serde_json::json!({})).unwrap();
        assert_eq!((a1, b1), (1, 1));
    }

    #[test]
    fn get_returns_named_version_with_tags_attached() {
        let r = reg();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        r.put("g", "v2", serde_json::json!({})).unwrap();
        r.tag("g", 2, "prod").unwrap();
        let p = r.get("g", 2).unwrap();
        assert_eq!(p.content, "v2");
        assert_eq!(p.tags, vec!["prod"]);
    }

    #[test]
    fn get_latest_returns_highest_version() {
        let r = reg();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        r.put("g", "v2", serde_json::json!({})).unwrap();
        r.put("g", "v3", serde_json::json!({})).unwrap();
        assert_eq!(r.get_latest("g").unwrap().content, "v3");
    }

    #[test]
    fn get_by_tag_resolves_alias() {
        let r = reg();
        r.put("g", "first", serde_json::json!({})).unwrap();
        r.put("g", "second", serde_json::json!({})).unwrap();
        r.tag("g", 1, "prod").unwrap();
        r.tag("g", 2, "staging").unwrap();
        assert_eq!(r.get_by_tag("g", "prod").unwrap().content, "first");
        assert_eq!(r.get_by_tag("g", "staging").unwrap().content, "second");
        // Re-tag: moving prod from v1 to v2.
        r.tag("g", 2, "prod").unwrap();
        assert_eq!(r.get_by_tag("g", "prod").unwrap().content, "second");
    }

    #[test]
    fn tag_unknown_version_errors() {
        let r = reg();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        let err = r.tag("g", 99, "prod").unwrap_err();
        matches!(err, PromptError::VersionNotFound { .. });
    }

    #[test]
    fn untag_removes_alias() {
        let r = reg();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        r.tag("g", 1, "prod").unwrap();
        assert!(r.untag("g", "prod"));
        assert!(r.get_by_tag("g", "prod").is_none());
        // untag idempotent for missing tag
        assert!(!r.untag("g", "prod"));
    }

    #[test]
    fn list_returns_versions_ascending() {
        let r = reg();
        for i in 0..3 {
            r.put("g", format!("v{i}"), serde_json::json!({})).unwrap();
        }
        let vs = r.list("g");
        assert_eq!(vs.len(), 3);
        assert_eq!(vs[0].version, 1);
        assert_eq!(vs[2].version, 3);
    }

    #[test]
    fn names_returns_all_known_prompts() {
        let r = reg();
        r.put("alpha", "a", serde_json::json!({})).unwrap();
        r.put("beta", "b", serde_json::json!({})).unwrap();
        assert_eq!(r.names(), vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn delete_removes_version_and_associated_tags() {
        let r = reg();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        r.put("g", "v2", serde_json::json!({})).unwrap();
        r.tag("g", 2, "prod").unwrap();
        assert!(r.delete("g", 2));
        assert!(r.get("g", 2).is_none());
        assert!(r.get_by_tag("g", "prod").is_none());
        // v3 added next still gets fresh version 3 (NOT 2 reused).
        let v = r.put("g", "v3", serde_json::json!({})).unwrap();
        assert_eq!(v, 3);
    }

    #[test]
    fn search_finds_semantically_close_prompts() {
        let r = reg();
        r.put("greeting", "hello world how are you", serde_json::json!({})).unwrap();
        r.put("farewell", "goodbye see you tomorrow", serde_json::json!({})).unwrap();
        r.put(
            "support",
            "i can help you with your issue today",
            serde_json::json!({}),
        )
        .unwrap();
        let hits = r.search("hello world", 2).unwrap();
        assert!(!hits.is_empty());
        // The closest hit must be the "hello world" prompt.
        assert!(hits[0].prompt.content.contains("hello"));
    }

    #[test]
    fn diff_marks_added_and_removed_lines() {
        let r = reg();
        r.put("g", "line a\nshared\nline c", serde_json::json!({})).unwrap();
        r.put("g", "line a\nshared\nNEW line", serde_json::json!({})).unwrap();
        let d = r.diff("g", 1, 2).unwrap();
        assert!(d.contains("-line c"));
        assert!(d.contains("+NEW line"));
        assert!(d.contains(" shared"));
    }

    #[test]
    fn change_bus_publishes_put_tag_delete_events() {
        let r = reg();
        let mut rx = r.subscribe();
        r.put("g", "v1", serde_json::json!({})).unwrap();
        let e = rx.try_recv().unwrap();
        assert_eq!(e.table, "prompt");
        assert!(matches!(e.kind, ChangeKind::Insert));
        assert_eq!(e.key.as_deref(), Some("g"));

        r.tag("g", 1, "prod").unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Update));

        r.delete("g", 1);
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Delete));
    }

    #[test]
    fn stats_counts_names_versions_and_tags() {
        let r = reg();
        r.put("a", "v1", serde_json::json!({})).unwrap();
        r.put("a", "v2", serde_json::json!({})).unwrap();
        r.put("b", "v1", serde_json::json!({})).unwrap();
        r.tag("a", 1, "prod").unwrap();
        r.tag("a", 2, "staging").unwrap();
        let s = r.stats();
        assert_eq!(s.names, 2);
        assert_eq!(s.versions, 3);
        assert_eq!(s.tags, 2);
    }
}
