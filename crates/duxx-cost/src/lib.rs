//! # `duxx-cost` — token + cost ledger with budgets (Phase 7.6)
//!
//! Every agent stack burns money. Each LLM call costs tokens, each
//! retry doubles them, each tool fan-out multiplies them again. Most
//! ops teams discover the bill at the end of the month and try to
//! attribute the burn back to specific users / prompts / models / use
//! cases — usually with a hand-rolled Python script and a CSV export
//! from OpenAI.
//!
//! `duxx-cost` makes the ledger a first-class primitive. Record each
//! call as it happens. Set per-tenant budgets. Watch alerts fire in
//! real time. And — the move competitors can't replicate — **cluster
//! your most expensive queries semantically** to spot which kinds of
//! requests are driving spend.
//!
//! ## Capabilities
//!
//! - **Append-only ledger.** Every `CostEntry` records `(tenant,
//!   model, tokens_in, tokens_out, cost_usd, timestamp)` plus
//!   optional links to `trace_id`, `run_id`, `prompt`, and free-form
//!   metadata.
//! - **Filtered queries + aggregations.** Group by tenant / model /
//!   prompt / day. Standard `sum` / `count` / `mean`. Rolling
//!   windows for budget calculation.
//! - **Budgets per tenant.** Daily / weekly / monthly / custom-secs
//!   periods. Warn at `warn_pct` (default 80%), block at 100%.
//! - **Active alert feed.** `alerts()` returns every tenant currently
//!   over warn / over budget. `PSUBSCRIBE cost.alerts` for live push.
//! - **Semantic clustering of expensive queries.** Pass an
//!   embedder; `cluster_expensive` groups the top-N spenders by
//!   `input_text` semantic similarity. Answers "what kind of
//!   queries cost me the most last week?" in one call.
//!
//! ## What's NOT in this crate
//!
//! - **Pricing tables.** Callers supply `cost_usd` directly. Models +
//!   pricing change weekly; baking them in would mean shipping
//!   DuxxDB every time OpenAI announces a price drop.
//! - **Billing integration.** Stripe / Recurly / etc. is a caller
//!   responsibility. We give you the numbers; you charge for them.
//! - **Persistence.** Phase 7.x.b alongside the other in-memory
//!   stores.

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

/// Errors surfaced by the cost ledger.
#[derive(Debug, Error)]
pub enum CostError {
    #[error("budget for tenant {0:?} not found")]
    BudgetNotFound(String),
    #[error("negative cost not allowed: {0}")]
    NegativeCost(f64),
    #[error("embedder error: {0}")]
    Embed(String),
}

pub type Result<T> = std::result::Result<T, CostError>;

// ---------------------------------------------------------------- Types

/// One billable event recorded in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    /// Stable id. UUID-assigned at `record` time.
    pub id: String,
    /// The owner of this spend — usually a user / workspace / agent id.
    pub tenant: String,
    pub model: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// Dollar cost as a float. Sub-cent precision is fine.
    pub cost_usd: f64,
    /// Optional link to the trace this call happened inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Optional eval-run id if this entry was incurred during eval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Optional prompt-registry reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<u64>,
    /// Canonical text representation of the input. Used as the
    /// embedding source for `cluster_expensive`. Empty means "skip
    /// this entry when clustering".
    #[serde(default)]
    pub input_text: String,
    /// Free-form: agent name, branch, request id, etc.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub recorded_at_unix_ns: u128,
}

/// Budget period. Custom periods are useful for sprint-length or
/// experiment-window caps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BudgetPeriod {
    Daily,
    Weekly,
    Monthly,
    Custom { secs: u64 },
}

impl BudgetPeriod {
    pub fn duration_secs(self) -> u64 {
        match self {
            Self::Daily => 24 * 3600,
            Self::Weekly => 7 * 24 * 3600,
            Self::Monthly => 30 * 24 * 3600,
            Self::Custom { secs } => secs,
        }
    }
}

/// Per-tenant cap with a warn threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub tenant: String,
    pub period: BudgetPeriod,
    pub amount_usd: f64,
    /// 0.0 .. 1.0. Default 0.8.
    pub warn_pct: f32,
    /// Free-form: webhook URL, slack channel, owner email.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_unix_ns: u128,
}

/// Where a tenant stands against its budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BudgetStatus {
    /// No budget configured.
    NoBudget,
    /// Spent < warn_pct * amount.
    Ok,
    /// warn_pct * amount <= spent < amount.
    Warning,
    /// Spent >= amount.
    Exceeded,
}

/// One row of `alerts()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetAlert {
    pub tenant: String,
    pub status: BudgetStatus,
    pub spent_usd: f64,
    pub budget_usd: f64,
    pub period: BudgetPeriod,
    /// `spent / budget` (clamped to a sane upper bound for display).
    pub utilization: f64,
}

/// Filter for queries + aggregations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostFilter {
    pub tenant: Option<String>,
    pub model: Option<String>,
    pub prompt_name: Option<String>,
    pub since_unix_ns: Option<u128>,
    pub until_unix_ns: Option<u128>,
    /// Cap the returned rows. 0 = no limit.
    #[serde(default)]
    pub limit: usize,
}

/// Grouping axis for `aggregate`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    Tenant,
    Model,
    Prompt,
    DayUtc,
    None,
}

/// Aggregated slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBucket {
    pub key: String,
    pub count: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub total_usd: f64,
    pub mean_usd: f64,
}

/// One semantic cluster of expensive queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpenseCluster {
    pub representative_entry_id: String,
    pub representative_input: String,
    pub total_usd: f64,
    pub members: Vec<ExpenseClusterMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpenseClusterMember {
    pub entry_id: String,
    pub cost_usd: f64,
    pub similarity: f32,
}

/// Counters for ops dashboards.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CostStats {
    pub entries: usize,
    pub tenants_with_budget: usize,
    pub total_usd: f64,
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- Ledger

/// In-process cost ledger. Cheaply clonable.
#[derive(Clone)]
pub struct CostLedger {
    inner: Arc<Inner>,
}

struct Inner {
    /// Append-only flat list. Linear scan for queries; that's fine
    /// for in-memory volumes. Phase 7.x.b adds a tantivy-backed
    /// time-windowed secondary index for million-row corpora.
    entries: RwLock<Vec<CostEntry>>,
    /// tenant -> Budget
    budgets: RwLock<HashMap<String, Budget>>,
    /// Shared embedder for clustering. None → clustering returns an error.
    embedder: Arc<dyn Embedder>,
    /// Vector index over `input_text` of every entry that has one.
    vector_index: RwLock<VectorIndex>,
    /// Internal HNSW id -> entry id.
    id_to_entry: RwLock<HashMap<u64, String>>,
    next_internal_id: RwLock<u64>,
    bus: ChangeBus,
}

impl std::fmt::Debug for CostLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("CostLedger")
            .field("entries", &s.entries)
            .field("tenants_with_budget", &s.tenants_with_budget)
            .field("total_usd", &s.total_usd)
            .finish()
    }
}

impl CostLedger {
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            inner: Arc::new(Inner {
                entries: RwLock::new(Vec::new()),
                budgets: RwLock::new(HashMap::new()),
                embedder,
                vector_index: RwLock::new(VectorIndex::with_capacity(dim, 100_000)),
                id_to_entry: RwLock::new(HashMap::new()),
                next_internal_id: RwLock::new(1),
                bus: ChangeBus::default(),
            }),
        }
    }

    /// Subscribe to change events. `table = "cost"`, `key =
    /// Some(tenant)`. Includes both ledger inserts and budget
    /// status transitions (use `kind: Update` for the latter).
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Append one cost entry. UUID assigned automatically if the
    /// caller didn't provide one. Returns the assigned id.
    pub fn record(&self, mut entry: CostEntry) -> Result<String> {
        if entry.cost_usd < 0.0 {
            return Err(CostError::NegativeCost(entry.cost_usd));
        }
        if entry.id.is_empty() {
            entry.id = uuid::Uuid::new_v4().simple().to_string();
        }
        if entry.recorded_at_unix_ns == 0 {
            entry.recorded_at_unix_ns = now_unix_ns();
        }
        let entry_id = entry.id.clone();
        let tenant = entry.tenant.clone();

        // Index for clustering if we have input text.
        if !entry.input_text.is_empty() {
            let emb = self
                .inner
                .embedder
                .embed(&entry.input_text)
                .map_err(|e| CostError::Embed(e.to_string()))?;
            let internal_id = {
                let mut n = self.inner.next_internal_id.write();
                let v = *n;
                *n += 1;
                v
            };
            self.inner
                .vector_index
                .write()
                .insert(internal_id, emb)
                .map_err(|e| CostError::Embed(format!("vector index: {e}")))?;
            self.inner
                .id_to_entry
                .write()
                .insert(internal_id, entry_id.clone());
        }

        self.inner.entries.write().push(entry);
        self.inner.bus.publish(ChangeEvent {
            table: "cost".to_string(),
            key: Some(tenant.clone()),
            row_id: 0,
            kind: ChangeKind::Insert,
        });
        // Emit a budget-status alert if applicable.
        if let Some(alert) = self.alert_for(&tenant) {
            if matches!(alert.status, BudgetStatus::Warning | BudgetStatus::Exceeded) {
                self.inner.bus.publish(ChangeEvent {
                    table: "cost.alerts".to_string(),
                    key: Some(tenant),
                    row_id: 0,
                    kind: ChangeKind::Update,
                });
            }
        }
        Ok(entry_id)
    }

    /// Query the ledger, returning entries newest first.
    pub fn query(&self, filter: &CostFilter) -> Vec<CostEntry> {
        let entries = self.inner.entries.read();
        let mut out: Vec<CostEntry> = entries
            .iter()
            .filter(|e| self.match_filter(e, filter))
            .cloned()
            .collect();
        out.sort_by_key(|e| std::cmp::Reverse(e.recorded_at_unix_ns));
        if filter.limit > 0 && out.len() > filter.limit {
            out.truncate(filter.limit);
        }
        out
    }

    fn match_filter(&self, e: &CostEntry, f: &CostFilter) -> bool {
        if let Some(t) = &f.tenant {
            if &e.tenant != t {
                return false;
            }
        }
        if let Some(m) = &f.model {
            if &e.model != m {
                return false;
            }
        }
        if let Some(p) = &f.prompt_name {
            if e.prompt_name.as_deref() != Some(p.as_str()) {
                return false;
            }
        }
        if let Some(s) = f.since_unix_ns {
            if e.recorded_at_unix_ns < s {
                return false;
            }
        }
        if let Some(u) = f.until_unix_ns {
            if e.recorded_at_unix_ns >= u {
                return false;
            }
        }
        true
    }

    /// Aggregate the filtered set by one of `Tenant` / `Model` /
    /// `Prompt` / `DayUtc` / `None` (one big bucket).
    pub fn aggregate(&self, filter: &CostFilter, group_by: GroupBy) -> Vec<CostBucket> {
        let entries = self.inner.entries.read();
        let mut buckets: HashMap<String, CostBucket> = HashMap::new();
        for e in entries.iter().filter(|e| self.match_filter(e, filter)) {
            let key = match group_by {
                GroupBy::Tenant => e.tenant.clone(),
                GroupBy::Model => e.model.clone(),
                GroupBy::Prompt => e
                    .prompt_name
                    .clone()
                    .unwrap_or_else(|| "(none)".to_string()),
                GroupBy::DayUtc => day_bucket(e.recorded_at_unix_ns),
                GroupBy::None => "(all)".to_string(),
            };
            let b = buckets.entry(key.clone()).or_insert(CostBucket {
                key,
                count: 0,
                tokens_in: 0,
                tokens_out: 0,
                total_usd: 0.0,
                mean_usd: 0.0,
            });
            b.count += 1;
            b.tokens_in += e.tokens_in;
            b.tokens_out += e.tokens_out;
            b.total_usd += e.cost_usd;
        }
        for b in buckets.values_mut() {
            b.mean_usd = if b.count > 0 {
                b.total_usd / b.count as f64
            } else {
                0.0
            };
        }
        let mut out: Vec<_> = buckets.into_values().collect();
        // Sort buckets by total_usd descending — biggest spenders first.
        out.sort_by(|a, b| b.total_usd.partial_cmp(&a.total_usd).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Convenience: total spend for a tenant within an optional window.
    pub fn total_for(&self, tenant: &str, since_unix_ns: Option<u128>, until_unix_ns: Option<u128>) -> f64 {
        let filter = CostFilter {
            tenant: Some(tenant.to_string()),
            since_unix_ns,
            until_unix_ns,
            ..Default::default()
        };
        self.inner
            .entries
            .read()
            .iter()
            .filter(|e| self.match_filter(e, &filter))
            .map(|e| e.cost_usd)
            .sum()
    }

    /// Set or replace a budget. `warn_pct` is clamped to `[0.0, 1.0]`;
    /// pass 0.0 to disable warnings.
    pub fn set_budget(
        &self,
        tenant: impl Into<String>,
        period: BudgetPeriod,
        amount_usd: f64,
        warn_pct: f32,
        metadata: serde_json::Value,
    ) -> Result<()> {
        let tenant = tenant.into();
        let warn_pct = warn_pct.clamp(0.0, 1.0);
        let budget = Budget {
            tenant: tenant.clone(),
            period,
            amount_usd,
            warn_pct,
            metadata,
            created_at_unix_ns: now_unix_ns(),
        };
        self.inner.budgets.write().insert(tenant.clone(), budget);
        self.inner.bus.publish(ChangeEvent {
            table: "cost".to_string(),
            key: Some(tenant),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Get a budget.
    pub fn get_budget(&self, tenant: &str) -> Option<Budget> {
        self.inner.budgets.read().get(tenant).cloned()
    }

    /// Remove a budget. Returns true if it existed.
    pub fn delete_budget(&self, tenant: &str) -> bool {
        let removed = self.inner.budgets.write().remove(tenant).is_some();
        if removed {
            self.inner.bus.publish(ChangeEvent {
                table: "cost".to_string(),
                key: Some(tenant.into()),
                row_id: 0,
                kind: ChangeKind::Delete,
            });
        }
        removed
    }

    /// Where the tenant stands right now against its budget. The
    /// rolling window is `period`-long, ending at "now".
    pub fn budget_status(&self, tenant: &str) -> BudgetStatus {
        let budget = match self.get_budget(tenant) {
            Some(b) => b,
            None => return BudgetStatus::NoBudget,
        };
        let now = now_unix_ns();
        let window_secs = budget.period.duration_secs();
        let since = now.saturating_sub((window_secs as u128) * 1_000_000_000);
        let spent = self.total_for(tenant, Some(since), Some(now));
        if spent >= budget.amount_usd {
            BudgetStatus::Exceeded
        } else if budget.warn_pct > 0.0
            && spent >= (budget.amount_usd * budget.warn_pct as f64)
        {
            BudgetStatus::Warning
        } else {
            BudgetStatus::Ok
        }
    }

    /// One-tenant alert helper.
    fn alert_for(&self, tenant: &str) -> Option<BudgetAlert> {
        let budget = self.get_budget(tenant)?;
        let status = self.budget_status(tenant);
        let now = now_unix_ns();
        let window_secs = budget.period.duration_secs();
        let since = now.saturating_sub((window_secs as u128) * 1_000_000_000);
        let spent = self.total_for(tenant, Some(since), Some(now));
        let utilization = if budget.amount_usd > 0.0 {
            spent / budget.amount_usd
        } else {
            0.0
        };
        Some(BudgetAlert {
            tenant: tenant.into(),
            status,
            spent_usd: spent,
            budget_usd: budget.amount_usd,
            period: budget.period,
            utilization,
        })
    }

    /// Every tenant currently above its `warn_pct` (or over budget).
    pub fn alerts(&self) -> Vec<BudgetAlert> {
        let tenants: Vec<String> = self.inner.budgets.read().keys().cloned().collect();
        let mut out = Vec::new();
        for t in tenants {
            if let Some(alert) = self.alert_for(&t) {
                if matches!(alert.status, BudgetStatus::Warning | BudgetStatus::Exceeded) {
                    out.push(alert);
                }
            }
        }
        // Show most-blown budgets first.
        out.sort_by(|a, b| {
            b.utilization
                .partial_cmp(&a.utilization)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Cluster the top-cost entries semantically — "what kinds of
    /// queries cost me the most?" Selects the top-`limit` entries by
    /// `cost_usd` from the filtered set (default 50), then greedy-
    /// clusters them by cosine similarity of `input_text`.
    ///
    /// `sim_threshold` defaults to 0.7. `max_clusters` defaults to 10.
    pub fn cluster_expensive(
        &self,
        filter: &CostFilter,
        sim_threshold: f32,
        max_clusters: usize,
        top_n: usize,
    ) -> Result<Vec<ExpenseCluster>> {
        let smt = if sim_threshold == 0.0 { 0.7 } else { sim_threshold };
        let max_k = if max_clusters == 0 { 10 } else { max_clusters };
        let top = if top_n == 0 { 50 } else { top_n };

        // Pick the top-N most expensive entries that have input text.
        let mut candidates: Vec<CostEntry> = self
            .inner
            .entries
            .read()
            .iter()
            .filter(|e| self.match_filter(e, filter) && !e.input_text.is_empty())
            .cloned()
            .collect();
        candidates.sort_by(|a, b| {
            b.cost_usd
                .partial_cmp(&a.cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top);

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut clusters: Vec<ExpenseCluster> = Vec::new();

        let candidate_ids: std::collections::HashSet<String> =
            candidates.iter().map(|c| c.id.clone()).collect();

        for seed in &candidates {
            if visited.contains(&seed.id) {
                continue;
            }
            visited.insert(seed.id.clone());

            let seed_emb = self
                .inner
                .embedder
                .embed(&seed.input_text)
                .map_err(|e| CostError::Embed(e.to_string()))?;
            let neighbors = self.inner.vector_index.read().search(&seed_emb, 64);

            let mut members: Vec<ExpenseClusterMember> = vec![ExpenseClusterMember {
                entry_id: seed.id.clone(),
                cost_usd: seed.cost_usd,
                similarity: 1.0,
            }];
            let mut total = seed.cost_usd;

            let id_to_entry = self.inner.id_to_entry.read();
            for (internal_id, dist) in neighbors {
                let other_id = match id_to_entry.get(&internal_id) {
                    Some(s) => s.clone(),
                    None => continue,
                };
                if other_id == seed.id || visited.contains(&other_id) {
                    continue;
                }
                // Only include if the entry is in our top-N candidate set.
                if !candidate_ids.contains(&other_id) {
                    continue;
                }
                let sim = (1.0 - dist).clamp(0.0, 1.0);
                if sim < smt {
                    continue;
                }
                if let Some(other_entry) = candidates.iter().find(|c| c.id == other_id) {
                    total += other_entry.cost_usd;
                    members.push(ExpenseClusterMember {
                        entry_id: other_id.clone(),
                        cost_usd: other_entry.cost_usd,
                        similarity: sim,
                    });
                    visited.insert(other_id);
                }
            }
            clusters.push(ExpenseCluster {
                representative_entry_id: seed.id.clone(),
                representative_input: seed.input_text.clone(),
                total_usd: total,
                members,
            });
        }
        // Sort clusters by total spend descending — biggest cost
        // sinks first.
        clusters.sort_by(|a, b| {
            b.total_usd
                .partial_cmp(&a.total_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        clusters.truncate(max_k);
        Ok(clusters)
    }

    /// Counters for ops dashboards.
    pub fn stats(&self) -> CostStats {
        let entries = self.inner.entries.read();
        let total_usd: f64 = entries.iter().map(|e| e.cost_usd).sum();
        CostStats {
            entries: entries.len(),
            tenants_with_budget: self.inner.budgets.read().len(),
            total_usd,
        }
    }
}

/// Bucket a unix-ns timestamp into a "YYYY-MM-DD" UTC string. Cheap
/// integer arithmetic — no chrono dependency.
fn day_bucket(unix_ns: u128) -> String {
    let secs = (unix_ns / 1_000_000_000) as u64;
    // Days since 1970-01-01.
    let days = secs / 86_400;
    // Civil-from-days algorithm (Howard Hinnant, public domain).
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn ledger() -> CostLedger {
        CostLedger::new(Arc::new(HashEmbedder::new(16)))
    }

    fn entry(tenant: &str, model: &str, cost: f64, input: &str) -> CostEntry {
        CostEntry {
            id: String::new(),
            tenant: tenant.into(),
            model: model.into(),
            tokens_in: 100,
            tokens_out: 200,
            cost_usd: cost,
            trace_id: None,
            run_id: None,
            prompt_name: None,
            prompt_version: None,
            input_text: input.into(),
            metadata: serde_json::Value::Null,
            recorded_at_unix_ns: 0,
        }
    }

    #[test]
    fn record_assigns_id_and_timestamp() {
        let l = ledger();
        let id = l.record(entry("acme", "gpt-4o", 0.01, "hi")).unwrap();
        assert_eq!(id.len(), 32); // uuid simple form
        let entries = l.query(&CostFilter::default());
        assert_eq!(entries[0].id, id);
        assert!(entries[0].recorded_at_unix_ns > 0);
    }

    #[test]
    fn record_rejects_negative_cost() {
        let l = ledger();
        let err = l.record(entry("acme", "gpt-4o", -1.0, "hi")).unwrap_err();
        matches!(err, CostError::NegativeCost(_));
    }

    #[test]
    fn query_filters_by_tenant_and_model() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 0.01, "a")).unwrap();
        l.record(entry("acme", "claude", 0.02, "b")).unwrap();
        l.record(entry("globex", "gpt-4o", 0.03, "c")).unwrap();
        let acme = l.query(&CostFilter {
            tenant: Some("acme".into()),
            ..Default::default()
        });
        assert_eq!(acme.len(), 2);
        let gpt = l.query(&CostFilter {
            model: Some("gpt-4o".into()),
            ..Default::default()
        });
        assert_eq!(gpt.len(), 2);
    }

    #[test]
    fn aggregate_groups_by_tenant() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 1.00, "")).unwrap();
        l.record(entry("acme", "gpt-4o", 2.00, "")).unwrap();
        l.record(entry("globex", "gpt-4o", 5.00, "")).unwrap();
        let buckets = l.aggregate(&CostFilter::default(), GroupBy::Tenant);
        assert_eq!(buckets.len(), 2);
        // Globex first (5.00 > 3.00).
        assert_eq!(buckets[0].key, "globex");
        assert!((buckets[0].total_usd - 5.0).abs() < 0.001);
        assert!((buckets[1].total_usd - 3.0).abs() < 0.001);
    }

    #[test]
    fn aggregate_by_model_sums_correctly() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 1.50, "")).unwrap();
        l.record(entry("acme", "claude", 0.75, "")).unwrap();
        let buckets = l.aggregate(&CostFilter::default(), GroupBy::Model);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].key, "gpt-4o");
    }

    #[test]
    fn total_for_aggregates_per_tenant() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 1.25, "")).unwrap();
        l.record(entry("acme", "claude", 0.50, "")).unwrap();
        l.record(entry("globex", "gpt-4o", 0.10, "")).unwrap();
        assert!((l.total_for("acme", None, None) - 1.75).abs() < 0.001);
    }

    #[test]
    fn budget_status_transitions_ok_warning_exceeded() {
        let l = ledger();
        l.set_budget(
            "acme",
            BudgetPeriod::Daily,
            10.0,
            0.8,
            serde_json::Value::Null,
        )
        .unwrap();
        // Empty: Ok.
        assert_eq!(l.budget_status("acme"), BudgetStatus::Ok);
        // Spend $7 -> still Ok.
        l.record(entry("acme", "gpt-4o", 7.0, "")).unwrap();
        assert_eq!(l.budget_status("acme"), BudgetStatus::Ok);
        // Spend +$2 -> warning (at 9 of 10, warn_pct=0.8).
        l.record(entry("acme", "gpt-4o", 2.0, "")).unwrap();
        assert_eq!(l.budget_status("acme"), BudgetStatus::Warning);
        // Spend +$2 more -> exceeded.
        l.record(entry("acme", "gpt-4o", 2.0, "")).unwrap();
        assert_eq!(l.budget_status("acme"), BudgetStatus::Exceeded);
    }

    #[test]
    fn budget_status_no_budget_returns_no_budget() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 99.0, "")).unwrap();
        assert_eq!(l.budget_status("acme"), BudgetStatus::NoBudget);
    }

    #[test]
    fn alerts_returns_only_at_or_above_warn() {
        let l = ledger();
        l.set_budget("acme", BudgetPeriod::Daily, 10.0, 0.8, serde_json::Value::Null).unwrap();
        l.set_budget("globex", BudgetPeriod::Daily, 100.0, 0.8, serde_json::Value::Null).unwrap();
        l.record(entry("acme", "gpt-4o", 9.0, "")).unwrap(); // 90% -> warning
        l.record(entry("globex", "gpt-4o", 1.0, "")).unwrap(); // 1% -> ok
        let alerts = l.alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].tenant, "acme");
        assert_eq!(alerts[0].status, BudgetStatus::Warning);
    }

    #[test]
    fn cluster_expensive_groups_by_semantic_similarity() {
        let l = ledger();
        // Expensive entries about phone-number queries (varying wording so
        // HashEmbedder produces distinct but similar vectors) + cheap
        // unrelated ones.
        l.record(entry("acme", "gpt-4o", 5.0, "user asked for a phone number lookup"))
            .unwrap();
        l.record(entry("acme", "gpt-4o", 4.5, "phone number lookup with extras"))
            .unwrap();
        l.record(entry("acme", "gpt-4o", 4.0, "look up phone number for the user"))
            .unwrap();
        l.record(entry("acme", "gpt-4o", 0.01, "what's the weather today"))
            .unwrap();
        l.record(entry("acme", "gpt-4o", 0.01, "completely unrelated query about cars"))
            .unwrap();
        let clusters = l
            .cluster_expensive(&CostFilter::default(), 0.5, 5, 50)
            .unwrap();
        assert!(!clusters.is_empty(), "expected at least one cluster");
        // The first cluster's representative MUST be one of the
        // expensive phone-number entries (top by cost). Tests that the
        // ordering by total_usd places the expensive cluster first.
        assert!(
            clusters[0].representative_input.contains("phone")
                || clusters[0].total_usd >= 4.0,
            "top cluster should cover the expensive phone queries; got: {:?}",
            clusters[0]
        );
    }

    #[test]
    fn change_bus_emits_insert_on_record() {
        let l = ledger();
        let mut rx = l.subscribe();
        l.record(entry("acme", "gpt-4o", 0.01, "")).unwrap();
        let e = rx.try_recv().unwrap();
        assert_eq!(e.table, "cost");
        assert_eq!(e.key.as_deref(), Some("acme"));
        assert!(matches!(e.kind, ChangeKind::Insert));
    }

    #[test]
    fn change_bus_emits_alert_when_warn_crossed() {
        let l = ledger();
        l.set_budget("acme", BudgetPeriod::Daily, 10.0, 0.8, serde_json::Value::Null).unwrap();
        // Drain initial budget-set event.
        let mut rx = l.subscribe();
        // First spend below warn — only normal insert.
        l.record(entry("acme", "gpt-4o", 5.0, "")).unwrap();
        let e = rx.try_recv().unwrap();
        assert_eq!(e.table, "cost");
        // Pull a second event if one exists; we DON'T expect cost.alerts here.
        if let Ok(e2) = rx.try_recv() {
            assert_ne!(e2.table, "cost.alerts");
        }
        // Cross the warn line.
        l.record(entry("acme", "gpt-4o", 4.0, "")).unwrap();
        // Drain to find the cost.alerts event.
        let mut saw_alert = false;
        while let Ok(ev) = rx.try_recv() {
            if ev.table == "cost.alerts" {
                saw_alert = true;
                break;
            }
        }
        assert!(saw_alert, "expected a cost.alerts event after crossing warn");
    }

    #[test]
    fn delete_budget_removes_it() {
        let l = ledger();
        l.set_budget("acme", BudgetPeriod::Daily, 10.0, 0.8, serde_json::Value::Null).unwrap();
        assert!(l.get_budget("acme").is_some());
        assert!(l.delete_budget("acme"));
        assert!(l.get_budget("acme").is_none());
        assert!(!l.delete_budget("acme"));
    }

    #[test]
    fn aggregate_group_by_day_buckets() {
        let l = ledger();
        // Two entries on different unix-ns days.
        let mut a = entry("acme", "gpt-4o", 1.0, "");
        a.recorded_at_unix_ns = 1_700_000_000 * 1_000_000_000;
        let mut b = entry("acme", "gpt-4o", 2.0, "");
        b.recorded_at_unix_ns = a.recorded_at_unix_ns + 86_400 * 1_000_000_000;
        l.record(a).unwrap();
        l.record(b).unwrap();
        let buckets = l.aggregate(&CostFilter::default(), GroupBy::DayUtc);
        assert_eq!(buckets.len(), 2);
    }

    #[test]
    fn stats_counts_entries_budgets_total() {
        let l = ledger();
        l.record(entry("acme", "gpt-4o", 1.0, "")).unwrap();
        l.record(entry("globex", "gpt-4o", 2.0, "")).unwrap();
        l.set_budget("acme", BudgetPeriod::Daily, 10.0, 0.8, serde_json::Value::Null).unwrap();
        let s = l.stats();
        assert_eq!(s.entries, 2);
        assert_eq!(s.tenants_with_budget, 1);
        assert!((s.total_usd - 3.0).abs() < 0.001);
    }

    #[test]
    fn query_filter_with_time_window_works() {
        let l = ledger();
        let mut old = entry("acme", "gpt-4o", 1.0, "");
        old.recorded_at_unix_ns = 1_000_000;
        let mut new = entry("acme", "gpt-4o", 2.0, "");
        new.recorded_at_unix_ns = 9_000_000;
        l.record(old).unwrap();
        l.record(new).unwrap();
        let just_old = l.query(&CostFilter {
            until_unix_ns: Some(5_000_000),
            ..Default::default()
        });
        assert_eq!(just_old.len(), 1);
        assert!((just_old[0].cost_usd - 1.0).abs() < 0.001);
    }
}
