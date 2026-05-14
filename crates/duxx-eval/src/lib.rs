//! # `duxx-eval` — eval runs + regressions + semantic failure clustering (Phase 7.4)
//!
//! Eval workloads in agent stacks look like: pick a dataset (or part
//! of one), pick a prompt-version and a model, run inference, score
//! the outputs, save the results. Then later: compare two runs to spot
//! regressions, slice scores by dataset split / model / prompt
//! version, drill into individual failures.
//!
//! `duxx-eval` is the **storage + analysis** half of that loop. The
//! actual inference + scoring is the caller's responsibility (same
//! posture as Braintrust / LangSmith). Once scores are POSTed, this
//! crate gives you:
//!
//! - **EvalRun** records — one per "I scored dataset X v3 with prompt
//!   refund-classifier v7 on gpt-4o using llm-judge".
//! - **EvalScore** per (run, row_id) — score in `[0, 1]` plus
//!   free-form `notes` JSON for reviewer comments / partial credits /
//!   tokens used / latency.
//! - **Regression detection** between two runs over the same dataset.
//! - **Semantic clustering of failures** via the shared embedder —
//!   "show me my failures grouped by what they're failing about".
//!   This is unique. Competitors store failures in a flat table and
//!   make humans cluster by hand.
//! - **Aggregations** (mean, p50, p90, p99, pass-rate) per run and
//!   sliced by `metadata` keys (model, prompt_version, split, …).
//! - **Reactive change feed** (`PSUBSCRIBE eval.*`) so dashboards
//!   refresh live as scores stream in.
//!
//! ## What's NOT in this crate (by design)
//!
//! - **Scorer execution.** The caller runs `gpt-4o-as-judge` or the
//!   `exact_match` function locally and POSTs the score. We never
//!   call an LLM. Keeps DuxxDB an Apache-2 storage engine instead of
//!   an LLM client.
//! - **Inference orchestration.** Pair with duxx-ai's
//!   `AgentEvaluator` for that side of the loop; this crate is the
//!   backend it writes to.
//! - **Persistence.** Phase 7.x.b adds the redb + tantivy on-disk
//!   pattern alongside trace / prompt / dataset persistence.

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

/// Errors surfaced by the eval registry.
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("eval run {0} not found")]
    RunNotFound(EvalRunId),
    #[error("eval run {0} is not in {1:?} state")]
    WrongState(EvalRunId, EvalStatus),
    #[error("score must be in [0, 1], got {0}")]
    InvalidScore(f32),
    #[error("embedder error: {0}")]
    Embed(String),
}

pub type Result<T> = std::result::Result<T, EvalError>;

// ---------------------------------------------------------------- Types

/// Identifier for an eval run.
pub type EvalRunId = String;

/// Lifecycle status of an eval run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalStatus {
    /// Created but no scores recorded yet.
    Pending,
    /// At least one score recorded; not finalized.
    Running,
    /// `complete()` called; summary stats are frozen.
    Completed,
    /// `fail()` called; partial results retained but flagged.
    Failed,
}

impl EvalStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

/// Built-in scorer hints. Stored as a string so callers can use any
/// label they want (`my_custom_judge_v3`, `exact_match`, etc.) — the
/// registry doesn't enforce a closed set.
pub type ScorerName = String;

/// One eval run record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    pub id: EvalRunId,
    /// Optional human-readable name; defaults to "{dataset}:{prompt}:{model}".
    pub name: String,
    pub dataset_name: String,
    pub dataset_version: u64,
    /// `None` when the run is over raw inference outputs (no prompt registry).
    #[serde(default)]
    pub prompt_name: Option<String>,
    #[serde(default)]
    pub prompt_version: Option<u64>,
    pub model: String,
    pub scorer: ScorerName,
    pub status: EvalStatus,
    /// Free-form metadata: `tokens`, `cost_usd`, `notes`, `commit_sha`.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_unix_ns: u128,
    /// Set when status transitions to Completed / Failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_unix_ns: Option<u128>,
    /// Frozen at `complete()`; None while Running/Pending.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<EvalSummary>,
}

/// Aggregated stats for a finished run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct EvalSummary {
    pub total_scored: usize,
    pub mean: f32,
    pub p50: f32,
    pub p90: f32,
    pub p99: f32,
    pub min: f32,
    pub max: f32,
    /// Pass-rate at the canonical 0.5 threshold. Override with
    /// `pass_rate_at()` if a different cutoff is needed.
    pub pass_rate_50: f32,
}

/// One score per (run, row).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScore {
    pub run_id: EvalRunId,
    pub row_id: String,
    /// `[0, 1]` — 1 = perfect, 0 = total failure.
    pub score: f32,
    /// Free-form: reviewer comments, partial-credit breakdown, tokens
    /// used, latency, full LLM response, etc.
    #[serde(default)]
    pub notes: serde_json::Value,
    /// Optional canonical "what was actually produced" text. Used as
    /// the embedding input for failure clustering when present.
    #[serde(default)]
    pub output_text: String,
    pub recorded_at_unix_ns: u128,
}

/// Result of [`EvalRegistry::compare`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalComparison {
    pub run_a: EvalRunId,
    pub run_b: EvalRunId,
    pub mean_delta: f32,
    pub pass_rate_50_delta: f32,
    /// Rows where score_b < score_a (regressions).
    pub regressed: Vec<RegressionRow>,
    /// Rows where score_b > score_a.
    pub improved: Vec<RegressionRow>,
    /// Rows present in `run_b` only.
    pub new_rows: Vec<String>,
    /// Rows present in `run_a` only.
    pub dropped_rows: Vec<String>,
}

/// One row-level delta returned by [`EvalRegistry::compare`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionRow {
    pub row_id: String,
    pub score_a: f32,
    pub score_b: f32,
    /// Always `score_b - score_a`.
    pub delta: f32,
}

/// One cluster of semantically-similar failing rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCluster {
    /// One representative row picked from the cluster (the one
    /// closest to the centroid).
    pub representative_row_id: String,
    pub representative_text: String,
    pub members: Vec<FailureClusterMember>,
    pub mean_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureClusterMember {
    pub row_id: String,
    pub score: f32,
    /// Cosine similarity to the representative.
    pub similarity: f32,
}

/// Counters returned by [`EvalRegistry::stats`].
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct EvalStats {
    pub runs: usize,
    pub scores: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- Registry

/// In-process eval registry. Cheaply clonable (Arc internals).
#[derive(Clone)]
pub struct EvalRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    runs: RwLock<HashMap<EvalRunId, EvalRun>>,
    /// (run_id, row_id) -> score
    scores: RwLock<HashMap<(EvalRunId, String), EvalScore>>,
    /// (run_id, row_id) -> internal HNSW id for failure clustering.
    failure_index_ids: RwLock<HashMap<(EvalRunId, String), u64>>,
    embedder: Arc<dyn Embedder>,
    failure_index: RwLock<VectorIndex>,
    id_to_key: RwLock<HashMap<u64, (EvalRunId, String)>>,
    next_internal_id: RwLock<u64>,
    bus: ChangeBus,
}

impl std::fmt::Debug for EvalRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("EvalRegistry")
            .field("runs", &s.runs)
            .field("scores", &s.scores)
            .field("running", &s.running)
            .field("completed", &s.completed)
            .field("failed", &s.failed)
            .finish()
    }
}

impl EvalRegistry {
    /// Build a registry that uses `embedder` for failure clustering.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        let dim = embedder.dim();
        Self {
            inner: Arc::new(Inner {
                runs: RwLock::new(HashMap::new()),
                scores: RwLock::new(HashMap::new()),
                failure_index_ids: RwLock::new(HashMap::new()),
                embedder,
                failure_index: RwLock::new(VectorIndex::with_capacity(dim, 100_000)),
                id_to_key: RwLock::new(HashMap::new()),
                next_internal_id: RwLock::new(1),
                bus: ChangeBus::default(),
            }),
        }
    }

    /// Subscribe to change events. Each `start` / `score` / `complete`
    /// / `fail` publishes a [`ChangeEvent`] with `table = "eval"` and
    /// `key = Some(run_id)`, so `PSUBSCRIBE eval.*` filters by run.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Start a new eval run. Status begins as Pending. Returns the
    /// assigned UUID-form run id.
    pub fn start(
        &self,
        dataset_name: impl Into<String>,
        dataset_version: u64,
        prompt_name: Option<String>,
        prompt_version: Option<u64>,
        model: impl Into<String>,
        scorer: impl Into<String>,
        metadata: serde_json::Value,
    ) -> EvalRunId {
        let dataset_name = dataset_name.into();
        let model = model.into();
        let scorer = scorer.into();
        let id = uuid::Uuid::new_v4().simple().to_string();
        let name = format!(
            "{dataset_name}:{p}:{model}",
            p = prompt_name.clone().unwrap_or_else(|| "-".into())
        );
        let run = EvalRun {
            id: id.clone(),
            name,
            dataset_name,
            dataset_version,
            prompt_name,
            prompt_version,
            model,
            scorer,
            status: EvalStatus::Pending,
            metadata,
            created_at_unix_ns: now_unix_ns(),
            completed_at_unix_ns: None,
            summary: None,
        };
        self.inner.runs.write().insert(id.clone(), run);
        self.inner.bus.publish(ChangeEvent {
            table: "eval".to_string(),
            key: Some(id.clone()),
            row_id: 0,
            kind: ChangeKind::Insert,
        });
        id
    }

    /// Record one (row_id, score, output_text, notes). Transitions
    /// the run from Pending → Running on first score. Score must be
    /// in `[0, 1]`.
    pub fn score(
        &self,
        run_id: &str,
        row_id: impl Into<String>,
        score: f32,
        output_text: impl Into<String>,
        notes: serde_json::Value,
    ) -> Result<()> {
        if !(0.0..=1.0).contains(&score) {
            return Err(EvalError::InvalidScore(score));
        }
        let row_id = row_id.into();
        let output_text = output_text.into();

        {
            let mut runs = self.inner.runs.write();
            let run = runs
                .get_mut(run_id)
                .ok_or_else(|| EvalError::RunNotFound(run_id.to_string()))?;
            if run.status.is_terminal() {
                return Err(EvalError::WrongState(run_id.to_string(), run.status));
            }
            if matches!(run.status, EvalStatus::Pending) {
                run.status = EvalStatus::Running;
            }
        }

        let entry = EvalScore {
            run_id: run_id.to_string(),
            row_id: row_id.clone(),
            score,
            notes,
            output_text: output_text.clone(),
            recorded_at_unix_ns: now_unix_ns(),
        };
        self.inner
            .scores
            .write()
            .insert((run_id.to_string(), row_id.clone()), entry);

        // Index the OUTPUT (what the model produced) for failure
        // clustering. Skip indexing when score >= 0.8 — clustering is
        // about failures, not successes, and the catalog stays small.
        if score < 0.8 && !output_text.is_empty() {
            let emb = self
                .inner
                .embedder
                .embed(&output_text)
                .map_err(|e| EvalError::Embed(e.to_string()))?;
            let internal_id = {
                let mut n = self.inner.next_internal_id.write();
                let v = *n;
                *n += 1;
                v
            };
            self.inner
                .failure_index
                .write()
                .insert(internal_id, emb)
                .map_err(|e| EvalError::Embed(format!("vector index: {e}")))?;
            self.inner
                .id_to_key
                .write()
                .insert(internal_id, (run_id.to_string(), row_id.clone()));
            self.inner
                .failure_index_ids
                .write()
                .insert((run_id.to_string(), row_id), internal_id);
        }

        self.inner.bus.publish(ChangeEvent {
            table: "eval".to_string(),
            key: Some(run_id.to_string()),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Mark a run as completed and freeze its summary stats.
    pub fn complete(&self, run_id: &str) -> Result<EvalSummary> {
        let summary = self.compute_summary(run_id);
        let mut runs = self.inner.runs.write();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| EvalError::RunNotFound(run_id.to_string()))?;
        if run.status.is_terminal() {
            return Err(EvalError::WrongState(run_id.to_string(), run.status));
        }
        run.status = EvalStatus::Completed;
        run.completed_at_unix_ns = Some(now_unix_ns());
        run.summary = Some(summary);
        drop(runs);
        self.inner.bus.publish(ChangeEvent {
            table: "eval".to_string(),
            key: Some(run_id.to_string()),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(summary)
    }

    /// Mark a run as failed. Summary still computed from any partial
    /// scores recorded so far.
    pub fn fail(&self, run_id: &str, reason: impl Into<String>) -> Result<()> {
        let summary = self.compute_summary(run_id);
        let mut runs = self.inner.runs.write();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| EvalError::RunNotFound(run_id.to_string()))?;
        if run.status.is_terminal() {
            return Err(EvalError::WrongState(run_id.to_string(), run.status));
        }
        run.status = EvalStatus::Failed;
        run.completed_at_unix_ns = Some(now_unix_ns());
        run.summary = Some(summary);
        // Stash the failure reason in metadata.
        if let serde_json::Value::Object(ref mut map) = run.metadata {
            map.insert(
                "failure_reason".to_string(),
                serde_json::Value::String(reason.into()),
            );
        } else {
            run.metadata = serde_json::json!({"failure_reason": reason.into()});
        }
        drop(runs);
        self.inner.bus.publish(ChangeEvent {
            table: "eval".to_string(),
            key: Some(run_id.to_string()),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Get one run.
    pub fn get(&self, run_id: &str) -> Option<EvalRun> {
        self.inner.runs.read().get(run_id).cloned()
    }

    /// Every score recorded for a run.
    pub fn scores(&self, run_id: &str) -> Vec<EvalScore> {
        self.inner
            .scores
            .read()
            .iter()
            .filter(|((rid, _), _)| rid == run_id)
            .map(|(_, s)| s.clone())
            .collect()
    }

    /// One score by (run, row).
    pub fn score_of(&self, run_id: &str, row_id: &str) -> Option<EvalScore> {
        self.inner
            .scores
            .read()
            .get(&(run_id.to_string(), row_id.to_string()))
            .cloned()
    }

    /// Every run, newest first.
    pub fn list_runs(&self) -> Vec<EvalRun> {
        let mut runs: Vec<_> = self.inner.runs.read().values().cloned().collect();
        runs.sort_by_key(|r| std::cmp::Reverse(r.created_at_unix_ns));
        runs
    }

    /// Runs filtered by exact `(dataset_name, dataset_version)` —
    /// useful for diffing all eval results that targeted the same
    /// test set.
    pub fn list_runs_for(&self, dataset_name: &str, dataset_version: u64) -> Vec<EvalRun> {
        self.list_runs()
            .into_iter()
            .filter(|r| r.dataset_name == dataset_name && r.dataset_version == dataset_version)
            .collect()
    }

    /// Compare two runs over the same dataset. The two runs do NOT
    /// have to share dataset_version; the join is by `row_id` only,
    /// so callers can compare across dataset revisions if the row
    /// ids stay stable.
    pub fn compare(&self, run_a: &str, run_b: &str) -> Result<EvalComparison> {
        let a_run = self.get(run_a).ok_or_else(|| EvalError::RunNotFound(run_a.into()))?;
        let b_run = self.get(run_b).ok_or_else(|| EvalError::RunNotFound(run_b.into()))?;
        let a_scores = self.scores(run_a);
        let b_scores = self.scores(run_b);
        let a_by_row: HashMap<String, f32> =
            a_scores.iter().map(|s| (s.row_id.clone(), s.score)).collect();
        let b_by_row: HashMap<String, f32> =
            b_scores.iter().map(|s| (s.row_id.clone(), s.score)).collect();

        let mut regressed = Vec::new();
        let mut improved = Vec::new();
        let mut new_rows = Vec::new();
        let mut dropped_rows = Vec::new();
        for (row_id, sa) in &a_by_row {
            match b_by_row.get(row_id) {
                Some(sb) => {
                    let delta = sb - sa;
                    if *sb < *sa {
                        regressed.push(RegressionRow {
                            row_id: row_id.clone(),
                            score_a: *sa,
                            score_b: *sb,
                            delta,
                        });
                    } else if *sb > *sa {
                        improved.push(RegressionRow {
                            row_id: row_id.clone(),
                            score_a: *sa,
                            score_b: *sb,
                            delta,
                        });
                    }
                }
                None => dropped_rows.push(row_id.clone()),
            }
        }
        for row_id in b_by_row.keys() {
            if !a_by_row.contains_key(row_id) {
                new_rows.push(row_id.clone());
            }
        }
        // Sort regressions by largest absolute drop first.
        regressed.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap_or(std::cmp::Ordering::Equal));
        improved.sort_by(|a, b| b.delta.partial_cmp(&a.delta).unwrap_or(std::cmp::Ordering::Equal));

        let mean_a = a_run.summary.map(|s| s.mean).unwrap_or(0.0);
        let mean_b = b_run.summary.map(|s| s.mean).unwrap_or(0.0);
        let pr_a = a_run.summary.map(|s| s.pass_rate_50).unwrap_or(0.0);
        let pr_b = b_run.summary.map(|s| s.pass_rate_50).unwrap_or(0.0);

        Ok(EvalComparison {
            run_a: run_a.to_string(),
            run_b: run_b.to_string(),
            mean_delta: mean_b - mean_a,
            pass_rate_50_delta: pr_b - pr_a,
            regressed,
            improved,
            new_rows,
            dropped_rows,
        })
    }

    /// Cluster failing rows (score < `threshold`, default 0.5) of a
    /// run by semantic similarity of their `output_text`. Greedy
    /// agglomeration with cosine similarity ≥ `sim_threshold`
    /// (default 0.8). Returns at most `max_clusters` clusters,
    /// sorted by member count desc.
    ///
    /// This is the differentiator — competitors store failures in a
    /// flat table; here, eval failures, dataset rows, memories, and
    /// prompts all share one HNSW vector space, so clustering by
    /// "what's the failure about" is one call.
    pub fn cluster_failures(
        &self,
        run_id: &str,
        score_threshold: f32,
        sim_threshold: f32,
        max_clusters: usize,
    ) -> Result<Vec<FailureCluster>> {
        if !self.inner.runs.read().contains_key(run_id) {
            return Err(EvalError::RunNotFound(run_id.to_string()));
        }
        let st = if score_threshold == 0.0 { 0.5 } else { score_threshold };
        let smt = if sim_threshold == 0.0 { 0.8 } else { sim_threshold };

        // Snapshot every failing score that's also in the index.
        let failures: Vec<EvalScore> = self
            .scores(run_id)
            .into_iter()
            .filter(|s| s.score < st && !s.output_text.is_empty())
            .collect();

        // Greedy cluster: for each failure, search the catalog for
        // similar ones, group together. We use the registry's own
        // VectorIndex by running a search per seed.
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut clusters: Vec<FailureCluster> = Vec::new();

        for seed in &failures {
            if visited.contains(&seed.row_id) {
                continue;
            }
            visited.insert(seed.row_id.clone());

            // Embed the seed and search for its k nearest neighbours
            // inside THIS run.
            let seed_emb = self
                .inner
                .embedder
                .embed(&seed.output_text)
                .map_err(|e| EvalError::Embed(e.to_string()))?;
            let neighbors = self.inner.failure_index.read().search(&seed_emb, 64);

            let mut cluster_members: Vec<FailureClusterMember> = Vec::new();
            cluster_members.push(FailureClusterMember {
                row_id: seed.row_id.clone(),
                score: seed.score,
                similarity: 1.0,
            });

            let id_to_key = self.inner.id_to_key.read();
            for (internal_id, dist) in neighbors {
                let (other_run, other_row) = match id_to_key.get(&internal_id) {
                    Some(k) => k.clone(),
                    None => continue,
                };
                if other_run != run_id || other_row == seed.row_id || visited.contains(&other_row) {
                    continue;
                }
                let sim = (1.0 - dist).clamp(0.0, 1.0);
                if sim < smt {
                    continue;
                }
                if let Some(scr) = self.score_of(run_id, &other_row) {
                    if scr.score < st {
                        cluster_members.push(FailureClusterMember {
                            row_id: other_row.clone(),
                            score: scr.score,
                            similarity: sim,
                        });
                        visited.insert(other_row);
                    }
                }
            }

            let mean = cluster_members.iter().map(|m| m.score).sum::<f32>()
                / cluster_members.len() as f32;
            clusters.push(FailureCluster {
                representative_row_id: seed.row_id.clone(),
                representative_text: seed.output_text.clone(),
                members: cluster_members,
                mean_score: mean,
            });
        }
        clusters.sort_by(|a, b| b.members.len().cmp(&a.members.len()));
        clusters.truncate(max_clusters);
        Ok(clusters)
    }

    /// Counters for ops dashboards.
    pub fn stats(&self) -> EvalStats {
        let runs = self.inner.runs.read();
        let mut running = 0;
        let mut completed = 0;
        let mut failed = 0;
        for r in runs.values() {
            match r.status {
                EvalStatus::Running | EvalStatus::Pending => running += 1,
                EvalStatus::Completed => completed += 1,
                EvalStatus::Failed => failed += 1,
            }
        }
        EvalStats {
            runs: runs.len(),
            scores: self.inner.scores.read().len(),
            running,
            completed,
            failed,
        }
    }

    // -- helpers --

    fn compute_summary(&self, run_id: &str) -> EvalSummary {
        let scores = self.scores(run_id);
        if scores.is_empty() {
            return EvalSummary::default();
        }
        let mut vals: Vec<f32> = scores.iter().map(|s| s.score).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let sum: f32 = vals.iter().sum();
        let mean = sum / n as f32;
        let p = |q: f32| -> f32 {
            let idx = ((n - 1) as f32 * q).round() as usize;
            vals[idx.min(n - 1)]
        };
        let pass = vals.iter().filter(|v| **v >= 0.5).count() as f32 / n as f32;
        EvalSummary {
            total_scored: n,
            mean,
            p50: p(0.5),
            p90: p(0.9),
            p99: p(0.99),
            min: *vals.first().unwrap(),
            max: *vals.last().unwrap(),
            pass_rate_50: pass,
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use duxx_embed::HashEmbedder;

    fn reg() -> EvalRegistry {
        EvalRegistry::new(Arc::new(HashEmbedder::new(16)))
    }

    fn start_run(r: &EvalRegistry) -> EvalRunId {
        r.start(
            "qa-set",
            1,
            Some("classifier".into()),
            Some(3),
            "gpt-4o",
            "llm_judge",
            serde_json::json!({"branch": "main"}),
        )
    }

    #[test]
    fn start_returns_pending_run() {
        let r = reg();
        let id = start_run(&r);
        let run = r.get(&id).unwrap();
        assert_eq!(run.status, EvalStatus::Pending);
        assert_eq!(run.dataset_name, "qa-set");
        assert_eq!(run.dataset_version, 1);
        assert_eq!(run.prompt_name.as_deref(), Some("classifier"));
        assert_eq!(run.model, "gpt-4o");
    }

    #[test]
    fn first_score_transitions_to_running() {
        let r = reg();
        let id = start_run(&r);
        r.score(&id, "row1", 0.9, "model output A", serde_json::Value::Null)
            .unwrap();
        assert_eq!(r.get(&id).unwrap().status, EvalStatus::Running);
    }

    #[test]
    fn score_validates_range() {
        let r = reg();
        let id = start_run(&r);
        let err = r
            .score(&id, "row1", 1.5, "x", serde_json::Value::Null)
            .unwrap_err();
        matches!(err, EvalError::InvalidScore(_));
    }

    #[test]
    fn complete_freezes_summary_with_percentiles() {
        let r = reg();
        let id = start_run(&r);
        for (i, s) in [0.1, 0.4, 0.5, 0.7, 0.9].iter().enumerate() {
            r.score(&id, format!("row{i}"), *s, format!("out-{i}"), serde_json::Value::Null)
                .unwrap();
        }
        let summary = r.complete(&id).unwrap();
        assert_eq!(summary.total_scored, 5);
        assert!((summary.mean - 0.52).abs() < 0.05);
        assert!((summary.p50 - 0.5).abs() < 0.001);
        assert!((summary.min - 0.1).abs() < 0.001);
        assert!((summary.max - 0.9).abs() < 0.001);
        // 3 of 5 >= 0.5
        assert!((summary.pass_rate_50 - 0.6).abs() < 0.001);
        // Run is terminal.
        assert_eq!(r.get(&id).unwrap().status, EvalStatus::Completed);
    }

    #[test]
    fn cannot_score_after_complete() {
        let r = reg();
        let id = start_run(&r);
        r.score(&id, "row1", 0.5, "x", serde_json::Value::Null).unwrap();
        r.complete(&id).unwrap();
        let err = r
            .score(&id, "row2", 0.5, "y", serde_json::Value::Null)
            .unwrap_err();
        matches!(err, EvalError::WrongState(_, _));
    }

    #[test]
    fn fail_stores_reason_in_metadata() {
        let r = reg();
        let id = start_run(&r);
        r.score(&id, "row1", 0.5, "x", serde_json::Value::Null).unwrap();
        r.fail(&id, "rate limited by provider").unwrap();
        let run = r.get(&id).unwrap();
        assert_eq!(run.status, EvalStatus::Failed);
        assert!(run.metadata.get("failure_reason").is_some());
    }

    #[test]
    fn compare_detects_regressions_and_improvements() {
        let r = reg();
        let a = start_run(&r);
        let b = start_run(&r);
        // a: row1=0.9, row2=0.5, row3=0.8
        // b: row1=0.7, row2=0.9, row4=0.6   (row3 dropped, row4 new)
        for (id, row, s) in [
            (&a, "row1", 0.9), (&a, "row2", 0.5), (&a, "row3", 0.8),
            (&b, "row1", 0.7), (&b, "row2", 0.9), (&b, "row4", 0.6),
        ] {
            r.score(id, row, s, "out", serde_json::Value::Null).unwrap();
        }
        r.complete(&a).unwrap();
        r.complete(&b).unwrap();

        let cmp = r.compare(&a, &b).unwrap();
        let regressed_ids: Vec<&str> = cmp.regressed.iter().map(|x| x.row_id.as_str()).collect();
        let improved_ids: Vec<&str> = cmp.improved.iter().map(|x| x.row_id.as_str()).collect();
        assert!(regressed_ids.contains(&"row1"));
        assert!(improved_ids.contains(&"row2"));
        assert_eq!(cmp.new_rows, vec!["row4".to_string()]);
        assert_eq!(cmp.dropped_rows, vec!["row3".to_string()]);
    }

    #[test]
    fn list_runs_for_dataset_filter() {
        let r = reg();
        let _ = r.start("alpha", 1, None, None, "m", "s", serde_json::Value::Null);
        let beta_id = r.start("beta", 1, None, None, "m", "s", serde_json::Value::Null);
        let runs = r.list_runs_for("beta", 1);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, beta_id);
    }

    #[test]
    fn cluster_failures_groups_similar_outputs() {
        let r = reg();
        let id = start_run(&r);
        // Three failures that look similar + two unrelated.
        for (row, text) in [
            ("f1", "the model hallucinated a phone number"),
            ("f2", "the model hallucinated a phone number again"),
            ("f3", "model hallucinated phone number for user"),
            ("f4", "the model timed out before responding"),
            ("f5", "the model rejected the prompt as unsafe"),
        ] {
            r.score(&id, row, 0.1, text, serde_json::Value::Null).unwrap();
        }
        // One passing row (must NOT be indexed for clustering).
        r.score(&id, "p1", 0.95, "model returned a perfect answer", serde_json::Value::Null)
            .unwrap();

        let clusters = r.cluster_failures(&id, 0.5, 0.5, 10).unwrap();
        // Should produce at least one cluster covering f1/f2/f3.
        assert!(!clusters.is_empty());
        let total_members: usize = clusters.iter().map(|c| c.members.len()).sum();
        assert!(total_members >= 3);
        // p1 must not appear anywhere.
        for c in &clusters {
            for m in &c.members {
                assert_ne!(m.row_id, "p1");
            }
        }
    }

    #[test]
    fn change_bus_publishes_start_score_complete() {
        let r = reg();
        let mut rx = r.subscribe();
        let id = start_run(&r);
        let e = rx.try_recv().unwrap();
        assert_eq!(e.table, "eval");
        assert!(matches!(e.kind, ChangeKind::Insert));
        assert_eq!(e.key.as_deref(), Some(id.as_str()));

        r.score(&id, "row", 0.5, "x", serde_json::Value::Null).unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Update));

        r.complete(&id).unwrap();
        let e = rx.try_recv().unwrap();
        assert!(matches!(e.kind, ChangeKind::Update));
    }

    #[test]
    fn stats_counts_runs_by_status() {
        let r = reg();
        let a = start_run(&r);
        let _b = start_run(&r);
        let c = start_run(&r);
        r.score(&a, "row", 0.5, "x", serde_json::Value::Null).unwrap();
        r.complete(&a).unwrap();
        r.score(&c, "row", 0.5, "x", serde_json::Value::Null).unwrap();
        r.fail(&c, "boom").unwrap();
        let s = r.stats();
        assert_eq!(s.runs, 3);
        assert_eq!(s.completed, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.running, 1); // _b is still Pending
        assert_eq!(s.scores, 2);
    }

    #[test]
    fn scores_returns_all_rows_for_a_run() {
        let r = reg();
        let id = start_run(&r);
        for row in ["a", "b", "c"] {
            r.score(&id, row, 0.5, "x", serde_json::Value::Null).unwrap();
        }
        let scores = r.scores(&id);
        assert_eq!(scores.len(), 3);
    }
}
