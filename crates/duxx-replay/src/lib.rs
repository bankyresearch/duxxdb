//! # `duxx-replay` — deterministic agent replay (Phase 7.5)
//!
//! Agent debugging usually goes: **find a failure → reproduce it →
//! tweak something → re-run → did it fix the cluster?** The middle
//! steps are where most time disappears, because most agent stacks
//! reproduce failures by hand: copy the inputs out of whatever
//! tracing tool captured them, paste into a Python REPL, hope you
//! got the prompt / model / seed right.
//!
//! `duxx-replay` makes the loop a first-class primitive. The crate
//! does **not** execute LLM calls — caller code still drives the
//! invocations. Instead it:
//!
//! 1. **Captures** each LLM/tool invocation against a `trace_id`
//!    while the agent runs normally.
//! 2. **Plays back** any captured trace, optionally with per-step
//!    overrides (swap model / prompt / inject a fake output / skip).
//! 3. **Diffs** the original outputs against the replay outputs at
//!    every step.
//!
//! Combine with the other Phase 7 primitives for the full loop:
//!
//! ```text
//!   EVAL.CLUSTER_FAILURES run -> a cluster of similar failing rows
//!   REPLAY.START source_trace [override: SwapPrompt v8]
//!   (caller re-executes step by step, REPLAY.RECORD on each)
//!   EVAL.SCORE   new-run row score
//!   EVAL.COMPARE old-run new-run -> "did v8 fix the cluster?"
//! ```
//!
//! ## What's NOT in this crate
//!
//! - **Inference orchestration.** Caller still calls the LLM. We
//!   record what they asked and what they got. This keeps DuxxDB
//!   pure Apache-2 storage with zero LLM-SDK coupling.
//! - **Persistence.** Phase 7.x.b alongside the other Phase 7
//!   in-memory stores.

use duxx_reactive::{ChangeBus, ChangeEvent, ChangeKind};
use duxx_storage::{Backend, MemoryBackend};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::broadcast;

mod tables {
    /// `trace_id` -> JSON(ReplaySession)
    pub const SESSIONS: &str = "replay.sessions";
    /// `run_id` -> JSON(ReplayRun)
    pub const RUNS: &str = "replay.runs";
}

/// Errors surfaced by the replay registry.
#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay session for trace {0:?} not found")]
    SessionNotFound(String),
    #[error("replay run {0:?} not found")]
    RunNotFound(String),
    #[error("replay run {0:?} is not in {1:?} state")]
    WrongState(String, ReplayStatus),
    #[error("invocation idx {0} is out of range for this session ({1} total)")]
    OutOfRange(usize, usize),
    #[error("captured invocation has no recorded output for cached mode")]
    NoCachedOutput,
    #[error("storage error: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, ReplayError>;

// ---------------------------------------------------------------- Types

/// Kind of invocation captured during an agent run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InvocationKind {
    /// LLM completion / chat call.
    LlmCall,
    /// External tool call.
    ToolCall { tool: String },
    /// Anything else worth capturing.
    Other { label: String },
}

impl InvocationKind {
    pub fn label(&self) -> String {
        match self {
            InvocationKind::LlmCall => "llm.call".into(),
            InvocationKind::ToolCall { tool } => format!("tool.{tool}"),
            InvocationKind::Other { label } => label.clone(),
        }
    }
}

/// One captured invocation within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayInvocation {
    /// Ordinal within the session (assigned in capture order).
    pub idx: usize,
    /// Span this invocation came from. Links back to `duxx-trace`.
    /// May be empty for callers that don't use traces.
    #[serde(default)]
    pub span_id: String,
    pub kind: InvocationKind,
    /// For LLM calls: which model. Optional for tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional reference into the prompt registry (Phase 7.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_version: Option<u64>,
    /// What the caller asked. Messages, args, etc.
    pub input: serde_json::Value,
    /// What was produced. None if capture happened before completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    /// Free-form: tokens, latency_ms, cost_usd, etc.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub recorded_at_unix_ns: u128,
}

/// A captured agent run — the full ordered sequence of invocations
/// tied to one `trace_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySession {
    pub trace_id: String,
    pub invocations: Vec<ReplayInvocation>,
    /// SHA-256 over the input payloads in order. Same fingerprint
    /// across two sessions means they sent the same inputs to the
    /// same models in the same order.
    pub fingerprint: String,
    pub captured_at_unix_ns: u128,
}

/// One per-invocation override applied during replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayOverride {
    pub at_idx: usize,
    pub kind: OverrideKind,
}

/// What the override does. Serialized with a `kind` tag so callers
/// can express overrides over the wire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OverrideKind {
    /// Swap the model (`gpt-4o` → `claude-4.5-sonnet`, etc.).
    SwapModel { model: String },
    /// Swap the prompt reference. Useful when comparing two prompt
    /// versions over the same dataset.
    SwapPrompt { prompt_name: String, prompt_version: u64 },
    /// Skip execution entirely; use this canned output instead.
    /// Useful for deterministic snapshots in tests.
    InjectOutput { output: serde_json::Value },
    /// Override the LLM `temperature` field in `input` (mutates the
    /// input JSON before returning it to the caller).
    SetTemperature { temperature: f32 },
    /// Skip this invocation entirely (no input, no output).
    Skip,
}

/// Replay execution mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    /// Return captured outputs verbatim. Caller never re-executes.
    /// Useful for UI playback / debugging walkthroughs.
    Cached,
    /// Caller re-executes each invocation. Each `RECORD` stores the
    /// new output; `DIFF` can compare against the original.
    Live,
    /// Same as `Live` but `STEP` returns one invocation at a time
    /// and the run blocks (state-wise) until that step is recorded.
    Stepped,
}

/// Lifecycle of a replay run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReplayStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl ReplayStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

/// One replay attempt. Cheaply clonable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRun {
    pub id: String,
    pub source_trace_id: String,
    pub mode: ReplayMode,
    pub overrides: Vec<ReplayOverride>,
    pub status: ReplayStatus,
    pub current_idx: usize,
    /// New outputs recorded during this run, keyed by invocation idx.
    pub outputs: HashMap<usize, serde_json::Value>,
    /// Tag stamped on the trace this replay produced (if the caller
    /// chose to emit one). Lets `TRACE.SEARCH` pull every replay of
    /// a given source.
    #[serde(default)]
    pub replay_trace_id: Option<String>,
    pub created_at_unix_ns: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_unix_ns: Option<u128>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// One row of a replay diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvocationDiff {
    pub idx: usize,
    pub kind: InvocationKind,
    pub original_output: Option<serde_json::Value>,
    pub replay_output: Option<serde_json::Value>,
    /// True iff the two outputs are not JSON-equal.
    pub differs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayDiff {
    pub source_trace_id: String,
    pub replay_run_id: String,
    pub invocation_diffs: Vec<InvocationDiff>,
    /// Count of invocations whose output differs (cheap summary).
    pub differing_count: usize,
}

/// Counters returned by [`ReplayRegistry::stats`].
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ReplayStats {
    pub sessions: usize,
    pub invocations: usize,
    pub runs: usize,
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

/// Tiny SHA-256 substitute — we don't need cryptographic strength,
/// just a stable hash for fingerprinting. Wyhash-like FNV-1a over
/// a deterministic serialization, hex-encoded.
fn fingerprint(invocations: &[ReplayInvocation]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for inv in invocations {
        let payload = serde_json::to_string(&inv.input).unwrap_or_default();
        for b in payload.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // Mix kind + model so calls with the same input but different
        // model/tool produce different fingerprints.
        for b in inv.kind.label().as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        if let Some(m) = &inv.model {
            for b in m.as_bytes() {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{h:016x}")
}

// ---------------------------------------------------------------- Registry

#[derive(Clone)]
pub struct ReplayRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    /// trace_id -> session
    sessions: RwLock<HashMap<String, ReplaySession>>,
    /// run_id -> run
    runs: RwLock<HashMap<String, ReplayRun>>,
    bus: ChangeBus,
    backend: Arc<dyn Backend>,
}

impl std::fmt::Debug for ReplayRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("ReplayRegistry")
            .field("sessions", &s.sessions)
            .field("invocations", &s.invocations)
            .field("runs", &s.runs)
            .finish()
    }
}

impl Default for ReplayRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayRegistry {
    /// Build a non-persistent registry. Equivalent to
    /// ``open(MemoryBackend::new())``.
    pub fn new() -> Self {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        Self::open(backend).expect("MemoryBackend cannot fail open")
    }

    /// Build a registry backed by the given persistence layer. On
    /// open, scans the ``replay.sessions`` and ``replay.runs`` tables
    /// to rebuild in-memory state.
    pub fn open(backend: Arc<dyn Backend>) -> Result<Self> {
        let me = Self {
            inner: Arc::new(Inner {
                sessions: RwLock::new(HashMap::new()),
                runs: RwLock::new(HashMap::new()),
                bus: ChangeBus::default(),
                backend,
            }),
        };
        me.rehydrate()?;
        Ok(me)
    }

    fn rehydrate(&self) -> Result<()> {
        let sessions = self
            .inner
            .backend
            .scan(tables::SESSIONS)
            .map_err(|e| ReplayError::Storage(format!("backend scan sessions: {e}")))?;
        for (_k, value_bytes) in sessions {
            let s: ReplaySession = serde_json::from_slice(&value_bytes)
                .map_err(|e| ReplayError::Storage(format!("session decode: {e}")))?;
            self.inner.sessions.write().insert(s.trace_id.clone(), s);
        }
        let runs = self
            .inner
            .backend
            .scan(tables::RUNS)
            .map_err(|e| ReplayError::Storage(format!("backend scan runs: {e}")))?;
        for (_k, value_bytes) in runs {
            let r: ReplayRun = serde_json::from_slice(&value_bytes)
                .map_err(|e| ReplayError::Storage(format!("run decode: {e}")))?;
            self.inner.runs.write().insert(r.id.clone(), r);
        }
        Ok(())
    }

    fn persist_session(&self, session: &ReplaySession) {
        if let Ok(bytes) = serde_json::to_vec(session) {
            if let Err(e) = self
                .inner
                .backend
                .put(tables::SESSIONS, session.trace_id.as_bytes(), &bytes)
            {
                tracing::warn!(error = %e, "backend persist replay session failed");
            }
        }
    }

    fn persist_run(&self, run: &ReplayRun) {
        if let Ok(bytes) = serde_json::to_vec(run) {
            if let Err(e) = self.inner.backend.put(tables::RUNS, run.id.as_bytes(), &bytes) {
                tracing::warn!(error = %e, "backend persist replay run failed");
            }
        }
    }

    /// Subscribe to change events. Each `capture` / `start` / `step` /
    /// `record_output` / `complete` / `fail` publishes a [`ChangeEvent`]
    /// with `table = "replay"`. The `key` is the trace_id for capture
    /// events and the run_id for run-lifecycle events.
    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.inner.bus.subscribe()
    }

    /// Capture one invocation against a `trace_id`. Append-only; idx
    /// is assigned in capture order. Returns the assigned idx.
    pub fn capture(
        &self,
        trace_id: impl Into<String>,
        mut invocation: ReplayInvocation,
    ) -> usize {
        let trace_id = trace_id.into();
        let mut sessions = self.inner.sessions.write();
        let session = sessions.entry(trace_id.clone()).or_insert_with(|| ReplaySession {
            trace_id: trace_id.clone(),
            invocations: Vec::new(),
            fingerprint: String::new(),
            captured_at_unix_ns: now_unix_ns(),
        });
        let idx = session.invocations.len();
        invocation.idx = idx;
        if invocation.recorded_at_unix_ns == 0 {
            invocation.recorded_at_unix_ns = now_unix_ns();
        }
        session.invocations.push(invocation);
        session.fingerprint = fingerprint(&session.invocations);
        let snapshot = session.clone();
        drop(sessions);
        self.persist_session(&snapshot);
        self.inner.bus.publish(ChangeEvent {
            table: "replay".to_string(),
            key: Some(trace_id),
            row_id: idx as u64,
            kind: ChangeKind::Insert,
        });
        idx
    }

    /// Get a captured session.
    pub fn get_session(&self, trace_id: &str) -> Option<ReplaySession> {
        self.inner.sessions.read().get(trace_id).cloned()
    }

    /// Every session, newest first.
    pub fn list_sessions(&self) -> Vec<ReplaySession> {
        let mut v: Vec<_> = self.inner.sessions.read().values().cloned().collect();
        v.sort_by_key(|s| std::cmp::Reverse(s.captured_at_unix_ns));
        v
    }

    /// Start a new replay run. Returns the assigned UUID-form run id.
    /// In `Cached` mode the run starts pre-populated with the session's
    /// recorded outputs (so `RECORD` is optional).
    pub fn start(
        &self,
        source_trace_id: impl Into<String>,
        mode: ReplayMode,
        overrides: Vec<ReplayOverride>,
        metadata: serde_json::Value,
    ) -> Result<String> {
        let source_trace_id = source_trace_id.into();
        if !self.inner.sessions.read().contains_key(&source_trace_id) {
            return Err(ReplayError::SessionNotFound(source_trace_id));
        }
        let id = uuid::Uuid::new_v4().simple().to_string();
        let mut outputs: HashMap<usize, serde_json::Value> = HashMap::new();
        if matches!(mode, ReplayMode::Cached) {
            if let Some(session) = self.inner.sessions.read().get(&source_trace_id) {
                for inv in &session.invocations {
                    if let Some(out) = &inv.output {
                        outputs.insert(inv.idx, out.clone());
                    }
                }
            }
        }
        let run = ReplayRun {
            id: id.clone(),
            source_trace_id: source_trace_id.clone(),
            mode,
            overrides,
            status: ReplayStatus::Pending,
            current_idx: 0,
            outputs,
            replay_trace_id: None,
            created_at_unix_ns: now_unix_ns(),
            completed_at_unix_ns: None,
            metadata,
        };
        self.persist_run(&run);
        self.inner.runs.write().insert(id.clone(), run);
        self.inner.bus.publish(ChangeEvent {
            table: "replay".to_string(),
            key: Some(id.clone()),
            row_id: 0,
            kind: ChangeKind::Insert,
        });
        Ok(id)
    }

    /// Return the next invocation to execute for a Live / Stepped
    /// replay, with any matching overrides applied to the input.
    /// Increments `current_idx`. Returns `None` when the session is
    /// exhausted (caller should call `complete`).
    ///
    /// Behavior on the returned invocation:
    /// - `SwapModel` rewrites `model`.
    /// - `SwapPrompt` rewrites `prompt_name` + `prompt_version`.
    /// - `SetTemperature` mutates `input["temperature"]` if `input`
    ///   is a JSON object.
    /// - `Skip` returns the original invocation but flags the caller
    ///   via `metadata["replay.skip"] = true` — caller should not
    ///   execute, just call `record_output` with whatever placeholder.
    /// - `InjectOutput` returns the invocation with `output` already
    ///   filled in; the caller's job is just to call `record_output`
    ///   echoing that output back, OR DuxxDB can auto-record (we do
    ///   the latter for ergonomics).
    pub fn step(&self, run_id: &str) -> Result<Option<ReplayInvocation>> {
        let mut runs = self.inner.runs.write();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| ReplayError::RunNotFound(run_id.into()))?;
        if run.status.is_terminal() {
            return Err(ReplayError::WrongState(run_id.into(), run.status));
        }
        let session = match self.inner.sessions.read().get(&run.source_trace_id).cloned() {
            Some(s) => s,
            None => return Err(ReplayError::SessionNotFound(run.source_trace_id.clone())),
        };
        // Promote to Running on first step.
        if matches!(run.status, ReplayStatus::Pending) {
            run.status = ReplayStatus::Running;
        }
        loop {
            if run.current_idx >= session.invocations.len() {
                let snapshot = run.clone();
                drop(runs);
                self.persist_run(&snapshot);
                return Ok(None);
            }
            let idx = run.current_idx;
            let original = session.invocations[idx].clone();
            // Find an override that targets this idx.
            let ov = run.overrides.iter().find(|o| o.at_idx == idx).cloned();
            run.current_idx += 1;
            let inv = match ov {
                Some(ReplayOverride {
                    kind: OverrideKind::Skip,
                    ..
                }) => {
                    // Auto-record an empty output so DIFF marks it.
                    run.outputs.insert(idx, serde_json::Value::Null);
                    continue;
                }
                Some(ReplayOverride {
                    kind: OverrideKind::InjectOutput { output },
                    ..
                }) => {
                    run.outputs.insert(idx, output.clone());
                    let mut inv = original.clone();
                    inv.output = Some(output);
                    inv
                }
                Some(ReplayOverride {
                    kind: OverrideKind::SwapModel { model },
                    ..
                }) => {
                    let mut inv = original.clone();
                    inv.model = Some(model);
                    inv
                }
                Some(ReplayOverride {
                    kind:
                        OverrideKind::SwapPrompt {
                            prompt_name,
                            prompt_version,
                        },
                    ..
                }) => {
                    let mut inv = original.clone();
                    inv.prompt_name = Some(prompt_name);
                    inv.prompt_version = Some(prompt_version);
                    inv
                }
                Some(ReplayOverride {
                    kind: OverrideKind::SetTemperature { temperature },
                    ..
                }) => {
                    let mut inv = original.clone();
                    if let serde_json::Value::Object(ref mut map) = inv.input {
                        map.insert(
                            "temperature".to_string(),
                            serde_json::json!(temperature),
                        );
                    }
                    inv
                }
                None => match run.mode {
                    ReplayMode::Cached => {
                        // Pre-populate output from session if available.
                        original.clone()
                    }
                    _ => original.clone(),
                },
            };
            let trace_id = run.source_trace_id.clone();
            let run_id_owned = run_id.to_string();
            let snapshot = run.clone();
            drop(runs);
            self.persist_run(&snapshot);
            self.inner.bus.publish(ChangeEvent {
                table: "replay".to_string(),
                key: Some(run_id_owned),
                row_id: idx as u64,
                kind: ChangeKind::Update,
            });
            // Re-borrow not strictly needed; return as-is.
            let _ = trace_id;
            return Ok(Some(inv));
        }
    }

    /// Caller posts the output produced for one invocation. Idempotent —
    /// repeat calls overwrite.
    pub fn record_output(
        &self,
        run_id: &str,
        invocation_idx: usize,
        output: serde_json::Value,
    ) -> Result<()> {
        let mut runs = self.inner.runs.write();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| ReplayError::RunNotFound(run_id.into()))?;
        if run.status.is_terminal() {
            return Err(ReplayError::WrongState(run_id.into(), run.status));
        }
        let session = self.inner.sessions.read().get(&run.source_trace_id).cloned();
        if let Some(s) = session {
            if invocation_idx >= s.invocations.len() {
                return Err(ReplayError::OutOfRange(invocation_idx, s.invocations.len()));
            }
        }
        run.outputs.insert(invocation_idx, output);
        if matches!(run.status, ReplayStatus::Pending) {
            run.status = ReplayStatus::Running;
        }
        let snapshot = run.clone();
        drop(runs);
        self.persist_run(&snapshot);
        self.inner.bus.publish(ChangeEvent {
            table: "replay".to_string(),
            key: Some(run_id.into()),
            row_id: invocation_idx as u64,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Attach a `replay_trace_id` — the trace produced by THIS replay
    /// run. Lets TRACE.* queries pull every replay of a source.
    pub fn set_replay_trace_id(
        &self,
        run_id: &str,
        replay_trace_id: impl Into<String>,
    ) -> Result<()> {
        let snapshot;
        {
            let mut runs = self.inner.runs.write();
            let run = runs
                .get_mut(run_id)
                .ok_or_else(|| ReplayError::RunNotFound(run_id.into()))?;
            run.replay_trace_id = Some(replay_trace_id.into());
            snapshot = run.clone();
        }
        self.persist_run(&snapshot);
        Ok(())
    }

    /// Finalize a replay run.
    pub fn complete(&self, run_id: &str) -> Result<()> {
        let snapshot;
        {
            let mut runs = self.inner.runs.write();
            let run = runs
                .get_mut(run_id)
                .ok_or_else(|| ReplayError::RunNotFound(run_id.into()))?;
            if run.status.is_terminal() {
                return Err(ReplayError::WrongState(run_id.into(), run.status));
            }
            run.status = ReplayStatus::Completed;
            run.completed_at_unix_ns = Some(now_unix_ns());
            snapshot = run.clone();
        }
        self.persist_run(&snapshot);
        self.inner.bus.publish(ChangeEvent {
            table: "replay".to_string(),
            key: Some(run_id.into()),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Mark a run as failed.
    pub fn fail(&self, run_id: &str, reason: impl Into<String>) -> Result<()> {
        let snapshot;
        {
            let mut runs = self.inner.runs.write();
            let run = runs
                .get_mut(run_id)
                .ok_or_else(|| ReplayError::RunNotFound(run_id.into()))?;
            if run.status.is_terminal() {
                return Err(ReplayError::WrongState(run_id.into(), run.status));
            }
            run.status = ReplayStatus::Failed;
            run.completed_at_unix_ns = Some(now_unix_ns());
            if let serde_json::Value::Object(ref mut map) = run.metadata {
                map.insert(
                    "failure_reason".to_string(),
                    serde_json::Value::String(reason.into()),
                );
            } else {
                run.metadata = serde_json::json!({"failure_reason": reason.into()});
            }
            snapshot = run.clone();
        }
        self.persist_run(&snapshot);
        self.inner.bus.publish(ChangeEvent {
            table: "replay".to_string(),
            key: Some(run_id.into()),
            row_id: 0,
            kind: ChangeKind::Update,
        });
        Ok(())
    }

    /// Get a replay run.
    pub fn get_run(&self, run_id: &str) -> Option<ReplayRun> {
        self.inner.runs.read().get(run_id).cloned()
    }

    /// Every replay run, newest first.
    pub fn list_runs(&self) -> Vec<ReplayRun> {
        let mut v: Vec<_> = self.inner.runs.read().values().cloned().collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.created_at_unix_ns));
        v
    }

    /// Runs filtered by their source trace.
    pub fn list_runs_for(&self, source_trace_id: &str) -> Vec<ReplayRun> {
        self.list_runs()
            .into_iter()
            .filter(|r| r.source_trace_id == source_trace_id)
            .collect()
    }

    /// Compare the original session against a replay run's recorded
    /// outputs, per invocation.
    pub fn diff(&self, source_trace_id: &str, replay_run_id: &str) -> Result<ReplayDiff> {
        let session = self
            .get_session(source_trace_id)
            .ok_or_else(|| ReplayError::SessionNotFound(source_trace_id.into()))?;
        let run = self
            .get_run(replay_run_id)
            .ok_or_else(|| ReplayError::RunNotFound(replay_run_id.into()))?;
        let mut diffs = Vec::with_capacity(session.invocations.len());
        let mut differing = 0;
        for inv in &session.invocations {
            let original = inv.output.clone();
            let replay = run.outputs.get(&inv.idx).cloned();
            let differs = match (&original, &replay) {
                (Some(a), Some(b)) => a != b,
                (None, None) => false,
                _ => true,
            };
            if differs {
                differing += 1;
            }
            diffs.push(InvocationDiff {
                idx: inv.idx,
                kind: inv.kind.clone(),
                original_output: original,
                replay_output: replay,
                differs,
            });
        }
        Ok(ReplayDiff {
            source_trace_id: source_trace_id.into(),
            replay_run_id: replay_run_id.into(),
            invocation_diffs: diffs,
            differing_count: differing,
        })
    }

    /// Counters for ops dashboards.
    pub fn stats(&self) -> ReplayStats {
        let sessions = self.inner.sessions.read();
        let invocations: usize = sessions.values().map(|s| s.invocations.len()).sum();
        let runs = self.inner.runs.read();
        let mut running = 0;
        let mut completed = 0;
        let mut failed = 0;
        for r in runs.values() {
            match r.status {
                ReplayStatus::Running | ReplayStatus::Pending => running += 1,
                ReplayStatus::Completed => completed += 1,
                ReplayStatus::Failed => failed += 1,
            }
        }
        ReplayStats {
            sessions: sessions.len(),
            invocations,
            runs: runs.len(),
            running,
            completed,
            failed,
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn inv(idx: usize, input: serde_json::Value, output: Option<serde_json::Value>) -> ReplayInvocation {
        ReplayInvocation {
            idx,
            span_id: format!("span-{idx}"),
            kind: InvocationKind::LlmCall,
            model: Some("gpt-4o".into()),
            prompt_name: None,
            prompt_version: None,
            input,
            output,
            metadata: serde_json::Value::Null,
            recorded_at_unix_ns: 0,
        }
    }

    fn captured_session(r: &ReplayRegistry) -> String {
        let trace_id = "trace-abc".to_string();
        r.capture(&trace_id, inv(0, serde_json::json!({"prompt": "hi"}), Some(serde_json::json!("hello world"))));
        r.capture(&trace_id, inv(0, serde_json::json!({"prompt": "what's 2+2?"}), Some(serde_json::json!("4"))));
        r.capture(&trace_id, inv(0, serde_json::json!({"prompt": "bye"}), Some(serde_json::json!("goodbye"))));
        trace_id
    }

    #[test]
    fn capture_appends_and_fingerprints() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let s = r.get_session(&id).unwrap();
        assert_eq!(s.invocations.len(), 3);
        assert_eq!(s.invocations[0].idx, 0);
        assert_eq!(s.invocations[2].idx, 2);
        assert!(!s.fingerprint.is_empty());
    }

    #[test]
    fn fingerprints_differ_when_inputs_differ() {
        let r = ReplayRegistry::new();
        r.capture("a", inv(0, serde_json::json!({"x": 1}), None));
        r.capture("b", inv(0, serde_json::json!({"x": 2}), None));
        let fa = r.get_session("a").unwrap().fingerprint;
        let fb = r.get_session("b").unwrap().fingerprint;
        assert_ne!(fa, fb);
    }

    #[test]
    fn start_pre_fills_cached_outputs() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Cached, vec![], serde_json::Value::Null)
            .unwrap();
        let run = r.get_run(&run_id).unwrap();
        // All 3 captured outputs should be pre-populated.
        assert_eq!(run.outputs.len(), 3);
        assert_eq!(run.outputs.get(&0).unwrap(), &serde_json::json!("hello world"));
    }

    #[test]
    fn start_on_missing_session_errors() {
        let r = ReplayRegistry::new();
        let err = r
            .start("no-such", ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap_err();
        matches!(err, ReplayError::SessionNotFound(_));
    }

    #[test]
    fn step_returns_invocations_in_order_then_none() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        let a = r.step(&run_id).unwrap().unwrap();
        let b = r.step(&run_id).unwrap().unwrap();
        let c = r.step(&run_id).unwrap().unwrap();
        let none = r.step(&run_id).unwrap();
        assert_eq!(a.idx, 0);
        assert_eq!(b.idx, 1);
        assert_eq!(c.idx, 2);
        assert!(none.is_none());
        // Run status promoted to Running on first step.
        assert_eq!(r.get_run(&run_id).unwrap().status, ReplayStatus::Running);
    }

    #[test]
    fn step_applies_swap_model_override() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(
                &id,
                ReplayMode::Live,
                vec![ReplayOverride {
                    at_idx: 1,
                    kind: OverrideKind::SwapModel {
                        model: "claude-4.5-sonnet".into(),
                    },
                }],
                serde_json::Value::Null,
            )
            .unwrap();
        r.step(&run_id).unwrap(); // idx 0 (no override)
        let inv = r.step(&run_id).unwrap().unwrap(); // idx 1 (override)
        assert_eq!(inv.model.as_deref(), Some("claude-4.5-sonnet"));
    }

    #[test]
    fn step_applies_swap_prompt_override() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(
                &id,
                ReplayMode::Live,
                vec![ReplayOverride {
                    at_idx: 0,
                    kind: OverrideKind::SwapPrompt {
                        prompt_name: "support".into(),
                        prompt_version: 7,
                    },
                }],
                serde_json::Value::Null,
            )
            .unwrap();
        let inv = r.step(&run_id).unwrap().unwrap();
        assert_eq!(inv.prompt_name.as_deref(), Some("support"));
        assert_eq!(inv.prompt_version, Some(7));
    }

    #[test]
    fn step_inject_output_skips_caller_execution() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let canned = serde_json::json!({"reply": "canned"});
        let run_id = r
            .start(
                &id,
                ReplayMode::Live,
                vec![ReplayOverride {
                    at_idx: 1,
                    kind: OverrideKind::InjectOutput {
                        output: canned.clone(),
                    },
                }],
                serde_json::Value::Null,
            )
            .unwrap();
        r.step(&run_id).unwrap(); // 0
        let inv = r.step(&run_id).unwrap().unwrap(); // 1
        assert_eq!(inv.output, Some(canned.clone()));
        // The run's outputs map already has it recorded.
        let run = r.get_run(&run_id).unwrap();
        assert_eq!(run.outputs.get(&1), Some(&canned));
    }

    #[test]
    fn step_skip_advances_without_returning() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(
                &id,
                ReplayMode::Live,
                vec![ReplayOverride {
                    at_idx: 1,
                    kind: OverrideKind::Skip,
                }],
                serde_json::Value::Null,
            )
            .unwrap();
        let a = r.step(&run_id).unwrap().unwrap();
        assert_eq!(a.idx, 0);
        let c = r.step(&run_id).unwrap().unwrap();
        // Idx 1 was skipped, so the NEXT returned is idx 2.
        assert_eq!(c.idx, 2);
        let run = r.get_run(&run_id).unwrap();
        // current_idx advanced past the skipped index.
        assert_eq!(run.current_idx, 3);
        assert_eq!(run.outputs.get(&1), Some(&serde_json::Value::Null));
    }

    #[test]
    fn step_set_temperature_mutates_input() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(
                &id,
                ReplayMode::Live,
                vec![ReplayOverride {
                    at_idx: 0,
                    kind: OverrideKind::SetTemperature { temperature: 0.0 },
                }],
                serde_json::Value::Null,
            )
            .unwrap();
        let inv = r.step(&run_id).unwrap().unwrap();
        assert_eq!(inv.input["temperature"], serde_json::json!(0.0));
        assert_eq!(inv.input["prompt"], serde_json::json!("hi"));
    }

    #[test]
    fn record_output_stores_per_idx() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        r.record_output(&run_id, 0, serde_json::json!("new hi"))
            .unwrap();
        r.record_output(&run_id, 1, serde_json::json!("new 4"))
            .unwrap();
        let run = r.get_run(&run_id).unwrap();
        assert_eq!(run.outputs.len(), 2);
        // Re-record overwrites.
        r.record_output(&run_id, 0, serde_json::json!("again"))
            .unwrap();
        assert_eq!(
            r.get_run(&run_id).unwrap().outputs.get(&0),
            Some(&serde_json::json!("again"))
        );
    }

    #[test]
    fn record_output_rejects_out_of_range() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        let err = r
            .record_output(&run_id, 99, serde_json::json!("x"))
            .unwrap_err();
        matches!(err, ReplayError::OutOfRange(_, _));
    }

    #[test]
    fn complete_freezes_run() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        r.record_output(&run_id, 0, serde_json::json!("x")).unwrap();
        r.complete(&run_id).unwrap();
        assert_eq!(r.get_run(&run_id).unwrap().status, ReplayStatus::Completed);
        let err = r.record_output(&run_id, 1, serde_json::json!("y")).unwrap_err();
        matches!(err, ReplayError::WrongState(_, _));
    }

    #[test]
    fn diff_marks_differing_outputs() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        // Idx 0: same as original. Idx 1: different. Idx 2: missing.
        r.record_output(&run_id, 0, serde_json::json!("hello world"))
            .unwrap();
        r.record_output(&run_id, 1, serde_json::json!("FIVE"))
            .unwrap();
        let diff = r.diff(&id, &run_id).unwrap();
        assert_eq!(diff.invocation_diffs.len(), 3);
        assert!(!diff.invocation_diffs[0].differs);
        assert!(diff.invocation_diffs[1].differs);
        assert!(diff.invocation_diffs[2].differs);
        assert_eq!(diff.differing_count, 2);
    }

    #[test]
    fn change_bus_emits_capture_start_step_complete() {
        let r = ReplayRegistry::new();
        let mut rx = r.subscribe();
        let id = captured_session(&r);
        // 3 capture events.
        for _ in 0..3 {
            let e = rx.try_recv().unwrap();
            assert_eq!(e.table, "replay");
            assert!(matches!(e.kind, ChangeKind::Insert));
            assert_eq!(e.key.as_deref(), Some(id.as_str()));
        }
        let run_id = r
            .start(&id, ReplayMode::Live, vec![], serde_json::Value::Null)
            .unwrap();
        let e = rx.try_recv().unwrap(); // start
        assert!(matches!(e.kind, ChangeKind::Insert));
        assert_eq!(e.key.as_deref(), Some(run_id.as_str()));
        let _ = r.step(&run_id).unwrap();
        let e = rx.try_recv().unwrap(); // step
        assert!(matches!(e.kind, ChangeKind::Update));
        r.complete(&run_id).unwrap();
        let e = rx.try_recv().unwrap(); // complete
        assert!(matches!(e.kind, ChangeKind::Update));
    }

    #[test]
    fn list_runs_for_source_filters() {
        let r = ReplayRegistry::new();
        let id_a = captured_session(&r);
        // A second session.
        r.capture("trace-b", inv(0, serde_json::json!({}), Some(serde_json::Value::Null)));
        let run1 = r.start(&id_a, ReplayMode::Live, vec![], serde_json::Value::Null).unwrap();
        let _run2 = r.start("trace-b", ReplayMode::Live, vec![], serde_json::Value::Null).unwrap();
        let only_a = r.list_runs_for(&id_a);
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].id, run1);
    }

    #[test]
    fn stats_counts_sessions_invocations_runs() {
        let r = ReplayRegistry::new();
        let id = captured_session(&r);
        let run_id = r.start(&id, ReplayMode::Live, vec![], serde_json::Value::Null).unwrap();
        r.record_output(&run_id, 0, serde_json::json!("x")).unwrap();
        r.complete(&run_id).unwrap();
        let s = r.stats();
        assert_eq!(s.sessions, 1);
        assert_eq!(s.invocations, 3);
        assert_eq!(s.runs, 1);
        assert_eq!(s.completed, 1);
    }

    // ---------------------------------------------------------------- persistence

    #[test]
    fn open_rehydrates_sessions_and_runs() {
        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        let trace_id;
        let run_id;
        {
            let r = ReplayRegistry::open(backend.clone()).unwrap();
            trace_id = captured_session(&r);
            run_id = r
                .start(&trace_id, ReplayMode::Live, vec![], serde_json::Value::Null)
                .unwrap();
            r.record_output(&run_id, 0, serde_json::json!("y")).unwrap();
            r.set_replay_trace_id(&run_id, "trace-out-123").unwrap();
            r.complete(&run_id).unwrap();
        }
        let r = ReplayRegistry::open(backend.clone()).unwrap();
        let session = r.get_session(&trace_id).unwrap();
        assert_eq!(session.invocations.len(), 3);
        let run = r.get_run(&run_id).unwrap();
        assert!(matches!(run.status, ReplayStatus::Completed));
        assert_eq!(run.replay_trace_id.as_deref(), Some("trace-out-123"));
        assert_eq!(run.outputs.get(&0), Some(&serde_json::json!("y")));
    }
}
