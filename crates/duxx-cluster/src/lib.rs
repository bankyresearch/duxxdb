//! # duxx-cluster — replication scaffold (Phase D)
//!
//! **This is a scaffold, not a production replication system.** It defines the
//! core primitive every leader/follower design is built on — a **sequenced
//! change feed** — with an in-memory implementation and convergence tests, so
//! the interfaces and semantics are pinned down. Real production replication
//! (network transport, failover/leader election, snapshots, backpressure,
//! cross-AZ, read-quorum) builds on this; it is explicitly out of scope here
//! and would be weeks of distributed-systems work.
//!
//! ## The primitive
//!
//! A leader assigns every workspace mutation a **monotonic sequence number**
//! and appends it to a [`ChangeLog`]. A [`Follower`] remembers the last
//! sequence it applied and, on `catch_up`, pulls everything newer and applies
//! it in order. This is the spine of:
//!
//! * **Leader/follower replication** — followers tail the leader's log.
//! * **Read replicas** — a replica is a caught-up follower serving reads.
//! * **Point-in-time / resume** — replay the log from any sequence.
//!
//! ```
//! use duxx_cluster::{MemoryLog, ChangeLog, Follower};
//!
//! let leader = MemoryLog::new();
//! leader.append("org/proj/prod", b"REMEMBER u1 hello".to_vec());
//! leader.append("org/proj/prod", b"REMEMBER u2 world".to_vec());
//!
//! let mut replica = Follower::new();
//! let mut applied = Vec::new();
//! replica.catch_up(&leader, |c| applied.push(c.seq));
//! assert_eq!(applied, vec![1, 2]);
//! assert_eq!(replica.applied_seq(), leader.latest_seq());
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A node's role in a replica set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationRole {
    /// Accepts writes, assigns sequence numbers, owns the authoritative log.
    Leader,
    /// Read-only; tails the leader and applies changes in order.
    Follower,
}

/// One replicated mutation: a monotonic `seq`, the workspace `namespace` it
/// belongs to, and an opaque `op` payload (a serialized command — the engine
/// decides the encoding; the change feed is payload-agnostic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub seq: u64,
    pub namespace: String,
    pub op: Vec<u8>,
}

/// An append-only, sequence-ordered feed of [`Change`]s. The leader appends;
/// followers read `since`. Implementations MUST assign strictly increasing
/// sequence numbers starting at 1 and return them in order from `since`.
pub trait ChangeLog: Send + Sync {
    /// Append a mutation, returning its assigned sequence number.
    fn append(&self, namespace: &str, op: Vec<u8>) -> u64;
    /// Every change with `seq > after_seq`, in ascending sequence order.
    fn since(&self, after_seq: u64) -> Vec<Change>;
    /// The highest sequence number assigned so far (0 if empty).
    fn latest_seq(&self) -> u64;
    /// The lowest sequence still present (0 if empty). Rises after WAL
    /// compaction; a follower whose cursor is below this has fallen off the log
    /// and must bootstrap from a snapshot rather than tailing.
    fn earliest_seq(&self) -> u64 {
        self.since(0).first().map(|c| c.seq).unwrap_or(0)
    }
}

/// In-memory [`ChangeLog`]. The production leader swaps this for a durable,
/// network-served log (e.g. WAL + a replication RPC); the semantics are
/// identical.
#[derive(Default)]
pub struct MemoryLog {
    inner: Mutex<Vec<Change>>,
}

impl MemoryLog {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ChangeLog for MemoryLog {
    fn append(&self, namespace: &str, op: Vec<u8>) -> u64 {
        let mut log = self.inner.lock().expect("changelog poisoned");
        let seq = log.len() as u64 + 1;
        log.push(Change {
            seq,
            namespace: namespace.to_string(),
            op,
        });
        seq
    }

    fn since(&self, after_seq: u64) -> Vec<Change> {
        let log = self.inner.lock().expect("changelog poisoned");
        log.iter().filter(|c| c.seq > after_seq).cloned().collect()
    }

    fn latest_seq(&self) -> u64 {
        self.inner
            .lock()
            .expect("changelog poisoned")
            .last()
            .map(|c| c.seq)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Durable WAL — a redb-backed ChangeLog
// ---------------------------------------------------------------------------

use duxx_storage::{open_backend, Backend};

const WAL_TABLE: &str = "wal";

/// Durable, redb-backed [`ChangeLog`]. The sequence feed survives process
/// restart, so a follower can resume from its last applied sequence after
/// either node bounces — the foundation for real (not in-memory) replication.
pub struct RedbLog {
    backend: Box<dyn Backend>,
    next: Mutex<u64>,
}

impl RedbLog {
    /// Open (or create) a durable WAL at `path` (a redb file). Recovers the
    /// next sequence from whatever is already on disk.
    pub fn open(path: &str) -> Result<Self, String> {
        let backend =
            open_backend(Some(&format!("redb:{path}"))).map_err(|e| format!("open wal: {e}"))?;
        let last = backend
            .scan(WAL_TABLE)
            .map_err(|e| format!("scan wal: {e}"))?
            .last()
            .map(|(k, _)| be_u64(k))
            .unwrap_or(0);
        Ok(Self {
            backend,
            next: Mutex::new(last),
        })
    }
}

fn be_u64(k: &[u8]) -> u64 {
    <[u8; 8]>::try_from(k).map(u64::from_be_bytes).unwrap_or(0)
}

impl ChangeLog for RedbLog {
    fn append(&self, namespace: &str, op: Vec<u8>) -> u64 {
        let mut n = self.next.lock().expect("wal next poisoned");
        let seq = *n + 1;
        // 8-byte big-endian key → lexicographic order == sequence order, so
        // `scan` returns changes already sorted.
        let value =
            bincode::serialize(&(namespace.to_string(), op)).expect("wal entry encode");
        self.backend
            .put(WAL_TABLE, &seq.to_be_bytes(), &value)
            .expect("wal put");
        *n = seq;
        seq
    }

    fn since(&self, after_seq: u64) -> Vec<Change> {
        self.backend
            .scan(WAL_TABLE)
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (be_u64(&k), v))
            .filter(|(seq, _)| *seq > after_seq)
            .map(|(seq, v)| {
                let (namespace, op): (String, Vec<u8>) =
                    bincode::deserialize(&v).unwrap_or_default();
                Change {
                    seq,
                    namespace,
                    op,
                }
            })
            .collect()
    }

    fn latest_seq(&self) -> u64 {
        *self.next.lock().expect("wal next poisoned")
    }

    fn earliest_seq(&self) -> u64 {
        self.backend
            .scan(WAL_TABLE)
            .unwrap_or_default()
            .first()
            .map(|(k, _)| be_u64(k))
            .unwrap_or(0)
    }
}

impl RedbLog {
    /// Drop every change with `seq <= up_to_seq` — call after a snapshot covers
    /// them, so the log doesn't grow without bound. The sequence counter is
    /// unchanged (new appends continue). Returns the number removed.
    pub fn compact(&self, up_to_seq: u64) -> usize {
        let stale: Vec<Vec<u8>> = self
            .backend
            .scan(WAL_TABLE)
            .unwrap_or_default()
            .into_iter()
            .map(|(k, _)| k)
            .filter(|k| be_u64(k) <= up_to_seq)
            .collect();
        let n = stale.len();
        for k in &stale {
            let _ = self.backend.delete(WAL_TABLE, k);
        }
        n
    }
}

/// Durable follower cursor: the last sequence this follower applied, persisted
/// to a file so a restart **resumes** instead of re-pulling (and re-applying)
/// the whole log from zero. Writes via temp-file + rename for crash safety.
pub struct DurableCursor {
    path: PathBuf,
}

impl DurableCursor {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Last applied sequence (0 if none persisted yet).
    pub fn load(&self) -> u64 {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Persist the applied sequence.
    pub fn save(&self, seq: u64) -> std::io::Result<()> {
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, seq.to_string())?;
        std::fs::rename(&tmp, &self.path)
    }
}

/// Snapshot bootstrap: capture a leader's durable data + the change-feed
/// sequence it represents, so a **brand-new** replica can start from the
/// snapshot and resume replication from that sequence — instead of replaying
/// the whole log (which may have been compacted away).
pub mod snapshot {
    use std::fs;
    use std::io;
    use std::path::Path;

    /// File at the snapshot root recording the sequence it was captured at.
    pub const MANIFEST: &str = "SNAPSHOT_SEQ";

    /// Capture `data_dir` into `dest` (a recursive copy) tagged with the
    /// change-feed `seq` it represents.
    ///
    /// The data must be **quiesced** (no active writer) for a consistent copy —
    /// snapshot from a cold/flushed state (e.g. a stopped node, or after
    /// evicting/flushing the workspaces). `seq` is the leader's `latest_seq()`
    /// at that moment.
    pub fn create(data_dir: &Path, dest: &Path, seq: u64) -> io::Result<()> {
        copy_dir_all(data_dir, dest)?;
        fs::write(dest.join(MANIFEST), seq.to_string())
    }

    /// Restore a snapshot into `target_dir`, returning the sequence it was
    /// captured at — the point the new replica resumes replication from. The
    /// manifest itself is not copied into the data.
    pub fn restore(snapshot_dir: &Path, target_dir: &Path) -> io::Result<u64> {
        let seq = fs::read_to_string(snapshot_dir.join(MANIFEST))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        fs::create_dir_all(target_dir)?;
        for entry in fs::read_dir(snapshot_dir)? {
            let entry = entry?;
            if entry.file_name() == MANIFEST {
                continue;
            }
            let from = entry.path();
            let to = target_dir.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                copy_dir_all(&from, &to)?;
            } else {
                fs::copy(&from, &to)?;
            }
        }
        Ok(seq)
    }

    fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let from = entry.path();
            let to = dst.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                copy_dir_all(&from, &to)?;
            } else {
                fs::copy(&from, &to)?;
            }
        }
        Ok(())
    }
}

/// A follower replica. Tracks the last applied sequence and pulls newer
/// changes from a leader's [`ChangeLog`] on `catch_up`.
#[derive(Debug, Default)]
pub struct Follower {
    applied_seq: u64,
}

impl Follower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sequence resumed from after a restart (e.g. read from local durable
    /// state) — so the follower doesn't re-apply what it already has.
    pub fn resume_from(applied_seq: u64) -> Self {
        Self { applied_seq }
    }

    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    /// Pull every change newer than `applied_seq` from `leader`, apply each in
    /// order via `apply`, and advance. Returns the number applied. Idempotent
    /// across calls: a second `catch_up` with no new changes applies nothing.
    pub fn catch_up(&mut self, leader: &dyn ChangeLog, mut apply: impl FnMut(&Change)) -> usize {
        let changes = leader.since(self.applied_seq);
        let n = changes.len();
        for c in &changes {
            apply(c);
            self.applied_seq = c.seq;
        }
        n
    }

    /// Whether this follower is fully caught up to `leader`.
    pub fn is_current(&self, leader: &dyn ChangeLog) -> bool {
        self.applied_seq >= leader.latest_seq()
    }
}

/// Round-robin read routing across a set of replicas. A real router would also
/// weight by replica lag (`is_current`/`applied_seq`) and health; this picks
/// the next index and is enough to pin the interface.
#[derive(Debug, Default)]
pub struct ReadRouter {
    next: usize,
}

impl ReadRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index of the replica to serve the next read from, or `None` if there
    /// are no replicas.
    pub fn pick(&mut self, replica_count: usize) -> Option<usize> {
        if replica_count == 0 {
            return None;
        }
        let i = self.next % replica_count;
        self.next = self.next.wrapping_add(1);
        Some(i)
    }

    /// **Lag-aware** routing: round-robin among replicas whose lag behind the
    /// leader is within `max_lag`; if none qualify (all stale), fall back to
    /// the freshest replica rather than returning nothing. This keeps reads off
    /// replicas that are too far behind to serve consistent-enough data.
    pub fn pick_fresh(
        &mut self,
        replicas: &[Replica],
        leader_seq: u64,
        max_lag: u64,
    ) -> Option<usize> {
        if replicas.is_empty() {
            return None;
        }
        let eligible: Vec<usize> = replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| leader_seq.saturating_sub(r.applied_seq) <= max_lag)
            .map(|(i, _)| i)
            .collect();
        if eligible.is_empty() {
            // All replicas are stale — serve from the freshest.
            return replicas
                .iter()
                .enumerate()
                .max_by_key(|(_, r)| r.applied_seq)
                .map(|(i, _)| i);
        }
        let pick = eligible[self.next % eligible.len()];
        self.next = self.next.wrapping_add(1);
        Some(pick)
    }
}

/// A replica's lag snapshot, for [`ReadRouter::pick_fresh`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replica {
    pub node_id: NodeId,
    pub applied_seq: u64,
}

// ---------------------------------------------------------------------------
// Coordinator — leader election & failover
// ---------------------------------------------------------------------------

/// A node's identity in the cluster.
pub type NodeId = String;

/// Current leadership view: who leads (if anyone) and the monotonic election
/// `term` (bumped on every leadership change — fencing token against split
/// brain).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leadership {
    pub leader: Option<NodeId>,
    pub term: u64,
}

/// Coordinates leader election and failover across nodes.
///
/// **Do not build Raft here.** The production implementation delegates
/// consensus to an external coordinator (etcd / Consul) via a lease +
/// compare-and-swap — turning failover from a research project into an
/// integration. This crate provides in-process implementations for single-node
/// ([`StaticCoordinator`]) and for tests / simulation ([`LocalCoordinator`]);
/// an `EtcdCoordinator` slots in behind this same trait (the way a
/// `PostgresStore` slots in behind the control plane's `Store` trait).
///
/// ## Implementing `EtcdCoordinator` (the production drop-in)
///
/// etcd is async and this trait is sync (it's polled on the hot path, e.g.
/// `is_leader()` per mutation), so the implementation **caches** leadership in
/// an `Arc<Mutex<Leadership>>` that a background task keeps fresh — the sync
/// methods just read the cache (no network call per write):
///
/// 1. `Client::connect(endpoints)`, then `lease_grant(ttl)` and a
///    `lease_keep_alive` loop so the node's lease stays live.
/// 2. `election_client().campaign(name, node_id, lease)` in the background —
///    it resolves only when this node becomes leader; on resolve, set the
///    cached `Leadership { leader: Some(node_id), term }`.
/// 3. `election_client().observe(name)` streams leadership changes — update the
///    cache (and bump `term`) whenever the leader key changes, so failover is
///    reflected within a lease TTL.
/// 4. `resign()` calls `election_client().resign(leader_key)` and drops the
///    lease, freeing the slot for another node — the same semantics
///    [`LocalCoordinator`] models in-process.
///
/// It needs a running etcd to build (gate it behind an `etcd` cargo feature)
/// and to test (integration test against an etcd container) — which is why this
/// crate ships the in-process impls and not a stub of unverifiable async code.
pub trait Coordinator: Send + Sync {
    /// This node's id.
    fn node_id(&self) -> &str;
    /// Attempt to acquire (or renew) leadership, returning the resulting view.
    /// Idempotent for the current leader; an observer just learns who leads.
    fn campaign(&self) -> Leadership;
    /// Observe leadership without campaigning.
    fn leadership(&self) -> Leadership;
    /// Voluntarily step down (graceful failover trigger). A no-op if this node
    /// is not the leader.
    fn resign(&self);
    /// Whether this node currently holds leadership.
    fn is_leader(&self) -> bool {
        self.leadership().leader.as_deref() == Some(self.node_id())
    }
}

/// Fixed-leader coordinator: single-node deployments or manually-pinned
/// leadership. Never fails over.
#[derive(Debug, Clone)]
pub struct StaticCoordinator {
    node_id: NodeId,
    leader: NodeId,
}

impl StaticCoordinator {
    /// A single-node coordinator where this node is always the leader.
    pub fn solo(node_id: impl Into<NodeId>) -> Self {
        let node_id = node_id.into();
        Self {
            leader: node_id.clone(),
            node_id,
        }
    }

    /// A node that defers to a fixed, externally-chosen `leader`.
    pub fn with_leader(node_id: impl Into<NodeId>, leader: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            leader: leader.into(),
        }
    }
}

impl Coordinator for StaticCoordinator {
    fn node_id(&self) -> &str {
        &self.node_id
    }
    fn campaign(&self) -> Leadership {
        Leadership {
            leader: Some(self.leader.clone()),
            term: 1,
        }
    }
    fn leadership(&self) -> Leadership {
        self.campaign()
    }
    fn resign(&self) {}
}

/// A shared in-process election — the backing store every [`LocalCoordinator`]
/// for a simulated cluster points at. Models the etcd lease/CAS semantics
/// (first writer wins; on resign, the slot frees for the next campaigner) so
/// election and failover can be tested deterministically, without a clock.
#[derive(Clone, Default)]
pub struct LocalElection {
    inner: Arc<Mutex<ElectionState>>,
}

#[derive(Default)]
struct ElectionState {
    leader: Option<NodeId>,
    term: u64,
}

impl LocalElection {
    pub fn new() -> Self {
        Self::default()
    }

    /// A coordinator for `node_id` participating in this election.
    pub fn coordinator(&self, node_id: impl Into<NodeId>) -> LocalCoordinator {
        LocalCoordinator {
            node_id: node_id.into(),
            election: self.clone(),
        }
    }
}

/// A node's handle into a [`LocalElection`].
pub struct LocalCoordinator {
    node_id: NodeId,
    election: LocalElection,
}

impl Coordinator for LocalCoordinator {
    fn node_id(&self) -> &str {
        &self.node_id
    }

    fn campaign(&self) -> Leadership {
        let mut s = self.election.inner.lock().expect("election poisoned");
        if s.leader.is_none() {
            // Win the empty slot (etcd: CAS create-if-absent on the lease key).
            s.leader = Some(self.node_id.clone());
            s.term += 1;
        }
        Leadership {
            leader: s.leader.clone(),
            term: s.term,
        }
    }

    fn leadership(&self) -> Leadership {
        let s = self.election.inner.lock().expect("election poisoned");
        Leadership {
            leader: s.leader.clone(),
            term: s.term,
        }
    }

    fn resign(&self) {
        let mut s = self.election.inner.lock().expect("election poisoned");
        if s.leader.as_deref() == Some(self.node_id.as_str()) {
            s.leader = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leader_assigns_monotonic_sequences() {
        let log = MemoryLog::new();
        assert_eq!(log.latest_seq(), 0);
        assert_eq!(log.append("ns", b"a".to_vec()), 1);
        assert_eq!(log.append("ns", b"b".to_vec()), 2);
        assert_eq!(log.append("ns", b"c".to_vec()), 3);
        assert_eq!(log.latest_seq(), 3);
    }

    #[test]
    fn follower_catches_up_and_converges() {
        let leader = MemoryLog::new();
        leader.append("ns", b"a".to_vec());
        leader.append("ns", b"b".to_vec());

        let mut f = Follower::new();
        assert!(!f.is_current(&leader));

        let mut seen = Vec::new();
        let n = f.catch_up(&leader, |c| seen.push((c.seq, c.op.clone())));
        assert_eq!(n, 2);
        assert_eq!(seen, vec![(1, b"a".to_vec()), (2, b"b".to_vec())]);
        assert!(f.is_current(&leader));

        // A second catch-up with no new writes applies nothing (idempotent).
        assert_eq!(f.catch_up(&leader, |_| panic!("should not re-apply")), 0);

        // New writes flow incrementally.
        leader.append("ns", b"c".to_vec());
        let mut seen2 = Vec::new();
        assert_eq!(f.catch_up(&leader, |c| seen2.push(c.seq)), 1);
        assert_eq!(seen2, vec![3]);
        assert_eq!(f.applied_seq(), leader.latest_seq());
    }

    #[test]
    fn follower_resumes_without_replaying() {
        let leader = MemoryLog::new();
        for b in [b"a", b"b", b"c"] {
            leader.append("ns", b.to_vec());
        }
        // Restarted follower that already applied up to seq 2.
        let mut f = Follower::resume_from(2);
        let mut seen = Vec::new();
        f.catch_up(&leader, |c| seen.push(c.seq));
        assert_eq!(seen, vec![3], "must not re-apply already-applied changes");
    }

    #[test]
    fn read_router_round_robins() {
        let mut r = ReadRouter::new();
        assert_eq!(r.pick(0), None);
        assert_eq!(r.pick(3), Some(0));
        assert_eq!(r.pick(3), Some(1));
        assert_eq!(r.pick(3), Some(2));
        assert_eq!(r.pick(3), Some(0));
    }

    #[test]
    fn roles_are_distinct() {
        assert_ne!(ReplicationRole::Leader, ReplicationRole::Follower);
    }

    #[test]
    fn static_coordinator_solo_is_always_leader() {
        let c = StaticCoordinator::solo("node-a");
        assert!(c.is_leader());
        assert_eq!(c.campaign().leader.as_deref(), Some("node-a"));
        c.resign(); // no-op
        assert!(c.is_leader());
    }

    #[test]
    fn local_election_elects_one_leader_then_fails_over() {
        let election = LocalElection::new();
        let a = election.coordinator("node-a");
        let b = election.coordinator("node-b");

        // A campaigns first and wins term 1; B observes A as leader.
        let la = a.campaign();
        assert_eq!(la.leader.as_deref(), Some("node-a"));
        assert_eq!(la.term, 1);
        assert!(a.is_leader());
        let lb = b.campaign();
        assert_eq!(lb.leader.as_deref(), Some("node-a"), "B must not steal leadership");
        assert!(!b.is_leader());

        // A fails / steps down → B campaigns and wins term 2 (failover).
        a.resign();
        assert!(b.campaign().leader.as_deref() == Some("node-b"));
        assert!(b.is_leader());
        assert!(!a.is_leader());
        assert_eq!(b.leadership().term, 2, "term advances on leadership change");
    }

    #[test]
    fn redb_wal_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.redb");
        let path = path.to_str().unwrap();

        {
            let log = RedbLog::open(path).unwrap();
            assert_eq!(log.append("ns", b"a".to_vec()), 1);
            assert_eq!(log.append("ns", b"b".to_vec()), 2);
            assert_eq!(log.latest_seq(), 2);
        } // dropped — flushed to disk

        // Reopen: sequence + entries survived.
        let log = RedbLog::open(path).unwrap();
        assert_eq!(log.latest_seq(), 2);
        assert_eq!(log.since(0).len(), 2);
        // New appends continue the sequence.
        assert_eq!(log.append("ns", b"c".to_vec()), 3);
        let tail: Vec<u64> = log.since(1).iter().map(|c| c.seq).collect();
        assert_eq!(tail, vec![2, 3]);
        assert_eq!(log.since(1)[0].op, b"b".to_vec());
    }

    #[test]
    fn snapshot_create_and_restore_round_trips() {
        use std::fs;
        let tmp = tempfile::tempdir().unwrap();

        // A leader's durable data dir (workspace files).
        let data = tmp.path().join("data");
        fs::create_dir_all(data.join("org/proj/prod")).unwrap();
        fs::write(data.join("org/proj/prod/memory.redb"), b"WORKSPACE-DATA").unwrap();
        fs::write(data.join("version.json"), b"{}").unwrap();

        // Snapshot at seq 42.
        let snap = tmp.path().join("snap");
        super::snapshot::create(&data, &snap, 42).unwrap();
        assert!(snap.join(super::snapshot::MANIFEST).exists());

        // A fresh replica restores it and learns the resume sequence.
        let restored = tmp.path().join("restored");
        let seq = super::snapshot::restore(&snap, &restored).unwrap();
        assert_eq!(seq, 42, "replica resumes replication from the snapshot seq");
        assert_eq!(
            fs::read(restored.join("org/proj/prod/memory.redb")).unwrap(),
            b"WORKSPACE-DATA"
        );
        assert!(
            !restored.join(super::snapshot::MANIFEST).exists(),
            "manifest must not be copied into the data dir"
        );
    }

    #[test]
    fn durable_cursor_persists_applied_seq() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("applied.seq");

        let c = DurableCursor::open(&path);
        assert_eq!(c.load(), 0, "no cursor yet → 0");
        c.save(5).unwrap();
        assert_eq!(c.load(), 5);

        // A restarted follower reads the same value.
        let c2 = DurableCursor::open(&path);
        assert_eq!(c2.load(), 5, "follower resumes from the persisted seq");
        c2.save(9).unwrap();
        assert_eq!(c.load(), 9);
    }

    #[test]
    fn wal_compaction_bounds_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.redb");
        let path = path.to_str().unwrap();
        let log = RedbLog::open(path).unwrap();
        for b in [b"a", b"b", b"c", b"d", b"e"] {
            log.append("ns", b.to_vec());
        }
        assert_eq!(log.earliest_seq(), 1);

        // Snapshot covered up to seq 3 → compact them away.
        assert_eq!(log.compact(3), 3);
        assert_eq!(log.earliest_seq(), 4, "earliest rises past the compaction point");
        assert_eq!(
            log.since(0).iter().map(|c| c.seq).collect::<Vec<_>>(),
            vec![4, 5]
        );
        // Sequence counter is unchanged — new appends continue.
        assert_eq!(log.latest_seq(), 5);
        assert_eq!(log.append("ns", b"f".to_vec()), 6);
    }

    #[test]
    fn read_router_prefers_fresh_replicas() {
        let replicas = vec![
            Replica { node_id: "r1".into(), applied_seq: 100 }, // lag 5  — fresh
            Replica { node_id: "r2".into(), applied_seq: 40 },  // lag 65 — stale
            Replica { node_id: "r3".into(), applied_seq: 103 }, // lag 2  — fresh
        ];
        let leader_seq = 105;
        let mut router = ReadRouter::new();
        // With max_lag=10, only r1 and r3 are eligible → round-robin between them.
        let picks: Vec<usize> = (0..4)
            .map(|_| router.pick_fresh(&replicas, leader_seq, 10).unwrap())
            .collect();
        assert!(picks.iter().all(|&i| i == 0 || i == 2), "stale r2 must be skipped: {picks:?}");

        // If ALL are stale, fall back to the freshest (r3 @ 103).
        let mut r2 = ReadRouter::new();
        assert_eq!(r2.pick_fresh(&replicas, leader_seq, 1), Some(2));
    }
}
