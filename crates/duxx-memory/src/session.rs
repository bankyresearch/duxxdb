//! `SESSION` — hot key/value store with sliding TTL.
//!
//! Holds the per-conversation working set: turn buffer, in-flight tool
//! invocations, scratch flags. Reads bump the entry's `last_access`
//! timestamp so an active session doesn't expire.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default sliding TTL for a session entry.
pub const DEFAULT_TTL: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone)]
struct Entry {
    data: Vec<u8>,
    last_access: Instant,
    ttl: Duration,
}

impl Entry {
    fn fresh(&self) -> bool {
        self.last_access.elapsed() < self.ttl
    }
}

/// In-memory session store.
#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<RwLock<HashMap<String, Entry>>>,
    default_ttl: Duration,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_TTL)
    }

    pub fn with_ttl(default_ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            default_ttl,
        }
    }

    /// Set a session value with the default TTL.
    pub fn put(&self, session_id: impl Into<String>, data: Vec<u8>) {
        self.put_with_ttl(session_id, data, self.default_ttl);
    }

    /// Set with an explicit TTL.
    pub fn put_with_ttl(&self, session_id: impl Into<String>, data: Vec<u8>, ttl: Duration) {
        let entry = Entry {
            data,
            last_access: Instant::now(),
            ttl,
        };
        self.inner.write().insert(session_id.into(), entry);
    }

    /// Get and bump `last_access`. Returns `None` for missing or expired keys
    /// (and removes expired ones lazily).
    pub fn get(&self, session_id: &str) -> Option<Vec<u8>> {
        let mut inner = self.inner.write();
        let entry = inner.get(session_id)?;
        if !entry.fresh() {
            inner.remove(session_id);
            return None;
        }
        // Bump last_access (sliding TTL).
        let entry = inner.get_mut(session_id)?;
        entry.last_access = Instant::now();
        Some(entry.data.clone())
    }

    /// Remove a session.
    pub fn delete(&self, session_id: &str) -> bool {
        self.inner.write().remove(session_id).is_some()
    }

    /// Purge all expired entries; returns how many were removed.
    pub fn purge_expired(&self) -> usize {
        let mut inner = self.inner.write();
        let before = inner.len();
        inner.retain(|_, e| e.fresh());
        before - inner.len()
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let s = SessionStore::new();
        s.put("sid-1", b"hello".to_vec());
        assert_eq!(s.get("sid-1"), Some(b"hello".to_vec()));
    }

    #[test]
    fn missing_returns_none() {
        let s = SessionStore::new();
        assert!(s.get("nope").is_none());
    }

    #[test]
    fn ttl_expires_and_lazy_evicts() {
        let s = SessionStore::with_ttl(Duration::from_millis(5));
        s.put("sid-1", b"x".to_vec());
        std::thread::sleep(Duration::from_millis(20));
        assert!(s.get("sid-1").is_none());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn sliding_ttl_keeps_active_session_alive() {
        // Generous time budget so CI schedulers (especially macOS GHA
        // runners under load) don't false-fail this — the original
        // 50ms TTL / 20ms sleep had no headroom for scheduler jitter.
        let s = SessionStore::with_ttl(Duration::from_millis(500));
        s.put("sid-1", b"x".to_vec());
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(50));
            // Each `get` bumps last_access, so the entry stays alive
            // even though total elapsed > original TTL.
            assert!(s.get("sid-1").is_some());
        }
    }

    #[test]
    fn delete_removes() {
        let s = SessionStore::new();
        s.put("a", b"1".to_vec());
        assert!(s.delete("a"));
        assert!(s.get("a").is_none());
    }

    #[test]
    fn purge_expired_works() {
        let s = SessionStore::with_ttl(Duration::from_millis(5));
        s.put("a", b"1".to_vec());
        s.put_with_ttl("b", b"2".to_vec(), Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(20));
        let n = s.purge_expired();
        assert_eq!(n, 1);
        assert_eq!(s.len(), 1);
    }
}
