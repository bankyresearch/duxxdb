//! # duxx-reactive
//!
//! Reactive subscription primitives for DuxxDB.
//!
//! Phase 1: a thin wrapper over `tokio::sync::broadcast`. Phase 4 adds
//! WAL tailing, predicate evaluation, and durable resume tokens.

use tokio::sync::broadcast;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A change event emitted when rows are written.
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub table: String,
    /// Optional partition / user / agent key. Used for per-key channel
    /// routing (`PSUBSCRIBE memory.alice*`).
    pub key: Option<String>,
    pub row_id: u64,
    pub kind: ChangeKind,
}

impl ChangeEvent {
    /// Routing channel: `<table>.<key>` if `key` is set, else `<table>`.
    pub fn channel(&self) -> String {
        match &self.key {
            Some(k) => format!("{}.{}", self.table, k),
            None => self.table.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangeKind {
    Insert,
    Update,
    Delete,
}

/// Central pub-sub for changes. One per DuxxDB instance.
#[derive(Debug, Clone)]
pub struct ChangeBus {
    tx: broadcast::Sender<ChangeEvent>,
}

impl ChangeBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publish(&self, event: ChangeEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ChangeEvent> {
        self.tx.subscribe()
    }
}

impl Default for ChangeBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pub_sub_roundtrip() {
        let bus = ChangeBus::new(8);
        let mut rx = bus.subscribe();
        bus.publish(ChangeEvent {
            table: "t".into(),
            key: None,
            row_id: 1,
            kind: ChangeKind::Insert,
        });
        let got = rx.recv().await.unwrap();
        assert_eq!(got.row_id, 1);
        assert_eq!(got.kind, ChangeKind::Insert);
        assert_eq!(got.channel(), "t");
    }

    #[tokio::test]
    async fn keyed_event_channel() {
        let e = ChangeEvent {
            table: "memory".into(),
            key: Some("alice".into()),
            row_id: 7,
            kind: ChangeKind::Insert,
        };
        assert_eq!(e.channel(), "memory.alice");
    }
}
