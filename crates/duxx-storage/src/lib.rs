//! # duxx-storage
//!
//! Three layers of storage abstraction:
//!
//! 1. [`Table`] — schema-aware row-oriented placeholder for the future
//!    Lance / Arrow columnar backend. See [PHASE_2_3_PLAN.md] for the
//!    integration plan.
//! 2. [`Storage`] — minimal byte-keyed key/value store keyed by
//!    `u64`. This is what `MemoryStore` uses for durable `by_id`
//!    storage. Pure Rust impls: [`MemoryStorage`] (in-memory) and
//!    [`RedbStorage`] (durable via [redb]).
//! 3. [`Backend`] — multi-table, string-keyed persistence trait used
//!    by the Phase 7 primitives (`PromptRegistry`, `DatasetRegistry`,
//!    `EvalRegistry`, `ReplayRegistry`, `TraceStore`, `CostLedger`).
//!    Pure Rust impls: [`MemoryBackend`] and [`RedbBackend`]. See
//!    `docs/V0_2_0_PLAN.md` for the wider integration plan.
//!
//! [PHASE_2_3_PLAN.md]: https://github.com/duxxdb/duxxdb/blob/main/docs/PHASE_2_3_PLAN.md
//! [redb]: https://github.com/cberner/redb

pub mod backend;
pub mod storage;
pub mod table;

pub use backend::{key, open_backend, Backend, BatchOp, MemoryBackend};
pub use storage::{MemoryStorage, Storage};
pub use table::{Row, RowId, Table};

#[cfg(feature = "redb-store")]
pub use backend::RedbBackend;
#[cfg(feature = "redb-store")]
pub use storage::RedbStorage;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
