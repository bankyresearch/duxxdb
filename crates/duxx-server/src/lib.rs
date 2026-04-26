//! # duxx-server
//!
//! Network daemon for DuxxDB — gRPC + RESP3 wire protocols.
//! Phase-3 target. This file is a stub so the workspace compiles.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder server handle. Real implementation lands in Phase 3.
#[derive(Debug, Default)]
pub struct Server;

impl Server {
    pub fn new() -> Self {
        Self
    }
}
