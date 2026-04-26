//! # duxx-mcp
//!
//! Model Context Protocol (MCP) server for DuxxDB.
//! Lets any Claude / GPT / Llama agent store and recall memories
//! via stdio or SSE with zero glue code. Phase-3 target.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder server. Real implementation lands in Phase 3.
#[derive(Debug, Default)]
pub struct McpServer;

impl McpServer {
    pub fn new() -> Self {
        Self
    }
}
