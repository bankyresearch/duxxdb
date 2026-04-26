//! `duxx-mcp` — DuxxDB MCP server binary.
//!
//! Speaks JSON-RPC 2.0 over stdio. Wire it up to any MCP-aware agent
//! (Claude Desktop, Cline, etc.) via its server config.
//!
//! Example Claude Desktop config (`claude_desktop_config.json`):
//! ```json
//! {
//!   "mcpServers": {
//!     "duxxdb": {
//!       "command": "duxx-mcp",
//!       "args": []
//!     }
//!   }
//! }
//! ```

use duxx_mcp::McpServer;
use std::io::{self, BufReader};

fn main() -> anyhow::Result<()> {
    // Logs go to stderr so they don't pollute the JSON-RPC stream on stdout.
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!(
        version = duxx_mcp::SERVER_VERSION,
        protocol = duxx_mcp::MCP_PROTOCOL_VERSION,
        "starting duxx-mcp"
    );

    let stdin = io::stdin().lock();
    let stdout = io::stdout().lock();
    let server = McpServer::new();
    server.run(BufReader::new(stdin), stdout)?;
    tracing::info!("duxx-mcp shut down cleanly");
    Ok(())
}
