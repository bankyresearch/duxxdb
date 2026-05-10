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
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Logs go to stderr so they don't pollute the JSON-RPC stream on stdout.
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Configurable embedder via DUXX_EMBEDDER env var; defaults to hash:32.
    let spec = std::env::var("DUXX_EMBEDDER").unwrap_or_else(|_| "hash:32".to_string());
    tracing::info!(
        version = duxx_mcp::SERVER_VERSION,
        protocol = duxx_mcp::MCP_PROTOCOL_VERSION,
        %spec,
        "starting duxx-mcp"
    );

    let server = match duxx_embed::from_spec(Some(&spec))? {
        Some(provider) => {
            let arc: Arc<dyn duxx_embed::Embedder> = Arc::from(provider);
            McpServer::with_provider(arc)
        }
        None => McpServer::new(),
    };

    let stdin = io::stdin().lock();
    let stdout = io::stdout().lock();
    server.run(BufReader::new(stdin), stdout)?;
    tracing::info!("duxx-mcp shut down cleanly");
    Ok(())
}
