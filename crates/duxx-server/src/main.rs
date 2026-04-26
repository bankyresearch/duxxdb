//! `duxx-server` — RESP2/3 TCP daemon.
//!
//! Drop-in compatible with `valkey-cli` and `redis-cli`:
//!
//! ```bash
//! duxx-server                          # listens on 127.0.0.1:6379
//! duxx-server --addr 0.0.0.0:9100      # custom address
//! ```
//!
//! Then from another shell:
//! ```bash
//! redis-cli -p 6379
//! 127.0.0.1:6379> PING
//! PONG
//! 127.0.0.1:6379> REMEMBER alice "I lost my wallet"
//! (integer) 1
//! 127.0.0.1:6379> RECALL alice "wallet"
//! 1) 1) (integer) 1
//!    2) "0.032787"
//!    3) "I lost my wallet"
//! ```

use duxx_server::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Tiny arg parse: only --addr <host:port> is supported.
    let mut args = std::env::args().skip(1);
    let mut addr = String::from("127.0.0.1:6379");
    while let Some(a) = args.next() {
        match a.as_str() {
            "--addr" | "-a" => {
                addr = args.next().ok_or_else(|| anyhow::anyhow!("--addr needs a value"))?;
            }
            "--help" | "-h" => {
                println!("duxx-server v{}\n", duxx_server::SERVER_VERSION);
                println!("USAGE: duxx-server [--addr HOST:PORT]");
                println!("Default: 127.0.0.1:6379 (Valkey/Redis-compatible)");
                return Ok(());
            }
            other => {
                anyhow::bail!("unknown arg: {other}");
            }
        }
    }

    tracing::info!(version = duxx_server::SERVER_VERSION, %addr, "starting duxx-server");
    let server = Server::new();
    server.serve(&addr).await?;
    Ok(())
}
