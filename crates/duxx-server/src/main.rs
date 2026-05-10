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
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let mut addr = String::from("127.0.0.1:6379");
    let mut embedder_spec: Option<String> = None;
    let mut storage_spec: Option<String> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--addr" | "-a" => {
                addr = args.next().ok_or_else(|| anyhow::anyhow!("--addr needs a value"))?;
            }
            "--embedder" | "-e" => {
                embedder_spec = Some(
                    args.next().ok_or_else(|| anyhow::anyhow!("--embedder needs a value"))?,
                );
            }
            "--storage" | "-s" => {
                storage_spec = Some(
                    args.next().ok_or_else(|| anyhow::anyhow!("--storage needs a value"))?,
                );
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let embedder_spec = embedder_spec
        .or_else(|| std::env::var("DUXX_EMBEDDER").ok())
        .unwrap_or_else(|| "hash:32".to_string());
    let storage_spec = storage_spec.or_else(|| std::env::var("DUXX_STORAGE").ok());

    tracing::info!(
        version = duxx_server::SERVER_VERSION,
        %addr,
        embedder = %embedder_spec,
        storage = ?storage_spec,
        "starting duxx-server"
    );

    let embedder: Arc<dyn duxx_embed::Embedder> = match duxx_embed::from_spec(Some(&embedder_spec))?
    {
        Some(provider) => Arc::from(provider),
        None => Arc::new(duxx_embed::HashEmbedder::new(32)),
    };

    let server = match storage_spec.as_deref() {
        Some(spec) => {
            let storage = open_storage(spec)?;
            duxx_server::Server::with_provider_and_storage(embedder, storage)?
        }
        None => duxx_server::Server::with_provider(embedder),
    };
    server.serve(&addr).await?;
    Ok(())
}

/// Parse a `--storage` spec and open the named backend.
fn open_storage(spec: &str) -> anyhow::Result<Arc<dyn duxx_storage::Storage>> {
    let (kind, rest) = spec.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("storage spec must be 'kind:path' (e.g. 'redb:./data/duxx.redb')")
    })?;
    match kind {
        "memory" => Ok(Arc::new(duxx_storage::MemoryStorage::new())),
        "redb" => {
            // Make sure the parent dir exists.
            let path = std::path::Path::new(rest);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Ok(Arc::new(duxx_storage::RedbStorage::open(path)?))
        }
        other => anyhow::bail!("unknown storage kind: {other} (built-in: memory, redb)"),
    }
}

fn print_help() {
    println!("duxx-server v{}", duxx_server::SERVER_VERSION);
    println!();
    println!("USAGE: duxx-server [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --addr HOST:PORT       Listen address     (default 127.0.0.1:6379)");
    println!("  --embedder SPEC        Embedder spec      (default hash:32)");
    println!("  --storage SPEC         Storage backend    (default in-memory only)");
    println!();
    println!("EMBEDDER SPECS:");
    println!("  hash:<dim>                          deterministic toy embedder");
    println!("  openai:text-embedding-3-small       requires OPENAI_API_KEY (1536-d)");
    println!("  openai:text-embedding-3-large       requires OPENAI_API_KEY (3072-d)");
    println!("  cohere:embed-english-v3.0           requires COHERE_API_KEY  (1024-d)");
    println!();
    println!("STORAGE SPECS:");
    println!("  memory:<ignored>                    in-memory only (lost on exit)");
    println!("  redb:./path/to/file.redb            durable, ACID, single-process");
    println!();
    println!("ENV: DUXX_EMBEDDER, DUXX_STORAGE, OPENAI_API_KEY, COHERE_API_KEY");
}
