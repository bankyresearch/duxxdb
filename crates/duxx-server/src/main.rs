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
    let mut token: Option<String> = None;
    let mut drain_secs: u64 = 30;
    let mut metrics_addr: Option<String> = None;
    let mut tls_cert: Option<String> = None;
    let mut tls_key: Option<String> = None;
    let mut max_memories: Option<usize> = None;

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
            "--token" | "-t" => {
                token = Some(
                    args.next().ok_or_else(|| anyhow::anyhow!("--token needs a value"))?,
                );
            }
            "--drain-secs" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--drain-secs needs a value"))?;
                drain_secs = v.parse()?;
            }
            "--metrics-addr" => {
                metrics_addr = Some(
                    args.next().ok_or_else(|| {
                        anyhow::anyhow!("--metrics-addr needs HOST:PORT")
                    })?,
                );
            }
            "--tls-cert" => {
                tls_cert = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--tls-cert needs a path"))?,
                );
            }
            "--tls-key" => {
                tls_key = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--tls-key needs a path"))?,
                );
            }
            "--max-memories" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--max-memories needs a value"))?;
                max_memories = Some(v.parse()?);
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let token = token.or_else(|| std::env::var("DUXX_TOKEN").ok());

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

    let mut server = match storage_spec.as_deref() {
        // dir:./path -- full persistence: redb rows + tantivy index +
        // HNSW dump under one directory. Reopens skip the rebuild.
        Some(spec) if spec.starts_with("dir:") => {
            let path = &spec[4..];
            std::fs::create_dir_all(path)?;
            duxx_server::Server::open_at(embedder, path)?
        }
        // memory:_, redb:./file -- byte-keyed Storage; rows persist,
        // indices are rebuilt from rows on open.
        Some(spec) => {
            let storage = open_storage(spec)?;
            duxx_server::Server::with_provider_and_storage(embedder, storage)?
        }
        None => duxx_server::Server::with_provider(embedder),
    };

    if let Some(t) = token {
        if t.is_empty() {
            anyhow::bail!("--token / DUXX_TOKEN must not be empty");
        }
        if t.len() < 16 {
            tracing::warn!(
                "auth token is shorter than 16 characters; use a stronger value in production"
            );
        }
        server = server.with_auth(t);
        tracing::info!("authentication ENABLED (clients must AUTH)");
    } else {
        tracing::warn!(
            "running without authentication. Bind to 127.0.0.1 only, OR set --token / DUXX_TOKEN."
        );
    }

    // Optional native TLS termination (Phase 6.2). Both --tls-cert and
    // --tls-key must be provided together, or neither.
    let tls_cert = tls_cert.or_else(|| std::env::var("DUXX_TLS_CERT").ok());
    let tls_key = tls_key.or_else(|| std::env::var("DUXX_TLS_KEY").ok());
    match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => {
            server = server.with_tls_files(&cert, &key)?;
            tracing::info!(cert = %cert, "TLS termination ENABLED");
        }
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("TLS requires BOTH --tls-cert and --tls-key");
        }
        (None, None) => {}
    }

    // Optional memory cap with importance-based eviction (Phase 6.2).
    let max_memories = max_memories.or_else(|| {
        std::env::var("DUXX_MAX_MEMORIES")
            .ok()
            .and_then(|s| s.parse().ok())
    });
    if let Some(cap) = max_memories {
        server.memory().set_max_rows(Some(cap));
        tracing::info!(
            max_memories = cap,
            "memory cap ENABLED (oldest decayed-importance evicted on overflow)"
        );
    }

    // Optional Prometheus / health endpoint on a separate listener.
    let metrics_addr = metrics_addr.or_else(|| std::env::var("DUXX_METRICS_ADDR").ok());
    if let Some(addr) = metrics_addr {
        let m = duxx_server::metrics::Metrics::new();
        server = server.with_metrics(m.clone());
        let parsed: std::net::SocketAddr = addr.parse()?;
        tokio::spawn(async move {
            if let Err(e) = duxx_server::metrics::serve(m, parsed).await {
                tracing::warn!(error = %e, "metrics endpoint stopped");
            }
        });
    }

    // Graceful shutdown: serve until Ctrl+C / SIGTERM, then drain
    // in-flight connections for `drain_secs` before exiting.
    let drain = std::time::Duration::from_secs(drain_secs);
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Ctrl+C received");
    };
    let still_open = server.serve_with_shutdown(&addr, shutdown, drain).await?;
    drop(server); // explicit: triggers MemoryStore Drop -> tantivy commit + HNSW dump
    if still_open > 0 {
        tracing::warn!(
            "{still_open} connections did not close within {drain_secs}s drain window"
        );
    }
    tracing::info!("duxx-server stopped");
    Ok(())
}

/// Parse a `--storage` spec and open the named byte-keyed backend.
/// `dir:` is handled separately upstream because it constructs a
/// full `MemoryStore` rather than a `Storage`.
fn open_storage(spec: &str) -> anyhow::Result<Arc<dyn duxx_storage::Storage>> {
    let (kind, rest) = spec.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("storage spec must be 'kind:path' (e.g. 'redb:./data/duxx.redb')")
    })?;
    match kind {
        "memory" => Ok(Arc::new(duxx_storage::MemoryStorage::new())),
        "redb" => {
            let path = std::path::Path::new(rest);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Ok(Arc::new(duxx_storage::RedbStorage::open(path)?))
        }
        other => anyhow::bail!(
            "unknown storage kind: {other} (built-in: memory, redb, dir)"
        ),
    }
}

fn print_help() {
    println!("duxx-server v{}", duxx_server::SERVER_VERSION);
    println!();
    println!("USAGE: duxx-server [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --addr HOST:PORT       Listen address       (default 127.0.0.1:6379)");
    println!("  --embedder SPEC        Embedder spec        (default hash:32)");
    println!("  --storage SPEC         Storage backend      (default in-memory only)");
    println!("  --token TOKEN          Require AUTH <TOKEN> from every client");
    println!("                         (default: no auth -- localhost-only safe)");
    println!("  --drain-secs N         Shutdown drain budget (default 30)");
    println!("  --metrics-addr HOST:PORT  Bind a Prometheus + /health endpoint");
    println!("                         (default: disabled)");
    println!("  --tls-cert PATH        PEM cert chain (Phase 6.2)");
    println!("  --tls-key  PATH        PEM private key (must accompany --tls-cert)");
    println!("  --max-memories N       Cap memory rows; oldest decayed-importance");
    println!("                         is evicted on overflow (default: unlimited)");
    println!();
    println!("EMBEDDER SPECS:");
    println!("  hash:<dim>                          deterministic toy embedder");
    println!("  openai:text-embedding-3-small       requires OPENAI_API_KEY (1536-d)");
    println!("  openai:text-embedding-3-large       requires OPENAI_API_KEY (3072-d)");
    println!("  cohere:embed-english-v3.0           requires COHERE_API_KEY  (1024-d)");
    println!();
    println!("STORAGE SPECS:");
    println!("  memory:<ignored>                    in-memory only (lost on exit)");
    println!("  redb:./path/to/file.redb            durable rows; indices rebuilt on open");
    println!("  dir:./path/to/dir                   FULLY persistent: rows + tantivy + HNSW");
    println!("                                      (skips rebuild on graceful reopen)");
    println!();
    println!("ENV: DUXX_EMBEDDER, DUXX_STORAGE, DUXX_TOKEN, DUXX_METRICS_ADDR,");
    println!("     DUXX_TLS_CERT, DUXX_TLS_KEY, DUXX_MAX_MEMORIES,");
    println!("     OPENAI_API_KEY, COHERE_API_KEY");
}
