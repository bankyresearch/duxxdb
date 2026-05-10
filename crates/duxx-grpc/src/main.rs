//! `duxx-grpc` daemon binary.
//!
//! Usage:
//! ```bash
//! duxx-grpc                                       # 127.0.0.1:50051, in-memory
//! duxx-grpc --addr 0.0.0.0:50051
//! duxx-grpc --storage dir:./data/duxx
//! duxx-grpc --embedder openai:text-embedding-3-small
//! ```

use duxx_grpc::DuxxService;
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
    let mut addr = String::from("127.0.0.1:50051");
    let mut embedder_spec: Option<String> = None;
    let mut storage_spec: Option<String> = None;
    let mut token: Option<String> = None;

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
        version = duxx_grpc::SERVER_VERSION,
        %addr,
        embedder = %embedder_spec,
        storage = ?storage_spec,
        "starting duxx-grpc"
    );

    let embedder: Arc<dyn duxx_embed::Embedder> =
        match duxx_embed::from_spec(Some(&embedder_spec))? {
            Some(provider) => Arc::from(provider),
            None => Arc::new(duxx_embed::HashEmbedder::new(32)),
        };

    let mut svc = match storage_spec.as_deref() {
        Some(spec) if spec.starts_with("dir:") => {
            let dir = &spec[4..];
            std::fs::create_dir_all(dir)?;
            DuxxService::open_at(embedder, dir)?
        }
        Some(spec) => {
            anyhow::bail!(
                "duxx-grpc currently supports only dir:./path or no storage; got {spec}"
            )
        }
        None => DuxxService::with_provider(embedder),
    };

    if let Some(t) = token {
        if t.is_empty() {
            anyhow::bail!("--token / DUXX_TOKEN must not be empty");
        }
        if t.len() < 16 {
            tracing::warn!(
                "auth token shorter than 16 chars; use a stronger value in production"
            );
        }
        svc = svc.with_auth(t);
        tracing::info!("authentication ENABLED (Bearer token required)");
    } else {
        tracing::warn!(
            "running without authentication. Bind to 127.0.0.1 only, or set --token / DUXX_TOKEN."
        );
    }

    tokio::select! {
        res = svc.clone().serve(&addr) => res?,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Ctrl+C received -- shutting down");
        }
    }
    drop(svc);
    tracing::info!("duxx-grpc stopped");
    Ok(())
}

fn print_help() {
    println!("duxx-grpc v{}", duxx_grpc::SERVER_VERSION);
    println!();
    println!("USAGE: duxx-grpc [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --addr HOST:PORT       Listen address     (default 127.0.0.1:50051)");
    println!("  --embedder SPEC        Embedder spec      (default hash:32)");
    println!("  --storage dir:./path   Persistent on-disk store (rows + indices)");
    println!("  --token TOKEN          Require Bearer TOKEN on every RPC");
    println!("                         (default: no auth -- localhost-only safe)");
    println!();
    println!("Health: grpc.health.v1.Health/Check is always available");
    println!("(no auth required) for k8s livenessProbe / readinessProbe.");
    println!();
    println!("ENV: DUXX_EMBEDDER, DUXX_STORAGE, DUXX_TOKEN,");
    println!("     OPENAI_API_KEY, COHERE_API_KEY");
}
