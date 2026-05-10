//! `duxx-export` — one-shot Parquet export from a persistent
//! DuxxDB directory.
//!
//! ```bash
//! duxx-export --storage dir:./data/duxx --out ./cold/memories-2026-05.parquet
//! ```

use duxx_coldtier::ParquetExporter;
use duxx_memory::MemoryStore;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let mut storage_spec: Option<String> = None;
    let mut out: Option<String> = None;
    let mut dim: usize = 32;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--storage" | "-s" => {
                storage_spec =
                    Some(args.next().ok_or_else(|| anyhow::anyhow!("--storage needs a value"))?);
            }
            "--out" | "-o" => {
                out = Some(args.next().ok_or_else(|| anyhow::anyhow!("--out needs a value"))?);
            }
            "--dim" | "-d" => {
                let v = args.next().ok_or_else(|| anyhow::anyhow!("--dim needs a value"))?;
                dim = v.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    let storage_spec = storage_spec
        .or_else(|| std::env::var("DUXX_STORAGE").ok())
        .ok_or_else(|| anyhow::anyhow!("--storage or DUXX_STORAGE required"))?;
    let out = out.ok_or_else(|| anyhow::anyhow!("--out required"))?;

    let store = match storage_spec.as_str() {
        spec if spec.starts_with("dir:") => {
            let path = &spec[4..];
            tracing::info!(path, dim, "opening MemoryStore::open_at");
            MemoryStore::open_at(dim, 100_000, path)?
        }
        other => anyhow::bail!(
            "duxx-export currently supports only 'dir:./path' storage; got {other}"
        ),
    };

    tracing::info!(
        memories = store.len(),
        out = %out,
        "starting parquet export"
    );
    let n = ParquetExporter::new().write(&store, &out)?;
    tracing::info!(count = n, out = %out, "export complete");
    println!("exported {n} memories -> {out}");
    Ok(())
}

fn print_help() {
    println!("duxx-export v{}", duxx_coldtier::VERSION);
    println!();
    println!("USAGE: duxx-export --storage dir:./path --out file.parquet [--dim N]");
    println!();
    println!("Reads every memory from a persistent DuxxDB directory and writes");
    println!("an Apache Parquet file readable by Spark / DuckDB / Polars / pandas.");
    println!();
    println!("ENV: DUXX_STORAGE (alternative to --storage)");
}
