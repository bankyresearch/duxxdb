//! `duxx` — interactive shell and admin CLI for DuxxDB.
//!
//! Phase-1 skeleton. Real command parsing lands with `clap`.

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("DuxxDB {} — interactive shell", duxx_core::VERSION);
    println!("(Phase-1 stub. See crates/duxx-cli/examples/chatbot_memory.rs for a demo.)");
    Ok(())
}
