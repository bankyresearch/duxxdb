//! `duxx-control` — control-plane demo / provisioning helper.
//!
//! This kickoff binary walks the managed-cloud flow end to end and prints the
//! exact `--auth-key` lines a data-plane node should serve. It is both a smoke
//! test and a worked example of how the control plane drives `duxx-server`.
//!
//! A real deployment replaces this with an HTTP/gRPC API + Cloud Console; the
//! [`duxx_control::ControlPlane`] library logic stays the same.

use duxx_control::{ControlPlane, Env, PlacementMode, RecordingBillingSink, Role};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("serve") => {
            let addr = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "127.0.0.1:7070".to_string());
            serve(addr);
        }
        Some("keygen") => {
            let dir = args.get(2).cloned().unwrap_or_else(|| ".".to_string());
            keygen(&dir);
        }
        Some("-h") | Some("--help") => {
            println!("duxx-control             run the provisioning demo");
            println!("duxx-control serve ADDR  start the HTTP API (default 127.0.0.1:7070)");
            println!("duxx-control keygen DIR  write an Ed25519 keypair (ed25519.key/.pub) to DIR");
            println!("\nenv for `serve`:");
            println!("  DUXX_CONTROL_ED25519_KEY  path to an Ed25519 private DER (asymmetric, preferred)");
            println!(
                "  DUXX_CONTROL_JWT_SECRET   HS256 signing secret (fallback when no Ed25519 key)"
            );
        }
        _ => demo(),
    }
}

/// Generate an Ed25519 keypair: private DER → `DIR/ed25519.key` (give to the
/// control plane), public DER → `DIR/ed25519.pub` (give to every data-plane
/// node via `duxx-server --jwt-public-key`).
fn keygen(dir: &str) {
    let (private_der, public_der) = duxx_control::generate_ed25519().expect("ed25519 keygen");
    std::fs::create_dir_all(dir).expect("create keygen dir");
    let priv_path = std::path::Path::new(dir).join("ed25519.key");
    let pub_path = std::path::Path::new(dir).join("ed25519.pub");
    std::fs::write(&priv_path, &private_der).expect("write private key");
    std::fs::write(&pub_path, &public_der).expect("write public key");
    println!("wrote private key -> {}", priv_path.display());
    println!("wrote public  key -> {}", pub_path.display());
    println!("\nrun the control plane with:");
    println!(
        "  $env:DUXX_CONTROL_ED25519_KEY=\"{}\"; duxx-control serve",
        priv_path.display()
    );
    println!("run each data-plane node with:");
    println!("  duxx-server --jwt-public-key {}", pub_path.display());
}

/// Start the control-plane HTTP API. Prefers an Ed25519 private key
/// (`DUXX_CONTROL_ED25519_KEY`, asymmetric); falls back to an HS256 secret.
fn serve(addr: String) {
    let cp = if let Ok(path) = std::env::var("DUXX_CONTROL_ED25519_KEY") {
        let der = std::fs::read(&path).expect("read DUXX_CONTROL_ED25519_KEY");
        println!("signing mode: Ed25519 (private key {path})");
        ControlPlane::with_ed25519(der)
    } else {
        let secret = std::env::var("DUXX_CONTROL_JWT_SECRET")
            .unwrap_or_else(|_| "dev-signing-key-change-me".to_string());
        println!("signing mode: HS256 (shared secret)");
        ControlPlane::with_signing_key(secret.into_bytes())
    };
    let cp = std::sync::Arc::new(cp);
    let sockaddr: std::net::SocketAddr = addr.parse().expect("invalid ADDR (host:port)");
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    println!("duxx-control API listening on http://{sockaddr}");
    println!("  try: curl -s http://{sockaddr}/healthz");
    rt.block_on(async move {
        if let Err(e) = duxx_control::api::serve(cp, sockaddr).await {
            eprintln!("serve error: {e}");
        }
    });
}

fn demo() {
    // A signing key lets the control plane also mint short-lived workspace
    // JWTs (the modern, signed path). Data-plane nodes verify with the same
    // secret via `duxx-server --jwt-secret`.
    let signing_key = b"shared-hs256-secret-between-cp-and-nodes".to_vec();
    let cp = ControlPlane::with_signing_key(signing_key);

    // 1. Onboard an org and a project.
    let org = cp.create_org("Acme Inc").expect("create org");
    let proj = cp
        .create_project(&org.id, "support-bot")
        .expect("create project");
    println!("org     = {} ({})", org.name, org.id);
    println!("project = {} ({})", proj.name, proj.id);

    // 2. Place the project on a shared data-plane node.
    let node = "node-1:6380";
    cp.place_project(&proj.id, node, PlacementMode::Shared)
        .expect("place project");
    println!("placed on {node} (shared)");

    // 3. Issue keys: a prod service account for the agent runtime, and a
    //    read-only observer for a dashboard.
    let svc = cp
        .issue_key(&proj.id, Env::Prod, Role::ServiceAccount, "agent-runtime")
        .expect("issue service key");
    let obs = cp
        .issue_key(&proj.id, Env::Prod, Role::Observer, "dashboard")
        .expect("issue observer key");
    println!("\nissued keys:");
    println!("  {} ({:?})  secret={}", svc.name, svc.role, svc.secret);
    println!("  {} ({:?})  secret={}", obs.name, obs.role, obs.secret);

    // 4. Materialize the node's auth catalog — these are the literal
    //    `--auth-key` arguments duxx-server consumes; the tenant field
    //    `org/project/env` becomes an isolated, durable workspace.
    println!("\ndata-plane args for {node}:");
    for entry in cp.data_plane_auth_entries(node) {
        println!("  --auth-key {entry}");
    }
    println!(
        "  (run the node with: duxx-server --tenants-dir /var/lib/duxx/ws <the --auth-key flags above>)"
    );

    // 4b. The signed path: exchange the service key for a short-lived JWT the
    //     node verifies with `--jwt-secret` (no static catalog needed).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    match cp.mint_jwt(&svc.secret, now, 900) {
        Ok(jwt) => {
            println!("\nshort-lived workspace JWT for {} (900s):", svc.name);
            println!("  {jwt}");
            println!(
                "  (node started with: duxx-server --jwt-secret <shared-secret> --tenants-dir …)"
            );
        }
        Err(e) => println!("\n(jwt mint skipped: {e})"),
    }

    // 5. Meter some usage and flush it to billing (Stripe stands in here).
    cp.record_usage(&proj.id, 1_200, 850, 0.0042);
    cp.record_usage(&proj.id, 900, 400, 0.0031);
    let usage = cp.usage(&proj.id);
    println!(
        "\nusage[{}]: {} requests, {} in / {} out tokens, ${:.4}",
        proj.name, usage.requests, usage.tokens_in, usage.tokens_out, usage.cost_usd
    );

    let sink = RecordingBillingSink::default();
    let n = cp.flush_billing(&sink).expect("flush billing");
    println!("flushed {n} project(s) to billing sink");
}
