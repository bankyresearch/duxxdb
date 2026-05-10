//! TLS support for `duxx-server`.
//!
//! rustls + tokio-rustls. No OpenSSL, no platform certs — operators
//! provide an explicit cert + key pair on disk (typically the same
//! files served by their LB / cert-manager / Let's Encrypt setup).
//!
//! ## What this gives you
//!
//! Once a server is built with [`Server::with_tls_files`], the accept
//! loop terminates TLS *inside* `duxx-server` and feeds the decrypted
//! stream to the same `handle_connection` path used for plain TCP. So:
//!
//! - The auth + drain + metrics paths all run unchanged.
//! - Clients connect with `redis-cli --tls` (or any RESP client that
//!   speaks rustls / OpenSSL).
//! - You no longer need a TLS-terminating sidecar to expose DuxxDB on
//!   a public network.
//!
//! ## What this does NOT do
//!
//! - mTLS (client cert auth) — that's Phase 6.3+.
//! - Automatic cert reloading — restart the daemon to pick up new
//!   files. cert-manager + a sidecar renewer is the standard pattern.
//! - SNI multiplexing — one cert per listener.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;

/// Load a PEM-encoded cert + key from disk and build a rustls
/// `ServerConfig` ready for `tokio_rustls::TlsAcceptor`.
///
/// The cert file may contain a chain (leaf first, then intermediates).
/// The key file must be a single PKCS#8 / RSA / SEC1 private key in
/// PEM form.
pub fn load_server_config(
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> anyhow::Result<Arc<ServerConfig>> {
    // Process-wide default crypto provider. Idempotent — install once,
    // ignore the "already installed" error if a different caller beat us.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_path = cert_path.as_ref();
    let key_path = key_path.as_ref();

    let certs = load_certs(cert_path)?;
    if certs.is_empty() {
        anyhow::bail!("no certificates found in {}", cert_path.display());
    }
    let key = load_private_key(key_path)?;

    let cfg = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| anyhow::anyhow!("rustls: invalid cert/key combination: {e}"))?;

    Ok(Arc::new(cfg))
}

fn load_certs(path: &Path) -> anyhow::Result<Vec<CertificateDer<'static>>> {
    let f = File::open(path)
        .map_err(|e| anyhow::anyhow!("opening cert file {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);
    let certs: Result<Vec<_>, _> = rustls_pemfile::certs(&mut r).collect();
    Ok(certs?)
}

fn load_private_key(path: &Path) -> anyhow::Result<PrivateKeyDer<'static>> {
    let f = File::open(path)
        .map_err(|e| anyhow::anyhow!("opening key file {}: {e}", path.display()))?;
    let mut r = BufReader::new(f);
    // rustls_pemfile::private_key tries PKCS#8, RSA, and SEC1 in order.
    rustls_pemfile::private_key(&mut r)?
        .ok_or_else(|| anyhow::anyhow!("no PEM private key found in {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a self-signed cert + key in a tempdir and load them
    /// back. Tests downstream (the TLS handshake test in lib.rs) reuse
    /// this helper.
    pub(crate) fn self_signed_pair() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        use std::io::Write;
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        File::create(&cert_path)
            .unwrap()
            .write_all(cert.cert.pem().as_bytes())
            .unwrap();
        File::create(&key_path)
            .unwrap()
            .write_all(cert.key_pair.serialize_pem().as_bytes())
            .unwrap();
        (dir, cert_path, key_path)
    }

    #[test]
    fn loads_self_signed_cert() {
        let (_dir, cert, key) = self_signed_pair();
        let cfg = load_server_config(&cert, &key).expect("load");
        // ServerConfig is opaque; just confirm we got an Arc out.
        assert!(Arc::strong_count(&cfg) >= 1);
    }

    #[test]
    fn missing_cert_file_errors() {
        let err = load_server_config("/nonexistent.pem", "/nonexistent.key").unwrap_err();
        assert!(err.to_string().contains("opening cert file"));
    }
}
