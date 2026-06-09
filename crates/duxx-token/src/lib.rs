//! # duxx-token — signed short-lived workspace credentials
//!
//! The control plane (`duxx-control`) **signs** a JWT scoping a caller to a
//! workspace `(org, project, env)` with a role; the data plane (`duxx-server`)
//! **verifies the signature** and resolves the namespace from the claims. This
//! is the forward path from static `--auth-key` catalog materialization to
//! short-lived, revocable-by-expiry credentials.
//!
//! Crypto is HS256 via the maintained [`jsonwebtoken`] crate — a shared secret
//! between the control plane and its data-plane nodes. (Asymmetric RS256/ES256,
//! where nodes hold only the public key, is a drop-in upgrade: swap the
//! `EncodingKey`/`DecodingKey` constructors.)
//!
//! Both sides depend on this crate, so the claim shape can never drift.

use serde::{Deserialize, Serialize};

/// Claims carried by a workspace credential. The `org`/`project`/`env` triple
/// is exactly what the data plane's `Namespace::parse` consumes (joined with
/// `/`), and `role` is a data-plane role string (`developer`, `service`, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — the issuing API key / principal id.
    pub sub: String,
    pub org: String,
    pub project: String,
    /// `dev` | `staging` | `prod`.
    pub env: String,
    pub role: String,
    /// Expiry, unix seconds. Validated automatically on `verify`.
    pub exp: usize,
    /// Issued-at, unix seconds.
    pub iat: usize,
}

impl Claims {
    /// Build claims valid for `ttl_secs` from `now_unix` (seconds since the
    /// epoch — passed in so this stays free of ambient clock reads and is
    /// trivially testable).
    pub fn new(
        sub: impl Into<String>,
        org: impl Into<String>,
        project: impl Into<String>,
        env: impl Into<String>,
        role: impl Into<String>,
        now_unix: u64,
        ttl_secs: u64,
    ) -> Self {
        Self {
            sub: sub.into(),
            org: org.into(),
            project: project.into(),
            env: env.into(),
            role: role.into(),
            iat: now_unix as usize,
            exp: now_unix.saturating_add(ttl_secs) as usize,
        }
    }

    /// The data-plane tenant field: `org/project/env`.
    pub fn tenant(&self) -> String {
        format!("{}/{}/{}", self.org, self.project, self.env)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("token sign failed: {0}")]
    Sign(String),
    #[error("token verify failed: {0}")]
    Verify(String),
}

/// Sign `claims` with the shared `secret` (HS256).
pub fn sign(claims: &Claims, secret: &[u8]) -> Result<String, TokenError> {
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);
    let key = jsonwebtoken::EncodingKey::from_secret(secret);
    jsonwebtoken::encode(&header, claims, &key).map_err(|e| TokenError::Sign(e.to_string()))
}

/// Verify a token's signature with `secret` and return its claims. Expiry is
/// validated automatically; a token whose `exp` has passed fails here.
pub fn verify(token: &str, secret: &[u8]) -> Result<Claims, TokenError> {
    let key = jsonwebtoken::DecodingKey::from_secret(secret);
    let validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);
    jsonwebtoken::decode::<Claims>(token, &key, &validation)
        .map(|data| data.claims)
        .map_err(|e| TokenError::Verify(e.to_string()))
}

// ---------------------------------------------------------------------------
// Asymmetric (EdDSA / Ed25519): the control plane holds the private key and
// signs; data-plane nodes hold only the PUBLIC key and verify. This removes
// the shared-secret weakness of HS256 — a compromised node cannot mint tokens.
// ---------------------------------------------------------------------------

/// Generate an Ed25519 keypair. Returns `(private_pkcs8_der, public_der)` —
/// give the private bytes to the control plane (signing) and the public bytes
/// to every data-plane node (verifying).
pub fn generate_ed25519() -> Result<(Vec<u8>, Vec<u8>), TokenError> {
    use ring::signature::{Ed25519KeyPair, KeyPair};
    let rng = ring::rand::SystemRandom::new();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|e| TokenError::Sign(format!("ed25519 keygen: {e}")))?;
    let kp = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|e| TokenError::Sign(format!("ed25519 keypair: {e}")))?;
    Ok((pkcs8.as_ref().to_vec(), kp.public_key().as_ref().to_vec()))
}

/// Sign with an Ed25519 private key (PKCS#8 DER from [`generate_ed25519`]).
pub fn sign_ed25519(claims: &Claims, private_pkcs8_der: &[u8]) -> Result<String, TokenError> {
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::EdDSA);
    let key = jsonwebtoken::EncodingKey::from_ed_der(private_pkcs8_der);
    jsonwebtoken::encode(&header, claims, &key).map_err(|e| TokenError::Sign(e.to_string()))
}

/// Verify with an Ed25519 public key (raw DER from [`generate_ed25519`]).
pub fn verify_ed25519(token: &str, public_der: &[u8]) -> Result<Claims, TokenError> {
    let key = jsonwebtoken::DecodingKey::from_ed_der(public_der);
    let validation = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::EdDSA);
    jsonwebtoken::decode::<Claims>(token, &key, &validation)
        .map(|data| data.claims)
        .map_err(|e| TokenError::Verify(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const SECRET: &[u8] = b"super-secret-signing-key-0123456789";

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn sign_then_verify_round_trips() {
        // `exp` is validated against the real clock, so a *valid* token must
        // use real time for `now`.
        let claims = Claims::new("key_1", "org_1", "proj_a", "prod", "developer", now(), 3600);
        let token = sign(&claims, SECRET).unwrap();
        let got = verify(&token, SECRET).unwrap();
        assert_eq!(got, claims);
        assert_eq!(got.tenant(), "org_1/proj_a/prod");
    }

    #[test]
    fn wrong_secret_fails_verification() {
        let claims = Claims::new("k", "o", "p", "prod", "service", 1_000_000, 3600);
        let token = sign(&claims, SECRET).unwrap();
        assert!(verify(&token, b"a-different-secret-key-9999").is_err());
    }

    #[test]
    fn expired_token_is_rejected() {
        // Issued far in the past with a tiny TTL → already expired.
        let claims = Claims::new("k", "o", "p", "prod", "developer", 1_000, 1);
        let token = sign(&claims, SECRET).unwrap();
        let err = verify(&token, SECRET).unwrap_err();
        assert!(matches!(err, TokenError::Verify(_)));
    }

    #[test]
    fn ed25519_asymmetric_round_trips() {
        let (priv_der, pub_der) = generate_ed25519().unwrap();
        let claims = Claims::new("k", "o", "p", "prod", "developer", now(), 3600);
        let token = sign_ed25519(&claims, &priv_der).unwrap();
        // The public key verifies a token the private key signed.
        assert_eq!(verify_ed25519(&token, &pub_der).unwrap(), claims);
        // A different keypair's public key does not.
        let (_, other_pub) = generate_ed25519().unwrap();
        assert!(verify_ed25519(&token, &other_pub).is_err());
        // And HS256 verify can't accept an EdDSA token.
        assert!(verify(&token, b"some-hmac-secret").is_err());
    }

    #[test]
    fn tampered_token_fails() {
        let claims = Claims::new("k", "o", "p", "prod", "observer", 1_000_000, 3600);
        let mut token = sign(&claims, SECRET).unwrap();
        // Flip a character in the payload section.
        token.insert(token.len() / 2, 'X');
        assert!(verify(&token, SECRET).is_err());
    }
}
