# Security Policy

## Supported Versions

Security fixes target the latest released minor version of DuxxDB. If
you operate DuxxDB in production, pin an explicit release tag and plan
to upgrade promptly when a security release is published.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub private vulnerability reporting or open a private security
advisory for this repository. Include:

- The affected version, commit, or container tag.
- A short impact summary.
- Reproduction steps or a minimal proof of concept.
- Whether the issue is already public.

We aim to acknowledge new reports within 3 business days and will keep
the reporter updated as the fix is triaged, patched, and released.

## Disclosure

Coordinated disclosure is expected. Public details should wait until a
fixed release or mitigation guidance is available, unless active
exploitation requires earlier operator notice.

## Production Guidance

For network-exposed deployments, enable TLS, token authentication,
persistent storage, metrics on a private interface, and memory caps.
Do not expose unauthenticated RESP, gRPC, metrics, or health endpoints
to the public internet.
