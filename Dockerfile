# DuxxDB Dockerfile (Linux x86_64 / aarch64).
#
# Multi-stage:
#   1. `builder`  — full Rust toolchain, builds release binaries.
#   2. `runtime`  — minimal debian-slim with binaries + ca-certs
#                  (so the OpenAI/Cohere HTTPS clients can verify TLS).
#
# Build:
#   docker build -t duxxdb:latest .
#
# Smoke run (in-memory only; lost on exit):
#   docker run --rm -p 6379:6379 duxxdb:latest
#
# Persistent quickstart (rows + indices on a host volume):
#   docker run --rm -p 6379:6379 \
#     -v duxxdb-data:/var/lib/duxxdb \
#     -e DUXX_STORAGE=dir:/var/lib/duxxdb \
#     duxxdb:latest
#
# With auth + Prometheus metrics (Phase 6.1):
#   docker run --rm -p 6379:6379 -p 9100:9100 \
#     -v duxxdb-data:/var/lib/duxxdb \
#     -e DUXX_STORAGE=dir:/var/lib/duxxdb \
#     -e DUXX_TOKEN=$YOUR_TOKEN \
#     -e DUXX_METRICS_ADDR=0.0.0.0:9100 \
#     duxxdb:latest
#
# For a complete production stack with persistent volumes, healthchecks,
# and Prometheus scrape config, see packaging/docker/docker-compose.yml.

# ---------------------------------------------------------------- builder ----
FROM rust:1.83-bookworm AS builder

WORKDIR /usr/src/duxxdb

# protoc is required by duxx-grpc's build.rs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

# Cache deps separately from source. Copy manifests first.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY bindings ./bindings
COPY rust-toolchain.toml* ./

# Pre-fetch the dep graph.
RUN cargo fetch

# Build everything ops needs at runtime: RESP, gRPC, MCP, cold-tier export.
RUN cargo build --release \
      -p duxx-server \
      -p duxx-grpc \
      -p duxx-mcp \
      -p duxx-coldtier \
      --features duxx-embed/http

# ---------------------------------------------------------------- runtime ----
FROM debian:bookworm-slim AS runtime

# ca-certificates  -- TLS for OpenAI / Cohere HTTPS embedders.
# tini             -- proper PID 1 + signal forwarding (graceful shutdown).
# redis-tools      -- ships redis-cli for HEALTHCHECK and on-host debug.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      tini \
      redis-tools \
 && rm -rf /var/lib/apt/lists/*

# Bring all four binaries.
COPY --from=builder /usr/src/duxxdb/target/release/duxx-server   /usr/local/bin/duxx-server
COPY --from=builder /usr/src/duxxdb/target/release/duxx-grpc     /usr/local/bin/duxx-grpc
COPY --from=builder /usr/src/duxxdb/target/release/duxx-mcp      /usr/local/bin/duxx-mcp
COPY --from=builder /usr/src/duxxdb/target/release/duxx-export   /usr/local/bin/duxx-export

# Non-root user for safety. UID matches packaging/debian/postinst.
RUN useradd --system --uid 10001 --home /var/lib/duxxdb --shell /usr/sbin/nologin duxxdb \
 && mkdir -p /var/lib/duxxdb /etc/duxxdb \
 && chown -R duxxdb:duxxdb /var/lib/duxxdb /etc/duxxdb

# Default config; overrideable via -e at run time.
ENV DUXX_EMBEDDER=hash:32 \
    RUST_LOG=info

USER duxxdb
WORKDIR /var/lib/duxxdb

EXPOSE 6379 9100

# HEALTHCHECK uses redis-cli PING; works pre-AUTH thanks to Phase 6.1
# allowlist (PING is permitted before AUTH).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD redis-cli -h 127.0.0.1 -p 6379 PING | grep -q PONG || exit 1

# tini reaps zombies and forwards SIGTERM/SIGINT to duxx-server, which
# triggers the Phase 6.1 graceful-shutdown drain.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/duxx-server"]
CMD ["--addr", "0.0.0.0:6379"]
