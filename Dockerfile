# DuxxDB Dockerfile (Linux x86_64).
#
# Multi-stage:
# 1. `builder` — full Rust toolchain, builds release binaries.
# 2. `runtime` — minimal debian-slim with just the binary + ca-certs
#    (so the OpenAI/Cohere HTTP clients can verify TLS certs).
#
# Build:  docker build -t duxxdb:0.1.0 .
# Run:    docker run --rm -p 6379:6379 \
#           -e DUXX_EMBEDDER=hash:32 \
#           duxxdb:0.1.0
#
# With OpenAI:
#   docker run --rm -p 6379:6379 \
#     -e DUXX_EMBEDDER=openai:text-embedding-3-small \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     duxxdb:0.1.0

# ---------------------------------------------------------------- builder ----
FROM rust:1.83-bookworm AS builder

WORKDIR /usr/src/duxxdb

# Cache deps separately from source. Copy manifests first.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY bindings ./bindings
COPY rust-toolchain.toml* ./

# Pre-fetch the dep graph.
RUN cargo fetch

# Real build.
RUN cargo build --release \
      -p duxx-server \
      -p duxx-mcp \
      --features duxx-embed/http

# ---------------------------------------------------------------- runtime ----
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Bring just the binaries.
COPY --from=builder /usr/src/duxxdb/target/release/duxx-server /usr/local/bin/duxx-server
COPY --from=builder /usr/src/duxxdb/target/release/duxx-mcp    /usr/local/bin/duxx-mcp

# Non-root user for safety.
RUN useradd --system --uid 10001 --shell /usr/sbin/nologin duxx
USER duxx

# Default config; overrideable via -e at run time.
ENV DUXX_EMBEDDER=hash:32 \
    RUST_LOG=info

EXPOSE 6379

# Default command is the RESP server. Override CMD to run duxx-mcp instead.
ENTRYPOINT ["/usr/local/bin/duxx-server"]
CMD ["--addr", "0.0.0.0:6379"]
