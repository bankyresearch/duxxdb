#!/usr/bin/env bash
# One-command reproducer for the DuxxDB comparative benchmark.
#
#   ./run.sh --quick          # embedded DuxxDB + exact reference; no services
#   ./run.sh                  # full suite: brings up redis/qdrant/pgvector +
#                             # the DuxxDB gRPC daemon (disk-backed), runs, tears down
#   ./run.sh --dims 128,768 --n 50000 --queries 500
#
# Results are printed as markdown and written to results-<timestamp>.json.
# See docs/BENCHMARKS.md for the methodology and hardware-reporting rules.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

QUICK=0
PASSTHRU=()
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1; PASSTHRU+=("$a") ;;
    *) PASSTHRU+=("$a") ;;
  esac
done

TS="$(date +%Y%m%d-%H%M%S)"
OUT="results-${TS}.json"

python -c "import numpy" 2>/dev/null || { echo "installing bench deps..."; pip install -q numpy; }

if [[ "$QUICK" == "1" ]]; then
  echo ">> quick mode: embedded DuxxDB + reference only"
  exec python bench.py "${PASSTHRU[@]}" --out "$OUT"
fi

# ---- full suite: stand up the network peers ----
echo ">> starting redis / qdrant / pgvector via docker compose"
docker compose up -d redis-stack qdrant pgvector

echo ">> waiting for services to accept connections"
for i in $(seq 1 60); do
  ready=1
  python - <<'PY' 2>/dev/null || ready=0
import socket
for host, port in [("localhost",6379),("localhost",6333),("localhost",5432)]:
    s = socket.create_connection((host, port), timeout=1); s.close()
PY
  [[ "$ready" == "1" ]] && break
  sleep 2
done

# ---- build + launch the DuxxDB gRPC daemon (disk-backed) ----
echo ">> building duxx-grpc (release)"
( cd "$ROOT" && cargo build --release -p duxx-grpc )
DUXX_DATA="$(mktemp -d)"
echo ">> launching duxx-grpc on 127.0.0.1:50051 (storage dir:$DUXX_DATA)"
"$ROOT/target/release/duxx-grpc" --addr 127.0.0.1:50051 --storage "dir:$DUXX_DATA" &
DUXX_PID=$!
cleanup() {
  echo ">> tearing down"
  kill "$DUXX_PID" 2>/dev/null || true
  docker compose down -v || true
  rm -rf "$DUXX_DATA" || true
}
trap cleanup EXIT
sleep 3

pip install -q redis qdrant-client psycopg2-binary grpcio 2>/dev/null || true

echo ">> running benchmark"
python bench.py "${PASSTHRU[@]}" --out "$OUT"
echo ">> wrote $OUT"
