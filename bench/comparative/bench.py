"""Comparative benchmark — DuxxDB vs LanceDB on identical workload.

Both targets run in the same Python process, against the same generated
corpus, on the same machine. No network in the embedded comparison —
keeps the numbers honest.

Usage:
    python bench.py                      # all targets, defaults
    python bench.py --n 5000 --queries 200 --dim 128
    python bench.py --backend duxxdb     # single target

For network targets (Redis Stack, Qdrant, pgvector) see
docker-compose.yml + the TODO stubs at the bottom of this file.
"""

import argparse
import random
import statistics
import time
from typing import Protocol

random.seed(42)


# ----------------------------------------------------------------------------
# Workload generation
# ----------------------------------------------------------------------------

def random_vec(dim: int) -> list[float]:
    """L2-normalized random Gaussian vector."""
    v = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


KEYS = ["alice", "bob", "charlie", "dave", "eve"]
TOPICS = [
    "refund", "order", "delivery", "wallet", "weather",
    "color", "tracking", "package", "invoice", "support",
    "billing", "discount", "schedule", "appointment", "feedback",
]


def generate_corpus(n: int, dim: int):
    out = []
    for i in range(n):
        key = KEYS[i % len(KEYS)]
        text = " ".join(random.sample(TOPICS, k=3)) + f" doc-{i}"
        out.append((key, text, random_vec(dim)))
    return out


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

class Backend(Protocol):
    name: str
    def open(self, dim: int) -> None: ...
    def insert(self, key: str, text: str, embedding: list[float]) -> None: ...
    def recall(self, key: str, query: str, embedding: list[float], k: int) -> int: ...
    def close(self) -> None: ...


class DuxxDBBackend:
    name = "duxxdb"

    def open(self, dim: int) -> None:
        import duxxdb
        self.store = duxxdb.MemoryStore(dim)

    def insert(self, key, text, embedding):
        self.store.remember(key=key, text=text, embedding=embedding)

    def recall(self, key, query, embedding, k):
        return len(self.store.recall(key=key, query=query,
                                     embedding=embedding, k=k))

    def close(self):
        pass


class LanceDBBackend:
    name = "lancedb"

    def open(self, dim: int) -> None:
        import os
        import shutil
        import lancedb
        self.tmp = os.path.join(os.path.dirname(__file__), "_lancedb_bench_tmp")
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)
        self.db = lancedb.connect(self.tmp)
        self.tbl = None
        self.dim = dim

    def insert(self, key, text, embedding):
        row = {"key": key, "text": text, "vector": embedding}
        if self.tbl is None:
            self.tbl = self.db.create_table("memories", data=[row])
        else:
            self.tbl.add([row])

    def recall(self, key, query, embedding, k):
        return len(
            self.tbl.search(embedding).limit(k).to_list()
        )

    def close(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class RedisStackBackend:
    """Redis Stack with RediSearch HNSW vector index."""
    name = "redis"

    def open(self, dim: int) -> None:
        import redis
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        self.r = redis.Redis(host="localhost", port=6379)
        self.r.ping()
        try:
            self.r.ft("idx:memories").dropindex(delete_documents=True)
        except Exception:
            pass
        try:
            for k in self.r.keys("mem:*"):
                self.r.delete(k)
        except Exception:
            pass
        schema = (
            TextField("$.key", as_name="key"),
            TextField("$.text", as_name="text"),
            VectorField(
                "$.vector",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                as_name="vector",
            ),
        )
        self.r.ft("idx:memories").create_index(
            schema,
            definition=IndexDefinition(prefix=["mem:"], index_type=IndexType.JSON),
        )
        self.dim = dim
        self.next_id = 1

    def insert(self, key, text, embedding):
        self.r.json().set(
            f"mem:{self.next_id}",
            "$",
            {"key": key, "text": text, "vector": [float(x) for x in embedding]},
        )
        self.next_id += 1

    def recall(self, key, query, embedding, k):
        import numpy as np
        from redis.commands.search.query import Query

        vec = np.asarray(embedding, dtype=np.float32).tobytes()
        q = (
            Query(f"*=>[KNN {k} @vector $vec AS dist]")
            .sort_by("dist")
            .return_fields("dist")
            .dialect(2)
        )
        res = self.r.ft("idx:memories").search(q, query_params={"vec": vec})
        return len(res.docs)

    def close(self):
        try:
            self.r.ft("idx:memories").dropindex(delete_documents=True)
        except Exception:
            pass
        try:
            self.r.close()
        except Exception:
            pass


class QdrantBackend:
    name = "qdrant"

    def open(self, dim: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host="localhost", port=6333)
        try:
            self.client.delete_collection("memories")
        except Exception:
            pass
        self.client.create_collection(
            collection_name="memories",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self.next_id = 1

    def insert(self, key, text, embedding):
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name="memories",
            points=[
                PointStruct(
                    id=self.next_id,
                    vector=list(embedding),
                    payload={"key": key, "text": text},
                )
            ],
        )
        self.next_id += 1

    def recall(self, key, query, embedding, k):
        hits = self.client.search(
            collection_name="memories",
            query_vector=list(embedding),
            limit=k,
        )
        return len(hits)

    def close(self):
        try:
            self.client.delete_collection("memories")
        except Exception:
            pass


class PgVectorBackend:
    name = "pgvector"

    def open(self, dim: int) -> None:
        import psycopg2

        self.conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="bench",
            dbname="duxx_bench",
        )
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS memories")
            cur.execute(
                "CREATE TABLE memories ("
                "id BIGSERIAL PRIMARY KEY, "
                "key TEXT, "
                "text TEXT, "
                f"vector vector({dim})"
                ")"
            )
            cur.execute(
                "CREATE INDEX ON memories USING hnsw (vector vector_cosine_ops)"
            )

    def insert(self, key, text, embedding):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (key, text, vector) VALUES (%s, %s, %s)",
                (key, text, str(list(embedding))),
            )

    def recall(self, key, query, embedding, k):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM memories ORDER BY vector <=> %s::vector LIMIT %s",
                (str(list(embedding)), k),
            )
            return len(cur.fetchall())

    def close(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS memories")
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass


class DuxxGrpcBackend:
    """DuxxDB through the gRPC daemon — apples-to-apples network parity
    with the other network backends."""
    name = "duxx-grpc"

    def open(self, dim: int) -> None:
        import os
        import sys

        # Add the generated proto stubs to path.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_proto"))
        import duxx_pb2 as pb
        import duxx_pb2_grpc as svc
        import grpc

        self.pb = pb
        self.svc = svc
        self.channel = grpc.insecure_channel("localhost:50051")
        self.client = svc.DuxxStub(self.channel)
        # Liveness check
        self.client.Ping(pb.PingRequest(nonce="hi"))

    def insert(self, key, text, embedding):
        self.client.Remember(
            self.pb.RememberRequest(key=key, text=text, embedding=list(embedding))
        )

    def recall(self, key, query, embedding, k):
        r = self.client.Recall(
            self.pb.RecallRequest(
                key=key, query=query, embedding=list(embedding), k=k
            )
        )
        return len(r.hits)

    def close(self):
        try:
            self.channel.close()
        except Exception:
            pass


BACKENDS = {
    b.name: b
    for b in [
        DuxxDBBackend,
        LanceDBBackend,
        RedisStackBackend,
        QdrantBackend,
        PgVectorBackend,
        DuxxGrpcBackend,
    ]
}


# ----------------------------------------------------------------------------
# Bench loop
# ----------------------------------------------------------------------------

def run_one(backend_cls, n: int, queries: int, dim: int) -> dict:
    backend = backend_cls()
    backend.open(dim)

    corpus = generate_corpus(n, dim)
    query_set = [
        (random.choice(KEYS), random.choice(TOPICS) + " " + random.choice(TOPICS),
         random_vec(dim))
        for _ in range(queries)
    ]

    # Insert phase
    insert_times = []
    t_insert_start = time.perf_counter()
    for key, text, vec in corpus:
        t0 = time.perf_counter()
        backend.insert(key, text, vec)
        insert_times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    t_insert_total = time.perf_counter() - t_insert_start

    # Recall phase
    recall_times = []
    for key, query_text, q_vec in query_set:
        t0 = time.perf_counter()
        backend.recall(key, query_text, q_vec, 10)
        recall_times.append((time.perf_counter() - t0) * 1e6)

    backend.close()

    insert_sorted = sorted(insert_times)
    recall_sorted = sorted(recall_times)
    return {
        "backend": backend.name,
        "n_insert": n,
        "n_query": queries,
        "dim": dim,
        "insert_us_p50": statistics.median(insert_times),
        "insert_us_p99": insert_sorted[int(len(insert_sorted) * 0.99) - 1],
        "insert_us_mean": statistics.mean(insert_times),
        "insert_total_s": t_insert_total,
        "insert_throughput": n / t_insert_total,
        "recall_us_p50": statistics.median(recall_times),
        "recall_us_p99": recall_sorted[int(len(recall_sorted) * 0.99) - 1],
        "recall_us_mean": statistics.mean(recall_times),
    }


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    if not results:
        return
    n = results[0]["n_insert"]
    q = results[0]["n_query"]
    dim = results[0]["dim"]
    print()
    print(f"## Workload: N={n}, Q={q}, dim={dim}")
    print()
    print(
        "| backend  | ins p50 (µs) | ins p99 (µs) | ins/s     "
        "| rec p50 (µs) | rec p99 (µs) |"
    )
    print(
        "|----------|-------------:|-------------:|----------:"
        "|-------------:|-------------:|"
    )
    for r in results:
        print(
            f"| {r['backend']:<8} "
            f"| {r['insert_us_p50']:>12.1f} "
            f"| {r['insert_us_p99']:>12.1f} "
            f"| {r['insert_throughput']:>9.0f} "
            f"| {r['recall_us_p50']:>12.1f} "
            f"| {r['recall_us_p99']:>12.1f} |"
        )
    print()


# ----------------------------------------------------------------------------
# Network-target stubs (TODO)
# ----------------------------------------------------------------------------
#
# To extend this harness to network-served systems:
#
# 1. Spin up the systems via docker-compose:
#       docker compose -f docker-compose.yml up -d
#
# 2. Add a backend class that wraps each one's Python client:
#
#       class RedisStackBackend:
#           name = "redis"
#           def open(self, dim):
#               import redis
#               self.r = redis.Redis(host="localhost", port=6379)
#               # FT.CREATE memory SCHEMA text TEXT vector VECTOR HNSW ...
#           def insert(self, ...): ...
#           def recall(self, ...): ...
#
#       class QdrantBackend:
#           name = "qdrant"
#           # qdrant-client.QdrantClient(host="localhost", port=6333)
#
#       class PgVectorBackend:
#           name = "pgvector"
#           # psycopg2 + vector ext
#
# 3. Add to BACKENDS dict and re-run.
# Network targets will show wall-clock latency that includes the
# per-RPC overhead. For DuxxDB-network parity, also bench duxx-grpc
# via grpcio in a third backend.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=[*BACKENDS, "all"], default="all")
    ap.add_argument("-n", type=int, default=1000, help="documents to insert")
    ap.add_argument("-q", "--queries", type=int, default=100,
                    help="queries to run")
    ap.add_argument("-d", "--dim", type=int, default=128,
                    help="embedding dimension")
    args = ap.parse_args()

    targets = list(BACKENDS) if args.backend == "all" else [args.backend]
    results = []
    for name in targets:
        try:
            r = run_one(BACKENDS[name], args.n, args.queries, args.dim)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"\n[{name}] ERROR: {e}")
            traceback.print_exc()

    print_table(results)


if __name__ == "__main__":
    main()
