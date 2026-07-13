"""Market-grade comparative benchmark for DuxxDB.

Measures three things the old microbenchmark did not:

  1. **Retrieval quality** — recall@k and nDCG@k against an *exact* cosine
     ground truth, on a clustered corpus where nearest neighbours are real
     (uniform-random vectors make every ANN index look either perfect or
     random, which measures nothing).
  2. **Sustained concurrency** — p50/p95/p99 latency and achieved QPS under
     N concurrent clients, not one sequential caller.
  3. **Fair configuration** — DuxxDB is run **disk-backed** (`open_at`) so it
     is compared like-for-like against disk-backed Redis / Qdrant / pgvector,
     not in-memory-vs-disk. Every system is queried in the mode it ships.

Design constraints that keep the numbers honest:

  * Doc ids are the *corpus index*, threaded through every backend, so a
    system's returned ids map back to ground truth exactly.
  * A pure-numpy `reference` backend (exact brute force) is always available.
    It needs no services, scores recall@k == 1.0 by construction, and is the
    self-check that the harness math is right.
  * DuxxDB recall is **hybrid** (vector + BM25 + RRF) — its real product
    behaviour. Ground truth is pure cosine, so this can only *understate*
    DuxxDB's vector recall, never inflate it. Stated, not hidden.
  * In-process backends (embedded DuxxDB, reference) hold the GIL during a
    call, so their concurrency numbers do not reflect true parallelism — they
    are reported single-client for latency only. The network backends
    (`duxx-grpc`, redis, qdrant, pgvector) carry the concurrency comparison.

Usage:
    python bench.py --quick                     # embedded + reference, no services
    python bench.py                             # full suite, all reachable backends
    python bench.py --dims 768,1536 --n 50000 --queries 500 --concurrency 1,8,32
    python bench.py --backend duxxdb --out results.json

See docs/BENCHMARKS.md for the methodology and how to reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

SEED = 42
GT_DEPTH = 100  # ground-truth depth kept per query (>= any evaluated k)


# ---------------------------------------------------------------------------
# Workload: a *clustered* corpus so nearest neighbours are meaningful.
# ---------------------------------------------------------------------------

KEYS = ["alice", "bob", "charlie", "dave", "eve"]
TOPICS = [
    "refund", "order", "delivery", "wallet", "weather",
    "color", "tracking", "package", "invoice", "support",
    "billing", "discount", "schedule", "appointment", "feedback",
]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def generate_corpus(n: int, dim: int, rng: np.random.Generator):
    """Gaussian-mixture corpus: `clusters` centroids, points scattered around
    them. Returns (vectors[n,dim] float32, texts[n], keys[n])."""
    clusters = max(8, n // 500)
    centroids = _normalize(rng.standard_normal((clusters, dim)).astype(np.float32))
    assign = rng.integers(0, clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.35
    vecs = _normalize(centroids[assign] + noise).astype(np.float32)

    texts, keys = [], []
    for i in range(n):
        # Topic words correlate with cluster so BM25 has real signal too.
        base = TOPICS[assign[i] % len(TOPICS)]
        extra = TOPICS[(assign[i] * 7 + i) % len(TOPICS)]
        texts.append(f"{base} {extra} doc-{i}")
        keys.append(KEYS[i % len(KEYS)])
    return vecs, texts, keys


def generate_queries(n_queries: int, corpus_vecs: np.ndarray, texts, keys,
                     rng: np.random.Generator):
    """Queries are perturbed corpus points (a realistic 'find similar' load)."""
    idx = rng.integers(0, len(corpus_vecs), size=n_queries)
    perturb = rng.standard_normal(corpus_vecs[idx].shape).astype(np.float32) * 0.15
    q_vecs = _normalize(corpus_vecs[idx] + perturb).astype(np.float32)
    q_texts = [texts[i].rsplit(" doc-", 1)[0] for i in idx]  # topic words only
    q_keys = [keys[i] for i in idx]
    return q_vecs, q_texts, q_keys


def exact_ground_truth(corpus_vecs: np.ndarray, q_vecs: np.ndarray, depth: int):
    """Exact cosine top-`depth` per query. Vectors are unit-norm, so cosine is
    the dot product. Returns an int array [n_queries, depth] of corpus ids."""
    depth = min(depth, corpus_vecs.shape[0])
    out = np.empty((q_vecs.shape[0], depth), dtype=np.int64)
    # Chunk the query set to bound the similarity matrix memory.
    chunk = max(1, 2_000_000 // max(1, corpus_vecs.shape[0]))
    for start in range(0, q_vecs.shape[0], chunk):
        block = q_vecs[start:start + chunk] @ corpus_vecs.T
        part = np.argpartition(-block, depth - 1, axis=1)[:, :depth]
        order = np.argsort(-np.take_along_axis(block, part, axis=1), axis=1)
        out[start:start + chunk] = np.take_along_axis(part, order, axis=1)
    return out


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def recall_at_k(returned: list[int], gt_topk: set[int], k: int) -> float:
    if not gt_topk:
        return 0.0
    hit = sum(1 for d in returned[:k] if d in gt_topk)
    return hit / min(k, len(gt_topk))


def ndcg_at_k(returned: list[int], gt_topk: set[int], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, d in enumerate(returned[:k]) if d in gt_topk
    )
    ideal_n = min(k, len(gt_topk))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return (dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Backends — insert(doc_id,...), query(...) -> ordered list[int] of corpus ids
# ---------------------------------------------------------------------------

class Backend:
    name = "base"
    in_process = True  # in-process backends skip the concurrency sweep

    def open(self, dim: int) -> None: ...
    def insert(self, doc_id: int, key: str, text: str, embedding) -> None: ...
    def query(self, key: str, text: str, embedding, k: int) -> list[int]: ...
    def close(self) -> None: ...


class ReferenceBackend(Backend):
    """Exact brute-force cosine. Self-check: recall@k must be 1.0."""
    name = "reference"

    def open(self, dim):
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []

    def insert(self, doc_id, key, text, embedding):
        self._ids.append(doc_id)
        self._vecs.append(np.asarray(embedding, dtype=np.float32))

    def _finalize(self):
        self._mat = _normalize(np.vstack(self._vecs))
        self._id_arr = np.asarray(self._ids)

    def query(self, key, text, embedding, k):
        if not hasattr(self, "_mat"):
            self._finalize()
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._mat @ q
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return self._id_arr[top].tolist()

    def close(self):
        pass


class DuxxDBBackend(Backend):
    """Embedded DuxxDB, **disk-backed** (open_at) for a fair comparison
    against the disk-backed network stores."""
    name = "duxxdb"

    def open(self, dim):
        import duxxdb
        self._dir = tempfile.mkdtemp(prefix="duxx_bench_")
        try:
            self.store = duxxdb.MemoryStore.open_at(
                dim=dim, capacity=1_000_000, dir=self._dir
            )
        except Exception:
            # Older wheels without open_at: fall back to in-memory, but say so.
            print("  [duxxdb] open_at unavailable; running IN-MEMORY (not disk-fair)",
                  file=sys.stderr)
            self.store = duxxdb.MemoryStore(dim)
        self._map: dict[int, int] = {}  # internal id -> corpus id

    def insert(self, doc_id, key, text, embedding):
        rid = self.store.remember(key=key, text=text, embedding=list(embedding))
        self._map[int(rid)] = doc_id

    def query(self, key, text, embedding, k):
        hits = self.store.recall(key=key, query=text, embedding=list(embedding), k=k)
        return [self._map.get(int(h.id), -1) for h in hits]

    def close(self):
        self.store = None
        shutil.rmtree(getattr(self, "_dir", ""), ignore_errors=True)


class DuxxGrpcBackend(Backend):
    """DuxxDB through the gRPC daemon — network parity with the other
    served systems (this is the one that carries the concurrency sweep)."""
    name = "duxx-grpc"
    in_process = False

    def open(self, dim):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_proto"))
        import duxx_pb2 as pb
        import duxx_pb2_grpc as svc
        import grpc

        self.pb, self.svc = pb, svc
        self.channel = grpc.insecure_channel("localhost:50051")
        self.client = svc.DuxxStub(self.channel)
        self.client.Ping(pb.PingRequest(nonce="hi"))
        self._map: dict[int, int] = {}

    def insert(self, doc_id, key, text, embedding):
        r = self.client.Remember(
            self.pb.RememberRequest(key=key, text=text, embedding=list(embedding))
        )
        self._map[int(r.id)] = doc_id

    def query(self, key, text, embedding, k):
        r = self.client.Recall(
            self.pb.RecallRequest(key=key, query=text, embedding=list(embedding), k=k)
        )
        return [self._map.get(int(h.id), -1) for h in r.hits]

    def close(self):
        try:
            self.channel.close()
        except Exception:
            pass


class RedisStackBackend(Backend):
    """Redis Stack, RediSearch HNSW, KNN vector query."""
    name = "redis"
    in_process = False

    def open(self, dim):
        import redis
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        self.r = redis.Redis(host="localhost", port=6379)
        self.r.ping()
        try:
            self.r.ft("idx:memories").dropindex(delete_documents=True)
        except Exception:
            pass
        schema = (
            TextField("$.key", as_name="key"),
            TextField("$.text", as_name="text"),
            VectorField("$.vector", "HNSW",
                        {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                        as_name="vector"),
        )
        self.r.ft("idx:memories").create_index(
            schema,
            definition=IndexDefinition(prefix=["mem:"], index_type=IndexType.JSON),
        )

    def insert(self, doc_id, key, text, embedding):
        self.r.json().set(f"mem:{doc_id}", "$",
                          {"key": key, "text": text,
                           "vector": [float(x) for x in embedding]})

    def query(self, key, text, embedding, k):
        from redis.commands.search.query import Query
        vec = np.asarray(embedding, dtype=np.float32).tobytes()
        q = (Query(f"*=>[KNN {k} @vector $vec AS dist]")
             .sort_by("dist").return_fields("dist").dialect(2))
        res = self.r.ft("idx:memories").search(q, query_params={"vec": vec})
        return [int(d.id.split(":", 1)[1]) for d in res.docs]

    def close(self):
        try:
            self.r.ft("idx:memories").dropindex(delete_documents=True)
            self.r.close()
        except Exception:
            pass


class QdrantBackend(Backend):
    name = "qdrant"
    in_process = False

    def open(self, dim):
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

    def insert(self, doc_id, key, text, embedding):
        from qdrant_client.models import PointStruct
        self.client.upsert(collection_name="memories",
                           points=[PointStruct(id=doc_id, vector=list(embedding),
                                               payload={"key": key, "text": text})])

    def query(self, key, text, embedding, k):
        hits = self.client.search(collection_name="memories",
                                  query_vector=list(embedding), limit=k)
        return [int(h.id) for h in hits]

    def close(self):
        try:
            self.client.delete_collection("memories")
        except Exception:
            pass


class PgVectorBackend(Backend):
    name = "pgvector"
    in_process = False

    def open(self, dim):
        import threading
        self._local = threading.local()  # a psycopg2 conn is not thread-safe
        with self._conn().cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS memories")
            cur.execute(f"CREATE TABLE memories (id BIGINT PRIMARY KEY, key TEXT, "
                        f"text TEXT, vector vector({dim}))")
            cur.execute("CREATE INDEX ON memories USING hnsw (vector vector_cosine_ops)")

    def _conn(self):
        # One connection per thread so the concurrency sweep is truly parallel.
        if not hasattr(self._local, "conn"):
            import psycopg2
            c = psycopg2.connect(host="localhost", port=5432, user="postgres",
                                 password="bench", dbname="duxx_bench")
            c.autocommit = True
            self._local.conn = c
        return self._local.conn

    def insert(self, doc_id, key, text, embedding):
        with self._conn().cursor() as cur:
            cur.execute("INSERT INTO memories (id, key, text, vector) VALUES (%s,%s,%s,%s)",
                        (doc_id, key, text, str(list(map(float, embedding)))))

    def query(self, key, text, embedding, k):
        with self._conn().cursor() as cur:
            cur.execute("SELECT id FROM memories ORDER BY vector <=> %s::vector LIMIT %s",
                        (str(list(map(float, embedding))), k))
            return [int(r[0]) for r in cur.fetchall()]

    def close(self):
        try:
            with self._conn().cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS memories")
            self._conn().close()
        except Exception:
            pass


BACKENDS: dict[str, type[Backend]] = {
    b.name: b for b in [
        ReferenceBackend, DuxxDBBackend, DuxxGrpcBackend,
        RedisStackBackend, QdrantBackend, PgVectorBackend,
    ]
}


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def _pct(sorted_us: list[float], p: float) -> float:
    if not sorted_us:
        return 0.0
    i = min(len(sorted_us) - 1, max(0, int(round(p * len(sorted_us))) - 1))
    return sorted_us[i]


def _lat_summary(times_us: list[float]) -> dict:
    s = sorted(times_us)
    return {
        "p50_us": round(_pct(s, 0.50), 1),
        "p95_us": round(_pct(s, 0.95), 1),
        "p99_us": round(_pct(s, 0.99), 1),
        "mean_us": round(statistics.mean(times_us), 1) if times_us else 0.0,
    }


# ---------------------------------------------------------------------------
# Run one backend: load, quality + single-client latency, concurrency sweep
# ---------------------------------------------------------------------------

def run_backend(backend_cls, corpus, queries, gt, k_values, concurrency, duration):
    vecs, texts, keys = corpus
    q_vecs, q_texts, q_keys = queries
    n = len(vecs)
    b = backend_cls()
    b.open(vecs.shape[1])

    # ---- load ----
    ins_us: list[float] = []
    t0 = time.perf_counter()
    for i in range(n):
        s = time.perf_counter()
        b.insert(i, keys[i], texts[i], vecs[i])
        ins_us.append((time.perf_counter() - s) * 1e6)
    load_s = time.perf_counter() - t0

    # ---- quality + single-client recall latency ----
    max_k = max(k_values)
    gt_sets = {k: [set(row[:k].tolist()) for row in gt] for k in k_values}
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    rec_us: list[float] = []
    for qi in range(len(q_vecs)):
        s = time.perf_counter()
        returned = b.query(q_keys[qi], q_texts[qi], q_vecs[qi], max_k)
        rec_us.append((time.perf_counter() - s) * 1e6)
        for k in k_values:
            recalls[k].append(recall_at_k(returned, gt_sets[k][qi], k))
            ndcgs[k].append(ndcg_at_k(returned, gt_sets[k][qi], k))

    result = {
        "backend": b.name,
        "n": n,
        "dim": int(vecs.shape[1]),
        "load_s": round(load_s, 3),
        "insert_throughput_per_s": round(n / load_s, 1) if load_s else 0.0,
        "insert_latency": _lat_summary(ins_us),
        "recall_latency_1client": _lat_summary(rec_us),
        "quality": {
            f"recall@{k}": round(statistics.mean(recalls[k]), 4) for k in k_values
        } | {
            f"ndcg@{k}": round(statistics.mean(ndcgs[k]), 4) for k in k_values
        },
        "concurrency": {},
    }

    # ---- concurrency sweep (network backends only) ----
    if not b.in_process:
        for c in concurrency:
            result["concurrency"][str(c)] = _concurrency_run(
                b, q_vecs, q_texts, q_keys, max_k, c, duration
            )

    b.close()
    return result


def _concurrency_run(b, q_vecs, q_texts, q_keys, k, clients, duration):
    """Fire queries from `clients` threads for `duration` seconds; report
    achieved QPS and latency percentiles."""
    nq = len(q_vecs)
    stop_at = time.perf_counter() + duration
    lat: list[float] = []
    count = [0]

    def worker(seed: int):
        local = []
        i = seed
        while time.perf_counter() < stop_at:
            qi = i % nq
            s = time.perf_counter()
            b.query(q_keys[qi], q_texts[qi], q_vecs[qi], k)
            local.append((time.perf_counter() - s) * 1e6)
            i += clients
        return local

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as ex:
        for part in ex.map(worker, range(clients)):
            lat.extend(part)
            count[0] += len(part)
    elapsed = time.perf_counter() - t0

    summary = _lat_summary(lat)
    summary["qps"] = round(count[0] / elapsed, 1) if elapsed else 0.0
    summary["ops"] = count[0]
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def machine_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def print_report(report: dict) -> None:
    m = report["machine"]
    print(f"\n# DuxxDB comparative benchmark")
    print(f"\n- host: {m['platform']} / {m['processor']} / {m['cpu_count']} cores"
          f" / py{m['python']}")
    print(f"- config: N={report['n']} queries={report['queries']} "
          f"dims={report['dims']} k={report['k_values']} "
          f"concurrency={report['concurrency']} duration={report['duration']}s")
    print("\n> DuxxDB runs disk-backed and queries in hybrid (vector+BM25) mode -- "
          "its real product behaviour. Ground truth is exact cosine, so DuxxDB's "
          "vector recall is if anything understated here. Reference == exact "
          "brute force (recall@k sanity == 1.0).")

    for dim_block in report["results"]:
        dim = dim_block["dim"]
        rows = dim_block["backends"]
        kk = report["k_values"]
        print(f"\n## dim = {dim}\n")
        head = ("| backend | ins/s | ins p50us | rec p50us (1c) | "
                + " | ".join(f"recall@{k}" for k in kk) + " | "
                + " | ".join(f"nDCG@{k}" for k in kk) + " |")
        print(head)
        print("|" + "---|" * (4 + 2 * len(kk)))
        for r in rows:
            q = r["quality"]
            print(f"| {r['backend']} | {r['insert_throughput_per_s']:.0f} "
                  f"| {r['insert_latency']['p50_us']:.0f} "
                  f"| {r['recall_latency_1client']['p50_us']:.0f} | "
                  + " | ".join(f"{q[f'recall@{k}']:.3f}" for k in kk) + " | "
                  + " | ".join(f"{q[f'ndcg@{k}']:.3f}" for k in kk) + " |")

        # concurrency table (only backends that ran it)
        conc_rows = [r for r in rows if r["concurrency"]]
        if conc_rows:
            print(f"\n### dim = {dim} -- sustained concurrency (QPS / p95us / p99us)\n")
            cs = report["concurrency"]
            print("| backend | " + " | ".join(f"{c} clients" for c in cs) + " |")
            print("|" + "---|" * (1 + len(cs)))
            for r in conc_rows:
                cells = []
                for c in cs:
                    d = r["concurrency"].get(str(c))
                    cells.append(f"{d['qps']:.0f} / {d['p95_us']:.0f} / {d['p99_us']:.0f}"
                                 if d else "-")
                print(f"| {r['backend']} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="all",
                    help="comma list of backends, or 'all' (default) / 'quick'")
    ap.add_argument("--quick", action="store_true",
                    help="embedded + reference only; no services needed")
    ap.add_argument("-n", type=int, default=5000, help="documents (default 5000)")
    ap.add_argument("-q", "--queries", type=int, default=200)
    ap.add_argument("--dims", default="128",
                    help="comma list, e.g. 128,768,1536 (default 128)")
    ap.add_argument("--k", default="1,10", help="recall/nDCG cutoffs (default 1,10)")
    ap.add_argument("--concurrency", default="1,8,32",
                    help="client counts for the concurrency sweep (network backends)")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="seconds per concurrency level (default 5)")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x]
    k_values = sorted(int(x) for x in args.k.split(",") if x)
    concurrency = [int(x) for x in args.concurrency.split(",") if x]

    if args.quick or args.backend == "quick":
        names = ["reference", "duxxdb"]
    elif args.backend == "all":
        names = list(BACKENDS)
    else:
        names = [b.strip() for b in args.backend.split(",")]

    report = {
        "machine": machine_info(),
        "n": args.n, "queries": args.queries, "dims": dims,
        "k_values": k_values, "concurrency": concurrency, "duration": args.duration,
        "results": [],
    }

    for dim in dims:
        rng = np.random.default_rng(SEED)
        vecs, texts, keys = generate_corpus(args.n, dim, rng)
        q_vecs, q_texts, q_keys = generate_queries(args.queries, vecs, texts, keys, rng)
        gt = exact_ground_truth(vecs, q_vecs, GT_DEPTH)
        corpus = (vecs, texts, keys)
        queries = (q_vecs, q_texts, q_keys)

        dim_block = {"dim": dim, "backends": []}
        for name in names:
            if name not in BACKENDS:
                print(f"[{name}] unknown backend, skipping", file=sys.stderr)
                continue
            try:
                r = run_backend(BACKENDS[name], corpus, queries, gt,
                                k_values, concurrency, args.duration)
                dim_block["backends"].append(r)
            except Exception as e:
                print(f"[{name}] dim={dim} skipped: {e}", file=sys.stderr)
        report["results"].append(dim_block)

    print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
