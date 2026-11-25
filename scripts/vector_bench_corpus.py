"""Write a vector-bench corpus from Wikipedia parquet shards.

usage: vector_bench_corpus.py <en.parquet> <hi.parquet> <n_en> <n_hi> <out>

<out>.docs    title + text (first 4,000 characters) of n_en + n_hi articles
<out>.queries titles of 200 English and 100 Hindi articles not in the corpus
Both files hold u32 little-endian length-prefixed UTF-8 strings.
"""
import struct
import sys

import pyarrow.parquet as pq


def rows(path, n):
    t = pq.read_table(path, columns=["title", "text"]).slice(0, n)
    return list(zip(t.column("title").to_pylist(), t.column("text").to_pylist()))


def write(path, items):
    with open(path, "wb") as f:
        for s in items:
            b = s.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)


en_path, hi_path, n_en, n_hi, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
en = rows(en_path, n_en + 200)
hi = rows(hi_path, n_hi + 100)
docs = [f"{t}\n{x}"[:4000] for t, x in en[:n_en] + hi[:n_hi]]
queries = [t for t, _ in en[n_en:n_en + 200] + hi[n_hi:n_hi + 100]]
write(out + ".docs", docs)
write(out + ".queries", queries)
print(f"docs {len(docs)} queries {len(queries)}")
