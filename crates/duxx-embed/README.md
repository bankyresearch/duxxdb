# duxx-embed

Embedding providers for DuxxDB. Every provider implements one trait,
`Embedder`, and is selected at run time by a spec string — the
`--embedder` flag or `DUXX_EMBEDDER` env of `duxx-server`, `duxx-grpc`
and `duxx-mcp`.

| Spec | Feature | Dim | What it is |
|---|---|---|---|
| `hash:<dim>` | `hash` (default) | any | Token-hash buckets. Deterministic toy for tests; no semantics. |
| `local:multilingual-e5-small` | `local` | 384 | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) running **in-process on [Candle](https://github.com/huggingface/candle)** (pure Rust, CPU). ~100 languages incl. English and Hindi. |
| `openai:<model>` | `openai` | 1536 / 3072 | OpenAI HTTP API (`OPENAI_API_KEY`). |
| `cohere:<model>` | `cohere` | 1024 / 384 | Cohere HTTP API (`COHERE_API_KEY`). |

## The local embedder

`local:` runs a real sentence-embedding transformer inside the DuxxDB
process: no Python, no ONNX runtime, no outside inference service. The
pipeline for each text is

1. the e5 prefix — `query: ` for searches (`RECALL`, `*.SEARCH`),
   `passage: ` for stored text (`REMEMBER*`, ingest) — via
   `Embedder::embed_query` / `Embedder::embed`;
2. the model's own XLM-R SentencePiece tokenizer (`tokenizer.json`,
   via the `tokenizers` crate), truncated to 512 tokens;
3. the 12-layer, 384-wide BERT encoder;
4. mean pooling over the real (non-padding) tokens;
5. L2 normalisation (cosine = dot product).

Batches (`REMEMBER.BATCH`, `Embedder::embed_batch`) are sorted by length
and padded per batch, with batch size capped so that
`rows × longest² ≤ 4 × 512²` — memory per forward pass stays bounded
whatever the input.

### Spec options

```
local:<model>[,max_tokens=N][,batch=N][,prefix=e5|none]
```

- `<model>` — the alias `multilingual-e5-small` (or its repo id
  `intfloat/multilingual-e5-small`), or a path to a directory holding
  `config.json`, `tokenizer.json` and `model.safetensors` (or
  `pytorch_model.bin`) of a BERT-architecture e5-family model.
- `max_tokens` — truncate inputs (default 512, the model's limit).
  Lowering it speeds up long passages; text past the limit is ignored.
- `batch` — max rows per forward pass (default 32).
- `prefix` — `e5` (default) adds `query: `/`passage: `; `none` embeds
  raw text (for models trained without prefixes).

### Model files

Built-in aliases are pinned to an exact Hugging Face commit and every
downloaded file is checked against a pinned SHA-256 before it is used.
Files are fetched once, on first start, into:

1. `DUXX_MODEL_DIR` (the Docker image sets `/var/lib/duxxdb/models`,
   on the data volume), else
2. `$XDG_CACHE_HOME/duxxdb/models`, else `$HOME/.cache/duxxdb/models`.

`DUXX_MODEL_OFFLINE=1` forbids downloads (air-gapped hosts: copy the
`intfloat--multilingual-e5-small@614241f622f5/` directory in by hand).
`DUXX_HF_ENDPOINT` points at a Hugging Face mirror.

Inference threads: Candle's CPU kernels use all cores by default; set
`RAYON_NUM_THREADS` to cap them on a shared host.

### Performance

Measured on an AWS **c7g.xlarge** (4 × Graviton3), release build, with
the host also serving a live site and crawler (load average 7–9 on 4
vCPUs, benchmark at `nice 15`; an idle host is faster):

| | 2 threads | 4 threads (≈1.5 effective cores) |
|---|---|---|
| query embed (8 tokens) | 20 ms | 25 ms |
| passages of ~80 tokens, batches of 64 | 11.7 / s | 17.2 / s |
| passages of 512 tokens, batches of 16 | 1.1 / s | 1.5 / s |

- **Memory**: a `duxx-server` with the e5 embedder sits at ~760–780 MB
  RSS (118 M f32 parameters, most of them the 250 k-token multilingual
  vocabulary), peaking at ~1.1 GB while the model loads and while
  embedding 512-token batches. A cross-encoder reranker on top adds the
  same again (see `crates/duxx-rerank/README.md`): give such a container
  ≥ 2.5 GB.
- **Load**: 8–10 s from the local model store; the first start also
  downloads 470 MB.
- A 512-token passage is ~27 GFLOP through 12 layers, so long passages
  dominate re-embedding time. For a search index, embed title + lead
  (≈80–128 tokens) rather than whole pages: 300 k pages at ~80 tokens is
  ~7 h on 2 contended cores; at 512 tokens it would be ~3 days.

## Switching embedders

Vectors from two embedders live in different spaces — even at the same
dimension a `hash:384` vector and an e5 vector are not comparable — so a
persistent store records its embedder and **refuses to open under a
different one**:

- `dir:` storage keeps `embedder.json` inside the directory (the Phase 7
  `--phase7-storage dir:` directory gets its own);
- `redb:<file>` storage keeps `<file>.embedder.json` beside the file.

```
Error: embedder mismatch: this store's vectors were written by `hash:128` (128-d)
but the server is configured with `local:intfloat/multilingual-e5-small@614241f622f5`
(384-d). ... Marker file: /var/lib/duxxdb/embedder.json
```

Stores created before markers existed adopt the embedder they are next
opened with — after the row-level dimension check has passed, so a
128-d store opened with a 384-d embedder is still refused (with the
same pointer here). `INFO` reports the active embedder (`embedder:`).

To move a store to a new embedder, **re-embed into a new, empty store**:

1. Start a second `duxx-server` with the new embedder on a **new**
   storage directory (or volume), e.g.
   `--embedder local:multilingual-e5-small --storage dir:/var/lib/duxxdb-e5`.
   The first start downloads the model.
2. Re-ingest into it from your source of truth — for a search index
   that is the crawler's page store; send `REMEMBER.BATCH key text1 text2 …`
   in batches of 16–64 so the model batches them.
   If DuxxDB itself holds the only copy, page through the old server
   with `MEMORY.SCAN <cursor> 500` (replies `[next, [[id, key, text], …]]`)
   and `REMEMBER.BATCH` each page into the new one. `MEMORY.SCAN` carries
   key and text only, so importance/kind/tags must be re-applied
   (`REMEMBER.META`) from your own records if you use them.
3. Point clients at the new server (or swap the volume and restart),
   then retire the old store.

Never point the new embedder at the old directory and delete
`embedder.json` to "force" it: the old vectors would stay in the index
next to the new ones and every search would mix two spaces.

## Testing

```text
cargo test -p duxx-embed                                   # hash, spec parsing, batching
DUXX_MODEL_DIR=/models cargo test -p duxx-embed --features local --release -- --include-ignored
```

The `local` model tests (tokenisation with prefixes/specials/Hindi,
truncation, 384-d unit-norm output, batched = single, and a ranking
sanity check in English, Hindi and cross-lingual) download the model on
first run, so they are `#[ignore]`d in the default test run.
