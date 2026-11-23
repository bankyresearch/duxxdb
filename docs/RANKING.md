# Ranking: titles, URLs, priors and search deadlines

DuxxDB 0.4.4 adds three things to `REMEMBER` and `RECALL`, for search over
web pages where a page *named* for the query, or the page of an important
site, should come before a page that only mentions the query in passing:

1. **Fielded text scoring.** The BM25 side scores a row's title and URL as
   their own fields, with boosts, plus a bonus when the whole query appears
   as a phrase in the title.
2. **A per-row prior.** `REMEMBER ... PRIOR p` stores a number from 0 to 1
   (for example a site-importance score) that `RECALL` can weigh into the
   final score.
3. **A search deadline.** `RECALL ... TIMEOUT ms` returns the best hits found
   when the time is up instead of waiting for the whole search.

Every existing command and argument works as before; the new arguments are
optional, and every new weight that changes the fused ranking defaults to 0.

## RESP syntax

```text
REMEMBER key text [PRIOR p] [TITLE title] [URL url]   -> (integer) id
RECALL   key query [k] [TIMEOUT ms]                    -> [[id, score, text], ...]
```

- The options of `REMEMBER` are trailing pairs after the text, in any order,
  each at most once, names in any case. Send the text as **one** argument:
  when the text is sent as several words, a trailing `PRIOR <number>`,
  `TITLE x` or `URL x` pair is read as an option, not as text. A trailing
  `PRIOR` whose value is not a number stays text, as before.
- `PRIOR` must be a number from 0 to 1; anything else is an error
  (`ERR PRIOR must be a number from 0 to 1, got ...`) and nothing is stored.
- `TIMEOUT` is in milliseconds, counted from when the server starts handling
  the command (query embedding included). `TIMEOUT 0` means no limit. A
  timeout is never an error: the reply has the usual shape and holds the best
  hits found in time (possibly fewer than `k`, possibly none). `INFO` counts
  cut-short recalls in `recall_timeouts`.

Examples:

```text
REMEMBER web "https://www.wipro.com/\nWipro\nIT services company\n..." PRIOR 0.82
REMEMBER web "Sign in to your documents" TITLE "DigiLocker login" URL https://www.digilocker.gov.in/ PRIOR 0.9
RECALL web "wipro" 10 TIMEOUT 150
```

## Fielded text scoring

Every row is indexed in three tantivy fields:

| Field   | Holds                                                         |
|---------|---------------------------------------------------------------|
| `text`  | the whole text, exactly as in 0.4.3                          |
| `title` | the `TITLE` given, else the text's second line (see below)   |
| `url`   | the `URL` given, else the text's first line, without its scheme and a leading `www.`, split at dots, slashes and hyphens (`https://www.digilocker.gov.in/` is indexed as `digilocker gov in`) |

**Title and URL from the text's layout.** When `REMEMBER` gives no `TITLE`
or `URL`, they are read from the text itself: if its first line is a single
`http://` or `https://` URL, that line is the URL and the second line the
title. A text whose first line is not such a URL has no title and no URL.
This is the layout Gaelxe already writes (URL, title, description, marker,
body), so existing rows get their fields when the text index is rebuilt,
without writing them again.

**Scoring.** A query is BM25 over `text` (weight 1), plus `title` matches
times the title boost, plus `url` matches times the URL boost, plus, for a
query of two words or more, the title-phrase boost when the whole query
appears in the title in order (a phrase match). Each field uses the text tokenizer
(`DUXX_TEXT_TOKENIZER`, so `words` applies to titles and URLs too). Because
`text` still holds the whole text, **a row with no title and no URL scores
exactly as in 0.4.3**, and with every boost set to 0 every row does.

| Variable                       | Default | Meaning                                  |
|--------------------------------|---------|------------------------------------------|
| `DUXX_TEXT_TITLE_BOOST`        | `3`     | weight of a title match (0 = off)        |
| `DUXX_TEXT_URL_BOOST`          | `1.5`   | weight of a URL match (0 = off)          |
| `DUXX_TEXT_TITLE_PHRASE_BOOST` | `2`     | weight of the whole query in the title (0 = off) |
| `DUXX_FIELD_MATCH_WEIGHT`      | `0`     | weight of the title/URL match in the fused score (below) |

**Why a field-match weight.** `RECALL` fuses the vector and text sides by
reciprocal rank (RRF, k = 60), which sees only each side's order. Moving the
page titled "Wipro" from third to first on the text side changes its fused
score by about 1%, so when the vector side prefers a deep page that repeats
the word, the deep page still wins. With `DUXX_FIELD_MATCH_WEIGHT = w > 0`,
each fused candidate's score is multiplied by `1 + w * m`, where `m` (0 to 1)
is the row's title/URL score relative to the best title/URL match for the
query. Rows without a title or URL have `m = 0`.

## Priors

`REMEMBER ... PRIOR p` stores `p` with the row. `RECALL` multiplies each
fused score by `1 + w * p`, with `w` from `DUXX_PRIOR_WEIGHT` (default `0`,
so priors are stored but change nothing until it is set). A row without a
prior counts as `p = 0`. `INFO` reports `priors:<rows that have one>`.

When either multiplier is on, `RECALL` fuses all `max(3k, 32)` candidates of
each side, applies the multipliers, and returns the best `k`. With both at 0
it is the plain fusion of 0.4.3.

The final score is therefore

```text
rrf(vector rank, text rank) * (1 + DUXX_FIELD_MATCH_WEIGHT * m) * (1 + DUXX_PRIOR_WEIGHT * prior)
```

## Deadline

With `TIMEOUT`, the vector and text searches run in parallel on their own
threads. At the deadline the server fuses what each side has: the vector
side's hits if it finished, and the text side's hits from every index segment
it finished (tantivy is searched one segment at a time and checks the
deadline between segments). A side still running finishes in the background
and its result is dropped. Without `TIMEOUT` nothing changes.

## INFO

`INFO` gains:

```text
priors:<rows with a prior>
recall_timeouts:<recalls cut short since start>
# Ranking
text_title_boost:3
text_url_boost:1.5
text_title_phrase_boost:2
field_match_weight:0
prior_weight:0
```

## Upgrading from 0.4.3 (on-disk format)

- **Row store (redb): no migration, both directions.** A row is written
  exactly as 0.4.3 wrote it unless it has a prior, title or URL; then those
  are appended after it. 0.4.3 reads such a row and ignores the appended
  bytes, and 0.4.4 reads 0.4.3 rows with no prior, title or URL. Rolling back
  to 0.4.3 keeps every row; it loses only the priors and given titles/URLs.
- **Text index (tantivy): rebuilt once, automatically.** The index schema
  gained the `title` and `url` fields. On the first start of 0.4.4 the old
  index no longer matches the schema; it is recreated empty and refilled from
  the row store (one log line, `text index schema differs ...; recreating it
  empty to be refilled`), as when `DUXX_TEXT_TOKENIZER` changes. This takes
  as long as a text reindex of the whole store (rows are read from disk and
  committed in batches of 50,000) and needs no action. Rolling back to 0.4.3
  rebuilds it once more the same way.
- **Vector index (HNSW): unchanged.**

## Settings for Gaelxe

```text
DUXX_TEXT_TOKENIZER=words          # as today
DUXX_LEAN_ROWS=1                   # as today
DUXX_TEXT_TITLE_BOOST=3            # defaults; listed for clarity
DUXX_TEXT_URL_BOOST=1.5
DUXX_TEXT_TITLE_PHRASE_BOOST=2
DUXX_FIELD_MATCH_WEIGHT=1          # let a title/URL match count in the fused score
DUXX_PRIOR_WEIGHT=0.6              # the weight Gaelxe's own ranker uses for the site prior
```

and send `REMEMBER <key> <page text> PRIOR <site prior>` and
`RECALL <key> <query> <k> TIMEOUT <ms>`. The page text needs no change: its
first two lines already give the URL and the title. The weights above are a
starting point to tune on the judged benchmark.
