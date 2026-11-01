# Results

Measurements recorded by the team while building the competition system. Each
file quotes the original record **verbatim** (Vietnamese, as written at the time)
and names the commit in this repository that contains it, so every number can be
checked with `git show <commit>`. Nothing here was re-measured, rounded or
re-computed for this replica.

| File | What it records | Source commit |
|---|---|---|
| [`query_latency_2026-09-25.md`](query_latency_2026-09-25.md) | Median query latency 5,862 ms → 294 ms after rewriting the lexical (BM25) tier, with the official score unchanged | `be743e3` |
| [`caption_query_language_2026-09-21.md`](caption_query_language_2026-09-21.md) | R@100 of the caption BM25 arm 11.7% → 60.1% after feeding it the English translation of the query | `cf05ca0` |
