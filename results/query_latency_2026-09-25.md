# Query latency: 5,862 ms → 294 ms (2026-09-25)

**Source:** commit `be743e3` in this repository,
*"perf(online): tầng văn bản 5.862 -> 294 ms, Final lệch đúng 0,0"*
(author: Trịnh Vỹ Triết, 2026-09-25). Verify with `git show be743e3`.

## What changed

The lexical tier (BM25 over captions, OCR and subtitles) dominated query time.
`src/mfe/online/bm25_thua.py` precomputes the query-independent part of BM25 as a
sparse CSC matrix, so scoring a query sums only the columns of its terms. The same
commit normalizes `subtitle_exact` documents once at load time, caches query
vectors, and disables two arms (`object`, `ocr_vietocr`) via `arms_disabled`.
Latency is measured with `scripts/measure_latency.py`.

## Recorded numbers (verbatim from the commit message)

Per-arm BM25 timing:

```
      caption_qwen  1.725 -> 4,3 ms  (400x)
      subtitle      1.237 -> 1,6 ms  (779x)
      ocr_qwen25      362 -> 0,9 ms  (405x)
```

Official score (`mfe.eval.official`), before vs. after, measured against the same
branch with the two edited files stashed:

```
    dev      0,5192239858906526 -> 0,5192239858906526   lệch 0,0
    heldout  0,5419444444444445 -> 0,5419444444444445   lệch 0,0
```

End-to-end query latency:

```
Độ trễ: trung vị 5.862 -> 294 ms (20x), p90 512 ms.
```

(Vietnamese number format: `5.862` = 5,862 and `0,5192` = 0.5192. *lệch* = difference,
*trung vị* = median.)
