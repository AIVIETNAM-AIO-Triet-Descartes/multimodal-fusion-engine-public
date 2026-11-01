# Caption arm query language: R@100 11.7% → 60.1% (2026-09-21)

**Source:** commit `cf05ca0` in this repository,
*"fix(online): caption_qwen phải ăn truy vấn đã dịch + đổi caption sang Gemini"*
(author: Trịnh Vỹ Triết, 2026-09-21). Verify with `git show cf05ca0`.

## What changed

Keyframe captions are written in English, but every text arm received the
Vietnamese query by default, so the caption BM25 arm searched an English index
with Vietnamese words. The fix adds a per-arm `query_language` to
`TEXT_ARM_SPECS` (`src/mfe/online/text_arms.py`) so the caption arm receives the
English translation. Measured with `scripts/measure_caption_query_lang.py`.

## Recorded numbers (verbatim from the commit message)

163 development queries, real translations, only the query string changed:

```
    vi      27/163   R@100 11,7%   hạng TB 72,9    <- hiện trạng
    en     122/163   R@100 60,1%   hạng TB 55,3
    vi+en  123/163   R@100 54,6%   hạng TB 77,2
```

```
So cặp `vi→en`: 112 lên / 10 xuống.
```

**Scope:** these are figures for the **caption BM25 arm on its own**, not for the
fused system. `vi` = Vietnamese query (state before the fix), `en` = English
translation, `vi+en` = both concatenated; *hạng TB* = mean rank of the target;
*112 lên / 10 xuống* = 112 queries improved and 10 got worse when switching from
`vi` to `en`. Vietnamese number format: `11,7%` = 11.7%.
