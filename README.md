# Multimodal Fusion Engine (MFE) — public replica

> **This is a replica.** Team AICare developed the Multimodal Fusion Engine in a
> **private repository** that we used for the Ho Chi Minh City AI Challenge 2026 and
> for writing our two papers. This public repository is a reproduction of that
> repository that keeps **only the core system**: the code needed to build the
> index and run retrieval, the web cockpit used during the final round, its tests,
> and a few measurements recorded from the official system.
>
> The commit history was carried over from the private repository for the files
> kept here. Authors, dates and commit messages are the original ones. Each
> member's identity is normalized to a single name and a GitHub `noreply` address,
> and AI-assistant co-author trailers were removed. Internal notes, paper
> manuscripts, experiment artifacts, competition data and model weights are not
> included.

## Team AICare — University of Science, VNU-HCM

| Member | Role |
|---|---|
| Trịnh Vỹ Triết | Team leader |
| Nguyễn Quốc Đạt | Member |
| Trần Công Quang | Member |
| Phạm Minh Trọng | Member |
| Bùi Nhật Bảo | Member |

**Competition:** Ho Chi Minh City AI Challenge 2026 — **Potential Prize (Top 35)**.

## What the system does

Given a Vietnamese description, MFE finds the matching moment in a large video
archive (news, documentaries, traffic cameras, sports) for three task types:

- **KIS** — known-item search for a described moment;
- **Q&A** — find the moment and answer a question about it;
- **TRAKE** — find a sequence of events in the correct temporal order.

Answers are given as `video_id` plus the source-video `frame_idx`, either exported
as submission files (`app/submit_lab.py`) or sent to the organizers' DRES
evaluation server from the web cockpit.

## Architecture

```
offline (Kaggle GPU workers, per-video shards on Modal Volumes)
  keyframes + keyframe→frame maps provided by the organizers
  captions (Gemini, English) · OCR (Qwen2.5-VL / Gemini) · ASR (faster-whisper)
  SigLIP SO400M image vectors (1,152-d) · BGE-M3 text vectors (1,024-d)
  15-second window records
          │  merged into one flat store: .npz matrices (memory-mapped) + JSON + SQLite
          ▼
online (CPU only — no Milvus, no Elasticsearch, no GPU at query time)
  query translation + QUEST query expansion
  8 retrieval arms ──► weighted Reciprocal Rank Fusion (k = 60)
                   ──► scene-order prior ──► temporal grouping ──► top-100
  task adapters: KIS ranking · Q&A visual reader · TRAKE alignment (DANTE-style DP)
          ▼
web cockpit (Starlette + WebSocket): shared/private search lanes, per-query arm
toggles and weights, candidate inspection, DRES submission with a duplicate lock
```

### Retrieval arms used in the final round

From `config/btc_b12.yaml` (`fusion_weights`, `arms_disabled`):

| Arm | Evidence | Method | Weight |
|---|---|---|---|
| `siglip` | keyframe image | SigLIP SO400M, dense | 2.0 |
| `bge_m3` | caption | BGE-M3, dense | 1.0 |
| `subtitle_bge_m3` | ASR transcript | BGE-M3, dense | 1.0 |
| `caption_gemini` | caption | BM25 | 1.0 |
| `ocr_qwen25` | on-screen text (batch 1) | BM25 | 1.0 |
| `ocr_gemini` | on-screen text (batch 2) | BM25 | 0.5 |
| `subtitle_partial` | ASR transcript | partial matching | 1.0 |
| `subtitle_exact` | ASR transcript | exact phrase matching | 1.0 |

Two window-level arms (`window_bm25`, `window_bge_m3`) are implemented and disabled
by default; operators could enable and weight them per query in the cockpit.

### Key parameters (from the code)

| Component | Setting | Source |
|---|---|---|
| Candidates per arm | 500 | `config/btc_b12.yaml` (`coarse_topk`) |
| RRF constant | k = 60 | `src/mfe/online/fusion.py` |
| Scene-order prior | weight 0.5, only for queries with ≥ 3 ordered clauses | `config/btc_b12.yaml`, `src/mfe/online/scene_order.py` |
| Temporal grouping | groups of ≤ 640 frames, 20 representatives promoted | `config/btc_b12.yaml`, `src/mfe/online/dedup.py` |
| TRAKE alignment | dynamic programming with gap penalty λ = 0.001 | `src/mfe/online/dante.py` |
| Lexical index | BM25 as a precomputed sparse matrix | `src/mfe/online/bm25_thua.py` |

## Results

Every number below is quoted from a record in this repository; the `results/`
files reproduce the original text verbatim and name the commit that contains it.

| Measurement | Before | After | Source |
|---|---|---|---|
| Median query latency (CPU) | 5,862 ms | 294 ms (p90 512 ms) | [`results/query_latency_2026-09-25.md`](results/query_latency_2026-09-25.md), commit `be743e3` |
| Official score while optimizing latency (dev / heldout) | 0.5192239858906526 / 0.5419444444444445 | unchanged (difference 0.0) | same as above |
| R@100 of the caption BM25 arm, 163 dev queries | 11.7% (Vietnamese query) | 60.1% (English translation) | [`results/caption_query_language_2026-09-21.md`](results/caption_query_language_2026-09-21.md), commit `cf05ca0` |

The final flat store held 446,926 keyframes (batch 1: 176,707; batch 2: 270,219),
about 4.9 GB (commit `438fe63`).

## Repository layout

```
src/mfe/        engine: contracts, offline ingest, flat stores, online retrieval, evaluation
web/            web cockpit used in the final round (server.py, index.html)
app/            Streamlit tools: submission bench and rerank lab
config/         run configs (btc_b12.yaml = final-round configuration) and ingest shards
scripts/        store building, merging, measurement (measure_latency.py, …) and tooling
notebooks/btc/  Kaggle notebooks of the offline pipeline (keyframes, captions, OCR, ASR, embeddings)
eval/           163-query development set (team-annotated) and cached query translations
results/        recorded measurements quoted in this README
tests/          unit and integration tests
```

## Installation

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,web,browser]"  # retrieval stack, web cockpit, Streamlit tools, tests
```

`web` pulls in the online retrieval stack (`online` → `stores`: NumPy, SciPy,
BM25, sentence-transformers/PyTorch). Other extras: `offline` (local ingest
tools) and `production-stores` (the Milvus/Elasticsearch prototype that the final
system replaced with the flat store). The GPU stages of the offline pipeline ran on
Kaggle; their dependencies are listed in `notebooks/requirements-kaggle-*.txt`.
Copy `.env.example` to `.env` for the optional services.

## Running

### Smoke test with a synthetic store (no data, no models)

```bash
python scripts/make_demo_store.py --out stores/demo
python scripts/query_flat.py "người đàn ông đạp xe" --root stores/demo \
    --config config/btc_b12.yaml --en "a man riding a bicycle" \
    --arms caption_gemini,ocr_qwen25,subtitle,subtitle_exact
```

`make_demo_store.py` writes a few **hand-written, synthetic** records in the flat
store layout, so the lexical arms and RRF fusion run end to end on a laptop. Its
output is not a competition result.

### Tests

```bash
pytest tests/unit -q
```

Tests that need inputs which are not part of this replica (the notebook generator
scripts, the raw per-annotator dev files and the organizers' official queryset) are
skipped with an explicit reason. Integration tests under `tests/integration` need
live Milvus/Elasticsearch services.

### Full system

The full system needs data and services that are **not distributed** with this
repository:

- the competition videos, keyframes and keyframe→frame maps, which belong to the
  organizers;
- the flat store built from them (`scripts/build_b12_relational.py`,
  `scripts/aggregate_latest_parts.py`, `scripts/embed_latest_caption_parts.py`),
  placed under `runtime/btc_b12/` or pointed to with `MFE_RUNTIME_ROOT`;
- the SigLIP SO400M and BGE-M3 models (downloaded on first use);
- optionally, an OpenAI-compatible LLM gateway for translation, QUEST and the
  visual readers (`TRAMMAY_API_KEY`, `MFE_TRAMMAY_BASE_URL`), and DRES
  credentials (`user_dres`, `pass_dres`, `MFE_DRES_BASE_URL`).

With these in place, start the cockpit with `web/start_html.sh` (port 8501) after
creating accounts with `web/create_accounts.py`.

## License

[MIT](LICENSE).
