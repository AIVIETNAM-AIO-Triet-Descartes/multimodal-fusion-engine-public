#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/web
export PYTHONPATH=/home/ubuntu/multimodal-fusion-engine/src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export MFE_WEB_TORCH_THREADS="${MFE_WEB_TORCH_THREADS:-16}"
export MFE_WEB_MAX_PARALLEL_LANES="${MFE_WEB_MAX_PARALLEL_LANES:-8}"

exec /home/ubuntu/multimodal-fusion-engine/.venv/bin/uvicorn server:app \
  --host 0.0.0.0 \
  --port 8501 \
  --workers 1 \
  --loop asyncio \
  --http h11 \
  --timeout-keep-alive 300
