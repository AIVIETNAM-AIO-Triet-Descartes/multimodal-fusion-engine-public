#!/usr/bin/env bash
set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$WEB_DIR/.." && pwd)"
cd "$WEB_DIR"
export PYTHONPATH="$REPO_DIR/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export MFE_WEB_TORCH_THREADS="${MFE_WEB_TORCH_THREADS:-16}"
export MFE_WEB_MAX_PARALLEL_LANES="${MFE_WEB_MAX_PARALLEL_LANES:-16}"
export MFE_WEB_VERIFY_PARALLEL="${MFE_WEB_VERIFY_PARALLEL:-16}"

exec "$REPO_DIR/.venv/bin/uvicorn" server:app \
  --host 0.0.0.0 \
  --port 8501 \
  --workers 1 \
  --loop asyncio \
  --http h11 \
  --timeout-keep-alive 300
