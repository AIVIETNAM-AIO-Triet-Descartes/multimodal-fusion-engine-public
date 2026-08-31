"""In-memory base-plus-delta overlay for causal flat-bundle A/B tests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from mfe.offline.augmentation.io import canonical_hash
from mfe.online.flat_arms import FlatBundle


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"vector artifact must be 2-D, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_overlay_flat_bundle(
    base_root: str | Path,
    delta_root: str | Path,
    include_ids: Iterable[str],
    *,
    sources=None,
    models=None,
    required_models: tuple[str, ...] = ("siglip",),
) -> FlatBundle:
    """Load base unchanged and expose only allowlisted delta records.

    No merged artifact is written during parameter sweeps.  Text rows may be
    absent when a detector found no text, but every selected frame must have all
    ``required_models`` vectors so visual retrieval does not silently ignore it.
    """

    selected = set(include_ids)
    base = FlatBundle.load(base_root, sources=sources, models=models)
    if not selected:
        return base
    delta = FlatBundle.load(delta_root, sources=sources, models=models)

    for source, delta_text in delta.texts.items():
        filtered = {key: value for key, value in delta_text.items() if key in selected}
        if not filtered:
            continue
        base_text = base.texts.setdefault(source, {})
        duplicates = set(base_text) & set(filtered)
        if duplicates:
            raise ValueError(
                f"delta source {source} duplicates BTC ids: {sorted(duplicates)[:3]}"
            )
        base_text.update(filtered)
        base.text_keys[source] = [*base.text_keys.get(source, []), *sorted(filtered)]
        base.missing.pop(source, None)
        # Any cached BM25 was built for the pre-overlay corpus and is invalid.
        base.bm25.pop(source, None)

    present_by_model: dict[str, set[str]] = {}
    for model, matrix in delta.vectors.items():
        keys = delta.vector_keys[model]
        if len(keys) != len(set(keys)):
            raise ValueError(f"delta model {model} contains duplicate keyframe ids")
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != len(keys):
            raise ValueError(
                f"delta model {model} has {len(keys)} ids for matrix {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"delta model {model} contains NaN/Inf vectors")
        indices = [index for index, key in enumerate(keys) if key in selected]
        if not indices:
            present_by_model[model] = set()
            continue
        filtered_keys = [keys[index] for index in indices]
        filtered_matrix = _normalise(matrix[indices])
        present_by_model[model] = set(filtered_keys)
        if model in base.vectors:
            base_keys = base.vector_keys[model]
            duplicates = set(base_keys) & set(filtered_keys)
            if duplicates:
                raise ValueError(
                    f"delta model {model} duplicates BTC ids: {sorted(duplicates)[:3]}"
                )
            base_matrix = np.asarray(base.vectors[model])
            if base_matrix.ndim != 2 or base_matrix.shape[1] != filtered_matrix.shape[1]:
                raise ValueError(
                    f"{model} dimension drift: base {base_matrix.shape}, "
                    f"delta {filtered_matrix.shape}"
                )
            base.vectors[model] = np.concatenate([base_matrix, filtered_matrix], axis=0)
            base.vector_keys[model] = [*base_keys, *filtered_keys]
        else:
            base.vectors[model] = filtered_matrix
            base.vector_keys[model] = filtered_keys
        base.missing.pop(model, None)

    for model in required_models:
        missing = selected - present_by_model.get(model, set())
        if missing:
            raise ValueError(
                f"delta is missing required {model} vectors for {len(missing)} id(s), "
                f"e.g. {sorted(missing)[:3]}"
            )
    return base


def overlay_fingerprint(
    base_root: str | Path,
    delta_root: str | Path,
    include_ids: Iterable[str],
) -> str:
    """Cheap logical fingerprint; artifact byte hashes belong in Gate 0 lock."""

    return canonical_hash(
        {
            "base_root": str(Path(base_root)),
            "delta_root": str(Path(delta_root)),
            "include_ids": sorted(set(include_ids)),
        }
    )
