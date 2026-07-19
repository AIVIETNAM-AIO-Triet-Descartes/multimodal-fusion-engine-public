"""Config loading — docs/01 §3.1.

Includes loading the real config/local.yaml, so a change that desyncs the YAML
from the Config schema fails here rather than at someone's first run.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.core.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_YAML = REPO_ROOT / "config" / "local.yaml"


class TestRealConfig:
    def test_local_yaml_loads_and_validates(self):
        config = load_config(LOCAL_YAML)
        assert isinstance(config, Config)

    def test_default_repo_root_is_the_repo(self):
        """config/local.yaml -> grandparent is the repo root."""
        config = load_config(LOCAL_YAML)
        assert Path(config.repo_root) == REPO_ROOT

    def test_paths_resolve_to_absolute_under_repo(self):
        config = load_config(LOCAL_YAML)
        raw_path = Path(config.paths.raw)
        assert raw_path.is_absolute()
        assert raw_path == REPO_ROOT / "data" / "raw"

    def test_paths_are_cwd_independent(self, tmp_path, monkeypatch):
        """The whole point of resolution: same stores from any working dir."""
        monkeypatch.chdir(tmp_path)
        config = load_config(LOCAL_YAML)
        assert Path(config.paths.relational_db) == REPO_ROOT / "stores" / "relational" / "app.db"

    def test_ocr_disabled_by_default(self):
        """models.ocr: null means the OCR module is off (docs/02 §3)."""
        config = load_config(LOCAL_YAML)
        assert config.models.ocr is None

    def test_modal_yaml_also_validates(self):
        """The stub config must stay in sync with the schema even while unused."""
        config = load_config(REPO_ROOT / "config" / "modal.yaml")
        assert config.storage.backend == "modal"

    def test_modal_absolute_paths_not_rerooted(self):
        """A POSIX volume path stays absolute, even loaded on Windows."""
        config = load_config(REPO_ROOT / "config" / "modal.yaml")
        assert config.paths.raw == "/vol/data/raw/"


class TestResolvePath:
    def test_resolves_relative_against_repo_root(self):
        config = load_config(LOCAL_YAML)
        expected = REPO_ROOT / "eval" / "queryset.jsonl"
        assert Path(config.resolve_path("eval/queryset.jsonl")) == expected

    def test_absolute_passes_through(self):
        config = load_config(LOCAL_YAML)
        assert config.resolve_path("/vol/x") == "/vol/x"


class TestFailures:
    def test_missing_file(self):
        with pytest.raises(ContractError) as exc:
            load_config(REPO_ROOT / "config" / "does_not_exist.yaml")
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT

    def test_unknown_key_is_rejected(self, tmp_path):
        """extra='forbid' turns a typo'd config key into a load-time error."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "paths:\n"
            "  raw: data/raw/\n"
            "  keyframes: data/keyframes/\n"
            "  segments: data/segments/\n"
            "  vector_store: stores/vector/\n"
            "  inverted_store: stores/inverted/\n"
            "  relational_db: stores/relational/app.db\n"
            "  eval_results: eval/results/\n"
            "  throughput: eval/throughput.json\n"
            "retrieval:\n"
            "  embed_dim: 512\n"
            "  coarse_topk: 500\n"
            "  rerank_topk: 50\n"
            "  rerank_budget_ms: 3000\n"
            "eval:\n"
            "  queryset: eval/queryset.jsonl\n"
            "  typo_key: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(bad, repo_root=REPO_ROOT)

    def test_coarse_topk_over_500_rejected(self, tmp_path):
        """docs/04 §2: top_k <= 500. The bound lives in the schema."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "paths:\n"
            "  raw: data/raw/\n"
            "  keyframes: data/keyframes/\n"
            "  segments: data/segments/\n"
            "  vector_store: stores/vector/\n"
            "  inverted_store: stores/inverted/\n"
            "  relational_db: stores/relational/app.db\n"
            "  eval_results: eval/results/\n"
            "  throughput: eval/throughput.json\n"
            "retrieval:\n"
            "  embed_dim: 512\n"
            "  coarse_topk: 9999\n"
            "  rerank_topk: 50\n"
            "  rerank_budget_ms: 3000\n"
            "eval:\n"
            "  queryset: eval/queryset.jsonl\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(bad, repo_root=REPO_ROOT)
