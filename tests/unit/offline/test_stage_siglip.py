"""Regression tests for Transformers' two SigLIP image-output APIs."""

from types import SimpleNamespace

import pytest

from mfe.offline.stages import _siglip_feature_tensor


class FakeTensor:
    """Small tensor-shaped object; importing torch is unnecessary for this boundary."""

    def float(self):
        return self


def test_accepts_legacy_tensor_output():
    tensor = FakeTensor()

    assert _siglip_feature_tensor(tensor) is tensor


def test_extracts_current_transformers_pooler_output():
    tensor = FakeTensor()
    output = SimpleNamespace(pooler_output=tensor, last_hidden_state=object())

    assert _siglip_feature_tensor(output) is tensor


@pytest.mark.parametrize("output", [object(), SimpleNamespace(pooler_output=None)])
def test_rejects_output_without_pooled_features(output):
    with pytest.raises(TypeError, match="non-empty pooler_output"):
        _siglip_feature_tensor(output)
