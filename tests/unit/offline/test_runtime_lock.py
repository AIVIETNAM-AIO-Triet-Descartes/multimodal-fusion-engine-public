from importlib.metadata import version

from mfe.offline.runtime_lock import resolved_requirement_lock, write_modal_runtime_lock


def test_modal_runtime_lock_is_exact_sorted_and_excludes_paddle_distributions(tmp_path):
    pins = resolved_requirement_lock()

    assert pins == sorted(pins, key=str.casefold)
    assert len(pins) == len({pin.split("==", 1)[0].casefold() for pin in pins})
    assert all("==" in pin for pin in pins)
    assert not any(pin.casefold().startswith("modal==") for pin in pins)
    assert not any(pin.casefold().startswith("paddlepaddle==") for pin in pins)
    assert not any(pin.casefold().startswith("paddlepaddle-gpu==") for pin in pins)

    # NumPy is a mandatory project dependency and must be copied exactly from
    # the validated Kaggle interpreter rather than resolved again on Modal.
    assert f"numpy=={version('numpy')}" in pins

    path = write_modal_runtime_lock(tmp_path / "runtime.lock.txt")
    assert path.read_text(encoding="utf-8").splitlines()[2:] == pins
