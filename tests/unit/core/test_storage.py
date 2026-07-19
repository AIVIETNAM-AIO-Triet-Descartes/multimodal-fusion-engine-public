"""Storage backend — docs/01 §3.2.

The seam that makes local -> Modal a config change. Tests here lock the local
backend's byte round-trip and, more importantly, its containment guard.
"""

import pytest

from mfe.contracts.errors import ContractError, ErrorCode
from mfe.core.backends.local import LocalBackend
from mfe.core.backends.modal import ModalBackend
from mfe.core.storage import get_backend


class TestLocalBackend:
    def test_write_then_read_round_trips(self, tmp_path):
        backend = LocalBackend(root=str(tmp_path))
        backend.write_bytes("keyframes/L01_V001/000250.jpg", b"\xff\xd8\xff")
        assert backend.read_bytes("keyframes/L01_V001/000250.jpg") == b"\xff\xd8\xff"

    def test_write_creates_parent_dirs(self, tmp_path):
        backend = LocalBackend(root=str(tmp_path))
        backend.write_bytes("a/b/c/d.bin", b"x")
        assert (tmp_path / "a" / "b" / "c" / "d.bin").exists()

    def test_exists(self, tmp_path):
        backend = LocalBackend(root=str(tmp_path))
        assert backend.exists("nope.bin") is False
        backend.write_bytes("yes.bin", b"x")
        assert backend.exists("yes.bin") is True

    def test_missing_read_is_structured_error(self, tmp_path):
        backend = LocalBackend(root=str(tmp_path))
        with pytest.raises(ContractError) as exc:
            backend.read_bytes("gone.bin")
        assert exc.value.code is ErrorCode.STORE_UNAVAILABLE

    def test_dotdot_escape_is_blocked(self, tmp_path):
        """A '..' in a manifest path must not read outside the root."""
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")
        root = tmp_path / "store"
        root.mkdir()
        backend = LocalBackend(root=str(root))
        with pytest.raises(ContractError) as exc:
            backend.read_bytes("../secret.txt")
        assert exc.value.code is ErrorCode.STORE_UNAVAILABLE

    def test_absolute_escape_is_blocked(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        backend = LocalBackend(root=str(root))
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        with pytest.raises(ContractError):
            backend.read_bytes(str(outside))

    def test_root_itself_is_allowed(self, tmp_path):
        """A file directly in the root (no subdir) must not trip the guard."""
        backend = LocalBackend(root=str(tmp_path))
        backend.write_bytes("top.bin", b"x")
        assert backend.read_bytes("top.bin") == b"x"


class TestGetBackend:
    def test_local(self, tmp_path):
        backend = get_backend("local", root=str(tmp_path))
        assert isinstance(backend, LocalBackend)

    def test_modal_constructs_but_is_a_stub(self):
        """It builds; using it raises (docs/01 §3.2 — stub, fails loudly)."""
        backend = get_backend("modal", volume="mfe-data")
        assert isinstance(backend, ModalBackend)
        with pytest.raises(ContractError) as exc:
            backend.read_bytes("x")
        assert exc.value.code is ErrorCode.NOT_IMPLEMENTED

    def test_modal_without_volume_rejected(self):
        with pytest.raises(ContractError) as exc:
            get_backend("modal")
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT

    def test_unknown_backend_rejected(self):
        with pytest.raises(ContractError) as exc:
            get_backend("s3")
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT
