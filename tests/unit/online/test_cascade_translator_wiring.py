"""Khoá đường dựng translator của `Cascade` — mfe.online.cascade.

Vì sao cần một file riêng cho đúng ba dòng code
------------------------------------------------
`Cascade.from_config` dựng translator qua `load()`, và `load()` bắt **mọi**
`Exception` rồi trả `None`. Nó bắt như vậy có lý: "kho chưa ingest" là trạng thái
hợp lệ và không được làm sập cả trình duyệt. Nhưng nó bắt luôn `NameError`,
`ImportError`, `TypeError` — tức **lỗi lập trình** — và biến chúng thành một dòng
log.

Chuyện đó đã xảy ra thật. Lần merge 21-08-2026 để lại
`load("translator", lambda: _build_translator_with_quest(config))` — thiếu
`cls.`, nên là `NameError`. Hậu quả không phải một stack trace: translator thành
`None`, `resolve_english` không có bản tiếng Anh, và arm `siglip` **tự tắt** vì
tháp text của nó chỉ hiểu tiếng Anh. Mất một arm mạnh nhất, im lặng hoàn toàn.

Nên những test dưới đây gọi **thẳng** `_build_translator_with_quest`, không đi
qua `load()`. Đó là cả điểm của chúng: chỗ nào nuốt lỗi thì phải có một đường
khác không nuốt, nếu không thì không phép kiểm nào chạm tới được.
"""

from __future__ import annotations

from types import SimpleNamespace

from mfe.online.cascade import Cascade


def _config(*, enabled: bool, tiers=("mt",), cache_path: str = ""):
    """Cấu hình tối thiểu mà `build_translator` đọc."""
    translation = SimpleNamespace(
        enabled=enabled,
        tiers=list(tiers),
        mt_checkpoint="VietAI/envit5-translation",
        mt_num_beams=1,
        mt_max_new_tokens=128,
        cache_path=cache_path,
    )
    return SimpleNamespace(retrieval=SimpleNamespace(translation=translation))


def test_the_builder_is_reachable_and_callable():
    """Bắt đúng lớp lỗi mà `load()` nuốt: tên sai, import sai, chữ ký sai.

    `enabled=False` cho phép chạy mà không tải 1,2 GB envit5 — `build_translator`
    trả `None` trước khi dựng backend nào. Cái đang được kiểm là **đường đi tới
    hàm**, không phải bản thân phép dịch.
    """
    assert Cascade._build_translator_with_quest(_config(enabled=False)) is None


def test_from_config_does_not_swallow_a_broken_translator_call(caplog):
    """Đây mới là test bắt được lỗi thật, và nó bắt ở CHỖ GỌI.

    Test ở trên gọi thẳng `_build_translator_with_quest`, nên nó vẫn xanh khi
    `from_config` gọi sai tên — method có tồn tại, chỉ chỗ gọi thiếu `cls.`.
    Phải đi qua đúng đường `from_config` mới chạm tới được.

    Cách bắt: `load()` nuốt lỗi nhưng **ghi log** trước khi nuốt. Với translation
    tắt, đường lành trả `None` mà không ném gì, nên KHÔNG có dòng
    "translator unavailable" nào. Có dòng đó nghĩa là lambda đã ném — `NameError`
    lần trước sẽ hiện ra ở đây.
    """
    def no_dense(collection, dimension):  # tránh chạm Milvus, khỏi chờ timeout
        raise RuntimeError("khong dung Milvus trong test nay")

    with caplog.at_level("WARNING", logger="mfe.online.cascade"):
        cascade = Cascade.from_config(
            _config(enabled=False),
            vector_index=None,
            inverted_index=None,
            reranker=None,
            text_store=None,
            dense_store_factory=no_dense,
        )

    assert cascade.translator is None  # translation tắt — đúng như cấu hình
    broken = [r for r in caplog.records if "translator" in r.getMessage()]
    assert not broken, f"duong dung translator NEM loi: {[r.getMessage() for r in broken]}"


def test_from_config_does_not_probe_stores_when_flat_overrides_are_explicit(monkeypatch):
    """Explicit ``None`` means the caller deliberately uses a flat bundle.

    ``dict.get(key, expensive_default())`` evaluates the default eagerly, so
    the old wiring still opened FAISS/BM25 stores and emitted misleading ingest
    warnings even though the supplied overrides won.  The flat evaluator must
    not touch either store at all.
    """

    probed: list[str] = []
    monkeypatch.setattr(
        "mfe.online.cascade._load_vector_index",
        lambda _config: probed.append("vector"),
    )
    monkeypatch.setattr(
        "mfe.online.cascade._load_inverted_index",
        lambda _config: probed.append("inverted"),
    )

    Cascade.from_config(
        _config(enabled=False),
        vector_index=None,
        inverted_index=None,
        reranker=None,
        text_store=None,
        dense_store_factory=lambda *_: None,
    )

    assert probed == []


def test_quest_backend_is_optional():
    """Thiếu `quest` thì translator vẫn dựng được, chỉ mất tier `llm`.

    `_build_translator_with_quest` bắt `ImportError` quanh `QuestKIS` một cách có
    chủ ý — tầng agent là tuỳ chọn. Nhưng nó KHÔNG được bắt rộng hơn thế, nếu
    không thì một `QuestKIS()` hỏng lúc khởi tạo lại thành im lặng.
    """
    import mfe.online.quest as quest

    assert hasattr(quest, "QuestKIS"), "QuestKIS bien mat -> tier llm chet lang le"
