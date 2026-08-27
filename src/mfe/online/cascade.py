"""The retrieval cascade — docs/04 §1.

    query text
      -> [A] late-fusion coarse: vector_search + text_search over everything -> top 200-500
      -> [B] dedup + temporal grouping                                        -> top 50-100
      -> [C] early-fusion rerank: cross-encoder on original images            -> top 10-20
      -> [D] browser: grid + timeline for human verification
      -> candidate (+ evidence_id) -> Validator (Nhật Bảo)

Wired to the real stores (CHỐT — Vỹ Triết). Stage A now runs both arms:

    vector : encode the query with clip-ViT-B-32, cosine over the Faiss index
    text   : BM25 over object labels / OCR / metadata

The vector arm was impossible until the checkpoint was pinned — BTC ships
keyframe features but not the model, so a text query had no way into the space.
Verified at cosine 0.9992-1.0000 against the shipped vectors
(`scripts/verify_clip_checkpoint.py`).

**Scoring shapes this file.** Official PDF §2.2: a team may submit up to 100
answers and the score is the mean of R@{1,5,20,50,100}. Returning fewer than 100
candidates throws away four of the five terms for free. So the cascade is built
to hand the browser a deep list, not a short one, and stage B keeps the tail
instead of discarding it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from mfe.contracts.tools import Candidate
from mfe.online.arms import Arm, ArmRegistry
from mfe.online.dedup import DEFAULT_TEMPORAL_WINDOW_FRAMES, dedup
from mfe.online.fusion import weighted_fusion
from mfe.online.rerank import Reranker

logger = logging.getLogger(__name__)


def _no_dense_store(collection: str, dimension: int):
    """Default dense-store factory: there isn't one.

    Raising rather than returning None keeps `build_dense_arms` on one path —
    every failure to open a collection, including "nobody configured Milvus",
    becomes an unavailable arm with a readable reason.
    """
    raise RuntimeError("Milvus not configured (set MFE_MILVUS_URI)")


def _load_vector_index(config: Any):
    """Mo Faiss index. Import o TRONG ham: `faiss` chi can khi that su dung kho."""
    from mfe.stores.vector.index import VectorIndex

    return VectorIndex.load(config.paths.vector_store, config.retrieval.embed_dim)


def _load_inverted_index(config: Any):
    from mfe.stores.inverted.index import InvertedIndex

    return InvertedIndex.load(config.paths.inverted_store)


def _milvus_store_factory(collection: str, dimension: int):
    """Open one Milvus collection, verifying it matches what the arm expects."""
    from mfe.stores.production.config import MilvusConfig
    from mfe.stores.production.milvus import MilvusVectorStore

    config = MilvusConfig.from_env().model_copy(
        update={"collection": collection, "dimension": dimension}
    )
    store = MilvusVectorStore(config)
    health = store.health()
    if not health.connection_ok:
        raise RuntimeError(health.message or "connection failed")
    if not health.schema_ready:
        raise RuntimeError("chưa ingest — chạy scripts/ingest_sota_vectors.py")
    return store


class Cascade:
    """Online retrieval cascade. Owner: Phạm Trọng.

    Stores are injected rather than opened here, so the eval harness, the
    browser and a test can each hand it whatever they already have open.
    Use `from_config` for the ordinary case.
    """

    def __init__(
        self,
        config: Any = None,
        vector_index: Any = None,
        inverted_index: Any = None,
        reranker: Reranker | None = None,
        encode_query: Any = None,
        text_store: Any = None,
        dense_store_factory: Any = None,
        translator: Any = None,
        flat_bundle: Any = None,
    ) -> None:
        self.config = config
        self.vector_index = vector_index
        self.inverted_index = inverted_index
        self.encode_query = encode_query
        self.text_store = text_store
        self.dense_store_factory = dense_store_factory
        #: `FlatBundle` da nap san -> chay KHONG can Elasticsearch/Milvus.
        #: Dat truoc `self.arms` vi `_build_arms` doc no.
        self.flat_bundle = flat_bundle
        #: Fills in `query_text_en` when the caller has none. None means the only
        #: English is hand-written, which is where the pilot started.
        self.translator = translator
        self.arms = self._build_arms(config)
        # Still no *default* reranker: stage C is the last word on order, so a
        # scorer that is missing, mis-wired or wrong is strictly worse than no
        # stage at all. That was not theoretical — the old hash stub took a
        # ranking with Motorcycle/Wheel/Tire on top for "a person riding a
        # motorbike" and replaced it with a Laptop/Glasses frame it happened to
        # hash higher. `from_config` now builds a real one when
        # `models.reranker` names a checkpoint, and leaves this None otherwise.
        self.reranker = reranker

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> Cascade:
        """Open the stores named in config and build a cascade over them.

        A store that will not load is a warning, not a failure: a repo where
        ingest has not run should still start the browser and say what is
        missing, rather than refusing at import time.
        """
        from mfe.online.encoder import DEFAULT_TEXT_CHECKPOINT, encode_query_text

        # `VectorIndex` va `InvertedIndex` duoc import BEN TRONG `load()`, khong
        # phai o day. `VectorIndex` keo theo `faiss`, va mot import o dau ham thi
        # chay VO DIEU KIEN — ke ca khi nguoi goi da truyen `vector_index=None`
        # vi ho khong dung kho local. Tren Kaggle (duong `flat_bundle`) do la
        # `ModuleNotFoundError: No module named 'faiss'` cho mot doan ma khong ai
        # dinh chay: phu thuoc 100 MB bat buoc cho mot nhanh bi bo qua.
        text_checkpoint = getattr(
            getattr(config, "retrieval", None), "clip_text_checkpoint", DEFAULT_TEXT_CHECKPOINT
        )

        def load(label: str, loader: Any) -> Any:
            try:
                return loader()
            except Exception as err:  # noqa: BLE001 — "not built yet" is expected
                logger.warning("%s unavailable (%s); run mfe-ingest", label, err)
                return None

        return cls(
            config=config,
            vector_index=overrides.get(
                "vector_index",
                load("vector index", lambda: _load_vector_index(config)),
            ),
            inverted_index=overrides.get(
                "inverted_index",
                load("inverted index", lambda: _load_inverted_index(config)),
            ),
            encode_query=overrides.get(
                "encode_query",
                lambda text: encode_query_text(text, checkpoint=text_checkpoint),
            ),
            reranker=(
                overrides["reranker"]
                if "reranker" in overrides
                else load("reranker", lambda: cls._build_reranker(config))
            ),
            text_store=(
                overrides["text_store"]
                if "text_store" in overrides
                else load("text store", cls._build_text_store)
            ),
            dense_store_factory=overrides.get("dense_store_factory", _milvus_store_factory),
            # PHAI chuyen tiep. `from_config` dung `cls(...)` bang danh sach khoa
            # tuong minh, nen mot `**overrides` khong duoc liet ke o day se bi BO
            # QUA IM LANG: `Cascade.from_config(cfg, flat_bundle=b)` chay binh
            # thuong roi tra ve 0/12 arm, vi `_build_arms` thay `self.flat_bundle`
            # la None va roi ve duong Elasticsearch/Milvus khong ton tai.
            flat_bundle=overrides.get("flat_bundle"),
            translator=(
                overrides["translator"]
                if "translator" in overrides
                # `cls.` — KHÔNG bỏ được. Thiếu nó thì đây là `NameError`, mà
                # `load()` bắt mọi lỗi và trả `None`, nên hỏng này không nổ:
                # translator = None, `resolve_english` không có bản tiếng Anh,
                # và arm `siglip` tự tắt vì tháp text của nó chỉ hiểu tiếng Anh.
                # Mất một arm mạnh nhất, dấu vết duy nhất là một dòng log.
                else load("translator", lambda: cls._build_translator_with_quest(config))
            ),
        )

    @staticmethod
    def _build_translator_with_quest(config):
        from mfe.online.translate import build_translator
        try:
            from mfe.online.quest import QuestKIS
            backends = {"llm": QuestKIS()}
        except ImportError:
            backends = None
        return build_translator(config, extra_backends=backends)

    @staticmethod
    def _build_text_store() -> Any:
        """Open Elasticsearch from the environment, or None if it is not there.

        Endpoint and index come from env (`MFE_ELASTICSEARCH_URL`,
        `MFE_ELASTICSEARCH_INDEX`) rather than YAML because they differ per
        machine — see ke-hoach-merge-sota.md §4. `health()` is checked here so a
        dead service becomes "arm unavailable, here is why" instead of an
        exception on the first query.
        """
        from mfe.stores.production.config import ElasticsearchConfig
        from mfe.stores.production.elasticsearch import ElasticsearchTextStore

        store = ElasticsearchTextStore(ElasticsearchConfig.from_env())
        health = store.health()
        if not health.connection_ok or not health.index_ready:
            raise RuntimeError(f"Elasticsearch not ready: {health.message}")
        return store

    @staticmethod
    def _build_reranker(config: Any) -> Reranker | None:
        """Stage C from config, or None when `models.reranker` is null.

        Needs the Asset Store: stage C scores the query against the *image*, and
        the Asset Store is what turns a `keyframe_id` into pixels. No Asset
        Store means no stage C — reordering on anything else would not be early
        fusion, it would be a second text pass wearing its name.
        """
        checkpoint = getattr(getattr(config, "models", None), "reranker", None)
        if not checkpoint:
            return None

        from mfe.core.storage import get_backend
        from mfe.stores.asset.manifest import AssetStore

        backend = get_backend(
            config.storage.backend, root=config.repo_root, volume=config.storage.volume
        )
        asset_store = AssetStore.load(config.paths.asset_store, backend=backend)
        return Reranker(
            asset_store=asset_store,
            model_name=checkpoint,
            budget_ms=config.retrieval.rerank_budget_ms,
        )

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        run_id: str = "cascade_run",
        query_text_en: str | None = None,
        ocr_keywords: str | None = None,
        clauses: Sequence[str] | None = None,
    ) -> list[Candidate]:
        """Run stage A -> B -> C and return a ranked candidate list.

        Args:
            query_text: The description to search for, in Vietnamese.
            top_k: How many to return. Defaults to `retrieval.rerank_topk`,
                which config sets to the 100 the scoring rewards.
            run_id: Tracing id; ends up inside every `evidence_id`.
            query_text_en: Optional English form. Arms whose model has no
                Vietnamese (SigLIP's text tower, CLAP, ColPali) need it and sit
                out without it — see `Arm.query_language`. Left unset, a
                configured translator fills it in (`mfe.online.translate`).
            ocr_keywords: Optional Vietnamese keywords for text/OCR arms.
            clauses: Scene clauses in **narrative order**, supplied by a person
                (or an LLM) instead of letting `scene_order.split_clauses` cut
                the query itself. This is the *better* source, not a fallback:
                on 38 real BTC KIS papers, 21 carry sequence cues but the rule
                only reaches ≥3 clauses on **3**, because the guards that stop
                it shredding a single-moment query also stop it splitting hard
                cases. A reader has no such trade-off, and at competition time
                there is a person at the keyboard. Ignored for TRAKE, whose
                events are already an explicit ordered list.
        """
        top_k = top_k or self._cfg("rerank_topk", 100)
        query_text_en = self.resolve_english(query_text, query_text_en)

        coarse = self._coarse(
            query_text, run_id=run_id,
            query_text_en=query_text_en, ocr_keywords=ocr_keywords,
        )
        if not coarse:
            return []

        coarse = self._scene_order(coarse, query_text, run_id=run_id, clauses=clauses)
        grouped = self.group(coarse)

        grouped = self._rerank_tiers(grouped, query_text, query_text_en)

        if self.reranker is None:
            return grouped[:top_k]

        # Stage C is scored under a budget and truncates rather than raising
        # (mfe.online.rerank), so there is no timeout to catch here.
        #
        # `rerank_depth` decides how deep stage C reaches; the tail keeps its
        # stage B order. Default is the full submission depth, not a shallow
        # head: measured on the 2026 sample, reranking all 100 moved a correct
        # keyframe from rank 92 to rank 3, which a depth-25 pass never sees.
        # See mfe.online.rerank for the numbers and what they cost.
        depth = self._cfg("rerank_depth", 100)
        head = grouped[:depth]
        tail = grouped[len(head) :]
        reranked = self._rerank(query_text, head, run_id=run_id)

        return (reranked + tail)[:top_k]

    def resolve_english(self, query_text: str, query_text_en: str | None) -> str | None:
        """The English form to hand the English-only arms, or None.

        Order of precedence, and the two skips that keep it cheap:

        1. A hand-written `query_text_en` wins outright — the test set stays
           reproducible and a bad machine translation stays overridable.
        2. No translator configured: nothing to do.
        3. **No active arm reads English:** do not translate. Otherwise a run
           with SigLIP switched off would still pay to load a 1.2 GB checkpoint
           and translate every query for nobody.

        A translator that returns None is not an error here. The arms handle it:
        they sit out, loudly. The Vietnamese is never substituted.

        Precedence itself lives in `QueryTranslator.translate`, not here — the
        hand-written form is passed *into* it rather than checked before it, so
        there is one implementation of "manual wins" and its usage counter sees
        every query rather than only the translated ones.
        """
        manual = query_text_en if query_text_en and query_text_en.strip() else None
        if self.translator is None or not self.arms.needs_english():
            return manual
        translation = self.translator.translate(query_text, manual=manual)
        return translation.text_en if translation is not None else None

    def _coarse(
        self,
        query_text: str,
        run_id: str = "cascade_run",
        query_text_en: str | None = None,
        ocr_keywords: str | None = None,
    ) -> list[Candidate]:
        """Stage A — late fusion over every active arm (`mfe.online.arms`)."""
        coarse_topk = self._cfg("coarse_topk", 500)
        result_lists = self.arms.run(
            query_text, coarse_topk, run_id, query_text_en=query_text_en, ocr_keywords=ocr_keywords
        )

        if not result_lists:
            logger.warning(
                "no arm produced candidates for %r; arms: %s",
                query_text,
                ", ".join(f"{a.name}={a.status()}" for a in self.arms),
            )
            return []

        fusion_cfg = getattr(getattr(self.config, "retrieval", None), "fusion", None)
        normalizer = fusion_cfg.resolved_normalizer() if fusion_cfg else "rrf"
        rrf_k = fusion_cfg.rrf_k if fusion_cfg else 60

        return weighted_fusion(
            result_lists,
            weights=self.arms.weights(),
            run_id=run_id,
            normalizer=normalizer,
            rrf_k=rrf_k,
        )

    def _scene_order(
        self, coarse: list[Candidate], query_text: str, run_id: str,
        clauses: Sequence[str] | None = None,
    ) -> list[Candidate]:
        """Thiên kiến thứ tự cảnh cho KIS nhiều cảnh (`mfe.online.scene_order`).

        Đề KIS thật của BTC thường tả một *đoạn* chứ không một khung — 21/38 đề
        KIS mang từ nối trình tự. Ta đánh chỉ mục từng khung rời nên cấu trúc đó
        bị vứt: hai mệnh đề đi vào cùng một túi từ, và một video có *một* trong
        hai cảnh ghi điểm y như video có **cả hai, đúng thứ tự**.

        Đo trên 14 câu KIS ba cảnh (`eval/queryset_order_l30.jsonl`, video L30
        chưa câu nào khác đụng tới): **RỘNG 0,800 → 0,843 · hạng TB 10,3 → 4,2 ·
        4 lên / 0 xuống**, và DANTE xếp **đúng video lên hạng nhất ở 14/14 câu**
        trên 873 video. Trên 189 câu dev: 4 câu bị chạm, +0,050; **111 câu còn
        lại giữ nguyên từng con số**.

        **Ba cửa thoát sớm, theo thứ tự rẻ dần** — vì hàm này chạy lại tầng A
        cho *từng* mệnh đề, tức nhân chi phí truy hồi lên 3 lần:

        1. `scene_order_weight <= 0` — tắt, và đây là mặc định.
        2. Câu tách ra dưới `MIN_CLAUSES_FOR_PRIOR` mệnh đề. Đây là đường ra của
           **111/115** câu KIS bộ dev, nên phép tách phải đứng trước mọi thứ đắt.
        3. Không video nào ghép được chuỗi tăng dần -> `prior` rỗng ->
           `apply_prior` là phép không-làm-gì thật sự.

        **Cần `RelationalMap`**, và không có thì tầng này **ngồi ngoài kèm cảnh
        báo** chứ không chạy bằng số khác. `EventHit.ordinal` phải là
        `keyframe_index`: `dante` nhân λ với nó, và nếu đưa *thứ hạng trong danh
        sách* vào thay thì hai keyframe cách nhau 200 frame bị phạt y như hai
        keyframe liền nhau (xem docstring `mfe.online.dante` §1). Đưa `frame_idx`
        vào cũng sai theo kiểu khác: khoảng cách gấp ~25 lần nên λ = 0,001 hoá ra
        mạnh gấp 25, tức một cấu hình chưa ai đo.
        """
        weight = float(self._cfg("scene_order_weight", 0.0) or 0.0)
        if weight <= 0:
            return coarse

        from mfe.online import scene_order  # noqa: PLC0415 - chi khi bat

        # Mệnh đề người/LLM tách sẵn thắng bộ luật — xem `search(clauses=…)`.
        # Ngưỡng ≥3 vẫn áp cho **cả hai** nguồn: nó nói về *số cảnh*, không nói
        # về cách tách, và câu hai cảnh đo được là luôn gây hại dù ai tách.
        split = (
            scene_order.clauses_from_texts(clauses) if clauses
            else scene_order.split_clauses(query_text)
        )
        if len(split) < scene_order.MIN_CLAUSES_FOR_PRIOR:
            return coarse

        relational = self._relational_map()
        if relational is None:
            logger.warning(
                "thu tu canh: bat (%.2f) nhung khong mo duoc Relational Map — "
                "tang ngoi ngoai. Can `keyframe_index` cho phep phat cua DANTE.",
                weight,
            )
            return coarse

        per_clause = []
        for index, clause in enumerate(split):
            # `-c{i}`, KHONG `#c{i}`: `run_id` di thang vao `evidence_id`, va `#`
            # truot pattern cua `Candidate` -> `ArmRegistry.run` nuot
            # `ValidationError` roi ghi "arm ... returned nothing". Moi arm tra
            # rong ma khong co gi do len. Da mat mot luot do vi dung chuyen nay.
            hits = self._coarse(
                clause.text, run_id=f"{run_id}-c{index}",
                query_text_en=self.resolve_english(clause.text, None),
            )
            events = []
            for candidate in hits:
                try:
                    row = relational.get_keyframe(candidate.keyframe_id)
                except Exception:  # noqa: BLE001 - keyframe la thi bo qua rieng no
                    continue
                events.append(scene_order.EventHit(
                    keyframe_id=candidate.keyframe_id,
                    video_id=candidate.video_id,
                    ordinal=row.keyframe_index,
                    frame_id=row.frame_idx,
                    score=float(candidate.score),
                ))
            per_clause.append(events)

        prior = scene_order.video_prior(per_clause)
        if not prior:
            return coarse
        logger.info(
            "thu tu canh: %d menh de, %d video ghep duoc chuoi tang dan",
            len(split), len(prior),
        )
        return scene_order.apply_prior(coarse, prior, weight=weight)

    def _rerank_tiers(
        self, grouped: list[Candidate], query_text: str, query_text_en: str | None
    ) -> list[Candidate]:
        """Các tầng trong `retrieval.rerank_tiers`, chạy theo đúng thứ tự khai.

        **Rỗng theo mặc định** — đó là hành vi đã chạy từ trước khi có trường
        này, và ba trong bốn tầng đo được là ÂM trên nền đã hiệu chỉnh (163 câu
        dev, 27-08): `ta` −0,010 · `hf_cross` −0,029 · `colpali` −0,033. Chỉ
        `vlm` dương, **+0,015 với 27,6 giây/câu**.

        Một tầng không chạy được thì **ghi cảnh báo rồi đi tiếp**, không ném:
        `TierOutcome.reason` nói vì sao (thiếu khoá API, hết quota, không lấy
        được ảnh), và mất một tầng còn hơn mất cả lượt nộp. Nhưng nó phải **nói
        ra** — một tầng im lặng không chạy in ra `+0,000` giống hệt một tầng
        chạy mà vô ích, và cột `chạy` là chỗ duy nhất phân biệt (CLAUDE.md §4).
        """
        names = self._cfg("rerank_tiers", None) or []
        if not names:
            return grouped

        from mfe.online.keyframe_images import KeyframeImages  # noqa: PLC0415
        from mfe.online.rerank_tiers import apply_tiers, build_tier  # noqa: PLC0415

        if getattr(self, "_tier_cache", None) is None:
            self._tier_images = KeyframeImages()
            self._tier_cache = [build_tier(name, self._tier_images) for name in names]

        depth = self._cfg("rerank_tier_depth", 20)
        texts = None
        if self.flat_bundle is not None:
            def texts(keyframe_id: str) -> str:  # noqa: E306 - dong bo voi tren
                parts = [
                    self.flat_bundle.texts[source].get(keyframe_id, "")
                    for source in self.flat_bundle.texts
                ]
                return " ".join(p for p in parts if p)

        after, outcomes = apply_tiers(
            grouped, self._tier_cache,
            query_vi=query_text, query_en=query_text_en,
            images=self._tier_images, texts=texts, depth=depth,
        )
        for outcome in outcomes:
            if outcome.ran:
                logger.info("%s", outcome.summary())
            else:
                logger.warning("tang %s NGOI NGOAI: %s", outcome.tier, outcome.reason)
        return after

    def _relational_map(self) -> Any:
        """Relational Map, mở một lần rồi giữ. `None` khi không mở được.

        Mở **lười** chứ không trong `__init__`: `Cascade` dựng được ở nhiều chỗ
        không có CSDL (test, bàn thử), và bắt nó mở bản đồ ngay lúc dựng sẽ biến
        một tính năng tuỳ chọn thành một điều kiện bắt buộc.

        Giữ cả lần thất bại (`False`) để không thử lại mỗi truy vấn — một câu
        cảnh báo mỗi lượt chạy là đủ, một câu mỗi truy vấn là rác.
        """
        cached = getattr(self, "_relational_cache", None)
        if cached is not None:
            return cached or None
        path = getattr(getattr(self.config, "paths", None), "relational_db", None)
        if not path:
            self._relational_cache = False
            return None
        try:
            from mfe.mapping.relational import RelationalMap  # noqa: PLC0415

            self._relational_cache = RelationalMap(path)
        except Exception as err:  # noqa: BLE001
            logger.warning("khong mo duoc Relational Map tu %s: %s", path, err)
            self._relational_cache = False
            return None
        return self._relational_cache

    def _vector_arm(
        self, query_text: str, top_k: int, run_id: str
    ) -> list[Candidate] | None:
        """Encode the query and search Faiss. None when the arm is unavailable.

        A failure here degrades to a text-only stage A and says so. That is a
        real loss of recall on descriptive queries, so it is logged at warning
        rather than swallowed — running one-armed without noticing is how a
        team spends a week tuning fusion weights that never applied.
        """
        if self.vector_index is None or self.encode_query is None:
            return None
        try:
            embedding = self.encode_query(query_text)
            return self.vector_index.search(
                query_embedding=embedding.tolist(), top_k=top_k, run_id=run_id
            )
        except Exception as err:  # noqa: BLE001 — degrade, do not abort the query
            logger.warning("vector arm unavailable (%s); stage A is text-only", err)
            return None

    def group(self, candidates: list[Candidate]) -> list[Candidate]:
        """Stage B — dedup + temporal grouping, applied only to the head.

        Diversity is worth something on the first screen an operator looks at,
        and worth nothing below it: R@20/50/100 read 100 deep and only ask
        whether a correct frame is present, not whether its neighbours are
        interesting. So only the first `retrieval.dedup_depth` slots are
        reordered; past that the fused ranking is passed through untouched.

        Promoting *every* representative ahead of *every* non-representative —
        what this did before — is a much larger intervention than it looks. On
        the 2026 sample it produced 236 representatives, so a candidate that
        fused at rank 60 came out at rank 270: outside the 100 answers we are
        allowed to submit, scoring 0.0 on a query the vector arm had ranked
        30th. Stage B was quietly deciding the score, and losing.
        """
        window = self._cfg("temporal_window_frames", DEFAULT_TEMPORAL_WINDOW_FRAMES)
        depth = self._cfg("dedup_depth", 20)
        if depth <= 0:
            return candidates

        representatives = dedup(candidates, temporal_window_frames=window)[:depth]
        chosen = {candidate.keyframe_id for candidate in representatives}
        remainder = [c for c in candidates if c.keyframe_id not in chosen]
        return representatives + remainder

    def _rerank(
        self,
        query_text: str,
        candidates: list[Candidate],
        run_id: str = "cascade_run",
    ) -> list[Candidate]:
        """Stage C — early fusion on original images."""
        return self.reranker.rerank(query=query_text, candidates=candidates, run_id=run_id)

    # --- config plumbing ---------------------------------------------------

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(getattr(self.config, "retrieval", None), name, default)

    def _build_arms(self, config: Any) -> ArmRegistry:
        """Register the arms this cascade can run.

        Arms are named after the config weight they read — `bm25`, `clip` — so
        there is no translation table between "what fusion calls it" and "what
        config calls it". That table existed and was one more place to drift.

        An arm whose store did not load is registered anyway, `available=False`
        with a reason. Leaving it out entirely would make the browser's switch
        list change shape depending on what happened to load, and would give an
        operator no way to find out why their arm is silent.
        """
        weights = getattr(getattr(config, "retrieval", None), "fusion_weights", None)

        def weight_of(name: str) -> float:
            return float(getattr(weights, name, 1.0)) if weights else 1.0

        registry = ArmRegistry()

        registry.register(
            Arm(
                name="bm25",
                search=(
                    (
                        lambda q, k, run: self.inverted_index.search(
                            query_text=q, top_k=k, run_id=run
                        )
                    )
                    if self.inverted_index is not None
                    else None
                ),
                weight=weight_of("bm25"),
                unavailable_reason=(
                    None if self.inverted_index is not None else "inverted index not loaded"
                ),
                description="BM25 over object labels / OCR / metadata",
            )
        )

        vector_ready = self.vector_index is not None and self.encode_query is not None
        registry.register(
            Arm(
                name="clip",
                search=(
                    (lambda q, k, run: self._vector_arm(q, k, run) or []) if vector_ready else None
                ),
                weight=weight_of("clip"),
                unavailable_reason=(
                    None
                    if vector_ready
                    else "vector index or query encoder not loaded"
                ),
                description="clip-ViT-B-32 cosine over the Faiss index",
            )
        )

        from mfe.online.dense_arms import DENSE_ARM_SPECS
        from mfe.online.text_arms import TEXT_ARM_SPECS

        arm_weights = {name: weight_of(name) for name in TEXT_ARM_SPECS}
        dense_weights = {name: weight_of(name) for name in DENSE_ARM_SPECS}

        if self.flat_bundle is not None:
            # Đường KHÔNG KHO: đọc bản gộp trong RAM. Cùng tên arm, cùng
            # `Candidate`, cùng `query_language` — nên fusion, dedup, DANTE và
            # bảng arm phía sau không biết chúng đang chạy trên kho nào. Đó là
            # điều kiện để so được kết quả hai đường; đổi tên arm là mất phép so.
            from mfe.online.flat_arms import build_flat_dense_arms, build_flat_text_arms

            for arm in build_flat_text_arms(self.flat_bundle, weights=arm_weights):
                registry.register(arm)
            for arm in build_flat_dense_arms(self.flat_bundle, weights=dense_weights):
                registry.register(arm)
            return registry

        # Elasticsearch-backed arms — one per `source` from SOTA_AIC's exports.
        # Registered even when the store is absent, marked unavailable with a
        # reason (mfe.online.text_arms).
        from mfe.online.text_arms import build_text_arms

        for arm in build_text_arms(self.text_store, weights=arm_weights):
            registry.register(arm)

        # Milvus-backed dense arms, one collection per embedding model.
        from mfe.online.dense_arms import build_dense_arms

        for arm in build_dense_arms(
            self.dense_store_factory or _no_dense_store, weights=dense_weights
        ):
            registry.register(arm)

        return registry
