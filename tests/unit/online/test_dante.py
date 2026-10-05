"""Khoa DANTE — quy hoach dong gan N su kien vao N vi tri tang dan.

    arXiv:2512.13169 §4.3 · docs/triet/2512.13169v1.pdf

Cai duoc kiem o day la TINH CHAT, khong phai con so: thu tu tang ngat, phep phat
uu tien gan ma khong cam xa, backtrack tra ve dung N vi tri, va video khong ghep
duoc thi bi BO chu khong duoc don frame.
"""

from __future__ import annotations

import pytest

from mfe.contracts.errors import ContractError
from mfe.online.dante import EventHit, align, align_video, normalise_events


def hit(video: str, ordinal: int, score: float) -> EventHit:
    """Mot ung vien. `frame_id` = ordinal * 90, du de phan biet hai truc."""
    return EventHit(
        keyframe_id=f"{video}#{ordinal * 90:06d}",
        video_id=video,
        ordinal=ordinal,
        frame_id=ordinal * 90,
        score=score,
    )


class TestTemporalOrder:
    def test_events_come_back_strictly_increasing(self):
        """Rang buoc lam TRAKE khac mot phep xep hang: su kien i+1 khong the
        xay ra truoc su kien i."""
        per_event = [
            [hit("V1", 10, 0.9), hit("V1", 50, 0.4)],
            [hit("V1", 20, 0.8), hit("V1", 5, 0.95)],
            [hit("V1", 40, 0.7), hit("V1", 15, 0.99)],
        ]
        result = align_video(per_event, penalty=0.0)
        ordinals = [h.ordinal for h in result.hits]
        assert ordinals == sorted(ordinals)
        assert len(set(ordinals)) == 3

    def test_the_highest_scoring_pick_is_rejected_when_it_breaks_order(self):
        """Su kien 1 co diem cao nhat o vi tri 90, nhung chon no thi su kien 2
        va 3 khong con cho — DP phai thay va bo qua."""
        per_event = [
            [hit("V1", 90, 1.0), hit("V1", 10, 0.5)],
            [hit("V1", 20, 0.5)],
            [hit("V1", 30, 0.5)],
        ]
        result = align_video(per_event, penalty=0.0)
        assert [h.ordinal for h in result.hits] == [10, 20, 30]

    def test_fewer_positions_than_events_is_unsolvable(self):
        per_event = [[hit("V1", 5, 1.0)], [hit("V1", 5, 1.0)], [hit("V1", 5, 1.0)]]
        assert align_video(per_event, penalty=0.0) is None

    def test_a_single_event_still_works(self):
        result = align_video([[hit("V1", 7, 0.3), hit("V1", 9, 0.8)]], penalty=0.0)
        assert [h.ordinal for h in result.hits] == [9]


class TestPenalty:
    def test_penalty_prefers_the_nearer_of_two_equal_scores(self):
        """λ la thu phan biet mot chuoi hanh dong voi hai khoanh khac tinh co
        giong nhau o hai dau video."""
        per_event = [
            [hit("V1", 10, 1.0)],
            [hit("V1", 12, 1.0), hit("V1", 900, 1.0)],
        ]
        assert [h.ordinal for h in align_video(per_event, penalty=0.01).hits] == [10, 12]

    def test_penalty_does_not_forbid_a_distant_match_that_scores_better(self):
        """Paper noi ro: phat KHUYEN KHICH gan chu khong LOAI xa. Mot chuoi su
        kien thua van ghep duoc neu do tuong dong du manh."""
        per_event = [
            [hit("V1", 10, 1.0)],
            [hit("V1", 12, 0.10), hit("V1", 900, 1.0)],
        ]
        assert [h.ordinal for h in align_video(per_event, penalty=0.001).hits] == [10, 900]

    def test_a_big_enough_penalty_flips_that_same_choice(self):
        """Cung du lieu, doi mot λ — de thay λ THAT SU dieu khien phep chon chu
        khong phai mot tham so trang tri."""
        per_event = [
            [hit("V1", 10, 1.0)],
            [hit("V1", 12, 0.10), hit("V1", 900, 1.0)],
        ]
        assert [h.ordinal for h in align_video(per_event, penalty=0.01).hits] == [10, 12]

    def test_zero_penalty_is_plain_max_sum_under_order(self):
        per_event = [
            [hit("V1", 1, 0.5), hit("V1", 2, 0.9)],
            [hit("V1", 3, 0.9), hit("V1", 9, 0.5)],
        ]
        assert [h.ordinal for h in align_video(per_event, penalty=0.0).hits] == [2, 3]

    def test_penalty_measures_ORDINAL_distance_not_list_position(self):
        """Chỗ dễ sai nhất khi thu `t` về danh sách thưa.

        Hai ứng viên đứng **cạnh nhau trong danh sách** nhưng cách nhau 800
        keyframe phải bị phạt như 800, không phải như 1. Tính theo chỉ số danh
        sách thì hai lựa chọn dưới đây hoà nhau và λ mất hết ý nghĩa.
        """
        per_event = [
            [hit("V1", 100, 1.0)],
            [hit("V1", 101, 0.5), hit("V1", 900, 0.5)],
        ]
        # Hai ung vien cua su kien 2 co diem BANG NHAU; chi khoang cach phan biet.
        assert [h.ordinal for h in align_video(per_event, penalty=0.01).hits] == [100, 101]

    def test_a_negative_penalty_is_refused(self):
        with pytest.raises(ContractError, match="penalty"):
            align_video([[hit("V1", 1, 1.0)]], penalty=-0.5)


class TestSparseScores:
    def test_an_event_may_land_on_a_position_it_did_not_retrieve(self):
        """Paper tinh S tren MOI t. Giu tinh chat do nghia la mot su kien yeu
        van 'di nho' qua vi tri do su kien khac tim ra, thay vi lam ca video mat
        kha nang ghep."""
        per_event = [
            [hit("V1", 10, 1.0), hit("V1", 20, 0.1)],
            [],                       # su kien 2 khong truy hoi duoc gi
            [hit("V1", 30, 1.0)],
        ]
        result = align_video(per_event, penalty=0.0)
        assert result is not None
        # Su kien 2 dat vao 20 — vi tri do su kien 1 tim ra, voi S = 0.
        assert [h.ordinal for h in result.hits] == [10, 20, 30]

    def test_an_event_with_no_hits_still_needs_a_free_position(self):
        """Khong the gan 3 su kien vao 2 vi tri, du sự kiện rỗng có 'miễn phí'."""
        per_event = [[hit("V1", 10, 1.0)], [], [hit("V1", 30, 1.0)]]
        assert align_video(per_event, penalty=0.0) is None

    def test_normalisation_makes_events_comparable(self):
        """Hai sự kiện có thang điểm lệch nhau 100 lần vẫn phải cân nhau.

        Điểm sau RRF phụ thuộc số arm đang bật, nên không chuẩn hoá thì một sự
        kiện tình cờ được nhiều arm bỏ phiếu sẽ áp đảo cả chuỗi.
        """
        per_event = [
            [hit("V1", 10, 100.0), hit("V1", 11, 99.0)],
            [hit("V1", 20, 1.0), hit("V1", 21, 0.5)],
        ]
        scaled = normalise_events(per_event)
        assert [round(h.score, 3) for h in scaled[0]] == [1.0, 0.99]
        assert [round(h.score, 3) for h in scaled[1]] == [1.0, 0.5]

    def test_normalisation_is_global_not_per_video(self):
        """Lỗi đã mắc rồi sửa: chuẩn hoá bên trong từng video làm video nào cũng
        có ứng viên tốt nhất = 1.0, nên `Alignment.score` bằng nhau và **thứ hạng
        video biến mất**. Với TRAKE đó là hỏng đúng thứ quan trọng nhất — sai
        video là 0 điểm ngay lập tức.
        """
        per_event = [
            [hit("A", 1, 0.2), hit("B", 1, 0.9)],
            [hit("A", 2, 0.2), hit("B", 2, 0.9)],
        ]
        scaled = normalise_events(per_event)
        by_video = {h.video_id: h.score for h in scaled[0]}
        assert by_video["B"] == 1.0
        assert by_video["A"] == pytest.approx(0.2 / 0.9)

    def test_all_zero_scores_do_not_divide_by_zero(self):
        per_event = [[hit("V1", 1, 0.0)], [hit("V1", 2, 0.0)]]
        result = align_video(per_event, penalty=0.0)
        assert [h.ordinal for h in result.hits] == [1, 2]


class TestAcrossVideos:
    def test_each_video_yields_at_most_one_answer(self):
        """Mot cau tra loi TRAKE neu DUNG MOT video (PDF §2.1.3)."""
        per_event = [
            [hit("V1", 1, 0.9), hit("V2", 1, 0.8)],
            [hit("V1", 5, 0.9), hit("V2", 5, 0.8)],
        ]
        results = align(per_event, penalty=0.0)
        assert [a.video_id for a in results] == ["V1", "V2"]
        assert len({a.video_id for a in results}) == len(results)

    def test_a_video_that_cannot_be_ordered_is_dropped_not_padded(self):
        """Mot frame don ghi diem nhu mot lan doan sai va khong phan biet duoc
        voi mot lan thu that."""
        per_event = [
            [hit("V1", 1, 0.9), hit("V2", 7, 0.99)],
            [hit("V1", 5, 0.9)],                       # V2 khong co ung vien thu 2
        ]
        results = align(per_event, penalty=0.0)
        assert [a.video_id for a in results] == ["V1"]

    def test_videos_are_ranked_by_alignment_score(self):
        per_event = [
            [hit("A", 1, 0.2), hit("B", 1, 0.9)],
            [hit("A", 2, 0.2), hit("B", 2, 0.9)],
        ]
        assert [a.video_id for a in align(per_event, penalty=0.0)] == ["B", "A"]

    def test_limit_caps_the_answer_list(self):
        per_event = [
            [hit(v, 1, 0.5) for v in "ABCDE"],
            [hit(v, 2, 0.5) for v in "ABCDE"],
        ]
        assert len(align(per_event, penalty=0.0, limit=3)) == 3

    def test_mixing_videos_into_align_video_is_refused(self):
        """`align_video` la thuat toan mot video; tron hai video vao no se cho
        ra mot chuoi 'tang dan' nhay giua hai video ma khong ai thay."""
        with pytest.raises(ContractError, match="nhieu video"):
            align_video([[hit("V1", 1, 1.0), hit("V2", 2, 1.0)]], penalty=0.0)

    def test_no_candidates_at_all_is_empty_not_an_error(self):
        assert align([[], []], penalty=0.0) == []

    def test_temporally_contradictory_evidence_is_dropped(self):
        """Su kien 1 tim thay o vi tri 9, su kien 2 o vi tri 2. Loi giai tang dan
        duy nhat gan ca hai vao dung cho CUA NHAU, cho tong diem 0.

        Cau tra loi do ghi 0 diem y nhu khong nop gi, nhung no chiem mot trong
        100 o va lam ban thu hang video. `min_support` loai no.
        """
        per_event = [[hit("V", 9, 0.9)], [hit("V", 2, 0.8)]]
        assert align(per_event, penalty=0.0) == []
        # Ha san xuong 0 thi no quay lai — de thay chinh `min_support` loai no,
        # khong phai mot nhanh nao khac.
        rescued = align(per_event, penalty=0.0, min_support=0)
        assert [h.ordinal for h in rescued[0].hits] == [2, 9]
        assert rescued[0].support == 0

    def test_support_counts_events_standing_on_their_own_evidence(self):
        per_event = [
            [hit("V", 10, 1.0), hit("V", 20, 0.1)],
            [],                       # di nho vi tri 20
            [hit("V", 30, 1.0)],
        ]
        result = align(per_event, penalty=0.0)[0]
        assert result.support == 2
        assert len(result.hits) == 3


class TestOutputShape:
    def test_frame_ids_are_source_video_frames_not_ordinals(self):
        """`ordinal` xep thu tu, `frame_id` la thu BTC cham. Nham hai cai nay la
        nop sai khung hinh — `mfe.contracts.ids` viet dai ve dung dieu do."""
        per_event = [[hit("V1", 3, 1.0)], [hit("V1", 8, 1.0)]]
        result = align_video(per_event, penalty=0.0)
        assert [h.ordinal for h in result.hits] == [3, 8]
        assert result.frame_ids == [270, 720]
        assert result.keyframe_ids == ["V1#000270", "V1#000720"]
