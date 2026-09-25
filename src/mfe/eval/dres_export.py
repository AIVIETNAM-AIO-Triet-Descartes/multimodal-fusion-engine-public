"""One selected answer, in BTC finals DRES format. Does not submit anything."""
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP

from mfe.eval.submission_bundle import apply_manual_order


def milliseconds(seconds):
    """Nearest integer millisecond, half up; never convert through FPS."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("Thiếu thời gian PTS đáng tin cậy.")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Thời gian phải hữu hạn và không âm.")
    return int((Decimal(str(seconds)) * 1000).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def candidate_time_ms(candidate, relational):
    # Explicit coordinates must not fall back to an old catalogue row on ID collision.
    if candidate.frame_idx is not None:
        return milliseconds(candidate.pts_time)
    row = relational.get_keyframe(candidate.keyframe_id)
    if row.video_id != candidate.video_id:
        raise ValueError("Video của khung không khớp bản đồ.")
    return milliseconds(row.pts_time)


def top1_payload(query, result, relational, *, manual_order=None, answer=""):
    if not result:
        raise ValueError("Chưa có ứng viên để xuất.")
    if query.type not in ("KIS", "QA", "TRAKE"):
        raise ValueError("Dạng câu hỏi không được hỗ trợ.")
    ranked = result if query.type == "TRAKE" else apply_manual_order(result, manual_order)
    selected = ranked[0]
    video = selected.video_id
    if not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_]+", video):
        raise ValueError("Tên video phải là ID không có đuôi file.")
    if query.type == "TRAKE":
        frames = list(selected.frame_ids)
        if query.n_events < 1 or len(frames) != query.n_events:
            raise ValueError("Chuỗi TRAKE phải đủ một frame cho mỗi sự kiện.")
        if any(type(f) is not int or f < 0 for f in frames):
            raise ValueError("Frame TRAKE phải là số nguyên không âm.")
        if any(a >= b for a, b in zip(frames, frames[1:])):
            raise ValueError("Frame TRAKE phải tăng nghiêm ngặt theo thứ tự sự kiện.")
        entry = {"text": f"TR-{video}-" + ",".join(map(str, frames))}
    else:
        time_ms = candidate_time_ms(selected, relational)
        if query.type == "QA":
            if not answer.strip():
                raise ValueError("Chưa nhập đáp án QA.")
            entry = {"text": f"QA-{answer.strip()}-{video}-{time_ms}"}
        else:
            entry = {"mediaItemName": video, "start": time_ms, "end": time_ms}
    return {"answerSets": [{"answers": [entry]}]}


def json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
