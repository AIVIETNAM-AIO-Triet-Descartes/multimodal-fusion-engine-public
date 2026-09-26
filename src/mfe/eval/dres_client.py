"""Small DRES v2 client and payload builders used by the HTML cockpit.

The browser never receives the DRES credentials or session token.  The server
reads them from ``.env`` and keeps the short lived session in this process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://eventretrieval.one"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,200}$")


class DresError(RuntimeError):
    """A DRES request failed without exposing credentials."""


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def dres_credentials() -> tuple[str | None, str | None]:
    # Accept the two spellings used by the operator's existing .env files.
    return _env_first("user_dres", "user_dress"), _env_first("pass_dres", "pass_dress")


def configured() -> bool:
    username, password = dres_credentials()
    return bool(username and password)


def submission_key(
    evaluation_id: str,
    state: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> str:
    """Stable key for one exact DRES answer.

    DRES rejects an identical answer, but it allows a different answer for the
    same query so a team can recover from an earlier wrong attempt.  The web
    guard therefore includes the serialized answer payload when available;
    older callers can still request the legacy query-only fingerprint.
    """
    identity = {
        "evaluation_id": evaluation_id,
        "query_type": str(state.get("query_type", "")).upper(),
        "query": str(state.get("query", "")),
        "events": [str(item) for item in state.get("events", [])]
        if isinstance(state.get("events"), list)
        else [],
    }
    if payload is not None:
        identity["payload"] = payload
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DresClient:
    username: str
    password: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 20
    session_id: str | None = None

    @classmethod
    def from_env(cls) -> DresClient | None:
        username, password = dres_credentials()
        if not username or not password:
            return None
        return cls(username, password, os.getenv("MFE_DRES_BASE_URL", DEFAULT_BASE_URL).rstrip("/"))

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "mfe-html/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise DresError(f"DRES HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DresError(f"DRES khong ket noi duoc: {type(exc).__name__}") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise DresError("DRES tra ve JSON khong hop le") from exc

    def login(self) -> str:
        result = self._request(
            "POST",
            "/api/v2/login",
            payload={"username": self.username, "password": self.password},
        )
        token = str(result.get("sessionId", "")) if isinstance(result, dict) else ""
        if not token:
            raise DresError("DRES dang nhap khong tra sessionId")
        self.session_id = token
        return token

    def _session(self) -> str:
        return self.session_id or self.login()

    def evaluations(self) -> list[dict[str, Any]]:
        result = self._request(
            "GET", "/api/v2/client/evaluation/list", query={"session": self._session()}
        )
        if not isinstance(result, list):
            raise DresError("DRES evaluation/list khong tra ve danh sach")
        return [item for item in result if isinstance(item, dict)]

    def submit(self, evaluation_id: str, payload: dict[str, Any]) -> Any:
        if not _ID_RE.fullmatch(evaluation_id):
            raise DresError("evaluation_id khong hop le")
        return self._request(
            "POST",
            f"/api/v2/submit/{urllib.parse.quote(evaluation_id, safe='')}",
            query={"session": self._session()},
            payload=payload,
        )


def build_selected_payload(
    query_type: str,
    item: dict[str, Any],
    *,
    answer: str = "",
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build exactly one DRES answer from a sanitized web result item."""
    kind = str(query_type).upper()
    video = str(item.get("video_id", ""))
    if not _ID_RE.fullmatch(video):
        raise DresError("video_id khong hop le")
    if kind == "KIS":
        time_ms = int(item.get("timestamp_ms", 0))
        entry = {"mediaItemName": video, "start": time_ms, "end": time_ms}
    elif kind == "QA":
        if not str(answer).strip():
            raise DresError("QA chua co dap an")
        time_ms = int(item.get("timestamp_ms", 0))
        entry = {"text": f"QA-{str(answer).strip()}-{video}-{time_ms}"}
    elif kind == "TRAKE":
        event_items = events if events is not None else item.get("events", [])
        frames = [int(event["frame_idx"]) for event in event_items if isinstance(event, dict)]
        if not frames or len(frames) != len(event_items):
            raise DresError("TRAKE thieu frame cho mot hoac nhieu su kien")
        if any(a >= b for a, b in zip(frames, frames[1:], strict=False)):
            raise DresError("frame TRAKE phai tang nghiem ngat")
        entry = {"text": f"TR-{video}-" + ",".join(map(str, frames))}
    else:
        raise DresError("query_type khong ho tro")
    return {"answerSets": [{"answers": [entry]}]}


def apply_selected_frame(item: dict[str, Any], selected: Any) -> dict[str, Any]:
    """Overlay one manually selected player frame onto a result item.

    The browser may seek to a frame that was not one of the retrieval
    candidates.  Keep the original result as the source of truth for the
    video, but allow the operator to replace its timestamp/frame after
    checking the video.  The video id is deliberately locked to the result so
    a client cannot turn a valid result into an answer for another video.
    """
    if not isinstance(item, dict) or not isinstance(selected, dict):
        raise DresError("frame da chot khong hop le")
    video = str(item.get("video_id", ""))
    selected_video = str(selected.get("video_id", video))
    if not _ID_RE.fullmatch(video) or selected_video != video:
        raise DresError("frame da chot khong cung video voi ket qua")
    try:
        frame_idx = int(selected["frame_idx"])
        timestamp_ms = int(selected["timestamp_ms"])
    except (KeyError, TypeError, ValueError):
        raise DresError("frame da chot thieu frame_idx hoac timestamp_ms") from None
    if frame_idx < 0 or timestamp_ms < 0:
        raise DresError("frame da chot phai khong am")
    updated = dict(item)
    updated.update(
        {
            "video_id": video,
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "seconds": round(timestamp_ms / 1000.0, 3),
            "keyframe_id": f"{video}#{frame_idx:06d}",
        }
    )
    return updated
