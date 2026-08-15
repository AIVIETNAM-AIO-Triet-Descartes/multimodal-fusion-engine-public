"""Minimal video browser Streamlit MVP — docs/04 §4.

    streamlit run src/mfe/browser/app.py

Four core capabilities:
    1. Grid ổn định                (top-k, diversity, no layout overflow)
    2. Timeline/neighbor           (prev/next frame in ONE action)
    3. Query refine                (edit query, add/remove concepts)
    4. Gated Submit Panel          (separate view and submit, confirm -> validate)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mfe.contracts.errors import ErrorCode
from mfe.contracts.tools import Candidate, ToolError, ToolResult
from mfe.online.cascade import Cascade
from mfe.online.tools.get_neighbors import get_neighbors
from mfe.online.tools.get_video_segment import get_video_segment

logger = logging.getLogger(__name__)

# Try importing Streamlit safely
try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def validate_submission_stub(candidate: Candidate) -> ToolResult[dict[str, Any]]:
    """Stub validator wrapper for TV4's Validator integration.

    Args:
        candidate: Candidate object selected for submission.

    Returns:
        ToolResult[dict] confirming valid submission status or Error.
    """
    if not candidate.evidence_id:
        return ToolResult(
            tool="validator",
            items=[],
            error=ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message="Candidate missing evidence_id provenance",
                tool="validator",
            ),
        )

    return ToolResult(
        tool="validator",
        items=[
            {
                "status": "VALID",
                "keyframe_id": candidate.keyframe_id,
                "video_id": candidate.video_id,
                "evidence_id": candidate.evidence_id,
            }
        ],
        error=None,
    )


def get_keyframe_image(keyframe_id: str, width: int = 320, height: int = 180) -> Any:
    """Load keyframe image from disk or generate a synthetic preview image.

    Args:
        keyframe_id: Keyframe identifier.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        String file path or PIL Image object.
    """
    from pathlib import Path

    video_id, kf_idx_str = keyframe_id.split("#") if "#" in keyframe_id else (keyframe_id, "001")

    # 1. Check local keyframe store disk paths
    possible_paths = [
        Path(f"data/keyframes/{video_id}/{kf_idx_str}.jpg"),
        Path(f"data/keyframes/{video_id}/{kf_idx_str}.png"),
        Path(f"data/raw/{video_id}/{kf_idx_str}.jpg"),
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)

    # 2. Dynamic synthetic preview image generator for development/demo
    try:
        from PIL import Image, ImageDraw

        hash_val = sum(ord(c) for c in keyframe_id)
        r = (hash_val * 37) % 160 + 40
        g = (hash_val * 59) % 160 + 40
        b = (hash_val * 83) % 160 + 40

        img = Image.new("RGB", (width, height), color=(r, g, b))
        draw = ImageDraw.Draw(img)

        # Draw frame header and labels
        draw.rectangle([4, 4, width - 5, height - 5], outline=(255, 255, 255), width=2)
        draw.text((15, 15), "KEYFRAME PREVIEW", fill=(255, 255, 255))
        draw.text((15, height // 2 - 10), f"ID: {keyframe_id}", fill=(255, 255, 0))
        draw.text((15, height - 30), f"Video: {video_id}", fill=(220, 220, 220))
        return img
    except Exception:
        # Fallback if PIL is unavailable
        return None


def render_grid(candidates: list[Candidate]) -> None:
    """Render top-k candidate results in a responsive grid layout."""
    if not HAS_STREAMLIT:
        return

    st.subheader(f"🔍 Keyframe Candidates ({len(candidates)} hits)")

    if not candidates:
        st.info("No candidates found. Try refining your query.")
        return

    cols_per_row = 3
    rows = [candidates[i : i + cols_per_row] for i in range(0, len(candidates), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, cand in zip(cols, row, strict=False):
            with col:
                with st.container(border=True):
                    # Keyframe visual card header
                    st.markdown(f"**`{cand.keyframe_id}`**")
                    st.caption(f"Video: `{cand.video_id}` | Score: `{cand.score:.4f}`")

                    # Image rendering (Disk image or PIL Preview)
                    img_data = get_keyframe_image(cand.keyframe_id, width=320, height=180)
                    if img_data is not None:
                        st.image(img_data, use_container_width=True)
                    else:
                        st.markdown(
                            f"""
                            <div style="background-color:#1e1e2e; color:#cdd6f4; height:120px; 
                                        display:flex; align-items:center; justify-content:center; 
                                        border-radius:6px; font-family:monospace; font-size:12px;">
                                🖼️ Keyframe: {cand.keyframe_id}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if cand.matched_terms:
                        st.caption(f"Matched: {', '.join(cand.matched_terms)}")

                    st.caption(f"Evidence: `{cand.evidence_id}`")

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("⏪ Timeline", key=f"tbl_{cand.keyframe_id}"):
                            st.session_state["active_timeline"] = cand.keyframe_id
                            logger.info("Operator clicked timeline for %s", cand.keyframe_id)

                    with btn_col2:
                        if st.button("✅ Select", key=f"sel_{cand.keyframe_id}"):
                            st.session_state["selected_candidate"] = cand
                            logger.info("Operator selected candidate %s", cand.keyframe_id)


def render_timeline(keyframe_id: str) -> None:
    """Render temporal neighbors around keyframe_id in ONE action."""
    if not HAS_STREAMLIT:
        return

    st.subheader(f"🎞️ Temporal Context Timeline around `{keyframe_id}`")

    # The tool cannot invent which frames are keyframes, so hand it the ones this
    # search already surfaced. A timeline built only from the result set is thin,
    # but it is real — see get_neighbors' docstring.
    known = [c.keyframe_id for c in st.session_state.get("search_results", [])]
    res = get_neighbors(keyframe_id=keyframe_id, window=5, known_keyframe_ids=known)
    if not res.ok or not res.items:
        st.error(f"Failed to load neighbors: {res.error}")
        return

    neighbors = res.items
    st.write(f"Showing {len(neighbors)} adjacent keyframes (window=5):")

    cols = st.columns(min(len(neighbors), 5))
    for idx, n in enumerate(neighbors):
        col = cols[idx % 5]
        with col:
            with st.container(border=True):
                st.markdown(f"**`{n.keyframe_id}`**")
                st.caption(f"Time: `{n.timestamp_ms} ms`")
                img_data = get_keyframe_image(n.keyframe_id, width=160, height=90)
                if img_data is not None:
                    st.image(img_data, use_container_width=True)
                else:
                    st.markdown(
                        f"""
                        <div style="background-color:#313244; color:#cdd6f4; height:80px; 
                                    display:flex; align-items:center; justify-content:center; 
                                    border-radius:4px; font-size:11px;">
                            {n.keyframe_id}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    seg_res = get_video_segment(keyframe_ids=[keyframe_id])
    if seg_res.ok and seg_res.items:
        seg_info = seg_res.items[0]
        st.caption(
            f"Segment Coordinates: Video `{seg_info['video_path']}` | "
            f"Range: `[{seg_info['start_ms']} ms - {seg_info['end_ms']} ms]`"
        )


def render_submit_panel(candidate: Candidate) -> None:
    """Render gated submission confirmation panel."""
    if not HAS_STREAMLIT:
        return

    st.subheader("🚀 Submission Confirmation Panel")
    st.warning("⚠️ Submission is gated. Confirming will validate through TV4 Validator.")

    st.write(f"**Target Keyframe:** `{candidate.keyframe_id}`")
    st.write(f"**Video ID:** `{candidate.video_id}`")
    st.write(f"**Evidence Provenance:** `{candidate.evidence_id}`")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("CONFIRM SUBMIT", type="primary", key="btn_confirm_submit"):
            logger.info("Operator clicked confirm submit for %s", candidate.keyframe_id)
            val_res = validate_submission_stub(candidate)
            if val_res.ok:
                st.success(f"✅ Submission VALIDATED & SUBMITTED: {val_res.items[0]}")
                st.balloons()
            else:
                st.error(f"❌ Submission REJECTED by Validator: {val_res.error.message}")

    with c2:
        if st.button("Cancel", key="btn_cancel_submit"):
            st.session_state["selected_candidate"] = None
            st.rerun()


def main() -> None:
    """Streamlit Application Entrypoint."""
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run: pip install -e '.[browser]'")
        return

    st.set_page_config(
        page_title="MFE Video Browser",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 Multimodal Fusion Engine (MFE) — Video Browser MVP")

    # Initialize Session State
    if "cascade" not in st.session_state:
        st.session_state["cascade"] = Cascade()
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []
    if "active_timeline" not in st.session_state:
        st.session_state["active_timeline"] = None
    if "selected_candidate" not in st.session_state:
        st.session_state["selected_candidate"] = None

    # Sidebar Query Refinement Controls
    with st.sidebar:
        st.header("⚙️ Query Refinement & Options")
        top_k = st.slider("Top-K Candidates", min_value=5, max_value=200, value=20, step=5)
        weight_text = st.slider(
            "Text Weight (BM25)", min_value=0.0, max_value=2.0, value=1.0, step=0.1
        )
        weight_vec = st.slider(
            "Vector Weight (CLIP)", min_value=0.0, max_value=2.0, value=1.0, step=0.1
        )

        st.session_state["cascade"].fusion_weights = {
            "search_text": weight_text,
            "vector_search": weight_vec,
        }

        st.markdown("---")
        st.caption("AIC 2026 — Multimodal Fusion Engine")

    # Search Bar Section
    query_text = st.text_input(
        "Enter Video Search Query:", value="a person riding a bicycle near the beach"
    )

    if st.button("🔍 Search Video Store", type="primary", use_container_width=True):
        if query_text.strip():
            start_t = time.perf_counter()
            results = st.session_state["cascade"].search(
                query_text=query_text, top_k=top_k, run_id="browser_query"
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            st.session_state["search_results"] = results
            st.session_state["active_timeline"] = None

            logger.info(
                "Query '%s' returned %d hits in %.2fms", query_text, len(results), elapsed_ms
            )
            st.toast(f"Retrieved {len(results)} hits in {elapsed_ms:.1f}ms", icon="⚡")

    # Display Selected Timeline panel if active
    if st.session_state.get("active_timeline"):
        render_timeline(st.session_state["active_timeline"])
        st.markdown("---")

    # Display Gated Submit Panel if candidate selected
    if st.session_state.get("selected_candidate"):
        render_submit_panel(st.session_state["selected_candidate"])
        st.markdown("---")

    # Render Main Candidate Grid
    render_grid(st.session_state.get("search_results", []))


if __name__ == "__main__":
    main()
