"""Merging the five members' query sets — mfe.eval.merge_queryset.

The checks here matter more than they look. Five people write ground truth on
five disjoint samples; every mistake in that arrangement fails *silently* at
score time — a query on somebody else's video simply scores 0.0 forever — so
the merge is the only place it can be caught.
"""

from __future__ import annotations

import json

import pytest

from mfe.eval import merge_queryset as mq
from mfe.eval.official import KISQuery


def _kis(query_id: str, author: str, split: str, video: str = "L21_V001") -> str:
    return json.dumps(
        {
            "query_id": query_id,
            "author": author,
            "split": split,
            "type": "KIS",
            "text": "người phụ nữ mặc áo dài vàng",
            "gt_video_id": video,
            "gt_frame_range": [100, 110],
        },
        ensure_ascii=False,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A whole eval/ tree with two members, wired into the module's paths."""
    dev = tmp_path / "dev"
    heldout = tmp_path / "heldout"
    manifests = tmp_path / "manifests"
    for directory in (dev, heldout, manifests):
        directory.mkdir()

    (manifests / "dat.json").write_text(
        json.dumps({"member": "dat", "videos": ["L21_V001", "L21_V002"]}), encoding="utf-8"
    )
    (manifests / "quang.json").write_text(
        json.dumps({"member": "quang", "videos": ["L21_V003", "L21_V004"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mq, "DEV_DIR", dev)
    monkeypatch.setattr(mq, "HELDOUT_DIR", heldout)
    monkeypatch.setattr(mq, "MANIFEST_DIR", manifests)
    monkeypatch.setattr(mq, "DEV_OUT", tmp_path / "queryset_dev.jsonl")
    monkeypatch.setattr(mq, "HELDOUT_OUT", tmp_path / "queryset_heldout.jsonl")
    return tmp_path, dev, heldout, manifests


def test_a_clean_set_merges(workspace):
    tmp_path, dev, heldout, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")
    (heldout / "quang.jsonl").write_text(
        _kis("quang_kis_01", "quang", "heldout", "L21_V003"), encoding="utf-8"
    )

    assert mq.run() == 0
    assert (tmp_path / "queryset_dev.jsonl").exists()
    assert (tmp_path / "queryset_heldout.jsonl").exists()


def test_query_on_another_members_video_is_refused(workspace, capsys):
    """The failure this whole script exists for.

    `L21_V003` belongs to quang. In dat's index it does not exist, so the query
    scores 0.0 every run and nothing ever says why.
    """
    _, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(
        _kis("dat_kis_01", "dat", "dev", "L21_V003"), encoding="utf-8"
    )

    assert mq.run() == 1
    assert "not in dat's sample" in capsys.readouterr().out


def test_two_members_claiming_the_same_video_is_refused(workspace, capsys):
    _, dev, _, manifests = workspace
    (manifests / "quang.json").write_text(
        json.dumps({"member": "quang", "videos": ["L21_V001"]}), encoding="utf-8"
    )
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")

    assert mq.run() == 1
    out = capsys.readouterr().out
    assert "both claim" in out
    assert "L21_V001" in out


def test_split_must_match_the_directory(workspace, capsys):
    """A held-out query labelled "dev" is a held-out query that gets debugged."""
    _, _, heldout, _ = workspace
    (heldout / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")

    assert mq.run() == 1
    assert "split says 'dev'" in capsys.readouterr().out


def test_author_must_match_the_file(workspace, capsys):
    _, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "quang", "dev"), encoding="utf-8")

    assert mq.run() == 1
    assert "author is 'quang'" in capsys.readouterr().out


def test_query_id_must_carry_the_author_prefix(workspace, capsys):
    """Prefixes are what keep ids unique once five files become one."""
    _, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("kis_01", "dat", "dev"), encoding="utf-8")

    assert mq.run() == 1
    assert "should start with 'dat_'" in capsys.readouterr().out


def test_duplicate_ids_across_splits_are_caught(workspace, capsys):
    _, dev, heldout, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")
    (heldout / "dat.jsonl").write_text(
        _kis("dat_kis_01", "dat", "heldout"), encoding="utf-8"
    )

    assert mq.run() == 1
    assert "appears 2 times" in capsys.readouterr().out


def test_an_inverted_frame_range_is_caught(workspace, capsys):
    _, dev, _, _ = workspace
    line = json.loads(_kis("dat_kis_01", "dat", "dev"))
    line["gt_frame_range"] = [110, 100]
    (dev / "dat.jsonl").write_text(json.dumps(line, ensure_ascii=False), encoding="utf-8")

    assert mq.run() == 1
    assert "ground truth is invalid" in capsys.readouterr().out


def test_nothing_is_written_when_anything_fails(workspace):
    """A partial merge is worse than none: the count is what docs/00 §1 gates on."""
    tmp_path, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(
        _kis("dat_kis_01", "dat", "dev") + "\n" + _kis("dat_kis_02", "dat", "dev", "L21_V003"),
        encoding="utf-8",
    )

    assert mq.run() == 1
    assert not (tmp_path / "queryset_dev.jsonl").exists()


def test_check_only_writes_nothing(workspace):
    tmp_path, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")

    assert mq.run(check_only=True) == 0
    assert not (tmp_path / "queryset_dev.jsonl").exists()


def test_missing_manifests_directory_is_reported(workspace):
    _, _, _, manifests = workspace
    for path in manifests.glob("*.json"):
        path.unlink()

    assert mq.run() == 2


def test_a_missing_anchor_warns_but_still_merges(workspace, capsys):
    """Wide-only is degraded, not broken — blocking would stop the other 114."""
    tmp_path, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")

    assert mq.run() == 0
    assert "no anchor frame" in capsys.readouterr().out
    assert (tmp_path / "queryset_dev.jsonl").exists()


def test_an_anchor_outside_its_scene_is_an_error_not_a_warning(workspace, capsys):
    """A contradictory annotation cannot be scored either way — that blocks."""
    _, dev, _, _ = workspace
    line = json.loads(_kis("dat_kis_01", "dat", "dev"))
    line["gt_anchor_frame"] = 900  # outside [100, 110]
    (dev / "dat.jsonl").write_text(json.dumps(line, ensure_ascii=False), encoding="utf-8")

    assert mq.run() == 1
    assert "strict window is invalid" in capsys.readouterr().out


def test_a_valid_anchor_produces_no_warning(workspace, capsys):
    _, dev, _, _ = workspace
    line = json.loads(_kis("dat_kis_01", "dat", "dev"))
    line["gt_anchor_frame"] = 105
    (dev / "dat.jsonl").write_text(json.dumps(line, ensure_ascii=False), encoding="utf-8")

    assert mq.run() == 0
    assert "CẢNH BÁO" not in capsys.readouterr().out


def test_merged_output_reloads(workspace):
    """The merged file must be readable by the same loader the harness uses."""
    tmp_path, dev, _, _ = workspace
    (dev / "dat.jsonl").write_text(_kis("dat_kis_01", "dat", "dev"), encoding="utf-8")

    mq.run()

    from mfe.eval.official import load_official_queryset

    reloaded = load_official_queryset(tmp_path / "queryset_dev.jsonl")
    assert len(reloaded) == 1
    assert isinstance(reloaded[0], KISQuery)
    assert reloaded[0].gt_frame_range == (100, 110)
