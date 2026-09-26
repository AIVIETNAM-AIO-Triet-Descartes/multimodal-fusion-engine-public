from mfe.eval.dres_client import build_selected_payload, dres_credentials


def test_dres_payload_matches_final_format_for_each_query_type():
    item = {"video_id": "L21_V001", "timestamp_ms": 1234, "frame_idx": 31}
    assert build_selected_payload("KIS", item) == {
        "answerSets": [{"answers": [{"mediaItemName": "L21_V001", "start": 1234, "end": 1234}]}]
    }
    assert build_selected_payload("QA", item, answer="áo xanh") == {
        "answerSets": [{"answers": [{"text": "QA-áo xanh-L21_V001-1234"}]}]
    }
    trake = {
        "video_id": "L21_V001",
        "events": [{"frame_idx": 31}, {"frame_idx": 80}],
    }
    assert build_selected_payload("TRAKE", trake) == {
        "answerSets": [{"answers": [{"text": "TR-L21_V001-31,80"}]}]
    }


def test_dres_credentials_accept_the_existing_aliases(monkeypatch):
    monkeypatch.delenv("user_dres", raising=False)
    monkeypatch.delenv("pass_dres", raising=False)
    monkeypatch.setenv("user_dress", "operator")
    monkeypatch.setenv("pass_dress", "secret")
    assert dres_credentials() == ("operator", "secret")
