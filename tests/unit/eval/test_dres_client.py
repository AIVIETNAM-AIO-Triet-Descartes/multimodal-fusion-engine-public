from mfe.eval.dres_client import build_selected_payload, dres_credentials, submission_key


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


def test_dres_submission_key_ignores_lane_and_result_rank():
    state = {"query_type": "KIS", "query": "một cảnh", "events": []}
    same_query = {**state, "scope": "private", "rank": 9}
    other_query = {**state, "query": "cảnh khác"}
    assert submission_key("eval-1", state) == submission_key("eval-1", same_query)
    assert submission_key("eval-1", state) != submission_key("eval-1", other_query)
    assert submission_key("eval-1", state) != submission_key("eval-2", state)


def test_dres_submission_key_allows_a_different_answer_for_same_query():
    state = {"query_type": "KIS", "query": "một cảnh", "events": []}
    first = {
        "answerSets": [
            {"answers": [{"mediaItemName": "L21_V001", "start": 100, "end": 100}]}
        ]
    }
    second = {
        "answerSets": [
            {"answers": [{"mediaItemName": "L21_V002", "start": 200, "end": 200}]}
        ]
    }
    assert submission_key("eval-1", state, first) != submission_key("eval-1", state, second)
    assert submission_key("eval-1", state, first) == submission_key("eval-1", state, first)
