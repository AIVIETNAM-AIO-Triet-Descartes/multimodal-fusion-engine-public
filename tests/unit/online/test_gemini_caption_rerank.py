from types import SimpleNamespace

from mfe.contracts.tools import Candidate
from mfe.online.gemini_caption_rerank import rerank_candidates


def _candidate(number: int) -> Candidate:
    return Candidate(
        keyframe_id=f"V#{number:06d}",
        video_id="V",
        score=float(number),
        evidence_id=f"caption_qwen:test:{number}",
    )


class FakeGateway:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def chat(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return self.reply


def test_rerank_preserves_candidate_objects_and_ignores_tail():
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    gateway = FakeGateway('["V#000003", "V#000001", "V#000002"]')
    bundle = SimpleNamespace(texts={"caption_qwen": {"V#000003": "sư tử"}})
    output, result = rerank_candidates(candidates, "sư tử", bundle, client=gateway, depth=3)
    assert [item.keyframe_id for item in output] == ["V#000003", "V#000001", "V#000002"]
    assert result.ran is True
    assert result.moved == 3
    assert "sư tử" in gateway.prompts[0]


def test_invalid_remote_ranking_falls_back_to_original_order():
    candidates = [_candidate(1), _candidate(2)]
    gateway = FakeGateway('["V#000001"]')
    output, result = rerank_candidates(
        candidates, "query", SimpleNamespace(texts={}), client=gateway, depth=2
    )
    assert output == candidates
    assert result.ran is False
    assert result.reason
