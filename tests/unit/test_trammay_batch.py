from mfe.online.trammay import TrammayClient


def test_chat_batch_encodes_images_and_preserves_ids(tmp_path, monkeypatch):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    captured = {}

    def fake_post(self, payload, *, endpoint="/chat/completions"):
        captured["payload"] = payload
        captured["endpoint"] = endpoint
        return {
            "data": [
                {"id": "f1", "content": "caption one"},
                {"id": "f2", "content": "caption two"},
            ]
        }

    monkeypatch.setattr(TrammayClient, "_post", fake_post)
    client = TrammayClient(api_key="test", max_retries=1)
    result = client.chat_batch(
        [
            {"id": "f1", "prompt": "caption", "images": [image]},
            {"id": "f2", "prompt": "caption", "images": [image]},
        ],
        max_tokens=1024,
    )

    assert result == {"f1": "caption one", "f2": "caption two"}
    assert captured["endpoint"] == "/chat/completions/batch"
    assert len(captured["payload"]["requests"]) == 2
    assert captured["payload"]["requests"][0]["images"][0].startswith(
        "data:image/jpeg;base64,"
    )
