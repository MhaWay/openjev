import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openjev.api import create_app
from openjev.config import Settings
from openjev.forjev import ForJevEngine, LABELS
from openjev.engine import Upstream


@pytest.fixture
def upstream(monkeypatch):
    calls = []
    ids = {label: i + 10 for i, label in enumerate(LABELS)}
    def respond(request):
        body = json.loads(request.content) if request.content else {}
        calls.append((request, body))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ids[body["prompt"]]]})
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 stream=httpx.ByteStream(b'data: {"delta":"ok"}\n\ndata: [DONE]\n\n'))
        candidates = body["logprob_token_ids"]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": f"token_id:{i}", "logprob": -0.1 if i == candidates[-1] else -3.0}
            for i in candidates]}]}}], "usage": {"prompt_tokens": 42, "completion_tokens": 1}})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        **{**kw, "transport": httpx.MockTransport(respond)}))
    return calls


def settings(**kw):
    return Settings(backend="forjev", upstream="http://qwen", upstream_model="qwen", **kw)


def body():
    return {"model": "forjev-qwen-next", "state": {"health": 4}, "questions": {
        "move": {"type": "choice", "instructions": {"goal": "survive"},
                 "criteria": {"stay": "Stay here", "flee": "Run away"}},
        "risk": {"type": "score", "criteria": ["low", "high"]},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"}}}


@pytest.mark.parametrize("images", [None, [], ["data:image/jpeg;base64,/9j/2Q=="]])
def test_native_api_typed_answers_and_optional_images(upstream, images):
    with TestClient(create_app(settings())) as client:
        response = client.post("/v1/systemone", json={**body(), "images": images})
        assert response.status_code == 200, response.text
        answers = response.json()["answers"]
        assert answers["move"]["choice"] == "flee"
        assert sum(answers["move"]["probabilities"].values()) == pytest.approx(1)
        assert 0 <= answers["risk"]["score"] <= 1
        assert 0 <= answers["urgent"]["noul"] <= 1
        assert response.json()["usage"] == {"input_tokens": 126, "output_tokens": 0}
    chats = [b for r, b in upstream if r.url.path == "/v1/chat/completions"]
    assert len(chats) == 3
    for b in chats:
        assert b["max_tokens"] == 1 and b["logprobs"] is True
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        assert b["messages"][1]["content"][0]["type"] == ("image_url" if images else "text")


@pytest.mark.parametrize("field,value", [("think", 1), ("sequential", True), ("steps", 2), ("samples", 2)])
def test_unsupported_options_rejected_before_inference(upstream, field, value):
    with TestClient(create_app(settings())) as client:
        assert client.post("/v1/systemone", json={**body(), field: value}).status_code == 400
    assert not upstream


def test_labels_beyond_single_characters_are_unique(upstream):
    async def run():
        engine = ForJevEngine(settings(forjev_max_choices=255))
        try:
            pairs = await engine._ids(255)
            assert len(set(t for _, t in pairs)) == 255
            assert pairs[62][0] == "AA"
            assert len({label for label, _ in pairs}) == 255
        finally:
            await engine.close()
    asyncio.run(run())


def test_missing_candidates_fail_closed():
    async def run():
        engine = ForJevEngine(settings())
        async def post(path, data):
            if path == "/tokenize":
                return {"tokens": [ord(data["prompt"])]}
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": "token_id:65", "logprob": -0.1}]}]}}], "usage": {"prompt_tokens": 1}}
        engine._post = post
        try:
            with pytest.raises(Upstream, match="incomplete"):
                await engine.decide(body()["questions"], "state", 0)
        finally:
            await engine.close()
    asyncio.run(run())


def test_chat_proxy_stream_tools_and_separate_credentials(upstream):
    with TestClient(create_app(settings(api_key="frontend", forjev_upstream_api_key="upstream"))) as client:
        assert client.post("/v1/chat/completions", json={}).status_code == 403
        headers = {"Authorization": "Bearer frontend"}
        models = client.get("/v1/models", headers=headers).json()
        assert {m["id"] for m in models["data"]} == {"qwen", "forjev-qwen-next"}
        payload = {"model": "qwen", "stream": True, "temperature": 0.8,
                   "messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "function", "function": {"name": "look"}}]}
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.content.endswith(b'data: [DONE]\n\n')
        assert upstream[-1][1] == payload
        assert upstream[-1][0].headers["authorization"] == "Bearer upstream"
        assert client.get("/ready").status_code == 200


def test_choice_cap_and_forced_answer(upstream):
    with TestClient(create_app(settings())) as client:
        data = body()
        data["questions"] = {"one": {"type": "choice", "criteria": {"only": "only"}}}
        assert client.post("/v1/systemone", json=data).json()["answers"]["one"]["choice"] == "only"
        assert not upstream
        data["questions"]["one"]["criteria"] = {str(i): str(i) for i in range(21)}
        assert client.post("/v1/systemone", json=data).status_code == 400
        assert not upstream
