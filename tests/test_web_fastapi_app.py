import json

import pytest
from fastapi.testclient import TestClient

from app import web_fastapi_app
from service.chat_service import ChatResponse
from utils.config_loader import ProfileRegistry, ProviderProfile


def build_registry() -> tuple[ProfileRegistry, ProviderProfile]:
    """Create a minimal in-memory registry for FastAPI tests.

    Args:
        None: This function does not accept parameters.
    """
    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "model-a",
        models = ["model-a", "model-b"],
    )
    registry = ProfileRegistry(
        profiles = {"p1": profile},
        default_profile_id = "p1",
    )
    return registry, profile


def test_fastapi_bootstrap_endpoint_returns_profile_catalog() -> None:
    """Verify bootstrap endpoint includes profile and model metadata.

    Args:
        None: This function does not accept parameters.
    """
    registry, profile = build_registry()
    app = web_fastapi_app.create_fastapi_app(
        registry = registry,
        profile = profile,
        model = "model-a",
    )
    client = TestClient(app)

    response = client.get("/api/bootstrap")
    payload = response.json()

    assert response.status_code == 200
    assert payload["default_profile"] == "p1"
    assert payload["default_model"] == "model-a"
    assert payload["profiles"][0]["profile_id"] == "p1"
    assert payload["profiles"][0]["models"] == ["model-a", "model-b"]


def test_fastapi_chat_endpoint_returns_stubbed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify chat endpoint returns normalized response payload.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    registry, profile = build_registry()

    def fake_send_chat(request, profile, model, enable_deep_thinking = None):
        """Return deterministic response for endpoint testing.

        Args:
            request: Normalized chat request payload.
            profile: Active provider profile.
            model: Active model name.
            enable_deep_thinking: Optional deep-thinking switch.
        """
        _ = request, profile, model, enable_deep_thinking
        return ChatResponse(
            assistant_text = "fake-assistant",
            reasoning_text = "fake-reasoning",
            usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            warning_messages = ["w1"],
            error_message = None,
        )

    monkeypatch.setattr(web_fastapi_app, "send_chat", fake_send_chat)

    app = web_fastapi_app.create_fastapi_app(
        registry = registry,
        profile = profile,
        model = "model-a",
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json = {
            "message": "hello",
            "profile": "p1",
            "model": "model-b",
            "history": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "enable_thinking": True,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["assistant_text"] == "fake-assistant"
    assert payload["reasoning_text"] == "fake-reasoning"
    assert payload["usage"]["total_tokens"] == 5
    assert payload["warning_messages"] == ["w1"]
    assert payload["error_message"] is None
    assert payload["profile"] == "p1"
    assert payload["model"] == "model-b"


def test_fastapi_stream_endpoint_emits_token_and_done_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify stream endpoint emits SSE token and done payload events.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    registry, profile = build_registry()

    def fake_stream_chat(
        request,
        profile,
        model,
        warning_messages = None,
        enable_deep_thinking = None,
    ):
        """Yield deterministic stream chunks for endpoint testing.

        Args:
            request: Normalized chat request payload.
            profile: Active provider profile.
            model: Active model name.
            warning_messages: Optional warning output list.
            enable_deep_thinking: Optional deep-thinking switch.
        """
        _ = request, profile, model, enable_deep_thinking
        if warning_messages is not None:
            warning_messages.append("stream-warning")
        yield "hello "
        yield "world"

    monkeypatch.setattr(web_fastapi_app, "stream_chat", fake_stream_chat)

    app = web_fastapi_app.create_fastapi_app(
        registry = registry,
        profile = profile,
        model = "model-a",
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat/stream",
        json = {
            "message": "hello",
            "profile": "p1",
            "model": "model-b",
            "temperature": 0.7,
            "top_p": 0.9,
            "enable_thinking": False,
        },
    )
    body = response.text

    assert response.status_code == 200
    assert "event: token" in body
    assert "event: done" in body
    assert "hello " in body

    done_json = ""
    chunks = body.split("\n\n")
    for chunk in chunks:
        if "event: done" not in chunk:
            continue
        for line in chunk.split("\n"):
            if line.startswith("data:"):
                done_json = line[len("data:"):].strip()
                break
        if done_json:
            break

    assert done_json
    done_payload = json.loads(done_json)
    assert done_payload["assistant_text"] == "hello world"
    assert done_payload["profile"] == "p1"
    assert done_payload["model"] == "model-b"
    assert done_payload["warning_messages"] == ["stream-warning"]
