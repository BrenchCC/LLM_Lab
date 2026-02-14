from types import SimpleNamespace

from service import chat_service
from service.chat_service import ChatRequest, send_chat
from utils.config_loader import ModelCapabilities, ProviderProfile


class FakeCompletions:
    """Fake completions endpoint for chat service tests."""

    def create(self, **kwargs):
        """Return fake completion object.

        Args:
            kwargs: Completion request keyword arguments.
        """
        _ = kwargs
        usage = SimpleNamespace(prompt_tokens = 3, completion_tokens = 5, total_tokens = 8)
        message = SimpleNamespace(content = "hello from fake model")
        choice = SimpleNamespace(message = message)
        completion = SimpleNamespace(
            choices = [choice],
            usage = usage,
            model_dump = lambda: {"ok": True},
        )
        return completion


class FakeClient:
    """Fake OpenAI-compatible client for chat service tests."""

    def __init__(self):
        """Initialize fake nested API attributes.

        Args:
            self: Fake client instance.
        """
        self.chat = SimpleNamespace(completions = FakeCompletions())


def build_profile() -> ProviderProfile:
    """Create a provider profile fixture object.

    Args:
        None: This function does not accept parameters.
    """
    return ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        capabilities = ModelCapabilities(),
    )


def test_send_chat_success(monkeypatch) -> None:
    """Verify text chat success path returns assistant content and usage.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_service, "build_client", lambda profile: FakeClient())
    monkeypatch.setattr(
        chat_service,
        "resolve_capabilities",
        lambda profile, model, client: ModelCapabilities(
            supports_text = True,
            supports_image = True,
            supports_video = False,
            supports_audio = False,
        ),
    )

    request = ChatRequest(user_text = "hello")
    response = send_chat(
        request = request,
        profile = build_profile(),
        model = "test-model",
    )

    assert response.error_message is None
    assert response.assistant_text == "hello from fake model"
    assert response.usage["total_tokens"] == 8


def test_send_chat_image_not_supported(monkeypatch) -> None:
    """Verify unsupported image input returns readable error.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_service, "build_client", lambda profile: FakeClient())
    monkeypatch.setattr(
        chat_service,
        "resolve_capabilities",
        lambda profile, model, client: ModelCapabilities(
            supports_text = True,
            supports_image = False,
            supports_video = False,
            supports_audio = False,
        ),
    )

    request = ChatRequest(
        user_text = "hello",
        image_paths = ["example.jpg"],
    )
    response = send_chat(
        request = request,
        profile = build_profile(),
        model = "test-model",
    )

    assert response.assistant_text == ""
    assert "does not support image" in str(response.error_message)

