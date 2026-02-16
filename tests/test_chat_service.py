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


class CaptureCompletions:
    """Capture completion call arguments for request-model mapping tests."""

    def __init__(self, tracker: dict[str, str]):
        """Store shared tracker reference.

        Args:
            tracker: Mutable tracker dictionary.
        """
        self.tracker = tracker

    def create(self, **kwargs):
        """Capture model argument and return normal fake completion.

        Args:
            kwargs: Completion request keyword arguments.
        """
        self.tracker["model"] = str(kwargs.get("model", ""))
        usage = SimpleNamespace(prompt_tokens = 3, completion_tokens = 5, total_tokens = 8)
        message = SimpleNamespace(content = "hello from fake model")
        choice = SimpleNamespace(message = message)
        completion = SimpleNamespace(
            choices = [choice],
            usage = usage,
            model_dump = lambda: {"ok": True},
        )
        return completion


class FakeReasoningCompletions:
    """Fake completions endpoint returning structured reasoning payload."""

    def create(self, **kwargs):
        """Return fake completion object with reasoning fields.

        Args:
            kwargs: Completion request keyword arguments.
        """
        _ = kwargs
        usage = SimpleNamespace(prompt_tokens = 4, completion_tokens = 7, total_tokens = 11)
        message = SimpleNamespace(
            content = [
                {"type": "reasoning", "text": "step one"},
                {"type": "text", "text": "final answer"},
            ],
            reasoning_content = "step one",
        )
        choice = SimpleNamespace(message = message)
        completion = SimpleNamespace(
            choices = [choice],
            usage = usage,
            model_dump = lambda: {"ok": True},
        )
        return completion


class FakeThinkTagCompletions:
    """Fake completions endpoint returning `<think>` wrapped content."""

    def create(self, **kwargs):
        """Return fake completion object with `<think>` tags.

        Args:
            kwargs: Completion request keyword arguments.
        """
        _ = kwargs
        usage = SimpleNamespace(prompt_tokens = 5, completion_tokens = 8, total_tokens = 13)
        message = SimpleNamespace(content = "<think>draft plan</think>\nvisible answer")
        choice = SimpleNamespace(message = message)
        completion = SimpleNamespace(
            choices = [choice],
            usage = usage,
            model_dump = lambda: {"ok": True},
        )
        return completion


class FallbackThinkingCompletions:
    """Fake completions endpoint that rejects thinking once then succeeds."""

    def __init__(self, tracker: dict[str, list[dict[str, object]]]):
        """Store request tracker for fallback verification.

        Args:
            tracker: Mutable request tracker dictionary.
        """
        self.tracker = tracker

    def create(self, **kwargs):
        """Fail on thinking payload then return normal completion.

        Args:
            kwargs: Completion request keyword arguments.
        """
        calls = self.tracker.setdefault("calls", [])
        calls.append(dict(kwargs))

        extra_body = kwargs.get("extra_body")
        has_enable_thinking = isinstance(extra_body, dict) and "enable_thinking" in extra_body
        if has_enable_thinking:
            raise RuntimeError("unsupported parameter: enable_thinking")

        usage = SimpleNamespace(prompt_tokens = 3, completion_tokens = 5, total_tokens = 8)
        message = SimpleNamespace(content = "fallback success")
        choice = SimpleNamespace(message = message)
        completion = SimpleNamespace(
            choices = [choice],
            usage = usage,
            model_dump = lambda: {"ok": True},
        )
        return completion


class StreamFallbackThinkingCompletions:
    """Fake stream completions that reject thinking once then stream normally."""

    def __init__(self, tracker: dict[str, list[dict[str, object]]]):
        """Store request tracker for stream fallback verification.

        Args:
            tracker: Mutable request tracker dictionary.
        """
        self.tracker = tracker

    def create(self, **kwargs):
        """Fail on thinking payload then return stream chunks.

        Args:
            kwargs: Completion request keyword arguments.
        """
        calls = self.tracker.setdefault("calls", [])
        calls.append(dict(kwargs))

        extra_body = kwargs.get("extra_body")
        has_enable_thinking = isinstance(extra_body, dict) and "enable_thinking" in extra_body
        if has_enable_thinking:
            raise RuntimeError("unsupported parameter: enable_thinking")

        chunk = SimpleNamespace(
            choices = [SimpleNamespace(delta = SimpleNamespace(content = "stream fallback success"))]
        )
        return [chunk]


class FakeClient:
    """Fake OpenAI-compatible client for chat service tests."""

    def __init__(self):
        """Initialize fake nested API attributes.

        Args:
            self: Fake client instance.
        """
        self.chat = SimpleNamespace(completions = FakeCompletions())


class CaptureClient:
    """Fake client that captures request model value."""

    def __init__(self, tracker: dict[str, str]):
        """Initialize fake nested API attributes.

        Args:
            tracker: Mutable tracker dictionary.
        """
        self.chat = SimpleNamespace(completions = CaptureCompletions(tracker = tracker))


class FakeReasoningClient:
    """Fake client for structured reasoning payload tests."""

    def __init__(self):
        """Initialize fake nested API attributes.

        Args:
            self: Fake client instance.
        """
        self.chat = SimpleNamespace(completions = FakeReasoningCompletions())


class FakeThinkTagClient:
    """Fake client for think-tag parsing tests."""

    def __init__(self):
        """Initialize fake nested API attributes.

        Args:
            self: Fake client instance.
        """
        self.chat = SimpleNamespace(completions = FakeThinkTagCompletions())


class FallbackThinkingClient:
    """Fake client for deep-thinking fallback tests."""

    def __init__(self, tracker: dict[str, list[dict[str, object]]]):
        """Initialize fake nested API attributes.

        Args:
            tracker: Mutable request tracker dictionary.
        """
        self.chat = SimpleNamespace(
            completions = FallbackThinkingCompletions(tracker = tracker)
        )


class StreamFallbackThinkingClient:
    """Fake client for stream deep-thinking fallback tests."""

    def __init__(self, tracker: dict[str, list[dict[str, object]]]):
        """Initialize fake nested API attributes.

        Args:
            tracker: Mutable request tracker dictionary.
        """
        self.chat = SimpleNamespace(
            completions = StreamFallbackThinkingCompletions(tracker = tracker)
        )


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
    assert response.reasoning_text == ""
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


def test_send_chat_extracts_structured_reasoning(monkeypatch) -> None:
    """Verify structured reasoning payload is extracted separately.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_service, "build_client", lambda profile: FakeReasoningClient())
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

    response = send_chat(
        request = ChatRequest(user_text = "hello"),
        profile = build_profile(),
        model = "test-model",
    )

    assert response.error_message is None
    assert response.assistant_text == "final answer"
    assert response.reasoning_text == "step one"


def test_send_chat_extracts_think_tag_reasoning(monkeypatch) -> None:
    """Verify `<think>` content is split into reasoning and final answer.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(chat_service, "build_client", lambda profile: FakeThinkTagClient())
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

    response = send_chat(
        request = ChatRequest(user_text = "hello"),
        profile = build_profile(),
        model = "test-model",
    )

    assert response.error_message is None
    assert response.assistant_text == "visible answer"
    assert response.reasoning_text == "draft plan"


def test_send_chat_uses_mapped_request_model(monkeypatch) -> None:
    """Verify send_chat uses mapped endpoint id for provider requests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tracker: dict[str, str] = {}

    def fake_resolve_capabilities(profile, model, client):
        _ = profile
        _ = client
        assert model == "endpoint-001"
        return ModelCapabilities(
            supports_text = True,
            supports_image = True,
            supports_video = False,
            supports_audio = False,
        )

    monkeypatch.setattr(chat_service, "build_client", lambda profile: CaptureClient(tracker = tracker))
    monkeypatch.setattr(chat_service, "resolve_capabilities", fake_resolve_capabilities)

    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        model_aliases = {"alias-model": "endpoint-001"},
        capabilities = ModelCapabilities(),
    )
    response = send_chat(
        request = ChatRequest(user_text = "hello"),
        profile = profile,
        model = "alias-model",
    )

    assert response.error_message is None
    assert tracker["model"] == "endpoint-001"


def test_send_chat_falls_back_when_deep_thinking_unsupported(monkeypatch) -> None:
    """Verify thinking mode falls back to normal mode with warning message.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tracker: dict[str, list[dict[str, object]]] = {}
    monkeypatch.setattr(
        chat_service,
        "build_client",
        lambda profile: FallbackThinkingClient(tracker = tracker),
    )
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

    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        enable_deep_thinking = True,
        capabilities = ModelCapabilities(),
    )
    response = send_chat(
        request = ChatRequest(user_text = "hello"),
        profile = profile,
        model = "test-model",
    )

    calls = tracker.get("calls", [])
    assert response.error_message is None
    assert response.assistant_text == "fallback success"
    assert len(calls) == 2
    assert isinstance(calls[0].get("extra_body"), dict)
    assert calls[0]["extra_body"]["enable_thinking"] is True
    assert "extra_body" not in calls[1]
    assert response.warning_messages


def test_send_chat_can_disable_deep_thinking_by_runtime_override(monkeypatch) -> None:
    """Verify runtime override can disable deep thinking for one request.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tracker: dict[str, list[dict[str, object]]] = {}
    monkeypatch.setattr(
        chat_service,
        "build_client",
        lambda profile: FallbackThinkingClient(tracker = tracker),
    )
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

    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        enable_deep_thinking = True,
        capabilities = ModelCapabilities(),
    )
    response = send_chat(
        request = ChatRequest(user_text = "hello"),
        profile = profile,
        model = "test-model",
        enable_deep_thinking = False,
    )

    calls = tracker.get("calls", [])
    assert response.error_message is None
    assert response.assistant_text == "fallback success"
    assert len(calls) == 1
    assert "extra_body" not in calls[0]
    assert not response.warning_messages


def test_stream_chat_falls_back_when_deep_thinking_unsupported(monkeypatch) -> None:
    """Verify stream mode fallback keeps working and exposes warning message.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tracker: dict[str, list[dict[str, object]]] = {}
    monkeypatch.setattr(
        chat_service,
        "build_client",
        lambda profile: StreamFallbackThinkingClient(tracker = tracker),
    )
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

    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        enable_deep_thinking = True,
        capabilities = ModelCapabilities(),
    )
    warning_messages: list[str] = []
    chunks = list(
        chat_service.stream_chat(
            request = ChatRequest(user_text = "hello", stream = True),
            profile = profile,
            model = "test-model",
            warning_messages = warning_messages,
        )
    )

    calls = tracker.get("calls", [])
    assert "".join(chunks) == "stream fallback success"
    assert len(calls) == 2
    assert isinstance(calls[0].get("extra_body"), dict)
    assert calls[0]["extra_body"]["enable_thinking"] is True
    assert "extra_body" not in calls[1]
    assert warning_messages


def test_stream_chat_can_disable_deep_thinking_by_runtime_override(monkeypatch) -> None:
    """Verify stream runtime override can disable deep thinking for one request.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tracker: dict[str, list[dict[str, object]]] = {}
    monkeypatch.setattr(
        chat_service,
        "build_client",
        lambda profile: StreamFallbackThinkingClient(tracker = tracker),
    )
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

    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "test-model",
        enable_deep_thinking = True,
        capabilities = ModelCapabilities(),
    )
    warning_messages: list[str] = []
    chunks = list(
        chat_service.stream_chat(
            request = ChatRequest(user_text = "hello", stream = True),
            profile = profile,
            model = "test-model",
            warning_messages = warning_messages,
            enable_deep_thinking = False,
        )
    )

    calls = tracker.get("calls", [])
    assert "".join(chunks) == "stream fallback success"
    assert len(calls) == 1
    assert "extra_body" not in calls[0]
    assert not warning_messages
