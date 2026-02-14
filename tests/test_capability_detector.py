from pathlib import Path

from service.capability_service import resolve_capabilities
from utils.config_loader import ModelCapabilities, ProviderProfile


class FakeModels:
    """Fake models endpoint object for capability tests."""

    def retrieve(self, model: str):
        """Return fake metadata payload.

        Args:
            model: Model name.
        """
        return {"id": model, "description": "multimodal image vision model"}


class FakeClient:
    """Fake OpenAI-compatible client object for capability tests."""

    def __init__(self):
        """Initialize fake client internals.

        Args:
            self: Fake client instance.
        """
        self.models = FakeModels()


def test_resolve_capabilities_from_metadata(tmp_path: Path) -> None:
    """Verify metadata heuristics can infer image capability.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "vision-model",
        capabilities = ModelCapabilities(),
    )
    cache_path = tmp_path / "cap_cache.json"

    capabilities = resolve_capabilities(
        profile = profile,
        model = "vision-model",
        client = FakeClient(),
        cache_path = cache_path,
    )

    assert capabilities.supports_text is True
    assert capabilities.supports_image is True
    assert capabilities.supports_video is False


def test_explicit_capability_has_priority(tmp_path: Path) -> None:
    """Verify explicit profile capability overrides auto-detection.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "vision-model",
        capabilities = ModelCapabilities(
            supports_text = True,
            supports_image = False,
            supports_video = False,
            supports_audio = False,
        ),
    )
    cache_path = tmp_path / "cap_cache.json"

    capabilities = resolve_capabilities(
        profile = profile,
        model = "vision-model",
        client = FakeClient(),
        cache_path = cache_path,
    )

    assert capabilities.supports_image is False

