from pathlib import Path

import pytest

from utils.config_loader import ProviderProfile, load_profiles, resolve_model, resolve_profile
from utils.config_loader import list_profile_models
from utils.config_loader import resolve_request_model


def write_profiles_file(
    tmp_path: Path,
    file_name: str = "profiles.yaml",
    enable_deep_thinking: bool = False,
) -> str:
    """Create a minimal profile YAML file for tests.

    Args:
        tmp_path: Pytest temporary directory fixture path.
        file_name: Target profile file name.
        enable_deep_thinking: Whether to set thinking mode flag in profile.
    """
    target_path = tmp_path / file_name
    target_path.write_text(
        (
            "default_profile: p1\n"
            "profiles:\n"
            "  p1:\n"
            "    base_url: https://api.example.com/v1\n"
            "    api_key_env: TEST_API_KEY\n"
            "    default_model: test-model\n"
            "    model_aliases:\n"
            "      test-model: endpoint-001\n"
            "    timeout_seconds: 30\n"
            f"    enable_deep_thinking: {'true' if enable_deep_thinking else 'false'}\n"
            "    capabilities:\n"
            "      supports_text: true\n"
            "      supports_image: false\n"
            "      supports_video: false\n"
            "      supports_audio: false\n"
        ),
        encoding = "utf-8",
    )
    return str(target_path)


def test_load_profiles_and_resolve_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify profile and model resolution with default precedence.

    Args:
        tmp_path: Pytest temporary directory fixture path.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("LLM_LAB_PROFILE", raising = False)
    monkeypatch.delenv("LLM_LAB_MODEL", raising = False)

    profiles_path = write_profiles_file(tmp_path = tmp_path)
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = None)
    model = resolve_model(profile = profile, cli_model = None)

    assert profile.profile_id == "p1"
    assert model == "test-model"


def test_resolve_profile_with_cli_override(tmp_path: Path) -> None:
    """Verify CLI profile override has highest priority.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    profiles_path = write_profiles_file(tmp_path = tmp_path)
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = "p1")
    assert profile.profile_id == "p1"


def test_missing_required_profile_fields(tmp_path: Path) -> None:
    """Verify missing required profile fields raise clear errors.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    target_path = tmp_path / "bad_profiles.yaml"
    target_path.write_text(
        (
            "default_profile: p1\n"
            "profiles:\n"
            "  p1:\n"
            "    base_url: https://api.example.com/v1\n"
        ),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_profiles(profiles_path = str(target_path))
    assert "missing required fields" in str(exc.value)


def test_resolve_model_uses_env_when_no_cli_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify env model wins when no explicit CLI profile is provided.

    Args:
        tmp_path: Pytest temporary directory fixture path.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("LLM_LAB_MODEL", "env-model")

    profiles_path = write_profiles_file(tmp_path = tmp_path)
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = None)
    model = resolve_model(
        profile = profile,
        cli_model = None,
        prefer_profile_default = False,
    )

    assert model == "env-model"


def test_resolve_model_prefers_profile_default_when_cli_profile_specified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI profile switch uses profile default when model is not specified.

    Args:
        tmp_path: Pytest temporary directory fixture path.
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("LLM_LAB_MODEL", "env-model")

    profiles_path = write_profiles_file(tmp_path = tmp_path)
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = "p1")
    model = resolve_model(
        profile = profile,
        cli_model = None,
        prefer_profile_default = True,
    )

    assert model == "test-model"


def test_load_profiles_with_deep_thinking_flag(tmp_path: Path) -> None:
    """Verify deep-thinking config can be loaded from profile YAML.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    profiles_path = write_profiles_file(
        tmp_path = tmp_path,
        enable_deep_thinking = True,
    )
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = "p1")

    assert profile.enable_deep_thinking is True


def test_resolve_request_model_with_alias_mapping(tmp_path: Path) -> None:
    """Verify display model can be mapped to provider request model id.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    profiles_path = write_profiles_file(tmp_path = tmp_path)
    registry = load_profiles(profiles_path = profiles_path)
    profile = resolve_profile(registry = registry, cli_profile = "p1")

    mapped = resolve_request_model(profile = profile, model_name = "test-model")
    passthrough = resolve_request_model(profile = profile, model_name = "custom-model")

    assert mapped == "endpoint-001"
    assert passthrough == "custom-model"


def test_list_profile_models_returns_default_then_aliases_without_duplicates() -> None:
    """Verify visible model list preserves default-first unique order.

    Args:
        None: This function does not accept parameters.
    """
    profile = ProviderProfile(
        profile_id = "p1",
        base_url = "https://api.example.com/v1",
        api_key_env = "TEST_API_KEY",
        default_model = "model-a",
        model_aliases = {
            "model-b": "endpoint-002",
            "model-a": "endpoint-001",
            "model-c": "endpoint-003",
        },
    )

    models = list_profile_models(profile = profile)
    assert models == ["model-a", "model-b", "model-c"]
