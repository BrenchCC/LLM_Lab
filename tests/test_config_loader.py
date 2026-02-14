from pathlib import Path

import pytest

from utils.config_loader import load_profiles, resolve_model, resolve_profile


def write_profiles_file(tmp_path: Path, file_name: str = "profiles.yaml") -> str:
    """Create a minimal profile YAML file for tests.

    Args:
        tmp_path: Pytest temporary directory fixture path.
        file_name: Target profile file name.
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
            "    timeout_seconds: 30\n"
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
