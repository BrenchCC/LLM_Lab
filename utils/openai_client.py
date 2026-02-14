import os
from typing import Any

from utils.config_loader import ProviderProfile


def resolve_api_key(
    profile: ProviderProfile,
    api_key_override = None,
) -> str:
    """Resolve API key for a profile from override or environment.

    Args:
        profile: The selected provider profile.
        api_key_override: Optional key from command line.
    """
    api_key = api_key_override or os.getenv(profile.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Please set environment variable `{profile.api_key_env}`."
        )
    return api_key


def build_client(
    profile: ProviderProfile,
    api_key_override = None,
) -> Any:
    """Create an OpenAI-compatible client for the active profile.

    Args:
        profile: The selected provider profile.
        api_key_override: Optional API key override from command line.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Package `openai` is required but not installed.") from exc

    api_key = resolve_api_key(
        profile = profile,
        api_key_override = api_key_override,
    )
    client = OpenAI(
        base_url = profile.base_url,
        api_key = api_key,
        timeout = profile.timeout_seconds,
    )
    return client

