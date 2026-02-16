import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ModelCapabilities:
    """Represent model modality support flags.

    Args:
        supports_text: Whether text input/output is supported.
        supports_image: Whether image input is supported.
        supports_video: Whether video input is supported.
        supports_audio: Whether audio input is supported.
    """

    supports_text: bool | None = None
    supports_image: bool | None = None
    supports_video: bool | None = None
    supports_audio: bool | None = None

    def merge(self, fallback: "ModelCapabilities") -> "ModelCapabilities":
        """Merge current capability values with fallback values.

        Args:
            fallback: The fallback capability object used for missing values.
        """
        return ModelCapabilities(
            supports_text = self.supports_text if self.supports_text is not None else fallback.supports_text,
            supports_image = self.supports_image if self.supports_image is not None else fallback.supports_image,
            supports_video = self.supports_video if self.supports_video is not None else fallback.supports_video,
            supports_audio = self.supports_audio if self.supports_audio is not None else fallback.supports_audio,
        )

    def with_defaults(self) -> "ModelCapabilities":
        """Fill unknown capability values with safe defaults.

        Args:
            self: The current capability object.
        """
        return ModelCapabilities(
            supports_text = True if self.supports_text is None else self.supports_text,
            supports_image = False if self.supports_image is None else self.supports_image,
            supports_video = False if self.supports_video is None else self.supports_video,
            supports_audio = False if self.supports_audio is None else self.supports_audio,
        )

    def as_dict(self) -> dict[str, bool | None]:
        """Convert capability object to dictionary.

        Args:
            self: The capability object instance.
        """
        return {
            "supports_text": self.supports_text,
            "supports_image": self.supports_image,
            "supports_video": self.supports_video,
            "supports_audio": self.supports_audio,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ModelCapabilities":
        """Build capability object from dictionary data.

        Args:
            payload: Raw dictionary that may include capability keys.
        """
        if payload is None:
            return cls()

        return cls(
            supports_text = payload.get("supports_text"),
            supports_image = payload.get("supports_image"),
            supports_video = payload.get("supports_video"),
            supports_audio = payload.get("supports_audio"),
        )


@dataclass
class ProviderProfile:
    """Represent one OpenAI-compatible provider profile.

    Args:
        profile_id: The unique profile identifier.
        base_url: The base URL for OpenAI-compatible endpoint.
        api_key_env: The environment variable name containing API key.
        default_model: The default display model name for this profile.
        model_aliases: Optional mapping from display model name to request model id.
        timeout_seconds: The HTTP timeout in seconds.
        capabilities: Explicit capability declaration from config if provided.
        enable_deep_thinking: Whether to request deep thinking mode when possible.
    """

    profile_id: str
    base_url: str
    api_key_env: str
    default_model: str
    model_aliases: dict[str, str] = field(default_factory = dict)
    timeout_seconds: int = 60
    capabilities: ModelCapabilities | None = None
    enable_deep_thinking: bool = False


@dataclass
class ProfileRegistry:
    """Hold all configured profiles and default selection.

    Args:
        profiles: Mapping from profile id to profile definition.
        default_profile_id: Profile id used when no override is provided.
    """

    profiles: dict[str, ProviderProfile]
    default_profile_id: str


def load_env_file(env_path = ".env") -> None:
    """Load environment variables from .env file.

    Args:
        env_path: File path to the .env file.
    """
    load_dotenv(dotenv_path = env_path, override = False)


def resolve_profiles_path(cli_profiles_path = None) -> str:
    """Resolve profile config path with command-line precedence.

    Args:
        cli_profiles_path: Optional command line override path.
    """
    if cli_profiles_path:
        return cli_profiles_path

    env_profiles_path = os.getenv("LLM_LAB_PROFILES_PATH")
    if env_profiles_path:
        return env_profiles_path

    default_path = Path("config/profiles.yaml")
    if default_path.exists():
        return str(default_path)

    return "config/profiles.example.yaml"


def load_profiles(profiles_path: str) -> ProfileRegistry:
    """Parse profile YAML and validate required fields.

    Args:
        profiles_path: YAML file path for profile definitions.
    """
    path = Path(profiles_path)
    if not path.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")

    with path.open("r", encoding = "utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    raw_profiles = raw_config.get("profiles", {})
    if not raw_profiles:
        raise ValueError(f"`profiles` section is missing or empty in {profiles_path}")

    profiles: dict[str, ProviderProfile] = {}
    required_keys = ["base_url", "api_key_env", "default_model"]
    for profile_id, profile_payload in raw_profiles.items():
        missing_keys = [key for key in required_keys if key not in profile_payload]
        if missing_keys:
            missing_text = ", ".join(missing_keys)
            raise ValueError(
                f"Profile `{profile_id}` missing required fields: {missing_text}"
            )

        capabilities = ModelCapabilities.from_dict(profile_payload.get("capabilities"))
        profiles[profile_id] = ProviderProfile(
            profile_id = profile_id,
            base_url = str(profile_payload["base_url"]),
            api_key_env = str(profile_payload["api_key_env"]),
            default_model = str(profile_payload["default_model"]),
            model_aliases = {
                str(key): str(value)
                for key, value in (profile_payload.get("model_aliases") or {}).items()
            },
            timeout_seconds = int(profile_payload.get("timeout_seconds", 60)),
            capabilities = capabilities,
            enable_deep_thinking = bool(profile_payload.get("enable_deep_thinking", False)),
        )

    default_profile_id = str(raw_config.get("default_profile") or "")
    if not default_profile_id:
        default_profile_id = next(iter(profiles.keys()))
    if default_profile_id not in profiles:
        raise ValueError(
            f"default_profile `{default_profile_id}` is not found in profiles"
        )

    return ProfileRegistry(
        profiles = profiles,
        default_profile_id = default_profile_id,
    )


def resolve_profile(
    registry: ProfileRegistry,
    cli_profile = None,
) -> ProviderProfile:
    """Resolve active profile from CLI override, then env, then YAML default.

    Args:
        registry: Parsed profile registry object.
        cli_profile: Optional profile id from command line.
    """
    profile_id = cli_profile or os.getenv("LLM_LAB_PROFILE") or registry.default_profile_id
    if profile_id not in registry.profiles:
        available = ", ".join(sorted(registry.profiles.keys()))
        raise ValueError(f"Profile `{profile_id}` not found. Available: {available}")
    return registry.profiles[profile_id]


def resolve_model(
    profile: ProviderProfile,
    cli_model = None,
    prefer_profile_default: bool = False,
) -> str:
    """Resolve model name with command-line precedence.

    Args:
        profile: The selected provider profile.
        cli_model: Optional model name from command line.
        prefer_profile_default: Whether profile default should win over env model.
    """
    if cli_model:
        model = cli_model
    elif prefer_profile_default:
        model = profile.default_model or os.getenv("LLM_LAB_MODEL")
    else:
        model = os.getenv("LLM_LAB_MODEL") or profile.default_model
    if not model:
        raise ValueError(f"No model resolved for profile `{profile.profile_id}`")
    return model


def resolve_request_model(profile: ProviderProfile, model_name: str) -> str:
    """Resolve provider request model id from a display model name.

    Args:
        profile: The selected provider profile.
        model_name: Display model name from CLI or UI.
    """
    if not model_name:
        return model_name
    return profile.model_aliases.get(model_name, model_name)


def list_profile_models(profile: ProviderProfile) -> list[str]:
    """List visible model options for one provider profile.

    Args:
        profile: The selected provider profile.
    """
    ordered_models: list[str] = []
    seen: set[str] = set()

    def push_model(model_name: str) -> None:
        """Push one model option into ordered list without duplicates.

        Args:
            model_name: Candidate model display name.
        """
        normalized = model_name.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        ordered_models.append(normalized)

    push_model(profile.default_model)
    for alias_name in profile.model_aliases.keys():
        push_model(alias_name)

    return ordered_models
