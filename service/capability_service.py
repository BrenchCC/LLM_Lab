import json
import logging
from pathlib import Path
from typing import Any

from utils.config_loader import ModelCapabilities, ProviderProfile
from utils.openai_client import build_client


logger = logging.getLogger(__name__)

DEFAULT_CAPABILITY_CACHE_PATH = Path("storage/logs/model_cap_cache.json")
FALLBACK_CAPABILITIES = ModelCapabilities(
    supports_text = True,
    supports_image = False,
    supports_video = False,
    supports_audio = False,
)


def capability_key(profile_id: str, model: str) -> str:
    """Build unique cache key for one profile/model pair.

    Args:
        profile_id: Profile identifier.
        model: Model name.
    """
    return f"{profile_id}::{model}"


def read_capability_cache(cache_path: Path) -> dict[str, dict[str, bool | None]]:
    """Read cached capability records from JSON file.

    Args:
        cache_path: File path for capability cache JSON.
    """
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("r", encoding = "utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("Failed to read capability cache: %s", exc)
    return {}


def write_capability_cache(
    cache_path: Path,
    payload: dict[str, dict[str, bool | None]],
) -> None:
    """Write capability cache payload into JSON file.

    Args:
        cache_path: File path for capability cache JSON.
        payload: Dictionary payload to be saved.
    """
    cache_path.parent.mkdir(parents = True, exist_ok = True)
    with cache_path.open("w", encoding = "utf-8") as file:
        json.dump(payload, file, ensure_ascii = False, indent = 2)


def capability_complete(capabilities: ModelCapabilities) -> bool:
    """Check whether all capability flags are explicitly known.

    Args:
        capabilities: Capability object to check.
    """
    return all(
        value is not None
        for value in [
            capabilities.supports_text,
            capabilities.supports_image,
            capabilities.supports_video,
            capabilities.supports_audio,
        ]
    )


def heuristic_capabilities(
    model: str,
    metadata_text: str = "",
) -> ModelCapabilities:
    """Infer capability flags from model name and metadata text heuristics.

    Args:
        model: Model name.
        metadata_text: Additional metadata string.
    """
    text = f"{model} {metadata_text}".lower()
    supports_image = any(
        keyword in text
        for keyword in ["vision", "vl", "image", "multimodal", "omni", "gpt-4o"]
    )
    supports_video = any(keyword in text for keyword in ["video", "videogen"])
    supports_audio = any(keyword in text for keyword in ["audio", "speech", "voice"])

    return ModelCapabilities(
        supports_text = True,
        supports_image = supports_image,
        supports_video = supports_video,
        supports_audio = supports_audio,
    )


def normalize_metadata_text(model_metadata: Any) -> str:
    """Normalize model metadata object into searchable text.

    Args:
        model_metadata: Metadata object returned by provider SDK.
    """
    if hasattr(model_metadata, "model_dump"):
        payload = model_metadata.model_dump()
        return json.dumps(payload, ensure_ascii = False, default = str)

    if isinstance(model_metadata, (dict, list)):
        return json.dumps(model_metadata, ensure_ascii = False, default = str)

    return str(model_metadata)


def detect_from_metadata(client: Any, model: str) -> ModelCapabilities | None:
    """Detect capabilities from model metadata endpoint when available.

    Args:
        client: OpenAI-compatible client instance.
        model: Model name.
    """
    try:
        model_metadata = client.models.retrieve(model)
        metadata_text = normalize_metadata_text(model_metadata = model_metadata)
        return heuristic_capabilities(model = model, metadata_text = metadata_text)
    except Exception as exc:
        logger.info("Metadata detection failed for %s: %s", model, exc)
        return None


def probe_image_support(client: Any, model: str) -> bool:
    """Probe image support by issuing a tiny multimodal request.

    Args:
        client: OpenAI-compatible client instance.
        model: Model name.
    """
    tiny_image_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )
    try:
        client.chat.completions.create(
            model = model,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ping"},
                        {"type": "image_url", "image_url": {"url": tiny_image_data_url}},
                    ],
                }
            ],
            max_tokens = 1,
        )
        return True
    except Exception:
        return False


def resolve_capabilities(
    profile: ProviderProfile,
    model: str,
    client: Any | None = None,
    cache_path: Path = DEFAULT_CAPABILITY_CACHE_PATH,
    force_refresh: bool = False,
) -> ModelCapabilities:
    """Resolve final capabilities with config first and auto-detection fallback.

    Args:
        profile: Active provider profile.
        model: Active model name.
        client: Optional existing client object.
        cache_path: Path to JSON cache file.
        force_refresh: Whether to ignore cache and re-detect.
    """
    explicit = profile.capabilities or ModelCapabilities()
    if capability_complete(explicit):
        return explicit.with_defaults()

    cache_payload = read_capability_cache(cache_path = cache_path)
    cache_record = cache_payload.get(capability_key(profile.profile_id, model))
    cached_capabilities = ModelCapabilities.from_dict(cache_record)

    merged = explicit.merge(fallback = cached_capabilities)
    if not force_refresh and capability_complete(merged):
        return merged.with_defaults()

    local_client = client or build_client(profile = profile)
    detected = detect_from_metadata(client = local_client, model = model)
    if detected:
        merged = merged.merge(fallback = detected)

    if merged.supports_image is None:
        merged = ModelCapabilities(
            supports_text = merged.supports_text,
            supports_image = probe_image_support(client = local_client, model = model),
            supports_video = merged.supports_video,
            supports_audio = merged.supports_audio,
        )

    resolved = merged.merge(fallback = FALLBACK_CAPABILITIES).with_defaults()
    cache_payload[capability_key(profile.profile_id, model)] = resolved.as_dict()
    write_capability_cache(cache_path = cache_path, payload = cache_payload)
    return resolved

