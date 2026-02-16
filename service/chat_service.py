from dataclasses import dataclass, field
import logging
import re
import warnings
from typing import Any
from typing import Iterator

from service.capability_service import resolve_capabilities
from utils.config_loader import ProviderProfile, resolve_request_model
from utils.media_utils import encode_image_to_data_url, extract_video_frames
from utils.openai_client import build_client

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """
你叫做 Brench's Bot，是 Brench 的专属 AI 助手。
你的核心目标是：准确、简洁、可执行地帮助 Brench 完成任务。

行为要求：
1. 默认使用中文回答，除非 Brench 明确要求英文。
2. 优先给出可直接执行的方案、命令或代码，不空泛。
3. 对不确定的信息要明确说明不确定，并给出验证建议。
4. 在编码场景下，尽量保持改动最小、结构清晰、风格一致。
5. 不编造事实，不虚构执行结果。

输出风格：
- 先给结论，再给关键细节。
- 保持专业、直接、礼貌。
- 避免冗长和无关内容。
""".strip()

REASONING_ITEM_TYPES = {
    "analysis",
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "thinking",
}

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags = re.IGNORECASE | re.DOTALL)
DEEP_THINKING_UNSUPPORTED_PATTERNS = {
    "does not support",
    "not support",
    "unsupported",
    "unknown parameter",
    "unrecognized parameter",
    "invalid parameter",
    "extra inputs are not permitted",
}


@dataclass
class ChatRequest:
    """Represent one unified chat request.

    Args:
        user_text: Input text from user.
        image_paths: Optional list of image file paths.
        video_paths: Optional list of video file paths.
        system_prompt: Optional system instruction text.
        conversation_history: Previous conversation messages for context.
        stream: Whether to use streaming response.
        temperature: Sampling temperature.
        top_p: Sampling top-p value.
        max_tokens: Optional output token limit.
    """

    user_text: str
    image_paths: list[str] = field(default_factory = list)
    video_paths: list[str] = field(default_factory = list)
    system_prompt: str | None = None
    conversation_history: list[dict[str, str]] = field(default_factory = list)
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None


@dataclass
class ChatResponse:
    """Represent one unified chat response.

    Args:
        assistant_text: Assistant output text.
        usage: Token usage summary.
        reasoning_text: Optional extracted reasoning text.
        warning_messages: Optional warning text list.
        error_message: Optional error message if request failed.
        raw_response: Optional raw provider response.
    """

    assistant_text: str
    usage: dict[str, int]
    reasoning_text: str = ""
    warning_messages: list[str] = field(default_factory = list)
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


def read_message_field(content: Any, field_name: str) -> Any:
    """Read one field from dict/object with best-effort compatibility.

    Args:
        content: Input object from provider payload.
        field_name: Field name to read.
    """
    if isinstance(content, dict):
        return content.get(field_name)
    return getattr(content, field_name, None)


def is_reasoning_item(content_item: Any) -> bool:
    """Check whether one content item is reasoning-type payload.

    Args:
        content_item: One message content block.
    """
    item_type = str(read_message_field(content_item, "type") or "").strip().lower()
    return item_type in REASONING_ITEM_TYPES


def normalize_message_text(content: Any) -> str:
    """Normalize provider message content into plain text.

    Args:
        content: Message content object from provider response.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if is_reasoning_item(content_item = item):
                continue
            maybe_text = read_message_field(item, "text")
            if maybe_text:
                text_parts.append(str(maybe_text))
                continue
            maybe_content = read_message_field(item, "content")
            if isinstance(maybe_content, str) and maybe_content.strip():
                text_parts.append(maybe_content)
        return "\n".join(text_parts)

    return str(content) if content is not None else ""


def normalize_reasoning_payload(content: Any) -> str:
    """Normalize arbitrary reasoning payload object into plain text.

    Args:
        content: Arbitrary reasoning payload object.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            normalized = normalize_reasoning_payload(content = item)
            if normalized:
                parts.append(normalized)
        return "\n\n".join(parts)

    text_candidates = [
        read_message_field(content, "text"),
        read_message_field(content, "content"),
        read_message_field(content, "summary"),
    ]
    for candidate in text_candidates:
        normalized = normalize_reasoning_payload(content = candidate)
        if normalized:
            return normalized

    return str(content).strip()


def extract_reasoning_from_message(message: Any) -> str:
    """Extract reasoning content from message payload when available.

    Args:
        message: Assistant message object from completion response.
    """
    reasoning_parts: list[str] = []

    for field_name in ["reasoning_content", "reasoning", "thinking", "analysis"]:
        field_value = read_message_field(message, field_name)
        normalized = normalize_reasoning_payload(content = field_value)
        if normalized:
            reasoning_parts.append(normalized)

    content = read_message_field(message, "content")
    if isinstance(content, list):
        for item in content:
            if not is_reasoning_item(content_item = item):
                continue
            reasoning_value = (
                read_message_field(item, "text")
                or read_message_field(item, "content")
                or read_message_field(item, "summary")
            )
            normalized = normalize_reasoning_payload(content = reasoning_value)
            if normalized:
                reasoning_parts.append(normalized)

    deduplicated_parts: list[str] = []
    seen: set[str] = set()
    for part in reasoning_parts:
        key = part.strip()
        if not key or key in seen:
            continue
        deduplicated_parts.append(key)
        seen.add(key)
    return "\n\n".join(deduplicated_parts)


def split_reasoning_think_blocks(text: str) -> tuple[str, str]:
    """Split `<think>...</think>` blocks from answer content.

    Args:
        text: Input assistant text.
    """
    matches = THINK_BLOCK_PATTERN.findall(text)
    if not matches:
        return "", text.strip()

    reasoning_parts = [match.strip() for match in matches if match and match.strip()]
    cleaned_answer = THINK_BLOCK_PATTERN.sub("", text).strip()
    return "\n\n".join(reasoning_parts), cleaned_answer


def separate_reasoning_text(
    assistant_text: str,
    explicit_reasoning_text: str = "",
) -> tuple[str, str]:
    """Separate final answer and reasoning text from one assistant output.

    Args:
        assistant_text: Assistant output text.
        explicit_reasoning_text: Reasoning text extracted from dedicated fields.
    """
    base_answer = assistant_text.strip()
    base_reasoning = explicit_reasoning_text.strip()
    tag_reasoning, cleaned_answer = split_reasoning_think_blocks(text = base_answer)

    if tag_reasoning and base_reasoning:
        if tag_reasoning != base_reasoning:
            base_reasoning = f"{base_reasoning}\n\n{tag_reasoning}"
    elif tag_reasoning:
        base_reasoning = tag_reasoning

    final_reasoning = normalize_reasoning_payload(content = base_reasoning)
    return cleaned_answer, final_reasoning


def merge_image_inputs(
    image_paths: list[str],
    video_paths: list[str],
) -> list[str]:
    """Merge direct image paths and extracted video frame paths.

    Args:
        image_paths: User supplied image paths.
        video_paths: User supplied video paths.
    """
    merged = list(image_paths)
    for video_path in video_paths:
        merged.extend(
            extract_video_frames(
                video_path = video_path,
                fps = 1,
                max_frames = 8,
                resize_max = 1024,
            )
        )
    return merged


def build_user_content(
    user_text: str,
    image_paths: list[str],
) -> str | list[dict[str, Any]]:
    """Build OpenAI-compatible user content blocks.

    Args:
        user_text: User input text.
        image_paths: Image paths for multimodal content.
    """
    if not image_paths:
        return user_text

    content: list[dict[str, Any]] = []
    if user_text:
        content.append({"type": "text", "text": user_text})

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image_path = image_path)},
            }
        )

    return content


def build_messages(
    request: ChatRequest,
    image_paths: list[str],
) -> list[dict[str, Any]]:
    """Build complete message list for one chat completion request.

    Args:
        request: Unified chat request payload.
        image_paths: Prepared image paths for this request.
    """
    messages: list[dict[str, Any]] = []
    system_prompt = request.system_prompt.strip() if request.system_prompt else DEFAULT_SYSTEM_PROMPT
    messages.append({"role": "system", "content": system_prompt})

    for history_item in request.conversation_history:
        role = history_item.get("role", "")
        content = history_item.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": build_user_content(
                user_text = request.user_text,
                image_paths = image_paths,
            ),
        }
    )
    return messages


def parse_usage(completion: Any) -> dict[str, int]:
    """Parse usage object from completion response.

    Args:
        completion: Completion response object.
    """
    usage = getattr(completion, "usage", None)
    if not usage:
        return {}

    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def build_completion_kwargs(
    request: ChatRequest,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
) -> dict[str, Any]:
    """Build completion call arguments.

    Args:
        request: Unified chat request payload.
        model: Target model name.
        messages: Provider message list.
        stream: Whether response stream mode is enabled.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    return kwargs


def with_deep_thinking_enabled(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build completion kwargs with deep-thinking switch enabled.

    Args:
        kwargs: Base completion keyword arguments.
    """
    payload = dict(kwargs)
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict):
        merged_extra_body = dict(extra_body)
    else:
        merged_extra_body = {}
    merged_extra_body["enable_thinking"] = True
    payload["extra_body"] = merged_extra_body
    return payload


def without_deep_thinking(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build completion kwargs with deep-thinking switch removed.

    Args:
        kwargs: Completion keyword arguments that may include thinking flag.
    """
    payload = dict(kwargs)
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        return payload

    merged_extra_body = dict(extra_body)
    merged_extra_body.pop("enable_thinking", None)
    if merged_extra_body:
        payload["extra_body"] = merged_extra_body
    else:
        payload.pop("extra_body", None)
    return payload


def is_deep_thinking_unsupported_error(error_message: str) -> bool:
    """Check whether an error indicates deep-thinking flag is not supported.

    Args:
        error_message: Provider exception message text.
    """
    message = (error_message or "").lower()
    if "unexpected keyword argument" in message and "extra_body" in message:
        return True
    if "enable_thinking" not in message and "thinking" not in message:
        return False
    return any(pattern in message for pattern in DEEP_THINKING_UNSUPPORTED_PATTERNS)


def build_deep_thinking_warning(profile: ProviderProfile, model: str) -> str:
    """Build warning text for deep-thinking fallback.

    Args:
        profile: Active provider profile.
        model: Active request model id.
    """
    return (
        f"Profile `{profile.profile_id}` configured `enable_deep_thinking = true`, "
        f"but model `{model}` does not support it. Falling back to normal mode."
    )


def create_completion_with_thinking_fallback(
    client: Any,
    profile: ProviderProfile,
    request_model: str,
    kwargs: dict[str, Any],
    enable_deep_thinking: bool | None = None,
) -> tuple[Any, list[str]]:
    """Create completion and gracefully fallback when thinking flag is unsupported.

    Args:
        client: OpenAI-compatible client.
        profile: Active provider profile.
        request_model: Request model id after alias resolution.
        kwargs: Base completion keyword arguments.
        enable_deep_thinking: Optional runtime override for thinking flag.
    """
    thinking_enabled = profile.enable_deep_thinking if enable_deep_thinking is None else enable_deep_thinking
    if not thinking_enabled:
        return client.chat.completions.create(**kwargs), []

    kwargs_with_thinking = with_deep_thinking_enabled(kwargs = kwargs)
    try:
        completion = client.chat.completions.create(**kwargs_with_thinking)
        return completion, []
    except Exception as exc:
        if not is_deep_thinking_unsupported_error(error_message = str(exc)):
            raise

        warning_message = build_deep_thinking_warning(
            profile = profile,
            model = request_model,
        )
        logger.warning("%s Provider error: %s", warning_message, exc)
        warnings.warn(warning_message, UserWarning, stacklevel = 2)

        kwargs_without_thinking = without_deep_thinking(kwargs = kwargs_with_thinking)
        completion = client.chat.completions.create(**kwargs_without_thinking)
        return completion, [warning_message]


def validate_modalities(
    request: ChatRequest,
    supports_image: bool,
) -> None:
    """Validate request modalities against capability flags.

    Args:
        request: Unified chat request payload.
        supports_image: Whether image input is supported by model.
    """
    if request.image_paths and not supports_image:
        raise ValueError("Current model does not support image input.")
    if request.video_paths and not supports_image:
        raise ValueError(
            "Current model does not support video/image input for frame-based video mode."
        )


def send_chat(
    request: ChatRequest,
    profile: ProviderProfile,
    model: str,
    enable_deep_thinking: bool | None = None,
) -> ChatResponse:
    """Send one non-stream chat request and return normalized response.

    Args:
        request: Unified chat request payload.
        profile: Active provider profile.
        model: Active model name.
        enable_deep_thinking: Optional runtime override for thinking flag.
    """
    try:
        client = build_client(profile = profile)
        request_model = resolve_request_model(profile = profile, model_name = model)
        capabilities = resolve_capabilities(
            profile = profile,
            model = request_model,
            client = client,
        )
        validate_modalities(
            request = request,
            supports_image = bool(capabilities.supports_image),
        )

        merged_images = merge_image_inputs(
            image_paths = request.image_paths,
            video_paths = request.video_paths,
        )
        messages = build_messages(request = request, image_paths = merged_images)
        kwargs = build_completion_kwargs(
            request = request,
            model = request_model,
            messages = messages,
            stream = False,
        )

        completion, warning_messages = create_completion_with_thinking_fallback(
            client = client,
            profile = profile,
            request_model = request_model,
            kwargs = kwargs,
            enable_deep_thinking = enable_deep_thinking,
        )
        message = completion.choices[0].message
        response_text = normalize_message_text(content = message.content)
        reasoning_text = extract_reasoning_from_message(message = message)
        response_text, reasoning_text = separate_reasoning_text(
            assistant_text = response_text,
            explicit_reasoning_text = reasoning_text,
        )

        raw_response: dict[str, Any] | None = None
        if hasattr(completion, "model_dump"):
            raw_response = completion.model_dump()

        return ChatResponse(
            assistant_text = response_text,
            reasoning_text = reasoning_text,
            usage = parse_usage(completion = completion),
            warning_messages = warning_messages,
            error_message = None,
            raw_response = raw_response,
        )
    except Exception as exc:
        return ChatResponse(
            assistant_text = "",
            reasoning_text = "",
            usage = {},
            warning_messages = [],
            error_message = str(exc),
            raw_response = None,
        )


def stream_chat(
    request: ChatRequest,
    profile: ProviderProfile,
    model: str,
    warning_messages: list[str] | None = None,
    enable_deep_thinking: bool | None = None,
) -> Iterator[str]:
    """Send one stream chat request and yield token chunks.

    Args:
        request: Unified chat request payload.
        profile: Active provider profile.
        model: Active model name.
        warning_messages: Optional warning message output list.
        enable_deep_thinking: Optional runtime override for thinking flag.
    """
    client = build_client(profile = profile)
    request_model = resolve_request_model(profile = profile, model_name = model)
    capabilities = resolve_capabilities(
        profile = profile,
        model = request_model,
        client = client,
    )
    validate_modalities(
        request = request,
        supports_image = bool(capabilities.supports_image),
    )

    merged_images = merge_image_inputs(
        image_paths = request.image_paths,
        video_paths = request.video_paths,
    )
    messages = build_messages(request = request, image_paths = merged_images)
    kwargs = build_completion_kwargs(
        request = request,
        model = request_model,
        messages = messages,
        stream = True,
    )

    stream_response, stream_warnings = create_completion_with_thinking_fallback(
        client = client,
        profile = profile,
        request_model = request_model,
        kwargs = kwargs,
        enable_deep_thinking = enable_deep_thinking,
    )
    if warning_messages is not None and stream_warnings:
        warning_messages.extend(stream_warnings)

    for chunk in stream_response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content is None:
            continue
        yield normalize_message_text(content = content)
