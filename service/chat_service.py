from dataclasses import dataclass, field
from typing import Any
from typing import Iterator

from service.capability_service import resolve_capabilities
from utils.config_loader import ProviderProfile
from utils.media_utils import encode_image_to_data_url, extract_video_frames
from utils.openai_client import build_client

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
        error_message: Optional error message if request failed.
        raw_response: Optional raw provider response.
    """

    assistant_text: str
    usage: dict[str, int]
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


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
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if maybe_text:
                    text_parts.append(str(maybe_text))
        return "\n".join(text_parts)

    return str(content) if content is not None else ""


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
) -> ChatResponse:
    """Send one non-stream chat request and return normalized response.

    Args:
        request: Unified chat request payload.
        profile: Active provider profile.
        model: Active model name.
    """
    try:
        client = build_client(profile = profile)
        capabilities = resolve_capabilities(
            profile = profile,
            model = model,
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
            model = model,
            messages = messages,
            stream = False,
        )

        completion = client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        response_text = normalize_message_text(content = message.content)

        raw_response: dict[str, Any] | None = None
        if hasattr(completion, "model_dump"):
            raw_response = completion.model_dump()

        return ChatResponse(
            assistant_text = response_text,
            usage = parse_usage(completion = completion),
            error_message = None,
            raw_response = raw_response,
        )
    except Exception as exc:
        return ChatResponse(
            assistant_text = "",
            usage = {},
            error_message = str(exc),
            raw_response = None,
        )


def stream_chat(
    request: ChatRequest,
    profile: ProviderProfile,
    model: str,
) -> Iterator[str]:
    """Send one stream chat request and yield token chunks.

    Args:
        request: Unified chat request payload.
        profile: Active provider profile.
        model: Active model name.
    """
    client = build_client(profile = profile)
    capabilities = resolve_capabilities(
        profile = profile,
        model = model,
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
        model = model,
        messages = messages,
        stream = True,
    )

    stream_response = client.chat.completions.create(**kwargs)
    for chunk in stream_response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content is None:
            continue
        yield normalize_message_text(content = content)
