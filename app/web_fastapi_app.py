import logging
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from service.chat_service import ChatRequest, send_chat, separate_reasoning_text, stream_chat
from service.session_service import append_message, create_session
from service.session_service import save_session as persist_session
from utils.config_loader import ProfileRegistry, ProviderProfile, list_profile_models


logger = logging.getLogger(__name__)


class HistoryMessage(BaseModel):
    """Represent one history message from frontend payload.

    Args:
        role: Message role, expected user or assistant.
        content: Message content text.
    """

    role: str
    content: str


class FastAPIChatRequest(BaseModel):
    """Represent one chat request payload from FastAPI frontend.

    Args:
        message: Current user message text.
        profile: Optional profile id selected on frontend.
        model: Optional model id selected on frontend.
        system_prompt: Optional system prompt text.
        history: Optional previous conversation history list.
        temperature: Sampling temperature.
        top_p: Sampling top-p value.
        max_tokens: Optional output token limit.
        enable_thinking: Optional runtime deep-thinking switch.
    """

    message: str = Field(min_length = 1)
    profile: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    history: list[HistoryMessage] = Field(default_factory = list)
    temperature: float = Field(default = 0.7, ge = 0.0, le = 2.0)
    top_p: float = Field(default = 1.0, ge = 0.0, le = 1.0)
    max_tokens: int | None = Field(default = None, ge = 1)
    enable_thinking: bool | None = None


class FastAPIChatResponse(BaseModel):
    """Represent one normalized chat response to frontend.

    Args:
        assistant_text: Assistant output text.
        reasoning_text: Optional reasoning text.
        usage: Token usage dictionary.
        warning_messages: Runtime warning message list.
        error_message: Optional error message text.
        profile: Profile id used for this request.
        model: Model id used for this request.
        session_file: Optional persisted session file path.
    """

    assistant_text: str
    reasoning_text: str
    usage: dict[str, int]
    warning_messages: list[str]
    error_message: str | None
    profile: str
    model: str
    session_file: str | None = None


def build_sse_event(event: str, payload: dict[str, object]) -> str:
    """Build one Server-Sent-Events message block.

    Args:
        event: SSE event type name.
        payload: JSON-serializable event payload.
    """
    serialized = json.dumps(payload, ensure_ascii = False)
    return f"event: {event}\ndata: {serialized}\n\n"


def load_fastapi_page_html() -> str:
    """Load FastAPI frontend HTML page content.

    Args:
        None: This function does not accept parameters.
    """
    page_path = Path(__file__).resolve().with_name("web_fastapi_page.html")
    if not page_path.exists():
        raise FileNotFoundError(f"FastAPI page file not found: {page_path}")
    return page_path.read_text(encoding = "utf-8")


def normalize_history(history: list[HistoryMessage]) -> list[dict[str, str]]:
    """Normalize frontend history list into service-compatible messages.

    Args:
        history: Frontend history message list.
    """
    normalized: list[dict[str, str]] = []
    for message in history:
        role = message.role.strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = message.content.strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def resolve_runtime_profile(
    registry: ProfileRegistry,
    fallback_profile: ProviderProfile,
    requested_profile_id: str | None,
) -> tuple[ProviderProfile, list[str]]:
    """Resolve profile for one request with warning fallback behavior.

    Args:
        registry: Loaded profile registry.
        fallback_profile: Profile selected during startup.
        requested_profile_id: Optional profile id from frontend request.
    """
    warnings: list[str] = []
    profile_id = (requested_profile_id or "").strip()
    if not profile_id:
        return fallback_profile, warnings
    if profile_id in registry.profiles:
        return registry.profiles[profile_id], warnings

    warnings.append(
        f"Unknown profile `{profile_id}`. Fallback to `{fallback_profile.profile_id}`."
    )
    return fallback_profile, warnings


def resolve_runtime_model(
    profile: ProviderProfile,
    fallback_profile: ProviderProfile,
    fallback_model: str,
    requested_model: str | None,
) -> str:
    """Resolve model for one request with profile-aware fallback.

    Args:
        profile: Profile resolved for current request.
        fallback_profile: Startup-selected profile.
        fallback_model: Startup-selected model.
        requested_model: Optional model from frontend request.
    """
    model_name = (requested_model or "").strip()
    if model_name:
        return model_name
    if profile.profile_id == fallback_profile.profile_id:
        return fallback_model
    return profile.default_model


def save_request_session(
    history: list[dict[str, str]],
    user_text: str,
    assistant_text: str,
    profile_id: str,
    model_name: str,
    error_message: str | None = None,
) -> str:
    """Persist current request conversation into session JSON file.

    Args:
        history: Conversation history before current user turn.
        user_text: Current user input text.
        assistant_text: Assistant response text.
        profile_id: Active profile id.
        model_name: Active model name.
        error_message: Optional error message text.
    """
    session_payload = create_session(
        profile_id = profile_id,
        model_name = model_name,
    )

    for message in history:
        append_message(
            session = session_payload,
            role = message["role"],
            content = message["content"],
        )
    append_message(
        session = session_payload,
        role = "user",
        content = user_text,
    )
    append_message(
        session = session_payload,
        role = "assistant",
        content = assistant_text,
        metadata = {"error_message": error_message} if error_message else None,
    )
    return persist_session(session = session_payload)


def build_bootstrap_payload(
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
) -> dict[str, object]:
    """Build bootstrap payload for frontend initialization.

    Args:
        registry: Loaded profile registry.
        profile: Startup-selected profile.
        model: Startup-selected model.
    """
    profile_options: list[dict[str, object]] = []
    for profile_id, profile_item in registry.profiles.items():
        profile_options.append(
            {
                "profile_id": profile_id,
                "default_model": profile_item.default_model,
                "models": list_profile_models(profile = profile_item),
                "enable_deep_thinking": profile_item.enable_deep_thinking,
            }
        )

    return {
        "default_profile": profile.profile_id,
        "default_model": model,
        "profiles": profile_options,
    }


def create_fastapi_app(
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
    save_session_enabled: bool = False,
) -> FastAPI:
    """Create FastAPI app instance for llm-lab web mode.

    Args:
        registry: Loaded profile registry.
        profile: Startup-selected profile.
        model: Startup-selected model.
        save_session_enabled: Whether to persist each request conversation.
    """
    page_html = load_fastapi_page_html()
    app = FastAPI(
        title = "LLM Lab FastAPI UI",
        version = "0.1.0",
        docs_url = None,
        redoc_url = None,
    )

    # 挂载静态文件目录
    from fastapi.staticfiles import StaticFiles
    import os
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    assets_path = os.path.join(project_root, "assets")
    app.mount("/assets", StaticFiles(directory = assets_path), name = "assets")

    @app.get("/", response_class = HTMLResponse)
    def render_home_page() -> HTMLResponse:
        """Render FastAPI web page.

        Args:
            None: This endpoint does not accept parameters.
        """
        return HTMLResponse(content = page_html)

    @app.get("/api/bootstrap")
    def bootstrap_runtime_info() -> dict[str, object]:
        """Return runtime bootstrap payload for frontend.

        Args:
            None: This endpoint does not accept parameters.
        """
        return build_bootstrap_payload(
            registry = registry,
            profile = profile,
            model = model,
        )

    @app.post("/api/chat", response_model = FastAPIChatResponse)
    def chat_endpoint(payload: FastAPIChatRequest) -> FastAPIChatResponse:
        """Handle one chat request from frontend page.

        Args:
            payload: Request payload body from frontend.
        """
        user_text = payload.message.strip()
        if not user_text:
            raise HTTPException(status_code = 400, detail = "message cannot be empty")

        selected_profile, profile_warnings = resolve_runtime_profile(
            registry = registry,
            fallback_profile = profile,
            requested_profile_id = payload.profile,
        )
        selected_model = resolve_runtime_model(
            profile = selected_profile,
            fallback_profile = profile,
            fallback_model = model,
            requested_model = payload.model,
        )
        history = normalize_history(history = payload.history)

        response = send_chat(
            request = ChatRequest(
                user_text = user_text,
                system_prompt = payload.system_prompt,
                conversation_history = history,
                stream = False,
                temperature = payload.temperature,
                top_p = payload.top_p,
                max_tokens = payload.max_tokens,
            ),
            profile = selected_profile,
            model = selected_model,
            enable_deep_thinking = payload.enable_thinking,
        )

        warning_messages = profile_warnings + response.warning_messages
        session_file: str | None = None
        if save_session_enabled:
            try:
                session_file = save_request_session(
                    history = history,
                    user_text = user_text,
                    assistant_text = response.assistant_text,
                    profile_id = selected_profile.profile_id,
                    model_name = selected_model,
                    error_message = response.error_message,
                )
            except Exception as exc:
                logger.warning("Failed to persist FastAPI session: %s", exc)
                warning_messages.append(f"Failed to persist session: {exc}")

        return FastAPIChatResponse(
            assistant_text = response.assistant_text,
            reasoning_text = response.reasoning_text,
            usage = response.usage,
            warning_messages = warning_messages,
            error_message = response.error_message,
            profile = selected_profile.profile_id,
            model = selected_model,
            session_file = session_file,
        )

    @app.post("/api/chat/stream")
    def chat_stream_endpoint(payload: FastAPIChatRequest) -> StreamingResponse:
        """Handle one streaming chat request via Server-Sent Events.

        Args:
            payload: Request payload body from frontend.
        """
        user_text = payload.message.strip()
        if not user_text:
            raise HTTPException(status_code = 400, detail = "message cannot be empty")

        selected_profile, profile_warnings = resolve_runtime_profile(
            registry = registry,
            fallback_profile = profile,
            requested_profile_id = payload.profile,
        )
        selected_model = resolve_runtime_model(
            profile = selected_profile,
            fallback_profile = profile,
            fallback_model = model,
            requested_model = payload.model,
        )
        history = normalize_history(history = payload.history)

        def event_generator():
            """Yield streaming events for one chat request.

            Args:
                None: This generator does not accept parameters.
            """
            warning_messages = list(profile_warnings)
            raw_text = ""
            assistant_text = ""
            reasoning_text = ""
            error_message: str | None = None

            request = ChatRequest(
                user_text = user_text,
                system_prompt = payload.system_prompt,
                conversation_history = history,
                stream = True,
                temperature = payload.temperature,
                top_p = payload.top_p,
                max_tokens = payload.max_tokens,
            )

            try:
                for chunk in stream_chat(
                    request = request,
                    profile = selected_profile,
                    model = selected_model,
                    warning_messages = warning_messages,
                    enable_deep_thinking = payload.enable_thinking,
                ):
                    raw_text += chunk
                    yield build_sse_event(
                        event = "token",
                        payload = {"delta": chunk},
                    )
                assistant_text, reasoning_text = separate_reasoning_text(
                    assistant_text = raw_text,
                )
            except Exception as exc:
                error_message = str(exc)
                logger.warning("FastAPI stream chat failed: %s", exc)
                yield build_sse_event(
                    event = "error",
                    payload = {"error_message": error_message},
                )

            session_file: str | None = None
            if save_session_enabled:
                try:
                    persisted_text = assistant_text
                    if error_message:
                        persisted_text = f"[ERROR] {error_message}"
                    session_file = save_request_session(
                        history = history,
                        user_text = user_text,
                        assistant_text = persisted_text,
                        profile_id = selected_profile.profile_id,
                        model_name = selected_model,
                        error_message = error_message,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist FastAPI stream session: %s", exc)
                    warning_messages.append(f"Failed to persist session: {exc}")

            done_payload: dict[str, object] = {
                "assistant_text": assistant_text,
                "reasoning_text": reasoning_text,
                "warning_messages": warning_messages,
                "error_message": error_message,
                "profile": selected_profile.profile_id,
                "model": selected_model,
                "session_file": session_file,
            }
            yield build_sse_event(event = "done", payload = done_payload)

        return StreamingResponse(
            content = event_generator(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def run_fastapi_app(
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
    host: str,
    port: int,
    save_session_enabled: bool = False,
) -> None:
    """Run FastAPI app with uvicorn server.

    Args:
        registry: Loaded profile registry.
        profile: Startup-selected profile.
        model: Startup-selected model.
        host: Bind host.
        port: Bind port.
        save_session_enabled: Whether to persist each request conversation.
    """
    import uvicorn

    app = create_fastapi_app(
        registry = registry,
        profile = profile,
        model = model,
        save_session_enabled = save_session_enabled,
    )
    uvicorn.run(app, host = host, port = port)
