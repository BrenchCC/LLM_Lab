import os
import inspect
import logging

from service.chat_service import ChatRequest, send_chat, separate_reasoning_text, stream_chat
from service.session_service import append_message, create_session
from service.session_service import save_session as persist_session
from utils.config_loader import ProfileRegistry, ProviderProfile, list_profile_models

logger = logging.getLogger(__name__)


GRADIO_CSS = """
:root {
  --ink: #0f3148;
  --muted: #60798d;
  --stroke: #cedfe9;
  --teal: #0f766e;
  --teal-dark: #0b5f59;
  --orange: #f97316;
  --card: #ffffff;
}

.gradio-container {
  background:
    radial-gradient(circle at 8% 6%, rgba(15,118,110,.12) 0, rgba(15,118,110,0) 26%),
    radial-gradient(circle at 90% 15%, rgba(249,115,22,.12) 0, rgba(249,115,22,0) 22%),
    linear-gradient(160deg, #f5fbff 0%, #edf7f3 100%);
}

.llm-hero {
  border: 1px solid var(--stroke);
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(15,118,110,.11), rgba(255,255,255,.85));
  padding: 18px 20px;
  margin-bottom: 10px;
}

.llm-hero .kicker {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--teal);
  font-weight: 700;
}

.llm-hero h2 {
  margin: 8px 0 6px 0;
  color: var(--ink);
}

.llm-hero p {
  margin: 0;
  color: var(--muted);
}

.control-panel,
.chat-panel {
  border: 1px solid var(--stroke);
  border-radius: 16px;
  background: rgba(255,255,255,.78);
  padding: 12px;
}

.status-box {
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 8px 12px;
  background: var(--card);
}

button.primary {
  background: var(--teal) !important;
}

button.primary:hover {
  background: var(--teal-dark) !important;
}

.reasoning-loader {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-weight: 600;
}

.reasoning-loader .spinner {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid rgba(15, 118, 110, 0.24);
  border-top-color: var(--teal);
  animation: spin .8s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
"""

GRADIO_HEAD = """
<script>
(() => {
  const INPUT_ROOT_ID = "llm-lab-chat-input";
  const SEND_BUTTON_ROOT_ID = "llm-lab-send-btn";

  function resolveTextarea() {
    const inputRoot = document.getElementById(INPUT_ROOT_ID);
    if (!inputRoot) {
      return null;
    }
    if (inputRoot.tagName && inputRoot.tagName.toLowerCase() === "textarea") {
      return inputRoot;
    }
    return inputRoot.querySelector("textarea");
  }

  function resolveSendButton() {
    const buttonRoot = document.getElementById(SEND_BUTTON_ROOT_ID);
    if (!buttonRoot) {
      return null;
    }
    if (buttonRoot.tagName && buttonRoot.tagName.toLowerCase() === "button") {
      return buttonRoot;
    }
    return buttonRoot.querySelector("button");
  }

  function bindEnterShortcut() {
    const textarea = resolveTextarea();
    const sendButton = resolveSendButton();
    if (!textarea || !sendButton) {
      return;
    }
    if (textarea.dataset.enterShortcutBound === "1") {
      return;
    }

    textarea.dataset.enterShortcutBound = "1";
    textarea.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") {
        return;
      }
      if (event.shiftKey || event.isComposing || event.keyCode === 229) {
        return;
      }
      event.preventDefault();
      sendButton.click();
    });
  }

  bindEnterShortcut();
  const observer = new MutationObserver(() => bindEnterShortcut());
  observer.observe(document.body, {childList: true, subtree: true});
})();
</script>
"""


def _set_no_proxy_entries(host: str, port: int) -> None:
    """Ensure local Gradio traffic bypasses system proxy.

    Args:
        host: Host used by Gradio server.
        port: Port used by Gradio server.
    """
    raw_entries = os.getenv("NO_PROXY", "") or os.getenv("no_proxy", "")
    current_entries = [item.strip() for item in raw_entries.split(",") if item.strip()]
    merged_entries = set(current_entries)

    candidates = {
        "127.0.0.1",
        "localhost",
        "::1",
        host.strip(),
    }
    for candidate in list(candidates):
        if not candidate:
            continue
        merged_entries.add(candidate)
        merged_entries.add(f"{candidate}:{port}")

    normalized = ",".join(sorted(merged_entries))
    os.environ["NO_PROXY"] = normalized
    os.environ["no_proxy"] = normalized


def prepare_gradio_runtime_env(host: str, port: int) -> None:
    """Prepare runtime environment for stable Gradio startup.

    Args:
        host: Host used by Gradio server.
        port: Port used by Gradio server.
    """
    _set_no_proxy_entries(host = host, port = port)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def launch_gradio_with_retry(demo, host: str, port: int, share: bool) -> None:
    """Launch Gradio app with one localhost fallback retry for startup 502.

    Args:
        demo: Gradio Blocks app instance.
        host: Preferred Gradio host.
        port: Preferred Gradio port.
        share: Whether to enable external share link.
    """
    queued_demo = demo.queue()
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share,
    }
    try:
        queued_demo.launch(**launch_kwargs)
        return
    except Exception as exc:
        error_text = str(exc)
        startup_502 = "startup-events" in error_text and "502" in error_text
        if not startup_502:
            raise

        fallback_host = ""
        if host == "127.0.0.1":
            fallback_host = "localhost"
        elif host == "localhost":
            fallback_host = "127.0.0.1"

        if not fallback_host:
            raise

        prepare_gradio_runtime_env(host = fallback_host, port = port)
        logger.warning(
            "Gradio startup event check failed on %s:%s. Retrying with host=%s.",
            host,
            port,
            fallback_host,
        )
        queued_demo.launch(
            server_name = fallback_host,
            server_port = port,
            share = share,
        )


def normalize_uploaded_paths(raw_files) -> list[str]:
    """Normalize Gradio upload payload into local path list.

    Args:
        raw_files: Upload payload from Gradio file components.
    """
    if raw_files is None:
        return []
    if isinstance(raw_files, list):
        return [str(item) for item in raw_files if item]
    return [str(raw_files)]


def build_status_text(
    profile_id: str,
    model_name: str,
    image_count: int,
    video_count: int,
    thinking_enabled: bool | None = None,
    total_tokens: int | None = None,
) -> str:
    """Build human-readable status summary.

    Args:
        profile_id: Active profile id.
        model_name: Active model name.
        image_count: Number of attached images in current turn.
        video_count: Number of attached videos in current turn.
        thinking_enabled: Optional deep thinking switch state.
        total_tokens: Optional total token count from latest response.
    """
    segments = [
        f"**Profile**: `{profile_id}`",
        f"**Model**: `{model_name}`",
        f"**Images**: {image_count}",
        f"**Videos**: {video_count}",
    ]
    if thinking_enabled is not None:
        segments.append(f"**Thinking**: {'ON' if thinking_enabled else 'OFF'}")

    base_text = " | ".join(segments)
    if total_tokens is not None:
        return f"{base_text} | **Total Tokens**: {total_tokens}"
    return base_text


def build_conversation_history_from_session(
    session_payload: dict,
) -> list[dict[str, str]]:
    """Build normalized history messages from session payload.

    Args:
        session_payload: Current chat session dictionary.
    """
    history: list[dict[str, str]] = []
    for message in session_payload.get("messages", []):
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        history.append(
            {
                "role": role,
                "content": content,
            }
        )
    return history


def build_assistant_display_text(
    assistant_text: str,
    reasoning_text: str,
) -> str:
    """Build assistant display content with optional reasoning section.

    Args:
        assistant_text: Final assistant answer text.
        reasoning_text: Extracted reasoning text.
    """
    content = assistant_text.strip() if assistant_text else ""
    reasoning = reasoning_text.strip() if reasoning_text else ""
    if not content:
        content = "(empty response)"
    if not reasoning:
        return content

    safe_reasoning = reasoning.replace("```", "'''")
    return (
        f"{content}\n\n"
        "<details><summary>Reasoning</summary>\n\n"
        f"```text\n{safe_reasoning}\n```\n"
        "</details>"
    )


def build_reasoning_loading_text() -> str:
    """Build temporary assistant bubble shown while model is reasoning.

    Args:
        None: This function does not accept parameters.
    """
    return (
        "<div class='reasoning-loader'>"
        "<span class='spinner'></span>"
        "<span>Reasoning...</span>"
        "</div>"
    )


def build_gradio_theme(gr):
    """Build Gradio theme object with backward-compatible fallback.

    Args:
        gr: Imported Gradio module.
    """
    try:
        return gr.themes.Soft(
            primary_hue = "teal",
            secondary_hue = "amber",
            neutral_hue = "slate",
            radius_size = "lg",
            font = [gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "sans-serif"],
            font_mono = [gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        )
    except Exception:
        return None


def build_chatbot_component(gr):
    """Create Chatbot component with compatibility fallbacks.

    Args:
        gr: Imported Gradio module.
    """
    base_kwargs = {
        "label": "Conversation",
        "height": 560,
    }
    chatbot_signature = inspect.signature(gr.Chatbot.__init__)
    supports_type = "type" in chatbot_signature.parameters

    if supports_type:
        candidate_specs = [
            ("messages", {**base_kwargs, "type": "messages", "bubble_full_width": False, "show_copy_button": True}),
            ("messages", {**base_kwargs, "type": "messages", "show_copy_button": True}),
            ("messages", {**base_kwargs, "type": "messages"}),
            ("tuples", {**base_kwargs, "type": "tuples", "bubble_full_width": False, "show_copy_button": True}),
            ("tuples", {**base_kwargs, "type": "tuples", "show_copy_button": True}),
            ("tuples", {**base_kwargs, "type": "tuples"}),
        ]
    else:
        candidate_specs = [
            ("messages", {**base_kwargs, "buttons": ["copy"]}),
            ("messages", {**base_kwargs}),
        ]

    for format_name, candidate_kwargs in candidate_specs:
        try:
            component = gr.Chatbot(**candidate_kwargs)
            return component, format_name
        except TypeError:
            continue

    return gr.Chatbot(label = "Conversation"), "messages"


def history_to_messages(history) -> list[dict[str, str]]:
    """Normalize Gradio history payload into role/content messages.

    Args:
        history: Gradio chatbot history object in messages or tuples format.
    """
    normalized: list[dict[str, str]] = []
    if not history:
        return normalized

    first_item = history[0]
    if isinstance(first_item, dict) or hasattr(first_item, "role"):
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
            else:
                role = getattr(item, "role", None)
                content = getattr(item, "content", "")
            if role not in {"user", "assistant"}:
                continue
            normalized.append({"role": role, "content": str(content)})
        return normalized

    for pair in history:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        user_text, assistant_text = pair[0], pair[1]
        normalized.append({"role": "user", "content": str(user_text or "")})
        normalized.append({"role": "assistant", "content": str(assistant_text or "")})
    return normalized


def messages_to_chatbot_history(
    messages: list[dict[str, str]],
    chatbot_format: str,
):
    """Convert normalized messages into current chatbot display format.

    Args:
        messages: Role/content message list.
        chatbot_format: Target chatbot format, messages or tuples.
    """
    if chatbot_format == "messages":
        return messages

    tuples_history: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in messages:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            if pending_user is not None:
                tuples_history.append((pending_user, ""))
            pending_user = content
            continue
        if role == "assistant":
            if pending_user is None:
                tuples_history.append(("", content))
            else:
                tuples_history.append((pending_user, content))
                pending_user = None

    if pending_user is not None:
        tuples_history.append((pending_user, ""))

    return tuples_history


def run_gradio_app(
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
    host: str,
    port: int,
    share: bool,
    save_session_enabled: bool,
) -> None:
    """Run Gradio chat web application.

    Args:
        registry: Loaded profile registry.
        profile: Startup selected profile.
        model: Startup selected model.
        host: Gradio host address.
        port: Gradio port.
        share: Whether to enable Gradio share link.
        save_session_enabled: Whether session persistence is enabled by default.
    """
    import gradio as gr
    prepare_gradio_runtime_env(host = host, port = port)

    theme = build_gradio_theme(gr = gr)
    chatbot_format = "messages"

    profile_options = sorted(registry.profiles.keys())
    active_session = create_session(
        profile_id = profile.profile_id,
        model_name = model,
    )

    def build_model_dropdown_options(
        profile_id: str,
        preferred_model: str = "",
    ) -> list[str]:
        """Build model preset choices for selected provider.

        Args:
            profile_id: Selected profile id.
            preferred_model: Optional model that should be pinned in choices.
        """
        selected_profile = registry.profiles[profile_id]
        options = list_profile_models(profile = selected_profile)
        normalized_preferred = preferred_model.strip()

        if normalized_preferred and normalized_preferred not in options:
            options = [normalized_preferred] + options
        if not options and selected_profile.default_model:
            options = [selected_profile.default_model]
        return options

    def resolve_effective_model_name(
        profile_id: str,
        model_preset: str,
        custom_model: str,
    ) -> str:
        """Resolve final model name from preset and optional custom override.

        Args:
            profile_id: Active profile id from UI.
            model_preset: Selected preset model from dropdown.
            custom_model: Custom model text from input.
        """
        custom_value = custom_model.strip() if isinstance(custom_model, str) else ""
        if custom_value:
            return custom_value

        preset_value = model_preset.strip() if isinstance(model_preset, str) else ""
        if preset_value:
            return preset_value
        return registry.profiles[profile_id].default_model

    def update_model_controls(profile_id: str):
        """Update model controls when user switches provider.

        Args:
            profile_id: Selected profile id.
        """
        nonlocal active_session
        selected_profile = registry.profiles[profile_id]
        model_options = build_model_dropdown_options(
            profile_id = profile_id,
            preferred_model = selected_profile.default_model,
        )
        active_session = create_session(
            profile_id = profile_id,
            model_name = selected_profile.default_model,
        )
        empty_history = messages_to_chatbot_history(
            messages = [],
            chatbot_format = chatbot_format,
        )
        status_text = build_status_text(
            profile_id = profile_id,
            model_name = selected_profile.default_model,
            image_count = 0,
            video_count = 0,
            thinking_enabled = bool(selected_profile.enable_deep_thinking),
        )
        return (
            gr.update(
                choices = model_options,
                value = selected_profile.default_model,
            ),
            "",
            bool(selected_profile.enable_deep_thinking),
            empty_history,
            status_text,
        )

    initial_model_options = build_model_dropdown_options(
        profile_id = profile.profile_id,
        preferred_model = model,
    )

    def run_turn(
        user_message: str,
        history,
        profile_id: str,
        model_preset: str,
        custom_model: str,
        image_files,
        video_file,
        temperature: float,
        top_p: float,
        thinking_enabled: bool,
        stream_enabled: bool,
        save_enabled: bool,
    ):
        """Run one Gradio chat turn.

        Args:
            user_message: User input text.
            history: Existing chatbot history.
            profile_id: Active profile id from UI.
            model_preset: Selected preset model from dropdown.
            custom_model: Optional custom model text from input.
            image_files: Uploaded image payload.
            video_file: Uploaded video payload.
            temperature: Current temperature value.
            top_p: Current top-p value.
            thinking_enabled: Whether deep thinking mode is enabled.
            stream_enabled: Whether stream output is enabled.
            save_enabled: Save switch in current UI.
        """
        nonlocal active_session

        effective_model_name = resolve_effective_model_name(
            profile_id = profile_id,
            model_preset = model_preset,
            custom_model = custom_model,
        )
        local_messages = history_to_messages(history = history)
        if not user_message:
            local_history = messages_to_chatbot_history(
                messages = local_messages,
                chatbot_format = chatbot_format,
            )
            status_text = build_status_text(
                profile_id = profile_id,
                model_name = effective_model_name,
                image_count = 0,
                video_count = 0,
                thinking_enabled = thinking_enabled,
            )
            yield local_history, "", status_text
            return

        selected_profile = registry.profiles[profile_id]
        if (
            active_session.get("profile_id") != profile_id
            or active_session.get("model_name") != effective_model_name
        ):
            active_session = create_session(
                profile_id = profile_id,
                model_name = effective_model_name,
            )

        conversation_history = build_conversation_history_from_session(
            session_payload = active_session,
        )
        image_paths = normalize_uploaded_paths(raw_files = image_files)
        video_paths = normalize_uploaded_paths(raw_files = video_file)

        append_message(
            session = active_session,
            role = "user",
            content = user_message,
            metadata = {"images": image_paths, "videos": video_paths},
        )

        request = ChatRequest(
            user_text = user_message,
            image_paths = image_paths,
            video_paths = video_paths,
            conversation_history = conversation_history,
            temperature = temperature,
            top_p = top_p,
            stream = stream_enabled,
        )

        local_messages.append({"role": "user", "content": user_message})
        local_messages.append(
            {
                "role": "assistant",
                "content": build_reasoning_loading_text(),
            }
        )
        pending_history = messages_to_chatbot_history(
            messages = local_messages,
            chatbot_format = chatbot_format,
        )
        base_status = build_status_text(
            profile_id = profile_id,
            model_name = effective_model_name,
            image_count = len(image_paths),
            video_count = len(video_paths),
            thinking_enabled = thinking_enabled,
        )
        pending_status = f"{base_status} | **Status**: Reasoning..."
        yield pending_history, "", pending_status
        usage: dict[str, int] = {}
        reasoning_text = ""
        assistant_text = ""
        assistant_display_text = ""
        warning_messages: list[str] = []

        if stream_enabled:
            chunk_text = ""
            try:
                for chunk in stream_chat(
                    request = request,
                    profile = selected_profile,
                    model = effective_model_name,
                    warning_messages = warning_messages,
                    enable_deep_thinking = thinking_enabled,
                ):
                    chunk_text += chunk
                    partial_answer, _ = separate_reasoning_text(
                        assistant_text = chunk_text,
                    )
                    local_messages[-1] = {
                        "role": "assistant",
                        "content": partial_answer if partial_answer.strip() else build_reasoning_loading_text(),
                    }
                    partial_history = messages_to_chatbot_history(
                        messages = local_messages,
                        chatbot_format = chatbot_format,
                    )
                    yield partial_history, "", f"{base_status} | **Status**: Streaming..."

                assistant_text, reasoning_text = separate_reasoning_text(
                    assistant_text = chunk_text,
                )
                assistant_display_text = build_assistant_display_text(
                    assistant_text = assistant_text,
                    reasoning_text = reasoning_text,
                )
            except Exception as exc:
                assistant_text = f"[ERROR] {exc}"
                reasoning_text = ""
                assistant_display_text = assistant_text
        else:
            response = send_chat(
                request = request,
                profile = selected_profile,
                model = effective_model_name,
                enable_deep_thinking = thinking_enabled,
            )
            assistant_text = response.assistant_text
            if response.error_message:
                assistant_text = f"[ERROR] {response.error_message}"
                reasoning_text = ""
            else:
                reasoning_text = response.reasoning_text
            usage = response.usage
            warning_messages = response.warning_messages
            assistant_display_text = build_assistant_display_text(
                assistant_text = assistant_text,
                reasoning_text = reasoning_text,
            )

        local_messages[-1] = {"role": "assistant", "content": assistant_display_text}
        local_history = messages_to_chatbot_history(
            messages = local_messages,
            chatbot_format = chatbot_format,
        )

        total_tokens = usage.get("total_tokens")
        append_message(
            session = active_session,
            role = "assistant",
            content = assistant_text,
            metadata = {
                "usage": usage,
                "reasoning": reasoning_text,
                "warnings": warning_messages,
            },
        )
        if save_session_enabled or save_enabled:
            persist_session(session = active_session)

        status_text = build_status_text(
            profile_id = profile_id,
            model_name = effective_model_name,
            image_count = len(image_paths),
            video_count = len(video_paths),
            thinking_enabled = thinking_enabled,
            total_tokens = total_tokens,
        )
        if warning_messages:
            warnings_text = " | ".join(warning_messages)
            status_text = f"{status_text}\n\n**Warning**: {warnings_text}"

        yield local_history, "", status_text
        return

    def clear_chat(
        profile_id: str,
        model_preset: str,
        custom_model: str,
        thinking_enabled: bool,
    ):
        """Reset active chat session and clear conversation panel.

        Args:
            profile_id: Active profile id from UI.
            model_preset: Selected preset model from dropdown.
            custom_model: Optional custom model text from input.
            thinking_enabled: Whether deep thinking mode is enabled.
        """
        nonlocal active_session
        effective_model_name = resolve_effective_model_name(
            profile_id = profile_id,
            model_preset = model_preset,
            custom_model = custom_model,
        )
        active_session = create_session(
            profile_id = profile_id,
            model_name = effective_model_name,
        )
        status_text = build_status_text(
            profile_id = profile_id,
            model_name = effective_model_name,
            image_count = 0,
            video_count = 0,
            thinking_enabled = thinking_enabled,
        )
        empty_history = messages_to_chatbot_history(
            messages = [],
            chatbot_format = chatbot_format,
        )
        return empty_history, status_text

    status_default = build_status_text(
        profile_id = profile.profile_id,
        model_name = model,
        image_count = 0,
        video_count = 0,
        thinking_enabled = bool(profile.enable_deep_thinking),
    )

    blocks_kwargs = {
        "title": "llm-lab Gradio UI",
        "css": GRADIO_CSS,
        "head": GRADIO_HEAD,
    }
    if theme is not None:
        blocks_kwargs["theme"] = theme

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <section class="llm-hero">
              <div class="kicker">LLM Lab Interface</div>
              <h2>Multimodal Prompt Playground</h2>
              <p>
                Test OpenAI-compatible models with clean controls for profile switching,
                media upload, and session replay.
              </p>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale = 3, elem_classes = ["control-panel"]):
                gr.Markdown("### Control Center")
                profile_dropdown = gr.Dropdown(
                    label = "Model Provider",
                    choices = profile_options,
                    value = profile.profile_id,
                )
                model_preset_dropdown = gr.Dropdown(
                    label = "Model",
                    choices = initial_model_options,
                    value = model,
                )
                custom_model_input = gr.Textbox(
                    label = "Custom model (optional)",
                    value = "",
                    placeholder = "Leave empty to use selected preset model",
                )
                image_files = gr.Files(
                    label = "Images",
                    file_count = "multiple",
                    type = "filepath",
                )
                video_file = gr.File(label = "Video", type = "filepath")
                temperature = gr.Slider(label = "Temperature", minimum = 0.0, maximum = 2.0, value = 0.7, step = 0.1)
                top_p = gr.Slider(label = "Top-p", minimum = 0.1, maximum = 1.0, value = 1.0, step = 0.1)
                thinking_enabled = gr.Checkbox(
                    label = "Thinking",
                    value = bool(profile.enable_deep_thinking),
                )
                stream_enabled = gr.Checkbox(label = "Stream response", value = False)
                save_enabled = gr.Checkbox(label = "Save session", value = save_session_enabled)

            with gr.Column(scale = 7, elem_classes = ["chat-panel"]):
                status_box = gr.Markdown(status_default, elem_classes = ["status-box"])
                chatbot, chatbot_format = build_chatbot_component(gr = gr)
                user_input = gr.Textbox(
                    label = "Message",
                    placeholder = "Type your prompt. Enter to send, Shift+Enter for newline",
                    lines = 3,
                    elem_id = "llm-lab-chat-input",
                )
                with gr.Row():
                    send_button = gr.Button(
                        "Send",
                        variant = "primary",
                        elem_id = "llm-lab-send-btn",
                    )
                    clear_button = gr.Button("Clear Chat")

        profile_dropdown.change(
            fn = update_model_controls,
            inputs = [profile_dropdown],
            outputs = [model_preset_dropdown, custom_model_input, thinking_enabled, chatbot, status_box],
        )
        model_preset_dropdown.change(
            fn = clear_chat,
            inputs = [profile_dropdown, model_preset_dropdown, custom_model_input, thinking_enabled],
            outputs = [chatbot, status_box],
        )
        custom_model_input.change(
            fn = clear_chat,
            inputs = [profile_dropdown, model_preset_dropdown, custom_model_input, thinking_enabled],
            outputs = [chatbot, status_box],
        )
        send_button.click(
            fn = run_turn,
            inputs = [
                user_input,
                chatbot,
                profile_dropdown,
                model_preset_dropdown,
                custom_model_input,
                image_files,
                video_file,
                temperature,
                top_p,
                thinking_enabled,
                stream_enabled,
                save_enabled,
            ],
            outputs = [chatbot, user_input, status_box],
        )
        clear_button.click(
            fn = clear_chat,
            inputs = [profile_dropdown, model_preset_dropdown, custom_model_input, thinking_enabled],
            outputs = [chatbot, status_box],
        )

    launch_gradio_with_retry(
        demo = demo,
        host = host,
        port = port,
        share = share,
    )
