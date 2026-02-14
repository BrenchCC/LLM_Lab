import inspect

from service.chat_service import ChatRequest, send_chat
from service.session_service import append_message, create_session
from service.session_service import save_session as persist_session
from utils.config_loader import ProfileRegistry, ProviderProfile


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
"""


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
    total_tokens: int | None = None,
) -> str:
    """Build human-readable status summary.

    Args:
        profile_id: Active profile id.
        model_name: Active model name.
        image_count: Number of attached images in current turn.
        video_count: Number of attached videos in current turn.
        total_tokens: Optional total token count from latest response.
    """
    base_text = (
        f"**Profile**: `{profile_id}` | "
        f"**Model**: `{model_name}` | "
        f"**Images**: {image_count} | "
        f"**Videos**: {video_count}"
    )
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

    theme = build_gradio_theme(gr = gr)
    chatbot_format = "messages"

    profile_options = sorted(registry.profiles.keys())
    active_session = create_session(
        profile_id = profile.profile_id,
        model_name = model,
    )

    def update_model(profile_id: str) -> str:
        """Return default model when user switches profile.

        Args:
            profile_id: Selected profile id.
        """
        return registry.profiles[profile_id].default_model

    def run_turn(
        user_message: str,
        history,
        profile_id: str,
        model_name: str,
        image_files,
        video_file,
        temperature: float,
        top_p: float,
        save_enabled: bool,
    ):
        """Run one Gradio chat turn.

        Args:
            user_message: User input text.
            history: Existing chatbot history.
            profile_id: Active profile id from UI.
            model_name: Active model name from UI.
            image_files: Uploaded image payload.
            video_file: Uploaded video payload.
            temperature: Current temperature value.
            top_p: Current top-p value.
            save_enabled: Save switch in current UI.
        """
        nonlocal active_session

        local_messages = history_to_messages(history = history)
        if not user_message:
            local_history = messages_to_chatbot_history(
                messages = local_messages,
                chatbot_format = chatbot_format,
            )
            status_text = build_status_text(
                profile_id = profile_id,
                model_name = model_name,
                image_count = 0,
                video_count = 0,
            )
            return local_history, "", status_text

        selected_profile = registry.profiles[profile_id]
        if (
            active_session.get("profile_id") != profile_id
            or active_session.get("model_name") != model_name
        ):
            active_session = create_session(
                profile_id = profile_id,
                model_name = model_name,
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
            stream = False,
        )
        response = send_chat(
            request = request,
            profile = selected_profile,
            model = model_name,
        )
        assistant_text = response.assistant_text or f"[ERROR] {response.error_message}"
        local_messages.append({"role": "user", "content": user_message})
        local_messages.append({"role": "assistant", "content": assistant_text})
        local_history = messages_to_chatbot_history(
            messages = local_messages,
            chatbot_format = chatbot_format,
        )

        total_tokens = response.usage.get("total_tokens")
        append_message(
            session = active_session,
            role = "assistant",
            content = assistant_text,
            metadata = {"usage": response.usage},
        )
        if save_session_enabled or save_enabled:
            persist_session(session = active_session)

        status_text = build_status_text(
            profile_id = profile_id,
            model_name = model_name,
            image_count = len(image_paths),
            video_count = len(video_paths),
            total_tokens = total_tokens,
        )

        return local_history, "", status_text

    def clear_chat(profile_id: str, model_name: str):
        """Reset active chat session and clear conversation panel.

        Args:
            profile_id: Active profile id from UI.
            model_name: Active model name from UI.
        """
        nonlocal active_session
        active_session = create_session(
            profile_id = profile_id,
            model_name = model_name,
        )
        status_text = build_status_text(
            profile_id = profile_id,
            model_name = model_name,
            image_count = 0,
            video_count = 0,
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
    )

    blocks_kwargs = {
        "title": "llm-lab Gradio UI",
        "css": GRADIO_CSS,
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
                    label = "Profile",
                    choices = profile_options,
                    value = profile.profile_id,
                )
                model_input = gr.Textbox(label = "Model", value = model)
                image_files = gr.Files(
                    label = "Images",
                    file_count = "multiple",
                    type = "filepath",
                )
                video_file = gr.File(label = "Video", type = "filepath")
                temperature = gr.Slider(label = "Temperature", minimum = 0.0, maximum = 2.0, value = 0.7, step = 0.1)
                top_p = gr.Slider(label = "Top-p", minimum = 0.1, maximum = 1.0, value = 1.0, step = 0.1)
                save_enabled = gr.Checkbox(label = "Save session", value = save_session_enabled)

            with gr.Column(scale = 7, elem_classes = ["chat-panel"]):
                status_box = gr.Markdown(status_default, elem_classes = ["status-box"])
                chatbot, chatbot_format = build_chatbot_component(gr = gr)
                user_input = gr.Textbox(
                    label = "Message",
                    placeholder = "Type your prompt and press Enter",
                    lines = 2,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant = "primary")
                    clear_button = gr.Button("Clear Chat")

        profile_dropdown.change(
            fn = update_model,
            inputs = [profile_dropdown],
            outputs = [model_input],
        )
        send_button.click(
            fn = run_turn,
            inputs = [
                user_input,
                chatbot,
                profile_dropdown,
                model_input,
                image_files,
                video_file,
                temperature,
                top_p,
                save_enabled,
            ],
            outputs = [chatbot, user_input, status_box],
        )
        user_input.submit(
            fn = run_turn,
            inputs = [
                user_input,
                chatbot,
                profile_dropdown,
                model_input,
                image_files,
                video_file,
                temperature,
                top_p,
                save_enabled,
            ],
            outputs = [chatbot, user_input, status_box],
        )
        clear_button.click(
            fn = clear_chat,
            inputs = [profile_dropdown, model_input],
            outputs = [chatbot, status_box],
        )

    demo.queue().launch(
        server_name = host,
        server_port = port,
        share = share,
    )
