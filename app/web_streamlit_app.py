import argparse
import html
import tempfile
from pathlib import Path

from service.chat_service import ChatRequest, send_chat, separate_reasoning_text, stream_chat
from service.session_service import append_message, create_session, save_session
from utils.config_loader import list_profile_models, load_env_file, load_profiles, resolve_model
from utils.config_loader import resolve_profile


STREAMLIT_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --ink: #10344d;
  --muted: #5f7688;
  --stroke: #c8dce9;
  --card: #ffffff;
  --teal: #0f766e;
  --teal-soft: #d8f0ec;
  --orange: #f97316;
  --bg-a: #f6fbff;
  --bg-b: #eef7f4;
}

html, body, [class*="css"] {
  font-family: "Space Grotesk", "Segoe UI", sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 8%, rgba(15,118,110,.12) 0, rgba(15,118,110,0) 32%),
    radial-gradient(circle at 88% 18%, rgba(249,115,22,.12) 0, rgba(249,115,22,0) 30%),
    linear-gradient(155deg, var(--bg-a) 0%, var(--bg-b) 100%);
}

[data-testid="stSidebar"] {
  border-right: 1px solid var(--stroke);
  background: linear-gradient(180deg, #f7fcfb 0%, #edf4fb 100%);
}

.llm-hero {
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 1.1rem 1.2rem 1.2rem 1.2rem;
  background: linear-gradient(140deg, rgba(15,118,110,.12), rgba(255,255,255,.85));
  margin-bottom: 0.8rem;
}

.llm-hero .kicker {
  font-family: "IBM Plex Mono", monospace;
  color: var(--teal);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.llm-hero h1 {
  margin: 0.2rem 0 0.35rem 0;
  color: var(--ink);
  font-size: 1.55rem;
}

.llm-hero p {
  margin: 0;
  color: var(--muted);
}

.stat-pill-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 0.65rem;
}

.stat-pill {
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 999px;
  padding: 0.34rem 0.75rem;
  color: var(--ink);
  font-size: 0.86rem;
}

.stat-pill b {
  color: var(--teal);
}

div[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.76);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 0.2rem 0.2rem;
}

.usage-card {
  border: 1px solid var(--stroke);
  border-radius: 14px;
  background: var(--card);
  padding: 0.65rem 0.85rem;
  margin-top: 0.6rem;
}

.usage-card h4 {
  margin: 0 0 0.35rem 0;
  color: var(--ink);
  font-size: 0.92rem;
}

.usage-card p {
  margin: 0;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.84rem;
}

[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.6);
  border: 1px dashed var(--stroke);
  border-radius: 14px;
  padding: 0.45rem 0.55rem;
}
</style>
"""


def parse_args() -> argparse.Namespace:
    """Parse Streamlit app arguments passed after `--`.

    Args:
        None: This function does not accept parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", type = str, default = ".env")
    parser.add_argument("--profiles-path", type = str, required = True)
    parser.add_argument("--profile", type = str, default = None)
    parser.add_argument("--model", type = str, default = None)
    parser.add_argument("--save-session", action = "store_true")
    args, _ = parser.parse_known_args()
    return args


def apply_streamlit_theme(st) -> None:
    """Inject custom style sheet into Streamlit page.

    Args:
        st: Imported Streamlit module.
    """
    st.markdown(STREAMLIT_THEME, unsafe_allow_html = True)


def render_hero(st) -> None:
    """Render branded hero section at page top.

    Args:
        st: Imported Streamlit module.
    """
    st.markdown(
        """
        <div class="llm-hero">
          <div class="kicker">LLM-LAB</div>
          <h1>LLM-LAB</h1>
          <p>
            Unified model lab for OpenAI-compatible services with a clear workflow
            across text, image, and video prompts.
          </p>
        </div>
        """,
        unsafe_allow_html = True,
    )


def render_runtime_pills(
    st,
    profile_id: str,
    model_name: str,
    stream_mode: bool,
    thinking_enabled: bool,
    temperature: float,
    top_p: float,
) -> None:
    """Render compact runtime status pills.

    Args:
        st: Imported Streamlit module.
        profile_id: Current selected profile id.
        model_name: Current selected model name.
        stream_mode: Whether stream mode is enabled.
        thinking_enabled: Whether deep thinking mode is enabled.
        temperature: Active temperature value.
        top_p: Active top-p value.
    """
    stream_text = "ON" if stream_mode else "OFF"
    thinking_text = "ON" if thinking_enabled else "OFF"
    st.markdown(
        f"""
        <div class="stat-pill-wrap">
          <div class="stat-pill"><b>Profile</b>: {profile_id}</div>
          <div class="stat-pill"><b>Model</b>: {model_name}</div>
          <div class="stat-pill"><b>Stream</b>: {stream_text}</div>
          <div class="stat-pill"><b>Thinking</b>: {thinking_text}</div>
          <div class="stat-pill"><b>Temp</b>: {temperature:.2f}</div>
          <div class="stat-pill"><b>Top-p</b>: {top_p:.2f}</div>
        </div>
        """,
        unsafe_allow_html = True,
    )


def persist_uploaded_file(uploaded_file, temp_dir: Path) -> str:
    """Persist one uploaded file object into temporary directory.

    Args:
        uploaded_file: Uploaded file object from Streamlit.
        temp_dir: Target temporary directory path.
    """
    safe_name = uploaded_file.name.replace("/", "_")
    target_path = temp_dir / safe_name
    target_path.write_bytes(uploaded_file.getvalue())
    return str(target_path)


def ensure_streamlit_state(profile_id: str, model_name: str) -> None:
    """Initialize Streamlit session state fields.

    Args:
        profile_id: Active profile identifier.
        model_name: Active model name.
    """
    import streamlit as st

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_payload" not in st.session_state:
        st.session_state.session_payload = create_session(
            profile_id = profile_id,
            model_name = model_name,
        )
        return

    payload = st.session_state.session_payload
    if payload.get("profile_id") != profile_id or payload.get("model_name") != model_name:
        st.session_state.session_payload = create_session(
            profile_id = profile_id,
            model_name = model_name,
        )
        st.session_state.messages = []
        st.session_state.last_usage = {}


def render_usage_card(st, usage: dict[str, int]) -> None:
    """Render token usage summary card.

    Args:
        st: Imported Streamlit module.
        usage: Token usage dictionary from model response.
    """
    if not usage:
        return

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    st.markdown(
        (
            "<div class='usage-card'>"
            "<h4>Token Usage</h4>"
            f"<p>prompt={prompt_tokens} | completion={completion_tokens} | total={total_tokens}</p>"
            "</div>"
        ),
        unsafe_allow_html = True,
    )


def render_reasoning_block(st, reasoning_text: str) -> None:
    """Render a simple collapsible reasoning block.

    Args:
        st: Imported Streamlit module.
        reasoning_text: Extracted reasoning text from model output.
    """
    content = reasoning_text.strip()
    if not content:
        return

    escaped = html.escape(content)
    st.markdown(
        (
            "<details>"
            "<summary><b>Reasoning</b></summary>"
            f"<div style='margin-top: 8px; white-space: pre-wrap;'>{escaped}</div>"
            "</details>"
        ),
        unsafe_allow_html = True,
    )


def build_conversation_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build normalized chat history payload for model context.

    Args:
        messages: Display message list from Streamlit session state.
    """
    history: list[dict[str, str]] = []
    for message in messages:
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


def build_model_options(
    profile,
    preferred_model: str = "",
) -> list[str]:
    """Build model options for one profile with optional preferred item.

    Args:
        profile: Active provider profile object.
        preferred_model: Optional model name to be pinned in options.
    """
    options = list_profile_models(profile = profile)
    normalized_preferred = preferred_model.strip()

    if normalized_preferred and normalized_preferred not in options:
        options = [normalized_preferred] + options

    if not options and profile.default_model:
        options = [profile.default_model]
    return options


def resolve_selected_model(
    st,
    profile,
    initial_profile_id: str,
    initial_model: str,
) -> str:
    """Resolve effective model from preset and optional custom override.

    Args:
        st: Imported Streamlit module.
        profile: Selected provider profile object.
        initial_profile_id: Profile id from startup arguments.
        initial_model: Model name resolved during startup.
    """
    base_model = initial_model if profile.profile_id == initial_profile_id else profile.default_model
    model_options = build_model_options(
        profile = profile,
        preferred_model = base_model,
    )
    option_state_key = f"streamlit_model_option_{profile.profile_id}"
    custom_state_key = f"streamlit_custom_model_{profile.profile_id}"

    if option_state_key not in st.session_state:
        st.session_state[option_state_key] = model_options[0] if model_options else base_model
    if custom_state_key not in st.session_state:
        st.session_state[custom_state_key] = ""
    if model_options and st.session_state[option_state_key] not in model_options:
        st.session_state[option_state_key] = model_options[0]

    st.selectbox(
        "Model",
        options = model_options,
        key = option_state_key,
    )
    st.text_input(
        "Custom model (optional)",
        key = custom_state_key,
        placeholder = "Leave empty to use selected preset model",
    )

    custom_model = str(st.session_state[custom_state_key]).strip()
    if custom_model:
        return custom_model
    return str(st.session_state[option_state_key]).strip()


def resolve_thinking_enabled(st, profile) -> bool:
    """Resolve current thinking switch value for selected provider.

    Args:
        st: Imported Streamlit module.
        profile: Selected provider profile object.
    """
    state_key = f"streamlit_thinking_{profile.profile_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = bool(profile.enable_deep_thinking)

    st.checkbox(
        "Thinking",
        key = state_key,
        help = "Enable provider deep thinking mode for current provider.",
    )
    return bool(st.session_state[state_key])


def main() -> None:
    """Run Streamlit web app.

    Args:
        None: This function does not accept parameters.
    """
    import streamlit as st

    args = parse_args()
    load_env_file(env_path = args.env_path)
    registry = load_profiles(profiles_path = args.profiles_path)
    initial_profile = resolve_profile(registry = registry, cli_profile = args.profile)
    initial_model = resolve_model(
        profile = initial_profile,
        cli_model = args.model,
        prefer_profile_default = bool(args.profile),
    )

    st.set_page_config(
        page_title = "LLM-LAB",
        layout = "wide",
        initial_sidebar_state = "expanded",
    )
    apply_streamlit_theme(st = st)
    render_hero(st = st)

    profile_options = sorted(registry.profiles.keys())
    default_profile_index = profile_options.index(initial_profile.profile_id)

    with st.sidebar:
        st.markdown("### LLM-LAB Theme Console")
        st.caption("LLM-LAB unified theme for provider/model switching, sampling, and session persistence.")

        selected_profile_id = st.selectbox(
            "Model Provider",
            options = profile_options,
            index = default_profile_index,
        )
        selected_profile = registry.profiles[selected_profile_id]

        selected_model = resolve_selected_model(
            st = st,
            profile = selected_profile,
            initial_profile_id = initial_profile.profile_id,
            initial_model = initial_model,
        )
        thinking_enabled = resolve_thinking_enabled(
            st = st,
            profile = selected_profile,
        )
        stream_mode = st.checkbox("Stream response", value = False)
        save_session_enabled = st.checkbox(
            "Save session",
            value = args.save_session,
        )
        temperature = st.slider("Temperature", min_value = 0.0, max_value = 2.0, value = 0.7, step = 0.1)
        top_p = st.slider("Top-p", min_value = 0.1, max_value = 1.0, value = 1.0, step = 0.1)

    ensure_streamlit_state(
        profile_id = selected_profile_id,
        model_name = selected_model,
    )

    render_runtime_pills(
        st = st,
        profile_id = selected_profile_id,
        model_name = selected_model,
        stream_mode = stream_mode,
        thinking_enabled = thinking_enabled,
        temperature = temperature,
        top_p = top_p,
    )

    st.markdown("### LLM-LAB Attachments")
    upload_col, video_col = st.columns(2, gap = "large")
    with upload_col:
        uploaded_images = st.file_uploader(
            "Upload images",
            type = ["png", "jpg", "jpeg", "webp"],
            accept_multiple_files = True,
        )
    with video_col:
        uploaded_video = st.file_uploader(
            "Upload video",
            type = ["mp4", "mov", "avi", "mkv"],
            accept_multiple_files = False,
        )

    if uploaded_images:
        st.caption(f"Image files attached: {len(uploaded_images)}")
    if uploaded_video:
        st.caption(f"Video attached: {uploaded_video.name}")

    st.markdown("### LLM-LAB Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_reasoning_block(
                    st = st,
                    reasoning_text = str(message.get("reasoning", "")),
                )

    if "last_usage" not in st.session_state:
        st.session_state.last_usage = {}
    render_usage_card(st = st, usage = st.session_state.last_usage)

    prompt = st.chat_input("Type your prompt here, then press Enter...")
    if not prompt:
        return

    temp_dir = Path(tempfile.mkdtemp(prefix = "llm_lab_upload_"))
    image_paths: list[str] = []
    video_paths: list[str] = []

    for uploaded in uploaded_images or []:
        image_paths.append(persist_uploaded_file(uploaded_file = uploaded, temp_dir = temp_dir))
    if uploaded_video:
        video_paths.append(persist_uploaded_file(uploaded_file = uploaded_video, temp_dir = temp_dir))

    conversation_history = build_conversation_history(messages = st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    append_message(
        session = st.session_state.session_payload,
        role = "user",
        content = prompt,
        metadata = {"images": image_paths, "videos": video_paths},
    )

    with st.chat_message("assistant"):
        warning_messages: list[str] = []
        if stream_mode:
            placeholder = st.empty()
            full_text = ""
            reasoning_text = ""
            try:
                request = ChatRequest(
                    user_text = prompt,
                    image_paths = image_paths,
                    video_paths = video_paths,
                    conversation_history = conversation_history,
                    stream = True,
                    temperature = temperature,
                    top_p = top_p,
                )
                for chunk in stream_chat(
                    request = request,
                    profile = selected_profile,
                    model = selected_model,
                    warning_messages = warning_messages,
                    enable_deep_thinking = thinking_enabled,
                ):
                    full_text += chunk
                    placeholder.markdown(full_text)
                assistant_text, reasoning_text = separate_reasoning_text(
                    assistant_text = full_text,
                )
                if assistant_text != full_text:
                    placeholder.markdown(assistant_text if assistant_text else "(empty response)")
                for warning_message in warning_messages:
                    st.warning(warning_message)
                render_reasoning_block(st = st, reasoning_text = reasoning_text)
            except Exception as exc:
                full_text = f"[ERROR] {exc}"
                placeholder.error(full_text)
                assistant_text = full_text
                reasoning_text = ""
            usage = {}
        else:
            request = ChatRequest(
                user_text = prompt,
                image_paths = image_paths,
                video_paths = video_paths,
                conversation_history = conversation_history,
                stream = False,
                temperature = temperature,
                top_p = top_p,
            )
            with st.spinner("Reasoning..."):
                response = send_chat(
                    request = request,
                    profile = selected_profile,
                    model = selected_model,
                    enable_deep_thinking = thinking_enabled,
                )
            assistant_text = response.assistant_text
            if response.error_message:
                assistant_text = f"[ERROR] {response.error_message}"
            reasoning_text = response.reasoning_text
            usage = response.usage
            warning_messages = response.warning_messages
            if response.error_message:
                st.error(assistant_text)
            else:
                for warning_message in warning_messages:
                    st.warning(warning_message)
                st.markdown(assistant_text)
                render_reasoning_block(st = st, reasoning_text = reasoning_text)
                render_usage_card(st = st, usage = usage)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "reasoning": reasoning_text,
        }
    )
    st.session_state.last_usage = usage
    append_message(
        session = st.session_state.session_payload,
        role = "assistant",
        content = assistant_text,
        metadata = {
            "usage": usage,
            "reasoning": reasoning_text,
            "warnings": warning_messages,
        },
    )

    if save_session_enabled:
        save_session(session = st.session_state.session_payload)


if __name__ == "__main__":
    main()
