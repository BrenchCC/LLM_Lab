import shlex
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown

from service.chat_service import ChatRequest, send_chat, stream_chat
from service.session_service import append_message, create_session, load_session
from service.session_service import save_session as persist_session
from utils.config_loader import ProfileRegistry, ProviderProfile


@dataclass
class CLIState:
    """Hold mutable CLI runtime state.

    Args:
        profile: Active provider profile.
        model: Active model name.
        stream: Whether stream mode is enabled.
        temperature: Current temperature value.
        top_p: Current top-p value.
        pending_images: Image list that will be attached to next message.
        pending_videos: Video list that will be attached to next message.
    """

    profile: ProviderProfile
    model: str
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0
    pending_images: list[str] = field(default_factory = list)
    pending_videos: list[str] = field(default_factory = list)


COMMAND_SPECS: list[tuple[str, str]] = [
    ("/help", "Show command help"),
    ("/status", "Show current runtime status"),
    ("/profiles", "List available profiles"),
    ("/use <profile_id>", "Switch profile and reset model"),
    ("/model <model_name>", "Set active model"),
    ("/stream on|off", "Toggle stream output"),
    ("/temp <float>", "Set temperature"),
    ("/top_p <float>", "Set top-p"),
    ("/image <path1,path2,...>", "Attach images for next turn only"),
    ("/video <path>", "Attach videos for next turn only"),
    ("/clear", "Clear current session context"),
    ("/save [file_name]", "Save session into storage/conversations"),
    ("/load <file_path>", "Load session file"),
    ("/exit", "Exit the CLI"),
    ("/quit", "Exit the CLI"),
]


def parse_cli_command(line: str) -> tuple[str, list[str]]:
    """Parse one slash command line into command and argument list.

    Args:
        line: Raw input command line.
    """
    tokens = shlex.split(line)
    if not tokens:
        return "", []
    command = tokens[0].lower()
    args = tokens[1:]
    return command, args


def build_status_text(state: CLIState) -> str:
    """Build rich-friendly status text for current runtime state.

    Args:
        state: Mutable CLI state object.
    """
    image_count = len(state.pending_images)
    video_count = len(state.pending_videos)
    return (
        f"[bold]profile[/]: {state.profile.profile_id}    "
        f"[bold]model[/]: {state.model}\n"
        f"[bold]stream[/]: {'on' if state.stream else 'off'}    "
        f"[bold]temperature[/]: {state.temperature:.2f}    "
        f"[bold]top_p[/]: {state.top_p:.2f}\n"
        f"[bold]pending[/]: images={image_count}, videos={video_count}"
    )


def render_banner(console: Console, state: CLIState) -> None:
    """Render startup banner for the CLI.

    Args:
        console: Rich console instance.
        state: Mutable CLI state object.
    """
    banner_text = (
        "[bold cyan]llm-lab CLI[/]\n"
        "[dim]Type /help for commands. Type /exit to quit.[/]"
    )
    console.print(
        Panel(
            banner_text,
            title = "Welcome",
            border_style = "cyan",
            expand = False,
        )
    )
    render_status(console = console, state = state)


def render_help(console: Console) -> None:
    """Render command help table.

    Args:
        console: Rich console instance.
    """
    table = Table(title = "Command Help", header_style = "bold cyan")
    table.add_column("Command", style = "green")
    table.add_column("Description", style = "white")
    for command, description in COMMAND_SPECS:
        table.add_row(command, description)
    console.print(table)


def render_status(console: Console, state: CLIState) -> None:
    """Render current runtime status panel.

    Args:
        console: Rich console instance.
        state: Mutable CLI state object.
    """
    console.print(
        Panel(
            build_status_text(state = state),
            title = "Runtime Status",
            border_style = "blue",
            expand = False,
        )
    )


def normalize_path_list(raw_arg: str) -> list[str]:
    """Normalize comma-separated path values.

    Args:
        raw_arg: Raw command argument string.
    """
    paths = [item.strip() for item in raw_arg.split(",") if item.strip()]
    return paths


def ensure_paths_exist(paths: list[str], media_type: str) -> None:
    """Validate that all provided local paths exist.

    Args:
        paths: List of file paths to validate.
        media_type: Logical media type label for error message.
    """
    missing_paths = [path for path in paths if not Path(path).exists()]
    if missing_paths:
        joined = ", ".join(missing_paths)
        raise FileNotFoundError(f"{media_type} file not found: {joined}")


def render_profiles(
    console: Console,
    registry: ProfileRegistry,
    active_profile_id: str,
) -> None:
    """Render available profiles table.

    Args:
        console: Rich console instance.
        registry: Profile registry object.
        active_profile_id: Currently active profile id.
    """
    table = Table(title = "Profiles", header_style = "bold cyan")
    table.add_column("Active", justify = "center")
    table.add_column("Profile ID")
    table.add_column("Default Model")

    for profile_id in sorted(registry.profiles.keys()):
        marker = "[green]*[/]" if profile_id == active_profile_id else ""
        default_model = registry.profiles[profile_id].default_model
        table.add_row(marker, profile_id, default_model)

    console.print(table)


def render_ok(console: Console, message: str) -> None:
    """Render a success system message.

    Args:
        console: Rich console instance.
        message: Message string.
    """
    console.print(f"[green]{message}[/]")


def render_warn(console: Console, message: str) -> None:
    """Render a warning system message.

    Args:
        console: Rich console instance.
        message: Message string.
    """
    console.print(f"[yellow]{message}[/]")


def render_error(console: Console, message: str) -> None:
    """Render an error panel.

    Args:
        console: Rich console instance.
        message: Error message string.
    """
    console.print(
        Panel(
            f"[red]{message}[/]",
            title = "Error",
            border_style = "red",
            expand = False,
        )
    )


def render_assistant_message(
    console: Console,
    message: str,
    usage: dict[str, int],
) -> None:
    """Render non-stream assistant message panel.

    Args:
        console: Rich console instance.
        message: Assistant message content.
        usage: Token usage payload.
    """
    content = message.strip()
    body = Markdown(content) if content else "(empty response)"
    console.print(
        Panel(
            body,
            title = "Assistant",
            border_style = "green",
            expand = True,
        )
    )
    if usage:
        token_text = (
            f"[dim]tokens: prompt={usage.get('prompt_tokens', 0)}, "
            f"completion={usage.get('completion_tokens', 0)}, "
            f"total={usage.get('total_tokens', 0)}[/]"
        )
        console.print(token_text)


def run_chat_turn(
    console: Console,
    state: CLIState,
    user_input: str,
) -> tuple[str, dict[str, int], str | None]:
    """Run one chat turn and return assistant text with usage.

    Args:
        console: Rich console instance.
        state: Mutable CLI state object.
        user_input: User input text.
    """
    request = ChatRequest(
        user_text = user_input,
        image_paths = list(state.pending_images),
        video_paths = list(state.pending_videos),
        stream = state.stream,
        temperature = state.temperature,
        top_p = state.top_p,
    )

    if state.stream:
        chunks: list[str] = []
        try:
            console.print("[bold green]assistant >[/] ", end = "")
            for chunk in stream_chat(
                request = request,
                profile = state.profile,
                model = state.model,
            ):
                console.print(chunk, end = "", markup = False, soft_wrap = True)
                chunks.append(chunk)
            console.print("")
            return "".join(chunks), {}, None
        except Exception as exc:
            console.print("")
            return "", {}, str(exc)

    response = send_chat(
        request = request,
        profile = state.profile,
        model = state.model,
    )
    return response.assistant_text, response.usage, response.error_message


def run_cli(
    registry: ProfileRegistry,
    initial_profile: ProviderProfile,
    initial_model: str,
    initial_stream: bool = False,
    save_session_enabled: bool = False,
) -> None:
    """Start interactive CLI chat loop.

    Args:
        registry: Loaded profile registry.
        initial_profile: Profile selected during startup.
        initial_model: Model selected during startup.
        initial_stream: Whether to start with stream mode enabled.
        save_session_enabled: Whether to auto-save session after each turn.
    """
    console = Console()

    state = CLIState(
        profile = initial_profile,
        model = initial_model,
        stream = initial_stream,
    )
    session = create_session(
        profile_id = state.profile.profile_id,
        model_name = state.model,
    )

    render_banner(console = console, state = state)
    while True:
        try:
            user_input = console.input("[bold cyan]you > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("")
            user_input = "/exit"

        if not user_input:
            continue

        if user_input.startswith("/"):
            command, args = parse_cli_command(line = user_input)

            if command == "/help":
                render_help(console = console)
                continue

            if command == "/status":
                render_status(console = console, state = state)
                continue

            if command == "/profiles":
                render_profiles(
                    console = console,
                    registry = registry,
                    active_profile_id = state.profile.profile_id,
                )
                continue

            if command == "/use":
                if not args:
                    render_warn(console = console, message = "Usage: /use <profile_id>")
                    continue
                profile_id = args[0]
                if profile_id not in registry.profiles:
                    render_error(console = console, message = f"Profile not found: {profile_id}")
                    continue
                state.profile = registry.profiles[profile_id]
                state.model = state.profile.default_model
                session = create_session(
                    profile_id = state.profile.profile_id,
                    model_name = state.model,
                )
                render_ok(
                    console = console,
                    message = f"Switched profile to {profile_id}, model reset to {state.model}",
                )
                render_status(console = console, state = state)
                continue

            if command == "/model":
                if not args:
                    render_warn(console = console, message = "Usage: /model <model_name>")
                    continue
                state.model = args[0]
                session["model_name"] = state.model
                render_ok(console = console, message = f"Active model: {state.model}")
                continue

            if command == "/stream":
                if not args or args[0] not in {"on", "off"}:
                    render_warn(console = console, message = "Usage: /stream on|off")
                    continue
                state.stream = args[0] == "on"
                render_ok(
                    console = console,
                    message = f"Stream mode: {'on' if state.stream else 'off'}",
                )
                continue

            if command == "/temp":
                if not args:
                    render_warn(console = console, message = "Usage: /temp <float>")
                    continue
                try:
                    state.temperature = float(args[0])
                    render_ok(console = console, message = f"Temperature: {state.temperature}")
                except ValueError:
                    render_error(console = console, message = "Temperature must be a float number.")
                continue

            if command == "/top_p":
                if not args:
                    render_warn(console = console, message = "Usage: /top_p <float>")
                    continue
                try:
                    state.top_p = float(args[0])
                    render_ok(console = console, message = f"Top-p: {state.top_p}")
                except ValueError:
                    render_error(console = console, message = "Top-p must be a float number.")
                continue

            if command == "/image":
                if not args:
                    render_warn(console = console, message = "Usage: /image <path1,path2,...>")
                    continue
                image_paths = normalize_path_list(raw_arg = args[0])
                try:
                    ensure_paths_exist(paths = image_paths, media_type = "Image")
                    state.pending_images = image_paths
                    render_ok(
                        console = console,
                        message = f"Attached images: {', '.join(image_paths)}",
                    )
                    render_status(console = console, state = state)
                except FileNotFoundError as exc:
                    render_error(console = console, message = str(exc))
                continue

            if command == "/video":
                if not args:
                    render_warn(console = console, message = "Usage: /video <path>")
                    continue
                video_paths = normalize_path_list(raw_arg = args[0])
                try:
                    ensure_paths_exist(paths = video_paths, media_type = "Video")
                    state.pending_videos = video_paths
                    render_ok(
                        console = console,
                        message = f"Attached videos: {', '.join(video_paths)}",
                    )
                    render_status(console = console, state = state)
                except FileNotFoundError as exc:
                    render_error(console = console, message = str(exc))
                continue

            if command == "/clear":
                session = create_session(
                    profile_id = state.profile.profile_id,
                    model_name = state.model,
                )
                state.pending_images = []
                state.pending_videos = []
                render_ok(console = console, message = "Session cleared.")
                continue

            if command == "/save":
                file_name = args[0] if args else None
                saved_path = persist_session(session = session, file_name = file_name)
                render_ok(console = console, message = f"Session saved to {saved_path}")
                continue

            if command == "/load":
                if not args:
                    render_warn(console = console, message = "Usage: /load <file_path>")
                    continue
                file_path = args[0]
                try:
                    loaded = load_session(file_path = file_path)
                    session = loaded
                    render_ok(console = console, message = f"Session loaded from {file_path}")
                except Exception as exc:
                    render_error(console = console, message = f"Load failed: {exc}")
                continue

            if command in {"/exit", "/quit"}:
                if save_session_enabled:
                    saved_path = persist_session(session = session)
                    render_ok(console = console, message = f"Auto-saved session to {saved_path}")
                console.print("[bold cyan]Bye.[/]")
                break

            render_warn(
                console = console,
                message = f"Unknown command: {command}. Type /help for command list.",
            )
            continue

        append_message(
            session = session,
            role = "user",
            content = user_input,
            metadata = {
                "images": list(state.pending_images),
                "videos": list(state.pending_videos),
            },
        )
        assistant_text, usage, error_message = run_chat_turn(
            console = console,
            state = state,
            user_input = user_input,
        )
        if error_message:
            output_text = f"[ERROR] {error_message}"
            render_error(console = console, message = output_text)
            append_message(
                session = session,
                role = "assistant",
                content = output_text,
                metadata = {},
            )
        else:
            if not state.stream:
                render_assistant_message(
                    console = console,
                    message = assistant_text,
                    usage = usage,
                )
            append_message(
                session = session,
                role = "assistant",
                content = assistant_text,
                metadata = {"usage": usage},
            )
            if save_session_enabled:
                persist_session(session = session)

        state.pending_images = []
        state.pending_videos = []
