import subprocess
import sys
from pathlib import Path

from utils.config_loader import ProfileRegistry, ProviderProfile


def run_streamlit(
    env_path: str,
    profiles_path: str,
    profile: ProviderProfile,
    model: str,
    host: str,
    port: int | None,
    save_session: bool,
) -> None:
    """Launch Streamlit app in a subprocess.

    Args:
        env_path: Path to .env file.
        profiles_path: Path to profile YAML.
        profile: Active provider profile.
        model: Active model name.
        host: Streamlit host address.
        port: Optional Streamlit port.
        save_session: Whether session persistence is enabled by default.
    """
    script_path = Path(__file__).resolve().with_name("web_streamlit_app.py")
    streamlit_port = port if port is not None else 8501

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.address",
        host,
        "--server.port",
        str(streamlit_port),
        "--",
        "--env-path",
        env_path,
        "--profiles-path",
        profiles_path,
        "--profile",
        profile.profile_id,
        "--model",
        model,
    ]
    if save_session:
        command.append("--save-session")

    subprocess.run(command, check = True)


def run_gradio(
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
    host: str,
    port: int | None,
    share: bool,
    save_session: bool,
) -> None:
    """Launch Gradio app in-process.

    Args:
        registry: Loaded profile registry.
        profile: Active provider profile.
        model: Active model name.
        host: Gradio host address.
        port: Optional Gradio port.
        share: Whether to enable Gradio share link.
        save_session: Whether session persistence is enabled by default.
    """
    from app.web_gradio_app import run_gradio_app

    gradio_port = port if port is not None else 7860
    run_gradio_app(
        registry = registry,
        profile = profile,
        model = model,
        host = host,
        port = gradio_port,
        share = share,
        save_session_enabled = save_session,
    )


def run_web(
    ui: str,
    registry: ProfileRegistry,
    profile: ProviderProfile,
    model: str,
    env_path: str,
    profiles_path: str,
    host: str,
    port: int | None,
    share: bool,
    save_session: bool,
) -> None:
    """Run selected web UI implementation.

    Args:
        ui: UI type, streamlit or gradio.
        registry: Loaded profile registry.
        profile: Active provider profile.
        model: Active model name.
        env_path: Path to .env file.
        profiles_path: Path to profile YAML.
        host: Server host address.
        port: Optional server port.
        share: Whether external share is enabled.
        save_session: Whether session persistence is enabled by default.
    """
    if ui == "streamlit":
        run_streamlit(
            env_path = env_path,
            profiles_path = profiles_path,
            profile = profile,
            model = model,
            host = host,
            port = port,
            save_session = save_session,
        )
        return

    if ui == "gradio":
        run_gradio(
            registry = registry,
            profile = profile,
            model = model,
            host = host,
            port = port,
            share = share,
            save_session = save_session,
        )
        return

    raise ValueError(f"Unsupported ui type: {ui}")
