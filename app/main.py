import argparse
import logging

from app.cli_runner import run_cli
from app.web_runner import run_web
from utils.config_loader import load_env_file, load_profiles, resolve_model
from utils.config_loader import resolve_profile, resolve_profiles_path
from utils.logger import setup_logging


logger = logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared command-line options to one parser.

    Args:
        parser: Target argparse parser instance.
    """
    parser.add_argument("--env-path", type = str, default = ".env")
    parser.add_argument("--profiles-path", type = str, default = None)
    parser.add_argument("--profile", type = str, default = None)
    parser.add_argument("--model", type = str, default = None)
    parser.add_argument("--log-level", type = str, default = "INFO")
    parser.add_argument("--save-session", action = "store_true")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for llm-lab entrypoint.

    Args:
        None: This function does not accept parameters.
    """
    parser = argparse.ArgumentParser(
        prog = "llm-lab",
        description = "Single-entry LLM chat testing tool.",
    )
    subparsers = parser.add_subparsers(dest = "command", required = True)

    chat_parser = subparsers.add_parser("chat", help = "Run interactive CLI mode.")
    add_common_arguments(parser = chat_parser)
    chat_parser.add_argument("--stream", action = "store_true")

    web_parser = subparsers.add_parser("web", help = "Run web UI mode.")
    add_common_arguments(parser = web_parser)
    web_parser.add_argument(
        "--ui",
        type = str,
        choices = ["streamlit", "gradio"],
        required = True,
    )
    web_parser.add_argument("--host", type = str, default = "127.0.0.1")
    web_parser.add_argument("--port", type = int, default = None)
    web_parser.add_argument("--share", action = "store_true")

    return parser.parse_args()


def main() -> int:
    """Run llm-lab main entrypoint and dispatch subcommands.

    Args:
        None: This function does not accept parameters.
    """
    args = parse_args()
    setup_logging(level_name = args.log_level)

    try:
        load_env_file(env_path = args.env_path)
        profiles_path = resolve_profiles_path(cli_profiles_path = args.profiles_path)
        registry = load_profiles(profiles_path = profiles_path)
        profile = resolve_profile(registry = registry, cli_profile = args.profile)
        model = resolve_model(profile = profile, cli_model = args.model)
    except Exception as exc:
        logger.error("Startup configuration failed: %s", exc)
        return 1

    if args.command == "chat":
        run_cli(
            registry = registry,
            initial_profile = profile,
            initial_model = model,
            initial_stream = args.stream,
            save_session_enabled = args.save_session,
        )
        return 0

    if args.command == "web":
        try:
            run_web(
                ui = args.ui,
                registry = registry,
                profile = profile,
                model = model,
                env_path = args.env_path,
                profiles_path = profiles_path,
                host = args.host,
                port = args.port,
                share = args.share,
                save_session = args.save_session,
            )
            return 0
        except Exception as exc:
            logger.error("Web mode failed: %s", exc)
            return 1

    logger.error("Unsupported command: %s", args.command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

