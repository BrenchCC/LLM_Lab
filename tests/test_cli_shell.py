from app.cli_runner import build_conversation_history_from_session, normalize_path_list, parse_cli_command


def test_parse_cli_command_with_quotes() -> None:
    """Verify command parser handles quoted arguments.

    Args:
        None: This function does not accept parameters.
    """
    command, args = parse_cli_command('/model "gpt-4o-mini"')
    assert command == "/model"
    assert args == ["gpt-4o-mini"]


def test_parse_cli_command_for_think_toggle() -> None:
    """Verify think command parser output.

    Args:
        None: This function does not accept parameters.
    """
    command, args = parse_cli_command("/think on")
    assert command == "/think"
    assert args == ["on"]


def test_normalize_path_list() -> None:
    """Verify comma-separated path parsing trims whitespace.

    Args:
        None: This function does not accept parameters.
    """
    paths = normalize_path_list("a.png, b.jpg ,c.webp")
    assert paths == ["a.png", "b.jpg", "c.webp"]


def test_build_conversation_history_from_session() -> None:
    """Verify CLI history builder keeps only user/assistant string messages.

    Args:
        None: This function does not accept parameters.
    """
    session = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "skip"},
            {"role": "assistant", "content": {"bad": "type"}},
        ]
    }
    history = build_conversation_history_from_session(session = session)
    assert history == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
