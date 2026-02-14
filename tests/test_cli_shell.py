from app.cli_runner import normalize_path_list, parse_cli_command


def test_parse_cli_command_with_quotes() -> None:
    """Verify command parser handles quoted arguments.

    Args:
        None: This function does not accept parameters.
    """
    command, args = parse_cli_command('/model "gpt-4o-mini"')
    assert command == "/model"
    assert args == ["gpt-4o-mini"]


def test_normalize_path_list() -> None:
    """Verify comma-separated path parsing trims whitespace.

    Args:
        None: This function does not accept parameters.
    """
    paths = normalize_path_list("a.png, b.jpg ,c.webp")
    assert paths == ["a.png", "b.jpg", "c.webp"]

