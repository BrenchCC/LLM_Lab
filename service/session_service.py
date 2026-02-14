import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def create_session(profile_id: str, model_name: str) -> dict[str, Any]:
    """Create a new chat session payload.

    Args:
        profile_id: Active profile identifier.
        model_name: Active model name.
    """
    session = {
        "session_id": str(uuid4()),
        "created_at": datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        "profile_id": profile_id,
        "model_name": model_name,
        "messages": [],
    }
    return session


def append_message(
    session: dict[str, Any],
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one message record into session payload.

    Args:
        session: Session payload dictionary.
        role: Message role, such as user or assistant.
        content: Message content text.
        metadata: Optional extra metadata dictionary.
    """
    message = {
        "timestamp": datetime.utcnow().isoformat(timespec = "seconds") + "Z",
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    session["messages"].append(message)


def clear_messages(session: dict[str, Any]) -> None:
    """Clear messages list in one session payload.

    Args:
        session: Session payload dictionary.
    """
    session["messages"] = []


def save_session(
    session: dict[str, Any],
    output_dir: str = "storage/conversations",
    file_name: str | None = None,
) -> str:
    """Persist session payload to local JSON file.

    Args:
        session: Session payload dictionary.
        output_dir: Directory path for session files.
        file_name: Optional JSON file name.
    """
    directory = Path(output_dir)
    directory.mkdir(parents = True, exist_ok = True)

    target_name = file_name or f"{session['session_id']}.json"
    if not target_name.endswith(".json"):
        target_name = f"{target_name}.json"

    target_path = directory / target_name
    with target_path.open("w", encoding = "utf-8") as file:
        json.dump(session, file, ensure_ascii = False, indent = 2)
    return str(target_path)


def load_session(file_path: str) -> dict[str, Any]:
    """Load one session payload from local JSON file.

    Args:
        file_path: Path to a session JSON file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {file_path}")

    with path.open("r", encoding = "utf-8") as file:
        payload = json.load(file)
    if "messages" not in payload:
        raise ValueError("Invalid session file. Required field `messages` is missing.")
    return payload

