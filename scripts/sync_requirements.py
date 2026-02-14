import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for requirements sync script.

    Args:
        None: This function does not accept parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type = str, default = "pyproject.toml")
    parser.add_argument("--output", type = str, default = "requirements.txt")
    parser.add_argument("--include-dev", action = "store_true")
    return parser.parse_args()


def load_toml(path: str) -> dict:
    """Load TOML file with Python-version-compatible parser.

    Args:
        path: Path to TOML file.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise RuntimeError(
                "Python 3.10 requires `tomli`. Install dependencies from requirements.txt first."
            ) from exc

    with Path(path).open("rb") as file:
        return tomllib.load(file)


def collect_dependencies(pyproject_data: dict, include_dev: bool) -> list[str]:
    """Collect dependency lines from pyproject sections.

    Args:
        pyproject_data: Parsed pyproject dictionary.
        include_dev: Whether to include dev dependency group.
    """
    project_section = pyproject_data.get("project", {})
    dependencies = list(project_section.get("dependencies", []))

    if include_dev:
        optional = project_section.get("optional-dependencies", {})
        dev_dependencies = optional.get("dev", [])
        dependencies.extend(dev_dependencies)

    unique_sorted = sorted(set(dependencies), key = lambda item: item.lower())
    return unique_sorted


def write_requirements(dependencies: list[str], output_path: str) -> None:
    """Write dependency lines into requirements.txt file.

    Args:
        dependencies: List of dependency requirement strings.
        output_path: Destination requirements file path.
    """
    target = Path(output_path)
    lines = [f"{dependency}\n" for dependency in dependencies]
    target.write_text("".join(lines), encoding = "utf-8")


def main() -> int:
    """Run sync process from pyproject.toml to requirements.txt.

    Args:
        None: This function does not accept parameters.
    """
    args = parse_args()
    pyproject_data = load_toml(path = args.pyproject)
    dependencies = collect_dependencies(
        pyproject_data = pyproject_data,
        include_dev = args.include_dev,
    )
    write_requirements(
        dependencies = dependencies,
        output_path = args.output,
    )
    print(f"Synced {len(dependencies)} dependencies to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
