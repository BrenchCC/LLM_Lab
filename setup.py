from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read_long_description(readme_path: str) -> str:
    """Read project long description from README file.

    Args:
        readme_path: Path to README markdown file.
    """
    return Path(readme_path).read_text(encoding = "utf-8")


setup(
    name = "llm-lab",
    version = "0.1.0",
    description = "Single-entry LLM chat testing tool for OpenAI-compatible APIs.",
    long_description = read_long_description(readme_path = "README.md"),
    long_description_content_type = "text/markdown",
    python_requires = ">=3.10",
    packages = find_packages(
        include = [
            "app",
            "app.*",
            "service",
            "service.*",
            "utils",
            "utils.*",
        ]
    ),
    include_package_data = True,
    install_requires = [
        "gradio>=5.0.0",
        "openai>=1.55.0",
        "pydantic>=2.10.0",
        "rich>=13.9.0",
        "tomli>=2.0.2",
        "PyYAML>=6.0.2",
        "streamlit>=1.40.0",
        "python-dotenv>=1.0.1",
        "opencv-python-headless>=4.10.0.84",
    ],
    extras_require = {
        "dev": [
            "pytest>=8.3.0",
            "pytest-mock>=3.14.0",
        ]
    },
    entry_points = {
        "console_scripts": [
            "llm-lab = app.main:main",
        ]
    },
)
