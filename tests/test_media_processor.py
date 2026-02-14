from pathlib import Path

import pytest

from utils.media_utils import encode_image_to_data_url, extract_video_frames


def test_encode_image_to_data_url(tmp_path: Path) -> None:
    """Verify image file can be encoded into data URL.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    data_url = encode_image_to_data_url(image_path = str(image_path))
    assert data_url.startswith("data:image/")
    assert ";base64," in data_url


def test_extract_video_frames_missing_file(tmp_path: Path) -> None:
    """Verify missing video path raises FileNotFoundError.

    Args:
        tmp_path: Pytest temporary directory fixture path.
    """
    missing_video_path = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        extract_video_frames(video_path = str(missing_video_path))

