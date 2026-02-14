import base64
import mimetypes
import tempfile
from pathlib import Path


def guess_image_mime(image_path: str) -> str:
    """Guess MIME type for an image path.

    Args:
        image_path: Local image file path.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        return "image/jpeg"
    return mime_type


def encode_image_to_data_url(image_path: str) -> str:
    """Encode local image into data URL format.

    Args:
        image_path: Local image file path.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type = guess_image_mime(image_path = image_path)
    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def resize_frame(frame, resize_max: int, cv2_module):
    """Resize a frame while preserving its aspect ratio.

    Args:
        frame: One video frame object from OpenCV.
        resize_max: Maximum width or height in pixels.
        cv2_module: Imported cv2 module object.
    """
    height, width = frame.shape[:2]
    if max(height, width) <= resize_max:
        return frame

    if width >= height:
        target_width = resize_max
        target_height = int(height * (resize_max / width))
    else:
        target_height = resize_max
        target_width = int(width * (resize_max / height))

    return cv2_module.resize(frame, (target_width, target_height))


def extract_video_frames(
    video_path: str,
    fps: int = 1,
    max_frames: int = 8,
    resize_max: int = 1024,
    output_dir: str | None = None,
) -> list[str]:
    """Extract key frames from a video and save to local images.

    Args:
        video_path: Local video file path.
        fps: Sampling frame rate for extraction.
        max_frames: Maximum number of frames to keep.
        resize_max: Maximum side length for each output frame.
        output_dir: Optional directory to store extracted frames.
    """
    if fps <= 0:
        raise ValueError("fps must be a positive integer.")
    if max_frames <= 0:
        raise ValueError("max_frames must be a positive integer.")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Package `opencv-python-headless` is required for video processing."
        ) from exc

    if output_dir:
        frame_dir = Path(output_dir)
        frame_dir.mkdir(parents = True, exist_ok = True)
    else:
        frame_dir = Path(tempfile.mkdtemp(prefix = "llm_lab_frames_"))

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Failed to read video file: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 0 else 1.0
    step = max(int(round(source_fps / fps)), 1)

    frame_paths: list[str] = []
    frame_index = 0

    while len(frame_paths) < max_frames:
        read_ok, frame = capture.read()
        if not read_ok:
            break

        if frame_index % step == 0:
            resized = resize_frame(
                frame = frame,
                resize_max = resize_max,
                cv2_module = cv2,
            )
            output_path = frame_dir / f"frame_{len(frame_paths):03d}.jpg"
            cv2.imwrite(str(output_path), resized)
            frame_paths.append(str(output_path))

        frame_index += 1

    capture.release()
    if not frame_paths:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frame_paths

