"""工具模块"""

from .video import get_video_duration, extract_frame, extract_multiple_frames
from .image import compress_image, image_to_base64
from .stats import TagStatistics

__all__ = [
    "get_video_duration", "extract_frame", "extract_multiple_frames",
    "compress_image", "image_to_base64",
    "TagStatistics"
]
