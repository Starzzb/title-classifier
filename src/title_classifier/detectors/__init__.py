"""检测器模块"""

from .base import BaseDetector
from .uhd import UHDDetector
from .yolo import YOLODetector
from .clip import CLIPClassifier

__all__ = ["BaseDetector", "UHDDetector", "YOLODetector", "CLIPClassifier"]
