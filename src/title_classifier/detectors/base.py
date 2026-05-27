"""检测器基类"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """检测器基类"""

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._loaded

    @abstractmethod
    def load_model(self) -> bool:
        """加载模型"""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测帧中的人体

        Args:
            frame: BGR格式的图像

        Returns:
            {
                "has_person": bool,
                "persons": List[Dict],
                "max_confidence": float
            }
        """
        pass

    def detect_from_file(self, image_path: str) -> Dict[str, Any]:
        """
        从文件检测人体

        Args:
            image_path: 图像文件路径

        Returns:
            检测结果
        """
        import cv2

        data = np.fromfile(image_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error(f"无法读取图像: {image_path}")
            return {"has_person": False, "persons": [], "max_confidence": 0.0}

        return self.detect(frame)
