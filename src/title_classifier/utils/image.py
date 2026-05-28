"""图片处理工具"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compress_image(input_path: str, output_path: str, max_size: int = 640, quality: int = 75) -> bool:
    """压缩图片，保持宽高比"""
    try:
        data = np.fromfile(input_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return Path(output_path).exists()
    except Exception as e:
        logger.error(f"图片压缩失败: {e}")
        return False


def image_to_base64(image_path: str, max_size: int = 640) -> str:
    """读取图片并压缩后转base64"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"无法解码图片: {image_path}")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        h, w = img.shape[:2]

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        result = base64.b64encode(buffer).decode("utf-8")
        
        logger.debug(f"图片转base64: {image_path}, 原始大小={w}x{h}, base64长度={len(result)}")
        
        return result
    except Exception as e:
        logger.error(f"图片转base64失败: {e}")
        return ""
