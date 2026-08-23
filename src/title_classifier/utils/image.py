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


def _resize_and_encode_jpeg(img: np.ndarray, max_size: int, quality: int) -> np.ndarray:
    """缩放到最长边 max_size 并编码为 JPEG buffer"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(
            img,
            (max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2),
            interpolation=cv2.INTER_AREA,
        )
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer


def image_array_to_base64(img_bgr: np.ndarray, max_size: int = 640, quality: int = 75) -> str:
    """BGR ndarray 直接转 base64 JPEG（避免落盘后再读盘）"""
    try:
        buffer = _resize_and_encode_jpeg(img_bgr, max_size, quality)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"图像数组转base64失败: {e}")
        return ""


def image_to_base64(image_path: str, max_size: int = 640) -> str:
    """读取图片并压缩后转base64"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"无法解码图片: {image_path}")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return image_array_to_base64(img, max_size=max_size)
    except Exception as e:
        logger.error(f"图片转base64失败: {e}")
        return ""


def get_image_info(image_path: str) -> dict:
    """获取图片元数据（分辨率）

    Returns:
        {"resolution": str, "width": int, "height": int}
    """
    info = {"resolution": "", "width": 0, "height": 0}
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return info
        h, w = img.shape[:2]
        info["width"] = w
        info["height"] = h
        info["resolution"] = f"{w}x{h}"
    except Exception as e:
        logger.warning(f"获取图片信息失败: {e}")
    return info
