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


def image_to_base64(image_path: str, max_size: int = 640, quality: int = 75) -> str:
    """读取图片并压缩后转base64"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"无法解码图片: {image_path}")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return image_array_to_base64(img, max_size=max_size, quality=quality)
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


def stitch_images(
    images: list,
    grid_shape: tuple = (2, 2),
    sub_max_size: int = 400,
    draw_labels: bool = True,
    timestamps: list = None,
    frame_ids: list = None,
) -> list:
    """将多张图片（路径或 numpy 数组）有条件地拼接成网格大图，适合 VLM 批量输入"""
    if not images:
        return []

    # 1. 解析 grid_shape
    if isinstance(grid_shape, str):
        try:
            r, c = map(int, grid_shape.lower().split("x"))
            grid_shape = (r, c)
        except Exception:
            grid_shape = (2, 2)
    rows, cols = grid_shape
    grid_capacity = rows * cols

    if grid_capacity <= 1:
        return images

    # 2. 读取并解码所有图片为 BGR numpy array
    decoded_imgs = []
    valid_ts = []
    valid_fids = []
    for idx, item in enumerate(images):
        ts = timestamps[idx] if (timestamps and idx < len(timestamps)) else None
        fid = frame_ids[idx] if (frame_ids and idx < len(frame_ids)) else f"F{idx+1:04d}"
        if isinstance(item, np.ndarray):
            decoded_imgs.append(item.copy())
            valid_ts.append(ts)
            valid_fids.append(fid)
        else:
            try:
                data = np.fromfile(str(item), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded_imgs.append(img)
                    valid_ts.append(ts)
                    valid_fids.append(fid)
            except Exception as e:
                logger.warning(f"读取图片失败 {item}: {e}")

    if not decoded_imgs:
        return []

    # 3. 确定子图的统一大小（基于第一张有效子图的比例缩放）
    first_img = decoded_imgs[0]
    h, w = first_img.shape[:2]
    scale = sub_max_size / max(h, w)
    sub_w = int(w * scale)
    sub_h = int(h * scale)
    sub_w = max(2, sub_w // 2 * 2)
    sub_h = max(2, sub_h // 2 * 2)

    # 4. 统一缩放并叠加标签
    processed_imgs = []
    for idx, img in enumerate(decoded_imgs):
        img_resized = cv2.resize(img, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        
        if draw_labels:
            ts = valid_ts[idx]
            fid = valid_fids[idx]
            text = fid
            if ts is not None:
                text += f" ({ts:.1f}s)"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 绘制半透明黑色背景
            sub_overlay = img_resized.copy()
            cv2.rectangle(sub_overlay, (2, 2), (tw + 12, th + 8), (0, 0, 0), -1)
            cv2.addWeighted(sub_overlay, 0.65, img_resized, 0.35, 0, img_resized)
            # 绘制白字
            cv2.putText(img_resized, text, (7, th + 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
        processed_imgs.append(img_resized)

    # 5. 分组拼接
    stitched_list = []
    for chunk_idx in range(0, len(processed_imgs), grid_capacity):
        chunk = processed_imgs[chunk_idx : chunk_idx + grid_capacity]
        
        # 创建空白画布
        canvas = np.zeros((rows * sub_h, cols * sub_w, 3), dtype=np.uint8)
        
        for i, img in enumerate(chunk):
            r_idx = i // cols
            c_idx = i % cols
            y_start = r_idx * sub_h
            y_end = y_start + sub_h
            x_start = c_idx * sub_w
            x_end = x_start + sub_w
            canvas[y_start:y_end, x_start:x_end] = img

        # 绘制分割灰色线条
        for c in range(1, cols):
            cv2.line(canvas, (c * sub_w, 0), (c * sub_w, rows * sub_h), (128, 128, 128), 1)
        for r in range(1, rows):
            cv2.line(canvas, (0, r * sub_h), (cols * sub_w, r * sub_h), (128, 128, 128), 1)

        stitched_list.append(canvas)

    return stitched_list

