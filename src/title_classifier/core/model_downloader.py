"""模型下载器 - 纯逻辑，可被 CLI/scripts/GUI 共用"""

import logging
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .model_registry import (
    YOLO_DIR, CLIP_DIR, CLIP_MODEL_OPTIONS, YOLO_DOWNLOAD_BASE,
)

logger = logging.getLogger(__name__)


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）"""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def get_dir_size_mb(path: Path) -> float:
    """获取目录大小（MB）"""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def get_model_status_yolo(model_name: str) -> dict:
    """检查 YOLO 模型下载状态"""
    path = YOLO_DIR / model_name
    return {
        "exists": path.exists(),
        "path": path,
        "size_mb": get_file_size_mb(path),
    }


def get_model_status_clip(model_name: str) -> dict:
    """检查 CLIP 模型下载状态

    Args:
        model_name: HF 仓库全名，如 "CLIP-ViT-B-16-laion2B-s34B-b88K"
    """
    clip_path = CLIP_DIR / f"models--laion--{model_name}"
    size = get_dir_size_mb(clip_path)
    return {
        "exists": clip_path.exists() and size > 100,
        "path": clip_path,
        "size_mb": size,
    }


def get_downloaded_size() -> float:
    """统计已下载模型总大小（MB）"""
    total = 0.0
    if YOLO_DIR.exists():
        total += get_dir_size_mb(YOLO_DIR)
    if CLIP_DIR.exists():
        total += get_dir_size_mb(CLIP_DIR)
    return total


def download_yolo_model(
    model_name: str,
    progress_cb: Optional[Callable] = None,
) -> bool:
    """下载 YOLO 模型

    Args:
        model_name: 模型文件名，如 "yolov8n.pt"
        progress_cb: 进度回调 count, block_size, total_size -> None

    Returns:
        True if successful
    """
    dest = YOLO_DIR / model_name
    if dest.exists():
        logger.info(f"YOLO 模型已存在: {dest}")
        return True

    url = YOLO_DOWNLOAD_BASE + model_name
    return _download_file(url, dest, progress_cb)


def download_clip_model(
    model_name: str,
    progress_cb: Optional[Callable] = None,
) -> bool:
    """下载 CLIP 模型（通过 huggingface_hub）

    Args:
        model_name: 模型名，如 "CLIP-ViT-B-16"
        progress_cb: 进度回调（当前不支持细粒度进度）

    Returns:
        True if successful
    """
    clip_path = CLIP_DIR / f"models--laion--{model_name}"
    if clip_path.exists() and get_dir_size_mb(clip_path) > 100:
        logger.info(f"CLIP 模型已存在: {clip_path}")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("需要安装 huggingface_hub: pip install huggingface_hub")
        return False

    try:
        repo_id = f"laion/{model_name}"
        logger.info(f"下载 CLIP 模型: {repo_id}")
        if progress_cb:
            progress_cb("downloading", 0, 0)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(clip_path),
            local_dir_use_symlinks=False,
        )
        if progress_cb:
            progress_cb("done", 0, 0)
        return True
    except Exception as e:
        logger.error(f"CLIP 模型下载失败: {e}")
        if progress_cb:
            progress_cb("error", 0, 0)
        return False


def _download_file(url: str, dest: Path, progress_cb=None) -> bool:
    """下载文件到指定路径"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest), reporthook=progress_cb)
        logger.info(f"下载完成: {dest}")
        return True
    except Exception as e:
        logger.error(f"下载失败 {url}: {e}")
        return False
