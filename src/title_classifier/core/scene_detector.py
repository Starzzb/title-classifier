"""场景检测模块 - 基于 OpenCV 的场景切换检测"""

import cv2
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_scenes(video_path: str, threshold: float = 0.3, sample_interval: float = 0.5) -> List[float]:
    """
    使用 OpenCV 直方图差异检测场景切换点。

    按 sample_interval 间隔采样帧，计算 HSV 直方图 Bhattacharyya 距离，
    超过 threshold 时标记为场景切换。

    Args:
        video_path: 视频文件路径
        threshold: 场景检测敏感度（0-1），值越低越灵敏，默认 0.3
        sample_interval: 采样间隔（秒），默认 0.5s

    Returns:
        场景切换时间点列表（秒），包含 0 和视频末尾
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}，回退到整段处理")
        return [0.0]

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0:
        cap.release()
        return [0.0]

    step = max(1, int(fps * sample_interval))
    prev_hist = None
    timestamps = [0.0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                ts = frame_idx / fps
                timestamps.append(ts)

        prev_hist = hist
        frame_idx += 1

    cap.release()

    timestamps.sort()
    logger.info(f"场景检测: 找到 {len(timestamps)} 个场景切换点 (threshold={threshold})")
    return timestamps


def build_segments(scene_points: List[float], duration: float, max_scenes: int = 10) -> List[Tuple[float, float]]:
    """
    将场景切换点构建为 (start, end) 段列表，合并小场景直到 ≤ max_scenes。

    Args:
        scene_points: 场景切换点列表（含 0）
        duration: 视频总时长
        max_scenes: 最大段数上限

    Returns:
        [(start, end), ...] 段列表
    """
    if not scene_points:
        return [(0.0, duration)]

    if scene_points[0] != 0.0:
        scene_points = [0.0] + scene_points
    if scene_points[-1] != duration:
        scene_points.append(duration)

    segments = []
    for i in range(len(scene_points) - 1):
        start = scene_points[i]
        end = scene_points[i + 1]
        if end - start > 0.5:
            segments.append((start, end))

    while len(segments) > max_scenes:
        min_gap = float("inf")
        merge_idx = 0
        for i in range(len(segments) - 1):
            gap = segments[i + 1][1] - segments[i][0]
            if gap < min_gap:
                min_gap = gap
                merge_idx = i

        merged = (segments[merge_idx][0], segments[merge_idx + 1][1])
        segments[merge_idx] = merged
        del segments[merge_idx + 1]
        logger.debug(f"合并相邻场景: 剩余 {len(segments)} 段")

    logger.info(f"场景分段: {len(segments)} 段 (上限 {max_scenes})")
    return segments


def get_segments(video_path: str, duration: float, threshold: float = 0.3, max_scenes: int = 10) -> List[Tuple[float, float]]:
    """
    一站式获取场景分段。

    Returns:
        [(start, end), ...]
    """
    scene_points = detect_scenes(video_path, threshold)
    return build_segments(scene_points, duration, max_scenes)
