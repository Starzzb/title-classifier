"""场景检测模块 - 基于 ffmpeg 的场景切换检测"""

import subprocess
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_scenes(video_path: str, threshold: float = 0.3) -> List[float]:
    """
    使用 ffmpeg scene detect 检测场景切换点。

    Args:
        video_path: 视频文件路径
        threshold: 场景检测敏感度（0-1），越低越灵敏，默认 0.3

    Returns:
        场景切换时间点列表（秒），包含 0 和视频末尾
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "frame=pts_time",
        "-of", "csv=print_key=1",
        "-f", "lavfi",
        f"movie={video_path},select='gt(scene,{threshold})'",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"场景检测失败: {result.stderr[:200]}")
            return [0.0]

        scenes = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    ts = float(parts[1])
                    scenes.append(ts)
                except ValueError:
                    continue

        scenes.sort()
        logger.info(f"场景检测: 找到 {len(scenes)} 个场景切换点 (threshold={threshold})")

        return scenes

    except FileNotFoundError:
        logger.warning("ffprobe 未安装，回退到整段处理")
        return [0.0]
    except subprocess.TimeoutExpired:
        logger.warning("场景检测超时（视频可能过大），回退到整段处理")
        return [0.0]
    except Exception as e:
        logger.warning(f"场景检测异常: {e}，回退到整段处理")
        return [0.0]


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
