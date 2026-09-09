"""场景检测模块 - 基于 OpenCV 的场景切换检测

性能注记（2026-09 实测）：
- 全片解码已达 libav 物理极限（~85x 实时），多线程/多进程并行解码无收益：
  cv2 ffmpeg 后端存在全局锁，多 cap 并发解码为负加速（4线程 16.5s vs 顺序 2.3s）
- 检测耗时占比小，场景模式的端到端大头在逐段分析（见 vision._process_video_by_scenes）
- 本模块的优化点：对超高分辨率帧降采样后再算直方图（4K 收益显著）
"""

import cv2
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 直方图计算的最大边长：超过则先降采样（只影响 >1080p 帧，阈值语义不变）
_HIST_MAX_SIDE = 320


def _frame_hist(frame: np.ndarray) -> np.ndarray:
    """计算帧的 HSV 直方图（超高分辨率帧先降采样，降低 CPU 开销）"""
    h, w = frame.shape[:2]
    if max(h, w) > 1080:
        scale = _HIST_MAX_SIDE / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


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
    last_log_pct = 0

    # 使用 grab() 快速跳过 + 仅对目标帧 retrieve() 解码（~3x 提速）
    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # 只对采样帧做昂贵解码
        ret, frame = cap.retrieve()
        if not ret:
            frame_idx += 1
            continue

        # 进度日志：每处理 10% 的视频量打印一次
        pct = int(frame_idx * 100 / total_frames) if total_frames > 0 else 0
        if pct >= last_log_pct + 10:
            last_log_pct = pct
            logger.info(f"场景检测进度: {frame_idx/fps:.0f}s / {duration:.0f}s ({pct}%)")

        hist = _frame_hist(frame)

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


def get_segments(video_path: str, duration: float, threshold: float = 0.3, max_scenes: int = 10,
                 sample_interval: float = 0.5) -> List[Tuple[float, float]]:
    """
    一站式获取场景分段。

    Returns:
        [(start, end), ...]
    """
    scene_points = detect_scenes(video_path, threshold, sample_interval)
    return build_segments(scene_points, duration, max_scenes)


def compute_frame_change_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    低成本计算相邻两帧画面的综合变化分数。
    综合三个层面的变化：
    1. HSV 直方图差异（55%）：捕获色彩和硬切变动；
    2. 灰度缩小图差值（30%）：捕获构图、镜头推进和主体运动；
    3. 边缘结构差异（15%）：捕获细节轮廓形态变化。
    
    Returns:
        float: 归一化的变化分数（0.0 到 1.0）
    """
    if frame1 is None or frame2 is None:
        return 0.0
    
    try:
        # 1. HSV 直方图差（主信号）
        hist1 = _frame_hist(frame1)
        hist2 = _frame_hist(frame2)
        hsv_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        
        # 2. 灰度结构差 (缩微图的平均绝对差)
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        g1_mini = cv2.resize(g1, (160, 90), interpolation=cv2.INTER_AREA)
        g2_mini = cv2.resize(g2, (160, 90), interpolation=cv2.INTER_AREA)
        # 高斯平滑
        g1_mini = cv2.GaussianBlur(g1_mini, (5, 5), 0)
        g2_mini = cv2.GaussianBlur(g2_mini, (5, 5), 0)
        
        abs_diff = cv2.absdiff(g1_mini, g2_mini)
        mean_gray_diff = float(np.mean(abs_diff)) / 255.0
        
        # 3. 边缘结构差
        edge1 = cv2.Canny(g1_mini, 50, 150)
        edge2 = cv2.Canny(g2_mini, 50, 150)
        edge_diff = float(np.mean(cv2.absdiff(edge1, edge2))) / 255.0
        
        # 4. 加权合并
        score = 0.55 * hsv_diff + 0.30 * mean_gray_diff + 0.15 * edge_diff
        return min(1.0, max(0.0, score))
    except Exception as e:
        logger.warning(f"计算画面变化分数失败: {e}")
        return 0.0

