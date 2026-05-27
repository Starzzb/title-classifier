"""视频处理工具"""

import subprocess
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
    return 0.0


def safe_timestamp(timestamp_seconds: float, duration: float, margin: float = 2.0) -> float:
    """安全的时间戳：确保不超过视频时长"""
    if duration <= 0:
        return timestamp_seconds
    max_safe = max(0, duration - margin)
    if max_safe <= 0:
        return 0.0
    return min(timestamp_seconds, max_safe)


def extract_frame(
    video_path: str,
    output_path: str,
    timestamp: str = None,
    max_size: int = 800,
) -> bool:
    """提取视频帧并压缩"""
    try:
        duration = get_video_duration(video_path)

        if timestamp is None:
            ts_seconds = duration / 4 if duration > 0 else 30.0
        else:
            parts = timestamp.split(":")
            if len(parts) == 3:
                ts_seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                ts_seconds = float(parts[0]) * 60 + float(parts[1])
            else:
                ts_seconds = float(timestamp)

        safe_ts = safe_timestamp(ts_seconds, duration)

        hours = int(safe_ts // 3600)
        minutes = int((safe_ts % 3600) // 60)
        seconds = safe_ts % 60
        safe_timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", safe_timestamp_str, "-i", video_path,
             "-vf", f"scale='if(gt(iw,{max_size}),{max_size},-2)':'if(gt(ih,{max_size}),{max_size},-2)'",
             "-frames:v", "1", "-q:v", "2", output_path],
            capture_output=True, timeout=15, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


def is_solid_color_frame(image_path: str, threshold: float = 15.0) -> bool:
    """检测图片是否为纯色"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return True

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        return std_dev < threshold
    except Exception:
        return True


def filter_solid_frames(frame_paths: list, threshold: float = 15.0) -> list:
    """过滤掉纯色帧"""
    valid_frames = []
    for fp in frame_paths:
        if Path(fp).exists() and not is_solid_color_frame(fp, threshold):
            valid_frames.append(fp)
    return valid_frames


def extract_multiple_frames(
    video_path: str,
    output_dir: str,
    n_frames: int = 5,
    max_size: int = 800,
    skip_start_end: bool = True,
) -> List[str]:
    """从视频中提取多个均匀分布的帧"""
    import hashlib

    duration = get_video_duration(video_path)
    if duration <= 0:
        logger.error(f"无法获取视频时长: {video_path}")
        return []

    if skip_start_end and duration > 10:
        start_offset = min(3.0, duration * 0.05)
        end_offset = min(3.0, duration * 0.05)
        effective_duration = duration - start_offset - end_offset
        timestamps = [start_offset + effective_duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]
    else:
        timestamps = [duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]

    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    frame_paths = []

    for i, ts in enumerate(timestamps):
        frame_path = Path(output_dir) / f"{video_hash}_frame_{i}_{ts:.1f}.jpg"
        if extract_frame(video_path, str(frame_path), timestamp=str(ts), max_size=max_size):
            frame_paths.append(str(frame_path))

    valid_frames = filter_solid_frames(frame_paths)

    logger.info(f"成功提取 {len(frame_paths)} 帧，有效帧 {len(valid_frames)} 帧")
    return valid_frames


def detect_keyframes(
    video_path: str,
    output_dir: str,
    max_frames: int = 8,
    threshold: float = 30.0,
    max_size: int = 800,
) -> List[str]:
    """基于帧差异的关键帧检测"""
    import hashlib

    duration = get_video_duration(video_path)
    if duration <= 0:
        return []

    start_offset = min(3.0, duration * 0.05)
    end_offset = min(3.0, duration * 0.05)
    effective_duration = duration - start_offset - end_offset

    n_samples = min(20, int(effective_duration / 2))
    if n_samples < 4:
        n_samples = 4

    timestamps = [start_offset + effective_duration * (i + 1) / (n_samples + 1) for i in range(n_samples)]

    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    tmp_dir = Path(output_dir) / f"_keyframe_tmp_{video_hash}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frames_data = []
    for i, ts in enumerate(timestamps):
        frame_path = str(tmp_dir / f"sample_{i}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=400):
            try:
                if is_solid_color_frame(frame_path):
                    continue

                img = cv2.imread(frame_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames_data.append({
                        "path": frame_path,
                        "timestamp": ts,
                        "gray": gray,
                        "index": i,
                    })
            except Exception:
                pass

    if len(frames_data) < 2:
        logger.warning("采样帧不足，跳过关键帧检测")
        return []

    keyframe_indices = [0]
    prev_gray = frames_data[0]["gray"]

    for i in range(1, len(frames_data)):
        curr_gray = frames_data[i]["gray"]
        diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))

        if diff > threshold:
            keyframe_indices.append(i)
            prev_gray = curr_gray

        if len(keyframe_indices) >= max_frames:
            break

    if len(keyframe_indices) < 3:
        additional = [i for i in range(len(frames_data)) if i not in keyframe_indices]
        step = max(1, len(additional) // (3 - len(keyframe_indices)))
        for i in range(0, len(additional), step):
            if len(keyframe_indices) >= 3:
                break
            keyframe_indices.append(additional[i])

    keyframe_indices.sort()

    keyframe_paths = []
    for idx in keyframe_indices:
        ts = frames_data[idx]["timestamp"]
        frame_path = str(Path(output_dir) / f"{video_hash}_keyframe_{idx}_{ts:.1f}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=max_size):
            keyframe_paths.append(frame_path)

    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    logger.info(f"检测到 {len(keyframe_paths)} 个关键帧")
    return keyframe_paths
