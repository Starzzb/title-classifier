"""视频处理工具 - 支持硬件加速和批量帧提取"""

import subprocess
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 硬件解码器配置（按平台）
HW_DECODERS = {
    "win32": ["d3d11va", "dxva2", "qsv"],  # Windows
    "linux": ["vaapi", "qsv"],              # Linux
    "darwin": ["videotoolbox"],              # macOS
}

# 缓存检测到的硬件解码器
_cached_hw_decoder = None
_hw_decoder_checked = False


def detect_hw_accel() -> Optional[str]:
    """
    自动检测可用的硬件解码器
    
    Returns:
        可用的硬件解码器名称，无可用时返回 None
    """
    global _cached_hw_decoder, _hw_decoder_checked
    
    if _hw_decoder_checked:
        return _cached_hw_decoder
    
    _hw_decoder_checked = True
    
    # 获取当前平台的解码器列表
    platform_decoders = HW_DECODERS.get(sys.platform, [])
    if not platform_decoders:
        logger.debug(f"平台 {sys.platform} 无硬件解码器配置")
        return None
    
    # 测试每个解码器
    for decoder in platform_decoders:
        try:
            # 使用一个简单的测试命令检查解码器是否可用
            result = subprocess.run(
                ["ffmpeg", "-hwaccel", decoder, "-f", "lavfi", "-i", 
                 "color=c=black:s=320x240:d=0.1", "-frames:v", "1", "-f", "null", "-"],
                capture_output=True, timeout=5, encoding="utf-8", errors="replace",
            )
            # 检查是否有解码器相关错误
            stderr = result.stderr.lower()
            if "hwaccel" not in stderr or "error" not in stderr:
                _cached_hw_decoder = decoder
                logger.info(f"检测到硬件解码器: {decoder}")
                return decoder
        except (subprocess.TimeoutExpired, Exception):
            continue
    
    logger.info("未检测到可用的硬件解码器，使用软解")
    return None


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    # 方式1: ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe超时，尝试cv2备用方案: {Path(video_path).name}")
    except Exception as e:
        logger.warning(f"ffprobe失败: {e}")

    # 方式2: cv2备用
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
    except Exception as e:
        logger.warning(f"cv2获取时长失败: {e}")

    logger.error(f"获取视频时长失败: {video_path}")
    return 0.0


def get_video_info(video_path: str) -> dict:
    """
    一次性获取视频元数据（时长 + 分辨率）

    Returns:
        {"duration": float, "resolution": str, "width": int, "height": int}
    """
    info = {"duration": 0.0, "resolution": "", "width": 0, "height": 0}

    # 方式1: ffprobe 一次性获取
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration:stream=width,height,codec_type",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)

            # 时长
            fmt = data.get("format", {})
            dur_str = fmt.get("duration", "")
            if dur_str:
                info["duration"] = float(dur_str)

            # 分辨率（取第一个视频流）
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    w = stream.get("width", 0)
                    h = stream.get("height", 0)
                    if w and h:
                        info["width"] = w
                        info["height"] = h
                        info["resolution"] = f"{w}x{h}"
                    break

            return info
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe超时: {Path(video_path).name}")
    except Exception as e:
        logger.warning(f"ffprobe失败: {e}")

    # 方式2: cv2 备用
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if fps > 0 and frame_count > 0:
                info["duration"] = frame_count / fps
            if w and h:
                info["width"] = w
                info["height"] = h
                info["resolution"] = f"{w}x{h}"
    except Exception as e:
        logger.warning(f"cv2获取信息失败: {e}")

    return info


def safe_timestamp(timestamp_seconds: float, duration: float, margin: float = 2.0) -> float:
    """安全的时间戳：确保不超过视频时长"""
    if duration <= 0:
        return timestamp_seconds
    max_safe = max(0, duration - margin)
    if max_safe <= 0:
        return 0.0
    return min(timestamp_seconds, max_safe)


def _parse_timestamp(timestamp: str) -> float:
    """解析时间戳字符串为秒数"""
    parts = timestamp.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    else:
        return float(timestamp)


def extract_frame(
    video_path: str,
    output_path: str,
    timestamp: str = None,
    max_size: int = 800,
    duration: float = None,
    hw_accel: Optional[str] = None,
) -> bool:
    """
    提取视频帧并压缩
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
        timestamp: 时间戳（秒或 HH:MM:SS.mmm 格式）
        max_size: 最大尺寸
        duration: 视频时长（可选，避免重复查询）
        hw_accel: 硬件解码器（None=自动检测）
    
    Returns:
        是否成功
    """
    try:
        if duration is None:
            duration = get_video_duration(video_path)

        if timestamp is None:
            ts_seconds = duration / 4 if duration > 0 else 30.0
        else:
            ts_seconds = _parse_timestamp(timestamp)

        safe_ts = safe_timestamp(ts_seconds, duration)

        hours = int(safe_ts // 3600)
        minutes = int((safe_ts % 3600) // 60)
        seconds = safe_ts % 60
        safe_timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        # 构建 ffmpeg 命令
        cmd = ["ffmpeg", "-y"]
        
        # 添加硬件加速
        if hw_accel:
            cmd.extend(["-hwaccel", hw_accel, "-hwaccel_output_format", "nv12"])
        
        cmd.extend([
            "-ss", safe_timestamp_str,
            "-i", video_path,
            "-vf", f"scale='if(gte(iw,ih),{max_size},-2)':'if(gte(ih,iw),{max_size},-2)'",
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ])

        result = subprocess.run(
            cmd, capture_output=True, timeout=15, encoding="utf-8", errors="replace",
        )
        
        # 硬件加速失败时回退到软解
        if result.returncode != 0 and hw_accel:
            logger.debug(f"硬件解码失败，回退到软解: {result.stderr[:200]}")
            return extract_frame(video_path, output_path, timestamp, max_size, duration, hw_accel=None)
        
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        logger.debug(f"帧提取失败: {e}")
        return False


def extract_frames_batch(
    video_path: str,
    output_dir: str,
    timestamps: List[float],
    max_size: int = 800,
    hw_accel: Optional[str] = None,
    prefix: str = "frame",
) -> List[str]:
    """
    批量提取多个帧 - 单次 ffmpeg 调用
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        timestamps: 时间戳列表（秒）
        max_size: 最大尺寸
        hw_accel: 硬件解码器
        prefix: 输出文件名前缀
    
    Returns:
        成功提取的帧文件路径列表
    """
    if not timestamps:
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建 select 滤镜表达式
    # 使用 between 选择特定时间点的帧
    select_parts = []
    for ts in timestamps:
        # 使用一个小窗口（0.1秒）来匹配时间点
        select_parts.append(f"between(t\\,{ts - 0.05:.3f}\\,{ts + 0.05:.3f})")
    select_expr = "+".join(select_parts)
    
    # 构建 ffmpeg 命令
    cmd = ["ffmpeg", "-y"]
    
    # 添加硬件加速
    if hw_accel:
        cmd.extend(["-hwaccel", hw_accel, "-hwaccel_output_format", "nv12"])
    
    cmd.extend([
        "-i", video_path,
        "-vf", f"select='{select_expr}',scale='if(gte(iw,ih),{max_size},-2)':'if(gte(ih,iw),{max_size},-2)'",
        "-vsync", "vfr",
        "-q:v", "2",
        str(output_path / f"{prefix}_%04d.jpg"),
    ])
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=60, encoding="utf-8", errors="replace",
        )
        
        # 硬件加速失败时回退到软解
        if result.returncode != 0 and hw_accel:
            logger.debug(f"批量硬件解码失败，回退到软解")
            return extract_frames_batch(
                video_path, output_dir, timestamps, max_size, hw_accel=None, prefix=prefix
            )
        
        # 收集生成的帧文件
        frame_files = sorted(output_path.glob(f"{prefix}_*.jpg"))
        frame_paths = [str(f) for f in frame_files]
        
        logger.info(f"批量提取完成: {len(frame_paths)}/{len(timestamps)} 帧")
        return frame_paths
        
    except Exception as e:
        logger.warning(f"批量帧提取失败: {e}")
        # 回退到逐帧提取
        return _extract_frames_fallback(video_path, output_dir, timestamps, max_size, hw_accel, prefix)


def _extract_frames_fallback(
    video_path: str,
    output_dir: str,
    timestamps: List[float],
    max_size: int = 800,
    hw_accel: Optional[str] = None,
    prefix: str = "frame",
) -> List[str]:
    """逐帧提取（回退方案）"""
    frame_paths = []
    for i, ts in enumerate(timestamps):
        frame_path = str(Path(output_dir) / f"{prefix}_{i:04d}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=max_size, hw_accel=hw_accel):
            frame_paths.append(frame_path)
    return frame_paths


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
    use_batch: bool = True,
    hw_accel: Optional[str] = None,
) -> List[str]:
    """
    从视频中提取多个均匀分布的帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        n_frames: 帧数
        max_size: 最大尺寸
        skip_start_end: 是否跳过开头和结尾
        use_batch: 是否使用批量提取
        hw_accel: 硬件解码器（None=自动检测）
    
    Returns:
        有效帧文件路径列表
    """
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
    
    # 自动检测硬件加速
    if hw_accel is None and use_batch:
        hw_accel = detect_hw_accel()
    
    if use_batch:
        # 批量提取
        frame_paths = extract_frames_batch(
            video_path, output_dir, timestamps, max_size, hw_accel, prefix=f"{video_hash}_frame"
        )
    else:
        # 逐帧提取
        frame_paths = []
        for i, ts in enumerate(timestamps):
            frame_path = Path(output_dir) / f"{video_hash}_frame_{i}_{ts:.1f}.jpg"
            if extract_frame(video_path, str(frame_path), timestamp=str(ts), max_size=max_size, hw_accel=hw_accel):
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

    # 使用批量提取采样帧
    hw_accel = detect_hw_accel()
    sample_paths = extract_frames_batch(
        video_path, str(tmp_dir), timestamps, max_size=400, hw_accel=hw_accel, prefix="sample"
    )
    
    frames_data = []
    for i, (frame_path, ts) in enumerate(zip(sample_paths, timestamps)):
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

    # 提取关键帧（使用批量提取）
    keyframe_timestamps = [frames_data[idx]["timestamp"] for idx in keyframe_indices]
    keyframe_paths = extract_frames_batch(
        video_path, output_dir, keyframe_timestamps, max_size=max_size, hw_accel=hw_accel, 
        prefix=f"{video_hash}_keyframe"
    )

    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    logger.info(f"检测到 {len(keyframe_paths)} 个关键帧")
    return keyframe_paths


def extract_frames_cv2(
    video_path: str,
    output_dir: str,
    timestamps: List[float],
    max_size: int = 400,
    quality: int = 90,
) -> List[Optional[str]]:
    """
    用 cv2 单次顺序解码批量提取帧（grab 跳帧 + retrieve 仅解码目标帧），
    替代逐帧 ffmpeg 子进程。

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        timestamps: 目标时间戳（秒）列表
        max_size: 输出图片最长边
        quality: JPEG 质量

    Returns:
        与 timestamps 对齐的路径列表；失败位置为 None。
        文件名: frame_{i:04d}_{ts:.1f}s.jpg（i 为 timestamps 下标）
    """
    if not timestamps:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(idx: int, ts: float, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(
                frame,
                (max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2),
                interpolation=cv2.INTER_AREA,
            )
        path = out_dir / f"frame_{idx:04d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return str(path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}")
        return [None] * len(timestamps)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        logger.warning(f"视频元数据无效，无法批量抽帧: {video_path}")
        return [None] * len(timestamps)

    # 目标帧索引 → 时间戳下标列表（多个时间戳可能落在同一帧）
    target_map: Dict[int, List[int]] = {}
    for i, ts in enumerate(timestamps):
        idx = min(max(int(round(ts * fps)), 0), total - 1)
        target_map.setdefault(idx, []).append(i)

    results: List[Optional[str]] = [None] * len(timestamps)
    frame_idx = -1
    while frame_idx < total - 1 and target_map:
        if not cap.grab():  # 快速跳过，不解码
            break
        frame_idx += 1
        if frame_idx not in target_map:
            continue
        ok, frame = cap.retrieve()  # 仅对目标帧解码
        wanted = target_map.pop(frame_idx)
        if not ok or frame is None:
            continue
        for i in wanted:
            results[i] = _save(i, timestamps[i], frame)

    cap.release()

    got = sum(1 for r in results if r is not None)
    logger.info(f"cv2批量抽帧: {got}/{len(timestamps)} 帧 ({video_path})")
    return results
