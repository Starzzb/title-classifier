import csv
import json
import os
import re
import sys
import time
import base64
import subprocess
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Union, List

import cv2
import numpy as np

# 人体检测模块（UHD）
try:
    from stage1c_frame_selector import (load_model, find_human_frame, get_video_duration, 
                                        safe_timestamp, detect_and_crop_human, crop_human_region)
    HAS_FRAME_SELECTOR = True
except ImportError:
    HAS_FRAME_SELECTOR = False

# YOLO 检测模块
try:
    from stage1c_yolo_detector import (YOLODetector, YOLOTracker, detect_human_yolo, 
                                       find_human_frame_yolo, YOLO_MODELS)
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# CLIP 分类模块
try:
    from stage1c_clip_classifier import CLIPClassifier, get_classifier
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# 标签统计模块
try:
    from tag_statistics import TagStatistics
    HAS_TAG_STATS = True
except ImportError:
    HAS_TAG_STATS = False

LOG_FILE = "logs/vision_refine_log.txt"

# 支持的媒体格式
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.ts'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff'}

# 使用统一的 Provider 配置
from providers import get_provider_config, get_api_key as get_provider_api_key, call_vision_api as provider_call_vision_api


def load_env(env_path: Path):
    if not env_path.exists():
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip())


def log_message(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    # 强制使用utf-8输出，避免Windows GBK编码问题
    sys.stdout.buffer.write((log_line + '\n').encode('utf-8', errors='replace'))
    sys.stdout.buffer.flush()
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')


def is_image_file(file_path: str) -> bool:
    """判断是否为图片文件"""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """判断是否为视频文件"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def is_vision_result_complete(row: dict) -> bool:
    """
    检查行是否有完整的vision结果
    
    完整结果定义：
    - vision_keywords 不为空
    - vision_description 不为空且不是错误信息
    - final_name 已更新（包含方括号格式）
    """
    keywords = row.get("vision_keywords", "").strip()
    description = row.get("vision_description", "").strip()
    final_name = row.get("final_name", "").strip()
    
    # 检查关键词是否存在
    if not keywords:
        return False
    
    # 检查描述是否存在且不是错误
    if not description or description.startswith("[ERROR]"):
        return False
    
    # 检查final_name是否已更新（包含方括号格式）
    if not final_name or not final_name.startswith("["):
        return False
    
    return True


def save_rows_to_csv(rows: list, fieldnames: list, output_path: str):
    """
    增量保存rows到CSV文件
    
    Args:
        rows: 数据行列表
        fieldnames: 列名列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compress_image(input_path: str, output_path: str, max_size: int = 800, quality: int = 85) -> bool:
    """
    压缩图片，保持宽高比
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        max_size: 最大边长（像素）
        quality: JPEG 质量 (1-100)
    
    Returns:
        是否成功
    """
    try:
        import cv2
        import numpy as np
        # 使用 numpy 读取以支持中文路径
        data = np.fromfile(input_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        # 计算缩放比例
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 保存为 JPEG
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return os.path.exists(output_path)
    except Exception as e:
        log_message(f"[压缩] 图片压缩失败: {e}")
        return False


def extract_frame(video_path: str, output_path: str, timestamp: str = None, 
                  max_size: int = 800) -> bool:
    """
    提取视频帧并压缩
    
    Args:
        video_path: 视频路径
        output_path: 输出帧路径
        timestamp: 时间点（HH:MM:SS 格式或秒数），None 表示使用视频 1/4 位置
        max_size: 最大边长（像素）
    
    Returns:
        是否成功
    """
    try:
        # 获取视频时长
        duration = get_video_duration(video_path)
        
        # 如果未指定时间点，使用视频 1/4 位置
        if timestamp is None:
            ts_seconds = duration / 4 if duration > 0 else 30.0
            log_message(f"[帧提取] 使用视频 1/4 位置: {ts_seconds:.1f}s (时长: {duration:.1f}s)")
        else:
            # 解析时间戳为秒数
            parts = timestamp.split(':')
            if len(parts) == 3:
                ts_seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                ts_seconds = float(parts[0]) * 60 + float(parts[1])
            else:
                ts_seconds = float(timestamp)
        
        # 安全处理时间戳
        safe_ts = safe_timestamp(ts_seconds, duration)
        if safe_ts != ts_seconds:
            log_message(f"[帧提取] 时间点调整: {ts_seconds:.1f}s -> {safe_ts:.1f}s (视频时长: {duration:.1f}s)")
        
        # 转换回 HH:MM:SS.mmm 格式
        hours = int(safe_ts // 3600)
        minutes = int((safe_ts % 3600) // 60)
        seconds = safe_ts % 60
        safe_timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        # 使用 ffmpeg 提取并同时缩放，避免大文件
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", safe_timestamp_str, "-i", video_path,
             "-vf", f"scale='if(gt(iw,{max_size}),{max_size},-2)':'if(gt(ih,{max_size}),{max_size},-2)'",
             "-frames:v", "1", "-q:v", "2", output_path],
            capture_output=True, timeout=15
        )
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def is_solid_color_frame(image_path: str, threshold: float = 15.0) -> bool:
    """
    检测图片是否为纯色（纯黑/纯白/纯灰等）
    
    Args:
        image_path: 图片路径
        threshold: 标准差阈值（越低越严格）
    
    Returns:
        True 如果是纯色帧
    """
    try:
        import cv2
        import numpy as np
        
        # 使用numpy读取以支持中文路径
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return True  # 无法读取视为无效
        
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算标准差
        std_dev = np.std(gray)
        
        # 标准差低于阈值认为是纯色
        return std_dev < threshold
    except Exception:
        return True  # 异常视为无效


def filter_solid_frames(frame_paths: list, threshold: float = 15.0) -> list:
    """
    过滤掉纯色帧，保留有效帧
    
    Args:
        frame_paths: 帧路径列表
        threshold: 纯色检测阈值
    
    Returns:
        有效帧路径列表
    """
    valid_frames = []
    for fp in frame_paths:
        if os.path.exists(fp) and not is_solid_color_frame(fp, threshold):
            valid_frames.append(fp)
    return valid_frames


def extract_multiple_frames(video_path: str, output_dir: str, n_frames: int = 5, 
                           max_size: int = 800, skip_start_end: bool = True) -> list:
    """
    从视频中提取多个均匀分布的帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        n_frames: 帧数
        max_size: 最大边长
        skip_start_end: 是否跳过开头和结尾（避免黑屏/片头片尾）
    
    Returns:
        成功提取的帧路径列表
    """
    import hashlib
    
    duration = get_video_duration(video_path)
    if duration <= 0:
        log_message(f"[多帧提取] 无法获取视频时长: {video_path}")
        return []
    
    # 跳过开头和结尾（避免黑屏/片头片尾）
    if skip_start_end and duration > 10:
        start_offset = min(3.0, duration * 0.05)  # 跳过前5%或3秒
        end_offset = min(3.0, duration * 0.05)    # 跳过后5%或3秒
        effective_duration = duration - start_offset - end_offset
        timestamps = [start_offset + effective_duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]
    else:
        timestamps = [duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]
    
    # 使用视频hash作为文件名前缀
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    frame_paths = []
    
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(output_dir, f"{video_hash}_frame_{i}_{ts:.1f}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=max_size):
            frame_paths.append(frame_path)
    
    # 过滤纯色帧
    valid_frames = filter_solid_frames(frame_paths)
    
    log_message(f"[多帧提取] 成功提取 {len(frame_paths)} 帧，有效帧 {len(valid_frames)} 帧")
    return valid_frames


def detect_keyframes(video_path: str, output_dir: str, max_frames: int = 8,
                     threshold: float = 30.0, max_size: int = 800) -> list:
    """
    基于帧差异的关键帧检测
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        max_frames: 最大帧数
        threshold: 差异阈值（越低越敏感）
        max_size: 最大边长
    
    Returns:
        关键帧路径列表
    """
    import hashlib
    import cv2
    import numpy as np
    
    duration = get_video_duration(video_path)
    if duration <= 0:
        return []
    
    # 跳过开头和结尾（避免黑屏/片头片尾）
    start_offset = min(3.0, duration * 0.05)  # 跳过前5%或3秒
    end_offset = min(3.0, duration * 0.05)    # 跳过后5%或3秒
    effective_duration = duration - start_offset - end_offset
    
    # 先均匀采样较多帧用于分析
    n_samples = min(20, int(effective_duration / 2))  # 每2秒一帧，最多20帧
    if n_samples < 4:
        n_samples = 4
    
    timestamps = [start_offset + effective_duration * (i + 1) / (n_samples + 1) for i in range(n_samples)]
    
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    tmp_dir = os.path.join(output_dir, f"_keyframe_tmp_{video_hash}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 提取所有采样帧
    frames_data = []
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(tmp_dir, f"sample_{i}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=400):  # 使用小尺寸加速
            try:
                # 检查是否为纯色帧
                if is_solid_color_frame(frame_path):
                    continue
                
                img = cv2.imread(frame_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames_data.append({
                        "path": frame_path,
                        "timestamp": ts,
                        "gray": gray,
                        "index": i
                    })
            except Exception:
                pass
    
    if len(frames_data) < 2:
        log_message(f"[关键帧] 采样帧不足，跳过关键帧检测")
        return []
    
    # 计算帧间差异
    keyframe_indices = [0]  # 第一帧总是关键帧
    prev_gray = frames_data[0]["gray"]
    
    for i in range(1, len(frames_data)):
        curr_gray = frames_data[i]["gray"]
        
        # 计算帧差异（平均像素差）
        diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
        
        if diff > threshold:
            keyframe_indices.append(i)
            prev_gray = curr_gray
        
        # 限制关键帧数量
        if len(keyframe_indices) >= max_frames:
            break
    
    # 如果关键帧太少，补充均匀分布的帧
    if len(keyframe_indices) < 3:
        additional = [i for i in range(len(frames_data)) if i not in keyframe_indices]
        step = max(1, len(additional) // (3 - len(keyframe_indices)))
        for i in range(0, len(additional), step):
            if len(keyframe_indices) >= 3:
                break
            keyframe_indices.append(additional[i])
    
    keyframe_indices.sort()
    
    # 重新提取高分辨率关键帧
    keyframe_paths = []
    for idx in keyframe_indices:
        ts = frames_data[idx]["timestamp"]
        frame_path = os.path.join(output_dir, f"{video_hash}_keyframe_{idx}_{ts:.1f}.jpg")
        if extract_frame(video_path, frame_path, timestamp=str(ts), max_size=max_size):
            keyframe_paths.append(frame_path)
    
    # 清理临时文件
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass
    
    log_message(f"[关键帧] 检测到 {len(keyframe_paths)} 个关键帧")
    return keyframe_paths


def image_to_base64(image_path: str, max_size: int = 800) -> str:
    """
    读取图片并压缩后转 base64
    
    Args:
        image_path: 图片路径
        max_size: 最大边长
    
    Returns:
        base64 字符串
    """
    import cv2
    import numpy as np
    
    # 使用 numpy 读取以支持中文路径
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # 降级：直接读取原文件
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    h, w = img.shape[:2]
    
    # 压缩
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 编码为 JPEG
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def build_vision_prompt(title: str, n_frames: int = 1, yolo_context: str = None) -> str:
    """构建VLM提示词，支持YOLO上下文增强"""
    
    # YOLO上下文部分
    yolo_section = ""
    if yolo_context:
        yolo_section = f"\n【视频预分析结果】\n{yolo_context}\n"
    
    if n_frames > 1:
        return (
            f'这是媒体文件 "{title}" 的{n_frames}个关键帧截图，用于文件管理归类。请进行纯技术性视觉分析：\n'
            f'{yolo_section}\n'
            f'1. 描述：综合所有帧，客观描述画面中的可见元素（场景、人物穿着、人物动作、水印等）\n'
            f'2. 关键词：提取 4-8 个关键词，用逗号分隔\n\n'
            f'【关键词提取规则】\n'
            f'- 第一优先级：可见文字/水印中的名称（个人昵称/艺名，非@开头的频道名）\n'
            f'- 第二优先级：人物穿着、人物行为，可见物体\n'
            f'- 综合所有帧的信息，提取最具代表性的标签\n\n'
            f'注意：@开头的频道名/群组名（如@ciyuanbb）不需要提取。水印中的广告内容不必提取\n\n'
            f'请始终返回结果，即使只能识别部分内容也请描述。严格按以下格式返回：\n'
            f'描述：xxx\n'
            f'关键词：xxx, xxx, xxx'
        )
    else:
        return (
            f'这是媒体文件 "{title}" 的截图，用于文件管理归类。请进行纯技术性视觉分析：\n'
            f'{yolo_section}\n'
            f'1. 描述：客观描述画面中的可见元素（场景、人物穿着、人物动作、水印等）\n'
            f'2. 关键词：提取 4-8 个关键词，用逗号分隔\n\n'
            f'【关键词提取规则】\n'
            f'- 第一优先级：可见文字/水印中的名称（个人昵称/艺名，非@开头的频道名）\n'
            f'- 第二优先级：人物穿着、人物行为，可见物体\n\n'
            f'注意：@开头的频道名/群组名（如@ciyuanbb）不需要提取。水印中的广告内容不必提取\n\n'
            f'请始终返回结果，即使只能识别部分内容也请描述。严格按以下格式返回：\n'
            f'描述：xxx\n'
            f'关键词：xxx, xxx, xxx'
        )


def build_yolo_context(yolo_results: list, selected_indices: list = None) -> str:
    """
    构建YOLO检测结果的上下文描述
    
    Args:
        yolo_results: YOLO检测结果列表
        selected_indices: 选中的帧索引（可选，只描述选中的帧）
    
    Returns:
        YOLO上下文描述字符串
    """
    if not yolo_results:
        return ""
    
    context_lines = []
    
    # 整体统计
    frames_with_person = [r for r in yolo_results if r.get("has_person")]
    if frames_with_person:
        # 统计姿态分布
        pose_counts = {}
        for r in frames_with_person:
            poses = r.get("pose_analysis", ["未知"])
            for pose in poses:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        # 主要姿态
        main_pose = max(pose_counts, key=pose_counts.get) if pose_counts else "未知"
        
        context_lines.append(f"- 视频中检测到人体，主要姿态：{main_pose}")
        context_lines.append(f"- 姿态分布：{', '.join([f'{k}({v}次)' for k, v in sorted(pose_counts.items(), key=lambda x: -x[1])])}")
        
        # 可见关键点统计
        avg_visible = sum(r.get("visible_count", 0) for r in frames_with_person) / len(frames_with_person)
        avg_conf = sum(r.get("avg_confidence", 0) for r in frames_with_person) / len(frames_with_person)
        context_lines.append(f"- 平均可见关键点：{avg_visible:.1f}/17，平均置信度：{avg_conf:.2f}")
    
    # 描述选中的帧
    if selected_indices and yolo_results:
        context_lines.append(f"\n选中的{n_frames}帧分析：")
        for i, idx in enumerate(selected_indices):
            if idx < len(yolo_results):
                r = yolo_results[idx]
                if r.get("has_person"):
                    poses = r.get("pose_analysis", ["未知"])
                    conf = r.get("avg_confidence", 0)
                    visible = r.get("visible_count", 0)
                    timestamp = r.get("timestamp", 0)
                    context_lines.append(f"  帧{i+1} ({timestamp:.1f}秒): {', '.join(poses)} (置信度:{conf:.2f}, 关键点:{visible}/17)")
                else:
                    context_lines.append(f"  帧{i+1}: 未检测到人体")
    
    return "\n".join(context_lines) if context_lines else ""


def select_best_frames_by_pose(yolo_results: list, max_frames: int = 5) -> list:
    """
    基于YOLO姿态分析选择最佳帧送给VLM
    
    策略：
    1. 优先选择姿态变化大的帧（捕捉动作多样性）
    2. 优先选择置信度高的帧（保证检测质量）
    3. 优先选择可见关键点多的帧（姿态信息完整）
    
    Args:
        yolo_results: YOLO检测结果列表
        max_frames: 最大帧数
    
    Returns:
        选中的帧索引列表
    """
    if not yolo_results:
        return []
    
    # 筛选有人体的帧
    frames_with_person = [(i, r) for i, r in enumerate(yolo_results) if r.get("has_person")]
    
    if not frames_with_person:
        # 如果没有检测到人体，返回均匀分布的帧
        step = max(1, len(yolo_results) // max_frames)
        return list(range(0, len(yolo_results), step))[:max_frames]
    
    if len(frames_with_person) <= max_frames:
        # 如果有人体的帧数不足，直接返回所有
        return [i for i, r in frames_with_person]
    
    # 计算每帧的得分
    scored_frames = []
    prev_pose = None
    
    for i, result in frames_with_person:
        score = 0.0
        
        # 置信度得分 (0-40分)
        conf = result.get("avg_confidence", 0)
        score += conf * 40
        
        # 关键点可见性得分 (0-30分)
        visible = result.get("visible_count", 0)
        score += (visible / 17) * 30
        
        # 姿态多样性得分 (0-30分)
        current_pose = tuple(result.get("pose_analysis", []))
        if prev_pose is not None and current_pose != prev_pose:
            score += 30
        prev_pose = current_pose
        
        scored_frames.append((i, score, result))
    
    # 按得分排序，选择top-N
    scored_frames.sort(key=lambda x: x[1], reverse=True)
    selected = [item[0] for item in scored_frames[:max_frames]]
    
    # 按时间顺序排序
    selected.sort()
    
    return selected


def save_yolo_results(output_dir: str, video_name: str, yolo_results: list, selected_indices: list):
    """
    保存YOLO检测结果到单独的JSON文件
    
    Args:
        output_dir: 输出目录
        video_name: 视频文件名
        yolo_results: YOLO检测结果
        selected_indices: 选中的帧索引
    """
    import json
    
    yolo_dir = Path(output_dir) / "yolo_results"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建保存的数据
    save_data = {
        "video_name": video_name,
        "total_frames_analyzed": len(yolo_results),
        "selected_frame_indices": selected_indices,
        "summary": {},
        "frames": yolo_results
    }
    
    # 生成摘要
    frames_with_person = [r for r in yolo_results if r.get("has_person")]
    if frames_with_person:
        pose_counts = {}
        for r in frames_with_person:
            poses = r.get("pose_analysis", ["未知"])
            for pose in poses:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        save_data["summary"] = {
            "frames_with_person": len(frames_with_person),
            "person_ratio": len(frames_with_person) / len(yolo_results) if yolo_results else 0,
            "pose_distribution": pose_counts,
            "main_pose": max(pose_counts, key=pose_counts.get) if pose_counts else "未知",
            "avg_confidence": sum(r.get("avg_confidence", 0) for r in frames_with_person) / len(frames_with_person),
            "avg_visible_keypoints": sum(r.get("visible_count", 0) for r in frames_with_person) / len(frames_with_person)
        }
    
    # 保存文件
    json_path = yolo_dir / f"{Path(video_name).stem}_yolo.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    log_message(f"[YOLO] 结果已保存: {json_path}")
    return json_path


def parse_vision_response(response: str) -> dict:
    result = {"description": "", "keywords": ""}
    desc_match = re.search(r'描述[：:]\s*(.+)', response)
    kw_match = re.search(r'关键词[：:]\s*(.+)', response)
    if desc_match:
        result["description"] = desc_match.group(1).strip()
    if kw_match:
        result["keywords"] = kw_match.group(1).strip()
    return result


def call_vision_api(image_b64: Union[str, List[str]], title: str, provider: str,
                    model: str = None, api_key: str = None, timeout: int = 90,
                    retries: int = 3, n_frames: int = 1, yolo_context: str = None) -> dict:
    """
    调用视觉 API
    
    Args:
        image_b64: 单个或多个 base64 编码的图片
        title: 标题
        provider: Provider 名称
        model: 模型名称（可选）
        api_key: API Key（可选）
        timeout: 超时时间
        retries: 重试次数
        n_frames: 帧数
        yolo_context: YOLO检测上下文（可选）
    
    Returns:
        {"description": str, "keywords": str}
    """
    prompt = build_vision_prompt(title, n_frames, yolo_context)
    
    result = provider_call_vision_api(
        provider_name=provider,
        image_b64=image_b64,
        prompt=prompt,
        model=model,
        api_key=api_key,
        timeout=timeout,
        retries=retries
    )
    
    return parse_vision_response(result)


def main():
    load_env(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(description="阶段 1c：使用视觉大模型从媒体文件提取关键词")
    parser.add_argument("-c", "--csv", type=str, default="output/title_review.csv", help="待审表路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出路径（默认覆盖原文件）")
    parser.add_argument("-p", "--provider", type=str, default="gcli",
                        choices=["mimo", "gcli"], help="API 提供商（默认 gcli）")
    parser.add_argument("-m", "--model", type=str, default="",
                        help="模型名称（默认使用 provider 对应的默认模型）")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key（也可通过 MIMO_API_KEY 或 GCLI_API_KEY 环境变量设置）")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="截取视频帧的时间点（默认自动使用视频 1/4 位置）")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="每批处理数量（默认 3）")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="批次间隔秒数（默认 2）")
    parser.add_argument("--retries", type=int, default=3,
                        help="API 失败重试次数（默认 3）")
    parser.add_argument("--retry-errors", action="store_true",
                        help="重新处理之前失败的行（vision_description 为 [ERROR] 的）")
    parser.add_argument("--single", type=str, default="",
                        help="仅处理指定的文件名（如 IMG_7940.MP4）")
    parser.add_argument("--all", action="store_true",
                        help="处理所有标题（忽略 needs_vision 标记）")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不写入文件")
    parser.add_argument("--no-frame-selector", action="store_true",
                        help="禁用人体检测预处理（默认启用）")
    parser.add_argument("--model-path", type=str, 
                        default="models/human_detection/ultratinyod_res_anc8_w128_64x64_loese_distill.onnx",
                        help="UHD 模型路径")
    parser.add_argument("--step-seconds", type=float, default=5.0,
                        help="帧提取步长（秒）")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="最大重试次数")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="人体检测置信度阈值")
    parser.add_argument("--max-image-size", type=int, default=800,
                        help="图片最大边长（像素），超过会压缩（默认 800）")
    # CLIP 相关参数
    parser.add_argument("--use-clip", action="store_true", default=False,
                        help="启用 CLIP 本地预分类（需安装 open-clip-torch）")
    parser.add_argument("--no-clip", action="store_true",
                        help="禁用 CLIP，直接使用云端 VLM")
    parser.add_argument("--clip-threshold", type=float, default=0.25,
                        help="CLIP 置信度阈值，低于此值触发云端 VLM（默认 0.25）")
    parser.add_argument("--clip-frames", type=int, default=5,
                        help="视频 CLIP 分析的帧数（默认 5）")
    parser.add_argument("--vlm-frames", type=int, default=5,
                        help="送入云端VLM的帧数（默认 5，多帧可提高准确度）")
    parser.add_argument("--multi-label", action="store_true", default=True,
                        help="CLIP 多标签模式（默认启用）")
    parser.add_argument("--single-label", action="store_true",
                        help="CLIP 单标签模式")
    parser.add_argument("--keyframe-threshold", type=float, default=30.0,
                        help="关键帧检测差异阈值（默认 30.0，越低越敏感）")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="最大关键帧数（默认 8）")
    # Embedding检测参数
    parser.add_argument("--use-embedding-detection", action="store_true", default=True,
                        help="使用embedding检测人体区域变化（默认启用）")
    parser.add_argument("--no-embedding-detection", action="store_true",
                        help="禁用embedding检测，使用标签比较")
    parser.add_argument("--embedding-threshold", type=float, default=0.75,
                        help="Embedding相似度阈值，低于此值认为有变化（默认0.75，稳健值）")
    # YOLO 检测参数
    parser.add_argument("--use-yolo", action="store_true", default=False,
                        help="使用 YOLOv8 替代 UHD 进行人体检测（需安装 ultralytics）")
    parser.add_argument("--yolo-model", type=str, default="detect",
                        choices=["detect", "pose", "segment"],
                        help="YOLO 模型类型（默认 detect）")
    parser.add_argument("--yolo-conf", type=float, default=0.5,
                        help="YOLO 置信度阈值（默认 0.5）")
    parser.add_argument("--yolo-pose", action="store_true", default=False,
                        help="使用 YOLO 姿态估计（自动启用 --use-yolo）")
    parser.add_argument("--yolo-segment", action="store_true", default=False,
                        help="使用 YOLO 实例分割（自动启用 --use-yolo）")
    args = parser.parse_args()

    # 使用统一的 Provider 配置
    provider = args.provider
    provider_config = get_provider_config(provider)
    
    if not provider_config:
        print(f"错误: 未知的 Provider '{provider}'")
        return
    
    model = args.model or provider_config.get("default_model", "")
    api_key = args.api_key or get_provider_api_key(provider)
    
    # 处理embedding检测参数
    if args.no_embedding_detection:
        args.use_embedding_detection = False

    if provider_config.get("requires_api_key", False) and not api_key:
        print(f"错误: 需要提供 --api-key 或设置 {provider_config.get('env_key', '')} 环境变量")
        return

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"错误: 未找到 {csv_path}")
        return

    output_path = Path(args.output).resolve() if args.output else csv_path

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("错误: CSV 文件为空")
            return
        fieldnames = list(reader.fieldnames)
        if "vision_description" not in fieldnames:
            fieldnames.append("vision_description")
        if "vision_keywords" not in fieldnames:
            fieldnames.append("vision_keywords")
        if "human_detected" not in fieldnames:
            fieldnames.append("human_detected")
        if "detection_confidence" not in fieldnames:
            fieldnames.append("detection_confidence")
        if "detection_timestamp" not in fieldnames:
            fieldnames.append("detection_timestamp")
        # CLIP 相关列
        for col in ["clip_clothing", "clip_action", "clip_hairstyle",
                     "clip_tags", "clip_tags_json", "clip_confidence", "vision_source"]:
            if col not in fieldnames:
                fieldnames.append(col)
        rows = list(reader)

    if not rows:
        print("CSV 中无数据")
        return

    has_needs_vision = "needs_vision" in fieldnames

    if args.retry_errors:
        for row in rows:
            if row.get("vision_description", "").startswith("[ERROR]"):
                row["vision_description"] = ""
                row["vision_keywords"] = ""
        pending = [(i, row) for i, row in enumerate(rows)
                   if row.get("original_title", "").strip()
                   and not row.get("vision_keywords", "").strip()
                   and (not has_needs_vision or row.get("needs_vision", "").strip().lower() == "true")]
    elif args.single:
        single = args.single.strip()
        single_stem = Path(single).stem.lower()
        pending = [(i, row) for i, row in enumerate(rows)
                   if single.lower() in row.get("original_title", "").lower()
                   or single.lower() in row.get("original_path", "").lower()
                   or single_stem == row.get("original_title", "").strip().lower()]
        if not pending:
            print(f"错误: 未找到匹配 '{single}' 的文件")
            return
    elif args.all:
        pending = [(i, row) for i, row in enumerate(rows)
                   if row.get("original_title", "").strip()
                   and not is_vision_result_complete(row)]
    elif has_needs_vision:
        pending = [(i, row) for i, row in enumerate(rows)
                   if row.get("original_title", "").strip()
                   and not is_vision_result_complete(row)
                   and row.get("needs_vision", "").strip().lower() == "true"]
    else:
        meaningless_pattern = re.compile(
            r'^[\d\-_.\s\(\)\[\]【】（）]+$'  # 纯数字+分隔符+括号
            r'|^[a-f0-9]{8,}$'                # 纯hex
            r'|^IMG_\d+$|^VID_'                # 设备前缀
            r'|^@\w+'                           # @开头
            r'|^[A-Z0-9\-]{20,}$'              # 长字母数字
        )
        def is_meaningless(title):
            t = title.strip()
            if meaningless_pattern.match(t):
                return True
            # 去括号后纯数字
            stripped = re.sub(r'[\(\)\[\]【】（）\s\-_.]+', '', t)
            if stripped.isdigit():
                return True
            # 无中文 + 含域名
            if not re.search(r'[\u4e00-\u9fff]', t) and re.search(r'(?i)(\.cc|\.com|\.net|\.org|\.vip)', t):
                return True
            # 无中文 + 数字占比>70%
            if not re.search(r'[\u4e00-\u9fff]', t):
                core = re.sub(r'[\(\)\[\]【】（）\-_.\s]+', '', t)
                if core and len(re.findall(r'\d', core)) / len(core) > 0.7:
                    return True
            return False
        pending = [(i, row) for i, row in enumerate(rows)
                   if row.get("original_title", "").strip()
                   and not is_vision_result_complete(row)
                   and is_meaningless(row["original_title"])]

    skip_count = len(rows) - len(pending)
    print(f"共 {len(rows)} 条记录，待处理 {len(pending)} 条，跳过 {skip_count} 条")
    print(f"使用 {args.provider} 模型: {model}，每批 {args.batch_size} 条，间隔 {args.delay}s")
    
    # YOLO 参数处理
    if args.yolo_pose:
        args.use_yolo = True
        args.yolo_model = "pose"
    elif args.yolo_segment:
        args.use_yolo = True
        args.yolo_model = "segment"
    
    use_frame_selector = not args.no_frame_selector
    use_yolo = args.use_yolo and HAS_YOLO
    
    if use_yolo:
        print(f"启用 YOLO 人体检测（模型: {args.yolo_model}，置信度: {args.yolo_conf}）")
        if not HAS_YOLO:
            print("警告: 无法导入 YOLO 检测模块，将使用 UHD 模型")
            use_yolo = False
    elif use_frame_selector:
        print(f"启用 UHD 人体检测预处理（仅视频），模型: {args.model_path}")
        if not HAS_FRAME_SELECTOR:
            print("警告: 无法导入人体检测模块，视频将使用默认帧提取")
    print(f"图片最大尺寸: {args.max_image_size}px")
    if args.dry_run:
        print("模拟模式，不会写入文件")

    # 加载人体检测模型
    frame_selector_session = None
    yolo_detector = None
    
    if use_yolo:
        # 加载 YOLO 模型
        yolo_detector = YOLODetector(
            model_type=args.yolo_model,
            conf_threshold=args.yolo_conf
        )
        if not yolo_detector.load_model():
            print("警告: YOLO 模型加载失败，将使用 UHD 模型")
            use_yolo = False
            yolo_detector = None
    elif use_frame_selector and HAS_FRAME_SELECTOR:
        # 加载 UHD 模型
        frame_selector_session = load_model(args.model_path)

    # 初始化标签统计
    tag_stats = None
    if HAS_TAG_STATS:
        tag_stats = TagStatistics()
        print(f"标签统计已加载: models/tag_statistics.json")
        print(tag_stats.get_stats_summary())

    # 初始化 CLIP 分类器
    use_clip = args.use_clip and not args.no_clip and HAS_CLIP
    clip_classifier = None
    if use_clip:
        clip_classifier = CLIPClassifier(tag_stats=tag_stats)
        if clip_classifier.load_model():
            print(f"CLIP 分类器已启用，阈值: {args.clip_threshold}")
        else:
            print("警告: CLIP 模型加载失败，将使用纯云端 VLM")
            use_clip = False
            clip_classifier = None

    tmp_dir = Path("logs/_vision_tmp")
    tmp_dir.mkdir(exist_ok=True)

    batch_size = args.batch_size

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size

        log_message(f"[批次 {batch_num}/{total_batches}] 处理 {len(batch)} 条")

        for idx, row in batch:
            original = row.get("original_title", "").strip()
            file_path = row.get("original_path", "").strip()

            if args.dry_run:
                log_message(f"  [模拟] {original[:40]}")
                continue

            if not file_path or not os.path.exists(file_path):
                log_message(f"  [{idx+1}] 跳过（文件不存在）: {original[:30]}")
                continue

            tmp_img = str(tmp_dir / f"frame_{idx}.jpg")
            is_image = is_image_file(file_path)

            # ===== CLIP 预分类 =====
            clip_result = None
            clip_confident = False
            if use_clip and clip_classifier is not None:
                # 先压缩图片供 CLIP 使用
                clip_img = tmp_img
                if is_image:
                    if not compress_image(file_path, clip_img, max_size=args.max_image_size):
                        clip_img = file_path
                # 对图片直接 CLIP；对视频先提取帧再 CLIP
                if is_image:
                    clip_result = clip_classifier.classify(
                        clip_img, threshold=args.clip_threshold,
                        multi_label=not args.single_label
                    )
                    if clip_result.get("avg_confidence", 0) >= args.clip_threshold:
                        clip_confident = True

            if is_image:
                # ========== 图片处理 ==========
                log_message(f"  [{idx+1}] 图片模式: {original[:30]}")
                row["human_detected"] = "N/A (图片)"
                row["detection_confidence"] = "N/A"
                row["detection_timestamp"] = "N/A"

                if not compress_image(file_path, tmp_img, max_size=args.max_image_size):
                    log_message(f"  [{idx+1}] 图片压缩失败，尝试直接读取")
                    tmp_img = file_path

                if clip_confident:
                    # CLIP 置信度高，直接使用，不调云端
                    row["clip_clothing"] = clip_result["clothing"]["label_cn"]
                    row["clip_action"] = clip_result["action"]["label_cn"]
                    row["clip_hairstyle"] = clip_result["hairstyle"]["label_cn"]
                    row["clip_tags"] = clip_result["tags"]
                    row["clip_tags_json"] = json.dumps(clip_result["tags_json"], ensure_ascii=False)
                    row["clip_confidence"] = f"{clip_result['avg_confidence']:.2f}"
                    row["vision_source"] = "clip_only"
                    row["vision_description"] = f"[CLIP] {clip_result['tags']}"
                    row["vision_keywords"] = clip_result["tags"]
                    # 用 CLIP 标签生成 final_name
                    if clip_result["tags"]:
                        kw_list = [k for k in clip_result["tags"].split("_") if k.strip()][:5]
                        if kw_list:
                            row["final_name"] = f"[{'_'.join(kw_list)}]_{original}"
                    log_message(f"  [{idx+1}] CLIP 直接分类: {clip_result['tags']} "
                              f"(置信度: {clip_result['avg_confidence']:.2f})")
                    if os.path.exists(tmp_img) and tmp_img != file_path:
                        os.remove(tmp_img)
                    continue

                # CLIP 置信度低或未启用，记录 CLIP 结果后继续调云端
                if clip_result:
                    row["clip_clothing"] = clip_result["clothing"]["label_cn"]
                    row["clip_action"] = clip_result["action"]["label_cn"]
                    row["clip_hairstyle"] = clip_result["hairstyle"]["label_cn"]
                    row["clip_tags"] = clip_result["tags"]
                    row["clip_tags_json"] = json.dumps(clip_result["tags_json"], ensure_ascii=False)
                    row["clip_confidence"] = f"{clip_result['avg_confidence']:.2f}"
                    row["vision_source"] = "clip+cloud"
                else:
                    row["vision_source"] = "cloud_only"
                
                # 设置图片帧用于VLM调用
                vlm_frames = [tmp_img]

            elif is_video_file(file_path):
                # ========== 视频处理（YOLO增强版） ==========
                duration = get_video_duration(file_path)
                log_message(f"  [{idx+1}] 视频模式: {original[:30]} (时长: {duration:.1f}s)")
                
                # 收集YOLO检测结果
                yolo_results = []
                
                # 策略1: 关键帧检测（检测场景变化）
                keyframe_paths = []
                if use_clip and clip_classifier is not None:
                    log_message(f"  [{idx+1}] 尝试关键帧检测...")
                    keyframe_paths = detect_keyframes(
                        file_path, str(tmp_dir), 
                        max_frames=8, threshold=30.0, 
                        max_size=args.max_image_size
                    )
                
                # 策略2: 人体检测帧（支持YOLO或UHD）
                human_frame_path = None
                if use_yolo and yolo_detector is not None:
                    # 使用 YOLO 检测人体
                    log_message(f"  [{idx+1}] 使用 YOLO 检测人体...")
                    human_result = find_human_frame_yolo(
                        file_path, model_type=args.yolo_model,
                        step_seconds=args.step_seconds,
                        max_retries=args.max_retries,
                        conf_threshold=args.yolo_conf
                    )
                    if human_result["found"]:
                        human_frame_path = human_result["frame_path"]
                        row["human_detected"] = "true"
                        row["detection_confidence"] = f"{human_result['max_confidence']:.2f}"
                        row["detection_timestamp"] = f"{human_result['timestamp']:.1f}s"
                        row["detection_method"] = "yolo"
                        log_message(f"  [{idx+1}] YOLO 检测到人体 @ {human_result['timestamp']:.1f}s, "
                                  f"置信度: {human_result['max_confidence']:.2f}")
                    else:
                        row["human_detected"] = "false"
                        row["detection_confidence"] = "0.00"
                        row["detection_timestamp"] = "N/A"
                        row["detection_method"] = "yolo"
                elif frame_selector_session is not None:
                    # 使用 UHD 检测人体
                    log_message(f"  [{idx+1}] 使用 UHD 检测人体...")
                    human_result = find_human_frame(
                        file_path, frame_selector_session,
                        step_seconds=args.step_seconds,
                        max_retries=args.max_retries,
                        conf_threshold=args.conf_threshold
                    )
                    if human_result["found"]:
                        human_frame_path = human_result["frame_path"]
                        row["human_detected"] = "true"
                        row["detection_confidence"] = f"{human_result['max_confidence']:.2f}"
                        row["detection_timestamp"] = f"{human_result['timestamp']:.1f}s"
                        row["detection_method"] = "uhd"
                        log_message(f"  [{idx+1}] UHD 检测到人体 @ {human_result['timestamp']:.1f}s, "
                                  f"置信度: {human_result['max_confidence']:.2f}")
                    else:
                        row["human_detected"] = "false"
                        row["detection_confidence"] = "0.00"
                        row["detection_timestamp"] = "N/A"
                        row["detection_method"] = "uhd"
                else:
                    row["human_detected"] = "N/A"
                    row["detection_confidence"] = "N/A"
                    row["detection_timestamp"] = "N/A"
                    row["detection_method"] = "none"
                
                # 策略3: 补充均匀采样帧
                all_frame_paths = list(keyframe_paths)
                if human_frame_path and human_frame_path not in all_frame_paths:
                    all_frame_paths.append(human_frame_path)
                
                # 如果帧数不足，补充均匀采样
                if len(all_frame_paths) < args.clip_frames:
                    n_extra = args.clip_frames - len(all_frame_paths)
                    extra_frames = extract_multiple_frames(
                        file_path, str(tmp_dir), n_frames=n_extra,
                        max_size=args.max_image_size
                    )
                    for fp in extra_frames:
                        if fp not in all_frame_paths:
                            all_frame_paths.append(fp)
                
                log_message(f"  [{idx+1}] 总共 {len(all_frame_paths)} 帧用于分析")
                
                # YOLO姿态分析（收集所有帧的姿态信息）
                if use_yolo and yolo_detector is not None:
                    log_message(f"  [{idx+1}] YOLO 姿态分析...")
                    for frame_path in all_frame_paths:
                        data = np.fromfile(frame_path, dtype=np.uint8)
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            pose_result = yolo_detector.estimate_pose(frame)
                            frame_data = {
                                "has_person": pose_result["has_person"],
                                "person_count": len(pose_result["poses"]) if pose_result["has_person"] else 0
                            }
                            if pose_result["has_person"]:
                                best_pose = max(pose_result["poses"], key=lambda x: x["avg_confidence"])
                                frame_data["keypoints"] = {kpt["name"]: {"x": kpt["x"], "y": kpt["y"], "conf": kpt["confidence"]} 
                                                          for kpt in best_pose["keypoints"] if kpt["visible"]}
                                frame_data["visible_count"] = best_pose["visible_count"]
                                frame_data["avg_confidence"] = best_pose["avg_confidence"]
                                frame_data["pose_analysis"] = analyze_pose_for_vlm(frame_data["keypoints"])
                            yolo_results.append(frame_data)
                        else:
                            yolo_results.append({"has_person": False})
                    
                    # 保存YOLO结果到JSON（延迟到selected_indices计算后）
                
                # 裁剪人体区域（用于embedding检测）
                human_crops = []
                if use_yolo and yolo_detector is not None and args.use_embedding_detection:
                    # 使用 YOLO 分割裁剪人体区域
                    log_message(f"  [{idx+1}] 使用 YOLO 裁剪人体区域...")
                    for frame_path in all_frame_paths:
                        # 读取图像
                        data = np.fromfile(frame_path, dtype=np.uint8)
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            if args.yolo_model == "segment":
                                # 使用实例分割
                                seg_result = yolo_detector.segment_persons(frame)
                                if seg_result["has_person"]:
                                    best_seg = max(seg_result["segments"], key=lambda x: x["area"])
                                    human_crops.append(best_seg["cropped"])
                                else:
                                    human_crops.append(None)
                            else:
                                # 使用检测+裁剪
                                crop_result = yolo_detector.detect_and_crop(frame, padding=0.15)
                                human_crops.append(crop_result["human_crop"])
                        else:
                            human_crops.append(None)
                    log_message(f"  [{idx+1}] 成功裁剪 {sum(1 for c in human_crops if c is not None)} 个人体区域")
                elif frame_selector_session is not None and args.use_embedding_detection:
                    # 使用 UHD 裁剪人体区域
                    log_message(f"  [{idx+1}] 使用 UHD 裁剪人体区域...")
                    for frame_path in all_frame_paths:
                        crop_result = detect_and_crop_human(
                            frame_path, frame_selector_session,
                            conf_threshold=args.conf_threshold, padding=0.15
                        )
                        human_crops.append(crop_result["human_crop"])
                    log_message(f"  [{idx+1}] 成功裁剪 {sum(1 for c in human_crops if c is not None)} 个人体区域")
                
                # CLIP 多帧分类（支持embedding检测）
                if use_clip and clip_classifier is not None and all_frame_paths:
                    # 对所有帧进行CLIP分类
                    change_result = clip_classifier.compare_frames(
                        all_frame_paths, threshold=args.clip_threshold,
                        multi_label=not args.single_label,
                        human_crops=human_crops if args.use_embedding_detection else None,
                        embedding_threshold=args.embedding_threshold
                    )
                    best_idx = change_result["best_frame_idx"]
                    best_tags = change_result["all_tags"][best_idx]
                    
                    # 记录相似度信息
                    if change_result.get("similarities"):
                        avg_sim = sum(change_result["similarities"]) / len(change_result["similarities"])
                        log_message(f"  [{idx+1}] 人体区域相似度: 平均={avg_sim:.3f}, "
                                  f"变化帧={change_result['changed_frames']}")
                    
                    # 记录 CLIP 结果
                    row["clip_clothing"] = best_tags["clothing"]["label_cn"]
                    row["clip_action"] = best_tags["action"]["label_cn"]
                    row["clip_hairstyle"] = best_tags["hairstyle"]["label_cn"]
                    row["clip_tags"] = best_tags["tags"]
                    row["clip_tags_json"] = json.dumps(best_tags["tags_json"], ensure_ascii=False)
                    row["clip_confidence"] = f"{best_tags['avg_confidence']:.2f}"
                    
                    if not change_result["has_significant_change"] and best_tags["avg_confidence"] >= args.clip_threshold:
                        # 静态内容 + 高置信度：跳过云端
                        row["vision_source"] = "clip_only"
                        row["vision_description"] = f"[CLIP] {best_tags['tags']}"
                        row["vision_keywords"] = best_tags["tags"]
                        if best_tags["tags"]:
                            kw_list = [k for k in best_tags["tags"].split("_") if k.strip()][:5]
                            if kw_list:
                                row["final_name"] = f"[{'_'.join(kw_list)}]_{original}"
                        log_message(f"  [{idx+1}] CLIP 视频静态分类: {best_tags['tags']} "
                                  f"(置信度: {best_tags['avg_confidence']:.2f})")
                        # 清理临时帧
                        for fp in all_frame_paths:
                            if os.path.exists(fp):
                                os.remove(fp)
                        continue
                    else:
                        # 动态内容或低置信度：使用智能帧选择
                        row["vision_source"] = "clip+cloud"
                        log_message(f"  [{idx+1}] CLIP 最佳帧置信度 {best_tags['avg_confidence']:.2f}，调云端 VLM")
                        
                        # 使用YOLO智能帧选择（如果有YOLO结果）
                        if yolo_results:
                            selected_indices = select_best_frames_by_pose(yolo_results, max_frames=args.vlm_frames)
                            vlm_frames = [all_frame_paths[i] for i in selected_indices if i < len(all_frame_paths)]
                            log_message(f"  [{idx+1}] YOLO智能选择 {len(vlm_frames)} 帧")
                        else:
                            # 降级到CLIP置信度选择
                            vlm_frames = []
                            vlm_frame_count = min(args.vlm_frames, len(all_frame_paths))
                            frame_confidences = [(i, all_tags["avg_confidence"]) 
                                                for i, all_tags in enumerate(change_result["all_tags"])]
                            frame_confidences.sort(key=lambda x: x[1], reverse=True)
                            
                            for i in range(vlm_frame_count):
                                if i < len(frame_confidences):
                                    frame_idx = frame_confidences[i][0]
                                    vlm_frames.append(all_frame_paths[frame_idx])
                        
                        # 如果有明确的人体检测帧，确保包含
                        if human_frame_path and human_frame_path not in vlm_frames:
                            vlm_frames.insert(0, human_frame_path)
                            vlm_frames = vlm_frames[:args.vlm_frames]
                        
                        log_message(f"  [{idx+1}] 送入VLM {len(vlm_frames)} 帧")
                        
                        # 清理不使用的帧
                        for fp in all_frame_paths:
                            if fp not in vlm_frames and os.path.exists(fp):
                                os.remove(fp)
                elif not use_clip or row.get("vision_source", "") == "":
                    # CLIP未启用，使用智能帧选择
                    row["vision_source"] = "cloud_only"
                    
                    # 使用YOLO智能帧选择
                    if yolo_results:
                        selected_indices = select_best_frames_by_pose(yolo_results, max_frames=args.vlm_frames)
                        vlm_frames = [all_frame_paths[i] for i in selected_indices if i < len(all_frame_paths)]
                        log_message(f"  [{idx+1}] YOLO智能选择 {len(vlm_frames)} 帧")
                    elif human_frame_path:
                        vlm_frames = [human_frame_path]
                    else:
                        if not extract_frame(file_path, tmp_img, args.timestamp, max_size=args.max_image_size):
                            log_message(f"  [{idx+1}] 帧提取失败: {original[:30]}")
                            continue
                        vlm_frames = [tmp_img]
            else:
                log_message(f"  [{idx+1}] 不支持的文件格式: {Path(file_path).suffix}")
                continue

            # 调用视觉 API（支持多帧）
            # 过滤掉不存在的帧文件
            valid_vlm_frames = [fp for fp in vlm_frames if os.path.exists(fp)]
            
            if not valid_vlm_frames:
                log_message(f"  [{idx+1}] 所有帧文件已被清理，跳过")
                continue
            
            # 构建YOLO上下文
            yolo_context = None
            if yolo_results and use_yolo:
                # 获取选中帧的索引
                selected_indices = [i for i, fp in enumerate(all_frame_paths) if fp in valid_vlm_frames]
                yolo_context = build_yolo_context(yolo_results, selected_indices)
                if yolo_context:
                    log_message(f"  [{idx+1}] 已构建YOLO上下文")
                
                # 保存YOLO结果到JSON
                save_yolo_results(str(output_path.parent), original, yolo_results, selected_indices)
            
            if is_video_file(file_path) and len(valid_vlm_frames) > 1:
                # 多帧模式
                images_b64 = [image_to_base64(fp, max_size=args.max_image_size) for fp in valid_vlm_frames]
                result = call_vision_api(images_b64, original, provider=provider,
                                        model=model, api_key=api_key,
                                        retries=args.retries, n_frames=len(valid_vlm_frames),
                                        yolo_context=yolo_context)
            else:
                # 单帧模式
                tmp_img = valid_vlm_frames[0]
                image_b64 = image_to_base64(tmp_img, max_size=args.max_image_size)
                result = call_vision_api(image_b64, original, provider=provider,
                                        model=model, api_key=api_key,
                                        retries=args.retries, yolo_context=yolo_context)

            description = result.get("description", "")
            keywords = result.get("keywords", "")

            row["vision_description"] = description
            row["vision_keywords"] = keywords

            # ===== 标签反馈：从 VLM 结果中学习新标签 =====
            if tag_stats and keywords and not description.startswith("[ERROR]"):
                tag_stats.update_from_vlm(keywords)
                # 如果 CLIP 已启用，更新后刷新 embedding
                if use_clip and clip_classifier is not None:
                    clip_classifier.reload_embeddings()

            if description.startswith("[ERROR]"):
                log_message(f"  [{idx+1}] {original[:30]} -> API 错误，已标记")
                if os.path.exists(tmp_img) and tmp_img != file_path:
                    os.remove(tmp_img)
                continue

            if keywords:
                kw_list = [k.strip() for k in keywords.split(',') if k.strip()][:5]
                if kw_list:
                    prefix = "_".join(kw_list)
                    row["final_name"] = f"[{prefix}]_{original}"
                else:
                    row["final_name"] = row.get("proposed_title", original)
            else:
                row["final_name"] = row.get("proposed_title", original)

            log_message(f"  [{idx+1}] {original[:30]}")
            log_message(f"      描述: {description[:80]}")
            log_message(f"      关键词: {keywords}")

            # 清理临时文件
            if os.path.exists(tmp_img) and tmp_img != file_path:
                os.remove(tmp_img)
            
            # 每处理完一个视频立即写入CSV
            if not args.dry_run:
                save_rows_to_csv(rows, fieldnames, str(output_path))
                log_message(f"  [保存] 已写入CSV")

        if batch_start + batch_size < len(pending):
            time.sleep(args.delay)

    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    if args.dry_run:
        print("模拟模式结束，未写入文件")
        return

    # 最终保存（确保所有数据已写入）
    save_rows_to_csv(rows, fieldnames, str(output_path))

    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
