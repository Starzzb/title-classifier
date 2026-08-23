"""视觉识别模块 - 视频全面分析版"""

import csv
import json
import os
import re
import time
import logging
import tempfile
import threading
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple

import cv2
import numpy as np

from ..providers import get_provider_config, get_api_key, call_vision_api
from ..detectors import YOLODetector, CLIPClassifier
from ..utils.video import get_video_duration, extract_frames_cv2
from ..utils.image import compress_image, image_to_base64, image_array_to_base64
from ..utils.stats import TagStatistics
from ..utils.prompt_loader import get_prompt

logger = logging.getLogger(__name__)

# GPU锁：CUDA不支持多线程并发推理，需要串行化
_gpu_lock = threading.Lock()


class VisionProcessor:
    """视觉处理器 - 支持视频全面分析"""

    def __init__(
        self,
        config: dict = None,
        provider: str = None,
        use_yolo: bool = None,
        yolo_model: str = None,
        yolo_models: List[str] = None,
        yolo_conf: float = None,
        use_clip: bool = None,
        clip_threshold: float = None,
        max_image_size: int = None,
        vlm_frames: int = None,
        analysis_step: float = None,
        max_sample_frames: int = None,
        debug_dir: str = None,
        covers_dir: str = None,
        db_store=None,
        device: str = None,
        motion_detection: bool = None,
        motion_threshold: float = None,
        motion_min_interval: float = None,
        backend: str = None,
        use_scene_detection: bool = None,
        scene_threshold: float = None,
        max_scenes: int = None,
        frames_per_scene: int = None,
    ):
        from ..utils.config import get_config_value
        if config is None:
            config = {}
        cv = lambda key, fallback: get_config_value(config, key, fallback)

        self.provider = provider or cv("providers.stage_providers.vision", "gcli")
        self.use_yolo = use_yolo if use_yolo is not None else cv("yolo.comprehensive.enabled", False)
        self.yolo_model = yolo_model or cv("yolo.model_type", "pose")
        self.yolo_models = yolo_models or cv("yolo.comprehensive.models", ["pose"])
        self.yolo_conf = yolo_conf if yolo_conf is not None else cv("yolo.confidence", 0.5)
        self.use_clip = use_clip if use_clip is not None else False
        self.clip_threshold = clip_threshold if clip_threshold is not None else cv("clip.threshold", 0.25)
        self.max_image_size = max_image_size if max_image_size is not None else cv("vision.max_image_size", 640)
        self.vlm_frames = vlm_frames if vlm_frames is not None else cv("vision.vlm_frames", 10)
        self.analysis_step = analysis_step if analysis_step is not None else cv("vision.analysis_step", 5.0)
        self.max_sample_frames = max_sample_frames if max_sample_frames is not None else cv("vision.max_sample_frames", 50)
        self.debug_dir = debug_dir
        self.covers_dir = covers_dir
        self.db_store = db_store
        self.device = self._resolve_device(device or cv("general.device", "cpu"))
        self.motion_detection = motion_detection if motion_detection is not None else cv("vision.motion_detection", True)
        self.motion_threshold = motion_threshold if motion_threshold is not None else cv("vision.motion_threshold", 8.0)
        self.motion_min_interval = motion_min_interval if motion_min_interval is not None else cv("vision.motion_min_interval", 5.0)
        self.backend = backend or cv("yolo.backend", "auto")
        self.use_scene_detection = use_scene_detection if use_scene_detection is not None else cv("scene_detection.enabled", True)
        self.scene_threshold = scene_threshold if scene_threshold is not None else cv("scene_detection.threshold", 0.3)
        self.max_scenes = max_scenes if max_scenes is not None else cv("scene_detection.max_scenes", 10)
        self.frames_per_scene = frames_per_scene if frames_per_scene is not None else cv("scene_detection.frames_per_scene", 10)

        self.provider_config = get_provider_config(provider)
        self.model = self.provider_config.get("default_model", "") if self.provider_config else ""
        self.api_key = get_api_key(provider)

        self.yolo_detector = None
        self.clip_classifier = None
        self.tag_stats = None

    def _resolve_device(self, device: str) -> str:
        """解析推理设备：auto/cuda/cpu"""
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA不可用，回退到CPU")
                    return "cpu"
                return "cuda"
            except ImportError:
                logger.warning("PyTorch未安装，使用CPU")
                return "cpu"
        # auto模式：自动检测
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_mem >= 4:
                    logger.info(f"检测到GPU ({torch.cuda.get_device_name(0)}, {gpu_mem:.1f}GB)，使用CUDA推理")
                    return "cuda"
                else:
                    logger.warning(f"显存不足 ({gpu_mem:.1f}GB < 4GB)，使用CPU推理")
                    return "cpu"
            return "cpu"
        except ImportError:
            return "cpu"

    def initialize(self) -> bool:
        """初始化检测器"""
        self.tag_stats = TagStatistics()

        # 始终使用YOLO检测器
        self.yolo_detector = YOLODetector(
            model_types=self.yolo_models,
            confidence=self.yolo_conf,
            device=self.device,
            backend=self.backend,
        )
        if not self.yolo_detector.load_model():
            logger.error("YOLO模型加载失败")
            return False

        # 记录后端信息
        if hasattr(self.yolo_detector, '_backend_type'):
            logger.info(f"YOLO后端: {self.yolo_detector._backend_type}")

        if self.use_clip:
            self.clip_classifier = CLIPClassifier(tag_stats=self.tag_stats, device=self.device)
            if not self.clip_classifier.load_model():
                logger.warning("CLIP模型加载失败，将使用纯云端VLM")
                self.use_clip = False
                self.clip_classifier = None

        return True

    def _detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                       threshold: float = None) -> Tuple[bool, float]:
        """
        轻量级运动检测 - 帧差法
        
        Args:
            prev_frame: 上一帧 (BGR)
            curr_frame: 当前帧 (BGR)
            threshold: 变化像素比例阈值（%），低于此值认为是静止画面
            
        Returns:
            (has_motion, change_ratio): 是否有运动，变化像素比例（0-100）
        """
        if threshold is None:
            threshold = self.motion_threshold
        
        try:
            # 1. 转灰度（兼容已灰度的帧）
            if len(prev_frame.shape) == 2:
                gray_prev = prev_frame
            else:
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if len(curr_frame.shape) == 2:
                gray_curr = curr_frame
            else:
                gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 2. 高斯模糊降噪（减少压缩伪影影响）
            gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
            gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)
            
            # 3. 计算绝对差值
            frame_diff = cv2.absdiff(gray_prev, gray_curr)
            
            # 4. 二值化（超过阈值的像素设为255）
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # 5. 计算变化像素比例
            change_ratio = (np.count_nonzero(thresh) / thresh.size) * 100
            
            # 6. 判断是否有运动
            has_motion = change_ratio > threshold
            
            return has_motion, change_ratio
            
        except Exception as e:
            logger.warning(f"运动检测失败: {e}")
            # 出错时默认有运动，避免误跳帧
            return True, 100.0

    def _should_skip_inference(
        self,
        prev_frame_gray: Optional[np.ndarray],
        curr_frame: np.ndarray,
        ts: float,
        last_forced_ts: float,
    ) -> bool:
        """
        判定当前帧是否可以跳过 YOLO 推理（静止画面复用上一帧结果）。

        仅当距上次强制推理不足 motion_min_interval 秒时才允许跳过，
        保证长时间静止场景也会周期性强制推理，防止状态漂移。
        """
        if not self.motion_detection:
            return False
        if prev_frame_gray is None:
            return False
        if ts - last_forced_ts >= self.motion_min_interval:
            return False  # 达到最小强制间隔，必须推理
        has_motion, _ = self._detect_motion(prev_frame_gray, curr_frame)
        return not has_motion

    def process_video(self, video_path: str, title: str, audio_context: str = "", subtitle_segments: List[Dict] = None) -> Dict:
        """处理视频 - 全面分析模式"""
        duration = get_video_duration(video_path)
        logger.info(f"视频模式: {title[:30]} (时长: {duration:.1f}s)")

        if audio_context:
            logger.info(f"检测到音频上下文，长度: {len(audio_context)} 字符")

        # 场景检测模式：长视频按场景分段分析后合并
        if self.use_scene_detection and duration >= 120:
            return self._process_video_by_scenes(video_path, title, duration, audio_context, subtitle_segments)

        # 默认使用YOLO全面分析模式
        return self._process_video_comprehensive(video_path, title, duration, audio_context, subtitle_segments)

    def _process_video_comprehensive(self, video_path: str, title: str, duration: float, audio_context: str = "", subtitle_segments: List[Dict] = None) -> Dict:
        """视频全面分析"""
        timing = {}  # 分步耗时追踪
        t_total_start = time.perf_counter()

        mode_name = "全面分析" if len(self.yolo_models) > 1 else "基础"
        logger.info(f"启动YOLO{mode_name}模式，采样间隔: {self.analysis_step}秒，模型: {self.yolo_models}")

        # 创建调试目录
        debug_subdir = None
        if self.debug_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = re.sub(r'[<>:"/\\|?*]', '_', Path(video_path).stem)[:30].rstrip(' .')
            debug_subdir = Path(self.debug_dir) / f"{timestamp}_{video_name}"
            debug_subdir.mkdir(parents=True, exist_ok=True)
            (debug_subdir / "detection").mkdir(exist_ok=True)
            (debug_subdir / "vlm_frames").mkdir(exist_ok=True)
            logger.info(f"调试目录: {debug_subdir}")

        # 1. 全面扫描视频（YOLO 推理）
        t1 = time.perf_counter()
        video_analysis = self._analyze_video_comprehensive(video_path, duration)
        timing["yolo_total"] = time.perf_counter() - t1

        # 2. CLIP 差异度评分
        clip_diff_scores = None
        if self.use_clip and self.clip_classifier and self.clip_classifier._loaded:
            t_clip = time.perf_counter()
            try:
                all_frames = video_analysis["frames"]
                decoded_all = video_analysis.get("decoded_frames", [])
                # 传完整对齐列表（含 None），保证分数索引 == 帧索引
                frames_for_diff = list(decoded_all) if decoded_all else \
                    [cv2.imread(f) for f in all_frames]
                if frames_for_diff:
                    clip_diff_scores = self.clip_classifier.compute_frame_diff_scores(frames_for_diff)
                    timing["clip_diff"] = time.perf_counter() - t_clip
                    logger.info(f"CLIP 差异度评分完成: min={min(clip_diff_scores):.3f}, max={max(clip_diff_scores):.3f}, 耗时={timing['clip_diff']:.2f}s")
            except Exception as e:
                timing["clip_diff"] = time.perf_counter() - t_clip
                logger.warning(f"CLIP 差异度评分失败: {e}, 耗时={timing['clip_diff']:.2f}s")
                clip_diff_scores = None

        # 3. 帧选择
        t2 = time.perf_counter()
        selected_indices = self._select_representative_frames(
            video_analysis["timeline"], max_frames=self.vlm_frames, clip_diff_scores=clip_diff_scores
        )
        selected_frames = video_analysis["frames"]
        timing["frame_selection"] = time.perf_counter() - t2

        logger.info(f"帧选择完成: 总帧数={len(selected_frames)}, 选中帧数={len(selected_indices)}, 耗时={timing['frame_selection']:.3f}s")

        # 3. 生成视频摘要
        video_summary = self._generate_video_summary(video_analysis["timeline"], duration)

        # 4. 为每帧生成描述
        frame_descriptions = self._generate_frame_descriptions(
            video_analysis["timeline"], selected_indices
        )

        # 5. 构建全面上下文
        comprehensive_context = self._build_comprehensive_context(
            video_summary, frame_descriptions, len(selected_indices)
        )

        # 5.5 构建每帧对应的字幕上下文
        per_frame_subtitle = ""
        if subtitle_segments:
            frame_timestamps = [
                video_analysis["timeline"][i]["timestamp"]
                for i in selected_indices
                if i < len(video_analysis["timeline"])
            ]
            per_frame_subtitle = self._build_per_frame_subtitle_context(frame_timestamps, subtitle_segments)

        # 5.6 构建差异度提示（基于选中帧的 VLM 编号）
        diff_hint = ""
        if clip_diff_scores:
            # 取选中帧的差异度分数，映射到 VLM 编号（1-based）
            selected_diff = []
            for vlm_idx, orig_idx in enumerate(selected_indices):
                if orig_idx < len(clip_diff_scores):
                    selected_diff.append((vlm_idx + 1, clip_diff_scores[orig_idx]))

            # 按差异度排序，取前 1/3
            selected_diff.sort(key=lambda x: x[1], reverse=True)
            top_k = max(1, len(selected_diff) // 3)
            top_frames = [str(vlm_num) for vlm_num, _ in selected_diff[:top_k]]

            diff_hint = (
                f"第 {', '.join(top_frames)} 帧与其他帧差异最大（场景变化最明显），"
                f"请重点分析这些帧中的穿着、动作和场景细节。"
            )

        # 6. 调用VLM（传入音频上下文和每帧字幕）
        frames_for_vlm = [selected_frames[i] for i in selected_indices if i < len(selected_frames)]
        logger.info(f"调用VLM: {len(frames_for_vlm)}帧")

        # 复用内存解码帧（缺失时回退到路径），result/debug 仍保留路径列表
        decoded_all = video_analysis.get("decoded_frames", [])
        vlm_input = [
            decoded_all[i] if i < len(decoded_all) and decoded_all[i] is not None else selected_frames[i]
            for i in selected_indices
        ]
        
        # 保存调试数据 - 检测结果
        if debug_subdir:
            self._save_detection_debug(video_analysis, debug_subdir)

        # 构建prompt（用于调试）
        prompt = self._build_comprehensive_prompt(title, len(frames_for_vlm), comprehensive_context, audio_context, per_frame_subtitle, diff_hint=diff_hint)

        # 保存调试数据 - VLM输入帧和prompt
        if debug_subdir:
            self._save_vlm_debug(frames_for_vlm, prompt, debug_subdir)

        t_vlm = time.perf_counter()
        result = self._call_vlm_comprehensive(
            vlm_input,
            title,
            comprehensive_context,
            audio_context,
            per_frame_subtitle,
            diff_hint=diff_hint,
        )
        timing["vlm_api"] = time.perf_counter() - t_vlm

        logger.info(f"VLM结果: 描述='{result.get('description', '')[:50]}...', 关键词='{result.get('keywords', '')[:50]}...', 耗时={timing['vlm_api']:.2f}s")

        # 7. 保存分析结果
        # 计算选中帧的时间戳
        selected_timestamps = [
            video_analysis["timeline"][i]["timestamp"]
            for i in selected_indices
            if i < len(video_analysis["timeline"])
        ]

        # 保存调试数据 - VLM响应和汇总
        if debug_subdir:
            self._save_debug_summary(result, video_summary, debug_subdir,
                                     frame_timestamps=selected_timestamps)

        analysis_result = {
            "description": result.get("description", ""),
            "keywords": result.get("keywords", ""),
            "video_summary": video_summary,
            "selected_frames": len(selected_indices),
            "total_analyzed": len(video_analysis["timeline"]),
            "pose_changes": len(video_summary.get("pose_changes", [])),
            "person_ratio": video_summary.get("person_ratio", 0),
            "frames_for_vlm": frames_for_vlm,
            "frame_timestamps": selected_timestamps,
        }

        # 记录调试目录路径
        if debug_subdir:
            analysis_result["debug_dir"] = str(debug_subdir)

        # 耗时汇总
        timing["total"] = time.perf_counter() - t_total_start
        analysis_result["timing"] = timing

        # 日志输出耗时汇总
        motion_skipped = video_analysis.get("motion_skipped_count", 0)
        yolo_inference = video_analysis.get("yolo_inference_count", len(video_analysis.get("timeline", [])))
        logger.info(
            f"处理完成: 总耗时={timing['total']:.2f}s | "
            f"YOLO={timing.get('yolo_total', 0):.2f}s({yolo_inference}帧推理,{motion_skipped}帧跳过) | "
            f"CLIP={timing.get('clip_diff', 0):.2f}s | "
            f"帧选择={timing.get('frame_selection', 0):.3f}s | "
            f"VLM={timing.get('vlm_api', 0):.2f}s"
        )

        # Debug 模式保存耗时数据
        if debug_subdir:
            timing_path = debug_subdir / "timing.json"
            with open(timing_path, "w", encoding="utf-8") as f:
                json.dump(timing, f, indent=2, ensure_ascii=False)

        return analysis_result

    def _process_video_by_scenes(self, video_path: str, title: str, duration: float, audio_context: str = "", subtitle_segments: List[Dict] = None) -> Dict:
        """场景分段分析模式：检测场景 → 逐段分析 → 合并结果"""
        from .scene_detector import get_segments
        from ..utils.prompt_loader import get_prompt

        timing = {}
        t_total_start = time.perf_counter()

        # 创建调试目录
        debug_subdir = None
        if self.debug_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = re.sub(r'[<>:"/\\|?*]', '_', Path(video_path).stem)[:30].rstrip(' .')
            debug_subdir = Path(self.debug_dir) / f"{timestamp}_{video_name}"
            debug_subdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"场景模式调试目录: {debug_subdir}")

        segments = get_segments(video_path, duration, self.scene_threshold, self.max_scenes)
        logger.info(f"场景分段数: {len(segments)}, 将逐段分析后合并")

        scene_results = []

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_duration = seg_end - seg_start
            logger.info(f"[场景 {seg_idx+1}/{len(segments)}] {seg_start:.1f}s - {seg_end:.1f}s (时长: {seg_duration:.1f}s)")

            t1 = time.perf_counter()
            seg_analysis = self._analyze_video_segment(video_path, seg_start, seg_end, seg_idx, duration)
            timing[f"scene_{seg_idx}_yolo"] = time.perf_counter() - t1

            if not seg_analysis or "error" in seg_analysis:
                logger.warning(f"[场景 {seg_idx+1}] 分析失败，跳过")
                continue

            # 保存每段的YOLO检测结果到调试目录
            if debug_subdir:
                seg_debug_dir = debug_subdir / f"scene_{seg_idx}"
                seg_debug_dir.mkdir(exist_ok=True)
                (seg_debug_dir / "detection").mkdir(exist_ok=True)
                self._save_detection_debug(seg_analysis, seg_debug_dir)

            t2 = time.perf_counter()
            seg_result = self._call_vlm_comprehensive(
                seg_analysis["decoded"],
                f"{title}[场景{seg_idx+1}]",
                seg_analysis["context"],
                audio_context,
            )
            timing[f"scene_{seg_idx}_vlm"] = time.perf_counter() - t2

            # 保存每段的VLM调试数据
            if debug_subdir:
                seg_debug_dir = debug_subdir / f"scene_{seg_idx}"
                (seg_debug_dir / "vlm_frames").mkdir(exist_ok=True)
                prompt = self._build_comprehensive_prompt(
                    f"{title}[场景{seg_idx+1}]",
                    len(seg_analysis["frames_for_vlm"]),
                    seg_analysis["context"],
                    audio_context,
                )
                self._save_vlm_debug(seg_analysis["frames_for_vlm"], prompt, seg_debug_dir)
                # 保存该段VLM响应
                response_text = (
                    f"描述: {seg_result.get('description', '')}\n"
                    f"关键词: {seg_result.get('keywords', '')}\n"
                )
                (seg_debug_dir / "vlm_response.txt").write_text(response_text, encoding="utf-8")

            scene_desc = seg_result.get("description", "") or seg_result.get("error", "分析失败")
            scene_kw = seg_result.get("keywords", "")
            scene_results.append({
                "index": seg_idx,
                "start": seg_start,
                "end": seg_end,
                "duration": seg_duration,
                "description": scene_desc,
                "keywords": scene_kw,
                "frames": len(seg_analysis["frames_for_vlm"]),
            })
            logger.info(f"  → 描述: {scene_desc[:60]}... 关键词: {scene_kw[:60]}...")

        if not scene_results:
            return {"error": "所有场景分析失败"}

        merged = self._merge_scene_descriptions(scene_results, title)
        timing["merge_vlm"] = merged.get("_timing", 0)
        timing["total"] = time.perf_counter() - t_total_start

        # 保存合并结果到调试目录
        if debug_subdir:
            video_summary = {
                "has_person": True,
                "duration": duration,
                "scenes": len(segments),
                "scene_results": scene_results,
            }
            self._save_debug_summary(merged, video_summary, debug_subdir)
            # 保存时间统计
            (debug_subdir / "timing.json").write_text(
                json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        logger.info(
            f"场景分析完成: {len(segments)}段, "
            f"总耗时={timing['total']:.2f}s, "
            f"合并VLM={timing.get('merge_vlm', 0):.2f}s"
        )

        result = {
            "description": merged.get("description", ""),
            "keywords": merged.get("keywords", ""),
            "video_summary": {
                "has_person": True,
                "duration": duration,
                "scenes": len(segments),
                "scene_results": scene_results,
            },
            "selected_frames": sum(s["frames"] for s in scene_results),
            "total_analyzed": sum(s["frames"] for s in scene_results),
            "timing": timing,
        }
        if debug_subdir:
            result["debug_dir"] = str(debug_subdir)
        return result

    def _analyze_video_segment(self, video_path: str, seg_start: float, seg_end: float, seg_idx: int, video_duration: float = None) -> Dict:
        """分析单个场景段：等距取帧 → YOLO 分析"""
        import cv2

        tmp_dir = Path("logs/_vision_tmp") / f"scene_{seg_idx}_{Path(video_path).stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        duration = seg_end - seg_start
        n_frames = min(self.frames_per_scene, max(3, int(duration / 0.5)))
        timestamps = np.linspace(seg_start, seg_end, n_frames)

        # 一次性批量抽帧（单次解码）
        extracted_paths = extract_frames_cv2(
            video_path, str(tmp_dir), [float(t) for t in timestamps], max_size=400
        )

        frames = []
        decoded = []
        timeline = []
        prev_frame_gray = None
        prev_result = None
        last_forced_timestamp = -float('inf')

        for i, (ts, frame_path) in enumerate(zip(timestamps, extracted_paths)):
            if frame_path is None:
                continue

            data = np.fromfile(frame_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frames.append(frame_path)
            decoded.append(frame)

            should_skip = self._should_skip_inference(prev_frame_gray, frame, ts, last_forced_timestamp)

            if should_skip and prev_result is not None:
                entry = prev_result.copy()
                entry["timestamp"] = ts
                entry["frame_path"] = frame_path
                entry["index"] = i
                entry["motion_skipped"] = True
                timeline.append(entry)
                prev_frame_gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue

            if self.yolo_detector:
                if self.device == "cuda":
                    with _gpu_lock:
                        result = self.yolo_detector.analyze_comprehensive(frame)
                else:
                    result = self.yolo_detector.analyze_comprehensive(frame)

                entry = {
                    "index": i,
                    "timestamp": ts,
                    "frame_path": frame_path,
                    "has_person": result.get("has_person", False),
                    "confidence": result.get("confidence", 0),
                    "motion_skipped": False,
                }
                timeline.append(entry)
                prev_result = entry
                last_forced_timestamp = ts

            prev_frame_gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        selected_indices = self._select_representative_frames(timeline, max_frames=self.frames_per_scene)
        selected_frames = [frames[i] for i in selected_indices if i < len(frames)]
        summary = self._generate_video_summary(timeline, duration)

        has_person_text = f"包含人物: {'是' if summary.get('has_person', False) else '否'}"
        context = (
            f"【场景 {seg_idx+1} 分析】\n"
            f"时间范围: {seg_start:.1f}s - {seg_end:.1f}s\n"
            f"总帧数: {len(timeline)}, 选中VLM帧: {len(selected_frames)}\n"
            f"{has_person_text}\n"
            f"主要姿态: {summary.get('main_pose', '未知')}"
        )

        return {
            "frames_for_vlm": selected_frames,
            "context": context,
            "frames": frames,
            "decoded": decoded,
            "timeline": timeline,
        }

    def _merge_scene_descriptions(self, scene_results: List[Dict], title: str) -> Dict:
        """合并所有场景描述为一组最终结果"""
        from ..providers import call_text_api
        from ..utils.prompt_loader import get_prompt

        t_start = time.perf_counter()

        scenes_text = ""
        for s in scene_results:
            scenes_text += (
                f"[场景 {s['index'] + 1}] ({s['start']:.1f}s - {s['end']:.1f}s)\n"
                f"描述: {s.get('description', '')}\n"
                f"关键词: {s.get('keywords', '')}\n\n"
            )

        system_header = get_prompt("vision_scene_merge", "system_header")
        task_instruction = get_prompt("vision_scene_merge", "task_instruction")
        output_format = get_prompt("vision_scene_merge", "output_format")

        prompt = (
            f"{system_header}\n\n"
            f"{task_instruction}\n\n"
            f"视频标题: {title}\n\n"
            f"{scenes_text}\n"
            f"{output_format}"
        )

        result = call_text_api(self.provider, prompt, model=self.model, api_key=self.api_key)

        timing = time.perf_counter() - t_start

        if not result or result.startswith("[ERROR]"):
            logger.error(f"场景合并VLM失败: {result[:100] if result else '空响应'}")
            descs = "；".join(s.get("description", "") for s in scene_results)
            kws = "，".join(s.get("keywords", "") for s in scene_results)
            return {
                "description": descs[:500],
                "keywords": kws[:500],
                "_timing": timing,
            }

        parsed = self._parse_vision_response(result)
        parsed["_timing"] = timing
        return parsed

    def _analyze_video_comprehensive(self, video_path: str, duration: float) -> Dict:
        """全面分析视频 - 高密度采样，使用多个YOLO模型，支持运动检测跳帧"""
        tmp_dir = Path("logs/_vision_tmp") / Path(video_path).stem.rstrip(" .")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        timeline = []
        decoded_frames = []

        # 计算采样时间点
        timestamps = np.arange(0, duration, self.analysis_step)
        if len(timestamps) > self.max_sample_frames:  # 限制最大采样数
            timestamps = np.linspace(0, duration, self.max_sample_frames)

        logger.info(f"YOLO分析: {len(timestamps)}个采样点, 模型: {self.yolo_models}")
        if self.motion_detection:
            logger.info(f"运动检测已启用: 阈值={self.motion_threshold}%, 最小强制间隔={self.motion_min_interval}s")

        # CUDA预热：第一次推理会编译kernel，耗时较长
        if self.device == "cuda":
            try:
                import torch
                logger.info(f"CUDA显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
                # 用空tensor预热CUDA
                _ = torch.zeros(1, device="cuda")
                del _
                torch.cuda.empty_cache()
                logger.info("CUDA预热完成")
            except Exception as e:
                logger.warning(f"CUDA预热失败: {e}")

        # 一次性批量抽帧（单次解码，替代循环内逐帧 ffmpeg 子进程）
        extracted_paths = extract_frames_cv2(
            video_path, str(tmp_dir), [float(t) for t in timestamps], max_size=400
        )

        # 运动检测相关变量
        prev_frame_gray = None
        prev_result = None
        last_forced_timestamp = -float('inf')  # 上次强制推理的时间戳
        motion_skipped_count = 0

        for i, (ts, frame_path) in enumerate(zip(timestamps, extracted_paths)):
            if frame_path is None:
                logger.debug(f"帧{i}: 提取失败（跳过）")
                continue

            # 全链路唯一一次解码（CLIP/VLM 均复用该内存数组）
            data = np.fromfile(frame_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                logger.debug(f"帧{i}: 解码失败（跳过）")
                continue
            frames.append(frame_path)
            decoded_frames.append(frame)
            logger.debug(f"[DEBUG] 帧{i}: 解码完成，开始YOLO推理")

            # 运动检测：判断是否可以跳过YOLO推理
            should_skip = self._should_skip_inference(prev_frame_gray, frame, ts, last_forced_timestamp)
            if should_skip:
                motion_skipped_count += 1

            if should_skip and prev_result is not None:
                # 复用上一帧结果，更新时间戳和帧路径
                timeline_entry = prev_result.copy()
                timeline_entry["timestamp"] = ts
                timeline_entry["frame_path"] = frame_path
                timeline_entry["index"] = i
                timeline_entry["motion_skipped"] = True
                timeline.append(timeline_entry)
                
                # 更新上一帧灰度图（用于下一次运动检测）
                if len(frame.shape) == 2:
                    prev_frame_gray = frame
                else:
                    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue

            # 有运动或首次帧，执行YOLO全面分析
            if self.yolo_detector:
                logger.debug(f"[DEBUG] 帧{i}: 开始analyze_comprehensive")
                # CUDA推理需要串行化（GPU不支持多线程并发推理）
                if self.device == "cuda":
                    with _gpu_lock:
                        comprehensive_result = self.yolo_detector.analyze_comprehensive(frame)
                else:
                    comprehensive_result = self.yolo_detector.analyze_comprehensive(frame)
                logger.debug(f"[DEBUG] 帧{i}: analyze_comprehensive完成")

                timeline_entry = {
                    "index": i,
                    "timestamp": ts,
                    "frame_path": frame_path,
                    "has_person": comprehensive_result.get("has_person", False),
                    "confidence": comprehensive_result.get("confidence", 0),
                    "models_used": comprehensive_result.get("models_used", []),
                    "vote_count": comprehensive_result.get("merged", {}).get("vote_count", 0),
                    "motion_skipped": False,
                }

                # 保存原始模型输出（用于调试）
                timeline_entry["raw_detection"] = comprehensive_result.get("detection")
                timeline_entry["raw_pose"] = comprehensive_result.get("pose")
                timeline_entry["raw_segment"] = comprehensive_result.get("segment")
                timeline_entry["merged"] = comprehensive_result.get("merged")

                # 提取姿态信息
                pose_result = comprehensive_result.get("pose")
                if pose_result and pose_result.get("has_person") and pose_result.get("poses"):
                    best_pose = pose_result["poses"][0]
                    timeline_entry["pose_analysis"] = best_pose.get("pose_analysis", [])
                    timeline_entry["visible_keypoints"] = best_pose.get("visible_count", 0)
                    timeline_entry["keypoints"] = best_pose.get("keypoints", {})
                    timeline_entry["pose_bbox"] = best_pose.get("bbox")
                    timeline_entry["pose_avg_confidence"] = best_pose.get("avg_confidence", 0)
                else:
                    timeline_entry["pose_analysis"] = []
                    timeline_entry["visible_keypoints"] = 0
                    timeline_entry["keypoints"] = {}
                    timeline_entry["pose_bbox"] = None
                    timeline_entry["pose_avg_confidence"] = 0

                # 提取检测信息
                detection_result = comprehensive_result.get("detection")
                if detection_result and detection_result.get("has_person"):
                    timeline_entry["detection_details"] = detection_result.get("persons", [])
                    timeline_entry["detection_max_confidence"] = detection_result.get("max_confidence", 0)
                else:
                    timeline_entry["detection_details"] = []
                    timeline_entry["detection_max_confidence"] = 0

                # 提取分割信息
                segment_result = comprehensive_result.get("segment")
                if segment_result and segment_result.get("has_person"):
                    timeline_entry["segment_details"] = segment_result.get("segments", [])
                    timeline_entry["segment_max_confidence"] = segment_result.get("max_confidence", 0)
                    # 提取穿着分析
                    if segment_result.get("segments"):
                        best_segment = segment_result["segments"][0]
                        timeline_entry["wearing_analysis"] = best_segment.get("wearing_analysis", {})
                        timeline_entry["segment_mask_ratio"] = best_segment.get("mask_ratio", 0)
                    else:
                        timeline_entry["wearing_analysis"] = {}
                        timeline_entry["segment_mask_ratio"] = 0
                else:
                    timeline_entry["segment_details"] = []
                    timeline_entry["wearing_analysis"] = {}
                    timeline_entry["segment_max_confidence"] = 0
                    timeline_entry["segment_mask_ratio"] = 0

                timeline.append(timeline_entry)
                
                # 更新状态
                prev_result = timeline_entry
                last_forced_timestamp = ts
                if len(frame.shape) == 2:
                    prev_frame_gray = frame
                else:
                    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (i + 1) % 10 == 0:
                logger.info(f"已分析 {i + 1}/{len(timestamps)} 帧")

        # 输出运动检测统计
        if self.motion_detection and motion_skipped_count > 0:
            logger.info(f"运动检测统计: 跳过 {motion_skipped_count} 帧静止画面，节省 {motion_skipped_count} 次YOLO推理")

        logger.info(f"全面分析完成: {len(timeline)}帧, 提取帧数: {len(frames)}")

        # 临时文件保留用于调试，不自动清理
        # 如需清理可取消下面注释
        # import shutil
        # shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "frames": frames,
            "decoded_frames": decoded_frames,
            "timeline": timeline,
            "duration": duration,
        }

    def _generate_video_summary(self, timeline: List[Dict], duration: float) -> Dict:
        """生成视频摘要（包含多模型统计和运动检测统计）"""
        frames_with_person = [t for t in timeline if t.get("has_person")]
        
        # 运动检测统计
        motion_skipped_count = sum(1 for t in timeline if t.get("motion_skipped", False))
        total_frames = len(timeline)
        yolo_inference_count = total_frames - motion_skipped_count

        if not frames_with_person:
            return {
                "has_person": False,
                "duration": duration,
                "person_ratio": 0,
                "motion_skipped_count": motion_skipped_count,
                "total_frames": total_frames,
                "yolo_inference_count": yolo_inference_count,
            }

        # 姿态变化时间线（带时序平滑：持续>=2帧才算真正变化）
        pose_changes = []
        prev_pose = None
        pose_persist_count = 0
        PERSIST_THRESHOLD = 2
        for t in frames_with_person:
            current_pose = tuple(t.get("pose_analysis", []))
            if current_pose == prev_pose:
                pose_persist_count += 1
            else:
                if pose_persist_count >= PERSIST_THRESHOLD and prev_pose is not None:
                    pose_changes.append({
                        "timestamp": t["timestamp"],
                        "from": list(prev_pose),
                        "to": list(current_pose),
                    })
                pose_persist_count = 1
                prev_pose = current_pose

        # 人体出现时间段
        person_appearances = []
        start = None
        for i, t in enumerate(timeline):
            if t.get("has_person") and start is None:
                start = t["timestamp"]
            elif not t.get("has_person") and start is not None:
                person_appearances.append({
                    "start": start,
                    "end": timeline[i - 1]["timestamp"],
                })
                start = None
        if start is not None:
            person_appearances.append({"start": start, "end": timeline[-1]["timestamp"]})

        # 主要姿态统计
        pose_counts = {}
        for t in frames_with_person:
            poses = t.get("pose_analysis", [])
            for pose in poses:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1

        main_pose = max(pose_counts, key=pose_counts.get) if pose_counts else "未知"

        # 平均置信度和关键点
        avg_confidence = sum(t.get("confidence", 0) for t in frames_with_person) / len(frames_with_person)
        avg_keypoints = sum(t.get("visible_keypoints", 0) for t in frames_with_person) / len(frames_with_person)

        # 多模型统计
        models_used = set()
        vote_counts = []
        wearing_stats = []
        for t in frames_with_person:
            models_used.update(t.get("models_used", []))
            vote_counts.append(t.get("vote_count", 0))
            wearing = t.get("wearing_analysis", {})
            if wearing.get("has_wearing"):
                wearing_stats.append(wearing.get("color_variance", 0))

        avg_vote = sum(vote_counts) / len(vote_counts) if vote_counts else 0
        avg_wearing_variance = sum(wearing_stats) / len(wearing_stats) if wearing_stats else 0

        return {
            "has_person": True,
            "duration": duration,
            "person_ratio": len(frames_with_person) / len(timeline),
            "person_appearances": person_appearances,
            "pose_changes": pose_changes,
            "main_pose": main_pose,
            "pose_distribution": pose_counts,
            "avg_confidence": avg_confidence,
            "avg_keypoints": avg_keypoints,
            "first_appearance": frames_with_person[0]["timestamp"],
            "last_appearance": frames_with_person[-1]["timestamp"],
            "models_used": list(models_used),
            "avg_vote": avg_vote,
            "avg_wearing_variance": avg_wearing_variance,
            "motion_skipped_count": motion_skipped_count,
            "total_frames": total_frames,
            "yolo_inference_count": yolo_inference_count,
        }

    def _select_representative_frames(self, timeline: List[Dict], max_frames: int = 10, clip_diff_scores: list = None) -> List[int]:
        """
        分区段选择代表性帧：将采样帧等分为 max_frames 个区段，
        每个区段内独立选最优帧，保证全视频均匀覆盖。
        clip_diff_scores: CLIP 差异度评分列表，用于优先选择差异大的帧。
        """
        n = len(timeline)
        if n == 0:
            return []
        if n <= max_frames:
            return list(range(n))

        # 将 timeline 等分为 max_frames 个区段
        seg_size = n / max_frames
        selected = []

        for seg_idx in range(max_frames):
            start = int(seg_idx * seg_size)
            end = int((seg_idx + 1) * seg_size)
            if seg_idx == max_frames - 1:
                end = n  # 最后一段包含末尾

            segment = [(i, timeline[i]) for i in range(start, end)]

            # 构建当前区段的差异度评分字典
            diff_scores_for_segment = None
            if clip_diff_scores:
                diff_scores_for_segment = {
                    i: clip_diff_scores[i] if i < len(clip_diff_scores) else 0.0
                    for i in range(start, end)
                }

            # 区段内按置信度+关键点加权选最优帧
            best_i = self._pick_best_from_segment(segment, diff_scores_for_segment)
            selected.append(best_i)

        return selected

    def _pick_best_from_segment(self, segment: List[tuple], diff_scores: dict = None) -> int:
        """从区段内选出最优帧索引，无人体时取中间帧"""
        frames_with_person = [(i, t) for i, t in segment if t.get("has_person")]

        if not frames_with_person:
            # 无人体帧，优先选择差异度高的帧，否则取中间帧
            if diff_scores:
                return max(segment, key=lambda x: diff_scores.get(x[0], 0.0))[0]
            return segment[len(segment) // 2][0]

        # 加权评分：confidence 30% + visible_keypoints/17 20% + 姿态变化 20% + diff_score 30%
        prev_pose = None
        best_score = -1
        best_idx = frames_with_person[0][0]

        for i, t in frames_with_person:
            score = t.get("confidence", 0) * 30
            score += (t.get("visible_keypoints", 0) / 17) * 20
            current_pose = tuple(t.get("pose_analysis", []))
            if prev_pose and current_pose != prev_pose:
                score += 20
            prev_pose = current_pose
            if diff_scores:
                score += diff_scores.get(i, 0.0) * 30

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _generate_frame_descriptions(self, timeline: List[Dict], selected_indices: List[int]) -> List[str]:
        """为选中帧生成描述（包含多个模型的结果）"""
        descriptions = []

        for i, idx in enumerate(selected_indices):
            if idx >= len(timeline):
                continue

            t = timeline[idx]
            ts = t.get("timestamp", 0)
            has_person = t.get("has_person", False)
            models_used = t.get("models_used", [])
            vote_count = t.get("vote_count", 0)

            if has_person:
                desc_parts = [f"图{i+1}@{ts:.1f}s:"]
                
                # 检测结果
                detection_details = t.get("detection_details", [])
                if detection_details:
                    det_conf = max(d.get("confidence", 0) for d in detection_details)
                    desc_parts.append(f"[检测]置信度={det_conf:.2f}")
                
                # 姿态结果
                poses = t.get("pose_analysis", [])
                kpts = t.get("visible_keypoints", 0)
                if poses:
                    pose_str = ", ".join(poses)
                    desc_parts.append(f"[姿态]{pose_str}, 关键点={kpts}/17")
                
                # 分割结果
                wearing = t.get("wearing_analysis", {})
                segment_details = t.get("segment_details", [])
                if segment_details:
                    seg_conf = max(s.get("confidence", 0) for s in segment_details)
                    desc_parts.append(f"[分割]置信度={seg_conf:.2f}")
                    if wearing.get("has_wearing"):
                        color_var = wearing.get("color_variance", 0)
                        desc_parts.append(f"穿着色彩变化={color_var:.1f}")
                
                # 投票信息
                desc_parts.append(f"投票={vote_count}/{len(models_used)}")
                
                desc = " ".join(desc_parts)
            else:
                desc = f"图{i+1}@{ts:.1f}s: 未检测到人体 (投票={vote_count}/{len(models_used)})"

            descriptions.append(desc)

        return descriptions

    def _build_comprehensive_context(
        self, video_summary: Dict, frame_descriptions: List[str], n_frames: int
    ) -> str:
        """构建全面上下文（包含多模型信息）"""
        context_lines = []

        if video_summary.get("has_person"):
            context_lines.append("【视频全面分析结果】")
            context_lines.append(f"- 视频时长: {video_summary.get('duration', 0):.1f}秒")
            context_lines.append(f"- 人体出现比例: {video_summary.get('person_ratio', 0) * 100:.1f}%")
            
            # 多模型信息
            models_used = video_summary.get("models_used", [])
            if models_used:
                context_lines.append(f"- 使用模型: {', '.join(models_used)}")
                context_lines.append(f"- 平均投票数: {video_summary.get('avg_vote', 0):.1f}/{len(models_used)}")

            # 姿态分布（结构化描述）
            pose_dist = video_summary.get("pose_distribution", {})
            if pose_dist:
                total_person = int(video_summary.get("person_ratio", 0) * video_summary.get("total_frames", 1))
                main_pose = video_summary.get("main_pose", "未知")
                main_count = pose_dist.get(main_pose, 0)
                context_lines.append(f"- 主要姿态: {main_pose} (占{main_count}/{total_person}帧)")
                # 其他姿态
                other_poses = {k: v for k, v in pose_dist.items() if k != main_pose}
                if other_poses:
                    other_str = ", ".join([f"{k}({v}次)" for k, v in sorted(other_poses.items(), key=lambda x: -x[1])])
                    context_lines.append(f"- 其他姿态: {other_str}")

            # 姿态变化（时间线描述）
            pose_changes = video_summary.get("pose_changes", [])
            if pose_changes:
                context_lines.append(f"- 姿态变化: 共{len(pose_changes)}次")
                for change in pose_changes[:5]:
                    from_pose = ", ".join(change["from"]) if change["from"] else "无"
                    to_pose = ", ".join(change["to"]) if change["to"] else "无"
                    context_lines.append(f"  * {change['timestamp']:.1f}s时从[{from_pose}]变为[{to_pose}]")

            # 人体出现时间段
            appearances = video_summary.get("person_appearances", [])
            if appearances:
                context_lines.append(f"- 人体出现时间段:")
                for app in appearances[:3]:  # 最多显示3个时间段
                    context_lines.append(f"  * {app['start']:.1f}s - {app['end']:.1f}s")

            # 统计信息
            context_lines.append(f"- 平均置信度: {video_summary.get('avg_confidence', 0):.2f}")
            context_lines.append(f"- 平均可见关键点: {video_summary.get('avg_keypoints', 0):.1f}/17")
            
            # 穿着分析
            avg_wearing_variance = video_summary.get("avg_wearing_variance", 0)
            if avg_wearing_variance > 0:
                context_lines.append(f"- 穿着色彩变化: {avg_wearing_variance:.1f}")
            
            # 运动检测统计
            motion_skipped = video_summary.get("motion_skipped_count", 0)
            if motion_skipped > 0:
                total_frames = video_summary.get("total_frames", 0)
                yolo_count = video_summary.get("yolo_inference_count", total_frames)
                context_lines.append(f"- 运动检测: 跳过{motion_skipped}帧静止画面 (YOLO推理{yolo_count}次)")

            context_lines.append("")
            context_lines.append("【各帧详细分析（图片序号对应下方描述）】")
            for desc in frame_descriptions:
                context_lines.append(f"- {desc}")
        else:
            context_lines.append("【视频分析结果】")
            context_lines.append("- 视频中未检测到人体")

        return "\n".join(context_lines)

    def _to_base64_item(self, item) -> str:
        """帧元素分派：str 路径读盘编码；ndarray 直接编码"""
        if isinstance(item, np.ndarray):
            return image_array_to_base64(item, max_size=self.max_image_size)
        return image_to_base64(item, max_size=self.max_image_size)

    def _call_vlm_comprehensive(self, frames, title: str, context: str = "", audio_context: str = "", per_frame_subtitle: str = "", diff_hint: str = "") -> Dict:
        """调用VLM - 全面分析模式，失败重试一次"""
        if not frames:
            return {"error": "无可用帧（所有帧提取失败）"}

        prompt = self._build_comprehensive_prompt(title, len(frames), context, audio_context, per_frame_subtitle, diff_hint=diff_hint)

        images_b64 = [self._to_base64_item(f) for f in frames]
        result = call_vision_api(
            self.provider, images_b64, prompt,
            model=self.model, api_key=self.api_key,
        )

        # 失败时重试一次（同样帧数）
        if result.startswith("[ERROR]") or not result.strip():
            logger.warning(f"VLM调用失败，重试一次: {len(frames)}帧")
            result = call_vision_api(
                self.provider, images_b64, prompt,
                model=self.model, api_key=self.api_key,
            )

        # 记录VLM响应（用于调试）
        logger.debug(f"VLM响应: {result[:500] if result else '空'}")

        # 检查VLM调用是否失败
        if not result or result.startswith("[ERROR]"):
            error_msg = f"VLM调用失败: {result[:100] if result else '空响应'}"
            logger.error(error_msg)
            return {"error": error_msg}

        parsed = self._parse_vision_response(result)

        # 关键词为空时重试（截断/格式异常）
        if not parsed.get("keywords"):
            logger.warning("VLM返回关键词为空，使用强调关键词格式的prompt重试")
            retry_prompt = (
                f"{get_prompt('vision_retry_video', 'system_header')}\n\n"
                f'分析媒体文件 "{title}"。\n\n'
                f"{get_prompt('vision_retry_video', 'task_instruction')}\n"
                f"{get_prompt('vision_retry_video', 'output_format')}"
            )

            images_b64_retry = [self._to_base64_item(f) for f in frames[:5]]

            result_retry = call_vision_api(
                self.provider, images_b64_retry, retry_prompt,
                model=self.model, api_key=self.api_key,
            )

            if result_retry and not result_retry.startswith("[ERROR]"):
                parsed_retry = self._parse_vision_response(result_retry)
                # 重试成功：关键词非空
                if parsed_retry.get("keywords"):
                    logger.info("关键词重试成功")
                    return parsed_retry
                logger.warning("关键词重试后仍为空，使用首次结果")

        # 最终检查：如果关键词仍为空，返回错误
        if not parsed.get("keywords"):
            error_msg = "VLM返回关键词为空（已重试）"
            logger.error(error_msg)
            return {"error": error_msg}

        return parsed

    def _save_detection_debug(self, video_analysis: Dict, debug_dir: Path):
        """保存检测结果调试数据"""
        import shutil

        detection_dir = debug_dir / "detection"
        timeline = video_analysis.get("timeline", [])
        frames = video_analysis.get("frames", [])

        # 保存每帧的检测结果
        for i, entry in enumerate(timeline):
            frame_path = entry.get("frame_path")
            if not frame_path or not Path(frame_path).exists():
                continue

            stem = f"frame_{i:04d}_{entry.get('timestamp', 0):.1f}s"

            # 复制原始帧
            original_dest = detection_dir / f"{stem}_original.jpg"
            shutil.copy2(frame_path, original_dest)

            # 绘制检测结果并保存（复用缓存推理结果，不重新推理）
            try:
                data = np.fromfile(frame_path, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                pose_result = entry.get("raw_pose")
                if frame is not None and pose_result is not None:
                    from ..detectors.yolo import draw_pose_on_frame
                    annotated = draw_pose_on_frame(frame, pose_result)

                    annotated_dest = detection_dir / f"{stem}_annotated.jpg"
                    cv2.imwrite(str(annotated_dest), annotated)
            except Exception as e:
                logger.warning(f"绘制检测结果失败: {e}")

            # 保存检测结果JSON
            json_dest = detection_dir / f"{stem}_result.json"
            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"检测调试数据已保存: {len(timeline)}帧")

    def _save_vlm_debug(self, frames: List[str], prompt: str, debug_dir: Path):
        """保存VLM输入调试数据"""
        import shutil

        vlm_dir = debug_dir / "vlm_frames"

        # 复制VLM输入帧
        for i, frame_path in enumerate(frames):
            if Path(frame_path).exists():
                dest = vlm_dir / f"selected_{i:03d}.jpg"
                shutil.copy2(frame_path, dest)

        # 保存prompt
        prompt_file = debug_dir / "vlm_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        logger.info(f"VLM调试数据已保存: {len(frames)}帧, prompt长度={len(prompt)}")

    def _save_debug_summary(self, vlm_result: Dict, video_summary: Dict, debug_dir: Path,
                           frame_timestamps: List[float] = None):
        """保存调试汇总"""
        # 保存VLM响应
        response_file = debug_dir / "vlm_response.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(f"描述: {vlm_result.get('description', '')}\n")
            f.write(f"关键词: {vlm_result.get('keywords', '')}\n")

        # 保存汇总JSON
        summary = {
            "vlm_result": vlm_result,
            "video_summary": video_summary,
        }
        if frame_timestamps:
            summary["frame_timestamps"] = frame_timestamps
        summary_file = debug_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"调试汇总已保存")

    def _build_comprehensive_prompt(self, title: str, n_frames: int, context: str, audio_context: str = "", per_frame_subtitle: str = "", diff_hint: str = "") -> str:
        """构建全面分析提示词"""
        
        # 构建音频上下文部分
        audio_section = ""
        if audio_context:
            audio_section = f"""

【音频转录】
{audio_context}"""

        # 构建每帧对应的字幕上下文
        subtitle_section = ""
        if per_frame_subtitle:
            subtitle_section = f"""

【各帧对应音频转录时间段】
{per_frame_subtitle}
（如果某帧无对应字幕，说明该时间段没有语音内容）"""

        # 差异度提示
        diff_section = ""
        if diff_hint:
            diff_section = f"""

【帧差异度提示】
{diff_hint}"""

        return f"""{get_prompt('vision_video', 'system_header')}

分析媒体文件 "{title}" 的{n_frames}个关键帧。

{context}{diff_section}{audio_section}{subtitle_section}

【任务说明】
{get_prompt('vision_video', 'task_instruction')}

【输出要求】
{get_prompt('vision_video', 'output_format')}

格式：
描述：xxx
关键词：xxx, xxx, xxx"""

    def process_image(self, image_path: str, title: str) -> Dict:
        """处理图片"""
        logger.info(f"图片模式: {title[:30]}")

        compressed_path = str(Path(image_path).parent / f"{Path(image_path).stem}_compressed.jpg")
        if not compress_image(image_path, compressed_path, max_size=self.max_image_size):
            compressed_path = image_path

        if self.use_clip and self.clip_classifier:
            clip_result = self.clip_classifier.classify(compressed_path, threshold=self.clip_threshold)
            if clip_result["avg_confidence"] >= self.clip_threshold:
                # 构建CLIP详细置信度JSON
                import json as _json
                clip_detail = {}
                for dim in ["clothing", "action", "hairstyle"]:
                    dim_results = clip_result.get("all_results", {}).get(dim, [])
                    clip_detail[dim] = [
                        {"label": r.get("label", ""), "label_cn": r.get("label_cn", ""), "confidence": round(r.get("confidence", 0), 4)}
                        for r in dim_results
                    ]
                return {
                    "description": f"[CLIP] {clip_result['tags']}",
                    "keywords": clip_result["tags"],
                    "source": "clip_only",
                    "clip_clothing": clip_result.get("clothing", {}).get("label_cn", ""),
                    "clip_action": clip_result.get("action", {}).get("label_cn", ""),
                    "clip_hairstyle": clip_result.get("hairstyle", {}).get("label_cn", ""),
                    "clip_confidence": round(clip_result.get("avg_confidence", 0), 4),
                    "clip_tags_json": _json.dumps(clip_result.get("tags_json", {}), ensure_ascii=False),
                    "clip_detail": _json.dumps(clip_detail, ensure_ascii=False),
                }

        image_b64 = image_to_base64(compressed_path, max_size=self.max_image_size)

        prompt = (
            f"{get_prompt('vision_image', 'system_header')}\n\n"
            f'分析图片 "{title}"。\n\n'
            f"{get_prompt('vision_image', 'task_instruction')}\n\n"
            f"{get_prompt('vision_image', 'output_format')}"
        )

        result = call_vision_api(
            self.provider, image_b64, prompt,
            model=self.model, api_key=self.api_key,
        )

        if compressed_path != image_path and Path(compressed_path).exists():
            try:
                Path(compressed_path).unlink()
            except:
                pass

        parsed = self._parse_vision_response(result)

        # 关键词为空时重试
        if not parsed.get("keywords"):
            logger.warning("图片VLM返回关键词为空，重试")
            retry_prompt = (
                f"{get_prompt('vision_retry_image', 'system_header')}\n\n"
                f'分析图片 "{title}"。\n\n'
                f"{get_prompt('vision_retry_image', 'task_instruction')}\n"
                f"{get_prompt('vision_retry_image', 'output_format')}"
            )
            result_retry = call_vision_api(
                self.provider, image_b64, retry_prompt,
                model=self.model, api_key=self.api_key,
            )
            if result_retry and not result_retry.startswith("[ERROR]"):
                parsed_retry = self._parse_vision_response(result_retry)
                if parsed_retry.get("keywords"):
                    logger.info("图片关键词重试成功")
                    return parsed_retry
                logger.warning("图片关键词重试后仍为空，使用首次结果")

        return parsed

    def _parse_vision_response(self, response: str) -> Dict:
        """解析VLM响应"""
        result = {"description": "", "keywords": ""}
        
        if not response:
            logger.warning("VLM响应为空")
            return result
        
        # 尝试多种格式解析
        # 格式1: 中文格式 "描述：xxx\n关键词：xxx"
        desc_match = re.search(r"描述[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL)
        kw_match = re.search(r"关键词[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL)
        
        if desc_match:
            result["description"] = desc_match.group(1).strip()
        if kw_match:
            result["keywords"] = kw_match.group(1).strip()
        
        # 格式1b: 英文格式 "description: xxx\nkeywords: xxx"
        if not result["description"]:
            desc_match_en = re.search(r"description[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
            if desc_match_en:
                result["description"] = desc_match_en.group(1).strip()
        
        if not result["keywords"]:
            kw_match_en = re.search(r"keywords?[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
            if kw_match_en:
                result["keywords"] = kw_match_en.group(1).strip()
        
        # 格式2: "1. 描述：xxx\n2. 关键词：xxx"
        if not result["description"]:
            desc_match2 = re.search(r"1[.、]?\s*描述[：:]\s*(.+?)(?:\n|2[.、]?\s*关键词)", response, re.DOTALL)
            if desc_match2:
                result["description"] = desc_match2.group(1).strip()
        
        if not result["keywords"]:
            kw_match2 = re.search(r"2[.、]?\s*关键词[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL)
            if kw_match2:
                result["keywords"] = kw_match2.group(1).strip()
        
        # 格式3: 如果还是没有，尝试从整个响应中提取
        if not result["description"] and not result["keywords"]:
            # 检查是否包含"描述"和"关键词"
            if "描述" in response and "关键词" in response:
                # 尝试按行分割
                lines = response.split("\n")
                for i, line in enumerate(lines):
                    if "描述" in line and "：" in line:
                        result["description"] = line.split("：", 1)[1].strip()
                    elif "关键词" in line and "：" in line:
                        result["keywords"] = line.split("：", 1)[1].strip()
        
        # 记录解析结果
        logger.debug(f"解析结果: 描述='{result['description'][:50]}...', 关键词='{result['keywords'][:50]}...'")
        
        # 调试：如果关键词为空，输出原始响应
        if not result["keywords"]:
            logger.warning(f"关键词为空，原始响应前200字: {response[:200]}")
        
        return result

    def generate_final_name(self, keywords: str, original_title: str) -> str:
        """
        从vision_keywords生成final_name
        水印博主名字最优先

        Args:
            keywords: 逗号分隔的关键词
            original_title: 原始文件名

        Returns:
            格式：[关键词1_关键词2_...]_原文件名
        """
        if not keywords:
            return original_title

        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        kw_list = kw_list[:12]  # 最多12个关键词

        if not kw_list:
            return original_title

        # 安全兜底：如果 original_title 仍带有 [kw]_ 前缀，剥离
        clean_title = re.sub(r"^\[[^\]]*\]_?", "", original_title, count=1)

        prefix = "_".join(kw_list)
        return f"[{prefix}]_{clean_title}"

    def generate_srt(
        self,
        video_path: str,
        description: str,
        keywords: str,
        final_name: str = None,
        video_summary: Dict = None,
        output_dir: str = "data/output/subtitles",
    ) -> str:
        """
        生成带元数据的SRT文件
        
        Args:
            video_path: 视频路径
            description: VLM描述
            keywords: VLM关键词
            final_name: 最终文件名（用于SRT文件名）
            video_summary: 视频摘要（姿态分析等）
            output_dir: 输出目录
        
        Returns:
            SRT文件路径
        """
        srt_dir = Path(output_dir)
        srt_dir.mkdir(parents=True, exist_ok=True)

        # 使用final_name作为SRT文件名（去掉扩展名）
        if final_name:
            srt_name = Path(final_name).stem
        else:
            srt_name = Path(video_path).stem
        
        srt_path = srt_dir / f"{srt_name}.srt"

        # 构建姿态摘要
        pose_summary = ""
        if video_summary and video_summary.get("has_person"):
            main_pose = video_summary.get("main_pose", "未知")
            pose_changes = len(video_summary.get("pose_changes", []))
            person_ratio = video_summary.get("person_ratio", 0) * 100
            pose_summary = f"主要姿态：{main_pose}，姿态变化{pose_changes}次，人体出现{person_ratio:.0f}%"

        # 写入SRT文件
        with open(srt_path, "w", encoding="utf-8") as f:
            # 元数据帧
            f.write("0\n")
            f.write("00:00:00,000 --> 00:00:01,000\n")
            f.write(f"【视频描述】{description}\n")
            f.write(f"【关键词】{keywords}\n")
            if pose_summary:
                f.write(f"【姿态分析】{pose_summary}\n")
            f.write("\n")

        logger.info(f"SRT文件已生成: {srt_path}")
        return str(srt_path)

    def _save_vlm_covers(self, media_id: int, frames: List[str], timestamps: List[float] = None):
        """
        保存 VLM 帧到 covers 目录（仅首次保存）
        """
        if not self.covers_dir or not frames:
            return

        cover_dir = Path(self.covers_dir) / str(media_id)
        cover_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, frame_path in enumerate(frames):
            dest = cover_dir / f"frame_{i:03d}.jpg"
            if dest.exists():
                continue  # 仅首次保存
            try:
                import shutil
                shutil.copy2(frame_path, str(dest))
                saved += 1
                # 记录到数据库
                if self.db_store:
                    ts = timestamps[i] if timestamps and i < len(timestamps) else None
                    self.db_store.save_vlm_frame(media_id, i, str(dest), ts)
            except Exception as e:
                logger.warning(f"保存VLM帧失败: {e}")

        if saved > 0:
            logger.info(f"保存VLM帧: {saved}张到 {cover_dir}")

    def _sync_to_db(self, media_id: int, result: dict, video_path: str = None, duration: float = None):
        """同步识别结果到数据库"""
        if not self.db_store:
            return

        from ..core.db_store import MediaDB
        db = self.db_store

        # 更新字段
        if result.get("description"):
            db.update_media(media_id, "vision_description", result["description"], "vision")
        if result.get("keywords"):
            db.update_media(media_id, "vision_keywords", result["keywords"], "vision")
            db.add_tags_from_keywords(media_id, result["keywords"], "vision")
        if result.get("final_name"):
            db.update_media(media_id, "final_name", result["final_name"], "vision")
        if result.get("srt_path"):
            db.update_media(media_id, "srt_path", result["srt_path"], "vision")

        # 从 video_summary 更新额外字段
        vs = result.get("video_summary", {})
        if vs:
            db.update_media(media_id, "human_detected", vs.get("has_person", False), "vision")
            if vs.get("has_person"):
                db.update_media(media_id, "detection_method", "yolo", "vision")

        # 补充文件元数据
        if video_path:
            try:
                import os
                file_size = os.path.getsize(video_path)
                db.update_media(media_id, "file_size", file_size, "vision")
            except Exception:
                pass
        if duration:
            db.update_media(media_id, "duration", duration, "vision")

        db.update_media(media_id, "needs_vision", 0, "vision")

        logger.debug(f"数据库同步完成: media_id={media_id}")

    def process_and_save(
        self,
        video_path: str,
        title: str,
        original_title: str = None,
        srt_output_dir: str = "data/output/subtitles",
    ) -> Dict:
        """
        处理视频并生成所有结果
        
        Args:
            video_path: 视频路径
            title: 标题
            original_title: 原始文件名（用于生成final_name）
            srt_output_dir: SRT输出目录
        
        Returns:
            {
                "description": str,
                "keywords": str,
                "final_name": str,
                "srt_path": str,
                "video_summary": Dict,
            }
        """
        if original_title is None:
            original_title = Path(video_path).name

        # 检查是否已有音频SRT
        audio_srt_path = self._find_audio_srt(original_title, srt_output_dir)
        audio_context = ""
        subtitle_segments = []
        
        if audio_srt_path:
            # 读取音频转录内容
            audio_context = self._read_audio_transcription(audio_srt_path)
            # 解析带时间戳的字幕段
            subtitle_segments = self._parse_audio_srt_with_timestamps(audio_srt_path)
            logger.info(f"检测到音频SRT，将作为VLM上下文: {audio_srt_path}")
            logger.info(f"音频上下文长度: {len(audio_context)} 字符, 字幕段数: {len(subtitle_segments)}")

        # 检测是否为图片文件
        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
        is_image = Path(video_path).suffix.lower() in IMAGE_EXT

        if is_image:
            result = self.process_image(video_path, title)
        else:
            # 处理视频（传入音频上下文和字幕段）
            result = self.process_video(video_path, title, audio_context, subtitle_segments)

        if "error" in result:
            return result

        description = result.get("description", "")
        keywords = result.get("keywords", "")
        video_summary = result.get("video_summary", {})

        # 生成final_name（剥离扩展名，避免重复）
        title_stem = Path(original_title).stem if "." in original_title else original_title
        final_name = self.generate_final_name(keywords, title_stem)

        # 生成/重命名SRT
        if audio_srt_path:
            # 重命名SRT文件并插入视觉描述
            srt_path = self._rename_and_update_srt(
                audio_srt_path, final_name, description, keywords, video_summary
            )
        else:
            # 正常生成SRT
            srt_path = self.generate_srt(
                video_path, description, keywords, final_name, video_summary, srt_output_dir
            )

        # 构建返回结果
        final_result = {
            "description": description,
            "keywords": keywords,
            "final_name": final_name,
            "srt_path": srt_path,
            "video_summary": video_summary,
            "debug_dir": result.get("debug_dir"),
        }

        # 保存 VLM 帧到 covers 目录（仅首次）
        if self.covers_dir and self.db_store:
            frames_for_vlm = result.get("frames_for_vlm", [])
            frame_timestamps = result.get("frame_timestamps", [])
            if frames_for_vlm:
                # 查找 media_id
                from ..core.db_store import MediaDB
                db = self.db_store
                media_record = db.find_by_path(str(video_path))
                if media_record:
                    self._save_vlm_covers(
                        media_record["id"],
                        frames_for_vlm,
                        frame_timestamps
                    )

        # 学习 VLM 返回的关键词到 CLIP 标签库
        if keywords and self.tag_stats:
            self.tag_stats.update_from_vlm(keywords)

        # 同步识别结果到数据库
        if self.db_store:
            media_record = self.db_store.find_by_path(str(video_path))
            if not media_record:
                media_record = self.db_store.find_by_current_path(str(video_path))
            if not media_record:
                # 尝试按文件名模糊匹配
                fname = Path(video_path).name
                media_record = self.db_store.find_match(original_title=fname, path=str(video_path))
            if not media_record:
                # 内容指纹兜底：重命名/移动未重扫的视频，找回原记录
                try:
                    from ..utils.fingerprint import compute_partial_hash
                    fhash = compute_partial_hash(video_path)
                    if fhash:
                        fp = self.db_store.find_fingerprint_by_hash(fhash)
                        if fp:
                            media_record = self.db_store.conn.execute(
                                "SELECT * FROM media_files WHERE fingerprint_id=?",
                                (fp["id"],)
                            ).fetchone()
                            if media_record:
                                media_record = dict(media_record)
                                self.db_store.update_media(
                                    media_record["id"], "current_path", str(video_path), "vision")
                except Exception as e:
                    logger.debug(f"指纹查找失败: {e}")
            if not media_record:
                # DB 中无记录，创建新记录
                logger.info(f"DB 中无记录，创建新记录: {video_path}")
                import os
                fsize = None
                try:
                    fsize = os.path.getsize(video_path)
                except Exception:
                    pass
                new_data = {
                    "original_title": Path(video_path).name,
                    "original_path": str(video_path),
                    "current_path": str(video_path),
                    "file_size": fsize,
                    "needs_vision": False,
                    "final_name": title or Path(video_path).stem,
                    "review_status": "已完成",
                }
                new_id = self.db_store.insert_media(new_data)
                media_record = {"id": new_id}
            
            duration = video_summary.get("duration") if video_summary else None
            self._sync_to_db(media_record["id"], final_result, video_path, duration)

        return final_result

    def _find_audio_srt(self, original_title: str, srt_output_dir: str) -> str:
        """
        查找已有的音频SRT文件
        
        Args:
            original_title: 原始文件名
            srt_output_dir: SRT输出目录
        
        Returns:
            音频SRT文件路径，如果不存在返回空字符串
        """
        srt_dir = Path(srt_output_dir)
        srt_name = Path(original_title).stem + ".srt"
        srt_path = srt_dir / srt_name
        
        if srt_path.exists():
            return str(srt_path)
        
        return ""

    def _read_audio_transcription(self, srt_path: str) -> str:
        """
        读取SRT文件中的音频转录内容
        
        Args:
            srt_path: SRT文件路径
        
        Returns:
            音频转录内容（纯文本）
        """
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 解析SRT格式，提取文本内容
            import re
            # 匹配SRT条目：序号 + 时间戳 + 文本
            pattern = r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+?)(?=\n\n|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            # 合并所有文本
            transcription = "\n".join(matches)
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"读取音频SRT失败: {e}")
            return ""

    def _parse_audio_srt_with_timestamps(self, srt_path: str) -> List[Dict]:
        """
        解析SRT文件，返回带时间戳的字幕段列表
        
        Args:
            srt_path: SRT文件路径
        
        Returns:
            字幕段列表: [{"start": 秒数, "end": 秒数, "text": 文本}, ...]
        """
        import re
        segments = []
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按空行分割字幕块
            blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # 解析时间戳行
                time_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1].strip()
                )
                if not time_match:
                    continue
                
                g = time_match.groups()
                start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
                end = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
                
                text = '\n'.join(lines[2:]).strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})
            
            return segments
            
        except Exception as e:
            logger.error(f"解析SRT时间戳失败: {e}")
            return []

    def _match_frame_to_subtitles(self, frame_timestamp: float, subtitle_segments: List[Dict]) -> Optional[Dict]:
        """
        将帧时间戳匹配到字幕段
        
        Args:
            frame_timestamp: 帧时间戳（秒）
            subtitle_segments: 字幕段列表
        
        Returns:
            匹配的字幕段，如果无匹配返回None
        """
        for seg in subtitle_segments:
            if seg["start"] <= frame_timestamp <= seg["end"]:
                return seg
        return None

    def _build_per_frame_subtitle_context(
        self, frame_timestamps: List[float], subtitle_segments: List[Dict]
    ) -> str:
        """
        构建每帧对应的字幕上下文
        
        Args:
            frame_timestamps: 各帧时间戳列表
            subtitle_segments: 字幕段列表
        
        Returns:
            格式化的每帧字幕上下文字符串
        """
        if not subtitle_segments:
            return ""
        
        def fmt_time(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"
        
        # 按字幕段分组帧，避免重复发送相同字幕内容
        # key: segment tuple (start, end), value: list of (frame_index, timestamp)
        from collections import OrderedDict
        seg_groups = OrderedDict()
        no_subtitle_frames = []
        
        for i, ts in enumerate(frame_timestamps, 1):
            seg = self._match_frame_to_subtitles(ts, subtitle_segments)
            if seg:
                seg_key = (seg['start'], seg['end'])
                if seg_key not in seg_groups:
                    seg_groups[seg_key] = {"seg": seg, "frames": []}
                seg_groups[seg_key]["frames"].append((i, ts))
            else:
                no_subtitle_frames.append((i, ts))
        
        lines = []
        for seg_key, group in seg_groups.items():
            seg = group["seg"]
            frames = group["frames"]
            time_range = f"[{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}]"
            text = seg['text']
            
            if len(frames) == 1:
                i, ts = frames[0]
                lines.append(f"- 图{i}@{ts:.1f}s: {time_range} {text}")
            else:
                # 多帧对应同一字幕段，合并显示
                frame_refs = ",".join(str(i) for i, _ in frames)
                ts_range = f"{frames[0][1]:.1f}s-{frames[-1][1]:.1f}s"
                lines.append(f"- 图{frame_refs}@{ts_range}: {time_range} {text}")
        
        for i, ts in no_subtitle_frames:
            lines.append(f"- 图{i}@{ts:.1f}s: (无对应字幕)")
        
        return "\n".join(lines)

    def _rename_and_update_srt(
        self,
        audio_srt_path: str,
        final_name: str,
        description: str,
        keywords: str,
        video_summary: Dict,
    ) -> str:
        """
        重命名音频SRT文件并插入视觉描述
        
        Args:
            audio_srt_path: 原音频SRT路径
            final_name: 最终文件名
            description: 视觉描述
            keywords: 关键词
            video_summary: 视频摘要
        
        Returns:
            新SRT文件路径
        """
        try:
            audio_srt = Path(audio_srt_path)
            
            # 生成新的SRT文件名（使用final_name）
            new_srt_name = Path(final_name).stem + ".srt"
            new_srt_path = audio_srt.parent / new_srt_name
            
            # 读取原音频SRT内容
            with open(audio_srt, "r", encoding="utf-8") as f:
                audio_content = f.read()
            
            # 构建视觉描述元数据
            pose_summary = ""
            if video_summary and video_summary.get("has_person"):
                main_pose = video_summary.get("main_pose", "未知")
                pose_changes = len(video_summary.get("pose_changes", []))
                person_ratio = video_summary.get("person_ratio", 0) * 100
                pose_summary = f"主要姿态：{main_pose}，姿态变化{pose_changes}次，人体出现{person_ratio:.0f}%"

            # 构建元数据帧
            metadata_frame = "0\n00:00:00,000 --> 00:00:01,000\n"
            metadata_frame += f"【视频描述】{description}\n"
            metadata_frame += f"【关键词】{keywords}\n"
            if pose_summary:
                metadata_frame += f"【姿态分析】{pose_summary}\n"
            metadata_frame += "\n"
            
            # 重新编号音频SRT条目（从1开始）
            import re
            # 替换SRT条目的序号
            def replace_index(match):
                return match.group(1) + "\n"
            
            # 在元数据帧后添加音频内容
            new_content = metadata_frame + audio_content
            
            # 写入新SRT文件
            with open(new_srt_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # 删除原音频SRT（如果新路径不同）
            if str(audio_srt) != str(new_srt_path):
                try:
                    audio_srt.unlink()
                    logger.info(f"已删除原音频SRT: {audio_srt}")
                except Exception as e:
                    logger.warning(f"删除原音频SRT失败: {e}")
            
            logger.info(f"SRT已重命名并更新: {new_srt_path}")
            return str(new_srt_path)
            
        except Exception as e:
            logger.error(f"重命名并更新SRT失败: {e}")
            return audio_srt_path
