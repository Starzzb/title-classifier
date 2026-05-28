"""视觉识别模块 - 视频全面分析版"""

import csv
import json
import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple

import cv2
import numpy as np

from ..providers import get_provider_config, get_api_key, call_vision_api
from ..detectors import YOLODetector, CLIPClassifier
from ..utils.video import get_video_duration, extract_frame, extract_multiple_frames, detect_keyframes
from ..utils.image import compress_image, image_to_base64
from ..utils.stats import TagStatistics

logger = logging.getLogger(__name__)


class VisionProcessor:
    """视觉处理器 - 支持视频全面分析"""

    def __init__(
        self,
        provider: str = "gcli",
        use_yolo: bool = False,
        yolo_model: str = "pose",
        yolo_models: List[str] = None,
        yolo_conf: float = 0.5,
        use_clip: bool = False,
        clip_threshold: float = 0.25,
        max_image_size: int = 480,
        vlm_frames: int = 10,
        analysis_step: float = 2.0,
        debug_dir: str = None,
    ):
        self.provider = provider
        self.use_yolo = use_yolo
        self.yolo_model = yolo_model  # 基础模式使用的模型
        self.yolo_models = yolo_models or ["pose"]  # 全面分析模式使用的模型列表
        self.yolo_conf = yolo_conf
        self.use_clip = use_clip
        self.clip_threshold = clip_threshold
        self.max_image_size = max_image_size
        self.vlm_frames = vlm_frames
        self.analysis_step = analysis_step
        self.debug_dir = debug_dir

        self.provider_config = get_provider_config(provider)
        self.model = self.provider_config.get("default_model", "") if self.provider_config else ""
        self.api_key = get_api_key(provider)

        self.yolo_detector = None
        self.clip_classifier = None
        self.tag_stats = None

    def initialize(self) -> bool:
        """初始化检测器"""
        self.tag_stats = TagStatistics()

        # 始终使用YOLO检测器
        self.yolo_detector = YOLODetector(
            model_types=self.yolo_models,
            confidence=self.yolo_conf,
        )
        if not self.yolo_detector.load_model():
            logger.error("YOLO模型加载失败")
            return False

        if self.use_clip:
            self.clip_classifier = CLIPClassifier(tag_stats=self.tag_stats)
            if not self.clip_classifier.load_model():
                logger.warning("CLIP模型加载失败，将使用纯云端VLM")
                self.use_clip = False
                self.clip_classifier = None

        return True

    def process_video(self, video_path: str, title: str, audio_context: str = "", subtitle_segments: List[Dict] = None) -> Dict:
        """处理视频 - 全面分析模式"""
        duration = get_video_duration(video_path)
        logger.info(f"视频模式: {title[:30]} (时长: {duration:.1f}s)")

        if audio_context:
            logger.info(f"检测到音频上下文，长度: {len(audio_context)} 字符")

        # 始终使用YOLO全面分析模式
        return self._process_video_comprehensive(video_path, title, duration, audio_context, subtitle_segments)

    def _process_video_comprehensive(self, video_path: str, title: str, duration: float, audio_context: str = "", subtitle_segments: List[Dict] = None) -> Dict:
        """视频全面分析"""
        logger.info(f"启动YOLO全面分析模式，采样间隔: {self.analysis_step}秒")

        # 创建调试目录
        debug_subdir = None
        if self.debug_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(video_path).stem[:30]
            debug_subdir = Path(self.debug_dir) / f"{timestamp}_{video_name}"
            debug_subdir.mkdir(parents=True, exist_ok=True)
            (debug_subdir / "detection").mkdir(exist_ok=True)
            (debug_subdir / "vlm_frames").mkdir(exist_ok=True)
            logger.info(f"调试目录: {debug_subdir}")

        # 1. 全面扫描视频
        video_analysis = self._analyze_video_comprehensive(video_path, duration)

        # 2. 智能选择代表性帧
        selected_indices = self._select_representative_frames(video_analysis["timeline"])
        selected_frames = video_analysis["frames"]

        logger.info(f"帧选择完成: 总帧数={len(selected_frames)}, 选中帧数={len(selected_indices)}")

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

        # 6. 调用VLM（传入音频上下文和每帧字幕）
        frames_for_vlm = [selected_frames[i] for i in selected_indices if i < len(selected_frames)]
        logger.info(f"调用VLM: {len(frames_for_vlm)}帧")
        
        # 保存调试数据 - 检测结果
        if debug_subdir:
            self._save_detection_debug(video_analysis, debug_subdir)

        # 构建prompt（用于调试）
        prompt = self._build_comprehensive_prompt(title, len(frames_for_vlm), comprehensive_context, audio_context, per_frame_subtitle)

        # 保存调试数据 - VLM输入帧和prompt
        if debug_subdir:
            self._save_vlm_debug(frames_for_vlm, prompt, debug_subdir)

        result = self._call_vlm_comprehensive(
            frames_for_vlm,
            title,
            comprehensive_context,
            audio_context,
            per_frame_subtitle,
        )

        logger.info(f"VLM结果: 描述='{result.get('description', '')[:50]}...', 关键词='{result.get('keywords', '')[:50]}...'")

        # 保存调试数据 - VLM响应和汇总
        if debug_subdir:
            self._save_debug_summary(result, video_summary, debug_subdir)

        # 7. 保存分析结果
        analysis_result = {
            "description": result.get("description", ""),
            "keywords": result.get("keywords", ""),
            "video_summary": video_summary,
            "selected_frames": len(selected_indices),
            "total_analyzed": len(video_analysis["timeline"]),
            "pose_changes": len(video_summary.get("pose_changes", [])),
            "person_ratio": video_summary.get("person_ratio", 0),
        }

        # 记录调试目录路径
        if debug_subdir:
            analysis_result["debug_dir"] = str(debug_subdir)

        return analysis_result

    def _process_video_traditional(self, video_path: str, title: str, duration: float, audio_context: str = "") -> Dict:
        """传统处理模式"""
        frames = self._extract_frames(video_path)
        if not frames:
            return {"error": "无法提取帧"}

        # 始终使用YOLO进行姿态分析
        yolo_results = self._analyze_poses(frames)

        selected_frames = self._select_frames(frames, yolo_results)
        yolo_context = self._build_yolo_context(yolo_results, selected_frames)

        # 如果有音频上下文，添加到yolo_context中
        if audio_context:
            yolo_context = f"{yolo_context}\n\n【音频转录内容】\n{audio_context}"

        result = self._call_vlm(selected_frames, title, yolo_context)

        return {
            "description": result.get("description", ""),
            "keywords": result.get("keywords", ""),
            "yolo_results": yolo_results,
            "selected_frames": selected_frames,
        }

    def _analyze_video_comprehensive(self, video_path: str, duration: float) -> Dict:
        """全面分析视频 - 高密度采样，使用多个YOLO模型"""
        tmp_dir = Path("logs/_vision_tmp") / Path(video_path).stem
        tmp_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        timeline = []

        # 计算采样时间点
        timestamps = np.arange(0, duration, self.analysis_step)
        if len(timestamps) > 50:  # 限制最大采样数
            timestamps = np.linspace(0, duration, 50)

        logger.info(f"全面分析: {len(timestamps)}个采样点, 模型: {self.yolo_models}")

        for i, ts in enumerate(timestamps):
            # 提取帧
            frame_path = str(tmp_dir / f"frame_{i:04d}_{ts:.1f}s.jpg")
            if not extract_frame(video_path, frame_path, timestamp=str(ts), max_size=400):
                logger.debug(f"帧提取失败: {frame_path}")
                continue

            frames.append(frame_path)

            # YOLO全面分析（使用多个模型）
            data = np.fromfile(frame_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if frame is not None and self.yolo_detector:
                comprehensive_result = self.yolo_detector.analyze_comprehensive(frame)

                timeline_entry = {
                    "index": i,
                    "timestamp": ts,
                    "frame_path": frame_path,
                    "has_person": comprehensive_result.get("has_person", False),
                    "confidence": comprehensive_result.get("confidence", 0),
                    "models_used": comprehensive_result.get("models_used", []),
                    "vote_count": comprehensive_result.get("merged", {}).get("vote_count", 0),
                }

                # 提取姿态信息
                pose_result = comprehensive_result.get("pose")
                if pose_result and pose_result.get("has_person") and pose_result.get("poses"):
                    best_pose = pose_result["poses"][0]
                    timeline_entry["pose_analysis"] = best_pose.get("pose_analysis", [])
                    timeline_entry["visible_keypoints"] = best_pose.get("visible_count", 0)
                    timeline_entry["keypoints"] = best_pose.get("keypoints", {})
                else:
                    timeline_entry["pose_analysis"] = []
                    timeline_entry["visible_keypoints"] = 0
                    timeline_entry["keypoints"] = {}

                # 提取检测信息
                detection_result = comprehensive_result.get("detection")
                if detection_result and detection_result.get("has_person"):
                    timeline_entry["detection_details"] = detection_result.get("persons", [])
                else:
                    timeline_entry["detection_details"] = []

                # 提取分割信息
                segment_result = comprehensive_result.get("segment")
                if segment_result and segment_result.get("has_person"):
                    timeline_entry["segment_details"] = segment_result.get("segments", [])
                    # 提取穿着分析
                    if segment_result.get("segments"):
                        best_segment = segment_result["segments"][0]
                        timeline_entry["wearing_analysis"] = best_segment.get("wearing_analysis", {})
                    else:
                        timeline_entry["wearing_analysis"] = {}
                else:
                    timeline_entry["segment_details"] = []
                    timeline_entry["wearing_analysis"] = {}

                timeline.append(timeline_entry)

            if (i + 1) % 10 == 0:
                logger.info(f"已分析 {i + 1}/{len(timestamps)} 帧")

        logger.info(f"全面分析完成: {len(timeline)}帧, 提取帧数: {len(frames)}")

        # 临时文件保留用于调试，不自动清理
        # 如需清理可取消下面注释
        # import shutil
        # shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "frames": frames,
            "timeline": timeline,
            "duration": duration,
        }

    def _generate_video_summary(self, timeline: List[Dict], duration: float) -> Dict:
        """生成视频摘要（包含多模型统计）"""
        frames_with_person = [t for t in timeline if t.get("has_person")]

        if not frames_with_person:
            return {
                "has_person": False,
                "duration": duration,
                "person_ratio": 0,
            }

        # 姿态变化时间线
        pose_changes = []
        prev_pose = None
        for t in frames_with_person:
            current_pose = tuple(t.get("pose_analysis", []))
            if prev_pose and current_pose != prev_pose:
                pose_changes.append({
                    "timestamp": t["timestamp"],
                    "from": list(prev_pose),
                    "to": list(current_pose),
                })
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
        }

    def _select_representative_frames(self, timeline: List[Dict], max_frames: int = 10) -> List[int]:
        """
        分区段选择代表性帧：将采样帧等分为 max_frames 个区段，
        每个区段内独立选最优帧，保证全视频均匀覆盖。
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

            # 区段内按置信度+关键点加权选最优帧
            best_i = self._pick_best_from_segment(segment)
            selected.append(best_i)

        return selected

    def _pick_best_from_segment(self, segment: List[tuple]) -> int:
        """从区段内选出最优帧索引，无人体时取中间帧"""
        frames_with_person = [(i, t) for i, t in segment if t.get("has_person")]

        if not frames_with_person:
            # 无人体帧，取区段中间帧保证覆盖
            return segment[len(segment) // 2][0]

        # 加权评分：confidence 40% + visible_keypoints/17 30% + 姿态变化 30%
        prev_pose = None
        best_score = -1
        best_idx = frames_with_person[0][0]

        for i, t in frames_with_person:
            score = t.get("confidence", 0) * 40
            score += (t.get("visible_keypoints", 0) / 17) * 30
            current_pose = tuple(t.get("pose_analysis", []))
            if prev_pose and current_pose != prev_pose:
                score += 30
            prev_pose = current_pose

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
            context_lines.append(f"- 主要姿态: {', '.join(video_summary.get('main_pose', ['未知']))}")
            
            # 多模型信息
            models_used = video_summary.get("models_used", [])
            if models_used:
                context_lines.append(f"- 使用模型: {', '.join(models_used)}")
                context_lines.append(f"- 平均投票数: {video_summary.get('avg_vote', 0):.1f}/{len(models_used)}")

            # 姿态分布
            pose_dist = video_summary.get("pose_distribution", {})
            if pose_dist:
                pose_str = ", ".join([f"{k}({v}次)" for k, v in sorted(pose_dist.items(), key=lambda x: -x[1])])
                context_lines.append(f"- 姿态分布: {pose_str}")

            # 姿态变化
            pose_changes = video_summary.get("pose_changes", [])
            if pose_changes:
                context_lines.append(f"- 姿态变化次数: {len(pose_changes)}")
                for change in pose_changes[:5]:  # 最多显示5次变化
                    from_pose = ", ".join(change["from"]) if change["from"] else "无"
                    to_pose = ", ".join(change["to"]) if change["to"] else "无"
                    context_lines.append(f"  * {change['timestamp']:.1f}s: {from_pose} -> {to_pose}")

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

            context_lines.append("")
            context_lines.append("【各帧详细分析（图片序号对应下方描述）】")
            for desc in frame_descriptions:
                context_lines.append(f"- {desc}")
        else:
            context_lines.append("【视频分析结果】")
            context_lines.append("- 视频中未检测到人体")

        return "\n".join(context_lines)

    def _call_vlm_comprehensive(self, frames: List[str], title: str, context: str, audio_context: str = "", per_frame_subtitle: str = "") -> Dict:
        """调用VLM - 全面分析模式，含帧数降级重试"""
        prompt = self._build_comprehensive_prompt(title, len(frames), context, audio_context, per_frame_subtitle)

        if len(frames) > 1:
            images_b64 = [image_to_base64(f, max_size=self.max_image_size) for f in frames]
            result = call_vision_api(
                self.provider, images_b64, prompt,
                model=self.model, api_key=self.api_key,
            )

            # 多帧失败或响应为空时降级：减少帧数重试
            need_fallback = result.startswith("[ERROR]") or not result.strip()
            if need_fallback and len(frames) > 2:
                reduced = max(2, len(frames) // 2)
                logger.warning(f"多帧VLM调用失败(响应为空或错误)，降级重试: {len(frames)}帧 → {reduced}帧")
                step = len(frames) / reduced
                sampled = [frames[int(i * step)] for i in range(reduced)]
                images_b64 = [image_to_base64(f, max_size=self.max_image_size) for f in sampled]
                prompt_reduced = self._build_comprehensive_prompt(title, reduced, context, audio_context, per_frame_subtitle)
                result = call_vision_api(
                    self.provider, images_b64, prompt_reduced,
                    model=self.model, api_key=self.api_key,
                )

            # 仍然失败，降级到单帧
            need_fallback = result.startswith("[ERROR]") or not result.strip()
            if need_fallback and len(frames) > 1:
                logger.warning("降级重试仍失败，使用单帧模式")
                image_b64 = image_to_base64(frames[0], max_size=self.max_image_size)
                prompt_single = self._build_comprehensive_prompt(title, 1, context, audio_context, per_frame_subtitle)
                result = call_vision_api(
                    self.provider, image_b64, prompt_single,
                    model=self.model, api_key=self.api_key,
                )
        else:
            image_b64 = image_to_base64(frames[0], max_size=self.max_image_size)
            result = call_vision_api(
                self.provider, image_b64, prompt,
                model=self.model, api_key=self.api_key,
            )

        # 记录VLM响应（用于调试）
        logger.debug(f"VLM响应: {result[:500] if result else '空'}")

        return self._parse_vision_response(result)

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

            # 绘制检测结果并保存
            try:
                data = np.fromfile(frame_path, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    # 始终使用YOLO绘制姿态
                    pose_result = self.yolo_detector.estimate_pose(frame)
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

    def _save_debug_summary(self, vlm_result: Dict, video_summary: Dict, debug_dir: Path):
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
        summary_file = debug_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"调试汇总已保存")

    def _build_comprehensive_prompt(self, title: str, n_frames: int, context: str, audio_context: str = "", per_frame_subtitle: str = "") -> str:
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

        return f"""分析媒体文件 "{title}" 的{n_frames}个关键帧。

{context}
{audio_section}
{subtitle_section}

【任务说明】
这是一个媒体文件管理任务，需要对视频/图片进行客观的技术分析和元数据提取，用于文件归类和检索系统。

【输出要求】
1. 视频描述（2-3句话）：
   - 客观描述画面中的视觉元素
   - 包含：场景环境、人物外观特征、动作姿态
   
2. 关键词提取（4-8个，逗号分隔）：
   按以下维度提取客观视觉特征：

   第一优先级 - 水印/标识文字：
   - 画面中出现的人名、昵称、水印文字
   - 只提取人物名称，过滤掉网址、域名、@群组名等

   第二优先级 - 人物外观特征：
   - 服装类型：女仆装、校服、JK制服、旗袍、护士装、泳衣、内衣等
   - 服饰细节：丝袜、过膝袜、高跟鞋、蕾丝、蝴蝶结等
   - 颜色描述：黑色丝袜、白色衬衫、红色裙子等
   - 发型特征：双马尾、长发、短发、马尾辫、丸子头等

   第三优先级 - 姿态动作：
   - 基本姿势：站立、坐姿、跪姿、蹲姿、躺卧、弯腰
   - 动作描述：自拍、跳舞、行走、摆拍、转身等

   第四优先级 - 场景环境：
   - 室内/室外、卧室、浴室、客厅等
   - 背景特征

【格式要求】
请严格按以下格式返回，不要添加其他内容：
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
                return {
                    "description": f"[CLIP] {clip_result['tags']}",
                    "keywords": clip_result["tags"],
                    "source": "clip_only",
                }

        image_b64 = image_to_base64(compressed_path, max_size=self.max_image_size)
        result = call_vision_api(
            self.provider, image_b64, title,
            model=self.model, api_key=self.api_key,
        )

        if compressed_path != image_path and Path(compressed_path).exists():
            try:
                Path(compressed_path).unlink()
            except:
                pass

        return self._parse_vision_response(result)

    def _extract_frames(self, video_path: str) -> List[str]:
        """提取视频帧"""
        tmp_dir = Path("logs/_vision_tmp") / Path(video_path).stem
        tmp_dir.mkdir(parents=True, exist_ok=True)
        frames = []

        keyframes = detect_keyframes(video_path, str(tmp_dir), max_frames=8, max_size=self.max_image_size)
        frames.extend(keyframes)

        if len(frames) < self.vlm_frames:
            extra = extract_multiple_frames(
                video_path, str(tmp_dir),
                n_frames=self.vlm_frames - len(frames),
                max_size=self.max_image_size,
            )
            for f in extra:
                if f not in frames:
                    frames.append(f)

        # 临时文件保留用于调试，不自动清理
        # 如需清理可取消下面注释
        # import shutil
        # shutil.rmtree(tmp_dir, ignore_errors=True)

        return frames

    def _analyze_poses(self, frames: List[str]) -> List[Dict]:
        """分析姿态"""
        results = []
        for frame_path in frames:
            data = np.fromfile(frame_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                pose_result = self.yolo_detector.estimate_pose(frame)
                results.append(pose_result)
            else:
                results.append({"has_person": False})
        return results

    def _select_frames(self, frames: List[str], yolo_results: List[Dict]) -> List[str]:
        """
        分区段选择帧：将帧等分为 vlm_frames 个区段，
        每个区段内按 YOLO 评分选最优帧，保证全视频均匀覆盖。
        """
        n = len(frames)
        if n == 0:
            return []
        if n <= self.vlm_frames:
            return list(frames)

        if not yolo_results:
            # 无 YOLO 结果，均匀取帧
            step = n / self.vlm_frames
            return [frames[int(i * step)] for i in range(self.vlm_frames)]

        # 计算每帧评分
        scores = []
        prev_pose = None
        for result in yolo_results:
            score = 0.0
            if result.get("has_person"):
                conf = result.get("max_confidence", 0)
                score += conf * 40
                visible = result.get("poses", [{}])[0].get("visible_count", 0)
                score += (visible / 17) * 30
                current_pose = tuple(result.get("poses", [{}])[0].get("pose_analysis", []))
                if prev_pose and current_pose != prev_pose:
                    score += 30
                prev_pose = current_pose
            scores.append(score)

        # 等分区段，每段内取最高分帧
        seg_size = n / self.vlm_frames
        selected_indices = []

        for seg_idx in range(self.vlm_frames):
            start = int(seg_idx * seg_size)
            end = int((seg_idx + 1) * seg_size)
            if seg_idx == self.vlm_frames - 1:
                end = n

            best_score = -1
            best_i = start
            for i in range(start, end):
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_i = i
            selected_indices.append(best_i)

        selected_indices.sort()
        return [frames[i] for i in selected_indices if i < len(frames)]

    def _build_yolo_context(self, yolo_results: List[Dict], selected_frames: List[str]) -> str:
        """构建YOLO上下文"""
        context_lines = []

        frames_with_person = [r for r in yolo_results if r.get("has_person")]
        if frames_with_person:
            pose_counts = {}
            for r in frames_with_person:
                poses = r.get("poses", [{}])[0].get("pose_analysis", ["未知"])
                for pose in poses:
                    pose_counts[pose] = pose_counts.get(pose, 0) + 1

            main_pose = max(pose_counts, key=pose_counts.get) if pose_counts else "未知"
            context_lines.append(f"- 视频中检测到人体，主要姿态：{main_pose}")

            avg_visible = sum(r.get("poses", [{}])[0].get("visible_count", 0) for r in frames_with_person) / len(frames_with_person)
            avg_conf = sum(r.get("max_confidence", 0) for r in frames_with_person) / len(frames_with_person)
            context_lines.append(f"- 平均可见关键点：{avg_visible:.1f}/17，平均置信度：{avg_conf:.2f}")

        return "\n".join(context_lines) if context_lines else ""

    def _call_vlm(self, frames: List[str], title: str, yolo_context: str = None) -> Dict:
        """调用VLM，含帧数降级重试"""
        prompt = self._build_vision_prompt(title, len(frames), yolo_context)

        if len(frames) > 1:
            images_b64 = [image_to_base64(f, max_size=self.max_image_size) for f in frames]
            result = call_vision_api(
                self.provider, images_b64, prompt,
                model=self.model, api_key=self.api_key,
            )

            # 多帧失败或响应为空时降级：减少帧数重试
            need_fallback = result.startswith("[ERROR]") or not result.strip()
            if need_fallback and len(frames) > 2:
                reduced = max(2, len(frames) // 2)
                logger.warning(f"多帧VLM调用失败(响应为空或错误)，降级重试: {len(frames)}帧 → {reduced}帧")
                step = len(frames) / reduced
                sampled = [frames[int(i * step)] for i in range(reduced)]
                images_b64 = [image_to_base64(f, max_size=self.max_image_size) for f in sampled]
                prompt_reduced = self._build_vision_prompt(title, reduced, yolo_context)
                result = call_vision_api(
                    self.provider, images_b64, prompt_reduced,
                    model=self.model, api_key=self.api_key,
                )

            # 仍然失败，降级到单帧
            need_fallback = result.startswith("[ERROR]") or not result.strip()
            if need_fallback and len(frames) > 1:
                logger.warning("降级重试仍失败，使用单帧模式")
                image_b64 = image_to_base64(frames[0], max_size=self.max_image_size)
                prompt_single = self._build_vision_prompt(title, 1, yolo_context)
                result = call_vision_api(
                    self.provider, image_b64, prompt_single,
                    model=self.model, api_key=self.api_key,
                )
        else:
            image_b64 = image_to_base64(frames[0], max_size=self.max_image_size)
            result = call_vision_api(
                self.provider, image_b64, prompt,
                model=self.model, api_key=self.api_key,
            )

        return self._parse_vision_response(result)

    def _build_vision_prompt(self, title: str, n_frames: int, yolo_context: str = None) -> str:
        """构建VLM提示词"""
        yolo_section = ""
        if yolo_context:
            yolo_section = f"\n【视频预分析结果】\n{yolo_context}\n"

        keyword_rules = """
【关键词提取规则 - 重要】

关键词必须聚焦以下维度（按优先级排序）：

第一优先级 - 水印博主名字（最优先，必须包含）：
- 博主昵称/艺名（如：Sexy Yuki、UUbabydoll、ciyuanbb等）
- 只提取人物名称，不提取频道名、群组名
- 过滤掉：网址、域名、@群组名、频道链接、广告内容
- 过滤规则：包含 .com .cc .net .org @ http www 等的内容一律忽略

第二优先级 - 人物穿着（必须包含）：
- 服装类型：女仆装、校服、JK制服、旗袍、护士装、泳衣、内衣等
- 服饰细节：丝袜、过膝袜、高跟鞋、蕾丝、蝴蝶结等
- 颜色描述：黑色丝袜、白色衬衫、红色裙子等
- 发型特征：双马尾、长发、短发、马尾辫、丸子头等

第三优先级 - 姿势动作（必须包含）：
- 基本姿势：站立、坐姿、跪姿、蹲姿、躺卧、弯腰
- 动作描述：自拍、跳舞、行走、摆拍、转身等
- 姿态特征：弓背、张腿、侧卧等

【关键词格式要求】
- 如果有水印博主名字，必须放在第一个
- 必须包含至少2个穿着类关键词
- 必须包含至少1个姿势类关键词
- 总共4-8个关键词
- 使用中文，用逗号分隔"""

        if n_frames > 1:
            return (
                f'这是媒体文件 "{title}" 的{n_frames}个关键帧截图，用于文件管理归类。\n'
                f"{yolo_section}\n"
                f"请综合所有帧，客观描述画面中的可见元素。\n"
                f"{keyword_rules}\n\n"
                f"请严格按以下格式返回：\n"
                f"描述：[重点描述穿着、姿势、行为]\n"
                f"关键词：[穿着1, 穿着2, 姿势1, 行为1, ...]"
            )
        else:
            return (
                f'这是媒体文件 "{title}" 的截图，用于文件管理归类。\n'
                f"{yolo_section}\n"
                f"请客观描述画面中的可见元素。\n"
                f"{keyword_rules}\n\n"
                f"请严格按以下格式返回：\n"
                f"描述：[重点描述穿着、姿势、行为]\n"
                f"关键词：[穿着1, 穿着2, 姿势1, 行为1, ...]"
            )

    def _parse_vision_response(self, response: str) -> Dict:
        """解析VLM响应"""
        result = {"description": "", "keywords": ""}
        
        if not response:
            logger.warning("VLM响应为空")
            return result
        
        # 尝试多种格式解析
        # 格式1: "描述：xxx\n关键词：xxx"
        desc_match = re.search(r"描述[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL)
        kw_match = re.search(r"关键词[：:]\s*(.+?)(?:\n|$)", response, re.DOTALL)
        
        if desc_match:
            result["description"] = desc_match.group(1).strip()
        if kw_match:
            result["keywords"] = kw_match.group(1).strip()
        
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
        kw_list = kw_list[:8]  # 最多8个关键词

        if not kw_list:
            return original_title

        prefix = "_".join(kw_list)
        return f"[{prefix}]_{original_title}"

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
        from pathlib import Path

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

        # 处理视频（传入音频上下文和字幕段）
        result = self.process_video(video_path, title, audio_context, subtitle_segments)

        if "error" in result:
            return result

        description = result.get("description", "")
        keywords = result.get("keywords", "")
        video_summary = result.get("video_summary", {})

        # 生成final_name
        final_name = self.generate_final_name(keywords, original_title)

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

        # 复制SRT到视频所在目录
        if srt_path and Path(srt_path).exists():
            video_dir = Path(video_path).parent
            srt_filename = Path(srt_path).name
            srt_in_video_dir = video_dir / srt_filename
            
            # 如果目标路径与源路径不同，则复制
            if str(Path(srt_path).resolve()) != str(srt_in_video_dir.resolve()):
                import shutil
                try:
                    shutil.copy2(srt_path, srt_in_video_dir)
                    logger.info(f"SRT已复制到视频目录: {srt_in_video_dir}")
                except Exception as e:
                    logger.warning(f"复制SRT到视频目录失败: {e}")

        return {
            "description": description,
            "keywords": keywords,
            "final_name": final_name,
            "srt_path": srt_path,
            "video_summary": video_summary,
            "debug_dir": result.get("debug_dir"),
        }

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

    def _append_audio_subtitles(self, video_path: str, srt_path: str) -> str:
        """
        追加音频字幕到SRT文件
        
        Args:
            video_path: 视频路径
            srt_path: 现有SRT文件路径
        
        Returns:
            更新后的SRT文件路径
        """
        try:
            from ..utils.audio import AudioProcessor, load_audio_config
            
            logger.info("开始生成音频字幕...")
            
            # 每次都重新加载配置（确保GUI修改后及时生效）
            audio_config = load_audio_config()
            logger.info(f"音频配置: 自适应分段={audio_config['adaptive_enabled']}, 静音跳过={audio_config['skip_silence']}, 阈值={audio_config['volume_threshold']}")
            
            # 使用配置初始化处理器
            audio_processor = AudioProcessor(provider="mimo", config=audio_config)
            
            # 生成临时音频字幕
            temp_srt = srt_path + ".audio.tmp"
            audio_result = audio_processor.process_video(video_path, output_srt=temp_srt)
            
            if audio_result and Path(temp_srt).exists():
                # 读取音频字幕内容
                with open(temp_srt, "r", encoding="utf-8") as f:
                    audio_content = f.read()
                
                # 追加到现有SRT文件
                with open(srt_path, "a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(audio_content)
                
                logger.info(f"音频字幕已追加到: {srt_path}")
                
                # 清理临时文件
                try:
                    Path(temp_srt).unlink()
                except:
                    pass
            else:
                logger.warning("音频字幕生成失败或无内容")
            
        except Exception as e:
            logger.error(f"音频字幕处理失败: {e}")
        
        return srt_path
