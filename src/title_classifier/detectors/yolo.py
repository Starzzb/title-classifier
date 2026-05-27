"""YOLOv8检测器 - 姿态检测专用"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

# YOLO 模型配置
YOLO_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "yolo"
YOLO_POSE_MODEL = YOLO_MODEL_DIR / "yolov8n-pose.pt"

# COCO 数据集人体类别 ID
PERSON_CLASS_ID = 0

# 关键点名称（COCO 17点格式）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class YOLODetector(BaseDetector):
    """YOLOv8姿态检测器"""

    def __init__(
        self,
        model_type: str = "pose",
        device: str = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        super().__init__(confidence)
        self.model_type = "pose"  # 固定使用pose模型
        self.iou_threshold = iou_threshold
        self.device = device or self._detect_device()
        self._yolo_model = None

    def _detect_device(self) -> str:
        """自动检测推理设备"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 4:
                    return "cuda"
                else:
                    logger.warning(f"显存不足 ({gpu_memory:.1f}GB)，使用CPU推理")
                    return "cpu"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def load_model(self) -> bool:
        """加载YOLO姿态模型"""
        if self._loaded:
            return True

        try:
            from ultralytics import YOLO

            model_path = YOLO_POSE_MODEL
            logger.info(f"加载YOLO姿态模型: {model_path} (设备: {self.device})")

            if not model_path.exists():
                logger.error(f"YOLO模型文件不存在: {model_path}")
                return False

            self._yolo_model = YOLO(str(model_path))

            if self.device == "cuda":
                self._yolo_model.to("cuda")

            self._loaded = True
            logger.info("YOLO姿态模型加载完成")
            return True

        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测帧中的人体"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "persons": [], "max_confidence": 0.0}

        try:
            results = self._yolo_model(
                frame, conf=self.confidence, iou=self.iou_threshold, verbose=False
            )

            persons = []
            max_conf = 0.0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    h_img, w_img = frame.shape[:2]
                    cx = (x1 + x2) / 2 / w_img
                    cy = (y1 + y2) / 2 / h_img
                    w = (x2 - x1) / w_img
                    h = (y2 - y1) / h_img

                    persons.append({
                        "bbox": [cx, cy, w, h],
                        "bbox_pixel": [x1, y1, x2, y2],
                        "confidence": conf,
                    })

                    if conf > max_conf:
                        max_conf = conf

            return {
                "has_person": len(persons) > 0,
                "persons": persons,
                "max_confidence": max_conf,
            }

        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return {"has_person": False, "persons": [], "max_confidence": 0.0}

    def estimate_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """估计人体姿态"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "poses": [], "max_confidence": 0.0}

        try:
            results = self._yolo_model(
                frame, conf=self.confidence, iou=self.iou_threshold, verbose=False
            )

            poses = []
            max_conf = 0.0

            for result in results:
                if result.keypoints is None:
                    continue

                for i, kpts in enumerate(result.keypoints):
                    kpts_data = kpts.data[0].cpu().numpy()

                    bbox = None
                    if result.boxes is not None and i < len(result.boxes):
                        box = result.boxes[i]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h_img, w_img = frame.shape[:2]

                        bbox = {
                            "bbox": [
                                (x1 + x2) / 2 / w_img,
                                (y1 + y2) / 2 / h_img,
                                (x2 - x1) / w_img,
                                (y2 - y1) / h_img,
                            ],
                            "bbox_pixel": [x1, y1, x2, y2],
                            "confidence": conf,
                        }

                    keypoints = []
                    for j, (x, y, conf_kpt) in enumerate(kpts_data):
                        keypoints.append({
                            "name": KEYPOINT_NAMES[j],
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf_kpt),
                            "visible": bool(conf_kpt > 0.5),
                        })

                    valid_kpts = [k for k in keypoints if k["visible"]]
                    avg_conf = sum(k["confidence"] for k in valid_kpts) / len(valid_kpts) if valid_kpts else 0.0

                    # 生成关键点字典用于姿态分析
                    keypoints_dict = {k["name"]: {"x": k["x"], "y": k["y"], "conf": k["confidence"]} 
                                     for k in valid_kpts}
                    
                    # 分析姿态
                    pose_analysis = analyze_pose_for_vlm(keypoints_dict)

                    pose_info = {
                        "keypoints": keypoints,
                        "keypoints_dict": keypoints_dict,
                        "bbox": bbox,
                        "avg_confidence": avg_conf,
                        "visible_count": len(valid_kpts),
                        "pose_analysis": pose_analysis,
                    }

                    poses.append(pose_info)

                    if avg_conf > max_conf:
                        max_conf = avg_conf

            return {
                "has_person": len(poses) > 0,
                "poses": poses,
                "max_confidence": max_conf,
            }

        except Exception as e:
            logger.error(f"YOLO姿态估计失败: {e}")
            return {"has_person": False, "poses": [], "max_confidence": 0.0}

    def detect_and_crop(self, frame: np.ndarray, padding: float = 0.15) -> Dict[str, Any]:
        """检测并裁剪人体区域"""
        result = self.detect(frame)
        if result["has_person"]:
            best_person = max(result["persons"], key=lambda x: x["confidence"])

            x1, y1, x2, y2 = best_person["bbox_pixel"]
            h, w = frame.shape[:2]

            bw = x2 - x1
            bh = y2 - y1
            x1 = max(0, int(x1 - padding * bw))
            y1 = max(0, int(y1 - padding * bh))
            x2 = min(w, int(x2 + padding * bw))
            y2 = min(h, int(y2 + padding * bh))

            cropped = frame[y1:y2, x1:x2]

            return {
                "has_human": True,
                "max_confidence": best_person["confidence"],
                "human_crop": cropped,
                "bbox": best_person["bbox"],
                "persons": result["persons"],
            }

        return {
            "has_human": False,
            "max_confidence": 0.0,
            "human_crop": None,
            "bbox": None,
            "persons": [],
        }


def analyze_pose_for_vlm(keypoints: dict) -> list:
    """
    分析姿态，返回姿态描述列表（用于VLM上下文）

    Args:
        keypoints: 关键点字典 {name: {x, y, conf}}

    Returns:
        姿态描述列表
    """
    analysis = []

    if "left_shoulder" in keypoints and "left_hip" in keypoints:
        shoulder_y = keypoints["left_shoulder"]["y"]
        hip_y = keypoints["left_hip"]["y"]
        if shoulder_y > hip_y:
            analysis.append("弯腰/前倾")

    if "left_knee" in keypoints and "left_hip" in keypoints:
        knee_y = keypoints["left_knee"]["y"]
        hip_y = keypoints["left_hip"]["y"]
        if knee_y < hip_y + 50:
            analysis.append("跪姿/蹲姿")

    if "left_ankle" in keypoints and "left_knee" in keypoints:
        ankle_y = keypoints["left_ankle"]["y"]
        knee_y = keypoints["left_knee"]["y"]
        if abs(ankle_y - knee_y) < 30:
            analysis.append("坐姿")

    if "left_ear" in keypoints and "right_ear" in keypoints:
        left_x = keypoints["left_ear"]["x"]
        right_x = keypoints["right_ear"]["x"]
        if left_x > right_x + 20:
            analysis.append("右侧朝向")
        elif right_x > left_x + 20:
            analysis.append("左侧朝向")

    if not analysis:
        analysis.append("站立/正常姿态")

    return analysis


# 保持向后兼容
def detect_human_yolo(frame_path: str, model_type: str = "pose", conf_threshold: float = 0.5) -> Dict:
    """检测人体（向后兼容）"""
    detector = YOLODetector(model_type=model_type, confidence=conf_threshold)
    return detector.detect_from_file(frame_path)


def find_human_frame_yolo(
    video_path: str,
    model_type: str = "pose",
    step_seconds: float = 5.0,
    max_retries: int = 3,
    conf_threshold: float = 0.5,
) -> Dict:
    """从视频中找到有人体的帧（向后兼容）"""
    import subprocess
    import hashlib

    detector = YOLODetector(model_type=model_type, confidence=conf_threshold)

    # 获取视频时长
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace",
        )
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except:
        duration = 0.0

    if duration <= 0:
        return {"found": False, "frame_path": None, "timestamp": None, "max_confidence": 0.0, "detections": [], "attempts": 0}

    tmp_dir = Path("logs/_yolo_frame_selector_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]

    for attempt in range(max_retries):
        timestamp = attempt * step_seconds
        frame_path = str(tmp_dir / f"frame_{video_hash}_{attempt}.jpg")

        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", timestamp_str, "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", frame_path],
                capture_output=True, timeout=15, encoding="utf-8", errors="replace",
            )

            if result.returncode != 0 or not Path(frame_path).exists():
                continue

            det_result = detector.detect_from_file(frame_path)

            if det_result["has_person"]:
                return {
                    "found": True,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "max_confidence": det_result["max_confidence"],
                    "detections": det_result["persons"],
                    "attempts": attempt + 1,
                }

            try:
                Path(frame_path).unlink()
            except:
                pass

        except Exception as e:
            logger.error(f"帧提取失败: {e}")
            continue

    return {"found": False, "frame_path": None, "timestamp": None, "max_confidence": 0.0, "detections": [], "attempts": max_retries}
