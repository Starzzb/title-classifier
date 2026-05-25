"""
YOLOv8 集成检测模块
支持目标检测、姿态估计、实例分割和多目标跟踪
替代原有的 UHD 超轻量人体检测模型
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO 模型配置
YOLO_MODELS = {
    "detect": "yolov8n.pt",      # 目标检测 (~6MB)
    "pose": "yolov8n-pose.pt",   # 姿态估计 (~13MB)
    "segment": "yolov8n-seg.pt", # 实例分割 (~12MB)
}

# COCO 数据集人体类别 ID
PERSON_CLASS_ID = 0

# 关键点名称（COCO 17点格式）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点连接关系（用于绘制骨架）
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上半身
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)  # 下半身
]


class YOLODetector:
    """
    YOLOv8 多功能检测器
    支持检测、姿态估计、实例分割和跟踪
    """
    
    def __init__(self, model_type: str = "detect", device: str = None, 
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        初始化 YOLO 检测器
        
        Args:
            model_type: 模型类型 ("detect", "pose", "segment")
            device: 推理设备 ("cuda", "cpu", None自动检测)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or self._detect_device()
        self.model = None
        self._model_loaded = False
        
    def _detect_device(self) -> str:
        """自动检测推理设备"""
        try:
            import torch
            if torch.cuda.is_available():
                # 检查显存大小
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 4:  # 4GB以上使用GPU
                    return "cuda"
                else:
                    logger.warning(f"显存不足 ({gpu_memory:.1f}GB)，使用CPU推理")
                    return "cpu"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def load_model(self) -> bool:
        """
        加载 YOLO 模型
        
        Returns:
            是否加载成功
        """
        if self._model_loaded:
            return True
            
        try:
            from ultralytics import YOLO
            
            model_name = YOLO_MODELS.get(self.model_type, YOLO_MODELS["detect"])
            logger.info(f"[YOLO] 加载模型: {model_name} (设备: {self.device})")
            
            # 加载模型（自动下载）
            self.model = YOLO(model_name)
            
            # 设置推理设备
            if self.device == "cuda":
                self.model.to("cuda")
            
            self._model_loaded = True
            logger.info(f"[YOLO] 模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"[YOLO] 模型加载失败: {e}")
            return False
    
    def detect_persons(self, frame: np.ndarray) -> Dict:
        """
        检测图像中的人体
        
        Args:
            frame: BGR格式的图像 (numpy数组)
            
        Returns:
            {
                "has_person": bool,
                "persons": List[Dict],  # 每个人体的信息
                "max_confidence": float,
                "count": int
            }
        """
        if not self._model_loaded and not self.load_model():
            return {"has_person": False, "persons": [], "max_confidence": 0.0, "count": 0}
        
        try:
            # 运行推理
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            persons = []
            max_conf = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    # 只保留人体类别
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue
                    
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 转换为中心点格式
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                    # 归一化坐标
                    h_img, w_img = frame.shape[:2]
                    cx_norm = cx / w_img
                    cy_norm = cy / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img
                    
                    persons.append({
                        "bbox": [cx_norm, cy_norm, w_norm, h_norm],  # 归一化中心点格式
                        "bbox_pixel": [x1, y1, x2, y2],  # 像素坐标
                        "confidence": conf,
                        "class_id": PERSON_CLASS_ID,
                        "class_name": "person"
                    })
                    
                    if conf > max_conf:
                        max_conf = conf
            
            return {
                "has_person": len(persons) > 0,
                "persons": persons,
                "max_confidence": max_conf,
                "count": len(persons)
            }
            
        except Exception as e:
            logger.error(f"[YOLO] 人体检测失败: {e}")
            return {"has_person": False, "persons": [], "max_confidence": 0.0, "count": 0}
    
    def estimate_pose(self, frame: np.ndarray) -> Dict:
        """
        估计图像中的人体姿态
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            {
                "has_person": bool,
                "poses": List[Dict],  # 每个人体的姿态
                "max_confidence": float
            }
        """
        if self.model_type != "pose":
            logger.warning("[YOLO] 姿态估计需要使用 pose 模型")
            # 尝试自动切换模型
            self.model_type = "pose"
            self._model_loaded = False
            if not self.load_model():
                return {"has_person": False, "poses": [], "max_confidence": 0.0}
        
        if not self._model_loaded and not self.load_model():
            return {"has_person": False, "poses": [], "max_confidence": 0.0}
        
        try:
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            poses = []
            max_conf = 0.0
            
            for result in results:
                if result.keypoints is None:
                    continue
                    
                for i, kpts in enumerate(result.keypoints):
                    # 获取关键点坐标和置信度
                    kpts_data = kpts.data[0].cpu().numpy()  # shape: (17, 3)
                    
                    # 获取对应的人体边界框
                    bbox = None
                    if result.boxes is not None and i < len(result.boxes):
                        box = result.boxes[i]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        h_img, w_img = frame.shape[:2]
                        
                        cx = (x1 + x2) / 2 / w_img
                        cy = (y1 + y2) / 2 / h_img
                        w = (x2 - x1) / w_img
                        h = (y2 - y1) / h_img
                        
                        bbox = {
                            "bbox": [cx, cy, w, h],
                            "bbox_pixel": [x1, y1, x2, y2],
                            "confidence": conf
                        }
                    
                    # 整理关键点信息
                    keypoints = []
                    for j, (x, y, conf_kpt) in enumerate(kpts_data):
                        keypoints.append({
                            "name": KEYPOINT_NAMES[j],
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf_kpt),
                            "visible": bool(conf_kpt > 0.5)
                        })
                    
                    # 计算平均置信度
                    valid_kpts = [k for k in keypoints if k["visible"]]
                    avg_conf = sum(k["confidence"] for k in valid_kpts) / len(valid_kpts) if valid_kpts else 0.0
                    
                    pose_info = {
                        "keypoints": keypoints,
                        "bbox": bbox,
                        "avg_confidence": avg_conf,
                        "visible_count": len(valid_kpts)
                    }
                    
                    poses.append(pose_info)
                    
                    if avg_conf > max_conf:
                        max_conf = avg_conf
            
            return {
                "has_person": len(poses) > 0,
                "poses": poses,
                "max_confidence": max_conf
            }
            
        except Exception as e:
            logger.error(f"[YOLO] 姿态估计失败: {e}")
            return {"has_person": False, "poses": [], "max_confidence": 0.0}
    
    def segment_persons(self, frame: np.ndarray) -> Dict:
        """
        对图像中的人体进行实例分割
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            {
                "has_person": bool,
                "segments": List[Dict],  # 每个人体的分割结果
                "max_confidence": float
            }
        """
        if self.model_type != "segment":
            logger.warning("[YOLO] 实例分割需要使用 segment 模型")
            self.model_type = "segment"
            self._model_loaded = False
            if not self.load_model():
                return {"has_person": False, "segments": [], "max_confidence": 0.0}
        
        if not self._model_loaded and not self.load_model():
            return {"has_person": False, "segments": [], "max_confidence": 0.0}
        
        try:
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            segments = []
            max_conf = 0.0
            
            for result in results:
                if result.masks is None or result.boxes is None:
                    continue
                    
                for i, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    # 只保留人体类别
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue
                    
                    conf = float(box.conf[0])
                    mask_data = mask.data[0].cpu().numpy()  # 二值mask
                    
                    # 获取边界框
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    h_img, w_img = frame.shape[:2]
                    
                    cx = (x1 + x2) / 2 / w_img
                    cy = (y1 + y2) / 2 / h_img
                    w = (x2 - x1) / w_img
                    h = (y2 - y1) / h_img
                    
                    # 裁剪人体区域（使用mask）
                    mask_uint8 = (mask_data * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_uint8, (w_img, h_img))
                    
                    # 应用mask裁剪
                    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_resized)
                    
                    # 裁剪边界框区域
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    cropped = masked_frame[y1_int:y2_int, x1_int:x2_int]
                    
                    segments.append({
                        "bbox": [cx, cy, w, h],
                        "bbox_pixel": [x1, y1, x2, y2],
                        "confidence": conf,
                        "mask": mask_resized,
                        "cropped": cropped,
                        "area": float(np.sum(mask_data > 0.5))
                    })
                    
                    if conf > max_conf:
                        max_conf = conf
            
            return {
                "has_person": len(segments) > 0,
                "segments": segments,
                "max_confidence": max_conf
            }
            
        except Exception as e:
            logger.error(f"[YOLO] 实例分割失败: {e}")
            return {"has_person": False, "segments": [], "max_confidence": 0.0}
    
    def detect_and_crop(self, frame: np.ndarray, padding: float = 0.15) -> Dict:
        """
        检测人体并裁剪区域（兼容原有接口）
        
        Args:
            frame: BGR格式的图像
            padding: 边界框扩展比例
            
        Returns:
            {
                "has_human": bool,
                "max_confidence": float,
                "human_crop": np.ndarray or None,
                "bbox": list or None,
                "persons": List[Dict]  # 所有检测到的人体
            }
        """
        # 根据模型类型选择检测方式
        if self.model_type == "segment":
            result = self.segment_persons(frame)
            if result["has_person"]:
                # 取最大面积的人体
                best_segment = max(result["segments"], key=lambda x: x["area"])
                return {
                    "has_human": True,
                    "max_confidence": best_segment["confidence"],
                    "human_crop": best_segment["cropped"],
                    "bbox": best_segment["bbox"],
                    "persons": result["segments"]
                }
        else:
            result = self.detect_persons(frame)
            if result["has_person"]:
                # 取最高置信度的人体
                best_person = max(result["persons"], key=lambda x: x["confidence"])
                
                # 裁剪人体区域
                x1, y1, x2, y2 = best_person["bbox_pixel"]
                h, w = frame.shape[:2]
                
                # 扩展边界框
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
                    "persons": result["persons"]
                }
        
        return {
            "has_human": False,
            "max_confidence": 0.0,
            "human_crop": None,
            "bbox": None,
            "persons": []
        }
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        全面分析帧图像（检测+姿态+分割）
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            综合分析结果
        """
        result = {
            "has_person": False,
            "detection": None,
            "pose": None,
            "segment": None,
            "best_crop": None,
            "summary": ""
        }
        
        # 1. 人体检测
        det_result = self.detect_persons(frame)
        result["detection"] = det_result
        
        if not det_result["has_person"]:
            result["summary"] = "未检测到人体"
            return result
        
        result["has_person"] = True
        
        # 2. 姿态估计（如果模型支持）
        if self.model_type == "pose":
            pose_result = self.estimate_pose(frame)
            result["pose"] = pose_result
        
        # 3. 实例分割（如果模型支持）
        if self.model_type == "segment":
            seg_result = self.segment_persons(frame)
            result["segment"] = seg_result
            
            if seg_result["has_person"]:
                best_seg = max(seg_result["segments"], key=lambda x: x["area"])
                result["best_crop"] = best_seg["cropped"]
        
        # 4. 生成摘要
        person_count = det_result["count"]
        max_conf = det_result["max_confidence"]
        
        summary_parts = [f"检测到 {person_count} 个人体"]
        summary_parts.append(f"最高置信度 {max_conf:.2f}")
        
        if result["pose"] and result["pose"]["has_person"]:
            best_pose = max(result["pose"]["poses"], key=lambda x: x["avg_confidence"])
            visible_kpts = best_pose["visible_count"]
            summary_parts.append(f"姿态关键点 {visible_kpts}/17")
        
        result["summary"] = ", ".join(summary_parts)
        
        return result


class YOLOTracker:
    """
    YOLO 多目标跟踪器
    用于视频中持续锁定并追踪人体
    """
    
    def __init__(self, model_type: str = "detect", tracker_type: str = "bytetrack.yaml"):
        """
        初始化跟踪器
        
        Args:
            model_type: 模型类型
            tracker_type: 跟踪器类型 ("bytetrack.yaml" 或 "botsort.yaml")
        """
        self.model_type = model_type
        self.tracker_type = tracker_type
        self.model = None
        self._model_loaded = False
        
    def load_model(self) -> bool:
        """加载模型"""
        if self._model_loaded:
            return True
            
        try:
            from ultralytics import YOLO
            
            model_name = YOLO_MODELS.get(self.model_type, YOLO_MODELS["detect"])
            logger.info(f"[YOLO Tracker] 加载模型: {model_name}")
            
            self.model = YOLO(model_name)
            self._model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"[YOLO Tracker] 模型加载失败: {e}")
            return False
    
    def track_video(self, video_path: str, max_frames: int = None) -> Dict:
        """
        跟踪视频中的人体
        
        Args:
            video_path: 视频路径
            max_frames: 最大处理帧数
            
        Returns:
            {
                "tracks": Dict[int, List[Dict]],  # track_id -> 帧列表
                "summary": Dict
            }
        """
        if not self._model_loaded and not self.load_model():
            return {"tracks": {}, "summary": {}}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"[YOLO Tracker] 无法打开视频: {video_path}")
                return {"tracks": {}, "summary": {}}
            
            tracks = {}
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames and frame_count >= max_frames:
                    break
                
                # 运行跟踪
                results = self.model.track(frame, persist=True, tracker=self.tracker_type, verbose=False)
                
                for result in results:
                    if result.boxes is None or result.boxes.id is None:
                        continue
                        
                    for box in result.boxes:
                        track_id = int(box.id[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        if cls_id != PERSON_CLASS_ID:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if track_id not in tracks:
                            tracks[track_id] = []
                        
                        tracks[track_id].append({
                            "frame": frame_count,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf
                        })
                
                frame_count += 1
            
            cap.release()
            
            # 生成摘要
            summary = {
                "total_frames": frame_count,
                "unique_persons": len(tracks),
                "avg_tracks_per_frame": sum(len(t) for t in tracks.values()) / max(1, frame_count)
            }
            
            return {"tracks": tracks, "summary": summary}
            
        except Exception as e:
            logger.error(f"[YOLO Tracker] 视频跟踪失败: {e}")
            return {"tracks": {}, "summary": {}}


# 便捷函数
def detect_human_yolo(frame_path: str, model_type: str = "detect", 
                      conf_threshold: float = 0.5) -> Dict:
    """
    使用YOLO检测图像中的人体（兼容原有接口）
    
    Args:
        frame_path: 图像路径
        model_type: 模型类型
        conf_threshold: 置信度阈值
        
    Returns:
        {
            "has_human": bool,
            "max_confidence": float,
            "detections": List[Dict]
        }
    """
    # 读取图像
    data = np.fromfile(frame_path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error(f"[YOLO] 无法读取图像: {frame_path}")
        return {"has_human": False, "max_confidence": 0.0, "detections": []}
    
    # 创建检测器
    detector = YOLODetector(model_type=model_type, conf_threshold=conf_threshold)
    
    # 检测人体
    result = detector.detect_persons(frame)
    
    # 转换为兼容格式
    detections = []
    for person in result["persons"]:
        detections.append({
            "score": person["confidence"],
            "bbox": person["bbox"],
            "bbox_pixel": person["bbox_pixel"]
        })
    
    return {
        "has_human": result["has_person"],
        "max_confidence": result["max_confidence"],
        "detections": detections
    }


def find_human_frame_yolo(video_path: str, model_type: str = "detect",
                          step_seconds: float = 5.0, max_retries: int = 3,
                          conf_threshold: float = 0.5) -> Dict:
    """
    从视频中找到第一帧包含人体的画面（兼容原有接口）
    
    Args:
        video_path: 视频路径
        model_type: 模型类型
        step_seconds: 帧提取步长
        max_retries: 最大重试次数
        conf_threshold: 置信度阈值
        
    Returns:
        {
            "found": bool,
            "frame_path": str or None,
            "timestamp": float or None,
            "max_confidence": float,
            "detections": list,
            "attempts": int
        }
    """
    import subprocess
    import hashlib
    
    # 获取视频时长
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except:
        duration = 0.0
    
    if duration <= 0:
        return {"found": False, "frame_path": None, "timestamp": None, 
                "max_confidence": 0.0, "detections": [], "attempts": 0}
    
    # 创建检测器
    detector = YOLODetector(model_type=model_type, conf_threshold=conf_threshold)
    
    # 临时目录
    tmp_dir = Path("logs/_yolo_frame_selector_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    
    for attempt in range(max_retries):
        timestamp = attempt * step_seconds
        
        # 提取帧
        frame_path = str(tmp_dir / f"frame_{video_hash}_{attempt}.jpg")
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", timestamp_str, "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", frame_path],
                capture_output=True, timeout=15
            )
            
            if result.returncode != 0 or not Path(frame_path).exists():
                continue
            
            # 检测人体
            det_result = detect_human_yolo(frame_path, model_type, conf_threshold)
            
            if det_result["has_human"]:
                logger.info(f"[YOLO] 检测到人体 @ {timestamp:.1f}s, 置信度: {det_result['max_confidence']:.2f}")
                return {
                    "found": True,
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "max_confidence": det_result["max_confidence"],
                    "detections": det_result["detections"],
                    "attempts": attempt + 1
                }
            
            # 清理临时文件
            try:
                Path(frame_path).unlink()
            except:
                pass
                
        except Exception as e:
            logger.error(f"[YOLO] 帧提取失败: {e}")
            continue
    
    return {"found": False, "frame_path": None, "timestamp": None, 
            "max_confidence": 0.0, "detections": [], "attempts": max_retries}


if __name__ == "__main__":
    """测试代码"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python stage1c_yolo_detector.py <图像/视频路径> [模型类型]")
        print("模型类型: detect (默认), pose, segment")
        sys.exit(1)
    
    path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "detect"
    
    if not Path(path).exists():
        print(f"错误: 文件不存在: {path}")
        sys.exit(1)
    
    # 判断是图像还是视频
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.ts'}
    is_video = Path(path).suffix.lower() in video_exts
    
    print(f"文件: {path}")
    print(f"类型: {'视频' if is_video else '图像'}")
    print(f"模型: {model_type}")
    print("=" * 50)
    
    if is_video:
        # 视频模式：查找有人帧
        result = find_human_frame_yolo(path, model_type=model_type)
        print(f"找到有人帧: {result['found']}")
        if result['found']:
            print(f"时间点: {result['timestamp']:.1f}s")
            print(f"置信度: {result['max_confidence']:.2f}")
            print(f"检测数量: {len(result['detections'])}")
    else:
        # 图像模式：检测人体
        result = detect_human_yolo(path, model_type=model_type)
        print(f"检测到人体: {result['has_human']}")
        if result['has_human']:
            print(f"置信度: {result['max_confidence']:.2f}")
            print(f"检测数量: {len(result['detections'])}")
            for i, det in enumerate(result['detections']):
                print(f"  人体 {i+1}: 置信度={det['score']:.2f}, 位置={det['bbox']}")
