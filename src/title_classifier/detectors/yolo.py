"""YOLOv8检测器 - 支持多模型（detect/pose/segment）+ OpenVINO 加速"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

# YOLO 模型配置
YOLO_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "yolo"
YOLO_MODELS = {
    "detect": YOLO_MODEL_DIR / "yolov8n.pt",
    "pose": YOLO_MODEL_DIR / "yolo11m-pose.pt",
    "segment": YOLO_MODEL_DIR / "yolov8n-seg.pt",
}

# OpenVINO 模型缓存目录
OPENVINO_MODEL_DIR = YOLO_MODEL_DIR / "openvino"

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
    """YOLOv8检测器 - 支持多模型 + OpenVINO 加速"""

    def __init__(
        self,
        model_types: List[str] = None,
        device: str = None,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        parallel_loading: bool = True,
        backend: str = "auto",
    ):
        """
        初始化YOLO检测器
        
        Args:
            model_types: 模型类型列表，可选 "detect", "pose", "segment"
            device: 推理设备 (cuda/cpu)
            confidence: 置信度阈值
            iou_threshold: IoU阈值
            parallel_loading: 是否并行加载模型
            backend: 推理后端 ("auto" / "openvino" / "pytorch")
        """
        super().__init__(confidence)
        self.model_types = model_types or ["pose"]
        self.iou_threshold = iou_threshold
        self.device = device or self._detect_device()
        self.parallel_loading = parallel_loading
        self.backend = self._resolve_backend(backend)
        self._models = {}  # 存储多个模型 {model_type: model}
        self._backend_type = {}  # 记录每个模型使用的后端 {model_type: "openvino"/"pytorch"}

    def _detect_device(self) -> str:
        """自动检测推理设备（CPU 优先）"""
        # 优先使用 CPU（当前优化目标）
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

    def _resolve_backend(self, backend: str) -> str:
        """解析推理后端：auto/openvino/pytorch"""
        if backend == "pytorch":
            return "pytorch"
        if backend == "openvino":
            if not self._check_openvino_available():
                logger.warning("OpenVINO 不可用，回退到 PyTorch")
                return "pytorch"
            return "openvino"
        # auto 模式：CPU 时优先 OpenVINO，CUDA 时使用 PyTorch
        if self.device == "cuda":
            return "pytorch"
        if self._check_openvino_available():
            logger.info("检测到 OpenVINO，使用 OpenVINO 后端加速 CPU 推理")
            return "openvino"
        return "pytorch"

    @staticmethod
    def _check_openvino_available() -> bool:
        """检查 OpenVINO 是否可用"""
        try:
            import openvino
            return True
        except ImportError:
            return False

    def _get_openvino_path(self, model_type: str) -> Path:
        """获取 OpenVINO 模型缓存路径"""
        return OPENVINO_MODEL_DIR / f"{model_type}_openvino_model"

    def _export_to_openvino(self, model_type: str, pt_path: Path) -> Path:
        """
        将 .pt 模型导出为 OpenVINO IR 格式（FP16）
        
        Args:
            model_type: 模型类型 (detect/pose/segment)
            pt_path: .pt 模型路径
            
        Returns:
            OpenVINO 模型目录路径
        """
        from ultralytics import YOLO
        
        openvino_path = self._get_openvino_path(model_type)
        
        logger.info(f"首次运行，正在将 {model_type} 导出为 OpenVINO FP16 格式...")
        logger.info(f"源模型: {pt_path}")
        logger.info(f"目标目录: {openvino_path}")
        
        try:
            # 加载 PyTorch 模型（必须指定 task，否则 pose/segment 模型导出时会丢失关键头）
            model = YOLO(str(pt_path), task=model_type)
            
            # 导出为 OpenVINO 格式 (FP16)
            export_dir = model.export(format="openvino", half=True)
            
            # ultralytics 导出后会创建一个目录，移动到我们的缓存位置
            export_path = Path(export_dir)
            if openvino_path.exists():
                shutil.rmtree(openvino_path)
            
            # 如果导出路径和目标路径不同，移动文件
            if export_path != openvino_path:
                shutil.move(str(export_path), str(openvino_path))
            
            logger.info(f"OpenVINO 模型导出完成: {openvino_path}")
            return openvino_path
            
        except Exception as e:
            logger.error(f"OpenVINO 导出失败: {e}")
            logger.warning("将回退使用 PyTorch 模型")
            raise

    def load_model(self) -> bool:
        """加载YOLO模型 - 支持 OpenVINO/ONNX Runtime/PyTorch"""
        if self._loaded:
            return True

        try:
            from ultralytics import YOLO

            loaded_count = 0
            for model_type in self.model_types:
                if model_type not in YOLO_MODELS:
                    logger.warning(f"未知的模型类型: {model_type}，跳过")
                    continue

                pt_path = YOLO_MODELS[model_type]
                openvino_path = self._get_openvino_path(model_type)
                
                model = None
                backend_used = "pytorch"

                # 优先尝试 OpenVINO
                if self.backend == "openvino":
                    try:
                        if openvino_path.exists():
                            # 直接加载缓存的 OpenVINO 模型目录
                            xml_files = list(openvino_path.glob("*.xml"))
                            if xml_files:
                                # ultralytics 需要加载目录而不是单个 .xml 文件
                                # 必须指定 task，否则 pose/segment 模型会被当作 detect 加载
                                logger.info(f"加载 OpenVINO {model_type} 模型: {openvino_path}")
                                model = YOLO(str(openvino_path), task=model_type)
                                backend_used = "openvino"
                            else:
                                raise FileNotFoundError(f"OpenVINO 模型目录中未找到 .xml 文件: {openvino_path}")
                        else:
                            # 首次运行：导出并缓存
                            if not pt_path.exists():
                                logger.error(f"YOLO 源模型文件不存在: {pt_path}")
                                continue
                            ov_path = self._export_to_openvino(model_type, pt_path)
                            xml_files = list(ov_path.glob("*.xml"))
                            if xml_files:
                                model = YOLO(str(ov_path), task=model_type)
                            else:
                                raise FileNotFoundError(f"导出后未找到 OpenVINO 模型文件: {ov_path}")
                            backend_used = "openvino"
                    except Exception as e:
                        logger.warning(f"OpenVINO 加载失败，回退到 PyTorch: {e}")
                        model = None

                # 回退到 PyTorch
                if model is None:
                    if not pt_path.exists():
                        logger.error(f"YOLO 模型文件不存在: {pt_path}")
                        continue
                    logger.info(f"加载 PyTorch {model_type} 模型: {pt_path} (设备: {self.device})")
                    model = YOLO(str(pt_path), task=model_type)
                    if self.device == "cuda":
                        model.to("cuda")
                    backend_used = "pytorch"

                self._models[model_type] = model
                self._backend_type[model_type] = backend_used
                loaded_count += 1
                logger.info(f"YOLO {model_type} 模型加载完成 (后端: {backend_used})")

            if loaded_count > 0:
                self._loaded = True
                backend_summary = ", ".join([f"{k}:{v}" for k, v in self._backend_type.items()])
                logger.info(f"成功加载 {loaded_count}/{len(self.model_types)} 个YOLO模型 [{backend_summary}]")
                return True
            else:
                logger.error("没有成功加载任何YOLO模型")
                return False

        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测帧中的人体（使用detect模型）"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "persons": [], "max_confidence": 0.0}

        # 优先使用detect模型，如果没有则使用pose模型
        model_type = "detect" if "detect" in self._backend_type else "pose"
        if model_type not in self._backend_type:
            logger.error("没有可用的检测模型")
            return {"has_person": False, "persons": [], "max_confidence": 0.0}

        try:
            model = self._models[model_type]
            results = model(
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
                        "model": model_type,
                    })

                    if conf > max_conf:
                        max_conf = conf

            return {
                "has_person": len(persons) > 0,
                "persons": persons,
                "max_confidence": max_conf,
                "model": model_type,
            }

        except Exception as e:
            logger.error(f"YOLO检测失败: {e}")
            return {"has_person": False, "persons": [], "max_confidence": 0.0}

    def estimate_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """估计人体姿态（使用pose模型）"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "poses": [], "max_confidence": 0.0}

        if "pose" not in self._backend_type:
            logger.error("pose模型未加载")
            return {"has_person": False, "poses": [], "max_confidence": 0.0}

        try:
            model = self._models["pose"]
            results = model(
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
                        "model": "pose",
                    }

                    poses.append(pose_info)

                    if avg_conf > max_conf:
                        max_conf = avg_conf

            return {
                "has_person": len(poses) > 0,
                "poses": poses,
                "max_confidence": max_conf,
                "model": "pose",
            }

        except Exception as e:
            logger.error(f"YOLO姿态估计失败: {e}")
            return {"has_person": False, "poses": [], "max_confidence": 0.0}

    def segment_instances(self, frame: np.ndarray) -> Dict[str, Any]:
        """实例分割（使用segment模型）"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "segments": [], "max_confidence": 0.0}

        if "segment" not in self._models:
            logger.error("segment模型未加载")
            return {"has_person": False, "segments": [], "max_confidence": 0.0}

        try:
            model = self._models["segment"]
            results = model(
                frame, conf=self.confidence, iou=self.iou_threshold, verbose=False
            )

            segments = []
            max_conf = 0.0

            for result in results:
                if result.masks is None or result.boxes is None:
                    continue

                for i, (mask, box) in enumerate(zip(result.masks, result.boxes)):
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    h_img, w_img = frame.shape[:2]

                    # 获取掩码
                    mask_data = mask.data[0].cpu().numpy()
                    
                    # 计算掩码面积比例
                    mask_area = np.sum(mask_data > 0.5)
                    total_area = h_img * w_img
                    mask_ratio = mask_area / total_area if total_area > 0 else 0

                    # 分析穿着（基于掩码区域的颜色分布）
                    wearing_analysis = analyze_wearing_from_mask(frame, mask_data)

                    segment_info = {
                        "bbox": [
                            (x1 + x2) / 2 / w_img,
                            (y1 + y2) / 2 / h_img,
                            (x2 - x1) / w_img,
                            (y2 - y1) / h_img,
                        ],
                        "bbox_pixel": [x1, y1, x2, y2],
                        "confidence": conf,
                        "mask_ratio": mask_ratio,
                        "wearing_analysis": wearing_analysis,
                        "model": "segment",
                    }

                    segments.append(segment_info)

                    if conf > max_conf:
                        max_conf = conf

            return {
                "has_person": len(segments) > 0,
                "segments": segments,
                "max_confidence": max_conf,
                "model": "segment",
            }

        except Exception as e:
            logger.error(f"YOLO分割失败: {e}")
            return {"has_person": False, "segments": [], "max_confidence": 0.0}

    def analyze_comprehensive(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        全面分析帧 - 使用所有可用模型
        
        Returns:
            {
                "has_person": bool,
                "detection": dict,  # detect模型结果
                "pose": dict,       # pose模型结果
                "segment": dict,    # segment模型结果
                "merged": dict,     # 合并后的结果
                "confidence": float, # 加权置信度
                "models_used": list, # 使用的模型列表
            }
        """
        if not self._loaded:
            if not self.load_model():
                return {
                    "has_person": False,
                    "detection": None,
                    "pose": None,
                    "segment": None,
                    "merged": None,
                    "confidence": 0.0,
                    "models_used": [],
                }

        results = {
            "has_person": False,
            "detection": None,
            "pose": None,
            "segment": None,
            "merged": None,
            "confidence": 0.0,
            "models_used": [],
        }

        # 并行运行三个模型
        if "detect" in self._models:
            try:
                logger.debug("[DEBUG] detect模型开始推理")
                results["detection"] = self.detect(frame)
                logger.debug("[DEBUG] detect模型推理完成")
                results["models_used"].append("detect")
            except Exception as e:
                logger.warning(f"detect模型推理失败: {e}")

        if "pose" in self._models:
            try:
                logger.debug("[DEBUG] pose模型开始推理")
                results["pose"] = self.estimate_pose(frame)
                logger.debug("[DEBUG] pose模型推理完成")
                results["models_used"].append("pose")
            except Exception as e:
                logger.warning(f"pose模型推理失败: {e}")

        if "segment" in self._models:
            try:
                logger.debug("[DEBUG] segment模型开始推理")
                results["segment"] = self.segment_instances(frame)
                logger.debug("[DEBUG] segment模型推理完成")
                results["models_used"].append("segment")
            except Exception as e:
                logger.warning(f"segment模型推理失败: {e}")

        # CUDA显存清理
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 合并结果
        results["merged"] = self._merge_results(
            results["detection"],
            results["pose"],
            results["segment"]
        )

        # 计算加权置信度
        results["confidence"] = self._calculate_weighted_confidence(
            results["detection"],
            results["pose"],
            results["segment"]
        )

        # 判断是否有人体
        results["has_person"] = results["merged"].get("has_person", False)

        return results

    def _merge_results(
        self,
        detection: Optional[Dict],
        pose: Optional[Dict],
        segment: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        合并三个模型的结果
        
        Args:
            detection: detect模型结果
            pose: pose模型结果
            segment: segment模型结果
            
        Returns:
            合并后的结果
        """
        merged = {
            "has_person": False,
            "persons": [],
            "poses": [],
            "segments": [],
            "detection_details": [],
            "pose_details": [],
            "segment_details": [],
        }

        # 收集有人体的结果
        if detection and detection.get("has_person"):
            merged["has_person"] = True
            merged["persons"] = detection.get("persons", [])
            merged["detection_details"] = detection.get("persons", [])

        if pose and pose.get("has_person"):
            merged["has_person"] = True
            merged["poses"] = pose.get("poses", [])
            merged["pose_details"] = pose.get("poses", [])

        if segment and segment.get("has_person"):
            merged["has_person"] = True
            merged["segments"] = segment.get("segments", [])
            merged["segment_details"] = segment.get("segments", [])

        # 投票决策：至少两个模型检测到人体才认为有人体
        vote_count = sum([
            1 if detection and detection.get("has_person") else 0,
            1 if pose and pose.get("has_person") else 0,
            1 if segment and segment.get("has_person") else 0,
        ])
        merged["vote_count"] = vote_count
        merged["vote_result"] = vote_count >= 2

        return merged

    def _calculate_weighted_confidence(
        self,
        detection: Optional[Dict],
        pose: Optional[Dict],
        segment: Optional[Dict],
    ) -> float:
        """
        计算加权置信度（根据置信度动态调整权重）
        
        Args:
            detection: detect模型结果
            pose: pose模型结果
            segment: segment模型结果
            
        Returns:
            加权置信度
        """
        confidences = []
        
        if detection and detection.get("has_person"):
            confidences.append(detection.get("max_confidence", 0.0))
        
        if pose and pose.get("has_person"):
            confidences.append(pose.get("max_confidence", 0.0))
        
        if segment and segment.get("has_person"):
            confidences.append(segment.get("max_confidence", 0.0))

        if not confidences:
            return 0.0

        # 动态权重：根据置信度分配权重
        total_conf = sum(confidences)
        if total_conf == 0:
            return 0.0

        # 加权平均
        weights = [c / total_conf for c in confidences]
        weighted_conf = sum(c * w for c, w in zip(confidences, weights))

        return weighted_conf

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
    使用人体自身尺寸（躯干长度）做归一化，兼容不同分辨率和镜头距离。

    Args:
        keypoints: 关键点字典 {name: {x, y, conf}}

    Returns:
        姿态描述列表
    """
    analysis = []

    # 计算人体参考尺度：躯干长度 = shoulder_y - hip_y（取双侧平均）
    def _avg_y(name_a, name_b):
        a = keypoints.get(name_a)
        b = keypoints.get(name_b)
        if a and b:
            return (a["y"] + b["y"]) / 2
        return a["y"] if a else (b["y"] if b else None)

    def _avg_x(name_a, name_b):
        a = keypoints.get(name_a)
        b = keypoints.get(name_b)
        if a and b:
            return (a["x"] + b["x"]) / 2
        return a["x"] if a else (b["x"] if b else None)

    shoulder_y = _avg_y("left_shoulder", "right_shoulder")
    hip_y = _avg_y("left_hip", "right_hip")

    if shoulder_y is None or hip_y is None:
        return ["站立/正常姿态"]

    torso = hip_y - shoulder_y
    if torso <= 0 or torso < 10:  # 最小10px躯干，防止误判
        return ["站立/正常姿态"]

    # --- 1. 弯腰/前倾 ---
    # 肩膀中点接近或超过髋部中点
    if shoulder_y > hip_y - torso * 0.2:
        analysis.append("弯腰/前倾")

    # --- 2. 跪姿/蹲姿 ---
    # 膝盖接近髋部高度
    knee_y = _avg_y("left_knee", "right_knee")
    if knee_y is not None and knee_y < hip_y + torso * 0.5:
        analysis.append("跪姿/蹲姿")

    # --- 3. 坐姿 ---
    # 膝盖与脚踝接近同一水平线
    ankle_y = _avg_y("left_ankle", "right_ankle")
    if knee_y is not None and ankle_y is not None:
        if abs(ankle_y - knee_y) < torso * 0.3:
            analysis.append("坐姿")

    # --- 4. 朝向 ---
    le = keypoints.get("left_ear")
    re = keypoints.get("right_ear")
    if le and re:
        dx = le["x"] - re["x"]
        if dx > 10:
            analysis.append("右侧朝向")
        elif dx < -10:
            analysis.append("左侧朝向")

    # --- 5. 手臂抬起 ---
    # 手腕高于肩膀
    wrist_y = _avg_y("left_wrist", "right_wrist")
    if wrist_y is not None and shoulder_y is not None:
        if wrist_y < shoulder_y:
            analysis.append("手臂抬起")

    # --- 6. 躺卧 ---
    # 关键点水平分布，垂直范围很小
    if torso > 0:
        all_y = [kp["y"] for kp in keypoints.values() if "y" in kp]
        all_x = [kp["x"] for kp in keypoints.values() if "x" in kp]
        if len(all_y) >= 6:
            vertical_range = max(all_y) - min(all_y)
            horizontal_range = max(all_x) - min(all_x)
            if vertical_range < torso * 0.5 and horizontal_range > vertical_range * 2:
                analysis.append("躺卧")

    # --- 7. 双腿张开 ---
    # 左右脚踝x距离大于躯干长度
    la = keypoints.get("left_ankle")
    ra = keypoints.get("right_ankle")
    if la and ra:
        ankle_x_dist = abs(la["x"] - ra["x"])
        if ankle_x_dist > torso * 1.2:
            analysis.append("双腿张开")

    # --- 8. 弓背/驼背 ---
    # 肩膀明显低于髋部（比弯腰更严重）
    if shoulder_y > hip_y + torso * 0.3:
        analysis.append("弓背/驼背")

    if not analysis:
        analysis.append("站立/正常姿态")

    return analysis


def analyze_wearing_from_mask(frame: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    基于掩码分析穿着
    
    Args:
        frame: 原始帧
        mask: 人体掩码
        
    Returns:
        穿着分析结果
    """
    try:
        # 将掩码转换为二值图
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # 提取人体区域
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 计算掩码区域的颜色分布
        mask_bool = mask_resized > 0
        if not np.any(mask_bool):
            return {"has_wearing": False}
        
        # 提取人体区域的颜色
        body_region = frame[mask_bool]
        
        # 计算平均颜色
        avg_color = np.mean(body_region, axis=0)
        
        # 计算颜色方差（用于判断是否有多彩衣物）
        color_std = np.std(body_region, axis=0)
        
        # 简单的穿着分析
        wearing_analysis = {
            "has_wearing": True,
            "avg_color_bgr": avg_color.tolist(),
            "color_std": color_std.tolist(),
            "color_variance": float(np.mean(color_std)),
        }
        
        return wearing_analysis
        
    except Exception as e:
        logger.warning(f"穿着分析失败: {e}")
        return {"has_wearing": False}


def draw_pose_on_frame(frame: np.ndarray, pose_result: Dict) -> np.ndarray:
    """在帧上绘制姿态检测结果"""
    annotated = frame.copy()

    if not pose_result.get("has_person") or not pose_result.get("poses"):
        return annotated

    for pose in pose_result["poses"]:
        bbox = pose.get("bbox")
        keypoints = pose.get("keypoints", [])

        # 绘制边界框
        if bbox and "bbox_pixel" in bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox["bbox_pixel"]]
            conf = bbox.get("confidence", 0)
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制关键点
        visible_kpts = []
        for kpt in keypoints:
            if kpt.get("visible") and kpt.get("confidence", 0) > 0.5:
                x, y = int(kpt["x"]), int(kpt["y"])
                visible_kpts.append((x, y, kpt["name"]))
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

        # 绘制骨架连接线
        skeleton = [
            ("nose", "left_eye"), ("nose", "right_eye"),
            ("left_eye", "left_ear"), ("right_eye", "right_ear"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ]

        kpt_dict = {k[2]: (k[0], k[1]) for k in visible_kpts}
        for k1, k2 in skeleton:
            if k1 in kpt_dict and k2 in kpt_dict:
                cv2.line(annotated, kpt_dict[k1], kpt_dict[k2], (255, 255, 0), 2)

        # 绘制姿态分析结果
        pose_analysis = pose.get("pose_analysis", [])
        if pose_analysis and bbox and "bbox_pixel" in bbox:
            x1, y1, _, _ = [int(v) for v in bbox["bbox_pixel"]]
            text = ", ".join(pose_analysis)
            cv2.putText(annotated, text, (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def draw_comprehensive_on_frame(frame: np.ndarray, comprehensive_result: Dict) -> np.ndarray:
    """
    在帧上绘制全面分析结果
    
    Args:
        frame: 原始帧
        comprehensive_result: 全面分析结果
        
    Returns:
        标注后的帧
    """
    annotated = frame.copy()
    
    if not comprehensive_result.get("has_person"):
        return annotated
    
    # 绘制检测结果（红色边框）
    detection = comprehensive_result.get("detection")
    if detection and detection.get("has_person"):
        for person in detection.get("persons", []):
            if "bbox_pixel" in person:
                x1, y1, x2, y2 = [int(v) for v in person["bbox_pixel"]]
                conf = person.get("confidence", 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"Det {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制姿态结果（绿色边框+关键点）
    pose = comprehensive_result.get("pose")
    if pose and pose.get("has_person"):
        for pose_info in pose.get("poses", []):
            bbox = pose_info.get("bbox")
            if bbox and "bbox_pixel" in bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox["bbox_pixel"]]
                conf = bbox.get("confidence", 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Pose {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制关键点
            for kpt in pose_info.get("keypoints", []):
                if kpt.get("visible") and kpt.get("confidence", 0) > 0.5:
                    x, y = int(kpt["x"]), int(kpt["y"])
                    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
    
    # 绘制分割结果（蓝色边框）
    segment = comprehensive_result.get("segment")
    if segment and segment.get("has_person"):
        for seg_info in segment.get("segments", []):
            if "bbox_pixel" in seg_info:
                x1, y1, x2, y2 = [int(v) for v in seg_info["bbox_pixel"]]
                conf = seg_info.get("confidence", 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"Seg {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 显示投票结果
    merged = comprehensive_result.get("merged", {})
    vote_count = merged.get("vote_count", 0)
    vote_result = merged.get("vote_result", False)
    confidence = comprehensive_result.get("confidence", 0)
    
    status_text = f"Vote: {vote_count}/3, Conf: {confidence:.2f}"
    color = (0, 255, 0) if vote_result else (0, 0, 255)
    cv2.putText(annotated, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return annotated


# 保持向后兼容
def detect_human_yolo(frame_path: str, model_type: str = "pose", conf_threshold: float = 0.5) -> Dict:
    """检测人体（向后兼容）"""
    detector = YOLODetector(model_types=[model_type], confidence=conf_threshold)
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

    detector = YOLODetector(model_types=[model_type], confidence=conf_threshold)

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
