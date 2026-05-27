"""UHD人体检测器"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import onnxruntime as ort

from .base import BaseDetector

logger = logging.getLogger(__name__)

# 模型配置
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "human_detection"
DEFAULT_MODEL = "ultratinyod_res_anc8_w128_64x64_loese_distill.onnx"


class UHDDetector(BaseDetector):
    """UHD超轻量人体检测器"""

    def __init__(self, model_path: str = None, confidence: float = 0.5):
        super().__init__(confidence)
        self.model_path = model_path or str(MODEL_DIR / DEFAULT_MODEL)
        self._session: Optional[ort.InferenceSession] = None

    def load_model(self) -> bool:
        """加载ONNX模型"""
        try:
            logger.info(f"加载UHD模型: {self.model_path}")
            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._loaded = True
            logger.info("UHD模型加载完成")
            return True
        except Exception as e:
            logger.error(f"UHD模型加载失败: {e}")
            return False

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预处理帧图像"""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
        arr = resized.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return chw[np.newaxis, ...]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的sigmoid"""
        x_clip = np.clip(x, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的softplus"""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _decode_decoded_output(self, detections: np.ndarray) -> List[Dict]:
        """解码已解码的检测结果 [1, N, 6]"""
        results = []
        for det in detections[0]:
            score, cls_id, cx, cy, w, h = det[:6]
            if score >= self.confidence:
                results.append({
                    "score": float(score),
                    "bbox": [float(cx), float(cy), float(w), float(h)],
                })
        return results

    def _non_max_suppression(self, detections: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            remaining = []
            for det in detections:
                cx1, cy1, w1, h1 = best["bbox"]
                cx2, cy2, w2, h2 = det["bbox"]

                x1_min, y1_min = cx1 - w1 / 2, cy1 - h1 / 2
                x1_max, y1_max = cx1 + w1 / 2, cy1 + h1 / 2
                x2_min, y2_min = cx2 - w2 / 2, cy2 - h2 / 2
                x2_max, y2_max = cx2 + w2 / 2, cy2 + h2 / 2

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - inter_area

                iou = inter_area / union_area if union_area > 0 else 0

                if iou < iou_thresh:
                    remaining.append(det)

            detections = remaining

        return keep

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测帧中的人体"""
        if not self._loaded:
            if not self.load_model():
                return {"has_person": False, "persons": [], "max_confidence": 0.0}

        try:
            input_tensor = self._preprocess(frame)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            raw_output = outputs[0]

            if raw_output.ndim == 3 and raw_output.shape[2] == 6:
                detections = self._decode_decoded_output(raw_output)
            else:
                logger.warning(f"未预期的输出形状 {raw_output.shape}")
                detections = []

            detections = self._non_max_suppression(detections)

            persons = []
            for det in detections:
                cx, cy, w, h = det["bbox"]
                h_img, w_img = frame.shape[:2]
                persons.append({
                    "bbox": [cx / w_img, cy / h_img, w / w_img, h / h_img],
                    "bbox_pixel": [
                        (cx - w / 2) * w_img,
                        (cy - h / 2) * h_img,
                        (cx + w / 2) * w_img,
                        (cy + h / 2) * h_img,
                    ],
                    "confidence": det["score"],
                })

            max_conf = max([p["confidence"] for p in persons]) if persons else 0.0

            return {
                "has_person": len(persons) > 0,
                "persons": persons,
                "max_confidence": max_conf,
            }

        except Exception as e:
            logger.error(f"UHD检测失败: {e}")
            return {"has_person": False, "persons": [], "max_confidence": 0.0}


# 保持向后兼容的函数接口
def load_model(model_path: str = None) -> ort.InferenceSession:
    """加载模型（向后兼容）"""
    if model_path is None:
        model_path = str(MODEL_DIR / DEFAULT_MODEL)
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def detect_human(frame_path: str, session: ort.InferenceSession, conf_threshold: float = 0.5) -> Dict:
    """检测人体（向后兼容）"""
    detector = UHDDetector(confidence=conf_threshold)
    detector._session = session
    detector._loaded = True
    return detector.detect_from_file(frame_path)


def find_human_frame(
    video_path: str,
    session: ort.InferenceSession,
    step_seconds: float = 5.0,
    max_retries: int = 3,
    conf_threshold: float = 0.5,
    tmp_dir: str = "logs/_frame_selector_tmp",
) -> Dict:
    """从视频中找到有人体的帧（向后兼容）"""
    import hashlib

    detector = UHDDetector(confidence=conf_threshold)
    detector._session = session
    detector._loaded = True

    # 获取视频时长
    duration = get_video_duration(video_path)
    if duration <= 0:
        return {"found": False, "frame_path": None, "timestamp": None, "max_confidence": 0.0, "detections": [], "attempts": 0}

    # 自适应步长
    adaptive_step = get_adaptive_step(duration, max_retries)

    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]

    for attempt in range(max_retries):
        timestamp = attempt * adaptive_step
        frame_path = str(tmp_path / f"frame_{video_hash}_{attempt}.jpg")

        if not extract_frame_at_timestamp(video_path, frame_path, timestamp, duration):
            continue

        result = detector.detect_from_file(frame_path)

        if result["has_person"]:
            return {
                "found": True,
                "frame_path": frame_path,
                "timestamp": timestamp,
                "max_confidence": result["max_confidence"],
                "detections": result["persons"],
                "attempts": attempt + 1,
            }

        try:
            Path(frame_path).unlink()
        except:
            pass

    return {"found": False, "frame_path": None, "timestamp": None, "max_confidence": 0.0, "detections": [], "attempts": max_retries}


def get_video_duration(video_path: str) -> float:
    """获取视频时长"""
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
    """安全的时间戳"""
    if duration <= 0:
        return timestamp_seconds
    max_safe = max(0, duration - margin)
    if max_safe <= 0:
        return 0.0
    return min(timestamp_seconds, max_safe)


def get_adaptive_step(duration: float, max_retries: int = 3, min_step: float = 2.0, max_step: float = 10.0) -> float:
    """根据视频时长自适应调整步长"""
    if duration <= 0:
        return min_step
    target_coverage = duration / 3
    step = target_coverage / max_retries
    return max(min_step, min(max_step, step))


def extract_frame_at_timestamp(video_path: str, output_path: str, timestamp_seconds: float, duration: float = 0.0) -> bool:
    """在指定时间点提取帧"""
    if duration <= 0:
        duration = get_video_duration(video_path)

    safe_ts = safe_timestamp(timestamp_seconds, duration)

    hours = int(safe_ts // 3600)
    minutes = int((safe_ts % 3600) // 60)
    seconds = safe_ts % 60
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", timestamp, "-i", video_path,
             "-frames:v", "1", "-q:v", "2", output_path],
            capture_output=True, timeout=15, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        logger.error(f"帧提取失败: {e}")
        return False


def crop_human_region(frame_path: str, bbox: list, padding: float = 0.15) -> Optional[np.ndarray]:
    """裁剪人体区域"""
    try:
        data = np.fromfile(frame_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        cx, cy, bw, bh = bbox

        x1 = int((cx - bw / 2 - padding * bw) * w)
        y1 = int((cy - bh / 2 - padding * bh) * h)
        x2 = int((cx + bw / 2 + padding * bw) * w)
        y2 = int((cy + bh / 2 + padding * bh) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        return crop
    except Exception as e:
        logger.error(f"人体裁剪失败: {e}")
        return None


def detect_and_crop_human(frame_path: str, session: ort.InferenceSession, conf_threshold: float = 0.5, padding: float = 0.15) -> Dict:
    """检测并裁剪人体区域"""
    detector = UHDDetector(confidence=conf_threshold)
    detector._session = session
    detector._loaded = True

    result = detector.detect_from_file(frame_path)

    human_crop = None
    bbox = None

    if result["has_person"] and result["persons"]:
        best_det = max(result["persons"], key=lambda x: x["confidence"])
        bbox = best_det["bbox"]
        human_crop = crop_human_region(frame_path, bbox, padding)

    return {
        "has_human": result["has_person"],
        "max_confidence": result["max_confidence"],
        "human_crop": human_crop,
        "bbox": bbox,
    }
