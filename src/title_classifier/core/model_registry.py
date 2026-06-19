"""模型注册表 - 声明式模型元数据，GUI 和 detectors 共用"""

from pathlib import Path
from typing import Dict, List, Optional

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
MODELS_DIR = PROJECT_DIR / "models"
YOLO_DIR = MODELS_DIR / "yolo"
CLIP_DIR = MODELS_DIR / "clip"


# ── YOLO 模型族 ──────────────────────────────────────────────────────────────

YOLO_MODEL_OPTIONS = {
    "yolov8n.pt": {"size": 6.25, "desc": "Nano - 最快，适合 CPU 实时推理"},
    "yolov8s.pt": {"size": 22.0, "desc": "Small - 速度精度平衡"},
    "yolov8m.pt": {"size": 52.0, "desc": "Medium - 较高精度"},
    "yolov8l.pt": {"size": 87.0, "desc": "Large - 高精度，建议 GPU"},
    "yolov8x.pt": {"size": 131.0, "desc": "ExtraLarge - 最高精度"},
}

YOLO_POSE_OPTIONS = {
    "yolov8n-pose.pt": {"size": 6.52, "desc": "Nano - 最快，适合 CPU"},
    "yolov8s-pose.pt": {"size": 22.42, "desc": "Small - 速度精度平衡"},
    "yolo11m-pose.pt": {"size": 40.49, "desc": "Medium - 推荐，精度较高"},
}

YOLO_SEGMENT_OPTIONS = {
    "yolov8n-seg.pt": {"size": 6.74, "desc": "Nano - 最快"},
    "yolov8s-seg.pt": {"size": 23.0, "desc": "Small - 速度精度平衡"},
    "yolov8m-seg.pt": {"size": 54.0, "desc": "Medium - 较高精度"},
}

YOLO_DOWNLOAD_BASE = "https://github.com/ultralytics/assets/releases/download/v8.2.0/"

# ── CLIP 模型族 ──────────────────────────────────────────────────────────────

CLIP_MODEL_OPTIONS = {
    "CLIP-ViT-B-16-laion2B-s34B-b88K": {
        "size": 650,
        "desc": "Base 级别，Patch size 16，推荐，平衡性能和精度",
        "pretrained": "laion2b_s34b_b88k",
        "open_clip_name": "ViT-B-16",
    },
    "CLIP-ViT-B-32-laion2B-s34B-b79K": {
        "size": 350,
        "desc": "Base 级别，Patch size 32，速度更快，精度略低",
        "pretrained": "laion2b_s34b_b79k",
        "open_clip_name": "ViT-B-32",
    },
    "CLIP-ViT-L-14-laion2B-s32B-b82K": {
        "size": 1700,
        "desc": "Large 级别，最高精度，需要更多显存",
        "pretrained": "laion2b_s32b_b82k",
        "open_clip_name": "ViT-L-14",
    },
}

# ── 模型注册表 ───────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "yolo_detect": {
        "name": "YOLO 检测模型",
        "category": "yolo",
        "task": "detect",
        "description": "目标检测，识别画面中物体的位置和类别（COCO 80类）",
        "pipeline_role": "判断画面是否包含人体，定位人体区域",
        "default": "yolov8n.pt",
        "options": YOLO_MODEL_OPTIONS,
        "download_base": YOLO_DOWNLOAD_BASE,
    },
    "yolo_pose": {
        "name": "YOLO 姿态模型",
        "category": "yolo",
        "task": "pose",
        "description": "人体姿态估计，检测 17 个关键点（COCO 格式）",
        "pipeline_role": "分析人体姿态：站立、坐姿、弯腰、跪姿、蹲姿、躺卧、弓背、张腿、侧向",
        "default": "yolo11m-pose.pt",
        "options": YOLO_POSE_OPTIONS,
        "download_base": YOLO_DOWNLOAD_BASE,
    },
    "yolo_segment": {
        "name": "YOLO 分割模型",
        "category": "yolo",
        "task": "segment",
        "description": "实例分割，像素级分割物体轮廓",
        "pipeline_role": "精确分割人体区域，提取穿着色彩特征（平均颜色、颜色方差）",
        "default": "yolov8n-seg.pt",
        "options": YOLO_SEGMENT_OPTIONS,
        "download_base": YOLO_DOWNLOAD_BASE,
    },
    "clip": {
        "name": "CLIP 视觉模型",
        "category": "clip",
        "task": "classify",
        "description": "视觉-语言对比模型，通过零样本分类将图像与文本描述匹配",
        "pipeline_role": (
            "对图像进行多维度零样本分类：\n"
            "  穿着分类（clothing）：识别服饰类型\n"
            "  动作分类（action）：识别人体动作\n"
            "  发型分类（hairstyle）：识别发型特征"
        ),
        "default": "CLIP-ViT-B-16-laion2B-s34B-b88K",
        "options": {
            name: {"size": info["size"], "desc": info["desc"]}
            for name, info in CLIP_MODEL_OPTIONS.items()
        },
    },
}

# config 中对应的 key 映射
_REGISTRY_TO_CONFIG_KEY = {
    "yolo_detect": "yolo_detect",
    "yolo_pose": "yolo_pose",
    "yolo_segment": "yolo_segment",
    "clip": "clip",
}


def get_model_info(registry_key: str) -> Optional[Dict]:
    """获取模型注册信息"""
    return MODEL_REGISTRY.get(registry_key)


def get_model_default(registry_key: str) -> str:
    """获取模型默认文件名"""
    info = MODEL_REGISTRY.get(registry_key)
    return info["default"] if info else ""


def get_model_from_config(registry_key: str, config: dict) -> str:
    """从 config 中读取用户选择的模型文件名，fallback 到默认值"""
    default = get_model_default(registry_key)
    config_key = _REGISTRY_TO_CONFIG_KEY.get(registry_key, registry_key)
    return config.get("models", {}).get(config_key, default)


def get_clip_pretrained(model_name: str) -> str:
    """获取 CLIP 模型的 pretrained tag"""
    info = CLIP_MODEL_OPTIONS.get(model_name)
    return info["pretrained"] if info else "laion2b_s34b_b88k"


def get_clip_open_clip_name(model_name: str) -> str:
    """获取 CLIP 模型的 open_clip 模型名（如 ViT-B-16）"""
    info = CLIP_MODEL_OPTIONS.get(model_name)
    return info.get("open_clip_name", "ViT-B-16") if info else "ViT-B-16"


def list_models(category: str = None) -> List[str]:
    """列出所有模型注册 key，可按 category 过滤"""
    if category:
        return [k for k, v in MODEL_REGISTRY.items() if v["category"] == category]
    return list(MODEL_REGISTRY.keys())
