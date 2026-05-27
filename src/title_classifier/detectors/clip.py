"""CLIP分类器"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

# 模型缓存目录
CLIP_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "clip"
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CLIP 模型配置
CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"

# 基础分类维度
CLOTHING_BASE = {
    "cosplay costume": "角色扮演服饰",
    "black stockings": "黑色丝袜",
    "maid outfit": "女仆装",
    "pantyhose": "连裤袜",
    "sailor uniform school uniform": "水手服",
    "school uniform": "校服/JK制服",
    "plaid skirt": "格子裙",
    "pleated skirt": "百褶裙",
    "lace lingerie": "蕾丝内衣",
    "white stockings": "白色丝袜",
    "knee-high socks": "过膝袜",
    "white shirt": "白色衬衫",
    "bunny girl outfit": "兔女郎服饰",
    "high heels": "高跟鞋",
    "nurse outfit": "护士装",
    "latex clothing": "乳胶服饰",
    "cow print outfit": "奶牛装",
    "bodysuit": "连体衣",
    "cat ear headband": "猫耳发箍",
    "chinese dress qipao": "旗袍",
    "nun outfit": "修女服饰",
    "leather harness": "皮质束带",
    "school swimsuit": "泳衣/死库水",
    "Mary Jane shoes": "玛丽珍鞋",
    "red dress": "红色服饰",
    "black dress": "黑色服饰",
    "white dress": "白色服饰",
}

ACTION_BASE = {
    "sitting": "坐姿",
    "kneeling": "跪姿",
    "squatting": "蹲姿",
    "lying down": "躺卧",
    "standing": "站立",
    "bending over": "弯腰",
    "taking a selfie": "自拍",
    "walking": "步行",
    "crawling": "爬行",
    "posing for photo": "摆拍",
    "arching back": "弓背",
    "spreading legs": "张腿",
}

HAIRSTYLE_BASE = {
    "long hair": "长发",
    "short hair": "短发",
    "twintails pigtails": "双马尾",
    "bangs fringe": "齐刘海",
    "blonde hair gold hair": "金发",
    "pink hair": "粉发",
    "purple hair": "紫发",
    "blue hair": "蓝发",
    "silver hair white hair": "银发/白发",
    "red hair": "红发",
    "orange hair": "橙发",
    "green hair": "绿发",
    "black hair": "黑发",
    "brown hair": "棕发",
    "bob cut": "波波头",
    "ponytail": "马尾辫",
    "bun hair updo": "丸子头/盘发",
}

# CLIP prompt 模板
CLOTHING_TEMPLATE = "a photo of a person wearing {}"
ACTION_TEMPLATE = "a photo of a person {}"
HAIRSTYLE_TEMPLATE = "a photo of a person with {}"


class CLIPClassifier:
    """CLIP零样本多维分类器"""

    def __init__(self, device: str = None, tag_stats=None):
        self.tag_stats = tag_stats
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._text_embeds = {}
        self._prompt_labels = {}
        self._loaded = False

        self._init_device(device)

    def _init_device(self, device: str = None):
        """初始化设备"""
        try:
            import torch
            if device:
                self._device = device
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

    def _find_local_model(self) -> Optional[str]:
        """查找本地缓存的模型"""
        if CLIP_CACHE_DIR.exists():
            for cache_dir in CLIP_CACHE_DIR.iterdir():
                if not cache_dir.is_dir():
                    continue
                snapshots = cache_dir / "snapshots"
                if snapshots.exists():
                    for snap_dir in snapshots.iterdir():
                        if snap_dir.is_dir():
                            for ext in (".safetensors", ".bin"):
                                for f in snap_dir.iterdir():
                                    if f.suffix == ext and "model" in f.name.lower():
                                        return str(f)
                            for f in snap_dir.iterdir():
                                if f.suffix in (".bin", ".safetensors"):
                                    return str(f)
                for ext in (".safetensors", ".bin"):
                    for f in cache_dir.iterdir():
                        if f.suffix == ext and "model" in f.name.lower():
                            return str(f)
        return None

    def load_model(self) -> bool:
        """加载CLIP模型"""
        try:
            import open_clip
            import torch

            logger.info(f"加载CLIP模型: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")

            local_model_path = self._find_local_model()

            if local_model_path:
                logger.info(f"使用本地模型: {local_model_path}")
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    CLIP_MODEL_NAME, pretrained=local_model_path, device=self._device
                )
                self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
                self._model.eval()
                self._precompute_all_embeddings()
                self._loaded = True
                return True

            # 本地没有，尝试在线下载
            import os
            os.environ["HF_HOME"] = str(CLIP_CACHE_DIR)
            logger.info("本地未找到模型，尝试在线下载...")

            attempts = [
                ("HF镜像", {"HF_ENDPOINT": "https://hf-mirror.com"}),
                ("原始源", {}),
            ]

            for name, env_override in attempts:
                try:
                    old_env = {}
                    for k, v in env_override.items():
                        old_env[k] = os.environ.get(k)
                        os.environ[k] = v

                    logger.info(f"尝试 {name}...")
                    self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self._device
                    )
                    self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
                    self._model.eval()

                    for k, v in old_env.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v

                    self._precompute_all_embeddings()
                    self._loaded = True
                    return True

                except Exception as e:
                    logger.warning(f"{name} 失败: {e}")
                    for k in env_override:
                        if k in os.environ and env_override[k] == os.environ.get(k):
                            os.environ.pop(k, None)
                    continue

            logger.error("所有加载方式均失败，请运行: python scripts/download_clip.py")
            return False

        except ImportError:
            logger.error("未安装 open-clip-torch，请运行: pip install open-clip-torch")
            return False
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            return False

    def _get_dimension_categories(self, dimension: str) -> Dict[str, str]:
        """获取某维度的完整候选集"""
        if dimension == "clothing":
            base = CLOTHING_BASE
        elif dimension == "action":
            base = ACTION_BASE
        elif dimension == "hairstyle":
            base = HAIRSTYLE_BASE
        else:
            return {}

        if self.tag_stats:
            return self.tag_stats.get_all_prompts(dimension, base)
        return base

    def _build_prompts(self, dimension: str) -> tuple:
        """构建某维度的prompt列表"""
        categories = self._get_dimension_categories(dimension)

        if dimension == "clothing":
            template = CLOTHING_TEMPLATE
        elif dimension == "action":
            template = ACTION_TEMPLATE
        elif dimension == "hairstyle":
            template = HAIRSTYLE_TEMPLATE
        else:
            template = "{}"

        prompts = []
        labels_cn = []
        for prompt_text, label_cn in categories.items():
            if " " in prompt_text and not prompt_text.startswith("a photo"):
                prompts.append(prompt_text)
            else:
                prompts.append(template.format(prompt_text))
            labels_cn.append(label_cn)

        return prompts, labels_cn

    def _encode_texts(self, texts: list) -> np.ndarray:
        """批量编码文本为embedding"""
        import torch

        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def _precompute_all_embeddings(self):
        """预计算所有维度的文本embedding"""
        for dim in ["clothing", "action", "hairstyle"]:
            prompts, labels_cn = self._build_prompts(dim)
            if prompts:
                self._text_embeds[dim] = self._encode_texts(prompts)
                self._prompt_labels[dim] = labels_cn
                logger.info(f"  {dim}: {len(prompts)} 个候选 prompt")

    def reload_embeddings(self):
        """重新加载embedding"""
        self._text_embeds.clear()
        self._prompt_labels.clear()
        self._precompute_all_embeddings()

    def _encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """编码单张图片为embedding"""
        try:
            import torch
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"图片编码失败: {e}")
            return None

    def _encode_image_array(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """编码numpy数组格式的图像为embedding"""
        try:
            import torch
            import cv2
            from PIL import Image

            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            img_tensor = self._preprocess(img_pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                image_features = self._model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"图像数组编码失败: {e}")
            return None

    def classify_single(self, image_path: str, dimension: str, top_k: int = 3) -> list:
        """对单张图片进行某维度分类"""
        if dimension not in self._text_embeds:
            return []

        image_embed = self._encode_image(image_path)
        if image_embed is None:
            return []

        text_embeds = self._text_embeds[dimension]
        similarities = (image_embed @ text_embeds.T)[0]

        exp_sim = np.exp(similarities - np.max(similarities))
        probs = exp_sim / exp_sim.sum()

        top_indices = probs.argsort()[::-1][:top_k]
        labels_cn = self._prompt_labels[dimension]

        results = []
        for idx in top_indices:
            prompts, _ = self._build_prompts(dimension)
            label_en = prompts[idx] if idx < len(prompts) else ""
            for prefix in [CLOTHING_TEMPLATE.split("{}")[0], ACTION_TEMPLATE.split("{}")[0], HAIRSTYLE_TEMPLATE.split("{}")[0]]:
                if label_en.startswith(prefix):
                    label_en = label_en[len(prefix):]
                    break

            results.append({
                "label": label_en.strip(),
                "label_cn": labels_cn[idx] if idx < len(labels_cn) else "",
                "confidence": float(probs[idx]),
            })

        return results

    def classify(self, image_path: str, threshold: float = 0.15, multi_label: bool = True) -> Dict[str, Any]:
        """多维分类"""
        result = {
            "clothing": {"label": "", "label_cn": "", "confidence": 0.0},
            "action": {"label": "", "label_cn": "", "confidence": 0.0},
            "hairstyle": {"label": "", "label_cn": "", "confidence": 0.0},
            "tags": "",
            "tags_json": {},
            "avg_confidence": 0.0,
            "all_results": {},
        }

        confidences = []
        tags_cn = []

        for dim in ["clothing", "action", "hairstyle"]:
            classifications = self.classify_single(image_path, dim, top_k=5)
            result["all_results"][dim] = classifications

            if classifications:
                best = classifications[0]
                result[dim] = best
                result["tags_json"][dim] = best["label"]

                if multi_label:
                    dim_labels = []
                    for cls in classifications:
                        if cls["confidence"] >= threshold:
                            if cls["label_cn"]:
                                dim_labels.append(cls["label_cn"])
                            confidences.append(cls["confidence"])
                    if dim_labels:
                        tags_cn.extend(dim_labels)
                else:
                    if best["confidence"] >= threshold:
                        confidences.append(best["confidence"])
                        if best["label_cn"]:
                            tags_cn.append(best["label_cn"])
            else:
                result["tags_json"][dim] = ""

        seen = set()
        unique_tags = []
        for tag in tags_cn:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        result["tags"] = "_".join(unique_tags)
        result["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0

        return result

    def detect_change_by_embedding(self, human_crops: list, threshold: float = 0.75) -> Dict[str, Any]:
        """基于人体区域embedding相似度检测穿着变化"""
        embeddings = []

        for crop in human_crops:
            if crop is not None:
                embed = self._encode_image_array(crop)
                embeddings.append(embed)
            else:
                embeddings.append(None)

        changed_frames = []
        similarities = []

        for i in range(1, len(embeddings)):
            if embeddings[i - 1] is None or embeddings[i] is None:
                similarities.append(1.0)
                continue

            sim = float(np.dot(embeddings[i - 1].flatten(), embeddings[i].flatten()))
            similarities.append(sim)

            if sim < threshold:
                changed_frames.append(i)

        return {
            "has_significant_change": len(changed_frames) > 0,
            "changed_frames": changed_frames,
            "similarities": similarities,
        }

    def compare_frames(
        self,
        image_paths: list,
        threshold: float = 0.15,
        multi_label: bool = True,
        human_crops: list = None,
        embedding_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """比较多个帧的变化"""
        if not image_paths:
            return {
                "all_tags": [],
                "has_significant_change": False,
                "changed_frames": [],
                "best_frame_idx": 0,
                "similarities": [],
            }

        all_tags = []
        all_confidences = []

        for path in image_paths:
            tag_result = self.classify(path, threshold=threshold, multi_label=multi_label)
            all_tags.append(tag_result)
            all_confidences.append(tag_result["avg_confidence"])

        if human_crops and len(human_crops) == len(image_paths):
            embed_result = self.detect_change_by_embedding(human_crops, embedding_threshold)

            best_frame_idx = 0
            if all_confidences:
                best_frame_idx = int(np.argmax(all_confidences))

            return {
                "all_tags": all_tags,
                "has_significant_change": embed_result["has_significant_change"],
                "changed_frames": embed_result["changed_frames"],
                "best_frame_idx": best_frame_idx,
                "similarities": embed_result["similarities"],
            }

        changed_frames = []
        for i in range(1, len(all_tags)):
            prev_tags = set(all_tags[i - 1]["tags"].split("_")) if all_tags[i - 1]["tags"] else set()
            curr_tags = set(all_tags[i]["tags"].split("_")) if all_tags[i]["tags"] else set()

            prev_tags.discard("")
            curr_tags.discard("")

            if not prev_tags and not curr_tags:
                continue

            intersection = len(prev_tags & curr_tags)
            union = len(prev_tags | curr_tags)

            if union == 0:
                similarity = 1.0
            else:
                similarity = intersection / union

            change_threshold = 0.5
            if similarity < change_threshold:
                changed_frames.append(i)

        has_change = len(changed_frames) > 0
        best_frame_idx = int(np.argmax(all_confidences)) if all_confidences else 0

        return {
            "all_tags": all_tags,
            "has_significant_change": has_change,
            "changed_frames": changed_frames,
            "best_frame_idx": best_frame_idx,
            "similarities": [],
        }


# 便捷函数
_global_classifier: Optional[CLIPClassifier] = None


def get_classifier(tag_stats=None) -> Optional[CLIPClassifier]:
    """获取全局CLIP分类器实例"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = CLIPClassifier(tag_stats=tag_stats)
        if not _global_classifier.load_model():
            _global_classifier = None
            return None
    return _global_classifier


def classify_image(image_path: str, tag_stats=None, threshold: float = 0.15) -> Dict[str, Any]:
    """便捷函数：对单张图片进行多维分类"""
    classifier = get_classifier(tag_stats)
    if classifier is None:
        return {"error": "CLIP model not available"}
    return classifier.classify(image_path, threshold=threshold)
