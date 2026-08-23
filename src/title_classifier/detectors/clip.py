"""CLIP分类器"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

# 模型缓存目录
CLIP_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "clip"
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CLIP 模型配置
def _load_clip_config() -> tuple:
    """从 config 读取用户选择的 CLIP 模型，fallback 到默认值"""
    from ..core.model_registry import MODEL_REGISTRY, get_clip_pretrained, get_clip_open_clip_name

    default_name = MODEL_REGISTRY["clip"]["default"]
    default_pretrained = get_clip_pretrained(default_name)
    default_open_clip = get_clip_open_clip_name(default_name)
    try:
        from ..utils.config import load_merged_config
        config = load_merged_config()
        model_name = config.get("models", {}).get("clip", default_name)
        pretrained = get_clip_pretrained(model_name)
        open_clip_name = get_clip_open_clip_name(model_name)
        return open_clip_name, pretrained
    except Exception:
        return default_open_clip, default_pretrained


CLIP_MODEL_NAME, CLIP_PRETRAINED = _load_clip_config()

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
        t0 = time.perf_counter()

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

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"[CLIP] 分类完成: 耗时={elapsed:.3f}s | "
            f"穿着={result['clothing']['label_cn']}({result['clothing']['confidence']:.3f}), "
            f"动作={result['action']['label_cn']}({result['action']['confidence']:.3f}), "
            f"发型={result['hairstyle']['label_cn']}({result['hairstyle']['confidence']:.3f}) | "
            f"标签={result['tags']}"
        )

        return result

    def compute_frame_diff_scores(self, frames: list) -> list:
        """计算每帧与所有其他帧的差异度分数

        差异度 = 1 - 该帧与所有其他帧的平均余弦相似度
        分数越高 = 该帧越"独特"（与其他帧差异越大）

        Args:
            frames: numpy array 列表（BGR 格式）

        Returns:
            每帧的差异度分数列表，范围 [0, 1]
        """
        if not frames:
            return []
        if len(frames) == 1:
            return [0.0]

        t0 = time.perf_counter()

        # 编码所有帧
        t_encode = time.perf_counter()
        embeddings = []
        for frame in frames:
            emb = self._encode_image_array(frame) if frame is not None else None
            embeddings.append(emb)
        encode_time = time.perf_counter() - t_encode

        n = len(embeddings)
        scores = []

        # 计算差异度
        t_calc = time.perf_counter()
        for i in range(n):
            if embeddings[i] is None:
                scores.append(0.0)
                continue

            similarities = []
            for j in range(n):
                if i == j or embeddings[j] is None:
                    continue
                sim = float(np.dot(embeddings[i].flatten(), embeddings[j].flatten()))
                similarities.append(sim)

            avg_sim = np.mean(similarities) if similarities else 1.0
            diff_score = 1.0 - avg_sim
            scores.append(max(0.0, diff_score))
        calc_time = time.perf_counter() - t_calc

        total_time = time.perf_counter() - t0
        logger.debug(
            f"[CLIP] 差异度计算: {n}帧, "
            f"编码={encode_time:.2f}s, 计算={calc_time:.3f}s, 总计={total_time:.2f}s | "
            f"分数: min={min(scores):.3f}, max={max(scores):.3f}, avg={np.mean(scores):.3f}"
        )

        return scores
