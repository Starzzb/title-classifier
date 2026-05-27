"""标签统计管理器"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STATS_FILE = "models/tag_statistics.json"

# 维度关键词映射规则
CLOTHING_INDICATORS = {
    "装", "服", "裙", "袜", "鞋", "衣", "裤", "帽", "饰", "链", "带", "领",
    "袖", "蕾丝", "纱", "绸", "绒", "皮", "布",
    "outfit", "dress", "skirt", "stocking", "sock", "shoe", "shirt", "pant",
    "hat", "glove", "lace", "silk", "leather", "latex", "satin", "velvet",
    "collar", "ribbon", "bow", "corset", "bodysuit", "swimsuit", "bikini",
    "uniform", "costume", "lingerie", "apron", "cape", "cloak", "hood",
    "heels", "boots", "sandals", "slippers", "mary janes",
    "头饰", "发饰", "耳环", "项链", "手链", "项圈", "choker",
    "headband", "earring", "necklace", "bracelet",
    "围裙", "斗篷", "披风", "兜帽", "手套", "口罩",
    "丝袜", "连裤袜", "长筒袜", "过膝袜", "白丝", "黑丝",
    "水手服", "女仆装", "护士装", "校服", "旗袍", "和服", "汉服",
}

ACTION_INDICATORS = {
    "坐", "站", "跪", "蹲", "躺", "卧", "趴", "弯", "走", "跑", "跳",
    "爬", "握", "持", "拿", "触摸", "抚摸", "自拍", "拍摄",
    "sitting", "standing", "kneeling", "squatting", "lying", "bending",
    "walking", "running", "jumping", "crawling", "holding", "touching",
    "posing", "dancing", "exercising", "stretching", "reaching",
    "selfie", "arching", "leaning", "hugging", "kissing",
    "仰卧", "俯卧", "侧卧", "跨坐", "盘腿", "翘腿", "张腿",
    "低头", "抬头", "转身", "回头", "俯身",
}

HAIRSTYLE_INDICATORS = {
    "发", "辫", "刘海", "马尾", "丸子", "卷发", "直发",
    "hair", "bangs", "fringe", "ponytail", "braid", "bun", "twintail",
    "pigtail", "bob", "pixie", "afro", "dreadlock",
    "长发", "短发", "双马尾", "单马尾", "散发", "盘发",
    "金发", "银发", "粉发", "紫发", "蓝发", "红发", "绿发", "橙发",
    "黑发", "棕发", "白发", "青发",
    "blonde", "brunette", "redhead", "pink hair", "blue hair",
    "purple hair", "silver hair", "green hair", "orange hair",
}

SCENE_INDICATORS = {
    "室", "房", "间", "床", "沙发", "地板", "墙", "窗", "门",
    "浴室", "厨房", "客厅", "卧室", "酒店", "影棚", "户外",
    "indoor", "outdoor", "bedroom", "bathroom", "living room",
    "studio", "hotel", "kitchen", "floor", "wall", "window",
    "sofa", "bed", "carpet", "curtain", "background",
    "场景", "环境", "背景", "灯光", "光线",
    "scene", "environment", "background", "lighting",
}


def _classify_tag_dimension(tag: str) -> Optional[str]:
    """判断一个标签属于哪个维度"""
    tag_lower = tag.lower().strip()
    if not tag_lower:
        return None

    clothing_score = sum(1 for w in CLOTHING_INDICATORS if w in tag_lower)
    action_score = sum(1 for w in ACTION_INDICATORS if w in tag_lower)
    hairstyle_score = sum(1 for w in HAIRSTYLE_INDICATORS if w in tag_lower)
    scene_score = sum(1 for w in SCENE_INDICATORS if w in tag_lower)

    scores = {
        "clothing": clothing_score,
        "action": action_score,
        "hairstyle": hairstyle_score,
    }

    max_dim = max(scores, key=scores.get)
    max_score = scores[max_dim]

    if scene_score > max_score:
        return None

    if max_score == 0:
        return None

    return max_dim


def _tag_to_clip_prompt(tag: str, dimension: str) -> str:
    """将中文/英文标签转换为CLIP prompt格式"""
    tag = tag.strip()
    if not tag:
        return ""

    if all(ord(c) < 128 for c in tag):
        if dimension == "clothing":
            return f"a photo of a person wearing {tag}"
        elif dimension == "action":
            return f"a photo of a person {tag}"
        elif dimension == "hairstyle":
            return f"a photo of a person with {tag}"
        return tag

    return tag


class TagStatistics:
    """标签统计管理器"""

    def __init__(self, stats_path: str = None):
        self.stats_path = Path(stats_path or STATS_FILE)
        self.data = self._load()

    def _load(self) -> dict:
        """加载统计数据"""
        if not self.stats_path.exists():
            return self._init_data()
        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for dim in ["clothing", "action", "hairstyle"]:
                if dim not in data:
                    data[dim] = {}
            if "meta" not in data:
                data["meta"] = {"total_updates": 0, "last_updated": ""}
            return data
        except (json.JSONDecodeError, IOError):
            return self._init_data()

    def _init_data(self) -> dict:
        return {
            "clothing": {},
            "action": {},
            "hairstyle": {},
            "meta": {"total_updates": 0, "last_updated": ""},
        }

    def save(self) -> None:
        """保存统计数据"""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.data["meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def update_from_vlm(self, keywords: str) -> None:
        """从VLM返回的关键词中学习新标签"""
        if not keywords:
            return

        tags = [t.strip() for t in keywords.split(",") if t.strip()]
        updated = False

        for tag in tags:
            tag = tag.strip("[]() ")
            if not tag or len(tag) < 2:
                continue

            dim = _classify_tag_dimension(tag)
            if dim is None:
                continue

            tag_lower = tag.lower()
            if tag_lower in self.data[dim]:
                self.data[dim][tag_lower]["count"] += 1
            else:
                self.data[dim][tag_lower] = {
                    "count": 1,
                    "source": "learned",
                    "label_cn": tag if any(ord(c) >= 128 for c in tag) else "",
                }
            updated = True

        if updated:
            self.data["meta"]["total_updates"] += 1
            self.save()

    def get_learned_prompts(self, dimension: str, min_count: int = 2) -> Dict[str, str]:
        """获取某维度的已学习标签"""
        if dimension not in self.data:
            return {}

        result = {}
        for tag, info in self.data[dimension].items():
            if info.get("source") == "learned" and info.get("count", 0) >= min_count:
                clip_prompt = _tag_to_clip_prompt(tag, dimension)
                if clip_prompt:
                    cn_label = info.get("label_cn", tag)
                    result[clip_prompt] = cn_label
        return result

    def get_all_prompts(self, dimension: str, base_categories: dict) -> Dict[str, str]:
        """获取某维度的完整候选集"""
        merged = dict(base_categories)
        learned = self.get_learned_prompts(dimension)
        merged.update(learned)
        return merged

    def get_stats_summary(self) -> str:
        """返回统计摘要"""
        lines = []
        for dim in ["clothing", "action", "hairstyle"]:
            total = len(self.data[dim])
            learned = sum(1 for v in self.data[dim].values() if v.get("source") == "learned")
            lines.append(f"  {dim}: {total} tags ({learned} learned)")
        lines.append(f"  total updates: {self.data['meta']['total_updates']}")
        return "\n".join(lines)
