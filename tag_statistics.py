"""
标签统计管理器
从云端 VLM 结果中提取新标签，动态扩充 CLIP 候选集
"""
import json
from pathlib import Path
from datetime import datetime

STATS_FILE = "models/tag_statistics.json"

# 维度关键词映射规则：用于自动判断 VLM 返回的标签属于哪个维度
CLOTHING_INDICATORS = {
    "装", "服", "裙", "袜", "鞋", "衣", "裤", "帽", "饰", "链", "带", "领",
    "袖", "袖", "蕾丝", "纱", "绸", "绒", "皮", "布",
    "outfit", "dress", "skirt", "stocking", "sock", "shoe", "shirt", "pant",
    "hat", "glove", "lace", "silk", "leather", "latex", "satin", "velvet",
    "collar", "ribbon", "bow", "corset", "bodysuit", "swimsuit", "bikini",
    "uniform", "costume", "lingerie", "apron", "cape", "cloak", "hood",
    "heels", "boots", "sandals", "slippers", "mary janes",
    "头饰", "发饰", "耳环", "项链", "手链", "项圈", "choker",
    "headband", "earring", "necklace", "bracelet", "collar",
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
    "selfie", "posing", "arching", "leaning", "hugging", "kissing",
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

# 场景指标（用于判断是否为场景标签，场景标签不纳入 CLIP 学习）
SCENE_INDICATORS = {
    "室", "房", "间", "床", "沙发", "地板", "墙", "窗", "门",
    "浴室", "厨房", "客厅", "卧室", "酒店", "影棚", "户外",
    "indoor", "outdoor", "bedroom", "bathroom", "living room",
    "studio", "hotel", "kitchen", "floor", "wall", "window",
    "sofa", "bed", "carpet", "curtain", "background",
    "场景", "环境", "背景", "灯光", "光线",
    "scene", "environment", "background", "lighting",
}


def _classify_tag_dimension(tag: str) -> str | None:
    """
    判断一个标签属于哪个维度
    返回: "clothing" / "action" / "hairstyle" / None（场景或其他，不学习）
    """
    tag_lower = tag.lower().strip()
    if not tag_lower:
        return None

    # 计算各维度匹配得分
    clothing_score = sum(1 for w in CLOTHING_INDICATORS if w in tag_lower)
    action_score = sum(1 for w in ACTION_INDICATORS if w in tag_lower)
    hairstyle_score = sum(1 for w in HAIRSTYLE_INDICATORS if w in tag_lower)
    scene_score = sum(1 for w in SCENE_INDICATORS if w in tag_lower)

    # 如果场景得分最高，跳过
    scores = {
        "clothing": clothing_score,
        "action": action_score,
        "hairstyle": hairstyle_score,
    }

    max_dim = max(scores, key=scores.get)
    max_score = scores[max_dim]

    # 场景标签优先排除
    if scene_score > max_score:
        return None

    # 需要至少一个匹配
    if max_score == 0:
        return None

    return max_dim


def _tag_to_clip_prompt(tag: str, dimension: str) -> str:
    """将中文/英文标签转换为 CLIP prompt 格式"""
    tag = tag.strip()
    if not tag:
        return ""

    # 如果已经是英文，直接构建 prompt
    if all(ord(c) < 128 for c in tag):
        if dimension == "clothing":
            return f"a photo of a person wearing {tag}"
        elif dimension == "action":
            return f"a photo of a person {tag}"
        elif dimension == "hairstyle":
            return f"a photo of a person with {tag}"
        return tag

    # 中文标签：直接返回原文（作为 CLIP prompt 的补充候选）
    # CLIP 的文本编码器对中文有一定理解能力，但不如英文
    # 这里保留中文，让 CLIP 尝试匹配
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
            # 确保结构完整
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
            "meta": {
                "total_updates": 0,
                "last_updated": "",
            }
        }

    def save(self):
        """保存统计数据"""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.data["meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def update_from_vlm(self, keywords: str):
        """
        从云端 VLM 返回的关键词中学习新标签
        
        Args:
            keywords: 逗号分隔的关键词字符串，如 "女仆装, 跪姿, 双马尾, 室内场景"
        """
        if not keywords:
            return

        tags = [t.strip() for t in keywords.split(",") if t.strip()]
        updated = False

        for tag in tags:
            # 去除可能的标签前缀（如 CLIP 标记）
            tag = tag.strip("[]() ")
            if not tag or len(tag) < 2:
                continue

            dim = _classify_tag_dimension(tag)
            if dim is None:
                # 场景或其他标签，不学习
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

    def get_learned_prompts(self, dimension: str, min_count: int = 2) -> dict[str, str]:
        """
        获取某维度的已学习标签（作为 CLIP 候选 prompt）
        
        Args:
            dimension: "clothing" / "action" / "hairstyle"
            min_count: 最小出现次数（过滤噪声）
        
        Returns:
            {clip_prompt: 中文标签} 字典
        """
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

    def get_all_prompts(self, dimension: str, base_categories: dict) -> dict[str, str]:
        """
        获取某维度的完整候选集（base + learned）
        
        Args:
            dimension: "clothing" / "action" / "hairstyle"
            base_categories: 基础分类 {英文prompt: 中文标签}
        
        Returns:
            合并后的 {prompt: label_cn} 字典
        """
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


if __name__ == "__main__":
    # 测试
    stats = TagStatistics()
    print("当前标签统计:")
    print(stats.get_stats_summary())

    # 模拟 VLM 返回的关键词
    test_keywords = "女仆装, 跪姿, 双马尾, 室内场景, 水印, 黑色丝袜, 角色扮演"
    print(f"\n学习新标签: {test_keywords}")
    stats.update_from_vlm(test_keywords)
    print("\n更新后统计:")
    print(stats.get_stats_summary())

    # 测试获取已学习标签
    learned = stats.get_learned_prompts("clothing")
    print(f"\n已学习的穿着标签: {learned}")
