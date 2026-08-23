"""测试 _build_comprehensive_context 的输出格式"""

import sys
from pathlib import Path

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _make_processor():
    """绕过 __init__ 的重依赖，构造裸实例"""
    from title_classifier.core.vision import VisionProcessor
    return VisionProcessor.__new__(VisionProcessor)


def test_main_pose_not_split_by_join():
    """main_pose 是字符串，不应被 join 逐字拆开"""
    p = _make_processor()
    summary = {
        "has_person": True,
        "duration": 60.0,
        "person_ratio": 0.5,
        "total_frames": 20,
        "main_pose": "站立",
        "pose_distribution": {"站立": 10},
        "pose_changes": [],
        "person_appearances": [],
        "avg_confidence": 0.9,
        "avg_keypoints": 12.0,
    }
    ctx = p._build_comprehensive_context(summary, [], 10)
    assert "站, 立" not in ctx  # 未被拆字（join 实际输出为逗号+空格）


def test_main_pose_appears_once():
    """prompt 中'- 主要姿态'只出现一次（结构化版本）"""
    p = _make_processor()
    summary = {
        "has_person": True,
        "duration": 60.0,
        "person_ratio": 0.5,
        "total_frames": 20,
        "main_pose": "站立",
        "pose_distribution": {"站立": 10},
        "pose_changes": [],
        "person_appearances": [],
        "avg_confidence": 0.9,
        "avg_keypoints": 12.0,
    }
    ctx = p._build_comprehensive_context(summary, [], 10)
    assert ctx.count("- 主要姿态") == 1