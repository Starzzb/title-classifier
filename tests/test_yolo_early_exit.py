"""analyze_comprehensive 早退逻辑：detect+pose 均阴性时跳过 segment 推理"""

import numpy as np

import title_classifier.core  # noqa: F401  先加载core打破detectors↔core循环依赖
from title_classifier.detectors.yolo import YOLODetector

_NO_PERSON_DETECT = {"has_person": False, "persons": [], "max_confidence": 0.0}
_NO_PERSON_POSE = {"has_person": False, "poses": [], "max_confidence": 0.0}


def _make_detector():
    det = YOLODetector.__new__(YOLODetector)
    det._loaded = True
    det._models = {"detect": object(), "pose": object(), "segment": object()}
    det._backend_type = {"detect": "test", "pose": "test", "segment": "test"}
    det.confidence = 0.5
    det.iou_threshold = 0.45
    det.device = "cpu"
    return det


def test_skips_segment_when_detect_and_pose_negative():
    det = _make_detector()
    calls = []

    def detect(frame):
        calls.append("detect")
        return dict(_NO_PERSON_DETECT)

    def pose(frame):
        calls.append("pose")
        return dict(_NO_PERSON_POSE)

    def segment(frame):
        calls.append("segment")
        return {"has_person": True, "segments": [{"confidence": 0.9}], "max_confidence": 0.9}

    det.detect = detect
    det.estimate_pose = pose
    det.segment_instances = segment

    result = det.analyze_comprehensive(np.zeros((32, 32, 3)))

    assert calls == ["detect", "pose"]  # segment 被跳过
    assert result["has_person"] is False
    assert result["segment"] is None


def test_segment_runs_when_pose_positive():
    det = _make_detector()
    calls = []

    def detect_pos(frame):
        calls.append("detect")
        return {"has_person": True, "persons": [{"confidence": 0.9}], "max_confidence": 0.9}

    def pose_pos(frame):
        calls.append("pose")
        return {"has_person": True, "poses": [{"avg_confidence": 0.9}], "max_confidence": 0.9}

    def segment(frame):
        calls.append("segment")
        return {"has_person": True, "segments": [{"confidence": 0.9}], "max_confidence": 0.9}

    det.detect = detect_pos
    det.estimate_pose = pose_pos
    det.segment_instances = segment

    result = det.analyze_comprehensive(np.zeros((32, 32, 3)))

    assert "segment" in calls  # 有人体时不早退（segment 提供穿着分析）
    assert result["has_person"] is True