"""debug 保存不得重新触发 YOLO 推理"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def test_save_detection_debug_reuses_raw_pose(tmp_path):
    from title_classifier.core.vision import VisionProcessor

    p = VisionProcessor.__new__(VisionProcessor)

    infer_calls = {"n": 0}

    class FakeDetector:
        def estimate_pose(self, frame):
            infer_calls["n"] += 1
            return {"has_person": False, "poses": []}

    p.yolo_detector = FakeDetector()

    # 准备一帧真实图片
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((60, 60, 3), dtype=np.uint8))

    pose_result = {
        "has_person": True,
        "poses": [{
            "bbox": {"bbox_pixel": [5, 5, 50, 50], "confidence": 0.9},
            "keypoints": [],
            "pose_analysis": ["站立"],
        }],
    }
    analysis = {
        "timeline": [{
            "index": 0,
            "timestamp": 1.0,
            "frame_path": str(frame_path),
            "raw_pose": pose_result,
        }],
        "frames": [str(frame_path)],
    }

    debug_dir = tmp_path / "dbg"
    debug_dir.mkdir()
    (debug_dir / "detection").mkdir()

    p._save_detection_debug(analysis, debug_dir)

    assert infer_calls["n"] == 0  # 未重新推理
    assert (debug_dir / "detection" / "frame_0000_1.0s_annotated.jpg").exists()