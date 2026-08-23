"""_should_skip_inference 公共跳帧判定"""

import sys
from pathlib import Path

import numpy as np

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _make(motion_threshold=8.0, min_interval=5.0):
    from title_classifier.core.vision import VisionProcessor

    p = VisionProcessor.__new__(VisionProcessor)
    p.motion_detection = True
    p.motion_threshold = motion_threshold
    p.motion_min_interval = min_interval
    return p


_STATIC_A = np.full((60, 60), 128, dtype=np.uint8)
_STATIC_B = np.full((60, 60), 130, dtype=np.uint8)   # 微小噪声，低于阈值
_MOVED = np.full((60, 60), 255, dtype=np.uint8)      # 剧烈变化


def test_first_frame_never_skips():
    p = _make()
    assert p._should_skip_inference(None, _STATIC_A, ts=0.0, last_forced_ts=-float("inf")) is False


def test_static_within_interval_skips():
    p = _make(min_interval=5.0)
    # 上次强制推理在 0s，当前 3s（< 5s），画面静止 → 跳过
    assert p._should_skip_inference(_STATIC_A, _STATIC_B, ts=3.0, last_forced_ts=0.0) is True


def test_static_beyond_interval_forced():
    p = _make(min_interval=5.0)
    # 上次强制推理在 0s，当前 6s（>= 5s），即使静止也强制推理
    assert p._should_skip_inference(_STATIC_A, _STATIC_B, ts=6.0, last_forced_ts=0.0) is False


def test_motion_never_skips():
    p = _make()
    assert p._should_skip_inference(_STATIC_A, _MOVED, ts=1.0, last_forced_ts=0.0) is False


def test_disabled_motion_detection_never_skips():
    p = _make()
    p.motion_detection = False
    assert p._should_skip_inference(_STATIC_A, _STATIC_B, ts=1.0, last_forced_ts=0.0) is False