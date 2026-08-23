"""测试 _analyze_video_comprehensive 的批量抽帧与内存复用"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


@pytest.fixture()
def synthetic_video(tmp_path):
    path = tmp_path / "synth.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    if not writer.isOpened():
        pytest.skip("VideoWriter 不可用")
    for i in range(30):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        x = i * 5 % 140
        cv2.rectangle(frame, (x, 40), (x + 20, 80), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def _make_processor(tmp_path, monkeypatch):
    """无 YOLO 检测器的裸 VisionProcessor（跳过推理分支）"""
    from title_classifier.core.vision import VisionProcessor

    monkeypatch.chdir(tmp_path)  # 让 logs/_vision_tmp 落在临时目录
    p = VisionProcessor.__new__(VisionProcessor)
    p.analysis_step = 1.0
    p.max_sample_frames = 10
    p.motion_detection = False
    p.motion_threshold = 8.0
    p.motion_min_interval = 5.0
    p.device = "cpu"
    p.backend = "pytorch"
    p.yolo_models = ["pose"]
    p.yolo_detector = None  # 关键：None 时循环只抽帧不推理
    p.debug_dir = None
    return p


def test_analyze_returns_aligned_decoded_frames(synthetic_video, tmp_path, monkeypatch):
    p = _make_processor(tmp_path, monkeypatch)
    analysis = p._analyze_video_comprehensive(synthetic_video, duration=3.0)

    assert len(analysis["frames"]) > 0
    assert len(analysis["decoded_frames"]) == len(analysis["frames"])
    # 每个成功抽取的位置都有解码数组
    for f, d in zip(analysis["frames"], analysis["decoded_frames"]):
        assert (f is None) == (d is None) or (f is not None and d is not None)
    assert all(isinstance(d, np.ndarray) for d in analysis["decoded_frames"] if d is not None)


def test_clip_diff_handles_none_entries(tmp_path):
    """compute_frame_diff_scores 对 None 帧给 0 分且长度对齐"""
    from title_classifier.detectors.clip import CLIPClassifier

    c = CLIPClassifier.__new__(CLIPClassifier)
    c._loaded = True
    c._device = "cpu"
    emb = np.array([[1.0, 0.0]])
    c._encode_image_array = lambda arr: emb if arr is not None else None

    scores = c.compute_frame_diff_scores([np.zeros((4, 4, 3)), None, np.zeros((4, 4, 3))])
    assert len(scores) == 3
    assert scores[1] == 0.0


def test_call_vlm_accepts_ndarray_frames(monkeypatch):
    """_call_vlm_comprehensive 接受 ndarray 帧（不再读盘）"""
    import title_classifier.core.vision as vmod

    p = _make_processor_for_vlm()
    captured = {}

    def fake_call(provider, images_b64, prompt, model=None, api_key=None):
        captured["images"] = images_b64
        return "描述：测试\n关键词：a, b"

    monkeypatch.setattr(vmod, "call_vision_api", fake_call)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    result = p._call_vlm_comprehensive([frame], "标题")

    assert "error" not in result
    assert len(captured["images"]) == 1
    assert isinstance(captured["images"][0], str) and len(captured["images"][0]) > 0


def _make_processor_for_vlm():
    from title_classifier.core.vision import VisionProcessor

    p = VisionProcessor.__new__(VisionProcessor)
    p.provider = "mock"
    p.model = "m"
    p.api_key = "k"
    p.max_image_size = 64
    return p