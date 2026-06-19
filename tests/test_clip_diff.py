import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_src = str(Path(__file__).parent.parent / "src")


def _get_clip_class():
    """Load CLIPClassifier by mocking out the circular-import chain."""
    # Pre-populate src on path
    if _src not in sys.path:
        sys.path.insert(0, _src)

    # Stub out core and its submodules so clip.py's module-level imports succeed
    # without pulling in vision.py (which re-imports detectors).
    stubs = {}
    for mod_name in [
        "title_classifier.core",
        "title_classifier.core.model_registry",
        "title_classifier.core.vision",
        "title_classifier.utils",
        "title_classifier.utils.config",
        "title_classifier.providers",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__package__ = mod_name.rsplit(".", 1)[0]
            sys.modules[mod_name] = stub
            stubs[mod_name] = stub

    # Provide the attributes that _load_clip_config needs
    mr = sys.modules["title_classifier.core.model_registry"]
    mr.MODEL_REGISTRY = {"clip": {"default": "test"}}
    mr.get_clip_pretrained = lambda name: "test"
    mr.get_clip_open_clip_name = lambda name: "test"

    cfg = sys.modules["title_classifier.utils.config"]
    cfg.load_merged_config = lambda: {}

    # Now load base and clip modules
    det_dir = Path(_src, "title_classifier", "detectors")

    # Ensure detectors package stub exists
    det_name = "title_classifier.detectors"
    if det_name not in sys.modules:
        det_pkg = types.ModuleType(det_name)
        det_pkg.__path__ = [str(det_dir)]
        det_pkg.__package__ = det_name
        sys.modules[det_name] = det_pkg

    # Load base
    base_name = "title_classifier.detectors.base"
    if base_name not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(
            base_name, det_dir / "base.py"
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules[base_name] = base_mod
        base_spec.loader.exec_module(base_mod)

    # Load clip
    clip_name = "title_classifier.detectors.clip"
    if clip_name not in sys.modules:
        clip_spec = importlib.util.spec_from_file_location(
            clip_name, det_dir / "clip.py"
        )
        clip_mod = importlib.util.module_from_spec(clip_spec)
        sys.modules[clip_name] = clip_mod
        clip_spec.loader.exec_module(clip_mod)

    return sys.modules[clip_name].CLIPClassifier


def test_compute_frame_diff_scores_returns_scores():
    """测试 compute_frame_diff_scores 返回每帧的差异度分数"""
    CLIPClassifier = _get_clip_class()

    classifier = CLIPClassifier.__new__(CLIPClassifier)
    classifier._loaded = True
    classifier._model = MagicMock()
    classifier._preprocess = MagicMock()
    classifier._device = "cpu"

    embeddings = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    ]
    call_count = 0

    def mock_encode(img_array):
        nonlocal call_count
        result = embeddings[call_count]
        call_count += 1
        return result

    classifier._encode_image_array = mock_encode

    frames = [np.zeros((100, 100, 3)) for _ in range(3)]
    result = classifier.compute_frame_diff_scores(frames)

    assert len(result) == 3
    assert result[1] > result[0]
    assert result[1] > result[2]


def test_compute_frame_diff_scores_empty_input():
    """测试空输入返回空列表"""
    CLIPClassifier = _get_clip_class()

    classifier = CLIPClassifier.__new__(CLIPClassifier)
    classifier._loaded = True

    result = classifier.compute_frame_diff_scores([])
    assert result == []


def test_compute_frame_diff_scores_single_frame():
    """测试单帧返回 [0.0]"""
    CLIPClassifier = _get_clip_class()

    classifier = CLIPClassifier.__new__(CLIPClassifier)
    classifier._loaded = True
    classifier._encode_image_array = lambda img: np.array([[1.0, 0.0]])

    result = classifier.compute_frame_diff_scores([np.zeros((100, 100, 3))])
    assert result == [0.0]
