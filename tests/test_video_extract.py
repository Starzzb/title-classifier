"""测试 extract_frames_cv2 与 image_array_to_base64"""

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
    """生成 3 秒、10fps、30 帧的合成视频（移动方块，帧间内容不同）"""
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


def test_extract_returns_aligned_paths(synthetic_video, tmp_path):
    from title_classifier.utils.video import extract_frames_cv2

    out_dir = tmp_path / "frames"
    timestamps = [0.0, 1.0, 2.0]
    results = extract_frames_cv2(synthetic_video, str(out_dir), timestamps, max_size=120)

    assert len(results) == len(timestamps)
    assert all(r is not None for r in results)
    for r in results:
        assert Path(r).exists()


def test_extract_resizes_to_max_size(synthetic_video, tmp_path):
    from title_classifier.utils.video import extract_frames_cv2

    out_dir = tmp_path / "frames"
    results = extract_frames_cv2(synthetic_video, str(out_dir), [1.0], max_size=64)
    img = cv2.imread(results[0])
    assert max(img.shape[:2]) <= 64


def test_extract_clamps_out_of_range_timestamp(synthetic_video, tmp_path):
    from title_classifier.utils.video import extract_frames_cv2

    out_dir = tmp_path / "frames"
    # 视频只有 3 秒，2.9s 应钳制到最后一帧而不是失败
    results = extract_frames_cv2(synthetic_video, str(out_dir), [2.9], max_size=120)
    assert results[0] is not None


def test_extract_handles_unopenable_file(tmp_path):
    from title_classifier.utils.video import extract_frames_cv2

    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    results = extract_frames_cv2(str(bad), str(tmp_path / "out"), [0.0, 1.0])
    assert results == [None, None]


def test_image_array_to_base64_roundtrip(tmp_path):
    from title_classifier.utils.image import image_array_to_base64

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :100] = (0, 0, 255)
    b64 = image_array_to_base64(img, max_size=50)
    assert isinstance(b64, str) and len(b64) > 0

    import base64
    data = base64.b64decode(b64)
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 50


def test_image_to_base64_still_works(tmp_path):
    """重构后旧接口保持可用"""
    from title_classifier.utils.image import image_to_base64

    img_path = tmp_path / "t.jpg"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
    b64 = image_to_base64(str(img_path), max_size=50)
    assert len(b64) > 0