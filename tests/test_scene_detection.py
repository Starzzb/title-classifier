"""场景检测测试：合成视频验证切换点检出 + 场景点缓存往返"""
import json
import sqlite3

import cv2
import numpy as np
import pytest

from title_classifier.core.scene_detector import detect_scenes, build_segments, get_segments


def _make_video(path, fps=10, seconds=12, cut_at=(4.0, 8.0), size=(160, 120)):
    """生成带硬切换的合成视频：纯色段落（红/绿/蓝），切换点处颜色突变"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, fps, size)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(int(fps * seconds)):
        t = i / fps
        idx = sum(1 for c in cut_at if t >= c)
        frame = np.full((size[1], size[0], 3), colors[idx % 3], dtype=np.uint8)
        w.write(frame)
    w.release()
    return fps


def test_detect_scenes_finds_cuts(tmp_path):
    video = tmp_path / "synthetic.mp4"
    fps = _make_video(video, cut_at=(4.0, 8.0))

    points = detect_scenes(str(video), threshold=0.3, sample_interval=0.5)

    # 切换点容差 ±1 个采样间隔
    assert points[0] == 0.0
    for expected in (4.0, 8.0):
        assert any(abs(p - expected) <= 0.6 for p in points), f"未检出切换点 {expected}: {points}"


def test_detect_scenes_high_res_downsample_consistent(tmp_path):
    """降采样路径(>1080p)与原始路径应检出相同切换点"""
    video = tmp_path / "large.mp4"
    fps = _make_video(video, cut_at=(4.0, 8.0), size=(1440, 1080))

    points = detect_scenes(str(video), threshold=0.3, sample_interval=0.5)
    for expected in (4.0, 8.0):
        assert any(abs(p - expected) <= 0.6 for p in points), f"未检出切换点 {expected}: {points}"


def test_build_segments_merges_to_max():
    # 20 个均匀切换点，max_scenes=5 → 合并到 5 段
    points = [float(i) for i in range(21)]
    segments = build_segments(points, duration=20.0, max_scenes=5)
    assert len(segments) <= 5
    assert segments[0][0] == 0.0
    assert segments[-1][1] == 20.0


def test_scene_cache_roundtrip(tmp_path):
    """db_store 场景缓存：未命中 → 写入 → 命中"""
    from title_classifier.core.db_store import MediaDB

    db_path = tmp_path / "media.db"
    store = MediaDB(str(db_path))
    store.init_schema()

    fp_id = store.get_fingerprint_id(12345, 60.0)
    assert store.get_scene_cache(fp_id, 0.3, 0.5) is None  # 未命中

    points = [0.0, 10.5, 20.0, 45.25]
    store.save_scene_cache(fp_id, 0.3, points, 0.5)
    got = store.get_scene_cache(fp_id, 0.3, 0.5)
    assert got == points

    # 不同阈值/间隔键控隔离
    assert store.get_scene_cache(fp_id, 0.4, 0.5) is None

    # 覆盖写
    store.save_scene_cache(fp_id, 0.3, [0.0, 30.0], 0.5)
    assert store.get_scene_cache(fp_id, 0.3, 0.5) == [0.0, 30.0]

    store.close()
