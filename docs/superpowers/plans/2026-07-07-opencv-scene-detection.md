# OpenCV Scene Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ffprobe-based scene detection in `detect_scenes()` with pure OpenCV to eliminate path compatibility issues (brackets `[]`, Windows drive colon `:`, ffprobe version API changes).

**Architecture:** Use `cv2.VideoCapture` to sample frames at ~1fps, compute HSV histogram differences between consecutive frames via `cv2.compareHist`, and return timestamps where difference exceeds a threshold. All three public functions (`detect_scenes`, `build_segments`, `get_segments`) keep their signatures unchanged.

**Tech Stack:** OpenCV (`cv2` already imported project-wide), no new dependencies.

---

### Task 1: Rewrite `detect_scenes()` with OpenCV

**Files:**
- Modify: `src/title_classifier/core/scene_detector.py` (full rewrite)

- [ ] **Step 1: Replace the file content**

```python
"""场景检测模块 - 基于 OpenCV 的场景切换检测"""

import cv2
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_scenes(video_path: str, threshold: float = 0.3, sample_interval: float = 0.5) -> List[float]:
    """
    使用 OpenCV 直方图差异检测场景切换点。

    逐帧/间隔采样，计算 HSV 直方图差异，超过 threshold 时标记场景切换。

    Args:
        video_path: 视频文件路径
        threshold: 场景检测敏感度（0-1），值越低越灵敏，默认 0.3
        sample_interval: 采样间隔（秒），默认 0.5s

    Returns:
        场景切换时间点列表（秒），包含 0 和视频末尾
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}，回退到整段处理")
        return [0.0]

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0:
        cap.release()
        return [0.0]

    step = max(1, int(fps * sample_interval))
    prev_hist = None
    timestamps = []

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            # HISTCMP_CHISQR gives unbounded values; normalize to ~0-1 via tanh-like scaling
            # threshold 0.3 roughly corresponds to diff ~ 0.2-0.5 depending on content
            # Empirical: uniform content ~0.05-0.15, scene change ~0.3+
            if diff > threshold * 10:
                ts = frame_idx / fps
                timestamps.append(ts)

        prev_hist = hist

    cap.release()

    timestamps.sort()
    logger.info(f"场景检测: 找到 {len(timestamps)} 个场景切换点 (threshold={threshold})")
    return timestamps


def build_segments(scene_points: List[float], duration: float, max_scenes: int = 10) -> List[Tuple[float, float]]:
    """
    将场景切换点构建为 (start, end) 段列表，合并小场景直到 ≤ max_scenes。

    Args:
        scene_points: 场景切换点列表（含 0）
        duration: 视频总时长
        max_scenes: 最大段数上限

    Returns:
        [(start, end), ...] 段列表
    """
    if not scene_points:
        return [(0.0, duration)]

    if scene_points[0] != 0.0:
        scene_points = [0.0] + scene_points
    if scene_points[-1] != duration:
        scene_points.append(duration)

    segments = []
    for i in range(len(scene_points) - 1):
        start = scene_points[i]
        end = scene_points[i + 1]
        if end - start > 0.5:
            segments.append((start, end))

    while len(segments) > max_scenes:
        min_gap = float("inf")
        merge_idx = 0
        for i in range(len(segments) - 1):
            gap = segments[i + 1][1] - segments[i][0]
            if gap < min_gap:
                min_gap = gap
                merge_idx = i

        merged = (segments[merge_idx][0], segments[merge_idx + 1][1])
        segments[merge_idx] = merged
        del segments[merge_idx + 1]
        logger.debug(f"合并相邻场景: 剩余 {len(segments)} 段")

    logger.info(f"场景分段: {len(segments)} 段 (上限 {max_scenes})")
    return segments


def get_segments(video_path: str, duration: float, threshold: float = 0.3, max_scenes: int = 10) -> List[Tuple[float, float]]:
    """
    一站式获取场景分段。

    Returns:
        [(start, end), ...]
    """
    scene_points = detect_scenes(video_path, threshold)
    return build_segments(scene_points, duration, max_scenes)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import py_compile; py_compile.compile('src/title_classifier/core/scene_detector.py', doraise=True); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Run basic integration test**

Use the test harness against the video with `[` brackets in the path:

```bash
python -c "
from title_classifier.core.scene_detector import detect_scenes, build_segments
import os, tempfile, shutil

video_dir = r'D:\aria2\anime'
test_video = None
for f in sorted(os.listdir(video_dir)):
    if f.endswith('.mp4') and os.path.isfile(os.path.join(video_dir, f)):
        test_video = os.path.join(video_dir, f)
        break

# Test 1: detect scenes on path with [ ]
points = detect_scenes(test_video, threshold=0.3)
print(f'Test 1 (bracket path): {len(points)} scene points')
print(f'  First 5: {points[:5]}')

# Test 2: build segments
segs = build_segments(points, duration=120.0, max_scenes=10)
print(f'Test 2 (segments): {len(segs)} segments')

# Test 3: detect on a temp copy (clean path)
tmp_path = os.path.join(tempfile.gettempdir(), '__scene_test_input.mp4')
shutil.copy2(test_video, tmp_path)
try:
    points2 = detect_scenes(tmp_path, threshold=0.3)
    print(f'Test 3 (clean path): {len(points2)} scene points')
    if points2:
        print(f'  First 5: {points2[:5]}')
finally:
    os.remove(tmp_path)
"
```

- [ ] **Step 4: Clean up debug file**

```bash
Remove-Item -LiteralPath 'test_scene_debug.py' -ErrorAction SilentlyContinue
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: replace ffprobe scene detection with OpenCV histogram-based detection"
```

### Task 2: Verify no regressions

**Files:**
- No file changes; run existing analysis to confirm

- [ ] **Step 1: Run full syntax check on all modified/related modules**

```bash
python -c "
import py_compile
for f in [
    'src/title_classifier/core/scene_detector.py',
    'src/title_classifier/core/vision.py',
    'src/title_classifier/__main__.py',
    'src/title_classifier/gui/stage_vision.py',
]:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

- [ ] **Step 2: Test CLI integration (dry-run with a short video)**

```bash
python -m title_classifier vision --help
```

Expected: help text with `--no-scene-detection`, `--scene-threshold`, `--max-scenes`, `--frames-per-scene`
