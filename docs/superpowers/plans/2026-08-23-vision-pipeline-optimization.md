# 视觉分析链路优化与冗余清理 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除视觉分析主链路（帧提取→YOLO→CLIP→VLM）中的重复 IO/推理冗余，删除死代码，并修复 prompt 构建中的主要姿态 bug。

**Architecture:** 用单次解码的 cv2 批量抽帧替换逐帧 ffmpeg 子进程；解码后的帧数组在 YOLO→CLIP→VLM 全链路内存复用；debug 绘图复用已有推理结果；YOLO 三模型加早退规则；两条采样路径的运动检测逻辑合并为一个公共方法。

**Tech Stack:** Python 3.10+, OpenCV (cv2), numpy, ultralytics YOLO, open_clip, pytest。

## Global Constraints

- 测试命令统一用：`& ".venv\Scripts\python.exe" -m pytest ...`（PowerShell，仓库根目录执行）
- 每个任务独立提交；提交信息用 conventional commits（perf:/fix:/refactor:/chore:）
- 不改变对外行为契约：`process_and_save` / `process_video` 返回结构不变；`timeline` 条目字段只增不减
- 合成测试视频用 `.avi` + MJPG 编码（Windows 上 OpenCV 必定可用），不用 mp4
- 新测试直接 `from title_classifier...` 导入（包已 editable 安装）；需要绕开重依赖时用 `VisionProcessor.__new__(VisionProcessor)` 手工设属性（参照现有 `tests/test_clip_diff.py` 的思路）
- 明确不做（后续计划）：硬件加速接入、CLIP diff 向量化、场景检测降采样、`logs/_vision_tmp` 清理策略、`process_image` 临时目录、场景模式字幕丢失修复

---

### Task 1: 修复主要姿态 bug（字符串 join + prompt 重复行）

**Files:**
- Modify: `src/title_classifier/core/vision.py`（`_build_comprehensive_context` 内，约 1089 行）
- Test: `tests/test_vision_context.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: `_build_comprehensive_context(video_summary: Dict, frame_descriptions: List[str], n_frames: int) -> str` 行为修正（签名不变）

背景：`video_summary["main_pose"]` 是字符串（来自 `max(pose_counts, key=...)` 或 `"未知"`）。当前代码 `', '.join(main_pose)` 会把字符串逐字拆开（如 `"站立"` → `"站,立"`）；且该方法输出两行"- 主要姿态"（1089 行的 join 版 + 1103 行的结构化版本），prompt 冗余。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_vision_context.py`：

```python
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
    assert "站,立" not in ctx  # 未被拆字


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
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_vision_context.py -v`
Expected: 两个测试 FAIL（存在 `"站,立"` 且 `- 主要姿态` 出现 2 次）

- [ ] **Step 3: 修改实现**

在 `src/title_classifier/core/vision.py` 的 `_build_comprehensive_context` 中，删除这一行：

```python
context_lines.append(f"- 主要姿态: {', '.join(video_summary.get('main_pose', ['未知']))}")
```

保留下方结构化的姿态分布块（含 `- 主要姿态: {main_pose} (占{main_count}/{total_person}帧)` 的那行）。

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_vision_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/title_classifier/core/vision.py tests/test_vision_context.py
git commit -m "fix(vision): 修复主要姿态被逐字join及prompt重复行"
```

---

### Task 2: cv2 单次解码批量抽帧 + 数组转 base64 工具

**Files:**
- Modify: `src/title_classifier/utils/video.py`（文件末尾追加新函数）
- Modify: `src/title_classifier/utils/image.py`（重构 + 新函数）
- Test: `tests/test_video_extract.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces:
  - `extract_frames_cv2(video_path: str, output_dir: str, timestamps: List[float], max_size: int = 400, quality: int = 90) -> List[Optional[str]]` —— 返回列表与输入 timestamps 一一对应，失败位为 None。文件名沿用现有模式 `frame_{i:04d}_{ts:.1f}s.jpg`（i 为输入下标）
  - `image_array_to_base64(img_bgr: np.ndarray, max_size: int = 640, quality: int = 75) -> str` —— BGR ndarray → base64 JPEG
  - 重构后 `compress_image` / `image_to_base64` 行为不变（`image_to_base64` 改为内部委托 `image_array_to_base64`）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_video_extract.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_video_extract.py -v`
Expected: ImportError / AttributeError（`extract_frames_cv2`、`image_array_to_base64` 不存在）

- [ ] **Step 3: 实现 `extract_frames_cv2`**

在 `src/title_classifier/utils/video.py` 末尾追加：

```python
def extract_frames_cv2(
    video_path: str,
    output_dir: str,
    timestamps: List[float],
    max_size: int = 400,
    quality: int = 90,
) -> List[Optional[str]]:
    """
    用 cv2 单次顺序解码批量提取帧（grab 跳帧 + retrieve 仅解码目标帧），
    替代逐帧 ffmpeg 子进程。

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        timestamps: 目标时间戳（秒）列表
        max_size: 输出图片最长边
        quality: JPEG 质量

    Returns:
        与 timestamps 对齐的路径列表；失败位置为 None。
        文件名: frame_{i:04d}_{ts:.1f}s.jpg（i 为 timestamps 下标）
    """
    if not timestamps:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(idx: int, ts: float, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(
                frame,
                (max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2),
                interpolation=cv2.INTER_AREA,
            )
        path = out_dir / f"frame_{idx:04d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return str(path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}")
        return [None] * len(timestamps)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        logger.warning(f"视频元数据无效，无法批量抽帧: {video_path}")
        return [None] * len(timestamps)

    # 目标帧索引 → 时间戳下标列表（多个时间戳可能落在同一帧）
    target_map: Dict[int, List[int]] = {}
    for i, ts in enumerate(timestamps):
        idx = min(max(int(round(ts * fps)), 0), total - 1)
        target_map.setdefault(idx, []).append(i)

    results: List[Optional[str]] = [None] * len(timestamps)
    frame_idx = -1
    while frame_idx < total - 1 and target_map:
        if not cap.grab():  # 快速跳过，不解码
            break
        frame_idx += 1
        if frame_idx not in target_map:
            continue
        ok, frame = cap.retrieve()  # 仅对目标帧解码
        wanted = target_map.pop(frame_idx)
        if not ok or frame is None:
            continue
        for i in wanted:
            results[i] = _save(i, timestamps[i], frame)

    cap.release()

    got = sum(1 for r in results if r is not None)
    logger.info(f"cv2批量抽帧: {got}/{len(timestamps)} 帧 ({video_path})")
    return results
```

注意：`video.py` 顶部 import 需补充 `Optional` 和 `Dict`（当前只有 `List, Optional, Tuple` —— 检查后补缺）。

- [ ] **Step 4: 重构 `utils/image.py`**

将压缩+编码公共逻辑抽出，新增 `image_array_to_base64`：

```python
def _resize_and_encode_jpeg(img: np.ndarray, max_size: int, quality: int) -> np.ndarray:
    """缩放到最长边 max_size 并编码为 JPEG buffer"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(
            img,
            (max(2, int(w * scale)) // 2 * 2, max(2, int(h * scale)) // 2 * 2),
            interpolation=cv2.INTER_AREA,
        )
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer


def image_array_to_base64(img_bgr: np.ndarray, max_size: int = 640, quality: int = 75) -> str:
    """BGR ndarray 直接转 base64 JPEG（避免落盘后再读盘）"""
    try:
        buffer = _resize_and_encode_jpeg(img_bgr, max_size, quality)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"图像数组转base64失败: {e}")
        return ""
```

`image_to_base64` 改为解码后委托：

```python
def image_to_base64(image_path: str, max_size: int = 640) -> str:
    """读取图片并压缩后转base64"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"无法解码图片: {image_path}")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return image_array_to_base64(img, max_size=max_size)
    except Exception as e:
        logger.error(f"图片转base64失败: {e}")
        return ""
```

（`compress_image` 可顺手改用 `_resize_and_encode_jpeg`，也可不动——不动更稳。）

- [ ] **Step 5: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_video_extract.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/title_classifier/utils/video.py src/title_classifier/utils/image.py tests/test_video_extract.py
git commit -m "feat(utils): 新增cv2单次解码批量抽帧与数组转base64工具"
```

---

### Task 3: 主链路接入批量抽帧 + 内存帧复用（comprehensive 与 segment 两条路径）

**Files:**
- Modify: `src/title_classifier/core/vision.py`（`_analyze_video_comprehensive` 约 672-857 行、CLIP diff 块约 245-259 行、VLM 调用约 1152-1225 行、`_analyze_video_segment` 约 519-600 行）
- Modify: `src/title_classifier/detectors/clip.py`（`compute_frame_diff_scores` 开头加 None 守卫）
- Test: `tests/test_vision_pipeline.py`（新建）

**Interfaces:**
- Consumes: Task 2 的 `extract_frames_cv2`、`image_array_to_base64`
- Produces:
  - `video_analysis` 字典新增键 `"decoded_frames": List[Optional[np.ndarray]]`（与 `"frames"` 对齐）；`"frames"`、`"timeline"` 键不变
  - `_call_vlm_comprehensive(frames, ...)` 中 `frames` 元素可为 `str`（路径）或 `np.ndarray`（BGR），通过内部 `_to_base64_item()` 分派
  - `compute_frame_diff_scores(frames)` 支持 `None` 元素（对应帧记 0.0 分，保持索引对齐）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_vision_pipeline.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_vision_pipeline.py -v`
Expected: FAIL（无 `decoded_frames` 键；`compute_frame_diff_scores` 对 None 抛异常或错位；ndarray 帧走不通）

- [ ] **Step 3: 修改 `_analyze_video_comprehensive`**

核心改动（vision.py）：

1. 循环前一次性批量抽帧，替代循环内 `extract_frame(...)`：

```python
from ..utils.video import get_video_duration, extract_frame, extract_multiple_frames, detect_keyframes, extract_frames_cv2  # 补充导入

# 计算采样时间点（原逻辑不变）
timestamps = np.arange(0, duration, self.analysis_step)
if len(timestamps) > self.max_sample_frames:
    timestamps = np.linspace(0, duration, self.max_sample_frames)

# 一次性批量抽帧（单次解码）
extracted_paths = extract_frames_cv2(
    video_path, str(tmp_dir), [float(t) for t in timestamps], max_size=400
)
```

2. 主循环改为遍历 `enumerate(extracted_paths)`，跳过 None：

```python
decoded_frames: list = []
frames: list = []
timeline: list = []

for i, (ts, frame_path) in enumerate(zip(timestamps, extracted_paths)):
    if frame_path is None:
        decoded_frames.append(None)
        continue
    data = np.fromfile(frame_path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 全链路唯一一次解码
    if frame is None:
        decoded_frames.append(None)
        continue
    frames.append(frame_path)
    decoded_frames.append(frame)
    # ……以下运动检测/YOLO 逻辑不变……
```

注意：原代码 `frames` 只 append 成功项导致其与时间戳错位；新实现中 `frames` 与 `timestamps`/`decoded_frames` 保持严格对齐（失败占位不进 `frames` 但 `decoded_frames` 占位 None）。`selected_indices` 是 timeline 下标而非 frames 下标——检查 `frames_for_vlm = [selected_frames[i] for i in selected_indices ...]` 这行：改为基于 `decoded_frames` 取选中帧数组（见第 4 点），并同步修正 `_process_video_by_scenes` 外其他引用。

3. 返回值增加键：

```python
return {
    "frames": frames,
    "decoded_frames": decoded_frames,
    "timeline": timeline,
    "duration": duration,
}
```

4. CLIP diff 块改为复用内存数组（vision.py 约 249-253 行）：

```python
all_frames = video_analysis["frames"]
decoded_all = video_analysis.get("decoded_frames", [])
# 传完整对齐列表（含 None），保证分数索引 == 帧索引
frames_for_diff = list(decoded_all) if decoded_all else \
    [cv2.imread(f) for f in all_frames]
```

5. VLM 帧传数组：`_call_vlm_comprehensive` 及关键词重试路径中，帧元素允许是 ndarray；新增私有方法：

```python
def _to_base64_item(self, item) -> str:
    """帧元素分派：str 路径读盘编码；ndarray 直接编码"""
    if isinstance(item, np.ndarray):
        return image_array_to_base64(item, max_size=self.max_image_size)
    return image_to_base64(item, max_size=self.max_image_size)
```

`_call_vlm_comprehensive` 内所有 `image_to_base64(f, ...)` 替换为 `self._to_base64_item(f)`；`process_video` 中：

```python
decoded_all = video_analysis.get("decoded_frames", [])
frames_for_vlm = [decoded_all[i] if i < len(decoded_all) and decoded_all[i] is not None
                  else selected_frames[i]
                  for i in selected_indices]
```

同时 `analysis_result["frames_for_vlm"]` 现在可能是 ndarray——下游 `_save_vlm_covers` 用 `shutil.copy2` 复制文件会失败，需改为仅当元素是 str 时复制（加 `isinstance(frame_path, (str, Path))` 守卫）。

- [ ] **Step 4: `compute_frame_diff_scores` 加 None 守卫（clip.py）**

编码循环处：

```python
for frame in frames:
    emb = self._encode_image_array(frame) if frame is not None else None
    embeddings.append(emb)
```

其余逻辑已天然支持 `embeddings[i] is None → 0.0`，无需再改。

- [ ] **Step 5: 同样改造 `_analyze_video_segment`**

- 用 `extract_frames_cv2` 一次性抽取该段 `n_frames` 个时间点（替换循环内 `extract_frame`）
- `seg_analysis` 增加 `"decoded"` 键（对齐数组），`_call_vlm_comprehensive(seg_analysis["decoded"], ...)` 直接收数组

- [ ] **Step 6: 运行全部相关测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_vision_pipeline.py tests/test_video_extract.py tests/test_vision_context.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/title_classifier/core/vision.py src/title_classifier/detectors/clip.py tests/test_vision_pipeline.py
git commit -m "perf(vision): 批量抽帧单次解码,CLIP/VLM复用内存帧数组"
```

---

### Task 4: debug 绘图复用已有推理结果（消除重复 YOLO 推理）

**Files:**
- Modify: `src/title_classifier/core/vision.py`（`_save_detection_debug` 约 1247-1260 行）
- Test: `tests/test_debug_no_reinfer.py`（新建）

**Interfaces:**
- Consumes: `timeline` 条目已有的 `raw_pose` 键（`_analyze_video_comprehensive` 已存入）
- Produces: `_save_detection_debug(video_analysis: Dict, debug_dir: Path)` 不再调用任何模型

- [ ] **Step 1: 写失败测试**

创建 `tests/test_debug_no_reinfer.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_debug_no_reinfer.py -v`
Expected: FAIL（`infer_calls["n"] == 1`，因为现实现对每帧重跑 `estimate_pose`）

- [ ] **Step 3: 修改实现**

`_save_detection_debug` 中，将：

```python
pose_result = self.yolo_detector.estimate_pose(frame)
```

替换为：

```python
pose_result = entry.get("raw_pose")
if pose_result is None:
    continue  # 无缓存结果则跳过绘制标注（不重新推理）
```

（外层的 `try` 结构和 `draw_pose_on_frame` 调用保持；若该条目没有 `raw_pose` 则只保存原图与 JSON。）

- [ ] **Step 4: 运行确认通过**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_debug_no_reinfer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/title_classifier/core/vision.py tests/test_debug_no_reinfer.py
git commit -m "perf(vision): debug绘图复用缓存的raw_pose结果,不再重复推理"
```

---

### Task 5: 移除每帧 torch.cuda.empty_cache()

**Files:**
- Modify: `src/title_classifier/detectors/yolo.py`（`analyze_comprehensive` 内约 563-569 行）

**Interfaces:**
- Consumes: 无
- Produces: `YOLODetector.analyze_comprehensive(frame)` 返回结构不变；仅删除性能反模式代码

- [ ] **Step 1: 删除代码块**

删除 `analyze_comprehensive` 中的整段：

```python
# CUDA显存清理
if self.device == "cuda":
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
```

理由：每帧调用 `empty_cache()` 会释放 CUDA 缓存池，迫使下一帧重新 cudaMalloc，显著拖慢连续推理。PyTorch 的缓存分配器自己管理显存，无需手动清理。（`_analyze_video_comprehensive` 里每个视频一次的 warmup 保留，不受影响。）

- [ ] **Step 2: 回归测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_detectors -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/title_classifier/detectors/yolo.py
git commit -m "perf(yolo): 移除每帧cuda.empty_cache反模式"
```

---

### Task 6: analyze_comprehensive 早退（detect+pose 均无人则跳过 segment）

**Files:**
- Modify: `src/title_classifier/detectors/yolo.py`（`analyze_comprehensive` 约 508-591 行）
- Test: `tests/test_yolo_early_exit.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: `analyze_comprehensive` 行为变化：当 detect 与 pose 都已运行且都返回 `has_person=False` 时，跳过 segment 推理（投票不可能达到 2）。返回结构不变。已知取舍：此前"仅 segment 检出人体"的帧 `has_person` 为 True，现在为 False（此类帧 vote_result 本来就是 False，影响限于 summary 的 person 统计口径）。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_yolo_early_exit.py`：

```python
"""analyze_comprehensive 早退逻辑"""

import numpy as np

_NO_PERSON_DETECT = {"has_person": False, "persons": [], "max_confidence": 0.0}
_NO_PERSON_POSE = {"has_person": False, "poses": [], "max_confidence": 0.0}


class _Recorder:
    """记录各模型是否被调用的假检测器"""

    def __init__(self, models):
        self._models = {m: object() for m in models}   # analyze_comprehensive 按 _models 分派
        self.calls = []

    def detect(self, frame):
        self.calls.append("detect")
        return dict(_NO_PERSON_DETECT)

    def estimate_pose(self, frame):
        self.calls.append("pose")
        return dict(_NO_PERSON_POSE)

    def segment_instances(self, frame):
        self.calls.append("segment")
        return {"has_person": True, "segments": [{"confidence": 0.9}], "max_confidence": 0.9}

    def _merge_results(self, d, p, s):
        from title_classifier.detectors.yolo import YOLODetector
        return YOLODetector._merge_results(None, d, p, s)

    def _calculate_weighted_confidence(self, d, p, s):
        from title_classifier.detectors.yolo import YOLODetector
        return YOLODetector._calculate_weighted_confidence(None, d, p, s)


def _make_detector(models):
    from title_classifier.detectors.yolo import YOLODetector
    det = YOLODetector.__new__(YOLODetector)
    BaseDetector_init_skip(det)
    det._models = {m: object() for m in models}
    det._backend_type = {m: "test" for m in models}
    det.confidence = 0.5
    det.iou_threshold = 0.45
    det.device = "cpu"
    return det


def BaseDetector_init_skip(det):
    from title_classifier.detectors.base import BaseDetector
    det._loaded = True


def test_skips_segment_when_detect_and_pose_negative():
    det = _make_detector(["detect", "pose", "segment"])
    rec = _Recorder(["detect", "pose", "segment"])
    # 把假方法挂到真实例上以记录调用
    det.detect = rec.detect
    det.estimate_pose = rec.estimate_pose
    det.segment_instances = rec.segment_instances

    result = det.analyze_comprehensive(np.zeros((32, 32, 3)))

    assert rec.calls == ["detect", "pose"]  # segment 被跳过
    assert result["has_person"] is False


def test_segment_runs_when_pose_positive():
    det = _make_detector(["detect", "pose", "segment"])
    rec = _Recorder(["detect", "pose", "segment"])

    def detect_pos(frame):
        rec.calls.append("detect")
        return {"has_person": True, "persons": [{"confidence": 0.9}], "max_confidence": 0.9}

    def pose_pos(frame):
        rec.calls.append("pose")
        return {"has_person": True, "poses": [{"avg_confidence": 0.9}], "max_confidence": 0.9}

    det.detect = detect_pos
    det.estimate_pose = pose_pos
    det.segment_instances = rec.segment_instances

    result = det.analyze_comprehensive(np.zeros((32, 32, 3)))

    assert "segment" in rec.calls  # 有人体时不早退（segment 提供穿着分析）
    assert result["has_person"] is True
```

注意：`_Recorder` 类在此测试里实际未用到（方法直接内联定义了），可删掉 `_Recorder` 仅保留内联版本，避免死代码——最终以简洁版为准。

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_yolo_early_exit.py -v`
Expected: 第一个测试 FAIL（`rec.calls == ["detect", "pose", "segment"]`，segment 未被跳过）

- [ ] **Step 3: 实现**

`analyze_comprehensive` 中，将三个独立的 `if "xxx" in self._models:` 块改为带早退判断：

```python
results = {...}  # 原样

model_timing = {}

detect_result = None
pose_result = None

if "detect" in self._models:
    try:
        t0 = time.perf_counter()
        detect_result = self.detect(frame)
        results["detection"] = detect_result
        model_timing["detect"] = time.perf_counter() - t0
        results["models_used"].append("detect")
    except Exception as e:
        model_timing["detect"] = 0
        logger.warning(f"detect模型推理失败: {e}")

if "pose" in self._models:
    try:
        t0 = time.perf_counter()
        pose_result = self.estimate_pose(frame)
        results["pose"] = pose_result
        model_timing["pose"] = time.perf_counter() - t0
        results["models_used"].append("pose")
    except Exception as e:
        model_timing["pose"] = 0
        logger.warning(f"pose模型推理失败: {e}")

# 早退：detect 与 pose 都运行且均为阴性 → 投票不可能≥2，跳过 segment
skip_segment = (
    "segment" in self._models
    and "detect" in self._models and "pose" in self._models
    and not (detect_result or {}).get("has_person", False)
    and not (pose_result or {}).get("has_person", False)
)
if skip_segment:
    logger.debug("detect与pose均未检出人体，跳过segment推理")
elif "segment" in self._models:
    try:
        t0 = time.perf_counter()
        results["segment"] = self.segment_instances(frame)
        model_timing["segment"] = time.perf_counter() - t0
        results["models_used"].append("segment")
    except Exception as e:
        model_timing["segment"] = 0
        logger.warning(f"segment模型推理失败: {e}")

# ……后续 merged/confidence/has_person 逻辑原样……
```

（保留原有的 debug 日志行可按需精简；`model_timing` 汇总到 `results["model_timing"]` 不变。）

- [ ] **Step 4: 运行确认通过 + 回归**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_yolo_early_exit.py tests/test_detectors -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/title_classifier/detectors/yolo.py tests/test_yolo_early_exit.py
git commit -m "perf(yolo): detect+pose均阴性时早退跳过segment推理"
```

---

### Task 7: 运动检测跳帧逻辑去重（公共 helper，segment 版获得 min_interval 保护）

**Files:**
- Modify: `src/title_classifier/core/vision.py`
  - 新增方法 `_should_skip_inference(prev_frame_gray, curr_frame, ts, last_forced_ts) -> bool`（放在 `_detect_motion` 之后）
  - `_analyze_video_comprehensive` 循环内（约 726-755 行）与 `_analyze_video_segment` 循环内（约 546-560 行）改用该 helper
- Test: `tests/test_motion_skip.py`（新建）

**Interfaces:**
- Consumes: `_detect_motion(prev, curr, threshold)`（已有）
- Produces: `_should_skip_inference(prev_frame_gray: Optional[np.ndarray], curr_frame: np.ndarray, ts: float, last_forced_ts: float) -> bool` —— True 表示可跳过本轮 YOLO 推理。语义：距上次强制推理不足 `motion_min_interval` 秒时才允许跳过静止帧（周期性强制推理防漂移）；`prev_frame_gray is None` 或运动明显时不跳过。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_motion_skip.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_motion_skip.py -v`
Expected: AttributeError（方法不存在）

- [ ] **Step 3: 实现 helper**

在 `_detect_motion` 后新增：

```python
def _should_skip_inference(
    self,
    prev_frame_gray: Optional[np.ndarray],
    curr_frame: np.ndarray,
    ts: float,
    last_forced_ts: float,
) -> bool:
    """
    判定当前帧是否可以跳过 YOLO 推理（静止画面复用上一帧结果）。

    仅当距上次强制推理不足 motion_min_interval 秒时才允许跳过，
    保证长时间静止场景也会周期性强制推理，防止状态漂移。
    """
    if not self.motion_detection:
        return False
    if prev_frame_gray is None:
        return False
    if ts - last_forced_ts >= self.motion_min_interval:
        return False  # 达到最小强制间隔，必须推理
    has_motion, _ = self._detect_motion(prev_frame_gray, curr_frame)
    return not has_motion
```

- [ ] **Step 4: 两条路径接入**

`_analyze_video_comprehensive` 循环内，原"运动检测：判断是否可以跳过YOLO推理"整段（`should_skip = False` 到 `logger.debug(...跳过YOLO推理)`）替换为：

```python
should_skip = self._should_skip_inference(prev_frame_gray, frame, ts, last_forced_timestamp)
if should_skip:
    motion_skipped_count += 1
```

`_analyze_video_segment` 循环内同样替换，并在函数开头初始化 `last_forced_timestamp = -float('inf')`，推理成功后更新 `last_forced_timestamp = ts`（原来该变量不存在，这是本次补齐的保护）。

- [ ] **Step 5: 运行全部相关测试**

Run: `& ".venv\Scripts\python.exe" -m pytest tests/test_motion_skip.py tests/test_vision_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/title_classifier/core/vision.py tests/test_motion_skip.py
git commit -m "refactor(vision): 抽取公共跳帧判定,segment路径补齐强制间隔保护"
```

---

### Task 8: 死代码删除

**Files:**
- Modify: `src/title_classifier/core/vision.py`
  - 删除方法：`_process_video_traditional`（约 647-670）、`_extract_frames`（约 1430-1454）、`_analyze_poses`（约 1456-1471）、`_select_frames`（约 1473-1524）、`_build_yolo_context`（约 1526-1545）、`_call_vlm`（约 1547-1604）、`_build_vision_prompt`（约 1606-1635）、`_append_audio_subtitles`（约 2254-2304）
- Modify: `src/title_classifier/detectors/yolo.py`
  - 删除模块级函数：`detect_human_yolo`（约 1004-1007）、`find_human_frame_yolo`（约 1010-1081）
- Modify: `src/title_classifier/detectors/clip.py`
  - 删除方法：`detect_change_by_embedding`（约 491-520）、`compare_frames`（约 522-595）
  - 删除模块级：`_global_classifier`、`get_classifier`、`classify_image`（约 598-618）
- Test: 无新测试（回归验证）

**Interfaces:**
- Consumes: 无
- Produces: 以上符号从包中移除（已 grep 确认全仓库零引用，包括 scripts/、gui/、tests/）

依据（2026-08-23 分析）：这些"传统模式"方法只被彼此引用，入口 `process_video` 仅路由到 `_process_video_by_scenes` / `_process_video_comprehensive`；`find_human_frame_yolo`、`compare_frames` 等在 src/scripts/gui/tests 中均无调用方。

- [ ] **Step 1: 删除前再次全局确认零引用**

Run: `rg "_process_video_traditional|_extract_frames\b|_analyze_poses|_select_frames\b|_build_yolo_context|_call_vlm\b|_build_vision_prompt|_append_audio_subtitles|detect_human_yolo|find_human_frame_yolo|compare_frames|detect_change_by_embedding|get_classifier|classify_image" src scripts gui tests`
Expected: 仅命中定义处（vision.py / yolo.py / clip.py 内部互调），无外部调用

- [ ] **Step 2: 执行删除**

按上面 Files 清单逐一删除。注意：
- 删除 `_extract_frames` 后，vision.py 顶部 `detect_keyframes`、`extract_multiple_frames` 导入若无其他使用一并移除（先 rg 确认）
- 删除 `find_human_frame_yolo` 后，yolo.py 若残留 `subprocess`/`hashlib` 局部导入随之消失（它们是函数内局部导入，随函数删除）
- clip.py 删除后检查 `json`、`time` 等顶部导入是否仍被使用

- [ ] **Step 3: 全量回归**

Run: `& ".venv\Scripts\python.exe" -m pytest -v`
Expected: 全部 PASS

- [ ] **Step 4: 冒烟验证 CLI 入口可加载**

Run: `& ".venv\Scripts\python.exe" -c "from title_classifier.core import VisionProcessor; from title_classifier.detectors.clip import CLIPClassifier; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 5: Commit**

```bash
git add -A src/title_classifier
git commit -m "chore(vision): 删除传统模式死链与未使用的兼容函数(~300行)"
```

---

## Self-Review 结论

- 覆盖对照：分析报告的性能五件套（#1 批量抽帧=Task 2/3、#2 内存复用=Task 3、#3 debug 重推理=Task 4、#4 empty_cache=Task 5、#5 早退=Task 6）、bug #16=Task 1、运动逻辑去重 #12=Task 7、死代码 #10/#11=Task 8。✅
- 类型一致性：`extract_frames_cv2` 返回 `List[Optional[str]]`（Task 2 定义，Task 3 消费一致）；`decoded_frames` 与 `frames` 对齐约定在 Task 3 内部闭环；`_to_base64_item` 在 Task 3 定义并即用。✅
- 占位符扫描：无 TBD/TODO；所有代码步骤给出具体代码。✅
- 风险点已显式记录：Task 6 的统计口径取舍、Task 3 中 `frames_for_vlm` 变 ndarray 对 `_save_vlm_covers` 的影响（已加守卫方案）。
