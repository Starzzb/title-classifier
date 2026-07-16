# Config-Driven Vision Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `config/default.toml` (+ `user.toml` override) the source of truth for all vision parameters, with CLI args and GUI as overrides. Currently these values are hardcoded in 3 places (Python default, argparse default, GUI StringVar) and `config/default.toml` is effectively ignored.

**Architecture:** Three-layer cascade: `user override → CLI arg / GUI input → config file → Python fallback`. VisionProcessor accepts an optional `config` dict; `__main__.py` reads config to set argparse defaults; GUI reads config to set form defaults.

**Tech Stack:** Existing `load_merged_config()` / `get_config_value()` utilities.

---

### Task 1: Make VisionProcessor accept config dict

**Files:**
- Modify: `src/title_classifier/core/vision.py` (lines 30-81)

- [ ] **Step 1: Modify `__init__` signature — add `config` param, change defaults to `None`**

Replace the entire `__init__` signature and body (lines 33-89):

```python
    def __init__(
        self,
        config: dict = None,
        provider: str = None,
        use_yolo: bool = None,
        yolo_model: str = None,
        yolo_models: List[str] = None,
        yolo_conf: float = None,
        use_clip: bool = None,
        clip_threshold: float = None,
        max_image_size: int = None,
        vlm_frames: int = None,
        analysis_step: float = None,
        max_sample_frames: int = None,
        debug_dir: str = None,
        covers_dir: str = None,
        db_store=None,
        device: str = None,
        motion_detection: bool = None,
        motion_threshold: float = None,
        motion_min_interval: float = None,
        backend: str = None,
        use_scene_detection: bool = None,
        scene_threshold: float = None,
        max_scenes: int = None,
        frames_per_scene: int = None,
    ):
        from ..utils.config import get_config_value
        if config is None:
            config = {}
        cv = lambda key, fallback: get_config_value(config, key, fallback)

        self.provider = provider or cv("providers.stage_providers.vision", "gcli")
        self.use_yolo = use_yolo if use_yolo is not None else cv("yolo.comprehensive.enabled", False)
        self.yolo_model = yolo_model or cv("yolo.model_type", "pose")
        self.yolo_models = yolo_models or cv("yolo.comprehensive.models", ["pose"])
        self.yolo_conf = yolo_conf if yolo_conf is not None else cv("yolo.confidence", 0.5)
        self.use_clip = use_clip if use_clip is not None else False
        self.clip_threshold = clip_threshold if clip_threshold is not None else cv("clip.threshold", 0.25)
        self.max_image_size = max_image_size if max_image_size is not None else cv("vision.max_image_size", 640)
        self.vlm_frames = vlm_frames if vlm_frames is not None else cv("vision.vlm_frames", 10)
        self.analysis_step = analysis_step if analysis_step is not None else cv("vision.analysis_step", 5.0)
        self.max_sample_frames = max_sample_frames if max_sample_frames is not None else cv("vision.max_sample_frames", 50)
        self.debug_dir = debug_dir
        self.covers_dir = covers_dir
        self.db_store = db_store
        self.device = self._resolve_device(device or cv("general.device", "cpu"))
        self.motion_detection = motion_detection if motion_detection is not None else cv("vision.motion_detection", True)
        self.motion_threshold = motion_threshold if motion_threshold is not None else cv("vision.motion_threshold", 8.0)
        self.motion_min_interval = motion_min_interval if motion_min_interval is not None else cv("vision.motion_min_interval", 5.0)
        self.backend = backend or cv("yolo.backend", "auto")
        self.use_scene_detection = use_scene_detection if use_scene_detection is not None else cv("scene_detection.enabled", True)
        self.scene_threshold = scene_threshold if scene_threshold is not None else cv("scene_detection.threshold", 0.3)
        self.max_scenes = max_scenes if max_scenes is not None else cv("scene_detection.max_scenes", 10)
        self.frames_per_scene = frames_per_scene if frames_per_scene is not None else cv("scene_detection.frames_per_scene", 10)
```

Note: the `cv` helper reads `get_config_value(config, dotted_key, fallback)`. Each param: if caller passed explicit value → use it; else if config has the key → use it; else use Python fallback.

- [ ] **Step 2: Verify syntax**

```bash
python -c "import py_compile; py_compile.compile('src/title_classifier/core/vision.py', doraise=True); print('OK')"
```

### Task 2: Make CLI argparse defaults read from config

**Files:**
- Modify: `src/title_classifier/__main__.py`

- [ ] **Step 1: Add config loading to vision arg setup, change `default=` to config-driven**

Replace the vision arg definitions block (lines 704-724) to read from config:

```python
    # vision 命令
    vision_cmd = subparsers.add_parser("vision", help="视觉识别")
    from .utils.config import load_merged_config, get_config_value
    cfg = load_merged_config()
    gv = lambda key, fallback: get_config_value(cfg, key, fallback)

    vision_cmd.add_argument("-c", "--csv", default="data/output/title_review.csv", help="CSV文件路径")
    vision_cmd.add_argument("-p", "--provider", default=gv("providers.stage_providers.vision", "gcli"), help="AI Provider")
    vision_cmd.add_argument("--use-yolo", action="store_true", help="使用YOLO姿态检测")
    vision_cmd.add_argument("--comprehensive", action="store_true", help="全面分析模式")
    vision_cmd.add_argument("--yolo-conf", type=float, default=gv("yolo.confidence", 0.5), help="YOLO置信度阈值")
    vision_cmd.add_argument("--use-clip", action="store_true", help="使用CLIP预分类")
    vision_cmd.add_argument("--clip-threshold", type=float, default=gv("clip.threshold", 0.25), help="CLIP置信度阈值")
    vision_cmd.add_argument("--max-image-size", type=int, default=gv("vision.max_image_size", 640), help="图片最大尺寸")
    vision_cmd.add_argument("--vlm-frames", type=int, default=gv("vision.vlm_frames", 10), help="VLM帧数")
    vision_cmd.add_argument("--analysis-step", type=float, default=gv("vision.analysis_step", 5.0), help="YOLO模式采样间隔（秒）")
    vision_cmd.add_argument("--max-sample-frames", type=int, default=gv("vision.max_sample_frames", 50), help="最大采样帧数上限")
    vision_cmd.add_argument("--device", default=gv("general.device", "cpu"), choices=["auto", "cuda", "cpu"], help="推理设备")
    vision_cmd.add_argument("--concurrent", type=int, default=4, help="并发处理视频数")
    vision_cmd.add_argument("--backend", default=gv("yolo.backend", "auto"), choices=["auto", "openvino", "pytorch"], help="YOLO推理后端")
    vision_cmd.add_argument("--no-motion-detection", action="store_true", help="禁用运动检测前置过滤")
    vision_cmd.add_argument("--motion-threshold", type=float, default=gv("vision.motion_threshold", 8.0), help="运动检测阈值（变化像素比例%%）")
    vision_cmd.add_argument("--no-scene-detection", action="store_true", help="禁用场景分段分析")
    vision_cmd.add_argument("--scene-threshold", type=float, default=gv("scene_detection.threshold", 0.3), help="场景检测敏感度")
    vision_cmd.add_argument("--max-scenes", type=int, default=gv("scene_detection.max_scenes", 10), help="最大场景段数")
    vision_cmd.add_argument("--frames-per-scene", type=int, default=gv("scene_detection.frames_per_scene", 10), help="每场景取帧数")
```

- [ ] **Step 2: Update `cmd_vision()` to pass config to VisionProcessor**

Modify the VisionProcessor instantiation in `cmd_vision()` (around line 169) to pass `config`:

```python
processor = VisionProcessor(
    config=cfg,
    provider=args.provider,
    use_yolo=args.use_yolo,
    yolo_model="pose" if args.use_yolo else "detect",
    yolo_models=yolo_models,
    yolo_conf=args.yolo_conf,
    use_clip=args.use_clip,
    clip_threshold=args.clip_threshold,
    max_image_size=args.max_image_size,
    vlm_frames=args.vlm_frames,
    analysis_step=args.analysis_step,
    max_sample_frames=args.max_sample_frames,
    debug_dir=debug_dir,
    device=args.device,
    motion_detection=not args.no_motion_detection,
    motion_threshold=args.motion_threshold,
    backend=args.backend,
    db_store=db,
    use_scene_detection=not args.no_scene_detection,
    scene_threshold=args.scene_threshold,
    max_scenes=args.max_scenes,
    frames_per_scene=args.frames_per_scene,
)
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import py_compile; py_compile.compile('src/title_classifier/__main__.py', doraise=True); print('OK')"
```

### Task 3: Make GUI defaults read from config

**Files:**
- Modify: `src/title_classifier/gui/stage_vision.py`

- [ ] **Step 1: Load config in the constructor and use for StringVar defaults**

In `_build_engine_section()`, after the existing code, load config and use it for defaults:

```python
        # Load config for defaults
        from ..utils.config import load_merged_config, get_config_value
        _cfg = load_merged_config()
        _gv = lambda key, fallback: get_config_value(_cfg, key, fallback)
```

Then change the hardcoded StringVar values:

| Old | New |
|-----|-----|
| `value="5.0"` (motion_threshold) | `value=str(_gv("vision.motion_threshold", 8.0))` |
| `value="2.0"` (analysis_step) → should already be `5.0` | `value=str(_gv("vision.analysis_step", 5.0))` |
| `value="50"` (max_sample_frames) | `value=str(_gv("vision.max_sample_frames", 50))` |
| `value="10"` (vlm_frames) | `value=str(_gv("vision.vlm_frames", 10))` |
| `value="640"` (max_image_size) | `value=str(_gv("vision.max_image_size", 640))` |
| `value="10"` (max_scenes) | `value=str(_gv("scene_detection.max_scenes", 10))` |
| `value="10"` (frames_per_scene) | `value=str(_gv("scene_detection.frames_per_scene", 10))` |
| `value="0.5"` (yolo_conf) | `value=str(_gv("yolo.confidence", 0.5))` |
| `value="0.25"` (clip_threshold) | `value=str(_gv("clip.threshold", 0.25))` |

Also update all the "skip if equals default" comparisons to compare against config values instead of hardcoded strings. For example, line 294:

```python
if threshold and threshold != str(_gv("vision.motion_threshold", 8.0)):
```

- [ ] **Step 2: Move the config loading import to the top of the method or reuse the existing load_merged_config calls**

Note: `load_merged_config` is already imported and called at lines 255-256 and 356-357 in `_run_vision()` / `_run_vision_retry()`. We should reuse the same pattern.

- [ ] **Step 3: Verify syntax**

```bash
python -c "import py_compile; py_compile.compile('src/title_classifier/gui/stage_vision.py', doraise=True); print('OK')"
```

### Task 4: Verify all call sites

**Files:**
- No changes; check that the existing single instantiation at `__main__.py:169` still works

- [ ] **Step 1: Run full syntax check**

```bash
python -c "
import py_compile
for f in [
    'src/title_classifier/core/vision.py',
    'src/title_classifier/__main__.py',
    'src/title_classifier/gui/stage_vision.py',
]:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

- [ ] **Step 2: Quick smoke test (CLI help)**

```bash
python -m title_classifier vision --help
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: make config/default.toml the source of truth for vision params

VisionProcessor now accepts optional config dict; params cascade:
user.toml > CLI arg/GUI > config/default.toml > Python fallback.
__main__.py reads config for argparse defaults.
GUI reads config for StringVar defaults."
```
