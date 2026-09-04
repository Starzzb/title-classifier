"""YOLO/CLIP 设备基准：OpenVINO-CPU vs CUDA-PyTorch

对同一视频采样 N 帧，分别以两种设备/后端跑完整 analyze_comprehensive
（detect+pose+segment 三模型）与可选的 CLIP 差异度评分，输出耗时对比。

用法:
    python scripts/bench_yolo_device.py <video_path> [--frames 30] [--no-clip]

说明:
    - CPU 基线 = device=cpu + backend=openvino（生产默认回退路径）
    - CUDA    = device=cuda + backend=pytorch
    - 前 3 帧为预热（CUDA kernel 编译/缓存），不计入统计
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import title_classifier.core  # noqa: F401  先导入 core 打破 detectors<->core 循环导入

import cv2
import numpy as np


def _get_duration(video_path: str) -> float:
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
    )
    return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else 0.0


def _sample_frames(video_path: str, n: int) -> list:
    from title_classifier.utils.video import extract_frames_cv2
    tmp = Path(tempfile.mkdtemp(prefix="bench_frames_"))
    duration = _get_duration(video_path)
    step = max(duration / (n + 1), 0.5)
    ts = [min(step * (i + 1), duration - 0.5) for i in range(n)]
    paths = extract_frames_cv2(video_path, str(tmp), ts, max_size=400)
    frames = []
    for p in paths:
        if p is None:
            continue
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def _bench_yolo(label: str, device: str, backend: str, frames: list) -> float:
    from title_classifier.detectors.yolo import YOLODetector
    det = YOLODetector(
        model_types=["detect", "pose", "segment"],
        device=device, backend=backend, confidence=0.5,
    )
    if not det.load_model():
        print(f"[{label}] 模型加载失败，跳过")
        return -1.0

    warm = min(3, len(frames))
    for f in frames[:warm]:
        det.analyze_comprehensive(f)

    t0 = time.perf_counter()
    for f in frames[warm:]:
        det.analyze_comprehensive(f)
    elapsed = time.perf_counter() - t0
    n = max(len(frames) - warm, 1)
    print(f"[{label}] YOLO三模型: {elapsed:.2f}s / {n}帧 = {elapsed/n*1000:.0f}ms/帧")
    return elapsed


def _bench_clip(label: str, device: str, frames: list) -> float:
    from title_classifier.detectors.clip import CLIPClassifier
    from title_classifier.utils.stats import TagStatistics
    clip = CLIPClassifier(device=device, tag_stats=TagStatistics())
    if not clip.load_model():
        print(f"[{label}] CLIP加载失败，跳过")
        return -1.0

    t0 = time.perf_counter()
    clip.compute_frame_diff_scores(frames)
    elapsed = time.perf_counter() - t0
    print(f"[{label}] CLIP差异度: {elapsed:.2f}s / {len(frames)}帧")
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO/CLIP 设备基准")
    ap.add_argument("video", help="测试视频路径")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--no-clip", action="store_true", help="跳过 CLIP 基准")
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"视频不存在: {args.video}")
        return 1

    print(f"采样 {args.frames} 帧: {args.video}")
    frames = _sample_frames(args.video, args.frames)
    if len(frames) < 5:
        print(f"有效帧不足: {len(frames)}")
        return 1
    print(f"有效帧: {len(frames)}\n")

    results = {}
    results["cpu_openvino_yolo"] = _bench_yolo("CPU/OpenVINO", "cpu", "openvino", frames)
    results["cuda_pytorch_yolo"] = _bench_yolo("CUDA/PyTorch", "cuda", "pytorch", frames)
    if not args.no_clip:
        results["cpu_openvino_clip"] = _bench_clip("CPU/OpenVINO", "cpu", frames)
        results["cuda_pytorch_clip"] = _bench_clip("CUDA/PyTorch", "cuda", frames)

    print("\n===== 对比 =====")
    yl = results["cpu_openvino_yolo"]
    yg = results["cuda_pytorch_yolo"]
    if yl > 0 and yg > 0:
        print(f"YOLO 加速比: {yl/yg:.2f}x  (CPU {yl:.2f}s vs CUDA {yg:.2f}s)")
    ck = results.get("cpu_openvino_clip", -1)
    cg = results.get("cuda_pytorch_clip", -1)
    if ck > 0 and cg > 0:
        print(f"CLIP 加速比: {ck/cg:.2f}x  (CPU {ck:.2f}s vs CUDA {cg:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
