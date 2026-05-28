"""
一键下载所有模型文件到 models/ 目录
运行方式: uv run python scripts/download_models.py
"""
import os
import sys
import shutil
from pathlib import Path

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ULTRALYTICS_HUB"] = "https://hf-mirror.com/ultralytics"

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def download_yolo_models():
    """下载 YOLO 模型"""
    print("=" * 50)
    print("1. 下载 YOLO 模型")
    print("=" * 50)

    yolo_dir = MODELS_DIR / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    models = ["yolov8n.pt", "yolov8n-pose.pt", "yolov8n-seg.pt"]

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[SKIP] ultralytics 未安装，跳过 YOLO 模型下载")
        print("       运行 `uv sync` 后重试")
        return

    for model_name in models:
        target_path = yolo_dir / model_name
        if target_path.exists():
            print(f"[SKIP] {model_name} 已存在")
            continue

        print(f"[下载] {model_name}...")
        try:
            model = YOLO(model_name)
            if Path(model_name).exists():
                shutil.move(model_name, str(target_path))
                print(f"[OK]   {model_name} -> {target_path}")
            else:
                print(f"[WARN] {model_name} 下载后未找到文件")
        except Exception as e:
            print(f"[FAIL] {model_name}: {e}")


def download_clip_model():
    """下载 CLIP 模型"""
    print()
    print("=" * 50)
    print("2. 下载 CLIP 模型（可选）")
    print("=" * 50)

    clip_dir = MODELS_DIR / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)

    model_id = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

    try:
        from huggingface_hub import snapshot_download
        print(f"[下载] {model_id}...")
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(clip_dir),
            resume_download=True,
        )
        print(f"[OK]   CLIP 模型下载完成: {path}")
    except ImportError:
        print("[SKIP] huggingface_hub 未安装，尝试 open_clip...")
        try:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="laion2b_s34b_b88k", device="cpu"
            )
            print("[OK]   CLIP 模型通过 open_clip 下载成功")
        except Exception as e:
            print(f"[FAIL] CLIP 模型下载失败: {e}")
            print(f"       手动下载: https://hf-mirror.com/{model_id}")
    except Exception as e:
        print(f"[FAIL] CLIP 模型下载失败: {e}")


def check_silero_vad():
    """检查 Silero VAD 模型"""
    print()
    print("=" * 50)
    print("3. 检查 Silero VAD 模型")
    print("=" * 50)

    try:
        import silero_vad
        print("[OK]   silero-vad 已安装，模型将在首次使用时自动下载")
    except ImportError:
        print("[SKIP] silero-vad 未安装")
        print("       运行 `uv sync` 后重试")


def main():
    print("模型下载工具")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print()

    # 确保模型目录存在
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    download_yolo_models()
    download_clip_model()
    check_silero_vad()

    print()
    print("=" * 50)
    print("下载完成！模型目录内容:")
    print("=" * 50)
    for p in sorted(MODELS_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size / 1024 / 1024
            print(f"  {p.relative_to(MODELS_DIR)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
