"""
CLIP 模型手动下载脚本
下载到项目 models/clip/ 目录
"""
import os
import sys
from pathlib import Path

# 模型缓存目录：项目内 models/clip/
CACHE_DIR = Path(__file__).parent.parent / "models" / "clip"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 设置 HF 缓存到项目目录
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"

def main():
    print(f"模型缓存目录: {CACHE_DIR}")
    print(f"模型: {MODEL_ID}")
    print(f"HF 镜像: {os.environ['HF_ENDPOINT']}")
    print()

    # 方式1：使用 huggingface_hub（推荐）
    try:
        from huggingface_hub import snapshot_download
        print("使用 huggingface_hub 下载...")
        path = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            resume_download=True,
        )
        print(f"下载成功: {path}")
        return
    except ImportError:
        print("huggingface_hub 未安装，尝试方式2...")
    except Exception as e:
        print(f"huggingface_hub 下载失败: {e}")
        print("尝试方式2...")

    # 方式2：使用 open_clip 自动下载
    print()
    print("使用 open_clip 自动下载...")
    try:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device="cpu"
        )
        print("open_clip 下载成功！模型已缓存。")
    except Exception as e:
        print(f"open_clip 下载失败: {e}")
        print()
        print("请手动下载模型文件：")
        print(f"  1. 访问 https://hf-mirror.com/{MODEL_ID}")
        print(f"  2. 下载 open_clip_pytorch_model.bin")
        print(f"  3. 放到 {CACHE_DIR} 对应目录下")

    print()
    print(f"缓存目录内容:")
    for p in sorted(CACHE_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size / 1024 / 1024
            print(f"  {p.relative_to(CACHE_DIR)} ({size:.1f} MB)")

if __name__ == "__main__":
    main()
