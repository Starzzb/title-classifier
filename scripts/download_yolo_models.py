"""
使用国内镜像下载YOLO模型到models/yolo目录
"""
import os
import sys
from pathlib import Path

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ULTRALYTICS_HUB"] = "https://hf-mirror.com/ultralytics"

# 模型目录
MODEL_DIR = Path(__file__).parent.parent / "models" / "yolo"

def download_model(model_name):
    """下载指定模型到models/yolo目录"""
    try:
        from ultralytics import YOLO
        
        target_path = MODEL_DIR / model_name
        print(f"正在下载 {model_name} 到 {target_path}...")
        
        # 下载模型
        model = YOLO(model_name)
        
        # 移动到目标目录
        import shutil
        if Path(model_name).exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(model_name, str(target_path))
            print(f"[OK] {model_name} 下载完成 -> {target_path}")
        else:
            print(f"[WARN] {model_name} 下载后未找到文件")
        
        return True
    except Exception as e:
        print(f"[FAIL] {model_name} 下载失败: {e}")
        return False

if __name__ == "__main__":
    models = ["yolov8n.pt", "yolov8n-pose.pt", "yolov8n-seg.pt"]
    
    # 确保目录存在
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查哪些模型已下载
    for model in models:
        target_path = MODEL_DIR / model
        if target_path.exists():
            print(f"[SKIP] {model} 已存在 ({target_path})")
        else:
            download_model(model)
