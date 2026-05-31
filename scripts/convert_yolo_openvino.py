#!/usr/bin/env python3
"""
YOLO OpenVINO INT8 量化校准脚本

独立运行此脚本，从视频中抽取帧作为校准数据，生成 INT8 量化的 OpenVINO 模型。
不影响主业务流程。

用法:
    python scripts/convert_yolo_openvino.py --model pose --calibrate-dir test/
    python scripts/convert_yolo_openvino.py --model all --calibrate-dir test/ --num-frames 500
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# YOLO 模型配置
YOLO_MODEL_DIR = project_root / "models" / "yolo"
YOLO_MODELS = {
    "detect": YOLO_MODEL_DIR / "yolov8n.pt",
    "pose": YOLO_MODEL_DIR / "yolov8s-pose.pt",
    "segment": YOLO_MODEL_DIR / "yolov8n-seg.pt",
}
OPENVINO_MODEL_DIR = YOLO_MODEL_DIR / "openvino"


def collect_calibration_frames(calibrate_dir: str, num_frames: int = 500) -> list:
    """
    从视频文件中抽取校准帧
    
    Args:
        calibrate_dir: 校准数据目录（包含视频文件）
        num_frames: 需要抽取的帧数
        
    Returns:
        帧图像路径列表
    """
    import cv2
    import numpy as np
    
    calibrate_path = Path(calibrate_dir)
    if not calibrate_path.exists():
        logger.error(f"校准目录不存在: {calibrate_dir}")
        return []
    
    # 收集所有视频文件
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
    video_files = []
    for ext in video_extensions:
        video_files.extend(calibrate_path.rglob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"未找到视频文件: {calibrate_dir}")
        return []
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 创建临时目录保存校准帧
    tmp_dir = Path(tempfile.mkdtemp(prefix="yolo_calibration_"))
    logger.info(f"校准帧保存目录: {tmp_dir}")
    
    frames_per_video = max(1, num_frames // len(video_files))
    collected_frames = []
    
    for video_path in video_files:
        if len(collected_frames) >= num_frames:
            break
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"无法打开视频: {video_path}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue
            
            # 均匀采样
            frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
            
            for idx in frame_indices:
                if len(collected_frames) >= num_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 保存帧
                    frame_path = tmp_dir / f"calib_{len(collected_frames):05d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    collected_frames.append(str(frame_path))
            
            cap.release()
            logger.info(f"从 {video_path.name} 抽取了 {min(frames_per_video, len(collected_frames))} 帧")
            
        except Exception as e:
            logger.warning(f"处理视频失败 {video_path}: {e}")
            continue
    
    logger.info(f"总共收集了 {len(collected_frames)} 帧校准数据")
    return collected_frames


def create_calibration_dataset(frame_paths: list, output_dir: str) -> str:
    """
    创建校准数据集 YAML 文件（ultralytics 格式）
    
    Args:
        frame_paths: 帧图像路径列表
        output_dir: 输出目录
        
    Returns:
        YAML 文件路径
    """
    import yaml
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 images 目录
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 复制帧到 images 目录
    import shutil
    for i, frame_path in enumerate(frame_paths):
        dest = images_dir / f"calib_{i:05d}.jpg"
        shutil.copy2(frame_path, dest)
    
    # 创建 YAML 配置
    yaml_content = {
        "path": str(output_path.absolute()),
        "train": "images",
        "val": "images",
        "names": {0: "person"},
    }
    
    yaml_path = output_path / "calibration.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    logger.info(f"校准数据集已创建: {yaml_path}")
    return str(yaml_path)


def convert_to_openvino_int8(model_type: str, calibrate_dir: str, num_frames: int = 500):
    """
    将 YOLO 模型转换为 OpenVINO INT8 格式
    
    Args:
        model_type: 模型类型 (detect/pose/segment/all)
        calibrate_dir: 校准数据目录
        num_frames: 校准帧数
    """
    from ultralytics import YOLO
    
    # 收集校准帧
    logger.info(f"开始收集校准帧 (目标: {num_frames} 帧)...")
    frame_paths = collect_calibration_frames(calibrate_dir, num_frames)
    
    if not frame_paths:
        logger.error("未收集到校准帧，退出")
        return
    
    # 创建校准数据集
    calib_dataset_dir = str(YOLO_MODEL_DIR / "calibration_dataset")
    yaml_path = create_calibration_dataset(frame_paths, calib_dataset_dir)
    
    # 确定要转换的模型
    if model_type == "all":
        models_to_convert = list(YOLO_MODELS.keys())
    else:
        if model_type not in YOLO_MODELS:
            logger.error(f"未知的模型类型: {model_type}")
            return
        models_to_convert = [model_type]
    
    # 转换每个模型
    for mt in models_to_convert:
        pt_path = YOLO_MODELS[mt]
        if not pt_path.exists():
            logger.warning(f"模型文件不存在: {pt_path}，跳过")
            continue
        
        output_dir = OPENVINO_MODEL_DIR / f"{mt}_int8_openvino_model"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"转换模型: {mt}")
        logger.info(f"  源模型: {pt_path}")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  校准数据: {yaml_path}")
        logger.info(f"{'='*60}")
        
        try:
            # 加载模型
            model = YOLO(str(pt_path))
            
            # 导出为 OpenVINO INT8 格式
            export_path = model.export(
                format="openvino",
                int8=True,
                data=yaml_path,
            )
            
            # 移动到目标目录
            export_path = Path(export_path)
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            
            if export_path != output_dir:
                import shutil
                shutil.move(str(export_path), str(output_dir))
            
            logger.info(f"✓ {mt} INT8 模型导出成功: {output_dir}")
            
        except Exception as e:
            logger.error(f"✗ {mt} INT8 模型导出失败: {e}")
            logger.info("提示: INT8 量化需要足够的校准数据，尝试增加 --num-frames")
    
    logger.info("\n转换完成!")
    logger.info(f"INT8 模型保存在: {OPENVINO_MODEL_DIR}")
    logger.info("\n使用方法:")
    logger.info("  在 config/default.toml 中设置:")
    logger.info("    [yolo.openvino]")
    logger.info('    precision = "INT8"')


def main():
    parser = argparse.ArgumentParser(
        description="YOLO OpenVINO INT8 量化校准脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 test/ 目录的视频进行校准
  python scripts/convert_yolo_openvino.py --model pose --calibrate-dir test/
  
  # 转换所有模型，使用 500 帧校准
  python scripts/convert_yolo_openvino.py --model all --calibrate-dir test/ --num-frames 500
  
  # 使用指定的校准图片目录
  python scripts/convert_yolo_openvino.py --model pose --calibrate-dir path/to/images/
        """
    )
    
    parser.add_argument(
        "--model", 
        default="pose",
        choices=["detect", "pose", "segment", "all"],
        help="要转换的模型类型 (默认: pose)"
    )
    parser.add_argument(
        "--calibrate-dir",
        required=True,
        help="校准数据目录（包含视频文件或图片）"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=500,
        help="校准帧数 (默认: 500)"
    )
    
    args = parser.parse_args()
    
    logger.info("YOLO OpenVINO INT8 量化校准脚本")
    logger.info(f"模型类型: {args.model}")
    logger.info(f"校准目录: {args.calibrate_dir}")
    logger.info(f"校准帧数: {args.num_frames}")
    
    convert_to_openvino_int8(args.model, args.calibrate_dir, args.num_frames)


if __name__ == "__main__":
    main()
