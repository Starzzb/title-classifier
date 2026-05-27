"""
YOLO模型效果测试脚本
测试检测和姿态估计功能
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1c_yolo_detector import YOLODetector

def extract_frames(video_path, n_frames=5):
    """从视频中提取多帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息: {total_frames}帧, {fps:.1f}fps, {duration:.1f}秒")
    
    # 均匀采样帧
    frame_indices = [int(total_frames * (i + 1) / (n_frames + 1)) for i in range(n_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    
    cap.release()
    return frames

def test_detection(frames, output_dir):
    """测试人体检测"""
    print("\n=== 测试人体检测 (yolov8n) ===")
    detector = YOLODetector(model_type="detect", conf_threshold=0.3)
    detector.load_model()
    
    for i, (frame_idx, frame) in enumerate(frames):
        result = detector.detect_persons(frame)
        status = "检测到人体" if result["has_person"] else "未检测到人体"
        print(f"帧 {i+1} (索引{frame_idx}): {status}, 数量={result['count']}, 置信度={result['max_confidence']:.2f}")
        
        # 保存结果图片
        output_path = output_dir / f"detect_frame_{i+1}.jpg"
        save_detection_result(frame, result, output_path)

def test_pose(frames, output_dir):
    """测试姿态估计"""
    print("\n=== 测试姿态估计 (yolov8n-pose) ===")
    detector = YOLODetector(model_type="pose", conf_threshold=0.3)
    detector.load_model()
    
    for i, (frame_idx, frame) in enumerate(frames):
        result = detector.estimate_pose(frame)
        status = "检测到人体" if result["has_person"] else "未检测到人体"
        
        if result["has_person"]:
            best_pose = max(result["poses"], key=lambda x: x["avg_confidence"])
            visible = best_pose["visible_count"]
            print(f"帧 {i+1} (索引{frame_idx}): {status}, 关键点={visible}/17")
        else:
            print(f"帧 {i+1} (索引{frame_idx}): {status}")
        
        # 保存结果图片
        output_path = output_dir / f"pose_frame_{i+1}.jpg"
        save_pose_result(frame, result, output_path)

def save_detection_result(frame, result, output_path):
    """保存检测结果图片"""
    img = frame.copy()
    
    if result["has_person"]:
        for person in result["persons"]:
            x1, y1, x2, y2 = [int(x) for x in person["bbox_pixel"]]
            conf = person["confidence"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"person {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)

def save_pose_result(frame, result, output_path):
    """保存姿态估计结果图片"""
    img = frame.copy()
    
    if result["has_person"]:
        for pose in result["poses"]:
            if pose["bbox"]:
                x1, y1, x2, y2 = [int(x) for x in pose["bbox"]["bbox_pixel"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制关键点
            for kpt in pose["keypoints"]:
                if kpt["visible"]:
                    x, y = int(kpt["x"]), int(kpt["y"])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imwrite(str(output_path), img)

def main():
    video_path = Path(__file__).parent / "test_video.mp4"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if not video_path.exists():
        print(f"视频不存在: {video_path}")
        return
    
    print(f"测试视频: {video_path}")
    print(f"输出目录: {output_dir}")
    
    # 提取帧
    frames = extract_frames(video_path, n_frames=3)
    if not frames:
        print("无法提取帧")
        return
    
    print(f"提取了 {len(frames)} 帧")
    
    # 测试检测
    test_detection(frames, output_dir)
    
    # 测试姿态估计
    test_pose(frames, output_dir)
    
    print(f"\n结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
