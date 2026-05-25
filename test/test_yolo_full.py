"""
YOLO视频分析测试脚本
支持整个视频分析，输出详细姿态数据
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1c_yolo_detector import YOLODetector

# 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def analyze_video(video_path, step_seconds=1.0, conf_threshold=0.3):
    """
    分析整个视频
    
    Args:
        video_path: 视频路径
        step_seconds: 每隔多少秒分析一帧
        conf_threshold: 置信度阈值
    
    Returns:
        分析结果字典
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    # 视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频: {video_path}")
    print(f"时长: {duration:.1f}秒, 帧数: {total_frames}, FPS: {fps:.1f}")
    print(f"分析间隔: {step_seconds}秒")
    print("=" * 60)
    
    # 初始化检测器
    detector = YOLODetector(model_type="pose", conf_threshold=conf_threshold)
    detector.load_model()
    
    # 分析帧
    step_frames = int(fps * step_seconds)
    results = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔分析
        if frame_idx % step_frames == 0:
            timestamp = frame_idx / fps
            
            # 姿态估计
            pose_result = detector.estimate_pose(frame)
            
            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "has_person": pose_result["has_person"],
                "person_count": len(pose_result["poses"]) if pose_result["has_person"] else 0
            }
            
            if pose_result["has_person"]:
                # 取最高置信度的人体
                best_pose = max(pose_result["poses"], key=lambda x: x["avg_confidence"])
                
                # 关键点数据
                keypoints_data = {}
                for kpt in best_pose["keypoints"]:
                    if kpt["visible"]:
                        keypoints_data[kpt["name"]] = {
                            "x": round(kpt["x"], 1),
                            "y": round(kpt["y"], 1),
                            "conf": round(kpt["confidence"], 3)
                        }
                
                frame_data["keypoints"] = keypoints_data
                frame_data["visible_count"] = best_pose["visible_count"]
                frame_data["avg_confidence"] = round(best_pose["avg_confidence"], 3)
                
                # 分析姿态
                frame_data["pose_analysis"] = analyze_pose(keypoints_data)
            
            results.append(frame_data)
            
            # 进度显示
            if len(results) % 10 == 0:
                print(f"已分析 {len(results)} 帧 ({timestamp:.1f}s)")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"分析完成，共 {len(results)} 帧")
    return {
        "video_info": {
            "path": str(video_path),
            "duration": round(duration, 2),
            "total_frames": total_frames,
            "fps": fps,
            "step_seconds": step_seconds
        },
        "frames": results,
        "summary": generate_summary(results, duration)
    }

def analyze_pose(keypoints):
    """分析姿态"""
    analysis = []
    
    # 检测动作
    if "left_shoulder" in keypoints and "left_hip" in keypoints:
        shoulder_y = keypoints["left_shoulder"]["y"]
        hip_y = keypoints["left_hip"]["y"]
        if shoulder_y > hip_y:
            analysis.append("弯腰/前倾")
    
    if "left_knee" in keypoints and "left_hip" in keypoints:
        knee_y = keypoints["left_knee"]["y"]
        hip_y = keypoints["left_hip"]["y"]
        if knee_y < hip_y + 50:  # 膝盖高于臀部
            analysis.append("跪姿/蹲姿")
    
    if "left_ankle" in keypoints and "left_knee" in keypoints:
        ankle_y = keypoints["left_ankle"]["y"]
        knee_y = keypoints["left_knee"]["y"]
        if abs(ankle_y - knee_y) < 30:
            analysis.append("坐姿")
    
    # 检测朝向
    if "left_ear" in keypoints and "right_ear" in keypoints:
        left_x = keypoints["left_ear"]["x"]
        right_x = keypoints["right_ear"]["x"]
        if left_x > right_x + 20:
            analysis.append("右侧朝向")
        elif right_x > left_x + 20:
            analysis.append("左侧朝向")
    
    if not analysis:
        analysis.append("站立/正常姿态")
    
    return analysis

def generate_summary(results, duration):
    """生成分析摘要"""
    frames_with_person = [f for f in results if f["has_person"]]
    person_ratio = len(frames_with_person) / len(results) if results else 0
    
    summary = {
        "total_frames_analyzed": len(results),
        "frames_with_person": len(frames_with_person),
        "person_presence_ratio": round(person_ratio * 100, 1),
        "person_detected": person_ratio > 0.1  # 超过10%帧有人体则认为有人
    }
    
    # 统计姿态
    pose_counts = {}
    for frame in frames_with_person:
        if "pose_analysis" in frame:
            for pose in frame["pose_analysis"]:
                pose_counts[pose] = pose_counts.get(pose, 0) + 1
    
    if pose_counts:
        summary["pose_statistics"] = pose_counts
        summary["dominant_pose"] = max(pose_counts, key=pose_counts.get)
    
    # 人体出现时间段
    if frames_with_person:
        first_appear = frames_with_person[0]["timestamp"]
        last_appear = frames_with_person[-1]["timestamp"]
        summary["first_appearance"] = first_appear
        summary["last_appearance"] = last_appear
    
    return summary

def save_results(output_path, results):
    """保存结果到JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")

def print_summary(summary):
    """打印摘要"""
    print("\n" + "=" * 60)
    print("分析摘要")
    print("=" * 60)
    print(f"分析帧数: {summary['total_frames_analyzed']}")
    print(f"检测到人体的帧数: {summary['frames_with_person']}")
    print(f"人体出现比例: {summary['person_presence_ratio']}%")
    print(f"是否有人: {'是' if summary['person_detected'] else '否'}")
    
    if "pose_statistics" in summary:
        print(f"\n姿态统计:")
        for pose, count in summary["pose_statistics"].items():
            print(f"  - {pose}: {count}次")
        print(f"主要姿态: {summary.get('dominant_pose', '未知')}")
    
    if "first_appearance" in summary:
        print(f"\n人体出现时间段:")
        print(f"  首次出现: {summary['first_appearance']}秒")
        print(f"  最后出现: {summary['last_appearance']}秒")

def main():
    video_path = Path(__file__).parent / "test_video.mp4"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if not video_path.exists():
        print(f"视频不存在: {video_path}")
        return
    
    # 分析整个视频，每5秒一帧
    results = analyze_video(video_path, step_seconds=5, conf_threshold=0.3)
    
    if results:
        # 打印摘要
        print_summary(results["summary"])
        
        # 保存结果
        json_path = output_dir / "analysis_result.json"
        save_results(json_path, results)
        
        # 打印前5帧详细数据
        print("\n" + "=" * 60)
        print("前5帧详细数据:")
        print("=" * 60)
        for frame in results["frames"][:5]:
            print(f"\n帧 {frame['frame_idx']} ({frame['timestamp']}秒):")
            print(f"  有人体: {frame['has_person']}")
            if frame["has_person"]:
                print(f"  可见关键点: {frame['visible_count']}/17")
                print(f"  平均置信度: {frame['avg_confidence']}")
                print(f"  姿态分析: {frame.get('pose_analysis', [])}")
                if "keypoints" in frame:
                    print(f"  关键点坐标:")
                    for name, pos in list(frame["keypoints"].items())[:5]:
                        print(f"    {name}: ({pos['x']}, {pos['y']}) conf={pos['conf']}")

if __name__ == "__main__":
    main()
