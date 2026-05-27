"""测试YOLO检测器"""

import pytest
import numpy as np


def test_yolo_import():
    """测试YOLO导入"""
    from title_classifier.detectors.yolo import YOLODetector
    assert YOLODetector is not None


def test_analyze_pose_for_vlm():
    """测试姿态分析"""
    from title_classifier.detectors.yolo import analyze_pose_for_vlm

    # 测试站立姿态
    keypoints = {
        "left_shoulder": {"x": 100, "y": 200, "conf": 0.9},
        "left_hip": {"x": 100, "y": 300, "conf": 0.9},
    }
    result = analyze_pose_for_vlm(keypoints)
    assert "站立/正常姿态" in result

    # 测试弯腰姿态
    keypoints = {
        "left_shoulder": {"x": 100, "y": 300, "conf": 0.9},
        "left_hip": {"x": 100, "y": 200, "conf": 0.9},
    }
    result = analyze_pose_for_vlm(keypoints)
    assert "弯腰/前倾" in result
