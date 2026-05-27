"""视频全面分析实战测试"""

import sys
import os
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 加载.env文件
def load_env(env_path: Path):
    """加载.env文件"""
    if not env_path.exists():
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip())

# 加载环境变量
load_env(Path(__file__).parent.parent / ".env")

from title_classifier.core.vision import VisionProcessor
from title_classifier.utils.video import get_video_duration


def test_comprehensive_analysis():
    """测试视频全面分析"""
    video_path = Path(__file__).parent / "test_video.mp4"

    if not video_path.exists():
        print(f"[错误] 视频不存在: {video_path}")
        return

    # 获取视频信息
    duration = get_video_duration(str(video_path))
    print(f"=" * 60)
    print(f"视频全面分析实战测试")
    print(f"=" * 60)
    print(f"视频: {video_path.name}")
    print(f"时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    print(f"大小: {video_path.stat().st_size / 1024 / 1024:.1f}MB")
    print()

    # 测试配置
    configs = [
        {"name": "标准模式", "step": 2.0, "frames": 10},
        {"name": "高密度模式", "step": 1.0, "frames": 15},
    ]

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"测试配置: {config['name']}")
        print(f"  采样间隔: {config['step']}秒")
        print(f"  VLM帧数: {config['frames']}")
        print(f"{'=' * 60}")

        # 创建处理器
        processor = VisionProcessor(
            provider="gcli",
            use_yolo=True,
            yolo_model="pose",
            yolo_conf=0.3,
            max_image_size=800,
            vlm_frames=config["frames"],
            analysis_step=config["step"],
        )

        # 初始化
        print("\n[1/3] 初始化模型...")
        start_time = time.time()
        if not processor.initialize():
            print("[错误] 初始化失败")
            continue
        init_time = time.time() - start_time
        print(f"  初始化完成: {init_time:.1f}秒")

        # 分析视频
        print("\n[2/3] 分析视频...")
        start_time = time.time()
        result = processor.process_video(str(video_path), "test_video")
        analysis_time = time.time() - start_time

        # 输出结果
        print(f"\n[3/3] 分析结果:")
        print(f"  分析耗时: {analysis_time:.1f}秒")

        if "error" in result:
            print(f"  错误: {result['error']}")
            continue

        # 视频摘要
        summary = result.get("video_summary", {})
        print(f"\n  【视频摘要】")
        print(f"  - 人体检测: {'是' if summary.get('has_person') else '否'}")
        print(f"  - 人体比例: {summary.get('person_ratio', 0)*100:.1f}%")
        print(f"  - 主要姿态: {summary.get('main_pose', '未知')}")
        print(f"  - 姿态变化: {result.get('pose_changes', 0)}次")
        print(f"  - 分析帧数: {result.get('total_analyzed', 0)}")
        print(f"  - 选中帧数: {result.get('selected_frames', 0)}")

        # 姿态分布
        pose_dist = summary.get("pose_distribution", {})
        if pose_dist:
            print(f"\n  【姿态分布】")
            for pose, count in sorted(pose_dist.items(), key=lambda x: -x[1]):
                print(f"  - {pose}: {count}次")

        # VLM结果
        print(f"\n  【VLM分析结果】")
        desc = result.get('description', '')
        kw = result.get('keywords', '')
        if desc:
            print(f"  描述: {desc[:200]}...")
        else:
            print(f"  描述: [空]")
        if kw:
            print(f"  关键词: {kw}")
        else:
            print(f"  关键词: [空]")
        
        # 检查是否有错误
        if not desc and not kw:
            print(f"\n  [警告] VLM返回为空，可能原因:")
            print(f"    1. API Key未配置")
            print(f"    2. API调用失败")
            print(f"    3. 响应解析失败")

    print(f"\n{'=' * 60}")
    print(f"测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_comprehensive_analysis()
