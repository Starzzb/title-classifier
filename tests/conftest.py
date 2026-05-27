"""测试配置"""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_video(project_root):
    """示例视频"""
    video = project_root / "test" / "test_video.mp4"
    if video.exists():
        return video
    pytest.skip("测试视频不存在")


@pytest.fixture
def tmp_output(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
