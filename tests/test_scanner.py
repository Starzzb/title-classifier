"""测试扫描器"""

import pytest
from pathlib import Path


def test_scanner_import():
    """测试扫描器导入"""
    from title_classifier.core.scanner import Scanner
    assert Scanner is not None


def test_has_chinese():
    """测试中文检测"""
    from title_classifier.core.scanner import has_chinese
    assert has_chinese("测试") == True
    assert has_chinese("test") == False
    assert has_chinese("测试test") == True


def test_is_already_classified():
    """测试已分类检测"""
    from title_classifier.core.scanner import is_already_classified
    assert is_already_classified("[分类]文件名") == True
    assert is_already_classified("[未分类]文件名") == False
    assert is_already_classified("文件名") == False


def test_is_needs_vision():
    """测试视觉识别需求检测"""
    from title_classifier.core.scanner import is_needs_vision
    assert is_needs_vision("IMG_7940") == True
    assert is_needs_vision("20240115") == True
    assert is_needs_vision("测试视频") == False
