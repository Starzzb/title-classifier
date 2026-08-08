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


def test_sync_replaced_file_creates_new_record(tmp_path):
    """路径命中但内容被替换 → 同步必须新建记录，旧记录保留。

    模拟真实 bug 场景：磁盘文件在路径上被替换成新内容（同路径、不同 size/hash），
    find_match 不得把新内容误判为旧记录，sync_db 应新建记录。
    """
    import os
    import shutil
    import sqlite3

    from title_classifier.core.scanner import Scanner
    from title_classifier.core.db_store import MediaDB

    # 两个真实小视频：v1 作为"旧记录内容"，v2 作为"替换后的新内容"
    v1 = Path(r"D:\Telegram\shortvideo\[阴部_阴道_手淫_特写_手指_湿润_阴唇_身体局部]_VID_20260403_215930_108.mp4")
    v2 = Path(r"D:\Telegram\shortvideo\[趴卧姿势_灰色破洞背心_露臀_挑逗行为_卧室场景_灰色床单_裸露下体_性暗示动作]_video_2026-04-20_13-34-50.mp4")
    if not v1.exists() or not v2.exists():
        pytest.skip("测试视频文件不存在")

    test_dir = tmp_path / "videos"
    test_dir.mkdir()
    target = test_dir / "目标视频.mp4"

    # 初始：磁盘上是 v1 内容，DB 记录指向该路径
    shutil.copy(v1, target)
    db = MediaDB(db_path=str(tmp_path / "media.db"))
    db.init_schema()
    db.insert_media({
        "original_title": "目标视频.mp4",
        "original_path": str(target),
        "current_path": str(target),
        "file_size": v1.stat().st_size,
        "needs_vision": False,
        "final_name": "[旧内容_关键词]_目标视频",
        "review_status": "已完成",
    })
    db.close()

    # 替换：磁盘上换成 v2 内容（同路径、不同 size）
    shutil.copy(v2, target)

    # 同步：应新建记录，旧记录保留
    sc = Scanner(db_store=MediaDB(db_path=str(tmp_path / "media.db")))
    sc.sync_db(str(test_dir))

    db = MediaDB(db_path=str(tmp_path / "media.db"))
    rows = db.conn.execute(
        "SELECT id, original_title, file_size, needs_vision, review_status FROM media_files WHERE current_path=?",
        (str(target),),
    ).fetchall()
    assert len(rows) == 2, f"替换后应有 2 条记录（旧+新），实际 {len(rows)}"
    new_row = [r for r in rows if r["needs_vision"] == 1]
    old_row = [r for r in rows if r["needs_vision"] == 0]
    assert len(new_row) == 1, "应新建 1 条待识别的新记录"
    assert len(old_row) == 1, "旧记录应保留"
    assert new_row[0]["file_size"] == v2.stat().st_size, "新记录 size 应为磁盘 v2 的 size"
    db.close()
