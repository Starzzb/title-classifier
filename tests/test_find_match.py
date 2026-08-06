"""find_match 指纹匹配级联测试"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from title_classifier.core.db_store import MediaDB


def _make_db(tmp_path) -> MediaDB:
    """创建临时数据库"""
    db = MediaDB(db_path=str(tmp_path / "test.db"))
    db.init_schema()
    return db


def test_l1_path_match(tmp_path):
    db = _make_db(tmp_path)
    db.insert_media({
        "original_title": "video1.mp4",
        "original_path": "D:\\test\\video1.mp4",
        "current_path": "D:\\test\\video1.mp4",
        "file_size": 1000,
        "duration": 10.0,
    })
    m = db.find_match(original_title="any.mp4", path="D:\\test\\video1.mp4")
    assert m is not None and m["original_title"] == "video1.mp4"
    db.close()


def test_l2_coarse_fingerprint(tmp_path):
    """size+duration 唯一候选 → 无需哈希即可追踪"""
    db = _make_db(tmp_path)
    fp_id = db.get_fingerprint_id(1000, 10.0, "hash_abc")
    db.insert_media({
        "original_title": "video1.mp4",
        "original_path": "D:\\test\\video1.mp4",
        "current_path": "D:\\test\\video1.mp4",
        "file_size": 1000,
        "duration": 10.0,
        "fingerprint_id": fp_id,
    })
    # 移动/重命名：路径不同、标题不同，但 size+duration 相同
    m = db.find_match(original_title="renamed.mp4", file_size=1000, duration=10.4)
    assert m is not None and m["original_title"] == "video1.mp4"
    db.close()


def test_l3_hash_fingerprint(tmp_path):
    """哈希精确匹配 → 内容真实相同，可追踪（即使 size+duration 有歧义）"""
    db = _make_db(tmp_path)
    fp_id = db.get_fingerprint_id(1000, 10.0, "hash_abc")
    db.insert_media({
        "original_title": "video1.mp4",
        "original_path": "D:\\test\\video1.mp4",
        "current_path": "D:\\test\\video1.mp4",
        "file_size": 1000,
        "duration": 10.0,
        "fingerprint_id": fp_id,
    })
    m = db.find_match(original_title="whatever.mp4", file_size=1000, duration=10.0, file_hash="hash_abc")
    assert m is not None and m["original_title"] == "video1.mp4"
    db.close()


def test_l4_title_fallback(tmp_path):
    """无指纹时：剥离 [前缀]_ 标题匹配"""
    db = _make_db(tmp_path)
    db.insert_media({
        "original_title": "video1.mp4",
        "original_path": "D:\\test\\old\\video1.mp4",
        "current_path": "D:\\test\\old\\video1.mp4",
        "file_size": 1000,
        "duration": 10.0,
    })
    # 重命名后文件已移走：旧路径不存在
    m = db.find_match(original_title="[kw1_kw2]video1.mp4", file_size=1000, duration=10.0)
    assert m is not None and m["original_title"] == "video1.mp4"
    db.close()


def test_size_duration_collision_disambiguated_by_hash(tmp_path):
    """两个不同文件同 size+duration → L2 不误匹配，L3 哈希区分"""
    db = _make_db(tmp_path)
    fp_a = db.get_fingerprint_id(1000, 10.0, "hash_A")
    fp_b = db.get_fingerprint_id(1000, 10.0, "hash_B")
    db.insert_media({
        "original_title": "fileA.mp4",
        "original_path": "D:\\test\\A.mp4",
        "current_path": "D:\\test\\A.mp4",
        "file_size": 1000,
        "duration": 10.0,
        "fingerprint_id": fp_a,
    })
    db.insert_media({
        "original_title": "fileB.mp4",
        "original_path": "D:\\test\\B.mp4",
        "current_path": "D:\\test\\B.mp4",
        "file_size": 1000,
        "duration": 10.0,
        "fingerprint_id": fp_b,
    })
    # L2 不应匹配（多个候选）
    m2 = db.find_match(original_title="new.mp4", file_size=1000, duration=10.0)
    assert m2 is None
    # L3 哈希精确区分
    ma = db.find_match(original_title="new.mp4", file_size=1000, duration=10.0, file_hash="hash_A")
    mb = db.find_match(original_title="new.mp4", file_size=1000, duration=10.0, file_hash="hash_B")
    assert ma is not None and ma["original_title"] == "fileA.mp4"
    assert mb is not None and mb["original_title"] == "fileB.mp4"
    db.close()


def test_no_match_returns_none(tmp_path):
    db = _make_db(tmp_path)
    m = db.find_match(original_title="ghost.mp4", file_size=999, duration=9.0)
    assert m is None
    db.close()
