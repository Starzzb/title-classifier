"""
从字幕文件补全数据库中的描述和关键词。

用法:
    python scripts/backfill_from_srt.py [--dry-run] [--db data/media.db] [--output-dir data/output]

匹配策略:
    SRT文件名格式: [kw1_kw2_...]_original_title_without_ext.srt
    数据库字段: original_title (带扩展名)
    匹配: 去掉SRT文件名中的 [..]_ 前缀和 .srt 后缀, 与数据库 original_title 去扩展名做前缀匹配
"""

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path


def parse_srt_filename(filename: str):
    """从SRT文件名解析出 final_name 和 original_title (不含扩展名)"""
    stem = Path(filename).stem  # 去掉 .srt

    # 匹配 [final_name]_original_title 格式
    m = re.match(r'^(\[.+?\])_(.+)$', stem)
    if m:
        final_name = m.group(1)
        original_title = m.group(2)
        return final_name, original_title

    # 没有 [..]_ 前缀, 整个 stem 就是 original_title
    return None, stem


def parse_srt_content(content: str):
    """从SRT内容中提取描述和关键词"""
    description = ""
    keywords = ""

    desc_m = re.search(r'【视频描述】(.+?)(?=【|\Z)', content, re.DOTALL)
    if desc_m:
        description = desc_m.group(1).strip()

    kw_m = re.search(r'【关键词】(.+?)(?=【|\Z)', content, re.DOTALL)
    if kw_m:
        keywords = kw_m.group(1).strip()

    return description, keywords


def find_srt_files(output_dir: str):
    """遍历所有 subtitles 子目录, 收集 SRT 文件"""
    output_path = Path(output_dir)
    srt_files = []
    for srt_dir in output_path.glob("*/subtitles"):
        for srt_file in srt_dir.glob("*.srt"):
            srt_files.append(srt_file)
    return srt_files


def build_db_index(db_path: str):
    """读取数据库, 建立 original_title_stem -> row 的索引"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, original_title, vision_description, vision_keywords, final_name, srt_path "
        "FROM media_files"
    ).fetchall()

    index = {}
    for row in rows:
        stem = Path(row["original_title"]).stem if "." in row["original_title"] else row["original_title"]
        # 用 stem 做 key, 如果有重复取最新的 (id 最大的)
        if stem not in index or row["id"] > index[stem]["id"]:
            index[stem] = {
                "id": row["id"],
                "original_title": row["original_title"],
                "stem": stem,
                "has_desc": bool(row["vision_description"]),
                "has_kw": bool(row["vision_keywords"]),
                "has_final": bool(row["final_name"]),
                "has_srt": bool(row["srt_path"]),
            }

    conn.close()
    return index


def match_srt_to_db(srt_title: str, db_index: dict):
    """匹配SRT文件名中的title到数据库条目"""
    # 精确匹配
    if srt_title in db_index:
        return db_index[srt_title]

    # 去掉末尾可能的空格/特殊字符后匹配
    srt_clean = srt_title.rstrip(" .")
    for stem, info in db_index.items():
        if stem.startswith(srt_clean) or srt_clean.startswith(stem):
            return info

    # 模糊匹配: SRT title 包含在 DB stem 中, 或反之
    for stem, info in db_index.items():
        if srt_title in stem or stem in srt_title:
            return info

    return None


def backfill(db_path: str, output_dir: str, dry_run: bool = False):
    """执行补全"""
    print(f"数据库: {db_path}")
    print(f"字幕目录: {output_dir}")
    print(f"模式: {'DRY RUN' if dry_run else '实际写入'}")
    print()

    # 1. 加载数据库索引
    db_index = build_db_index(db_path)
    print(f"数据库条目: {len(db_index)}")

    # 2. 收集所有 SRT 文件
    srt_files = find_srt_files(output_dir)
    print(f"SRT文件: {len(srt_files)}")

    # 3. 匹配并补全
    conn = sqlite3.connect(db_path)
    matched = 0
    updated_desc = 0
    updated_kw = 0
    updated_final = 0
    updated_srt = 0
    skipped_has_data = 0
    unmatched = []

    for srt_file in srt_files:
        final_name, srt_title = parse_srt_filename(srt_file.name)
        if not srt_title:
            continue

        # 匹配数据库
        db_row = match_srt_to_db(srt_title, db_index)
        if not db_row:
            unmatched.append(srt_file.name)
            continue

        matched += 1

        # 解析 SRT 内容
        try:
            content = srt_file.read_text(encoding="utf-8")
        except Exception:
            content = srt_file.read_text(encoding="utf-8-sig")

        description, keywords = parse_srt_content(content)

        # 跳过已有完整数据的条目
        if db_row["has_desc"] and db_row["has_kw"] and db_row["has_final"]:
            skipped_has_data += 1
            continue

        media_id = db_row["id"]
        updates = []

        if description and not db_row["has_desc"]:
            if not dry_run:
                conn.execute(
                    "UPDATE media_files SET vision_description = ?, updated_at = datetime('now','localtime') WHERE id = ?",
                    (description, media_id),
                )
            updates.append(f"  描述: {description[:60]}...")
            updated_desc += 1

        if keywords and not db_row["has_kw"]:
            if not dry_run:
                conn.execute(
                    "UPDATE media_files SET vision_keywords = ?, updated_at = datetime('now','localtime') WHERE id = ?",
                    (keywords, media_id),
                )
            updates.append(f"  关键词: {keywords[:60]}")
            updated_kw += 1

        if final_name and not db_row["has_final"]:
            if not dry_run:
                conn.execute(
                    "UPDATE media_files SET final_name = ?, updated_at = datetime('now','localtime') WHERE id = ?",
                    (final_name, media_id),
                )
            updates.append(f"  final_name: {final_name[:60]}")
            updated_final += 1

        if not db_row["has_srt"]:
            srt_path = str(srt_file)
            if not dry_run:
                conn.execute(
                    "UPDATE media_files SET srt_path = ?, updated_at = datetime('now','localtime') WHERE id = ?",
                    (srt_path, media_id),
                )
            updates.append(f"  srt_path: {srt_path}")
            updated_srt += 1

        if not dry_run:
            # 有描述后标记 needs_vision=0
            conn.execute(
                "UPDATE media_files SET needs_vision = 0, updated_at = datetime('now','localtime') WHERE id = ? AND needs_vision = 1",
                (media_id,),
            )

        if updates:
            print(f"[匹配] id={media_id} | {db_row['original_title'][:50]}")
            for u in updates:
                print(u)

    if not dry_run:
        conn.commit()
    conn.close()

    # 4. 统计
    print()
    print("=" * 50)
    print(f"匹配成功: {matched}")
    print(f"跳过(已有数据): {skipped_has_data}")
    print(f"补全描述: {updated_desc}")
    print(f"补全关键词: {updated_kw}")
    print(f"补全final_name: {updated_final}")
    print(f"补全srt_path: {updated_srt}")
    print(f"未匹配: {len(unmatched)}")

    if unmatched:
        print()
        print("未匹配的SRT文件 (前20个):")
        for name in unmatched[:20]:
            print(f"  {name[:80]}")


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="从字幕文件补全数据库描述和关键词")
    parser.add_argument("--db", default="data/media.db", help="数据库路径")
    parser.add_argument("--output-dir", default="data/output", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写入")
    args = parser.parse_args()

    # 相对于项目根目录
    project_root = Path(__file__).parent.parent
    db_path = project_root / args.db
    output_dir = project_root / args.output_dir

    if not db_path.exists():
        print(f"数据库不存在: {db_path}")
        sys.exit(1)

    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        sys.exit(1)

    backfill(str(db_path), str(output_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
