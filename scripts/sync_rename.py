"""磁盘文件名与数据库 final_name 双向统一脚本（dry_run 模式）

逻辑：
  遍历 D:\\Telegram 和 F:\\ 下所有存在文件的记录，
  比较磁盘 stem 与 DB final_name：
  - 磁盘未规范化 且 DB 已规范化（[关键词]_ 前缀）→ 磁盘改为 DB final_name
  - 磁盘已规范化 且 DB 未规范化 → DB final_name 同步为磁盘 stem（保留磁盘规范化名）
  - 两者都规范化但不同 → 以 DB final_name 为准改磁盘（DB 是视觉识别正式结果）
  - 两者都未规范化但不同 → 跳过并记录

用法：python sync_rename.py [--dry-run|--apply]
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DRIVES = ["D:\\Telegram", "F:\\"]

def is_classified(name: str) -> bool:
    return bool(name) and name.startswith("[")

def main():
    mode = "apply"
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        mode = "dry-run"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root, "data", "media.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {"rename_disk_to_db": 0, "sync_db_to_disk": 0, "skip": 0, "error": 0}
    detail = []

    for drive in DRIVES:
        rows = conn.execute(
            """SELECT id, final_name, current_path, original_path, review_status
               FROM media_files
               WHERE current_path LIKE ? AND review_status NOT IN ('文件已移走','文件损毁已删除')""",
            (drive + "%",),
        ).fetchall()

        for r in rows:
            p = r["current_path"] or r["original_path"]
            if not p or not os.path.exists(p):
                continue
            disk_stem = Path(p).stem
            final_name = r["final_name"] or ""
            if disk_stem == final_name:
                continue

            disk_std = is_classified(disk_stem)
            db_std = is_classified(final_name)

            if not disk_std and db_std:
                # 磁盘未规范化、DB 已规范化 → 磁盘改为 DB final_name
                new_path = os.path.join(os.path.dirname(p), final_name + Path(p).suffix)
                action = "rename_disk_to_db"
            elif disk_std and not db_std:
                # 磁盘已规范化、DB 未规范化 → DB 同步为磁盘 stem
                action = "sync_db_to_disk"
                new_path = None
            elif disk_std and db_std:
                # 都规范化但不同 → 以 DB 为准改磁盘
                new_path = os.path.join(os.path.dirname(p), final_name + Path(p).suffix)
                action = "rename_disk_to_db"
            else:
                stats["skip"] += 1
                detail.append(f"SKIP  id={r['id']} 磁盘={disk_stem[:35]} final={final_name[:35]}")
                continue

            detail.append(
                f"{action.upper()} id={r['id']} :: {Path(p).name[:45]} -> {final_name[:45]}"
            )
            stats[action] += 1

            if mode == "apply" and action == "rename_disk_to_db":
                try:
                    os.rename(p, new_path)
                    conn.execute(
                        "UPDATE media_files SET current_path=?, updated_at=datetime('now','localtime') WHERE id=?",
                        (new_path, r["id"]),
                    )
                    conn.commit()
                except Exception as e:
                    stats["error"] += 1
                    detail.append(f"ERROR id={r['id']} {e}")
            elif mode == "apply" and action == "sync_db_to_disk":
                try:
                    conn.execute(
                        "UPDATE media_files SET final_name=?, updated_at=datetime('now','localtime') WHERE id=?",
                        (disk_stem, r["id"]),
                    )
                    conn.commit()
                except Exception as e:
                    stats["error"] += 1
                    detail.append(f"ERROR id={r['id']} {e}")

    print(f"模式: {mode}")
    print(f"  磁盘改为 DB final_name: {stats['rename_disk_to_db']}")
    print(f"  DB final_name 同步为磁盘名: {stats['sync_db_to_disk']}")
    print(f"  跳过: {stats['skip']}")
    print(f"  错误: {stats['error']}")
    print(f"\n明细（前 20 条）:")
    for d in detail[:20]:
        print(f"  {d}")
    if len(detail) > 20:
        print(f"  ... 共 {len(detail)} 条操作")

    conn.close()

if __name__ == "__main__":
    main()
