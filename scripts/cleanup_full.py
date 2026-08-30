#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""media.db 与磁盘双向对齐(以磁盘为准)。

流程:
  1. 从 DB 路径推导遍历根目录,遍历磁盘视频文件建立全集
  2. 重复 current_path 的记录组只保留信息最全的一条
  3. 路径存在的记录 → 有效
  4. 路径不存在的记录 → 按 (file_size, 文件名stem) 在未被认领的磁盘文件中找唯一匹配 → 更新 current_path(搬迁)
  5. 仍无匹配的记录 → 硬删除(全部字段先写入报告 JSON 留底)
  6. 未被任何记录认领的磁盘文件 → 插入最小记录(不跑视觉描述)
  7. 复查:每条记录路径存在、每个磁盘文件有记录

用法:
  python cleanup_full.py            # dry-run,只统计不写库
  python cleanup_full.py --execute  # 实际执行(先自动备份)
"""
import argparse
import collections
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT / 'data' / 'media.db'
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.wmv', '.mov', '.flv', '.webm',
              '.m4v', '.ts', '.rmvb', '.rm', '.mpg', '.mpeg', '.3gp', '.m2ts'}
TS = time.strftime('%Y%m%d_%H%M%S')


def log(msg):
    print(msg, flush=True)


def norm(p):
    return os.path.normcase(os.path.normpath(p))


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


def derive_roots(rows_paths):
    """从所有 DB 路径推导 盘符+一级目录 根;直接位于盘根的文件按散文件处理。"""
    roots = {}   # norm_root -> display_root
    loose = {}   # norm_path -> display_path
    for p in rows_paths:
        if not p:
            continue
        p = p.replace('/', '\\')
        drive, rest = os.path.splitdrive(p)
        if not drive:
            continue
        parts = [x for x in rest.split('\\') if x]
        if len(parts) <= 1:
            full = drive + ('\\' + parts[0] if parts else '\\')
            n = norm(full)
            if os.path.isfile(full):
                loose.setdefault(n, full)
            continue
        root = drive + '\\' + parts[0]
        roots.setdefault(norm(root), root)
    return roots, loose


def walk_universe(roots, loose):
    """遍历根目录收集视频文件。返回 files_by_path, files_by_key, skipped_roots。"""
    files_by_path = {}
    files_by_key = {}
    skipped = []
    for nroot, root in sorted(roots.items()):
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in VIDEO_EXTS:
                        continue
                    fp = os.path.join(dirpath, fn)
                    try:
                        size = os.stat(fp).st_size
                    except OSError:
                        continue
                    n = norm(fp)
                    if n not in files_by_path:
                        files_by_path[n] = size
        elif os.path.isfile(root):
            size = os.stat(root).st_size
            files_by_path[norm(root)] = size
        else:
            skipped.append(root)
    for npath, path in loose.items():
        size = os.stat(path).st_size
        files_by_path.setdefault(npath, size)
    for npath, size in files_by_path.items():
        files_by_key.setdefault((size, stem_of(npath)), []).append(npath)
    return files_by_path, files_by_key, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true', help='实际写库(默认 dry-run)')
    args = ap.parse_args()
    mode = 'EXECUTE' if args.execute else 'DRY-RUN'
    log(f'=== media.db 对齐清理 [{mode}] {TS} ===')
    log(f'DB: {DB_PATH}')
    if not DB_PATH.exists():
        sys.exit('数据库不存在: ' + str(DB_PATH))

    report = {'mode': mode, 'timestamp': TS, 'db': str(DB_PATH)}

    con = sqlite3.connect(str(DB_PATH), timeout=30)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    total = cur.execute('SELECT COUNT(*) FROM media_files').fetchone()[0]
    rows = cur.execute(
        'SELECT id, original_title, original_path, current_path, file_size, final_name, '
        'vision_description, vision_keywords, fingerprint_id, review_status, srt_path '
        'FROM media_files').fetchall()
    log(f'DB 记录总数: {total}')

    # ---------- Phase 1: 磁盘全集 ----------
    all_paths = set()
    for r in rows:
        if r['current_path']:
            all_paths.add(r['current_path'])
        if r['original_path']:
            all_paths.add(r['original_path'])
    roots, loose = derive_roots(all_paths)
    log(f'推导遍历根: {len(roots)} 个目录根 + {len(loose)} 个盘根散文件')
    files_by_path, files_by_key, skipped = walk_universe(roots, loose)
    log(f'磁盘视频文件全集: {len(files_by_path)} 个')
    if skipped:
        log(f'跳过的不存在根: {skipped}')
    report['skipped_roots'] = skipped
    report['disk_file_count'] = len(files_by_path)

    # ---------- Phase 2: 重复 current_path 去重 ----------
    groups = {}
    for r in rows:
        if r['current_path']:
            groups.setdefault(norm(r['current_path']), []).append(r)
    dup_delete_ids = set()
    dup_removed = []
    for npath, grp in groups.items():
        if len(grp) < 2:
            continue
        actual_size = files_by_path.get(npath)
        def rank(r):
            return (
                1 if (actual_size is not None and r['file_size'] == actual_size) else 0,
                1 if r['vision_description'] else 0,
                1 if r['fingerprint_id'] else 0,
                -r['id'],
            )
        grp_sorted = sorted(grp, key=rank, reverse=True)
        for r in grp_sorted[1:]:
            dup_delete_ids.add(r['id'])
            dup_removed.append(dict(r))
    log(f'重复路径记录组去重: 删除 {len(dup_delete_ids)} 条冗余记录')
    report['duplicate_removed'] = dup_removed

    # ---------- Phase 3: 路径存在 → 有效 ----------
    claimed = set()
    valid, missing = [], []
    for r in rows:
        if r['id'] in dup_delete_ids or not r['current_path']:
            continue
        n = norm(r['current_path'])
        if n in files_by_path:
            valid.append(r)
            claimed.add(n)
        else:
            missing.append(r)
    log(f'路径有效: {len(valid)} 条; 路径不存在待处理: {len(missing)} 条')

    # ---------- Phase 4: 搬迁匹配 (size, stem) ----------
    avail_by_key = {}
    for n, size in files_by_path.items():
        if n not in claimed:
            avail_by_key.setdefault((size, stem_of(n)), []).append(n)
    for k in avail_by_key:
        avail_by_key[k].sort()

    relocations, no_size_key = [], 0
    missing_still = []
    for r in sorted(missing, key=lambda x: x['id']):
        size = r['file_size']
        if size is None:
            no_size_key += 1
            missing_still.append(r)
            continue
        # 优先用 current_path 的 stem,退化用 final_name / original_path 的 stem
        keys = []
        base = r['current_path'] or ''
        for cand in [base, r['final_name'] or '', r['original_path'] or '']:
            s = stem_of(cand) if cand else ''
            k = (size, s)
            if s and k not in keys:
                keys.append(k)
        hit = None
        for k in keys:
            cands = avail_by_key.get(k)
            if cands:
                hit = (k, cands[0])
                break
        if hit:
            k, newpath = hit
            avail_by_key[k].pop(0)
            claimed.add(newpath)
            relocations.append({'id': r['id'], 'old': r['current_path'],
                                'new': newpath, 'row': dict(r)})
        else:
            missing_still.append(r)
    log(f'搬迁匹配: {len(relocations)} 条 (file_size 为空无法匹配: {no_size_key} 条)')

    # ---------- Phase 4b: 仅凭大小的唯一匹配(疑似改名) ----------
    # 仅当该大小在"剩余孤儿记录"与"剩余未认领文件"中都恰好唯一时才配对,避免歧义。
    avail_by_size = {}
    for (size, _stem), lst in avail_by_key.items():
        for n in lst:
            avail_by_size.setdefault(size, []).append(n)
    orphan_size_count = collections.Counter(
        r['file_size'] for r in missing_still if r['file_size'] is not None)
    size_relocated = []
    still_missing = []
    for r in sorted(missing_still, key=lambda x: x['id']):
        size = r['file_size']
        cands = avail_by_size.get(size, []) if size is not None else []
        if size is not None and orphan_size_count.get(size, 0) == 1 and len(cands) == 1:
            newpath = cands[0]
            avail_by_size[size] = []
            claimed.add(newpath)
            size_relocated.append({'id': r['id'], 'old': r['current_path'],
                                   'new': os.path.normpath(newpath), 'size': size})
        else:
            still_missing.append(r)
    relocations.extend(size_relocated)
    log(f'大小唯一匹配(疑似改名): {len(size_relocated)} 条')
    report['relocations'] = [{k: v for k, v in x.items() if k != 'row'} for x in relocations]

    # ---------- Phase 5: 孤儿删除 ----------
    orphans = still_missing
    log(f'孤儿记录(磁盘无对应文件, 将删除): {len(orphans)} 条')
    report['orphans_deleted'] = [dict(r) for r in orphans]

    # ---------- Phase 6: 未认领文件 → 新增最小记录 ----------
    unclaimed = [n for n in files_by_path if n not in claimed]
    unclaimed.sort()
    log(f'磁盘有文件但无记录(将新增最小记录): {len(unclaimed)} 个')
    report['inserted'] = [{'path': os.path.normpath(n), 'size': files_by_path[n]}
                          for n in unclaimed]

    # ---------- 统计汇总 ----------
    summary = {
        'db_before': total,
        'disk_files': len(files_by_path),
        'valid': len(valid),
        'relocated': len(relocations),
        'duplicates_removed': len(dup_delete_ids),
        'orphans_deleted': len(orphans),
        'inserted': len(unclaimed),
        'db_after_expected': len(valid) + len(relocations) + len(unclaimed),
    }
    log('--- 统计 ---')
    for k, v in summary.items():
        log(f'  {k}: {v}')
    report['summary'] = summary

    rep_path = PROJECT / 'data' / f'align_report_{"exec" if args.execute else "dryrun"}_{TS}.json'
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    log(f'报告已写入: {rep_path}')

    # ---------- Phase 7: 写库 ----------
    if not args.execute:
        log('[dry-run] 未做任何改动。确认后运行: python cleanup_full.py --execute')
        con.close()
        return

    backup = PROJECT / 'data' / f'media.db.bak_align_{TS}'
    con.close()
    shutil.copy2(DB_PATH, backup)
    if os.path.exists(str(DB_PATH) + '-wal'):
        shutil.copy2(str(DB_PATH) + '-wal', str(backup) + '-wal')
    log(f'已备份: {backup} ({backup.stat().st_size} bytes)')

    con = sqlite3.connect(str(DB_PATH), timeout=30)
    cur = con.cursor()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        for rid in dup_delete_ids:
            cur.execute('DELETE FROM media_files WHERE id=?', (rid,))
        for x in relocations:
            cur.execute('UPDATE media_files SET current_path=?, updated_at=? WHERE id=?',
                        (os.path.normpath(x['new']), now, x['id']))
        for r in orphans:
            cur.execute('DELETE FROM media_files WHERE id=?', (r['id'],))
        for n in unclaimed:
            path = os.path.normpath(n)
            size = files_by_path[n]
            title = os.path.basename(path)
            cur.execute(
                'INSERT INTO media_files (original_title, final_name, original_path, current_path, '
                'file_size, created_at, updated_at) VALUES (?,?,?,?,?,?,?)',
                (title, stem_of(path), path, path, size, now, now))
        con.commit()
    except Exception:
        con.rollback()
        con.close()
        raise

    # ---------- Phase 8: 复查 ----------
    log('--- 复查验证 ---')
    bad_records = cur.execute(
        "SELECT COUNT(*) FROM media_files WHERE current_path IS NULL OR current_path=''").fetchone()[0]
    missing_on_disk = 0
    db_paths = set()
    for (p,) in cur.execute("SELECT current_path FROM media_files WHERE current_path IS NOT NULL"):
        db_paths.add(norm(p))
        if not os.path.isfile(p):
            missing_on_disk += 1
    no_record_files = [n for n in files_by_path if n not in db_paths]
    total_after = cur.execute('SELECT COUNT(*) FROM media_files').fetchone()[0]
    log(f'  记录总数(改后): {total_after}')
    log(f'  路径不存在的记录: {missing_on_disk} (应为 0)')
    log(f'  无记录的磁盘文件: {len(no_record_files)} (应为 0)')
    log(f'  空路径记录: {bad_records} (应为 0)')
    ok = missing_on_disk == 0 and not no_record_files and bad_records == 0
    report['verification'] = {'total_after': total_after, 'missing_on_disk': missing_on_disk,
                              'no_record_files': len(no_record_files),
                              'empty_path_records': bad_records, 'ok': ok}
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    con.close()
    log('=== 完成: 验证通过 ===' if ok else '=== 完成但验证未通过, 请检查报告 ===')
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
