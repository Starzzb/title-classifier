#!/usr/bin/env python3
"""
转换 PowerShell 导出的 CSV 为程序识别的标准格式

用法:
    python scripts/convert_csv.py data/冰凌紫萱_视频文件.csv
    python scripts/convert_csv.py data/我的资源_视频文件.csv
"""

import csv
import re
import sys
from pathlib import Path

# 广告文件关键词
AD_KEYWORDS = [
    "suu26.com", "suu27.com", "suu28.com", "suu33.com", "suu37.com",
    "tuu93.com", "tuu96.com", "tuu98.com", "tuu32.com",
    "fuu95.com", "2048.vip-直播", "UU直播", "台灣UU", "激情隨時看",
    "社 區 最 新 情 報", "台 妹 子 線 上", "同城少妇美女",
    "N房间的精彩直播", "20年信誉保证",
]

# 标准 CSV 字段
STANDARD_FIELDS = [
    "original_title", "original_path", "needs_vision", "final_name",
    "review_status", "audio_recognized", "srt_path",
    "vision_description", "vision_keywords",
    "human_detected", "detection_confidence", "detection_timestamp", "detection_method",
    "clip_clothing", "clip_action", "clip_hairstyle", "clip_tags", "clip_tags_json", "clip_confidence",
    "vision_source", "vision_failed",
]


def is_ad_file(name: str) -> bool:
    """检查是否是广告文件"""
    for keyword in AD_KEYWORDS:
        if keyword in name:
            return True
    return False


def is_small_file(length: str, threshold: int = 50_000_000) -> bool:
    """检查是否是小文件（小于 50MB，可能是广告或短片）"""
    try:
        return int(length) < threshold
    except ValueError:
        return False


def strip_bracket_prefix(name: str) -> str:
    """去除文件名开头的 [关键词]_ 前缀"""
    return re.sub(r"^\[[^\]]*\]_?", "", name, count=1)


def detect_needs_vision(name: str) -> bool:
    """判断是否需要视觉识别"""
    # 已有中括号分类
    if re.match(r"^\[[^\]]+\]", name):
        return False
    
    # 无意义标题模式
    patterns = [
        r"^(IMG|VID|VIDEO|MOV|MP4|DCIM|P\d+)[_\-\s]?\d+",  # 设备前缀
        r"^[a-f0-9]{8,}$",  # 纯 hex
        r"^\d{8,}$",  # 纯数字
        r"^mp4_\s*\(\d+\)",  # mp4_ (1) 格式
    ]
    
    for pattern in patterns:
        if re.match(pattern, name, re.IGNORECASE):
            return True
    
    # 短标题
    clean = re.sub(r'[\(\)\[\]【】（）\d\-_.\s]+', '', Path(name).stem)
    if len(clean) <= 2:
        return True
    
    return False


def convert_csv(input_path: str, output_path: str = None):
    """转换 CSV 文件"""
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_标准格式.csv"
    else:
        output_path = Path(output_path)
    
    print(f"读取: {input_path}")
    
    # 读取原始 CSV
    rows = []
    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"原始记录数: {len(rows)}")
    
    # 转换
    converted = []
    skipped_ad = 0
    skipped_small = 0
    skipped_dup = 0
    
    seen_paths = set()
    
    for row in rows:
        full_name = row.get("FullName", "")
        name = row.get("Name", "")
        dir_name = row.get("DirectoryName", "")
        length = row.get("Length", "")
        
        # 跳过广告文件
        if is_ad_file(name) or is_ad_file(full_name):
            skipped_ad += 1
            continue
        
        # 跳过小于 10MB 的文件（广告/短片）
        if is_small_file(length, 10_000_000):
            skipped_small += 1
            continue
        
        # Z: 改成 X:
        full_name = full_name.replace("Z:\\", "X:\\")
        dir_name = dir_name.replace("Z:\\", "X:\\")
        
        # 去重
        if full_name in seen_paths:
            skipped_dup += 1
            continue
        seen_paths.add(full_name)
        
        # 提取标题（去除扩展名）
        original_title = Path(name).stem
        
        # 判断是否需要视觉识别
        needs_vision = detect_needs_vision(name)
        
        # 构建标准行
        std_row = {
            "original_title": original_title,
            "original_path": full_name,
            "needs_vision": str(needs_vision).lower(),
            "final_name": "",  # 待填写
            "review_status": "待确认",
            "audio_recognized": "false",
            "srt_path": "",
            "vision_description": "",
            "vision_keywords": "",
            "human_detected": "",
            "detection_confidence": "",
            "detection_timestamp": "",
            "detection_method": "",
            "clip_clothing": "",
            "clip_action": "",
            "clip_hairstyle": "",
            "clip_tags": "",
            "clip_tags_json": "",
            "clip_confidence": "",
            "vision_source": "",
            "vision_failed": "false",
        }
        
        converted.append(std_row)
    
    # 写入标准 CSV
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_FIELDS)
        writer.writeheader()
        writer.writerows(converted)
    
    print(f"\n转换完成:")
    print(f"  输出文件: {output_path}")
    print(f"  有效记录: {len(converted)}")
    print(f"  跳过广告: {skipped_ad}")
    print(f"  跳过小文件: {skipped_small}")
    print(f"  跳过重复: {skipped_dup}")
    print(f"  需要视觉: {sum(1 for r in converted if r['needs_vision'] == 'true')}")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/convert_csv.py <input.csv> [output.csv]")
        print("示例: python scripts/convert_csv.py data/冰凌紫萱_视频文件.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_csv(input_path, output_path)


if __name__ == "__main__":
    main()
