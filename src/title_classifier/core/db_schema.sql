-- title-classifier 数据库表结构
-- SQLite 3

-- 视频指纹表（用于去重）
CREATE TABLE IF NOT EXISTS video_fingerprints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_size       INTEGER NOT NULL,
    duration        REAL NOT NULL,
    file_hash       TEXT,
    first_seen      TEXT DEFAULT (datetime('now', 'localtime')),
    last_seen       TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(file_size, duration)
);

-- 媒体文件主表
CREATE TABLE IF NOT EXISTS media_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- 原始信息
    original_title  TEXT NOT NULL,
    original_path   TEXT NOT NULL,
    current_path    TEXT,
    file_size       INTEGER,
    duration        REAL,
    resolution      TEXT,
    file_hash       TEXT,
    -- 识别结果
    final_name      TEXT,
    vision_description TEXT,
    vision_keywords TEXT,
    human_detected  INTEGER DEFAULT 0,
    detection_method TEXT,
    -- 状态
    needs_vision    INTEGER DEFAULT 1,
    audio_recognized INTEGER DEFAULT 0,
    review_status   TEXT DEFAULT '待确认',
    -- 关联文件
    srt_path        TEXT,
    -- 去重
    fingerprint_id  INTEGER REFERENCES video_fingerprints(id),
    -- 时间戳
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 标签表
CREATE TABLE IF NOT EXISTS tags (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    category        TEXT,
    created_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 媒体-标签关联表
CREATE TABLE IF NOT EXISTS media_tags (
    media_id        INTEGER REFERENCES media_files(id) ON DELETE CASCADE,
    tag_id          INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    confidence      REAL,
    source          TEXT,
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    PRIMARY KEY (media_id, tag_id)
);

-- 改动记录表
CREATE TABLE IF NOT EXISTS change_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id        INTEGER REFERENCES media_files(id) ON DELETE CASCADE,
    field_name      TEXT NOT NULL,
    old_value       TEXT,
    new_value       TEXT,
    change_source   TEXT,
    changed_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

-- VLM 帧表
CREATE TABLE IF NOT EXISTS vlm_frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id        INTEGER REFERENCES media_files(id) ON DELETE CASCADE,
    frame_index     INTEGER NOT NULL,
    frame_path      TEXT NOT NULL,
    timestamp       REAL,
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(media_id, frame_index)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_media_original_path ON media_files(original_path);
CREATE INDEX IF NOT EXISTS idx_media_current_path ON media_files(current_path);
CREATE INDEX IF NOT EXISTS idx_media_final_name ON media_files(final_name);
CREATE INDEX IF NOT EXISTS idx_media_file_hash ON media_files(file_hash);
CREATE INDEX IF NOT EXISTS idx_media_fingerprint ON media_files(fingerprint_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
CREATE INDEX IF NOT EXISTS idx_media_tags_media ON media_tags(media_id);
CREATE INDEX IF NOT EXISTS idx_media_tags_tag ON media_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_change_log_media ON change_log(media_id);
CREATE INDEX IF NOT EXISTS idx_change_log_time ON change_log(changed_at);
CREATE INDEX IF NOT EXISTS idx_vlm_frames_media ON vlm_frames(media_id);
