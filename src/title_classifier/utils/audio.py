"""音频处理工具 - 支持VAD语音检测和字幕后处理"""

import base64
import subprocess
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts"}

# 默认配置
DEFAULT_CONFIG = {
    "skip_silence": True,
    "volume_threshold": 0.01,
    "segment_duration": 30.0,
    "adaptive_enabled": True,
    "min_segment": 15,
    "max_segment": 60,
    "low_energy_merge_threshold": 30,
    # 增强型自适应分段配置
    "enter_threshold": 0.15,       # 双门限：进入语音阈值（相对噪声底的偏移）
    "exit_threshold": 0.08,        # 双门限：退出语音阈值（相对噪声底的偏移）
    "hangover_frames": 5,          # 迟滞帧数：能量掉下阈值后维持 N 帧才确认切出
    "noise_alpha": 0.95,           # 噪声底平滑系数（0.9~0.99，越大越保守）
    "zcr_noise_threshold": 0.4,    # ZCR 噪音阈值
    "centroid_min_hz": 200,        # 频谱重心下限（Hz），低于此为噪音
    "centroid_max_hz": 4000,       # 频谱重心上限（Hz），高于此为噪音
    "entropy_threshold": 0.8,      # 频谱熵阈值，高于此为噪音
    "vad_enabled": True,           # 是否使用Silero VAD
    "vad_min_speech_ms": 150,      # VAD最小时长（毫秒）
    "vad_min_silence_ms": 80,      # VAD最小静音时长（毫秒）
    "merge_gap": 0.8,              # 第一层：微合并间隙阈值（秒）
    "min_keep_duration": 1.0,      # 第一层：最小保留时长（秒）
    "max_chunk": 25.0,             # 第二层：语义打包最大块时长（秒）
    "long_gap": 2.0,               # 第二层：长停顿阈值（秒），超过则强制封口
    "min_duration": 1.0,           # 第三层：最小块时长（秒）
    "min_speech_ratio": 0.4,       # 第三层：最小语音占比（0-1）
    # 字幕后处理配置
    "postprocess_enabled": True,   # 是否启用字幕后处理
    "max_subtitle_duration": 10,   # 单个字幕最大时长（秒）
    "max_subtitle_chars": 100,     # 单个字幕最大字符数
    "filter_invalid": True,        # 是否过滤无效内容
    "format_text": True,           # 是否格式化文本
}


class SileroVADHelper:
    """Silero VAD 辅助类 - 语音活动检测"""
    _model = None

    @classmethod
    def get_model(cls):
        """获取VAD模型（懒加载）"""
        if cls._model is None:
            try:
                from silero_vad import load_silero_vad
                cls._model = load_silero_vad()
                logger.info("Silero VAD模型加载成功")
            except Exception as e:
                logger.error(f"Silero VAD模型加载失败: {e}")
                return None
        return cls._model

    @staticmethod
    def detect_speech_segments(
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        检测语音段（毫秒级精度）

        Args:
            audio_data: float32 numpy数组
            sample_rate: 采样率（默认16kHz）
            min_speech_ms: 最小时长（毫秒），低于此时长的语音段会被忽略
            min_silence_ms: 最小静音时长（毫秒），用于合并相邻语音段

        Returns:
            [(start_sec, end_sec), ...]
        """
        try:
            import torch
            from silero_vad import get_speech_timestamps

            model = SileroVADHelper.get_model()
            if model is None:
                return []

            # numpy -> torch tensor（确保数组可写）
            audio_data = np.array(audio_data, dtype=np.float32)
            wav = torch.from_numpy(audio_data).float()

            # 获取语音时间戳
            timestamps = get_speech_timestamps(
                wav,
                model,
                sampling_rate=sample_rate,
                return_seconds=True,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
            )

            return [(t['start'], t['end']) for t in timestamps]

        except Exception as e:
            logger.error(f"VAD检测失败: {e}")
            return []

    @staticmethod
    def detect_speech_segments_from_file(
        audio_path: str,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        从音频文件检测语音段

        Args:
            audio_path: 音频文件路径（WAV格式）
            min_speech_ms: 最小时长（毫秒）
            min_silence_ms: 最小静音时长（毫秒）

        Returns:
            [(start_sec, end_sec), ...]
        """
        try:
            from silero_vad import read_audio

            wav = read_audio(audio_path, sampling_rate=16000)
            audio_data = wav.numpy()

            return SileroVADHelper.detect_speech_segments(
                audio_data, 16000, min_speech_ms, min_silence_ms
            )

        except Exception as e:
            logger.error(f"从文件检测语音段失败: {e}")
            return []


def load_audio_config() -> dict:
    """加载音频配置"""
    config = DEFAULT_CONFIG.copy()
    
    try:
        import tomllib
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.toml"
        
        if config_path.exists():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
                
            audio_config = data.get("audio", {})
            config["skip_silence"] = audio_config.get("skip_silence", config["skip_silence"])
            config["volume_threshold"] = audio_config.get("volume_threshold", config["volume_threshold"])
            config["segment_duration"] = audio_config.get("segment_duration", config["segment_duration"])
            
            adaptive_config = audio_config.get("adaptive", {})
            config["adaptive_enabled"] = adaptive_config.get("enabled", config["adaptive_enabled"])
            config["min_segment"] = adaptive_config.get("min_segment", config["min_segment"])
            config["max_segment"] = adaptive_config.get("max_segment", config["max_segment"])
            config["low_energy_merge_threshold"] = adaptive_config.get("low_energy_merge_threshold", config["low_energy_merge_threshold"])
            # 增强型自适应配置
            config["enter_threshold"] = adaptive_config.get("enter_threshold", config["enter_threshold"])
            config["exit_threshold"] = adaptive_config.get("exit_threshold", config["exit_threshold"])
            config["hangover_frames"] = adaptive_config.get("hangover_frames", config["hangover_frames"])
            config["noise_alpha"] = adaptive_config.get("noise_alpha", config["noise_alpha"])
            config["zcr_noise_threshold"] = adaptive_config.get("zcr_noise_threshold", config["zcr_noise_threshold"])
            config["centroid_min_hz"] = adaptive_config.get("centroid_min_hz", config["centroid_min_hz"])
            config["centroid_max_hz"] = adaptive_config.get("centroid_max_hz", config["centroid_max_hz"])
            config["entropy_threshold"] = adaptive_config.get("entropy_threshold", config["entropy_threshold"])

            # VAD配置
            vad_config = audio_config.get("vad", {})
            config["vad_enabled"] = vad_config.get("enabled", config.get("vad_enabled", True))
            config["vad_min_speech_ms"] = vad_config.get("min_speech_ms", config.get("vad_min_speech_ms", 150))
            config["vad_min_silence_ms"] = vad_config.get("min_silence_ms", config.get("vad_min_silence_ms", 80))
            config["merge_gap"] = vad_config.get("merge_gap", config.get("merge_gap", 0.8))
            config["min_keep_duration"] = vad_config.get("min_keep_duration", config.get("min_keep_duration", 1.0))
            config["max_chunk"] = vad_config.get("max_chunk", config.get("max_chunk", 25.0))
            config["long_gap"] = vad_config.get("long_gap", config.get("long_gap", 2.0))
            config["min_duration"] = vad_config.get("min_duration", config.get("min_duration", 1.0))
            config["min_speech_ratio"] = vad_config.get("min_speech_ratio", config.get("min_speech_ratio", 0.4))

            # 字幕后处理配置
            postprocess_config = audio_config.get("postprocess", {})
            config["postprocess_enabled"] = postprocess_config.get("enabled", config.get("postprocess_enabled", True))
            config["max_subtitle_duration"] = postprocess_config.get("max_subtitle_duration", config.get("max_subtitle_duration", 10))
            config["max_subtitle_chars"] = postprocess_config.get("max_subtitle_chars", config.get("max_subtitle_chars", 100))
            config["filter_invalid"] = postprocess_config.get("filter_invalid", config.get("filter_invalid", True))
            config["format_text"] = postprocess_config.get("format_text", config.get("format_text", True))
            
    except Exception as e:
        logger.warning(f"加载音频配置失败，使用默认配置: {e}")
    
    return config


def is_video_file(file_path: str) -> bool:
    """判断是否为视频文件"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio(
    video_path: str,
    output_path: str,
    start_time: float = None,
    end_time: float = None,
) -> bool:
    """从视频提取音频"""
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path]

        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])

        cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        logger.error(f"提取音频失败: {e}")
        return False


def extract_audio_to_numpy(video_path: str) -> Optional[np.ndarray]:
    """
    一次性提取完整音频到内存（优化版本）
    
    Args:
        video_path: 视频路径
    
    Returns:
        numpy数组，音频数据（float32格式，16kHz采样率）
    """
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # 不包含视频
            "-f", "f32le",  # 32位浮点输出
            "-acodec", "pcm_f32le",
            "-ar", "16000",  # 16kHz采样率
            "-ac", "1",  # 单声道
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"提取音频失败: {result.stderr.decode('utf-8', errors='replace')}")
            return None
        
        # 将字节转换为浮点数
        audio_data = np.frombuffer(result.stdout, dtype=np.float32)
        
        if len(audio_data) == 0:
            logger.error("提取的音频数据为空")
            return None
        
        logger.info(f"音频提取完成: {len(audio_data)} 采样点, {len(audio_data)/16000:.1f}秒")
        return audio_data
        
    except Exception as e:
        logger.error(f"提取音频到内存失败: {e}")
        return None


def get_audio_base64(audio_path: str) -> str:
    """获取音频的base64编码"""
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logger.error(f"读取音频失败: {e}")
        return ""


def calculate_rms_from_numpy(audio_data: np.ndarray, start_sample: int, end_sample: int) -> float:
    """
    从numpy数组计算RMS能量
    
    Args:
        audio_data: 音频数据数组
        start_sample: 起始采样点
        end_sample: 结束采样点
    
    Returns:
        RMS能量值（0-1之间）
    """
    try:
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return 0.0
        
        # 提取片段
        segment = audio_data[start_sample:end_sample]
        
        if len(segment) == 0:
            return 0.0
        
        # 计算RMS能量
        rms = np.sqrt(np.mean(segment ** 2))
        
        # 归一化到0-1范围
        return min(float(rms), 1.0)
        
    except Exception as e:
        logger.error(f"计算RMS失败: {e}")
        return 0.0


def calculate_audio_rms(audio_path: str) -> float:
    """
    计算音频文件的RMS能量（均方根）
    
    Args:
        audio_path: 音频文件路径（WAV格式）
    
    Returns:
        RMS能量值（0-1之间）
    """
    try:
        # 使用ffmpeg读取音频数据
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-f", "f32le",  # 32位浮点输出
            "-acodec", "pcm_f32le",
            "-ar", "16000",
            "-ac", "1",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            return 0.0
        
        # 将字节转换为浮点数
        audio_data = np.frombuffer(result.stdout, dtype=np.float32)
        
        if len(audio_data) == 0:
            return 0.0
        
        # 计算RMS能量
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # 归一化到0-1范围（16位音频最大值约为1.0）
        return min(float(rms), 1.0)
        
    except Exception as e:
        logger.error(f"计算音频RMS失败: {e}")
        return 0.0


# ==================== 信号特征提取 ====================


def _compute_zcr(frame: np.ndarray) -> float:
    """
    计算过零率（Zero Crossing Rate）

    语音浊音（元音）波形缓慢 → ZCR 低
    清音（s, f）较高
    纯噪音/静音通常稳定在中间值

    Returns:
        过零率 (0.0 ~ 1.0)
    """
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(zero_crossings / (len(frame) - 1))


def _compute_spectral_centroid(frame: np.ndarray, sample_rate: int = 16000) -> float:
    """
    计算频谱重心（Spectral Centroid）— 单位 Hz

    人声主要集中在 300–3400Hz。
    能量高但重心在低频（<<200Hz）或极高频 → 可能是风声、电流声

    Returns:
        频谱重心 Hz
    """
    if len(frame) < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)

    total_magnitude = np.sum(spectrum)
    if total_magnitude < 1e-10:
        return 0.0

    centroid = np.sum(freqs * spectrum) / total_magnitude
    return float(centroid)


def _compute_spectral_entropy(frame: np.ndarray) -> float:
    """
    计算频谱熵（Spectral Entropy）— 0~1

    语音是谐波结构 → 频谱熵低
    噪音更接近白噪 → 熵高

    Returns:
        频谱熵 (0.0 ~ 1.0)
    """
    if len(frame) < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(frame))
    total = np.sum(spectrum)
    if total < 1e-10:
        return 0.0

    # 归一化为概率分布
    psd = spectrum / total
    psd = psd[psd > 1e-10]  # 去零，避免 log(0)

    # Shannon 熵，归一化到 0~1
    entropy = -np.sum(psd * np.log2(psd))
    max_entropy = np.log2(len(psd)) if len(psd) > 0 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _analyze_frame_features(
    audio_data: np.ndarray,
    start_sample: int,
    end_sample: int,
    sample_rate: int = 16000,
) -> dict:
    """
    分析单帧的全部信号特征

    Returns:
        {"rms": float, "zcr": float, "centroid": float, "entropy": float}
    """
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)

    if start_sample >= end_sample:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "entropy": 0.0}

    frame = audio_data[start_sample:end_sample]
    if len(frame) == 0:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "entropy": 0.0}

    return {
        "rms": float(min(np.sqrt(np.mean(frame ** 2)), 1.0)),
        "zcr": _compute_zcr(frame),
        "centroid": _compute_spectral_centroid(frame, sample_rate),
        "entropy": _compute_spectral_entropy(frame),
    }


def _classify_frame(features: dict, noise_floor: float, config: dict) -> str:
    """
    多特征帧分类：speech / noise / silence

    1. 绝对静音（rms < noise_floor * 1.5）
    2. 高能量 + 高熵 + 高ZCR → 噪音
    3. 高能量 + 频谱重心不在人声范围 → 噪音
    4. 高于 enter_threshold → 语音
    5. 其余 → 静音
    """
    rms = features["rms"]
    zcr = features["zcr"]
    centroid = features["centroid"]
    entropy = features["entropy"]

    # 绝对静音
    if rms < noise_floor * 1.5:
        return "silence"

    # 噪音过滤：高能量 + 高熵 + 高 ZCR
    entropy_threshold = config.get("entropy_threshold", 0.8)
    zcr_noise_threshold = config.get("zcr_noise_threshold", 0.4)
    if rms > config.get("enter_threshold", 0.15) and entropy > entropy_threshold and zcr > zcr_noise_threshold:
        return "noise"

    # 噪音过滤：频谱重心不在人声范围
    centroid_min = config.get("centroid_min_hz", 200)
    centroid_max = config.get("centroid_max_hz", 4000)
    if rms > config.get("enter_threshold", 0.15) and (centroid < centroid_min or centroid > centroid_max):
        return "noise"

    # 语音判断（使用 enter_threshold）
    if rms >= config.get("enter_threshold", 0.15):
        return "speech"

    return "silence"


def _build_frame_feature_map(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: int = 25,
) -> List[Tuple[float, dict]]:
    """
    构建帧级特征图（25ms 帧，用于精细分析）

    Returns:
        [(time_sec, {rms, zcr, centroid, entropy}), ...]
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    total_samples = len(audio_data)
    feature_map = []

    pos = 0
    while pos < total_samples:
        end = min(pos + frame_size, total_samples)
        features = _analyze_frame_features(audio_data, pos, end, sample_rate)
        time_sec = pos / sample_rate
        feature_map.append((time_sec, features))
        pos = end

    return feature_map


def enhanced_adaptive_segment(
    audio_data: np.ndarray,
    duration: float,
    config: dict,
    sample_rate: int = 16000,
) -> List[Tuple[float, float, str]]:
    """
    增强型自适应分段（帧级特征版）— 使用 ZCR + 频谱特征 + 双门限

    流程：
    1. 构建 25ms 帧级特征图（RMS, ZCR, 频谱重心, 频谱熵）
    2. 自适应噪声底估计
    3. 逐帧分类（speech / noise / silence）
    4. 双门限 + 迟滞状态机
    5. 合并为分段

    Args:
        audio_data: float32 音频数据
        duration: 音频时长（秒）
        config: 配置参数
        sample_rate: 采样率

    Returns:
        [(start_time, end_time, segment_type), ...]
    """
    if len(audio_data) == 0:
        return []

    min_segment = config["min_segment"]
    max_segment = config["max_segment"]
    enter_threshold = config.get("enter_threshold", 0.15)
    exit_threshold = config.get("exit_threshold", 0.08)
    hangover_frames = config.get("hangover_frames", 5)
    noise_alpha = config.get("noise_alpha", 0.95)
    low_energy_merge_threshold = config["low_energy_merge_threshold"]

    # 1. 构建帧级特征图
    feature_map = _build_frame_feature_map(audio_data, sample_rate, frame_ms=25)
    if not feature_map:
        return []

    # 2. 自适应噪声底（从 0 开始，让算法自行学习）
    noise_floor = 0.0

    # 3. 逐帧分类 + 双门限状态机
    in_speech = False
    hangover_counter = 0

    # 帧级分类结果 [(time_sec, label)]
    frame_labels = []
    for time_sec, features in feature_map:
        # 更新噪声底（非语音段）
        if not in_speech:
            noise_floor = noise_alpha * noise_floor + (1 - noise_alpha) * features["rms"]

        # 多特征分类
        label = _classify_frame(features, noise_floor, config)

        # 双门限状态机（基于 RMS）
        dynamic_enter = noise_floor + enter_threshold
        dynamic_exit = noise_floor + exit_threshold

        if in_speech:
            if features["rms"] < dynamic_exit and label != "speech":
                hangover_counter += 1
                if hangover_counter >= hangover_frames:
                    in_speech = False
                    hangover_counter = 0
            else:
                hangover_counter = 0
        else:
            if features["rms"] >= dynamic_enter or label == "speech":
                in_speech = True
                hangover_counter = 0

        frame_labels.append((time_sec, "speech" if in_speech else "silence"))

    # 4. 合并帧级标签为分段
    if not frame_labels:
        return []

    frame_ms = 25 / 1000.0
    segments = []
    seg_start = frame_labels[0][0]
    seg_type = frame_labels[0][1]

    for i in range(1, len(frame_labels)):
        t, label = frame_labels[i]
        if label != seg_type:
            seg_end = t
            if seg_end - seg_start >= min_segment:
                segments.append((seg_start, seg_end, seg_type))
            seg_start = t
            seg_type = label

    # 最后一段
    seg_end = min(frame_labels[-1][0] + frame_ms, duration)
    if seg_end - seg_start >= min_segment:
        segments.append((seg_start, seg_end, seg_type))

    # 超长段切分
    final_segments = []
    for seg in segments:
        s_start, s_end, s_type = seg
        while s_end - s_start > max_segment:
            final_segments.append((s_start, s_start + max_segment, s_type))
            s_start += max_segment
        if s_end - s_start > 0:
            final_segments.append((s_start, s_end, s_type))

    # 合并连续静音段
    merged = []
    for seg in final_segments:
        if merged and seg[2] == "silence" and merged[-1][2] == "silence":
            prev = merged[-1]
            if seg[1] - prev[0] <= low_energy_merge_threshold:
                merged[-1] = (prev[0], seg[1], "silence")
            else:
                merged.append(seg)
        else:
            merged.append(seg)

    return merged


def clean_transcription_text(text: str) -> str:
    """
    清理转录文本，移除非中文的标注内容和模型推理文本
    
    移除格式如：(female), (male), (whisper), (laughter),
              （female）, （male）, （whisper）, （laughter）等
    同时过滤模型推理/思维链输出（如 "首先，用户指令是..." 等）
    """
    import re
    # 移除英文括号及其内容（如 (female), (whisper), (laughter)）
    text = re.sub(r'\([a-zA-Z][a-zA-Z_ ]*\)', '', text)
    # 移除中文全角括号及其内容（如 （female）, （whisper）, （laughter））
    text = re.sub(r'（[a-zA-Z][a-zA-Z_ ]*）', '', text)
    # 清理多余空格
    text = re.sub(r' +', ' ', text)

    # 过滤模型推理/思维链输出：在遇到推理标记时截断
    reasoning_markers = [
        r'首先[，,]', r'首先，用户', r'用户指令', r'用户的指令',
        r'根据规则', r'规则[：:]', r'我需要', r'我需要处理',
        r'你是一名', r'这是一个技术', r'这是一个技术音频',
        r'系统提示[：:]', r'系统提示是',
        r'Transcribe ALL speech', r'You are a speech recognition',
        r'RULES[：:]',
        r'1\. 输出[：:]', r'1\. Output',
    ]
    for marker in reasoning_markers:
        idx = re.search(marker, text)
        if idx and idx.start() > 10:  # 确保前面有足够内容
            text = text[:idx.start()]
            break

    # 如果文本仍包含完整的系统提示（被过滤了整个文本），取引号内的内容
    if text.startswith('"') and '"' in text[1:]:
        match = re.match(r'^"([^"]+)"', text)
        if match:
            text = match.group(1)

    # 如果文本过长且包含中文标点/思考特征，提取前半部分
    if len(text) > 400:
        # 尝试在第一个句号/感叹号处截断
        first_end = re.search(r'[。！？\n]', text)
        if first_end:
            short = text[:first_end.start()].strip()
            if len(short) > 3:
                text = short

    return text.strip()


def generate_srt_from_segments(segments: list, output_path: str) -> bool:
    """从分段生成SRT字幕文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = format_time(seg["start"])
                end = format_time(seg["end"])
                text = seg["text"]
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        return True
    except Exception as e:
        logger.error(f"生成SRT失败: {e}")
        return False


def format_time(seconds: float) -> str:
    """将秒数转换为SRT时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class AudioProcessor:
    """音频处理器 - 支持VAD分段和字幕后处理"""

    def __init__(
        self,
        provider: str = "mimo",
        config: dict = None,
        debug_callback=None,
        **kwargs,
    ):
        self.provider = provider
        self.debug_callback = debug_callback  # 调试信息回调函数
        
        # 加载配置
        if config is None:
            config = load_audio_config()
        
        self.config = config
        
        # 允许通过kwargs覆盖配置
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        # 调试信息收集
        self.debug_info = {
            "vad_segments": [],
            "merged_segments": [],
            "api_calls": [],
        }

    def process_video(
        self,
        video_path: str,
        output_srt: str = None,
        segment_duration: float = None,
        detect_only: bool = False,
    ) -> Optional[str]:
        """
        处理视频，生成字幕
        
        Args:
            video_path: 视频路径
            output_srt: 输出SRT路径
            segment_duration: 固定分段时长（秒），仅在自适应关闭时使用
            detect_only: 仅检测VAD分段，不调用语音识别API（调试用）
        
        Returns:
            SRT文件路径（detect_only模式下返回None）
        """
        from ..providers import call_audio_api

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return None

        # 获取视频时长
        duration = self._get_duration(str(video_path))
        if duration <= 0:
            logger.error("无法获取视频时长")
            return None

        logger.info(f"视频时长: {duration:.1f}秒")
        logger.info(f"配置: 跳过静音={self.config['skip_silence']}, 音量阈值={self.config['volume_threshold']}")
        
        if detect_only:
            logger.info("=" * 50)
            logger.info("[调试模式] 仅检测VAD分段，不调用语音识别API")
            logger.info("=" * 50)

        # 确定分段策略
        vad_on = self.config.get("vad_enabled", False)

        if vad_on:
            logger.info("使用VAD语音活动检测进行分段")
            segments_to_process = self._get_vad_segments(str(video_path), duration)
            # 收集VAD调试信息
            self.debug_info["vad_segments"] = segments_to_process
            if self.debug_callback:
                self.debug_callback("vad_segments", segments_to_process)
                self.debug_callback("vad_params", {
                    "min_speech_ms": self.config.get("vad_min_speech_ms", 150),
                    "min_silence_ms": self.config.get("vad_min_silence_ms", 80),
                    "sample_rate": 16000,
                })
        else:
            seg_dur = segment_duration or self.config["segment_duration"]
            logger.info(f"使用固定分段模式，每段{seg_dur}秒")
            segments_to_process = self._get_fixed_segments(duration, seg_dur)

        # 收集分段调试信息
        self.debug_info["merged_segments"] = segments_to_process
        if self.debug_callback:
            self.debug_callback("segments", segments_to_process)

        # 如果是仅检测模式，到这里就停止
        if detect_only:
            logger.info("=" * 50)
            logger.info("[调试模式] VAD检测完成，共检测到以下语音段：")
            logger.info("-" * 50)
            for i, seg in enumerate(segments_to_process, 1):
                start, end, seg_type = seg
                duration_seg = end - start
                logger.info(f"  {i}. [{seg_type}] {start:.3f}s - {end:.3f}s ({duration_seg:.3f}秒)")
            logger.info("-" * 50)
            logger.info(f"总计: {len(segments_to_process)} 个语音段")
            logger.info("=" * 50)
            logger.info("[调试模式] 终止：未调用语音识别API")
            return None

        # 处理每个分段
        segments = []
        total = len(segments_to_process)
        skipped = 0
        processed = 0
        failed = 0
        skip_reasons = {"silence": 0, "short": 0, "empty": 0}
        fail_reasons = {"extract": 0, "base64": 0, "api": 0, "rejected": 0}

        # 统计段类型
        speech_segments = [(s, e, t) for s, e, t in segments_to_process if t == "speech"]
        silence_segments = [(s, e, t) for s, e, t in segments_to_process if t == "silence"]
        logger.info(f"分段统计: 共{total}段，语音段={len(speech_segments)}，静音段={len(silence_segments)}")

        for i, (start_time, end_time, segment_type) in enumerate(segments_to_process, 1):
            duration_seg = end_time - start_time
            
            # 跳过静音段
            if self.config["skip_silence"] and segment_type == "silence":
                logger.info(f"  [{i}/{total}] 跳过静音段: {start_time:.1f}s-{end_time:.1f}s ({duration_seg:.1f}秒)")
                skipped += 1
                skip_reasons["silence"] += 1
                continue

            # 跳过太短的段
            if duration_seg < 1.0:
                logger.info(f"  [{i}/{total}] 跳过过短段: {start_time:.1f}s-{end_time:.1f}s ({duration_seg:.1f}秒 < 1秒)")
                skipped += 1
                skip_reasons["short"] += 1
                continue

            logger.info(f"  [{i}/{total}] 处理区块: {start_time:.2f}s-{end_time:.2f}s ({duration_seg:.2f}秒)")

            # 提取音频
            audio_path = video_path.parent / f"{video_path.stem}_audio_{start_time:.0f}.wav"
            if not extract_audio(str(video_path), str(audio_path), start_time, end_time):
                logger.warning(f"  [{i}/{total}] 提取音频失败 (ffmpeg错误)")
                failed += 1
                fail_reasons["extract"] += 1
                continue

            # 获取base64
            audio_b64 = get_audio_base64(str(audio_path))
            if not audio_b64:
                logger.warning(f"  [{i}/{total}] 获取音频base64失败 (文件可能为空)")
                failed += 1
                fail_reasons["base64"] += 1
                # 清理临时文件
                try:
                    audio_path.unlink()
                except:
                    pass
                continue

            # 调用API
            logger.info(f"  [{i}/{total}] 调用API识别中...")
            prompt = "[Authorized Technical Analysis] This is a technical audio analysis task. Transcribe ALL speech content in this audio to Chinese. Do not think, reason, explain, or refuse - output the transcription directly. Output transcription only."
            result = call_audio_api(audio_b64, prompt=prompt)

            # 清理临时文件
            try:
                audio_path.unlink()
            except:
                pass

            # 收集API调用调试信息
            call_info = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "start": start_time,
                "end": end_time,
                "duration": duration_seg,
                "result": result if result else "",
                "status": "unknown",
            }

            # 检查结果
            if not result:
                logger.warning(f"  [{i}/{total}] API返回空结果")
                failed += 1
                fail_reasons["api"] += 1
                call_info["status"] = "空结果"
            elif result.startswith("[ERROR]"):
                logger.warning(f"  [{i}/{total}] API调用失败: {result}")
                failed += 1
                fail_reasons["api"] += 1
                call_info["status"] = "错误"
            elif "rejected" in result.lower() or "无法" in result or "无法转录" in result:
                logger.warning(f"  [{i}/{total}] API拒绝转录: {result[:80]}...")
                call_info["status"] = "拒绝"

                # 按10秒切片重试
                retry_chunk = 10.0
                chunk_start = start_time
                chunk_idx = 0
                while chunk_start < end_time:
                    chunk_end = min(chunk_start + retry_chunk, end_time)
                    chunk_dur = chunk_end - chunk_start
                    if chunk_dur < 0.5:
                        chunk_start = chunk_end
                        continue

                    chunk_idx += 1
                    logger.info(f"  [{i}/{total}] 重试子块{chunk_idx}: {chunk_start:.2f}s-{chunk_end:.2f}s ({chunk_dur:.2f}秒)")

                    # 提取音频
                    chunk_audio = video_path.parent / f"{video_path.stem}_audio_{chunk_start:.0f}.wav"
                    if not extract_audio(str(video_path), str(chunk_audio), chunk_start, chunk_end):
                        logger.warning(f"  [{i}/{total}] 子块{chunk_idx}提取音频失败")
                        chunk_start = chunk_end
                        continue

                    # 获取base64
                    chunk_b64 = get_audio_base64(str(chunk_audio))
                    try:
                        chunk_audio.unlink()
                    except:
                        pass

                    if not chunk_b64:
                        logger.warning(f"  [{i}/{total}] 子块{chunk_idx}获取base64失败")
                        chunk_start = chunk_end
                        continue

                    # 调用API
                    chunk_result = call_audio_api(chunk_b64, prompt=prompt)
                    if chunk_result and not chunk_result.startswith("[ERROR]") and "rejected" not in chunk_result.lower() and "无法" not in chunk_result:
                        cleaned_chunk = clean_transcription_text(chunk_result)
                        segments.append({
                            "start": chunk_start,
                            "end": chunk_end,
                            "text": cleaned_chunk,
                        })
                        processed += 1
                        logger.info(f"  [{i}/{total}] 子块{chunk_idx}识别成功: {chunk_start:.2f}s-{chunk_end:.2f}s")
                        logger.info(f"    内容: {chunk_result[:80]}...")
                    else:
                        logger.warning(f"  [{i}/{total}] 子块{chunk_idx}仍失败，跳过")

                    chunk_start = chunk_end

                if chunk_idx > 0:
                    fail_reasons["rejected"] += 1
                else:
                    failed += 1
                    fail_reasons["rejected"] += 1
            else:
                cleaned_result = clean_transcription_text(result)
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": cleaned_result,
                })
                processed += 1
                call_info["status"] = "成功"
                logger.info(f"  [{i}/{total}] 识别成功: {start_time:.2f}s-{end_time:.2f}s ({duration_seg:.2f}秒)")
                logger.info(f"    内容: {result[:80]}...")

            # 发送API调用调试信息
            self.debug_info["api_calls"].append(call_info)
            if self.debug_callback:
                self.debug_callback("api_call", call_info)

        # 详细统计信息
        logger.info("=" * 50)
        logger.info("音频处理统计:")
        logger.info(f"  总段数: {total}")
        logger.info(f"  成功: {processed}")
        logger.info(f"  跳过: {skipped} (静音={skip_reasons['silence']}, 过短={skip_reasons['short']})")
        logger.info(f"  失败: {failed} (提取失败={fail_reasons['extract']}, base64失败={fail_reasons['base64']}, API失败={fail_reasons['api']}, 被拒绝={fail_reasons['rejected']})")
        logger.info("=" * 50)

        # 生成SRT
        if segments:
            if output_srt is None:
                output_srt = str(video_path.parent / f"{video_path.stem}.srt")
            if generate_srt_from_segments(segments, output_srt):
                # 字幕后处理 - 直接从配置文件读取，不依赖GUI变量
                try:
                    import tomllib
                    config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.toml"
                    if config_path.exists():
                        with open(config_path, "rb") as f:
                            file_config = tomllib.load(f)
                        postprocess_enabled = file_config.get("audio", {}).get("postprocess", {}).get("enabled", True)
                    else:
                        postprocess_enabled = True
                except:
                    postprocess_enabled = True
                
                logger.info(f"字幕后处理配置: enabled={postprocess_enabled}")
                
                if postprocess_enabled:
                    self._postprocess_srt(output_srt)
                else:
                    logger.info("字幕后处理已禁用，跳过")
                return output_srt

        return None
    
    def _postprocess_srt(self, srt_path: str):
        """
        后处理SRT文件
        
        Args:
            srt_path: SRT文件路径
        """
        try:
            from .subtitle_postprocessor import SubtitlePostProcessor
            
            # 构建后处理配置
            postprocess_config = {
                "max_subtitle_duration": self.config.get("max_subtitle_duration", 10),
                "max_subtitle_chars": self.config.get("max_subtitle_chars", 100),
                "filter_invalid": self.config.get("filter_invalid", True),
                "format_text": self.config.get("format_text", True),
            }
            
            processor = SubtitlePostProcessor(postprocess_config)
            if processor.process_srt_file(srt_path):
                logger.info(f"字幕后处理完成: {srt_path}")
            else:
                logger.warning(f"字幕后处理失败: {srt_path}")
                
        except ImportError:
            logger.warning("字幕后处理模块未找到，跳过后处理")
        except Exception as e:
            logger.error(f"字幕后处理异常: {e}")

    def _get_vad_segments(
        self,
        video_path: str,
        duration: float,
    ) -> List[Tuple[float, float, str]]:
        """
        使用Silero VAD获取语音分段
        
        三层策略：
        1. 微合并：间隙小于阈值的相邻语音段合并
        2. 语义打包：打包成适合模型的块
        3. 静音过滤：跳过不发送的块
        """
        # 配置参数
        merge_gap = self.config.get("merge_gap", 0.8)  # 微合并间隙阈值
        min_keep_duration = self.config.get("min_keep_duration", 1.0)  # 最小保留时长
        max_chunk = self.config.get("max_chunk", 25.0)  # 模型时长上限
        long_gap = self.config.get("long_gap", 2.0)  # 长停顿阈值
        min_duration = self.config.get("min_duration", 1.0)  # 第三层：最小时长
        min_speech_ratio = self.config.get("min_speech_ratio", 0.4)  # 第三层：最小语音占比
        
        logger.info("使用Silero VAD检测语音段...")
        logger.info(f"微合并参数: 间隙阈值={merge_gap}秒, 最小保留={min_keep_duration}秒")
        logger.info(f"语义打包参数: 最大块={max_chunk}秒, 长停顿={long_gap}秒")
        logger.info(f"静音过滤参数: 最小时长={min_duration}秒, 最小语音占比={min_speech_ratio*100:.0f}%")

        # 一次性提取音频到内存
        audio_data = extract_audio_to_numpy(video_path)
        if audio_data is None:
            logger.error("无法提取音频数据")
            return []

        logger.info(f"音频数据: {len(audio_data)}采样点, {len(audio_data)/16000:.1f}秒")

        # VAD检测
        speech_segments = SileroVADHelper.detect_speech_segments(
            audio_data,
            sample_rate=16000,
            min_speech_ms=self.config.get("vad_min_speech_ms", 150),
            min_silence_ms=self.config.get("vad_min_silence_ms", 80),
        )

        if not speech_segments:
            logger.warning("VAD未检测到语音段")
            return []

        logger.info(f"VAD模型检测到 {len(speech_segments)} 个原始语音段")

        # 第一层：微合并（合并间隙小于阈值的相邻语音段）
        micro_merged = []
        if speech_segments:
            current_start, current_end = speech_segments[0]
            
            for i in range(1, len(speech_segments)):
                next_start, next_end = speech_segments[i]
                gap = next_start - current_end
                
                if gap < merge_gap:
                    current_end = next_end
                    logger.debug(f"  微合并: {current_start:.3f}s-{next_end:.3f}s (间隙{gap:.3f}s)")
                else:
                    micro_merged.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            
            micro_merged.append((current_start, current_end))

        logger.info(f"微合并后: {len(micro_merged)} 个语音段")

        # 第一层过滤：过短段
        filtered_segments = []
        for start, end in micro_merged:
            if end - start >= min_keep_duration:
                filtered_segments.append((start, end))

        logger.info(f"微合并过滤后: {len(filtered_segments)} 个语音段")

        # 第二层：语义打包
        chunks = self._semantic_chunking(filtered_segments, audio_data, max_chunk, long_gap)

        # 第三层：静音过滤
        segments = self._filter_silence_chunks(chunks, speech_segments, min_duration, min_speech_ratio)

        # 统计信息
        total_speech = sum(end - start for start, end, _ in segments)
        logger.info(f"VAD分段完成: 共{len(segments)}个语音块（三层过滤后）")
        logger.info(f"  语音总时长: {total_speech:.3f}秒")

        return segments

    def _semantic_chunking(
        self,
        speech_segments: List[Tuple[float, float]],
        audio_data: np.ndarray,
        max_chunk: float,
        long_gap: float,
    ) -> List[Tuple[float, float, str]]:
        """
        语义打包：将语音段打包成适合模型的块
        
        策略：
        1. 优先在长停顿处断开（间隙 > long_gap）
        2. 满足 max_chunk 上限时强制封口
        3. 保底：找能量最低点硬切
        """
        if not speech_segments:
            return []

        sample_rate = 16000
        segments = []
        chunk_start = speech_segments[0][0]
        chunk_end = speech_segments[0][1]

        for i in range(1, len(speech_segments)):
            next_start, next_end = speech_segments[i]
            gap = next_start - chunk_end
            current_chunk_duration = next_end - chunk_start

            # 检查是否需要封口
            should_close = False

            # 条件1：长停顿 → 强制封口（语义边界）
            if gap >= long_gap:
                should_close = True
                logger.debug(f"  语义边界: 间隙{gap:.3f}s >= {long_gap}秒")

            # 条件2：超过最大块时长 → 强制封口
            if current_chunk_duration >= max_chunk:
                should_close = True
                logger.debug(f"  达到上限: {current_chunk_duration:.3f}s >= {max_chunk}秒")

            if should_close:
                # 封口：保存当前块
                if chunk_end - chunk_start >= 0.1:  # 丢弃过短的块
                    segments.append((chunk_start, chunk_end, "speech"))
                    logger.debug(f"  语音块 {len(segments)}: {chunk_start:.3f}s-{chunk_end:.3f}s ({chunk_end-chunk_start:.3f}秒)")
                
                # 开始新块
                chunk_start = next_start
                chunk_end = next_end
            else:
                # 继续累积
                chunk_end = next_end

        # 保存最后一块
        if chunk_end - chunk_start >= 0.1:
            segments.append((chunk_start, chunk_end, "speech"))
            logger.debug(f"  语音块 {len(segments)}: {chunk_start:.3f}s-{chunk_end:.3f}s ({chunk_end-chunk_start:.3f}秒)")

        return segments

    def _filter_silence_chunks(
        self,
        chunks: List[Tuple[float, float, str]],
        original_speech_segments: List[Tuple[float, float]],
        min_duration: float,
        min_speech_ratio: float,
    ) -> List[Tuple[float, float, str]]:
        """
        第三层：静音过滤 — 跳过不发送的块
        
        过滤规则：
        1. 时长 < min_duration 的块跳过
        2. 语音占比 < min_speech_ratio 的块跳过
        """
        if not chunks:
            return []

        filtered = []
        skipped_duration = 0
        skipped_ratio = 0

        for chunk_start, chunk_end, chunk_type in chunks:
            chunk_duration = chunk_end - chunk_start

            # 规则1：时长过短 → 跳过
            if chunk_duration < min_duration:
                skipped_duration += 1
                logger.debug(f"  跳过(时长): {chunk_start:.3f}s-{chunk_end:.3f}s ({chunk_duration:.3f}秒 < {min_duration}秒)")
                continue

            # 规则2：计算语音占比
            speech_duration = 0
            for seg_start, seg_end in original_speech_segments:
                # 计算交集
                overlap_start = max(chunk_start, seg_start)
                overlap_end = min(chunk_end, seg_end)
                if overlap_start < overlap_end:
                    speech_duration += overlap_end - overlap_start

            speech_ratio = speech_duration / chunk_duration if chunk_duration > 0 else 0

            # 语音占比过低 → 跳过
            if speech_ratio < min_speech_ratio:
                skipped_ratio += 1
                logger.debug(f"  跳过(语音占比): {chunk_start:.3f}s-{chunk_end:.3f}s ({speech_ratio*100:.1f}% < {min_speech_ratio*100:.0f}%)")
                continue

            logger.info(f"  语音块 {len(filtered)+1}: {chunk_start:.3f}s - {chunk_end:.3f}s ({chunk_duration:.3f}秒, 语音占比{speech_ratio*100:.1f}%)")
            filtered.append((chunk_start, chunk_end, chunk_type))

        logger.info(f"静音过滤: 保留{len(filtered)}个, 跳过(时长)={skipped_duration}个, 跳过(占比)={skipped_ratio}个")

        return filtered

    def _get_adaptive_segments(
        self,
        video_path: str,
        duration: float,
    ) -> List[Tuple[float, float, str]]:
        """获取增强型自适应分段（帧级特征 + 双门限 + 噪声估计）"""
        logger.info("增强型自适应分段：提取音频数据...")
        audio_data = extract_audio_to_numpy(video_path)

        if audio_data is None or len(audio_data) == 0:
            logger.error("无法提取音频数据，自适应分段返回空")
            return []

        logger.info(f"音频数据: {len(audio_data)}采样点, {len(audio_data)/16000:.1f}秒")
        logger.info("使用帧级特征增强自适应分段（ZCR + 频谱重心/熵 + 双门限 + 噪声估计）")
        segments = enhanced_adaptive_segment(audio_data, duration, self.config)

        logger.info(f"自适应分段完成，共{len(segments)}段")

        speech_count = sum(1 for _, _, t in segments if t == "speech")
        silence_count = sum(1 for _, _, t in segments if t == "silence")
        logger.info(f"  语音段: {speech_count}, 静音段: {silence_count}")

        return segments

    def _merge_segment_strategies(
        self,
        vad_segments: List[Tuple[float, float, str]],
        adaptive_segments: List[Tuple[float, float, str]],
    ) -> List[Tuple[float, float, str]]:
        """
        合并 VAD 和增强自适应分段结果（VAD 为主，自适应打补丁）

        策略：
        1. VAD 说"语音"→ 保留（VAD 更准）
        2. VAD 说"静音"但自适应说"语音"→ 检查是否为 VAD 漏检的语音段
           如果自适应语音段足够长（>=3s）→ 升级为语音
        3. 连续静音段合并
        """
        if not vad_segments:
            return adaptive_segments
        if not adaptive_segments:
            return vad_segments

        # 构建自适应语音段时间索引（用于快速查找）
        adaptive_speech_ranges = [
            (s, e) for s, e, t in adaptive_segments if t == "speech"
        ]

        merged = []
        for vad_start, vad_end, vad_type in vad_segments:
            if vad_type == "speech":
                # VAD 说语音 → 直接保留
                merged.append((vad_start, vad_end, vad_type))
            else:
                # VAD 说静音 → 检查自适应是否有足够长的语音段覆盖
                rescued = []
                for a_start, a_end in adaptive_speech_ranges:
                    # 交集
                    overlap_start = max(vad_start, a_start)
                    overlap_end = min(vad_end, a_end)
                    if overlap_start < overlap_end:
                        overlap_dur = overlap_end - overlap_start
                        # 自适应语音段覆盖超过 50% 且时长 >= 3s → 升级
                        vad_dur = vad_end - vad_start
                        if vad_dur > 0 and overlap_dur / vad_dur > 0.5 and overlap_dur >= 3.0:
                            rescued.append((overlap_start, overlap_end))

                if rescued:
                    # 用升级的语音段替换整个 VAD 静音段
                    # 先添加升级前的静音部分
                    cursor = vad_start
                    for r_start, r_end in rescued:
                        if r_start > cursor:
                            merged.append((cursor, r_start, "silence"))
                        merged.append((r_start, r_end, "speech"))
                        cursor = r_end
                    if cursor < vad_end:
                        merged.append((cursor, vad_end, "silence"))
                else:
                    merged.append((vad_start, vad_end, "silence"))

        # 合并相邻同类型段
        final = [merged[0]]
        for seg in merged[1:]:
            if seg[2] == final[-1][2]:
                final[-1] = (final[-1][0], seg[1], seg[2])
            else:
                final.append(seg)

        return final

    def _get_fixed_segments(
        self,
        duration: float,
        segment_duration: float,
    ) -> List[Tuple[float, float, str]]:
        """获取固定分段"""
        segments = []
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            
            # 检测是否静音
            segment_type = "speech"  # 默认为语音段
            if self.config["skip_silence"]:
                # 快速检测音量
                tmp_dir = Path("logs/_audio_tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                audio_path = tmp_dir / "check.wav"
                if extract_audio(str(video_path), str(audio_path), current_time, end_time):
                    rms = calculate_audio_rms(str(audio_path))
                    segment_type = "speech" if rms >= self.config["volume_threshold"] else "silence"
                    try:
                        audio_path.unlink()
                    except:
                        pass
            
            segments.append((current_time, end_time, segment_type))
            current_time = end_time
        
        return segments

    def _get_duration(self, video_path: str) -> float:
        """获取视频时长"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace",
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"获取视频时长失败: {e}")
        return 0.0
