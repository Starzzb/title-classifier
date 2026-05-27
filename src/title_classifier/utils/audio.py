"""音频处理工具 - 支持自适应分段和音量阈值过滤"""

import base64
import subprocess
import logging
import tempfile
import shutil
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
    "scan_interval": 5,
    "min_segment": 15,
    "max_segment": 60,
    "energy_threshold": 0.01,
    "change_threshold": 0.15,
    "high_energy_min_duration": 15,
    "low_energy_merge_threshold": 30,
}


def load_audio_config() -> dict:
    """加载音频配置"""
    config = DEFAULT_CONFIG.copy()
    
    try:
        import tomllib
        config_path = Path(__file__).parent.parent.parent / "config" / "default.toml"
        
        if config_path.exists():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
                
            audio_config = data.get("audio", {})
            config["skip_silence"] = audio_config.get("skip_silence", config["skip_silence"])
            config["volume_threshold"] = audio_config.get("volume_threshold", config["volume_threshold"])
            config["segment_duration"] = audio_config.get("segment_duration", config["segment_duration"])
            
            adaptive_config = audio_config.get("adaptive", {})
            config["adaptive_enabled"] = adaptive_config.get("enabled", config["adaptive_enabled"])
            config["scan_interval"] = adaptive_config.get("scan_interval", config["scan_interval"])
            config["min_segment"] = adaptive_config.get("min_segment", config["min_segment"])
            config["max_segment"] = adaptive_config.get("max_segment", config["max_segment"])
            config["energy_threshold"] = adaptive_config.get("energy_threshold", config["energy_threshold"])
            config["change_threshold"] = adaptive_config.get("change_threshold", config["change_threshold"])
            config["high_energy_min_duration"] = adaptive_config.get("high_energy_min_duration", config["high_energy_min_duration"])
            config["low_energy_merge_threshold"] = adaptive_config.get("low_energy_merge_threshold", config["low_energy_merge_threshold"])
            
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


def scan_audio_energy(
    video_path: str,
    duration: float,
    scan_interval: float = 5.0,
) -> List[Tuple[float, float, float]]:
    """
    扫描音频能量分布（优化版本 - 一次性提取音频）
    
    Args:
        video_path: 视频路径
        duration: 视频时长
        scan_interval: 扫描间隔（秒）
    
    Returns:
        [(start_time, end_time, rms_energy), ...]
    """
    # 一次性提取完整音频到内存
    logger.info("正在提取音频数据...")
    audio_data = extract_audio_to_numpy(video_path)
    
    if audio_data is None:
        logger.error("无法提取音频数据")
        return []
    
    # 计算每个采样点对应的时间
    sample_rate = 16000  # 16kHz
    total_samples = len(audio_data)
    
    logger.info(f"开始扫描能量分布，扫描间隔: {scan_interval}秒")
    
    energy_map = []
    current_time = 0.0
    
    while current_time < duration:
        end_time = min(current_time + scan_interval, duration)
        
        # 计算采样点索引
        start_sample = int(current_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 计算RMS能量
        rms = calculate_rms_from_numpy(audio_data, start_sample, end_sample)
        energy_map.append((current_time, end_time, rms))
        
        current_time = end_time
    
    logger.info(f"能量扫描完成: {len(energy_map)}个采样点")
    return energy_map


def adaptive_segment(
    energy_map: List[Tuple[float, float, float]],
    config: dict,
) -> List[Tuple[float, float, str]]:
    """
    自适应分段
    
    Args:
        energy_map: 能量分布 [(start, end, rms), ...]
        config: 配置参数
    
    Returns:
        [(start_time, end_time, segment_type), ...]
        segment_type: "speech" 或 "silence"
    """
    if not energy_map:
        return []
    
    min_segment = config["min_segment"]
    max_segment = config["max_segment"]
    # 使用 volume_threshold 作为能量阈值（统一配置）
    energy_threshold = config.get("volume_threshold", config.get("energy_threshold", 0.01))
    change_threshold = config["change_threshold"]
    high_energy_min_duration = config["high_energy_min_duration"]
    low_energy_merge_threshold = config["low_energy_merge_threshold"]
    
    segments = []
    current_start = energy_map[0][0]
    current_type = "speech" if energy_map[0][2] >= energy_threshold else "silence"
    current_energy_sum = energy_map[0][2]
    current_count = 1
    
    for i in range(1, len(energy_map)):
        start, end, rms = energy_map[i]
        segment_type = "speech" if rms >= energy_threshold else "silence"
        
        # 检查是否需要分段
        need_split = False
        
        # 类型变化
        if segment_type != current_type:
            need_split = True
        
        # 能量变化超过阈值
        if current_count > 0:
            avg_energy = current_energy_sum / current_count
            if abs(rms - avg_energy) > change_threshold:
                need_split = True
        
        # 超过最大时长
        if end - current_start >= max_segment:
            need_split = True
        
        if need_split:
            # 确定分段类型
            avg_energy = current_energy_sum / current_count
            final_type = "speech" if avg_energy >= energy_threshold else "silence"
            
            # 检查最小时长
            duration = end - current_start
            if duration >= min_segment:
                segments.append((current_start, end, final_type))
                current_start = end
                current_type = segment_type
                current_energy_sum = rms
                current_count = 1
            else:
                # 继续累积
                current_energy_sum += rms
                current_count += 1
        else:
            current_energy_sum += rms
            current_count += 1
    
    # 添加最后一段
    if current_count > 0:
        avg_energy = current_energy_sum / current_count
        final_type = "speech" if avg_energy >= energy_threshold else "silence"
        segments.append((current_start, energy_map[-1][1], final_type))
    
    # 合并连续的低能量段
    merged_segments = []
    for seg in segments:
        if merged_segments and seg[2] == "silence" and merged_segments[-1][2] == "silence":
            # 合并
            prev = merged_segments[-1]
            if seg[1] - prev[0] <= low_energy_merge_threshold:
                merged_segments[-1] = (prev[0], seg[1], "silence")
            else:
                merged_segments.append(seg)
        else:
            merged_segments.append(seg)
    
    return merged_segments


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
    """音频处理器 - 支持自适应分段和音量阈值过滤"""

    def __init__(
        self,
        provider: str = "mimo",
        config: dict = None,
        **kwargs,
    ):
        self.provider = provider
        
        # 加载配置
        if config is None:
            config = load_audio_config()
        
        self.config = config
        
        # 允许通过kwargs覆盖配置
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value

    def process_video(
        self,
        video_path: str,
        output_srt: str = None,
        segment_duration: float = None,
    ) -> Optional[str]:
        """
        处理视频，生成字幕
        
        Args:
            video_path: 视频路径
            output_srt: 输出SRT路径
            segment_duration: 固定分段时长（秒），仅在自适应关闭时使用
        
        Returns:
            SRT文件路径
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
        logger.info(f"配置: 跳过静音={self.config['skip_silence']}, 阈值={self.config['volume_threshold']}")
        logger.info(f"自适应分段: {self.config['adaptive_enabled']}")

        # 确定分段策略
        if self.config["adaptive_enabled"]:
            segments_to_process = self._get_adaptive_segments(str(video_path), duration)
        else:
            seg_dur = segment_duration or self.config["segment_duration"]
            segments_to_process = self._get_fixed_segments(duration, seg_dur)

        # 处理每个分段
        segments = []
        total = len(segments_to_process)
        skipped = 0
        processed = 0

        for i, (start_time, end_time, segment_type) in enumerate(segments_to_process, 1):
            # 跳过静音段
            if self.config["skip_silence"] and segment_type == "silence":
                logger.debug(f"  [{i}/{total}] 静音片段 ({start_time:.0f}s-{end_time:.0f}s)，跳过")
                skipped += 1
                continue

            # 提取音频
            audio_path = video_path.parent / f"{video_path.stem}_audio_{start_time:.0f}.wav"
            if not extract_audio(str(video_path), str(audio_path), start_time, end_time):
                logger.warning(f"  [{i}/{total}] 提取音频失败，跳过")
                continue

            # 获取base64
            audio_b64 = get_audio_base64(str(audio_path))
            if not audio_b64:
                continue

            # 调用API
            logger.info(f"  [{i}/{total}] 正在识别: {start_time:.0f}s - {end_time:.0f}s")
            prompt = f"转录音频中的所有语音内容为中文。"
            result = call_audio_api(audio_b64, prompt=prompt, model="mimo-v2.5-pro")

            if result and not result.startswith("[ERROR]"):
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": result,
                })
                processed += 1

            # 清理临时文件
            try:
                audio_path.unlink()
            except:
                pass

        # 统计信息
        logger.info(f"音频处理完成: 共{total}段，处理{processed}段，跳过{skipped}段静音")

        # 生成SRT
        if segments:
            if output_srt is None:
                output_srt = str(video_path.parent / f"{video_path.stem}.srt")
            if generate_srt_from_segments(segments, output_srt):
                return output_srt

        return None

    def _get_adaptive_segments(
        self,
        video_path: str,
        duration: float,
    ) -> List[Tuple[float, float, str]]:
        """获取自适应分段"""
        logger.info("扫描音频能量分布...")
        energy_map = scan_audio_energy(
            video_path,
            duration,
            self.config["scan_interval"],
        )
        
        logger.info(f"扫描完成，共{len(energy_map)}个采样点")
        
        # 自适应分段
        segments = adaptive_segment(energy_map, self.config)
        
        logger.info(f"自适应分段完成，共{len(segments)}段")
        
        # 统计
        speech_count = sum(1 for _, _, t in segments if t == "speech")
        silence_count = sum(1 for _, _, t in segments if t == "silence")
        logger.info(f"  语音段: {speech_count}, 静音段: {silence_count}")
        
        return segments

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
                tmp_dir = Path(tempfile.mkdtemp())
                audio_path = tmp_dir / "check.wav"
                if extract_audio(str(video_path), str(audio_path), current_time, end_time):
                    rms = calculate_audio_rms(str(audio_path))
                    segment_type = "speech" if rms >= self.config["volume_threshold"] else "silence"
                    try:
                        audio_path.unlink()
                    except:
                        pass
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
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
