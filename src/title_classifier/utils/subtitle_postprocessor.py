"""字幕后处理器 - 拆分长字幕、过滤无效内容、格式化文本"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SubtitlePostProcessor:
    """字幕后处理器"""
    
    def __init__(self, config: dict = None):
        """
        初始化后处理器
        
        Args:
            config: 配置参数，包含：
                - max_subtitle_duration: 单个字幕最大时长（秒），默认10
                - max_subtitle_chars: 单个字幕最大字符数，默认100
                - filter_invalid: 是否过滤无效内容，默认True
                - format_text: 是否格式化文本，默认True
        """
        default_config = {
            "max_subtitle_duration": 10,
            "max_subtitle_chars": 100,
            "filter_invalid": True,
            "format_text": True,
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
    
    def process_srt_file(self, srt_path: str) -> bool:
        """
        处理SRT文件
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            是否成功
        """
        try:
            # 读取SRT文件
            segments = self._parse_srt(srt_path)
            if not segments:
                logger.warning(f"SRT文件为空或解析失败: {srt_path}")
                return False
            
            logger.info(f"解析到 {len(segments)} 个字幕段")
            
            # 过滤无效内容
            if self.config["filter_invalid"]:
                original_count = len(segments)
                segments = self._filter_invalid_segments(segments)
                filtered_count = original_count - len(segments)
                if filtered_count > 0:
                    logger.info(f"过滤了 {filtered_count} 个无效字幕段")
            
            # 拆分长字幕
            segments = self._split_long_subtitles(segments)
            logger.info(f"拆分后共 {len(segments)} 个字幕段")
            
            # 格式化文本
            if self.config["format_text"]:
                segments = self._format_segments(segments)
            
            # 重新编号
            segments = self._renumber_segments(segments)
            
            # 写入SRT文件
            self._write_srt(srt_path, segments)
            
            logger.info(f"字幕后处理完成: {srt_path}")
            return True
            
        except Exception as e:
            logger.error(f"字幕后处理失败: {e}")
            return False
    
    def _parse_srt(self, srt_path: str) -> List[Dict]:
        """
        解析SRT文件
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            字幕段列表
        """
        segments = []
        
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按空行分割字幕段
            blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # 解析序号
                try:
                    index = int(lines[0].strip())
                except ValueError:
                    continue
                
                # 解析时间戳
                time_match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    lines[1].strip()
                )
                if not time_match:
                    continue
                
                start_time = self._time_to_seconds(time_match.group(1))
                end_time = self._time_to_seconds(time_match.group(2))
                
                # 解析文本内容（可能有多行）
                text = '\n'.join(lines[2:]).strip()
                
                segments.append({
                    "index": index,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"解析SRT文件失败: {e}")
            return []
    
    def _time_to_seconds(self, time_str: str) -> float:
        """
        将SRT时间格式转换为秒数
        
        Args:
            time_str: SRT时间格式 (00:00:00,000)
            
        Returns:
            秒数
        """
        parts = time_str.replace(',', '.').split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _seconds_to_time(self, seconds: float) -> str:
        """
        将秒数转换为SRT时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            SRT时间格式 (00:00:00,000)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _filter_invalid_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        过滤无效字幕段
        
        Args:
            segments: 字幕段列表
            
        Returns:
            过滤后的字幕段列表
        """
        valid_segments = []
        
        for seg in segments:
            text = seg["text"]
            
            # 过滤时间戳列表（API返回异常）
            if self._is_timestamp_list(text):
                logger.debug(f"过滤时间戳列表: {text[:50]}...")
                continue
            
            # 过滤拒绝响应
            if self._is_rejected_response(text):
                logger.debug(f"过滤拒绝响应: {text[:50]}...")
                continue
            
            # 过滤安全拒绝
            if self._is_safety_rejection(text):
                logger.debug(f"过滤安全拒绝: {text[:50]}...")
                continue
            
            # 过滤长分析报告
            if self._is_analysis_report(text):
                logger.debug(f"过滤分析报告: {text[:50]}...")
                continue
            
            # 过滤空内容或无意义内容
            if not text or len(text.strip()) < 2:
                continue
            
            valid_segments.append(seg)
        
        return valid_segments
    
    def _is_timestamp_list(self, text: str) -> bool:
        """检查是否为时间戳列表"""
        # 匹配连续的时间戳格式
        timestamp_pattern = r'\d{2}:\d{2}(?::\d{2})?'
        timestamps = re.findall(timestamp_pattern, text)
        
        # 如果文本主要是时间戳，则认为是时间戳列表
        if len(timestamps) > 5:
            # 计算时间戳占比
            timestamp_chars = sum(len(t) for t in timestamps)
            total_chars = len(text.replace('\n', '').replace(' ', ''))
            if total_chars > 0 and timestamp_chars / total_chars > 0.5:
                return True
        
        return False
    
    def _is_rejected_response(self, text: str) -> bool:
        """检查是否为拒绝响应"""
        rejected_patterns = [
            r"rejected.*high risk",
            r"request was rejected",
            r"被拒绝",
            r"无法处理",
            r"无法转录",
            r"无法提供",
        ]
        
        for pattern in rejected_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_safety_rejection(self, text: str) -> bool:
        """检查是否为安全拒绝"""
        safety_patterns = [
            r"很抱歉",
            r"无法处理这个请求",
            r"安全准则",
            r"不当内容",
            r"色情内容",
        ]
        
        for pattern in safety_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_analysis_report(self, text: str) -> bool:
        """检查是否为分析报告"""
        # 长文本且包含分析相关关键词
        if len(text) > 500:
            analysis_keywords = ["分析", "报告", "总结", "转录报告", "音频转录"]
            for keyword in analysis_keywords:
                if keyword in text:
                    return True
        
        return False
    
    def _split_long_subtitles(self, segments: List[Dict]) -> List[Dict]:
        """
        拆分长字幕
        
        Args:
            segments: 字幕段列表
            
        Returns:
            拆分后的字幕段列表
        """
        result = []
        
        for seg in segments:
            duration = seg["end"] - seg["start"]
            text = seg["text"]
            char_count = len(text)
            
            # 检查是否需要拆分
            need_split = (
                duration > self.config["max_subtitle_duration"] or
                char_count > self.config["max_subtitle_chars"]
            )
            
            if not need_split:
                result.append(seg)
                continue
            
            # 拆分字幕
            split_segments = self._split_segment(seg)
            result.extend(split_segments)
        
        return result
    
    def _split_segment(self, segment: Dict) -> List[Dict]:
        """
        拆分单个字幕段
        
        Args:
            segment: 字幕段
            
        Returns:
            拆分后的字幕段列表
        """
        text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        total_duration = end_time - start_time
        
        # 按边界拆分文本
        lines = self._split_by_boundary(text)
        
        if not lines:
            return [segment]
        
        # 计算总字符数
        total_chars = sum(len(line) for line in lines if line.strip())
        if total_chars == 0:
            return [segment]
        
        # 生成拆分后的字幕段
        result = []
        current_time = start_time
        
        for line in lines:
            if not line.strip():
                continue
            
            # 根据字符数比例分配时间
            line_chars = len(line)
            duration = (line_chars / total_chars) * total_duration
            
            # 限制时长范围（但不小于0.5秒）
            duration = max(min(duration, self.config["max_subtitle_duration"]), 0.5)
            
            seg_start = current_time
            seg_end = current_time + duration
            
            # 确保不超过原始结束时间
            if seg_end > end_time:
                seg_end = end_time
            
            result.append({
                "start": seg_start,
                "end": seg_end,
                "text": line.strip(),
            })
            
            current_time = seg_end
        
        return result
    
    def _split_by_boundary(self, text: str) -> List[str]:
        """
        按边界拆分文本
        
        Args:
            text: 文本
            
        Returns:
            拆分后的文本列表
        """
        # 首先按换行符分割
        lines = text.split('\n')
        
        result = []
        current_chunk = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # 空行作为分隔符
                if current_chunk:
                    result.append('\n'.join(current_chunk))
                    current_chunk = []
                continue
            
            # 检查当前行是否需要进一步拆分（按句子边界）
            if len(line) > self.config["max_subtitle_chars"]:
                # 先保存当前chunk
                if current_chunk:
                    result.append('\n'.join(current_chunk))
                    current_chunk = []
                
                # 按句子边界拆分长行
                split_lines = self._split_by_sentence_boundary(line)
                result.extend(split_lines)
            else:
                current_chunk.append(line)
                
                # 检查是否需要拆分
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > self.config["max_subtitle_chars"]:
                    result.append('\n'.join(current_chunk))
                    current_chunk = []
        
        # 添加剩余内容
        if current_chunk:
            result.append('\n'.join(current_chunk))
        
        return result
    
    def _split_by_sentence_boundary(self, text: str) -> List[str]:
        """
        按句子边界拆分文本
        
        Args:
            text: 文本
            
        Returns:
            拆分后的文本列表
        """
        # 句子结束符
        sentence_endings = r'[。！？!?]'
        
        # 按句子结束符分割
        sentences = re.split(f'({sentence_endings})', text)
        
        result = []
        current_chunk = ""
        
        for i, part in enumerate(sentences):
            if not part:
                continue
            
            # 如果是句子结束符，添加到当前chunk
            if re.match(sentence_endings, part):
                current_chunk += part
                continue
            
            # 检查添加当前部分后是否超过限制
            test_chunk = current_chunk + part if current_chunk else part
            
            if len(test_chunk) > self.config["max_subtitle_chars"]:
                # 保存当前chunk
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk = test_chunk
        
        # 添加剩余内容
        if current_chunk:
            result.append(current_chunk.strip())
        
        return result
    
    def _format_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        格式化字幕段
        
        Args:
            segments: 字幕段列表
            
        Returns:
            格式化后的字幕段列表
        """
        result = []
        
        for seg in segments:
            text = seg["text"]
            
            # 统一说话人标签格式
            text = self._format_speaker_labels(text)
            
            # 清理多余空格
            text = re.sub(r'\s+', ' ', text)
            
            # 清理多余换行
            text = re.sub(r'\n\s*\n', '\n', text)
            
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text.strip(),
            })
        
        return result
    
    def _format_speaker_labels(self, text: str) -> str:
        """
        格式化说话人标签
        
        Args:
            text: 文本
            
        Returns:
            格式化后的文本
        """
        # 统一中文说话人标签格式
        # （男声，...）-> (男声，...)
        text = re.sub(r'（(男|女)声[，,]([^）]*)）', r'(\1声，\2)', text)
        
        # (male, ...) -> (男声，...)
        text = re.sub(r'\(male[，,]([^)]*)\)', r'(男声，\1)', text, flags=re.IGNORECASE)
        text = re.sub(r'\(female[，,]([^)]*)\)', r'(女声，\1)', text, flags=re.IGNORECASE)
        
        # (male) -> (男声)
        text = re.sub(r'\(male\)', '(男声)', text, flags=re.IGNORECASE)
        text = re.sub(r'\(female\)', '(女声)', text, flags=re.IGNORECASE)
        
        # 男声：-> (男声)
        text = re.sub(r'男声[：:]', '(男声)', text)
        text = re.sub(r'女声[：:]', '(女声)', text)
        
        return text
    
    def _renumber_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        重新编号字幕段
        
        Args:
            segments: 字幕段列表
            
        Returns:
            重新编号后的字幕段列表
        """
        result = []
        
        for i, seg in enumerate(segments, 1):
            result.append({
                "index": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
        
        return result
    
    def _write_srt(self, srt_path: str, segments: List[Dict]) -> bool:
        """
        写入SRT文件
        
        Args:
            srt_path: SRT文件路径
            segments: 字幕段列表
            
        Returns:
            是否成功
        """
        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    index = seg["index"]
                    start = self._seconds_to_time(seg["start"])
                    end = self._seconds_to_time(seg["end"])
                    text = seg["text"]
                    
                    f.write(f"{index}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
            
            return True
            
        except Exception as e:
            logger.error(f"写入SRT文件失败: {e}")
            return False


def postprocess_srt(srt_path: str, config: dict = None) -> bool:
    """
    后处理SRT文件的便捷函数
    
    Args:
        srt_path: SRT文件路径
        config: 配置参数
        
    Returns:
        是否成功
    """
    processor = SubtitlePostProcessor(config)
    return processor.process_srt_file(srt_path)


if __name__ == "__main__":
    # 测试用
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python subtitle_postprocessor.py <srt文件路径>")
        sys.exit(1)
    
    srt_path = sys.argv[1]
    result = postprocess_srt(srt_path)
    
    if result:
        print(f"后处理完成: {srt_path}")
    else:
        print(f"后处理失败: {srt_path}")
