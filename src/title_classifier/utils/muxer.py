"""字幕封装模块 - 将SRT字幕封装到视频容器中"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class SubtitleMuxer:
    """字幕封装器"""
    
    def __init__(self, config: dict = None):
        """
        初始化字幕封装器
        
        Args:
            config: 配置参数，包含：
                - output_format: 输出格式 (auto/mkv/mp4)
                - file_handling: 文件处理方式 (new/overwrite)
                - subtitle_processing: 字幕处理方式 (direct/convert)
                - encoding: 字幕编码 (auto/utf-8)
                - language_detection: 是否自动检测语言
                - track_naming: 轨道命名方式 (auto/manual)
                - default_track: 是否设置为默认轨道
        """
        default_config = {
            "output_format": "auto",
            "file_handling": "new",
            "subtitle_processing": "direct",
            "encoding": "utf-8",
            "language_detection": True,
            "track_naming": "auto",
            "default_track": True,
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        # 语言检测映射
        self.language_map = {
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
        }
    
    def mux_subtitle(self, video_path: str, srt_path: str, 
                    output_path: str = None, progress_callback: Callable = None) -> Dict:
        """
        将SRT字幕封装到视频中
        
        Args:
            video_path: 视频文件路径
            srt_path: SRT字幕文件路径
            output_path: 输出文件路径（可选）
            progress_callback: 进度回调函数
            
        Returns:
            {
                "success": bool,
                "output_path": str,
                "error": str,
                "details": dict
            }
        """
        try:
            # 验证文件存在
            if not Path(video_path).exists():
                return {"success": False, "error": f"视频文件不存在: {video_path}"}
            
            if not Path(srt_path).exists():
                return {"success": False, "error": f"字幕文件不存在: {srt_path}"}
            
            # 确定输出路径
            if output_path is None:
                output_path = self._get_output_path(video_path)
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 检测字幕语言
            language = None
            if self.config["language_detection"]:
                language = self._detect_language(srt_path)
            
            # 构建FFmpeg命令
            cmd = self._build_ffmpeg_command(video_path, srt_path, output_path, language)
            
            # 执行FFmpeg命令
            result = self._execute_ffmpeg(cmd, progress_callback)
            
            if result["success"]:
                # overwrite模式：用临时文件替换原文件
                if self.config["file_handling"] == "overwrite":
                    import shutil
                    if Path(output_path).exists():
                        try:
                            logger.info(f"[覆写] 替换原文件: {Path(video_path).name}")
                            shutil.move(output_path, str(video_path))
                            output_path = str(video_path)
                            logger.info("[覆写] 替换完成")
                        except Exception as move_err:
                            logger.error(f"[覆写] 替换失败: {move_err}")
                            return {"success": False, "error": f"覆写替换失败: {move_err}"}
                    else:
                        return {"success": False, "error": f"临时文件不存在: {output_path}"}
                
                # 验证输出文件
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    return {
                        "success": True,
                        "output_path": output_path,
                        "details": {
                            "file_size": file_size,
                            "language": language,
                            "format": Path(output_path).suffix[1:],
                        }
                    }
                else:
                    return {"success": False, "error": "输出文件未生成"}
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"封装字幕失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_output_path(self, video_path: str) -> str:
        """
        根据配置生成输出文件路径
        
        Args:
            video_path: 原始视频路径
            
        Returns:
            输出文件路径
        """
        video_path = Path(video_path)
        
        # 根据配置确定输出格式
        if self.config["output_format"] == "auto":
            output_format = video_path.suffix
        elif self.config["output_format"] == "mkv":
            output_format = ".mkv"
        elif self.config["output_format"] == "mp4":
            output_format = ".mp4"
        else:
            output_format = video_path.suffix
        
        stem = video_path.stem
        parent = video_path.parent
        
        if self.config["file_handling"] == "overwrite":
            # overwrite: 临时文件，mux_subtitle 成功后会替换原文件
            return str(parent / f"{stem}_muxed_tmp{output_format}")
        else:
            # new: 保留 _muxed 后缀新文件
            return str(parent / f"{stem}_muxed{output_format}")
    
    def _detect_language(self, srt_path: str) -> Optional[str]:
        """
        检测字幕语言
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            语言代码或None
        """
        try:
            # 读取字幕文件内容
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 简单的语言检测（基于字符特征）
            # 中文字符检测
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            # 英文字符检测
            english_chars = len(re.findall(r'[a-zA-Z]', content))
            # 日文字符检测
            japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', content))
            # 韩文字符检测
            korean_chars = len(re.findall(r'[\uac00-\ud7af]', content))
            
            # 计算比例
            total_chars = chinese_chars + english_chars + japanese_chars + korean_chars
            if total_chars == 0:
                return None
            
            # 找出占比最高的语言
            ratios = {
                "zh": chinese_chars / total_chars,
                "en": english_chars / total_chars,
                "ja": japanese_chars / total_chars,
                "ko": korean_chars / total_chars,
            }
            
            # 找出最大比例的语言
            max_lang = max(ratios, key=ratios.get)
            max_ratio = ratios[max_lang]
            
            # 如果比例超过阈值，则认为是该语言
            if max_ratio > 0.3:
                return max_lang
            
            return None
            
        except Exception as e:
            logger.warning(f"语言检测失败: {e}")
            return None
    
    def _build_ffmpeg_command(self, video_path: str, srt_path: str, 
                            output_path: str, language: str = None) -> List[str]:
        """
        构建FFmpeg命令
        
        Args:
            video_path: 视频路径
            srt_path: 字幕路径
            output_path: 输出路径
            language: 语言代码
            
        Returns:
            FFmpeg命令列表
        """
        cmd = ["ffmpeg", "-y"]
        
        # 输入文件
        cmd.extend(["-i", video_path])
        cmd.extend(["-i", srt_path])
        
        # 复制视频和音频流
        cmd.extend(["-c:v", "copy"])
        cmd.extend(["-c:a", "copy"])
        
        # 字幕编码
        output_format = Path(output_path).suffix.lower()
        if output_format == ".mkv":
            # MKV支持SRT直接封装
            cmd.extend(["-c:s", "srt"])
        elif output_format == ".mp4":
            # MP4需要转换为mov_text
            cmd.extend(["-c:s", "mov_text"])
        else:
            # 其他格式尝试直接复制
            cmd.extend(["-c:s", "copy"])
        
        # 字幕语言标签
        if language and language in self.language_map:
            cmd.extend(["-metadata:s:s:0", f"language={language}"])
        
        # 轨道命名
        if self.config["track_naming"] == "auto" and language:
            track_name = self.language_map.get(language, "Unknown")
            cmd.extend(["-metadata:s:s:0", f"title={track_name}"])
        
        # 默认轨道设置
        if self.config["default_track"]:
            cmd.extend(["-disposition:s:0", "default"])
        
        # 输出文件
        cmd.append(output_path)
        
        return cmd
    
    def _execute_ffmpeg(self, cmd: List[str], progress_callback: Callable = None) -> Dict:
        """
        执行FFmpeg命令
        
        Args:
            cmd: FFmpeg命令
            progress_callback: 进度回调函数
            
        Returns:
            {"success": bool, "error": str}
        """
        try:
            logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                encoding="utf-8",
                errors="replace"
            )
            
            if result.returncode == 0:
                logger.info("FFmpeg执行成功")
                if progress_callback:
                    progress_callback(100, "封装完成")
                return {"success": True}
            else:
                error_msg = result.stderr
                logger.error(f"FFmpeg执行失败 (返回码:{result.returncode}): {error_msg[:500]}")
                return {"success": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            error_msg = "FFmpeg执行超时"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"FFmpeg执行异常: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def batch_mux(self, video_srt_pairs: List[Tuple[str, str]], 
                 progress_callback: Callable = None) -> Dict:
        """
        批量封装字幕
        
        Args:
            video_srt_pairs: 视频和字幕文件对列表 [(video_path, srt_path), ...]
            progress_callback: 进度回调函数
            
        Returns:
            {
                "success": bool,
                "total": int,
                "success_count": int,
                "failed_count": int,
                "results": list,
                "failed_files": list
            }
        """
        total = len(video_srt_pairs)
        success_count = 0
        failed_count = 0
        results = []
        failed_files = []
        
        for i, (video_path, srt_path) in enumerate(video_srt_pairs):
            # 更新进度
            if progress_callback:
                progress = int((i / total) * 100)
                progress_callback(progress, f"处理 {i+1}/{total}: {Path(video_path).name}")
            
            # 封装单个文件
            result = self.mux_subtitle(video_path, srt_path)
            results.append({
                "video": video_path,
                "srt": srt_path,
                "result": result
            })
            
            if result["success"]:
                success_count += 1
            else:
                failed_count += 1
                failed_files.append({
                    "video": video_path,
                    "srt": srt_path,
                    "error": result["error"]
                })
        
        # 完成进度
        if progress_callback:
            progress_callback(100, f"批量封装完成: 成功 {success_count}, 失败 {failed_count}")
        
        return {
            "success": failed_count == 0,
            "total": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results,
            "failed_files": failed_files
        }
    
    def retry_failed(self, failed_files: List[Dict], 
                    progress_callback: Callable = None) -> Dict:
        """
        重试失败的封装操作
        
        Args:
            failed_files: 失败文件列表
            progress_callback: 进度回调函数
            
        Returns:
            {
                "success": bool,
                "total": int,
                "success_count": int,
                "failed_count": int,
                "results": list
            }
        """
        total = len(failed_files)
        success_count = 0
        failed_count = 0
        results = []
        
        for i, file_info in enumerate(failed_files):
            video_path = file_info["video"]
            srt_path = file_info["srt"]
            
            # 更新进度
            if progress_callback:
                progress = int((i / total) * 100)
                progress_callback(progress, f"重试 {i+1}/{total}: {Path(video_path).name}")
            
            # 封装单个文件
            result = self.mux_subtitle(video_path, srt_path)
            results.append({
                "video": video_path,
                "srt": srt_path,
                "result": result
            })
            
            if result["success"]:
                success_count += 1
            else:
                failed_count += 1
        
        # 完成进度
        if progress_callback:
            progress_callback(100, f"重试完成: 成功 {success_count}, 失败 {failed_count}")
        
        return {
            "success": failed_count == 0,
            "total": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }