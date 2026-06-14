"""Stage1c 音频识别 Tab"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

from .context import AppContext
from .app import ToolTip

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = __import__('sys').executable
DEFAULT_CSV = "data/output/title_review.csv"


class StageAudioTab(ttk.Frame):
    """Stage1c 音频识别"""

    def __init__(self, parent, ctx: AppContext, run_command_callback=None):
        super().__init__(parent)
        self.ctx = ctx
        self._run_command = run_command_callback
        self._build_ui()
        self._load_audio_config_to_gui()

    def _build_ui(self):
        """构建Stage1c音频识别标签页"""
        tab = self

        # 说明文本
        desc_frame = ttk.LabelFrame(tab, text="说明")
        desc_frame.pack(fill=tk.X, padx=4, pady=4)
        desc_label = ttk.Label(desc_frame, text=(
            "音频识别是视觉识别的前置步骤，用于提高视觉识别的准确率。\n"
            "运行音频识别后，视觉识别会自动读取音频转录内容作为上下文，\n"
            "从而生成更准确的视频描述和关键词。"
        ), justify=tk.LEFT)
        desc_label.pack(padx=4, pady=4)

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        csv_entry = ttk.Entry(csv_frame, textvariable=self.ctx.csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "Stage1生成的CSV文件，音频识别会处理needs_vision=TRUE的行")

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1ca_provider_var = tk.StringVar(value="mimo")
        from ..providers import get_providers_for_gui
        providers = get_providers_for_gui("audio")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.s1ca_provider_var, values=providers, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(provider_combo, "选择音频AI服务提供商\n- mimo: 小米MiMo（推荐，支持音频理解）")

        # 音频配置
        audio_frame = ttk.LabelFrame(tab, text="音频配置")
        audio_frame.pack(fill=tk.X, padx=4, pady=4)

        # 第一行：基本配置
        audio_row1 = ttk.Frame(audio_frame)
        audio_row1.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(audio_row1, text="音量阈值:").pack(side=tk.LEFT, padx=4)
        self.s1ca_volume_threshold_var = tk.StringVar(value="0.01")
        vol_entry = ttk.Entry(audio_row1, textvariable=self.s1ca_volume_threshold_var, width=8)
        vol_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(vol_entry, "静音检测阈值（RMS能量，0-1之间）\n\n"
                "- 较低值：更敏感，可能误判噪音为语音\n"
                "- 较高值：更严格，可能漏掉轻声说话\n"
                "- 推荐：0.01")

        self.s1ca_skip_silence_var = tk.BooleanVar(value=True)
        skip_cb = ttk.Checkbutton(audio_row1, text="跳过静音", variable=self.s1ca_skip_silence_var)
        skip_cb.pack(side=tk.LEFT, padx=8)
        ToolTip(skip_cb, "勾选后会跳过静音片段，节省API调用")

        # 第二行：VAD配置
        audio_row2 = ttk.Frame(audio_frame)
        audio_row2.pack(fill=tk.X, padx=4, pady=2)

        self.s1ca_vad_enabled_var = tk.BooleanVar(value=True)
        vad_cb = ttk.Checkbutton(audio_row2, text="使用VAD语音检测", variable=self.s1ca_vad_enabled_var)
        vad_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(vad_cb, "使用Silero VAD进行语音活动检测（推荐）\n\n"
                "- 基于深度学习模型，检测精度高\n"
                "- 能区分人声vs噪音/音乐\n"
                "- 毫秒级语音边界检测\n"
                "- 显著减少无效API调用")

        ttk.Label(audio_row2, text="最小时长(ms):").pack(side=tk.LEFT, padx=8)
        self.s1ca_vad_min_speech_var = tk.StringVar(value="250")
        vad_speech_entry = ttk.Entry(audio_row2, textvariable=self.s1ca_vad_min_speech_var, width=6)
        vad_speech_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(vad_speech_entry, "VAD检测的最小时长（毫秒）\n\n"
                "- 低于此时长的语音段会被忽略\n"
                "- 推荐：250ms")

        ttk.Label(audio_row2, text="最小静音(ms):").pack(side=tk.LEFT, padx=8)
        self.s1ca_vad_min_silence_var = tk.StringVar(value="100")
        vad_silence_entry = ttk.Entry(audio_row2, textvariable=self.s1ca_vad_min_silence_var, width=6)
        vad_silence_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(vad_silence_entry, "VAD检测的最小静音时长（毫秒）\n\n"
                "- 用于合并相邻的语音段\n"
                "- 推荐：100ms")

        # 第三行：字幕后处理配置
        audio_row3 = ttk.Frame(audio_frame)
        audio_row3.pack(fill=tk.X, padx=4, pady=2)

        self.s1ca_postprocess_var = tk.BooleanVar(value=True)
        postprocess_cb = ttk.Checkbutton(audio_row3, text="字幕后处理", variable=self.s1ca_postprocess_var)
        postprocess_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(postprocess_cb, "启用字幕后处理（推荐）\n\n"
                "- 拆分长字幕为多个短字幕\n"
                "- 过滤无效内容（时间戳列表、拒绝响应等）\n"
                "- 格式化字幕文本\n"
                "- 显著提升字幕可读性")

        ttk.Label(audio_row3, text="最长时长(秒):").pack(side=tk.LEFT, padx=8)
        self.s1ca_max_duration_var = tk.StringVar(value="10")
        max_dur_entry = ttk.Entry(audio_row3, textvariable=self.s1ca_max_duration_var, width=6)
        max_dur_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(max_dur_entry, "单个字幕的最大时长（秒）\n\n"
                "- 较小值：字幕更短，阅读更轻松\n"
                "- 较大值：字幕更长，上下文更完整\n"
                "- 推荐：10秒")

        ttk.Label(audio_row3, text="最大字符数:").pack(side=tk.LEFT, padx=8)
        self.s1ca_max_chars_var = tk.StringVar(value="100")
        max_chars_entry = ttk.Entry(audio_row3, textvariable=self.s1ca_max_chars_var, width=6)
        max_chars_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(max_chars_entry, "单个字幕的最大字符数\n\n"
                "- 较小值：字幕更短，适合手机阅读\n"
                "- 较大值：字幕更长，适合大屏幕\n"
                "- 推荐：100字符")

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1ca_all_var = tk.BooleanVar()
        all_cb = ttk.Checkbutton(opt_frame, text="处理所有未识别文件", variable=self.s1ca_all_var)
        all_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(all_cb, "勾选后会处理所有audio_recognized!=true的文件\n\n"
                "- 不勾选：只处理needs_vision=TRUE的文件\n"
                "- 勾选：忽略needs_vision字段，处理所有未识别文件")

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        audio_btn = ttk.Button(btn_frame, text="音频识别", command=self._run_audio)
        audio_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(audio_btn, "对视频进行音频识别\n\n"
                "流程：\n"
                "1. VAD检测语音段/静音段\n"
                "2. 调用AI进行语音转录\n"
                "3. 字幕后处理（拆分长字幕、过滤无效内容）\n"
                "4. 生成SRT字幕文件\n"
                "5. 更新CSV中的audio_recognized和srt_path\n\n"
                "完成后，视觉识别会自动读取音频转录内容")

        save_config_btn = ttk.Button(btn_frame, text="保存配置", command=self._save_audio_config_from_gui)
        save_config_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(save_config_btn, "保存音频配置到config文件")

    def _browse_csv(self):
        """浏览CSV文件（音频识别）"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.ctx.csv_var.set(file_path)

    def _gui_sync_to_db(self, original_path: str, updates: dict, source: str = "gui"):
        """GUI 操作同步到数据库"""
        if not self.ctx.db:
            return
        try:
            media = self.ctx.db.find_by_path(original_path)
            if not media:
                data = {"original_path": original_path, "original_title": Path(original_path).name}
                data.update(updates)
                self.ctx.db.insert_media(data)
            else:
                for field, value in updates.items():
                    self.ctx.db.update_media(media["id"], field, value, source)
        except Exception as e:
            print(f"[警告] 数据库同步失败: {e}")

    def _run_audio(self):
        """运行音频识别"""
        csv_path = self.ctx.csv_var.get()
        provider = self.s1ca_provider_var.get()

        # 保存音频配置
        self._save_audio_config_from_gui()

        # 在后台线程中运行音频处理
        def run_audio_task():
            import csv
            from pathlib import Path

            try:
                # 读取CSV文件
                csv_file = Path(csv_path)
                if not csv_file.exists():
                    print(f"[错误] CSV文件不存在: {csv_path}")
                    return

                with open(csv_file, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    fieldnames = list(reader.fieldnames)
                    rows = list(reader)

                if not rows:
                    print("[警告] CSV为空")
                    return

                # 确保字段存在
                for col in ["audio_recognized", "srt_path"]:
                    if col not in fieldnames:
                        fieldnames.append(col)

                # 筛选需要处理的记录
                process_all = self.s1ca_all_var.get()
                pending = []
                for i, row in enumerate(rows):
                    if process_all:
                        if row.get("original_path", "").strip():
                            pending.append((i, row))
                    else:
                        if (row.get("needs_vision", "").strip().lower() == "true" and
                            row.get("audio_recognized", "").strip().lower() != "true"):
                            pending.append((i, row))

                print(f"共 {len(rows)} 条记录，待处理 {len(pending)} 条")

                if not pending:
                    print("[完成] 无需处理")
                    return

                # 导入AudioProcessor
                from ..utils.audio import AudioProcessor, load_audio_config
                from ..utils.file_resolve import resolve_media_path
                from datetime import datetime

                # 加载配置并创建处理器
                config = load_audio_config()
                processor = AudioProcessor(
                    provider=provider,
                    config=config,
                )

                # 处理每个文件
                srt_dir = str(csv_file.parent / "subtitles")
                Path(srt_dir).mkdir(parents=True, exist_ok=True)

                total = len(pending)
                success = 0
                failed = 0

                for idx, (row_idx, row) in enumerate(pending):
                    original_path = row.get("original_path", "").strip()
                    original_title = row.get("original_title", "").strip()
                    final_name = row.get("final_name", "").strip()

                    # 解析实际文件路径（Stage2重命名后用final_name回退查找）
                    resolved = resolve_media_path(original_path, final_name, original_title)
                    if not resolved:
                        print(f"[{idx+1}/{total}] 跳过（文件不存在）: {original_title[:40]}")
                        failed += 1
                        continue
                    if resolved != original_path:
                        print(f"  [路径回退] {Path(original_path).name} → {Path(resolved).name}")
                    original_path = resolved

                    print(f"[{idx+1}/{total}] 处理: {original_title[:40]}")

                    start_time = datetime.now()

                    try:
                        srt_name = Path(original_title).stem + ".srt"
                        srt_path = str(Path(srt_dir) / srt_name)

                        result_path = processor.process_video(
                            video_path=original_path,
                            output_srt=srt_path,
                        )

                        elapsed = (datetime.now() - start_time).total_seconds()

                        if result_path:
                            rows[row_idx]["audio_recognized"] = "true"
                            rows[row_idx]["srt_path"] = result_path
                            print(f"  [完成] {elapsed:.1f}秒")
                            print(f"  SRT: {result_path}")
                            success += 1
                            # 同步到数据库
                            self._gui_sync_to_db(original_path, {
                                "audio_recognized": True,
                                "srt_path": result_path,
                            }, "audio")
                        else:
                            print(f"  [警告] 音频识别无结果")
                            failed += 1

                    except Exception as e:
                        print(f"  [错误] {e}")
                        failed += 1

                    # 每处理完一条立即保存CSV（原子化写入）
                    from ..utils.atomic_csv import atomic_write_csv
                    atomic_write_csv(csv_file, rows, fieldnames)

                print(f"\n[统计]")
                print(f"  成功: {success}")
                print(f"  失败: {failed}")
                print(f"  结果已保存至: {csv_path}")

            except Exception as e:
                print(f"[错误] 音频处理失败: {e}")
                import traceback
                traceback.print_exc()

        # 在后台线程中执行
        thread = threading.Thread(target=run_audio_task, daemon=True)
        thread.start()

    def _save_audio_config_from_gui(self):
        """从音频识别标签页保存配置"""
        try:
            import tomllib
            config_path = PROJECT_DIR / "config" / "default.toml"

            # 读取现有配置
            config = {}
            if config_path.exists():
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)

            # 更新音频配置
            if "audio" not in config:
                config["audio"] = {}

            config["audio"]["skip_silence"] = self.s1ca_skip_silence_var.get()
            config["audio"]["volume_threshold"] = float(self.s1ca_volume_threshold_var.get())

            # VAD配置
            if "vad" not in config["audio"]:
                config["audio"]["vad"] = {}

            config["audio"]["vad"]["enabled"] = self.s1ca_vad_enabled_var.get()
            config["audio"]["vad"]["min_speech_ms"] = int(self.s1ca_vad_min_speech_var.get())
            config["audio"]["vad"]["min_silence_ms"] = int(self.s1ca_vad_min_silence_var.get())

            # 字幕后处理配置
            if "postprocess" not in config["audio"]:
                config["audio"]["postprocess"] = {}

            config["audio"]["postprocess"]["enabled"] = self.s1ca_postprocess_var.get()
            config["audio"]["postprocess"]["max_subtitle_duration"] = int(self.s1ca_max_duration_var.get())
            config["audio"]["postprocess"]["max_subtitle_chars"] = int(self.s1ca_max_chars_var.get())

            # 写入配置文件
            import tomli_w
            with open(config_path, "wb") as f:
                tomli_w.dump(config, f)

            print(f"[配置] 音频配置已保存到: {config_path}")

        except ImportError:
            print("[警告] tomli_w 未安装，无法保存配置文件")
        except Exception as e:
            print(f"[警告] 保存音频配置失败: {e}")

    def _load_audio_config_to_gui(self):
        """从config文件加载音频配置到GUI"""
        try:
            import tomllib
            config_path = PROJECT_DIR / "config" / "default.toml"

            if not config_path.exists():
                return

            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            audio_config = config.get("audio", {})
            vad_config = audio_config.get("vad", {})
            postprocess_config = audio_config.get("postprocess", {})

            if "volume_threshold" in audio_config:
                self.s1ca_volume_threshold_var.set(str(audio_config["volume_threshold"]))
            if "skip_silence" in audio_config:
                self.s1ca_skip_silence_var.set(audio_config["skip_silence"])

            # VAD配置
            if "enabled" in vad_config:
                self.s1ca_vad_enabled_var.set(vad_config["enabled"])
            if "min_speech_ms" in vad_config:
                self.s1ca_vad_min_speech_var.set(str(vad_config["min_speech_ms"]))
            if "min_silence_ms" in vad_config:
                self.s1ca_vad_min_silence_var.set(str(vad_config["min_silence_ms"]))

            # 字幕后处理配置
            if "enabled" in postprocess_config:
                self.s1ca_postprocess_var.set(postprocess_config["enabled"])
            if "max_subtitle_duration" in postprocess_config:
                self.s1ca_max_duration_var.set(str(postprocess_config["max_subtitle_duration"]))
            if "max_subtitle_chars" in postprocess_config:
                self.s1ca_max_chars_var.set(str(postprocess_config["max_subtitle_chars"]))

            print(f"[配置] 已加载音频配置: VAD={vad_config.get('enabled', True)}, 后处理={postprocess_config.get('enabled', True)}")

        except Exception as e:
            print(f"[警告] 加载音频配置失败: {e}")
