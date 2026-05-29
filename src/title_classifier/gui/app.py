"""视频标题分类工具 - 图形界面"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import sys
import logging
from pathlib import Path
from datetime import datetime

from ..providers import (
    get_available_providers, get_provider_config, get_api_key,
    check_provider_availability, get_provider_display_name,
    get_providers_for_gui, call_text_api, test_provider_connection,
)
from ..core.refiner import Refiner
from ..utils.muxer import SubtitleMuxer

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = sys.executable
DEFAULT_CSV = "data/output/title_review.csv"


class ToolTip:
    """鼠标悬停提示工具类"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Microsoft YaHei", 9),
            wraplength=350,
        )
        label.pack(ipadx=4, ipady=2)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class LogRedirector:
    """将print输出重定向到GUI文本框"""

    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, msg):
        if msg.strip():
            self.text_widget.after(0, self._append, msg)

    def _append(self, msg):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n", self.tag)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


class GUILogHandler(logging.Handler):
    """将logging输出重定向到GUI文本框"""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.after(0, self._append, msg)

    def _append(self, msg):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n", "stdout")
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")


class TitleClassifierApp(tk.Tk):
    """视频标题分类工具主窗口"""

    def __init__(self):
        super().__init__()
        self.title("视频标题分类工具 v7.2")
        self.geometry("900x850")
        self.minsize(800, 700)

        self.process = None
        self.running = False

        # 加载.env文件
        self._load_env()

        # 共享CSV路径变量
        self.csv_var = tk.StringVar(value=DEFAULT_CSV)

        self._build_ui()
        self._load_audio_config_to_gui()
        self._sync_csv()

    def _load_env(self):
        """加载.env文件"""
        env_path = PROJECT_DIR / ".env"
        if not env_path.exists():
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())
        except Exception:
            pass

    def _build_ui(self):
        """构建UI"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_stage1_tab()
        self._build_stage1b_tab()
        self._build_stage1c_audio_tab()  # 新增：音频识别标签页
        self._build_stage1c_tab()
        self._build_stage2_tab()

        # 切换标签页时同步CSV
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(log_toolbar, text="清空日志", command=self._clear_log).pack(side=tk.RIGHT)
        self.stop_btn = ttk.Button(log_toolbar, text="停止", command=self._stop_process, state="disabled")
        self.stop_btn.pack(side=tk.RIGHT, padx=4)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, state="disabled", font=("Consolas", 9), wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log_text.tag_configure("stdout", foreground="#cccccc")
        self.log_text.tag_configure("stderr", foreground="#ff6666")
        self.log_text.tag_configure("info", foreground="#66ccff")

        sys.stdout = LogRedirector(self.log_text, "stdout")
        sys.stderr = LogRedirector(self.log_text, "stderr")

        # 将logging也重定向到GUI日志框
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logging.getLogger().addHandler(gui_handler)

    def _sync_csv(self):
        """同步所有标签页的CSV路径"""
        csv = self.csv_var.get()
        self.s1_output_var.set(csv)
        self.s1b_csv_var.set(csv)
        self.s1ca_csv_var.set(csv)
        self.s1c_csv_var.set(csv)
        self.s2_csv_var.set(csv)

    def _on_tab_changed(self, event=None):
        """切换标签页时同步"""
        tab = self.notebook.select()
        tab_name = self.notebook.tab(tab, "text")
        if "Stage1 " in tab_name:
            current = self.s1_output_var.get()
        elif "Stage1b" in tab_name:
            current = self.s1b_csv_var.get()
        elif "音频识别" in tab_name:
            current = self.s1ca_csv_var.get()
        elif "视觉识别" in tab_name:
            current = self.s1c_csv_var.get()
        elif "Stage2" in tab_name:
            current = self.s2_csv_var.get()
        else:
            return
        self.csv_var.set(current)
        self._sync_csv()

    def _build_stage1_tab(self):
        """构建Stage1扫描标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1 扫描")

        # 目录/文件选择
        dir_frame = ttk.LabelFrame(tab, text="扫描目标")
        dir_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_dir_var = tk.StringVar()
        dir_entry = ttk.Entry(dir_frame, textvariable=self.s1_dir_var, width=60)
        dir_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(dir_frame, text="浏览目录...", command=self._browse_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_frame, text="选择文件...", command=self._browse_file).pack(side=tk.LEFT, padx=2)
        ToolTip(dir_entry, "选择要扫描的视频/图片目录或单个文件\n\n"
                "- 选择目录：递归扫描所有子目录\n"
                "- 选择文件：只处理选中的单个文件")

        # 输出文件
        out_frame = ttk.LabelFrame(tab, text="输出文件")
        out_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_output_var = tk.StringVar(value=DEFAULT_CSV)
        out_entry = ttk.Entry(out_frame, textvariable=self.s1_output_var, width=60)
        out_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(out_entry, "扫描结果保存的CSV文件路径")

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1_append_var = tk.BooleanVar()
        append_cb = ttk.Checkbutton(opt_frame, text="追加模式", variable=self.s1_append_var)
        append_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(append_cb, "勾选后新扫描结果追加到现有CSV文件，否则覆盖")

        self.s1_force_var = tk.BooleanVar()
        force_cb = ttk.Checkbutton(opt_frame, text="强制重新分类", variable=self.s1_force_var)
        force_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(force_cb, "勾选后即使文件已有分类标签也会重新处理")

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        scan_btn = ttk.Button(btn_frame, text="开始扫描", command=self._run_scan)
        scan_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(scan_btn, "扫描目录中的媒体文件，提取关键词，生成待审CSV表")

    def _build_stage1b_tab(self):
        """构建Stage1b AI优化标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1b AI优化")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1b_csv_var = tk.StringVar(value=DEFAULT_CSV)
        csv_entry = ttk.Entry(csv_frame, textvariable=self.s1b_csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s1b).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "Stage1生成的CSV文件，AI会优化其中的标题")

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1b_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1b")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.s1b_provider_var, values=providers, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(provider_combo, "选择AI服务提供商\n- gcli: Google Gemini（推荐）\n- zhipu: 智谱GLM\n- ollama: 本地模型")

        # 过滤选项
        filter_frame = ttk.LabelFrame(tab, text="过滤选项")
        filter_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1b_filter_vision_var = tk.BooleanVar(value=True)
        filter_cb = ttk.Checkbutton(filter_frame, text="只加载 needs_vision=FALSE 的行", variable=self.s1b_filter_vision_var)
        filter_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(filter_cb, "勾选后只加载不需要视觉识别的行\n\n"
                "- needs_vision=FALSE：文件名有意义，可用AI优化标题\n"
                "- needs_vision=TRUE：文件名无意义，需要视觉识别")

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        load_btn = ttk.Button(btn_frame, text="加载CSV", command=self._load_s1b_preview)
        load_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(load_btn, "加载CSV文件到预览表格\n\n"
                "- 勾选过滤：只加载needs_vision=FALSE的行\n"
                "- 不勾选：加载所有行")

        use_original_btn = ttk.Button(btn_frame, text="填入原标题", command=self._s1b_fill_original_selected)
        use_original_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(use_original_btn, "将选中行的原标题直接填入final_name\n\n"
                "跳过AI优化，直接使用原标题作为最终文件名")

        use_original_all_btn = ttk.Button(btn_frame, text="全部填入原标题", command=self._s1b_fill_original_all)
        use_original_all_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(use_original_all_btn, "将所有行的原标题直接填入final_name\n\n"
                "跳过AI优化，直接使用原标题作为最终文件名")

        refine_btn = ttk.Button(btn_frame, text="AI优化选中行", command=self._run_refine_selected)
        refine_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(refine_btn, "对选中的行进行AI标题优化\n\n"
                "操作步骤：\n"
                "1. 在表格中选择要优化的行（可多选）\n"
                "2. 点击此按钮进行AI优化\n"
                "3. 优化结果会显示在'AI优化结果'列")

        refine_all_btn = ttk.Button(btn_frame, text="AI优化全部", command=self._run_refine_all)
        refine_all_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(refine_all_btn, "对表格中所有行进行AI标题优化")

        confirm_btn = ttk.Button(btn_frame, text="确认写入CSV", command=self._confirm_s1b_results)
        confirm_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(confirm_btn, "将已修改的标题写入CSV文件\n\n"
                "只写入通过以下方式修改过的行：\n"
                "- AI优化选中行/全部\n"
                "- 双击编辑标题\n"
                "- 右键采用原标题\n\n"
                "写入格式: [关键词]_原标题\n"
                "未修改的行不会被影响\n\n"
                "注意：needs_vision/audio 切换会即时写入，不需要点此按钮")

        # 预览表格
        preview_frame = ttk.LabelFrame(tab, text="优化结果预览（右键菜单可编辑）")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 创建树形视图（多选模式）- 添加needs_vision和audio_recognized列
        columns = ("original", "needs_vision", "audio_recognized", "refined")
        self.s1b_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", selectmode="extended")
        self.s1b_tree.heading("original", text="原始标题")
        self.s1b_tree.heading("needs_vision", text="需要视觉识别")
        self.s1b_tree.heading("audio_recognized", text="音频已识别")
        self.s1b_tree.heading("refined", text="AI优化结果")
        self.s1b_tree.column("original", width=200)
        self.s1b_tree.column("needs_vision", width=80, anchor="center")
        self.s1b_tree.column("audio_recognized", width=80, anchor="center")
        self.s1b_tree.column("refined", width=300)

        # 滚动条
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.s1b_tree.yview)
        self.s1b_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.s1b_tree.pack(fill=tk.BOTH, expand=True)

        # 右键菜单（按功能分组）
        self.s1b_context_menu = tk.Menu(self, tearoff=0)

        # 标题操作组
        self.s1b_context_menu.add_command(label="编辑标题", command=self._s1b_edit)
        self.s1b_context_menu.add_command(label="采用原标题", command=self._s1b_use_original)
        self.s1b_context_menu.add_command(label="重置为原标题（取消优化）", command=self._s1b_reset_to_original)

        self.s1b_context_menu.add_separator()

        # 状态切换组（即时生效）- 使用占位标签，show 时动态更新
        self.s1b_ctx_vision_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="需要视觉识别: -", command=self._s1b_toggle_needs_vision)
        self.s1b_ctx_audio_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="音频已识别: -", command=self._s1b_toggle_audio_recognized)

        self.s1b_context_menu.add_separator()

        # 其他
        self.s1b_context_menu.add_command(label="删除选中行", command=self._s1b_delete)

        self.s1b_tree.bind("<Button-3>", self._s1b_show_context_menu)
        self.s1b_tree.bind("<Double-1>", self._s1b_edit)

        # 存储优化结果
        self.s1b_results = {}
        self.s1b_modified = set()  # 跟踪已修改的 item_id

        # 修改行高亮样式
        self.s1b_tree.tag_configure("modified", background="#e6f3ff")

    def _build_stage1c_audio_tab(self):
        """构建Stage1c音频识别标签页（为视觉识别做准备）"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1c 音频识别")

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

        self.s1ca_csv_var = tk.StringVar(value=DEFAULT_CSV)
        csv_entry = ttk.Entry(csv_frame, textvariable=self.s1ca_csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s1ca).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "Stage1生成的CSV文件，音频识别会处理needs_vision=TRUE的行")

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1ca_provider_var = tk.StringVar(value="mimo")
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

    def _build_stage1c_tab(self):
        """构建Stage1c视觉识别标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage1c 视觉识别")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_csv_var = tk.StringVar(value=DEFAULT_CSV)
        csv_entry = ttk.Entry(csv_frame, textvariable=self.s1c_csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s1c).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "Stage1生成的CSV文件，视觉识别会分析视频内容并生成描述和关键词")

        # Provider选择
        provider_frame = ttk.LabelFrame(tab, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1c")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.s1c_provider_var, values=providers, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(provider_combo, "选择视觉AI服务提供商\n- gcli: Google Gemini（推荐）\n- mimo: 小米MiMo\n- zhipu: 智谱GLM")

        # 检测器选择
        det_frame = ttk.LabelFrame(tab, text="检测器")
        det_frame.pack(fill=tk.X, padx=4, pady=4)

        # YOLO检测器说明
        yolo_label = ttk.Label(det_frame, text="YOLO姿态检测（默认）")
        yolo_label.pack(side=tk.LEFT, padx=4)
        ToolTip(yolo_label, "使用YOLO Pose模型检测人体姿态\n\n"
                "功能：\n"
                "- 检测视频中是否有人体\n"
                "- 分析人体姿态（站立、跪姿、弯腰等）\n"
                "- 智能选择包含人体的代表性帧\n"
                "- 将姿态信息作为上下文传给VLM")

        # 全面分析模式选项
        self.s1c_comprehensive_var = tk.BooleanVar(value=False)
        comprehensive_cb = ttk.Checkbutton(det_frame, text="全面分析模式", variable=self.s1c_comprehensive_var)
        comprehensive_cb.pack(side=tk.LEFT, padx=8)
        ToolTip(comprehensive_cb, "使用三个YOLO模型进行全面分析\n\n"
                "功能：\n"
                "- 同时使用detect、pose、segment三个模型\n"
                "- 投票决策：至少两个模型检测到人体才认为有人体\n"
                "- 动态权重：根据置信度自动调整模型权重\n"
                "- 提供姿态分析、穿着分割等详细信息\n\n"
                "适用场景：需要更全面、准确的视频分析时使用\n"
                "注意：全面分析模式会使用更多GPU内存和时间")

        # CLIP选项
        self.s1c_use_clip_var = tk.BooleanVar()
        clip_cb = ttk.Checkbutton(det_frame, text="使用CLIP预分类", variable=self.s1c_use_clip_var)
        clip_cb.pack(side=tk.LEFT, padx=8)
        ToolTip(clip_cb, "使用CLIP模型进行图像预分类\n\n"
                "功能：\n"
                "- 快速识别图片内容类别\n"
                "- 如果置信度足够高，可跳过VLM调用\n\n"
                "适用场景：大量图片需要快速分类时使用")

        # 分析参数
        param_frame = ttk.LabelFrame(tab, text="分析参数")
        param_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(param_frame, text="采样间隔(秒):").pack(side=tk.LEFT, padx=4)
        self.s1c_analysis_step_var = tk.StringVar(value="2.0")
        step_entry = ttk.Entry(param_frame, textvariable=self.s1c_analysis_step_var, width=6)
        step_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(step_entry, "视频采样间隔\n\n"
                "- 默认2秒取一帧进行分析\n"
                "- 较小值：分析更细致，但耗时更长\n"
                "- 较大值：分析更快，但可能遗漏细节\n\n"
                "108秒视频，间隔2秒 = 约54帧（自动限制最多50帧）")

        ttk.Label(param_frame, text="VLM帧数:").pack(side=tk.LEFT, padx=8)
        self.s1c_vlm_frames_var = tk.StringVar(value="10")
        frames_entry = ttk.Entry(param_frame, textvariable=self.s1c_vlm_frames_var, width=6)
        frames_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(frames_entry, "传给VLM分析的帧数\n\n"
                "此参数由采样间隔决定，通常不需要手动设置")

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_all_var = tk.BooleanVar()
        all_cb = ttk.Checkbutton(opt_frame, text="处理所有未识别文件", variable=self.s1c_all_var)
        all_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(all_cb, "勾选后会处理所有vision_keywords为空的文件\n\n"
                "- 不勾选：只处理needs_vision=TRUE的文件\n"
                "- 勾选：忽略needs_vision字段，处理所有未识别文件")

        self.s1c_debug_var = tk.BooleanVar()
        debug_cb = ttk.Checkbutton(opt_frame, text="启用调试", variable=self.s1c_debug_var)
        debug_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(debug_cb, "启用调试模式，保存检测结果和VLM输入输出\n\n"
                "- 保存每帧的检测结果（原始帧+标注帧+JSON）\n"
                "- 保存VLM输入帧和Prompt\n"
                "- 保存VLM响应\n"
                "- 处理完成后自动打开调试窗口")

        # 废弃提醒
        deprecation_frame = ttk.Frame(tab)
        deprecation_frame.pack(fill=tk.X, padx=4, pady=4)
        deprecation_label = ttk.Label(
            deprecation_frame,
            text="注意：音频识别功能已转移到独立的 'Stage1c 音频识别' 标签页",
            foreground="red",
            font=("Microsoft YaHei", 9, "bold")
        )
        deprecation_label.pack(side=tk.LEFT, padx=4)
        ToolTip(deprecation_label, "vision命令的--audio参数已废弃\n\n"
                "音频识别现在由独立的audio子命令提供\n"
                "请使用 'Stage1c 音频识别' 标签页进行音频识别")

        # 字幕封装选项
        mux_frame = ttk.LabelFrame(tab, text="字幕封装")
        mux_frame.pack(fill=tk.X, padx=4, pady=4)

        # 提示信息
        mux_tip = ttk.Label(
            mux_frame,
            text="注意：封装字幕需要先运行音频识别，产出字幕文件",
            foreground="blue",
            font=("Microsoft YaHei", 8)
        )
        mux_tip.pack(fill=tk.X, padx=4, pady=2)

        # 第一行：封装开关和输出格式
        mux_row1 = ttk.Frame(mux_frame)
        mux_row1.pack(fill=tk.X, padx=4, pady=2)

        self.s1c_mux_enabled_var = tk.BooleanVar(value=False)
        mux_cb = ttk.Checkbutton(mux_row1, text="启用字幕封装", variable=self.s1c_mux_enabled_var)
        mux_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(mux_cb, "在视觉识别后自动将字幕封装到视频中\n\n"
                "需要先运行音频识别生成字幕文件\n"
                "封装后的视频会保存在原目录")

        ttk.Label(mux_row1, text="输出格式:").pack(side=tk.LEFT, padx=8)
        self.s1c_mux_format_var = tk.StringVar(value="auto")
        format_combo = ttk.Combobox(mux_row1, textvariable=self.s1c_mux_format_var, 
                                   values=["auto", "mkv", "mp4"], state="readonly", width=8)
        format_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(format_combo, "选择输出视频格式\n\n"
                "- auto: 保持原视频格式\n"
                "- mkv: MKV容器（推荐，支持SRT无损封装）\n"
                "- mp4: MP4容器（SRT会转为mov_text格式）")

        # 第二行：文件处理和字幕处理
        mux_row2 = ttk.Frame(mux_frame)
        mux_row2.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(mux_row2, text="文件处理:").pack(side=tk.LEFT, padx=4)
        self.s1c_mux_handling_var = tk.StringVar(value="new")
        handling_combo = ttk.Combobox(mux_row2, textvariable=self.s1c_mux_handling_var,
                                     values=["new", "overwrite"], state="readonly", width=10)
        handling_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(handling_combo, "选择文件处理方式\n\n"
                "- new: 创建新文件（原文件名_muxed.扩展名）\n"
                "- overwrite: 覆盖原文件（谨慎使用）")

        ttk.Label(mux_row2, text="字幕处理:").pack(side=tk.LEFT, padx=8)
        self.s1c_mux_processing_var = tk.StringVar(value="direct")
        processing_combo = ttk.Combobox(mux_row2, textvariable=self.s1c_mux_processing_var,
                                       values=["direct", "convert"], state="readonly", width=8)
        processing_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(processing_combo, "选择字幕处理方式\n\n"
                "- direct: 直接封装SRT文件\n"
                "- convert: 转换为UTF-8编码后封装")

        # 第三行：封装按钮和重试按钮
        mux_row3 = ttk.Frame(mux_frame)
        mux_row3.pack(fill=tk.X, padx=4, pady=4)

        mux_btn = ttk.Button(mux_row3, text="封装字幕", command=self._run_mux_subtitle)
        mux_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(mux_btn, "将字幕封装到视频中\n\n"
                "操作步骤：\n"
                "1. 确保已运行音频识别生成字幕文件\n"
                "2. 确保已运行视觉识别生成final_name\n"
                "3. 点击此按钮执行封装")

        retry_btn = ttk.Button(mux_row3, text="重试失败", command=self._retry_failed_mux)
        retry_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(retry_btn, "重试之前失败的封装操作\n\n"
                "如果封装过程中有文件失败，\n"
                "可以点击此按钮重新尝试")

        # 进度条
        self.s1c_mux_progress_var = tk.DoubleVar(value=0.0)
        mux_progress = ttk.Progressbar(mux_frame, variable=self.s1c_mux_progress_var, maximum=100)
        mux_progress.pack(fill=tk.X, padx=4, pady=2)

        # 状态标签
        self.s1c_mux_status_var = tk.StringVar(value="就绪")
        mux_status = ttk.Label(mux_frame, textvariable=self.s1c_mux_status_var)
        mux_status.pack(fill=tk.X, padx=4, pady=2)

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        vision_btn = ttk.Button(btn_frame, text="视觉识别", command=self._run_vision)
        vision_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(vision_btn, "对视频进行视觉分析\n\n"
                "YOLO模式流程：\n"
                "1. 每2秒提取一帧\n"
                "2. 用YOLO Pose分析每帧姿态\n"
                "3. 智能选择代表性帧\n"
                "4. 将帧图片+姿态信息传给VLM\n"
                "5. 生成描述、关键词、final_name\n"
                "6. 生成SRT元数据文件")

    def _build_stage2_tab(self):
        """构建Stage2重命名标签页"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Stage2 重命名")

        # CSV文件
        csv_frame = ttk.LabelFrame(tab, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s2_csv_var = tk.StringVar(value=DEFAULT_CSV)
        csv_entry = ttk.Entry(csv_frame, textvariable=self.s2_csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv_s2).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "包含final_name的CSV文件，用于批量重命名")

        # 批量操作
        batch_frame = ttk.LabelFrame(tab, text="批量操作")
        batch_frame.pack(fill=tk.X, padx=4, pady=4)

        confirm_btn = ttk.Button(batch_frame, text="一键确认所有记录", command=self._batch_confirm)
        confirm_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(confirm_btn, "将CSV中所有记录的review_status设置为'已确认'\n\n"
                "只有review_status='已确认'的记录才会被重命名")

        clear_btn = ttk.Button(batch_frame, text="一键清空final_name", command=self._batch_clear_final_name)
        clear_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(clear_btn, "清空CSV中所有记录的final_name字段\n\n"
                "用于重置优化结果，重新开始处理")

        # 选项
        opt_frame = ttk.LabelFrame(tab, text="选项")
        opt_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s2_dry_run_var = tk.BooleanVar(value=True)
        dry_run_cb = ttk.Checkbutton(opt_frame, text="模拟运行", variable=self.s2_dry_run_var)
        dry_run_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(dry_run_cb, "勾选后只显示重命名预览，不实际修改文件\n\n"
                "- 勾选：安全模式，只预览不执行\n"
                "- 取消勾选：实际执行重命名操作")

        # 执行按钮
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        rename_btn = ttk.Button(btn_frame, text="执行重命名", command=self._run_rename)
        rename_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(rename_btn, "根据CSV中的final_name批量重命名文件\n\n"
                "重命名规则：\n"
                "- 只处理review_status='已确认'的记录\n"
                "- 新文件名 = final_name + 原扩展名\n"
                "- 如果目标文件已存在，自动添加序号")

    # ==================== 文件浏览 ====================

    def _browse_dir(self):
        """浏览目录"""
        dir_path = filedialog.askdirectory(title="选择扫描目录")
        if dir_path:
            self.s1_dir_var.set(dir_path)

    def _browse_file(self):
        """浏览文件"""
        filetypes = [
            ("媒体文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.ts *.jpg *.jpeg *.png *.bmp *.webp"),
            ("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.ts"),
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*"),
        ]
        file_path = filedialog.askopenfilename(title="选择媒体文件", filetypes=filetypes)
        if file_path:
            self.s1_dir_var.set(file_path)

    def _browse_csv_s1b(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1b_csv_var.set(file_path)

    def _browse_csv_s1c(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1c_csv_var.set(file_path)

    def _browse_csv_s1ca(self):
        """浏览CSV文件（音频识别）"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1ca_csv_var.set(file_path)

    def _browse_csv_s2(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s2_csv_var.set(file_path)

    # ==================== 命令执行 ====================

    def _run_command(self, cmd, callback=None):
        """运行命令"""
        if self.running:
            messagebox.showwarning("警告", "有命令正在运行")
            return

        self.running = True
        self.stop_btn.configure(state="normal")

        def run():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in self.process.stdout:
                    print(line.rstrip())
                self.process.wait()
                if self.process.returncode == 0:
                    print("\n[完成] 命令执行成功")
                else:
                    print(f"\n[错误] 命令执行失败，返回码: {self.process.returncode}")
            except Exception as e:
                print(f"\n[错误] {e}")
            finally:
                self.running = False
                self.stop_btn.configure(state="disabled")
                if callback:
                    callback()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _stop_process(self):
        """停止进程"""
        if self.process:
            self.process.terminate()
            self.process = None
            self.running = False
            self.stop_btn.configure(state="disabled")
            print("[停止] 命令已终止")

    def _clear_log(self):
        """清空日志"""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

    def _run_scan(self):
        """运行扫描"""
        dir_path = self.s1_dir_var.get()
        if not dir_path:
            messagebox.showwarning("警告", "请选择扫描目录")
            return

        output = self.s1_output_var.get()
        cmd = [PYTHON, "-m", "title_classifier", "scan", "-d", dir_path, "-o", output]

        if self.s1_append_var.get():
            cmd.append("-a")
        if self.s1_force_var.get():
            cmd.append("--force")

        self._run_command(cmd)

    def _load_s1b_preview(self):
        """加载CSV到预览表格"""
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        try:
            import csv
            # 清空表格
            for item in self.s1b_tree.get_children():
                self.s1b_tree.delete(item)
            self.s1b_results.clear()
            self.s1b_modified.clear()

            # 是否过滤needs_vision
            filter_vision = self.s1b_filter_vision_var.get()

            # 读取CSV
            loaded_count = 0
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    needs_vision = row.get("needs_vision", "").strip().lower()
                    
                    # 如果启用过滤，只加载needs_vision=FALSE的行
                    if filter_vision and needs_vision == "true":
                        continue

                    original = row.get("original_title", "")
                    final = row.get("final_name", "")
                    audio_recognized = row.get("audio_recognized", "").strip().lower()
                    
                    # 插入到表格
                    item_id = self.s1b_tree.insert("", tk.END, values=(
                        original, 
                        needs_vision.upper(), 
                        audio_recognized.upper() if audio_recognized else "FALSE",
                        final
                    ))
                    self.s1b_results[item_id] = {
                        "row_index": i,
                        "original_title": original,
                        "final_name": final,
                        "needs_vision": needs_vision,
                        "audio_recognized": audio_recognized,
                    }
                    loaded_count += 1

            filter_desc = "（已过滤needs_vision=TRUE）" if filter_vision else ""
            print(f"[完成] 加载 {loaded_count} 条记录{filter_desc}")

        except Exception as e:
            print(f"[错误] 加载CSV失败: {e}")

    def _run_refine_selected(self):
        """对选中的行进行AI优化"""
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要优化的行")
            return

        provider = self.s1b_provider_var.get()

        # 收集选中行的原始标题（去除扩展名后发送给AI）
        items_to_refine = []
        for item_id in selected:
            values = self.s1b_tree.item(item_id, "values")
            original = values[0]
            needs_vision = values[1]  # keep needs_vision
            # 去除扩展名后发送给AI
            stem = Path(original).stem
            items_to_refine.append((item_id, original, needs_vision, stem))

        total = len(items_to_refine)
        print(f"[开始] 正在优化 {total} 条记录（每5条一批）...")

        # 禁用按钮防止重复点击
        self._set_refine_buttons_state("disabled")

        def run_refine():
            try:
                refiner = Refiner(provider=provider)

                # 定义进度回调
                def on_progress(current, total_count, title):
                    batch_num = current // 5 + 1
                    total_batches = (total_count + 4) // 5
                    self.after(0, lambda: print(f"  [批次 {batch_num}/{total_batches}] 正在处理: {title[:40]}..."))

                # 批量优化（发送去除扩展名的标题）
                stems = [t[3] for t in items_to_refine]
                refined_stems = refiner.refine_batch(stems, progress_callback=on_progress)

                # 在主线程中更新GUI
                def update_gui():
                    for (item_id, original, needs_vision, _), refined in zip(items_to_refine, refined_stems):
                        # 保留 audio_recognized 列
                        audio_recognized = self.s1b_tree.item(item_id, "values")[2]
                        self.s1b_tree.item(item_id, values=(original, needs_vision, audio_recognized, refined))
                        if item_id in self.s1b_results:
                            self.s1b_results[item_id]["final_name"] = refined
                        self._mark_modified(item_id)
                    print(f"[完成] 已优化 {total} 条记录")
                    self._set_refine_buttons_state("normal")

                self.after(0, update_gui)

            except Exception as e:
                def show_error():
                    print(f"[错误] AI优化失败: {e}")
                    messagebox.showerror("错误", f"AI优化失败: {e}")
                    self._set_refine_buttons_state("normal")
                self.after(0, show_error)

        # 在后台线程中执行
        threading.Thread(target=run_refine, daemon=True).start()

    def _run_refine_all(self):
        """对所有行进行AI优化"""
        all_items = self.s1b_tree.get_children()
        if not all_items:
            messagebox.showwarning("警告", "表格为空，请先加载CSV")
            return

        total = len(all_items)
        if not messagebox.askyesno("确认", f"确定要优化所有 {total} 条记录吗？"):
            return

        provider = self.s1b_provider_var.get()

        # 收集所有行的原始标题（去除扩展名后发送给AI）
        items_to_refine = []
        for item_id in all_items:
            values = self.s1b_tree.item(item_id, "values")
            original = values[0]
            needs_vision = values[1]  # keep needs_vision
            # 去除扩展名后发送给AI
            stem = Path(original).stem
            items_to_refine.append((item_id, original, needs_vision, stem))

        print(f"[开始] 正在优化 {total} 条记录（每5条一批）...")

        # 禁用按钮防止重复点击
        self._set_refine_buttons_state("disabled")

        def run_refine():
            try:
                refiner = Refiner(provider=provider)

                # 定义进度回调
                def on_progress(current, total_count, title):
                    batch_num = current // 5 + 1
                    total_batches = (total_count + 4) // 5
                    self.after(0, lambda: print(f"  [批次 {batch_num}/{total_batches}] 正在处理: {title[:40]}..."))

                # 批量优化（发送去除扩展名的标题）
                stems = [t[3] for t in items_to_refine]
                refined_stems = refiner.refine_batch(stems, progress_callback=on_progress)

                # 在主线程中更新GUI
                def update_gui():
                    for (item_id, original, needs_vision, _), refined in zip(items_to_refine, refined_stems):
                        # 保留 audio_recognized 列
                        audio_recognized = self.s1b_tree.item(item_id, "values")[2]
                        self.s1b_tree.item(item_id, values=(original, needs_vision, audio_recognized, refined))
                        if item_id in self.s1b_results:
                            self.s1b_results[item_id]["final_name"] = refined
                        self._mark_modified(item_id)
                    print(f"[完成] 已优化 {total} 条记录")
                    self._set_refine_buttons_state("normal")

                self.after(0, update_gui)

            except Exception as e:
                def show_error():
                    print(f"[错误] AI优化失败: {e}")
                    messagebox.showerror("错误", f"AI优化失败: {e}")
                    self._set_refine_buttons_state("normal")
                self.after(0, show_error)

        # 在后台线程中执行
        threading.Thread(target=run_refine, daemon=True).start()

    def _set_refine_buttons_state(self, state):
        """设置优化按钮的状态"""
        # 遍历Stage1b标签页中的所有按钮，设置其状态
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Notebook):
                for tab in widget.winfo_children():
                    for child in tab.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for btn in child.winfo_children():
                                if isinstance(btn, ttk.Button) and btn.cget("text") in ["AI优化选中行", "AI优化全部"]:
                                    btn.configure(state=state)

    def _mark_modified(self, item_id):
        """标记行为已修改（高亮 + 加入 modified 集合）"""
        self.s1b_modified.add(item_id)
        self.s1b_tree.item(item_id, tags=("modified",))

    def _s1b_show_context_menu(self, event):
        """显示右键菜单（动态标签）"""
        item = self.s1b_tree.identify_row(event.y)
        if not item:
            return
        self.s1b_tree.selection_set(item)
        values = self.s1b_tree.item(item, "values")

        # 动态更新状态切换标签
        nv = values[1]  # needs_vision
        ar = values[2]  # audio_recognized
        nv_next = "FALSE" if nv.upper() == "TRUE" else "TRUE"
        ar_next = "FALSE" if ar.upper() == "TRUE" else "TRUE"
        self.s1b_context_menu.entryconfigure(self.s1b_ctx_vision_idx, label=f"需要视觉识别: {nv} → {nv_next}")
        self.s1b_context_menu.entryconfigure(self.s1b_ctx_audio_idx, label=f"音频已识别: {ar} → {ar_next}")

        self.s1b_context_menu.post(event.x_root, event.y_root)

    def _s1b_edit(self, event=None):
        """编辑选中项"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        
        # 弹出编辑对话框
        dialog = tk.Toplevel(self)
        dialog.title("编辑标题")
        dialog.geometry("500x150")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="原始标题:").grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Label(dialog, text=values[0][:50]).grid(row=0, column=1, padx=4, pady=4, sticky=tk.W)

        ttk.Label(dialog, text="优化结果:").grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)
        edit_var = tk.StringVar(value=values[2])  # index 2 is refined
        ttk.Entry(dialog, textvariable=edit_var, width=50).grid(row=1, column=1, padx=4, pady=4)

        def confirm():
            new_value = edit_var.get()
            self.s1b_tree.item(item, values=(values[0], values[1], values[2], new_value))  # keep all columns
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = new_value
            self._mark_modified(item)
            dialog.destroy()

        ttk.Button(dialog, text="确认", command=confirm).grid(row=2, column=0, columnspan=2, pady=8)

    def _s1b_use_original(self):
        """采用原标题（单条）"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        original = values[0]
        needs_vision = values[1]  # keep needs_vision
        audio_recognized = values[2]  # keep audio_recognized
        
        self.s1b_tree.item(item, values=(original, needs_vision, audio_recognized, original))
        if item in self.s1b_results:
            self.s1b_results[item]["final_name"] = original
        self._mark_modified(item)

    def _s1b_reset_to_original(self):
        """重置为原标题（取消优化，不标记为修改）"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        for item in selected:
            values = self.s1b_tree.item(item, "values")
            original = values[0]
            needs_vision = values[1]
            audio_recognized = values[2]
            self.s1b_tree.item(item, values=(original, needs_vision, audio_recognized, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            # 从 modified 中移除（取消优化）
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())

    def _s1b_toggle_needs_vision(self):
        """切换选中行的needs_vision值（即时写入CSV）"""
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[1].upper()  # needs_vision is at index 1
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            self.s1b_tree.item(item, values=(values[0], new_value, values[2], values[3]))
            if item in self.s1b_results:
                self.s1b_results[item]["needs_vision"] = new_value.lower()
            count += 1

        # 即时写入CSV
        self._s1b_write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的needs_vision值（已写入CSV）")

    def _s1b_toggle_audio_recognized(self):
        """切换选中行的audio_recognized值（即时写入CSV）"""
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[2].upper()  # audio_recognized is at index 2
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            self.s1b_tree.item(item, values=(values[0], values[1], new_value, values[3]))
            if item in self.s1b_results:
                self.s1b_results[item]["audio_recognized"] = new_value.lower()
            count += 1

        # 即时写入CSV
        self._s1b_write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的audio_recognized值（已写入CSV）")

    def _s1b_write_toggles_to_csv(self):
        """将 needs_vision/audio_recognized 的修改即时写入 CSV"""
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            return

        try:
            from ..utils.atomic_csv import atomic_write_csv, safe_read_csv
            fieldnames, rows = safe_read_csv(csv_path)
            if not rows:
                return

            for item_id, result in self.s1b_results.items():
                row_index = result["row_index"]
                if row_index < len(rows):
                    nv = result.get("needs_vision", "")
                    ar = result.get("audio_recognized", "")
                    if nv:
                        rows[row_index]["needs_vision"] = nv
                    if ar:
                        rows[row_index]["audio_recognized"] = ar

            atomic_write_csv(csv_path, rows, fieldnames)
        except Exception as e:
            print(f"[错误] 写入状态失败: {e}")

    def _s1b_fill_original_selected(self):
        """将选中行的原标题填入final_name（不标记为修改）"""
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要填入的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            original = values[0]
            needs_vision = values[1]
            audio_recognized = values[2]
            self.s1b_tree.item(item, values=(original, needs_vision, audio_recognized, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            # 从 modified 中移除（批量重置）
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _s1b_fill_original_all(self):
        """将所有行的原标题填入final_name（不标记为修改）"""
        all_items = self.s1b_tree.get_children()
        if not all_items:
            messagebox.showwarning("警告", "表格为空")
            return

        if not messagebox.askyesno("确认", f"确定要将所有 {len(all_items)} 条记录的final_name设为原标题吗？"):
            return

        count = 0
        for item in all_items:
            values = self.s1b_tree.item(item, "values")
            original = values[0]
            needs_vision = values[1]
            audio_recognized = values[2]
            self.s1b_tree.item(item, values=(original, needs_vision, audio_recognized, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            # 从 modified 中移除（批量重置）
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _s1b_delete(self):
        """删除选中项"""
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        self.s1b_tree.delete(item)
        if item in self.s1b_results:
            del self.s1b_results[item]
        self.s1b_modified.discard(item)

    def _confirm_s1b_results(self):
        """确认写入优化结果到CSV（只写已修改行的 final_name）"""
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not self.s1b_modified:
            messagebox.showwarning("警告", "没有已修改的行可写入\n\n"
                                 "提示：只有通过AI优化、编辑、采用原标题 修改过的行才会被写入")
            return

        count = len(self.s1b_modified)
        if not messagebox.askyesno("确认", f"确定要将 {count} 条已修改的标题写入CSV吗？\n\n"
                                   "写入格式: [关键词]_原标题\n"
                                   "注意：未修改的行不会被影响"):
            return

        try:
            from ..utils.atomic_csv import atomic_write_csv, safe_read_csv
            fieldnames, rows = safe_read_csv(csv_path)
            if not rows:
                return

            # 只处理已修改的行
            updated = 0
            for item_id in self.s1b_modified:
                if item_id not in self.s1b_results:
                    continue
                result = self.s1b_results[item_id]
                row_index = result["row_index"]
                if row_index < len(rows):
                    final_name = result["final_name"]
                    # 自动加中括号
                    if final_name and not final_name.startswith("["):
                        final_name = f"[{final_name}]"
                    rows[row_index]["final_name"] = final_name
                    updated += 1

            # 保存CSV（原子化写入）
            atomic_write_csv(csv_path, rows, fieldnames)

            # 清空 modified 集合
            self.s1b_modified.clear()
            # 清除高亮
            for item_id in self.s1b_tree.get_children():
                self.s1b_tree.item(item_id, tags=())

            print(f"[完成] 已更新 {updated} 条记录")
            messagebox.showinfo("完成", f"已更新 {updated} 条记录")

        except Exception as e:
            print(f"[错误] 写入CSV失败: {e}")
            messagebox.showerror("错误", str(e))

    def _run_audio(self):
        """运行音频识别"""
        csv_path = self.s1ca_csv_var.get()
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

                    if not original_path or not Path(original_path).exists():
                        print(f"[{idx+1}/{total}] 跳过（文件不存在）: {original_title[:40]}")
                        failed += 1
                        continue

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

    def _run_vision(self):
        """运行视觉识别"""
        csv = self.s1c_csv_var.get()
        provider = self.s1c_provider_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "vision", "-c", csv, "-p", provider]

        # 始终使用YOLO
        cmd.append("--use-yolo")
        
        # 全面分析模式
        if self.s1c_comprehensive_var.get():
            cmd.append("--comprehensive")

        if self.s1c_use_clip_var.get():
            cmd.append("--use-clip")

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        if self.s1c_all_var.get():
            cmd.append("--all")

        # 调试模式
        if self.s1c_debug_var.get():
            cmd.append("--debug")
            self._debug_enabled = True
        else:
            self._debug_enabled = False

        # 定义完成回调，用于打开调试窗口
        def on_vision_complete():
            if getattr(self, '_debug_enabled', False):
                self._open_latest_debug_dir()

        self._run_command(cmd, callback=on_vision_complete)

    def _open_latest_debug_dir(self):
        """打开最新的调试目录"""
        debug_dir = PROJECT_DIR / "data" / "debug"
        if not debug_dir.exists():
            print("[调试] 未找到调试目录")
            return

        # 查找最新的子目录
        subdirs = sorted(debug_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not subdirs:
            print("[调试] 调试目录为空")
            return

        latest_dir = subdirs[0]
        print(f"[调试] 正在打开调试窗口: {latest_dir.name}")

        try:
            from .debug_window import DebugWindow
            DebugWindow(self, str(latest_dir))
        except Exception as e:
            print(f"[错误] 打开调试窗口失败: {e}")
            messagebox.showerror("错误", f"打开调试窗口失败: {e}")

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

            # 更新GUI变量（Stage1c音频识别标签页）
            if hasattr(self, 's1ca_volume_threshold_var'):
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
                postprocess_config = audio_config.get("postprocess", {})
                if "enabled" in postprocess_config:
                    self.s1ca_postprocess_var.set(postprocess_config["enabled"])
                if "max_subtitle_duration" in postprocess_config:
                    self.s1ca_max_duration_var.set(str(postprocess_config["max_subtitle_duration"]))
                if "max_subtitle_chars" in postprocess_config:
                    self.s1ca_max_chars_var.set(str(postprocess_config["max_subtitle_chars"]))
            
            print(f"[配置] 已加载音频配置: VAD={vad_config.get('enabled', True)}, 后处理={postprocess_config.get('enabled', True)}")
            
        except Exception as e:
            print(f"[警告] 加载音频配置失败: {e}")

    def _run_mux_subtitle(self):
        """运行字幕封装"""
        # 检查是否启用封装
        if not self.s1c_mux_enabled_var.get():
            print("[提示] 字幕封装未启用，请在'字幕封装'区域勾选'启用字幕封装'")
            return
        
        csv_path = self.s1c_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return
        
        # 获取配置
        config = {
            "output_format": self.s1c_mux_format_var.get(),
            "file_handling": self.s1c_mux_handling_var.get(),
            "subtitle_processing": self.s1c_mux_processing_var.get(),
        }
        
        # 在后台线程中运行封装
        def run_mux_task():
            try:
                # 导入必要的模块
                import csv
                from ..utils.muxer import SubtitleMuxer
                
                # 初始化封装器
                muxer = SubtitleMuxer(config)
                
                # 读取CSV
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if not rows:
                    print("[警告] CSV为空")
                    return
                
                # 查找需要封装的文件对
                video_srt_pairs = []
                srt_dir = str(Path(csv_path).parent / "subtitles")
                
                for row in rows:
                    original_path = row.get("original_path", "").strip()
                    final_name = row.get("final_name", "").strip()
                    srt_path = row.get("srt_path", "").strip()
                    
                    if not original_path or not Path(original_path).exists():
                        continue
                    
                    # 确定字幕文件路径
                    if srt_path and Path(srt_path).exists():
                        # 使用CSV中记录的字幕路径
                        pass
                    else:
                        # 尝试查找同名字幕文件
                        if final_name:
                            srt_name = Path(final_name).stem + ".srt"
                            srt_path = str(Path(srt_dir) / srt_name)
                        else:
                            srt_name = Path(original_path).stem + ".srt"
                            srt_path = str(Path(srt_dir) / srt_name)
                    
                    if Path(srt_path).exists():
                        video_srt_pairs.append((original_path, srt_path))
                    else:
                        print(f"[跳过] 未找到字幕文件: {Path(original_path).name}")
                
                if not video_srt_pairs:
                    print("[警告] 未找到需要封装的文件对")
                    return
                
                print(f"[开始] 共 {len(video_srt_pairs)} 个文件需要封装")
                
                # 更新进度条
                def progress_callback(progress, status):
                    self.s1c_mux_progress_var.set(progress)
                    self.s1c_mux_status_var.set(status)
                    self.update_idletasks()
                
                # 执行批量封装
                result = muxer.batch_mux(video_srt_pairs, progress_callback)
                
                # 显示结果
                if result["success"]:
                    print(f"[完成] 批量封装成功: {result['success_count']} 个文件")
                    messagebox.showinfo("完成", f"批量封装成功: {result['success_count']} 个文件")
                else:
                    print(f"[警告] 批量封装完成: 成功 {result['success_count']}, 失败 {result['failed_count']}")
                    
                    # 保存失败文件列表，供重试使用
                    self._failed_mux_files = result["failed_files"]
                    
                    if result["failed_files"]:
                        print("[失败文件列表]")
                        for file_info in result["failed_files"]:
                            print(f"  - {Path(file_info['video']).name}: {file_info['error']}")
                    
                    messagebox.showwarning("完成", 
                                          f"批量封装完成: 成功 {result['success_count']}, 失败 {result['failed_count']}\n"
                                          f"失败文件已记录，可点击'重试失败'按钮重试")
                
            except Exception as e:
                print(f"[错误] 字幕封装失败: {e}")
                messagebox.showerror("错误", f"字幕封装失败: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=run_mux_task, daemon=True)
        thread.start()

    def _retry_failed_mux(self):
        """重试失败的封装操作"""
        if not hasattr(self, '_failed_mux_files') or not self._failed_mux_files:
            print("[提示] 没有失败的封装操作需要重试")
            messagebox.showinfo("提示", "没有失败的封装操作需要重试")
            return
        
        # 获取配置
        config = {
            "output_format": self.s1c_mux_format_var.get(),
            "file_handling": self.s1c_mux_handling_var.get(),
            "subtitle_processing": self.s1c_mux_processing_var.get(),
        }
        
        # 在后台线程中运行重试
        def run_retry_task():
            try:
                from ..utils.muxer import SubtitleMuxer
                
                # 初始化封装器
                muxer = SubtitleMuxer(config)
                
                print(f"[开始] 重试 {len(self._failed_mux_files)} 个失败文件")
                
                # 更新进度条
                def progress_callback(progress, status):
                    self.s1c_mux_progress_var.set(progress)
                    self.s1c_mux_status_var.set(status)
                    self.update_idletasks()
                
                # 执行重试
                result = muxer.retry_failed(self._failed_mux_files, progress_callback)
                
                # 显示结果
                if result["success"]:
                    print(f"[完成] 重试成功: {result['success_count']} 个文件")
                    messagebox.showinfo("完成", f"重试成功: {result['success_count']} 个文件")
                    # 清空失败列表
                    self._failed_mux_files = []
                else:
                    print(f"[警告] 重试完成: 成功 {result['success_count']}, 失败 {result['failed_count']}")
                    
                    # 更新失败文件列表
                    self._failed_mux_files = [
                        {"video": r["video"], "srt": r["srt"], "error": r["result"]["error"]}
                        for r in result["results"]
                        if not r["result"]["success"]
                    ]
                    
                    messagebox.showwarning("完成", 
                                          f"重试完成: 成功 {result['success_count']}, 失败 {result['failed_count']}\n"
                                          f"仍有 {len(self._failed_mux_files)} 个文件失败")
                
            except Exception as e:
                print(f"[错误] 重试失败: {e}")
                messagebox.showerror("错误", f"重试失败: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=run_retry_task, daemon=True)
        thread.start()

    def _batch_confirm(self):
        """批量确认所有记录"""
        csv_path = self.s2_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not messagebox.askyesno("确认", "确定要将所有记录的review_status设置为'已确认'吗？"):
            return

        try:
            import csv
            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # 确保字段存在
            if "review_status" not in fieldnames:
                fieldnames.append("review_status")

            # 批量设置
            count = 0
            for row in rows:
                if row.get("review_status") != "已确认":
                    row["review_status"] = "已确认"
                    count += 1

            # 保存CSV（原子化写入）
            from ..utils.atomic_csv import atomic_write_csv
            atomic_write_csv(csv_path, rows, fieldnames)

            print(f"[完成] 已确认 {count} 条记录")
            messagebox.showinfo("完成", f"已确认 {count} 条记录")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    def _batch_clear_final_name(self):
        """批量清空final_name"""
        csv_path = self.s2_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        if not messagebox.askyesno("确认", "确定要清空所有记录的final_name吗？"):
            return

        try:
            import csv
            # 读取CSV
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # 批量清空
            count = 0
            for row in rows:
                if row.get("final_name"):
                    row["final_name"] = ""
                    count += 1

            # 保存CSV（原子化写入）
            from ..utils.atomic_csv import atomic_write_csv
            atomic_write_csv(csv_path, rows, fieldnames)

            print(f"[完成] 已清空 {count} 条记录的final_name")
            messagebox.showinfo("完成", f"已清空 {count} 条记录的final_name")

        except Exception as e:
            print(f"[错误] {e}")
            messagebox.showerror("错误", str(e))

    def _run_rename(self):
        """运行重命名"""
        csv = self.s2_csv_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "rename", "-c", csv]

        if self.s2_dry_run_var.get():
            cmd.append("--dry-run")

        self._run_command(cmd)


def main():
    """启动GUI"""
    app = TitleClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
