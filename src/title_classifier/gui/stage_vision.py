"""Stage1c 视觉识别 Tab - 从主窗口提取的独立组件"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from .context import AppContext
from .app import ToolTip

from ..providers import get_providers_for_gui
from ..utils.file_resolve import resolve_media_path

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = sys.executable
DEFAULT_CSV = "data/output/title_review.csv"


class StageVisionTab(ttk.Frame):
    """Stage1c 视觉识别标签页"""

    def __init__(self, master, ctx: AppContext, run_command, **kwargs):
        super().__init__(master, **kwargs)
        self.ctx = ctx
        self._run_command = run_command
        self._failed_mux_files = []
        self._debug_enabled = False
        self._build()

    def _build(self):
        # 可滚动容器
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮绑定：只在鼠标悬停canvas时生效
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # CSV文件
        csv_frame = ttk.LabelFrame(scroll_frame, text="CSV文件")
        csv_frame.pack(fill=tk.X, padx=4, pady=4)

        csv_entry = ttk.Entry(csv_frame, textvariable=self.ctx.csv_var, width=60)
        csv_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(csv_frame, text="浏览...", command=self._browse_csv).pack(side=tk.LEFT, padx=4)
        ToolTip(csv_entry, "Stage1生成的CSV文件，视觉识别会分析视频内容并生成描述和关键词")

        # Provider选择
        provider_frame = ttk.LabelFrame(scroll_frame, text="AI Provider")
        provider_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_provider_var = tk.StringVar(value="gcli")
        providers = get_providers_for_gui("1c")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.s1c_provider_var, values=providers, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=4)
        ToolTip(provider_combo, "选择视觉AI服务提供商\n- gcli: Google Gemini（推荐）\n- mimo: 小米MiMo\n- zhipu: 智谱GLM")

        # 推理引擎配置
        engine_frame = ttk.LabelFrame(scroll_frame, text="推理引擎")
        engine_frame.pack(fill=tk.X, padx=4, pady=4)

        # 第一行：推理设备 + YOLO后端
        engine_row1 = ttk.Frame(engine_frame)
        engine_row1.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(engine_row1, text="推理设备:").pack(side=tk.LEFT, padx=(0, 4))
        self.s1c_device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(engine_row1, textvariable=self.s1c_device_var, values=["cpu", "auto", "cuda"], state="readonly", width=8)
        device_combo.pack(side=tk.LEFT, padx=(0, 12))
        ToolTip(device_combo, "推理设备\n- cpu: CPU多核并行（推荐）\n- auto: 自动检测\n- cuda: GPU加速")

        ttk.Label(engine_row1, text="YOLO后端:").pack(side=tk.LEFT, padx=(0, 4))
        self.s1c_backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(engine_row1, textvariable=self.s1c_backend_var,
                                     values=["auto", "openvino", "pytorch"], state="readonly", width=10)
        backend_combo.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(backend_combo, "YOLO推理后端\n\n"
                "- auto: 自动选择（CPU时用OpenVINO，推荐）\n"
                "- openvino: Intel/AMD CPU加速（FP16，2-3x）\n"
                "- pytorch: 原始PyTorch\n\n"
                "首次使用OpenVINO时会自动导出模型\n"
                "支持detect/pose/segment三个模型")

        # 第二行：状态显示
        engine_row2 = ttk.Frame(engine_frame)
        engine_row2.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.s1c_engine_status = ttk.Label(engine_row2, text="", foreground="#888888")
        self.s1c_engine_status.pack(side=tk.LEFT)
        self._update_engine_status()

        # 检测器选项
        det_frame = ttk.LabelFrame(scroll_frame, text="检测器")
        det_frame.pack(fill=tk.X, padx=4, pady=4)

        # 全面分析模式选项
        self.s1c_comprehensive_var = tk.BooleanVar(value=False)
        comprehensive_cb = ttk.Checkbutton(det_frame, text="全面分析模式（3模型投票）", variable=self.s1c_comprehensive_var)
        comprehensive_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(comprehensive_cb, "使用三个YOLO模型进行全面分析\n\n"
                "- detect + pose + segment 三个模型\n"
                "- 投票决策：至少两个模型检测到人体才认为有人体\n"
                "- 提供姿态分析、穿着分割等详细信息\n\n"
                "注意：会使用更多内存和时间")

        # CLIP选项
        self.s1c_use_clip_var = tk.BooleanVar()
        clip_cb = ttk.Checkbutton(det_frame, text="CLIP预分类", variable=self.s1c_use_clip_var)
        clip_cb.pack(side=tk.LEFT, padx=8)
        ToolTip(clip_cb, "使用CLIP模型进行图像预分类\n\n"
                "- 快速识别图片内容类别\n"
                "- 如果置信度足够高，可跳过VLM调用")

        # 运动检测选项
        motion_frame = ttk.LabelFrame(scroll_frame, text="运动检测")
        motion_frame.pack(fill=tk.X, padx=4, pady=4)

        self.s1c_motion_var = tk.BooleanVar(value=True)
        motion_cb = ttk.Checkbutton(motion_frame, text="启用运动检测前置过滤", variable=self.s1c_motion_var)
        motion_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(motion_cb, "跳过静止画面的YOLO推理\n\n"
                "原理：\n"
                "- 使用帧差法检测画面变化\n"
                "- 静止帧复用上一帧结果\n"
                "- 监控等静态场景可减少60-80%推理\n\n"
                "建议：保持启用，对动态场景无负面影响")

        ttk.Label(motion_frame, text="阈值(%):").pack(side=tk.LEFT, padx=(12, 4))
        self.s1c_motion_threshold_var = tk.StringVar(value="5.0")
        threshold_entry = ttk.Entry(motion_frame, textvariable=self.s1c_motion_threshold_var, width=6)
        threshold_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(threshold_entry, "运动检测阈值（变化像素比例）\n\n"
                "- 默认5%：变化超过5%认为有运动\n"
                "- 降低（如2%）：更敏感，更多帧执行推理\n"
                "- 提高（如10%）：更不敏感，更多帧被跳过")

        # 分析参数
        param_frame = ttk.LabelFrame(scroll_frame, text="分析参数")
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

        ttk.Label(param_frame, text="最大采样帧数:").pack(side=tk.LEFT, padx=8)
        self.s1c_max_sample_var = tk.StringVar(value="50")
        max_sample_entry = ttk.Entry(param_frame, textvariable=self.s1c_max_sample_var, width=6)
        max_sample_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(max_sample_entry, "采样帧数上限\n\n"
                "- 默认50帧，覆盖整个视频\n"
                "- 增大：分析更细致，但YOLO推理时间更长\n"
                "- 减小：分析更快，但可能遗漏细节\n\n"
                "超过此数时，会均匀分布到整个视频")

        ttk.Label(param_frame, text="VLM帧数:").pack(side=tk.LEFT, padx=8)
        self.s1c_vlm_frames_var = tk.StringVar(value="10")
        frames_entry = ttk.Entry(param_frame, textvariable=self.s1c_vlm_frames_var, width=6)
        frames_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(frames_entry, "传给VLM分析的帧数\n\n"
                "从采样帧中智能选择，传给VLM进行内容分析")

        # 第二行参数
        param_frame2 = ttk.Frame(scroll_frame)
        param_frame2.pack(fill=tk.X, padx=4, pady=(0, 2))

        ttk.Label(param_frame2, text="YOLO置信度:").pack(side=tk.LEFT, padx=4)
        self.s1c_yolo_conf_var = tk.StringVar(value="0.4")
        yolo_conf_entry = ttk.Entry(param_frame2, textvariable=self.s1c_yolo_conf_var, width=6)
        yolo_conf_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(yolo_conf_entry, "YOLO人体检测置信度阈值\n\n"
                "- 默认0.4：平衡检测率和误检率\n"
                "- 降低（如0.3）：检测更多人体，但可能误检\n"
                "- 提高（如0.6）：更严格，但可能漏检\n\n"
                "建议：保持0.4，除非有明显误检或漏检")

        # 选项
        opt_frame = ttk.LabelFrame(scroll_frame, text="选项")
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
        deprecation_frame = ttk.Frame(scroll_frame)
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
        mux_frame = ttk.LabelFrame(scroll_frame, text="字幕封装")
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
        btn_frame = ttk.Frame(scroll_frame)
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

        retry_btn = ttk.Button(btn_frame, text="重试失败行", command=self._run_vision_retry)
        retry_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(retry_btn, "重试之前视觉识别失败的行\n\n"
                "只处理 vision_failed=true 的行\n"
                "成功后自动清除失败标记")

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        debug_btn = ttk.Button(btn_frame, text="查看调试结果", command=self._open_debug_browser)
        debug_btn.pack(side=tk.LEFT, padx=4)
        ToolTip(debug_btn, "浏览并打开已有的调试结果\n\n"
                "查看之前视觉识别保存的调试数据：\n"
                "- 每帧的YOLO检测结果（detect/pose/segment）\n"
                "- 投票决策详情\n"
                "- 姿态关键点\n"
                "- VLM输入输出")

    # ==================== 文件浏览 ====================

    def _browse_csv(self):
        """浏览CSV文件"""
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.ctx.csv_var.set(file_path)

    # ==================== 视觉识别 ====================

    def _run_vision(self):
        """运行视觉识别"""
        csv = self.ctx.csv_var.get()
        provider = self.s1c_provider_var.get()
        device = self.s1c_device_var.get()
        backend = self.s1c_backend_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "vision", "-c", csv, "-p", provider]

        # 始终使用YOLO
        cmd.append("--use-yolo")

        # 推理设备
        if device and device != "auto":
            cmd.extend(["--device", device])

        # YOLO推理后端
        if backend and backend != "auto":
            cmd.extend(["--backend", backend])

        # 并发数：根据设备自动调整
        if device == "cuda":
            cmd.extend(["--concurrent", "1"])  # GPU串行，避免CUDA死锁
        elif device == "cpu":
            cpu_workers = min(os.cpu_count() - 1, 4)
            cmd.extend(["--concurrent", str(cpu_workers)])  # CPU多核并行
        # auto模式不传concurrent，让CLI自动决定

        # 全面分析模式
        if self.s1c_comprehensive_var.get():
            cmd.append("--comprehensive")

        if self.s1c_use_clip_var.get():
            cmd.append("--use-clip")

        # 运动检测
        if not self.s1c_motion_var.get():
            cmd.append("--no-motion-detection")
        else:
            threshold = self.s1c_motion_threshold_var.get()
            if threshold and threshold != "5.0":
                cmd.extend(["--motion-threshold", threshold])

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        max_sample = self.s1c_max_sample_var.get()
        if max_sample and max_sample != "50":
            cmd.extend(["--max-sample-frames", max_sample])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        # YOLO置信度（校验范围）
        yolo_conf = self.s1c_yolo_conf_var.get()
        if yolo_conf and yolo_conf != "0.4":
            try:
                conf_val = float(yolo_conf)
                if not (0.1 <= conf_val <= 0.9):
                    messagebox.showwarning("警告", "YOLO置信度应在 0.1-0.9 之间")
                    return
            except ValueError:
                messagebox.showwarning("警告", "YOLO置信度必须是数字")
                return
            cmd.extend(["--yolo-conf", yolo_conf])

        if self.s1c_all_var.get():
            cmd.append("--all")

        # 调试模式
        if self.s1c_debug_var.get():
            cmd.append("--debug")
            self._debug_enabled = True
        else:
            self._debug_enabled = False

        # 定义完成回调，用于同步数据库
        def on_vision_complete():
            if getattr(self, '_debug_enabled', False):
                print('[调试] 调试数据已保存，点击"查看调试结果"按钮可查看')
            # 同步视觉识别结果到数据库
            self._sync_csv_to_db(csv)

        self._run_command(cmd, callback=on_vision_complete)

    def _run_vision_retry(self):
        """重试失败的视觉识别行"""
        csv = self.ctx.csv_var.get()
        provider = self.s1c_provider_var.get()
        device = self.s1c_device_var.get()
        backend = self.s1c_backend_var.get()

        cmd = [PYTHON, "-m", "title_classifier", "vision", "-c", csv, "-p", provider]

        # 始终使用YOLO
        cmd.append("--use-yolo")

        # 推理设备
        if device and device != "auto":
            cmd.extend(["--device", device])

        # YOLO推理后端
        if backend and backend != "auto":
            cmd.extend(["--backend", backend])

        # 并发数：根据设备自动调整
        if device == "cuda":
            cmd.extend(["--concurrent", "1"])
        elif device == "cpu":
            cpu_workers = min(os.cpu_count() - 1, 4)
            cmd.extend(["--concurrent", str(cpu_workers)])

        # 全面分析模式
        if self.s1c_comprehensive_var.get():
            cmd.append("--comprehensive")

        if self.s1c_use_clip_var.get():
            cmd.append("--use-clip")

        # 运动检测
        if not self.s1c_motion_var.get():
            cmd.append("--no-motion-detection")
        else:
            threshold = self.s1c_motion_threshold_var.get()
            if threshold and threshold != "5.0":
                cmd.extend(["--motion-threshold", threshold])

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        max_sample = self.s1c_max_sample_var.get()
        if max_sample and max_sample != "50":
            cmd.extend(["--max-sample-frames", max_sample])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        # YOLO置信度（校验范围）
        yolo_conf = self.s1c_yolo_conf_var.get()
        if yolo_conf and yolo_conf != "0.4":
            try:
                conf_val = float(yolo_conf)
                if not (0.1 <= conf_val <= 0.9):
                    messagebox.showwarning("警告", "YOLO置信度应在 0.1-0.9 之间")
                    return
            except ValueError:
                messagebox.showwarning("警告", "YOLO置信度必须是数字")
                return
            cmd.extend(["--yolo-conf", yolo_conf])

        # 重试失败行
        cmd.append("--retry-failed")

        # 定义完成回调
        def on_retry_complete():
            self._sync_csv_to_db(csv)

        self._run_command(cmd, callback=on_retry_complete)

    # ==================== 推理引擎状态 ====================

    def _update_engine_status(self):
        """更新推理引擎状态显示"""
        device = self.s1c_device_var.get()
        backend = self.s1c_backend_var.get()

        # 检测 OpenVINO
        openvino_ok = False
        try:
            import openvino
            openvino_ok = True
        except ImportError:
            pass

        # 检测 CUDA
        cuda_ok = False
        try:
            import torch
            if torch.cuda.is_available():
                cuda_ok = True
        except ImportError:
            pass

        # 构建状态文本
        if backend == "openvino" and not openvino_ok:
            status = "⚠ OpenVINO 未安装，将回退到 PyTorch"
            color = "#cc4444"
        elif device == "cuda" and not cuda_ok:
            status = "⚠ CUDA 不可用，将使用 CPU"
            color = "#cc4444"
        elif backend == "openvino" and openvino_ok:
            status = "✓ OpenVINO 加速（FP16，2-3x）"
            color = "#44aa44"
        elif device == "cuda" and cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            status = f"✓ CUDA: {gpu_name}"
            color = "#44aa44"
        else:
            status = "✓ CPU 推理（多核并行）"
            color = "#44aa44"

        self.s1c_engine_status.config(text=status, foreground=color)

    # ==================== 调试窗口 ====================

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

    def _open_debug_browser(self):
        """浏览并选择调试目录"""
        debug_dir = PROJECT_DIR / "data" / "debug"
        if not debug_dir.exists():
            messagebox.showinfo("提示", "调试目录不存在\n\n请先运行视觉识别并启用调试模式。")
            return

        # 收集所有调试子目录
        subdirs = sorted(
            [d for d in debug_dir.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not subdirs:
            messagebox.showinfo("提示", "调试目录为空\n\n请先运行视觉识别并启用调试模式。")
            return

        # 创建选择对话框
        dialog = tk.Toplevel(self)
        dialog.title("选择调试结果")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="选择要查看的调试结果：", font=("Microsoft YaHei", 10, "bold")).pack(padx=10, pady=(10, 5), anchor=tk.W)

        # 搜索框
        search_frame = ttk.Frame(dialog)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=5)

        # 列表框
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("name", "time", "frames")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("name", text="目录名")
        tree.heading("time", text="创建时间")
        tree.heading("frames", text="帧数")
        tree.column("name", width=280)
        tree.column("time", width=150)
        tree.column("frames", width=60)

        tree_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=tree_scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 填充数据
        def fill_tree(filter_text=""):
            for item in tree.get_children():
                tree.delete(item)
            for d in subdirs:
                name = d.name
                if filter_text and filter_text.lower() not in name.lower():
                    continue
                mtime = d.stat().st_mtime
                time_str = __import__("datetime").datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                # 统计帧数
                detection_dir = d / "detection"
                frame_count = len(list(detection_dir.glob("*_result.json"))) if detection_dir.exists() else 0
                tree.insert("", tk.END, values=(name, time_str, frame_count), iid=str(d))

        fill_tree()

        # 搜索过滤
        def on_search(*_):
            fill_tree(search_var.get())

        search_var.trace_add("write", on_search)

        # 双击打开
        def on_double_click(event):
            selected = tree.selection()
            if selected:
                dir_path = selected[0]
                dialog.destroy()
                self._open_debug_window(dir_path)

        tree.bind("<Double-1>", on_double_click)

        # 按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def on_open():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("提示", "请先选择一个调试结果")
                return
            dir_path = selected[0]
            dialog.destroy()
            self._open_debug_window(dir_path)

        ttk.Button(btn_frame, text="打开", command=on_open).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # 也支持手动选择目录
        def on_browse():
            selected_dir = filedialog.askdirectory(
                title="选择调试目录",
                initialdir=str(debug_dir)
            )
            if selected_dir:
                dialog.destroy()
                self._open_debug_window(selected_dir)

        ttk.Button(btn_frame, text="浏览其他目录...", command=on_browse).pack(side=tk.LEFT, padx=5)

    def _open_debug_window(self, debug_dir: str):
        """打开调试窗口"""
        try:
            from .debug_window import DebugWindow
            DebugWindow(self, debug_dir)
        except Exception as e:
            print(f"[错误] 打开调试窗口失败: {e}")
            messagebox.showerror("错误", f"打开调试窗口失败: {e}")

    # ==================== 字幕封装 ====================

    def _run_mux_subtitle(self):
        """运行字幕封装"""
        # 检查是否启用封装
        if not self.s1c_mux_enabled_var.get():
            print("[提示] 字幕封装未启用，请在'字幕封装'区域勾选'启用字幕封装'")
            return

        csv_path = self.ctx.csv_var.get()
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
                    original_title = row.get("original_title", "").strip()
                    final_name = row.get("final_name", "").strip()
                    srt_path = row.get("srt_path", "").strip()

                    # 解析实际文件路径（Stage2重命名后用final_name回退查找）
                    original_path = resolve_media_path(original_path, final_name, original_title)

                    if not original_path:
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
        if not self._failed_mux_files:
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

    # ==================== 数据库同步 ====================

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

    def _sync_csv_to_db(self, csv_path: str):
        """从 CSV 同步数据到数据库"""
        if not self.ctx.db:
            return
        try:
            import csv as csv_module
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    original_path = row.get("original_path", "")
                    if not original_path:
                        continue

                    # 尝试获取元数据
                    file_size = None
                    fs_str = row.get("file_size", "").strip()
                    if fs_str:
                        try:
                            file_size = int(fs_str)
                        except ValueError:
                            pass

                    duration = None
                    dur_str = row.get("duration", "").strip()
                    if dur_str:
                        try:
                            duration = float(dur_str)
                        except ValueError:
                            pass

                    resolution = row.get("resolution", "").strip()

                    existing = self.ctx.db.find_match(
                        original_title=row.get("original_title", ""),
                        file_size=file_size,
                        duration=duration,
                    )
                    if existing:
                        continue

                    data = {
                        "original_title": row.get("original_title", ""),
                        "original_path": original_path,
                        "current_path": original_path,
                        "needs_vision": row.get("needs_vision", "").lower() == "true",
                        "final_name": row.get("final_name", ""),
                        "review_status": row.get("review_status", "待确认"),
                        "audio_recognized": row.get("audio_recognized", "").lower() == "true",
                        "srt_path": row.get("srt_path", ""),
                        "vision_description": row.get("vision_description", ""),
                        "vision_keywords": row.get("vision_keywords", ""),
                        "human_detected": row.get("human_detected", "").lower() == "true",
                        "detection_method": row.get("detection_method", ""),
                        "file_size": file_size,
                        "duration": duration,
                        "resolution": resolution,
                    }
                    self.ctx.db.insert_media(data)
            print("[数据库] 视觉识别结果已同步")
        except Exception as e:
            print(f"[警告] 数据库同步失败: {e}")
