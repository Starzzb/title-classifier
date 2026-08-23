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

from ..utils.file_resolve import resolve_media_path

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
PYTHON = sys.executable
DEFAULT_CSV = "data/output/title_review.csv"


class CollapsibleFrame(ttk.Frame):
    """可折叠的 LabelFrame"""

    def __init__(self, parent, text="", expanded=True, **kwargs):
        super().__init__(parent, **kwargs)
        self._expanded = expanded

        # 标题栏
        self._header = ttk.Frame(self)
        self._header.pack(fill=tk.X)

        self._toggle_btn = ttk.Button(
            self._header, text=f"{'▼' if expanded else '▶'} {text}",
            command=self._toggle, bootstyle="link",
        )
        self._toggle_btn.pack(side=tk.LEFT)

        # 内容区
        self._content = ttk.Frame(self)
        if expanded:
            self._content.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

    def _toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self._content.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
            self._toggle_btn.configure(text=self._toggle_btn.cget("text").replace("▶", "▼"))
        else:
            self._content.pack_forget()
            self._toggle_btn.configure(text=self._toggle_btn.cget("text").replace("▼", "▶"))

    @property
    def content(self):
        return self._content


class StageVisionTab(ttk.Frame):
    """Stage1c 视觉识别标签页"""

    def __init__(self, master, ctx: AppContext, run_command=None, gui_sync_to_db=None, sync_csv_to_db=None, **kwargs):
        super().__init__(master, **kwargs)
        from ..utils.config import load_merged_config, get_config_value
        self._cfg = load_merged_config()
        self._gv = lambda key, fallback: get_config_value(self._cfg, key, fallback)
        self.ctx = ctx
        self._run_command = run_command
        self._gui_sync_to_db = gui_sync_to_db
        self._sync_csv_to_db = sync_csv_to_db
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

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mousewheel(event):
            try:
                canvas.unbind_all("<MouseWheel>")
            except tk.TclError:
                pass
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # CSV + Provider（顶部常驻）
        top_bar = ttk.Frame(scroll_frame)
        top_bar.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(top_bar, text="CSV:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(top_bar, textvariable=self.ctx.csv_var, width=50).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="浏览", width=5, command=self._browse_csv).pack(side=tk.LEFT, padx=2)



        # ===== 折叠区1: 推理配置（默认展开） =====
        sec1 = CollapsibleFrame(scroll_frame, text="推理配置", expanded=True)
        sec1.pack(fill=tk.X, padx=4, pady=2)
        self._build_inference_section(sec1.content)

        # ===== 折叠区2: 分析参数（默认折叠） =====
        sec2 = CollapsibleFrame(scroll_frame, text="分析参数", expanded=False)
        sec2.pack(fill=tk.X, padx=4, pady=2)
        self._build_params_section(sec2.content)

        # ===== 折叠区3: 字幕封装（默认折叠） =====
        sec3 = CollapsibleFrame(scroll_frame, text="字幕封装", expanded=False)
        sec3.pack(fill=tk.X, padx=4, pady=2)
        self._build_mux_section(sec3.content)

        # 执行按钮（底部常驻）
        btn_frame = ttk.Frame(scroll_frame)
        btn_frame.pack(fill=tk.X, padx=4, pady=8)

        ttk.Button(btn_frame, text="视觉识别", command=self._run_vision).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="重试失败行", command=self._run_vision_retry).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="查看调试结果", command=self._open_debug_browser).pack(side=tk.LEFT, padx=4)

    def _build_inference_section(self, parent):
        """推理配置折叠区"""
        # 推理引擎
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row1, text="推理设备:").pack(side=tk.LEFT, padx=(0, 4))
        self.s1c_device_var = tk.StringVar(value=self._gv("general.device", "cpu"))
        ttk.Combobox(row1, textvariable=self.s1c_device_var, values=["cpu", "auto", "cuda"], state="readonly", width=8).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row1, text="YOLO后端:").pack(side=tk.LEFT, padx=(0, 4))
        self.s1c_backend_var = tk.StringVar(value=self._gv("yolo.backend", "auto"))
        ttk.Combobox(row1, textvariable=self.s1c_backend_var, values=["auto", "openvino", "pytorch"], state="readonly", width=10).pack(side=tk.LEFT)

        # 状态
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.s1c_engine_status = ttk.Label(row2, text="", foreground="#888888")
        self.s1c_engine_status.pack(side=tk.LEFT)
        self._update_engine_status()

        # 检测模式
        row3 = ttk.Frame(parent)
        row3.pack(fill=tk.X, padx=4, pady=2)
        self.s1c_comprehensive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="全面分析模式（3模型投票）", variable=self.s1c_comprehensive_var).pack(side=tk.LEFT)
        self.s1c_use_clip_var = tk.BooleanVar()
        ttk.Checkbutton(row3, text="CLIP预分类", variable=self.s1c_use_clip_var).pack(side=tk.LEFT, padx=12)

        # 运动检测
        row4 = ttk.Frame(parent)
        row4.pack(fill=tk.X, padx=4, pady=2)
        self.s1c_motion_var = tk.BooleanVar(value=self._gv("vision.motion_detection", True))
        ttk.Checkbutton(row4, text="运动检测", variable=self.s1c_motion_var).pack(side=tk.LEFT)
        ttk.Label(row4, text="阈值(%):").pack(side=tk.LEFT, padx=(12, 4))
        self.s1c_motion_threshold_var = tk.StringVar(value=str(self._gv("vision.motion_threshold", 8.0)))
        mt_entry = ttk.Entry(row4, textvariable=self.s1c_motion_threshold_var, width=6)
        mt_entry.pack(side=tk.LEFT)
        ToolTip(mt_entry, "跳过YOLO推理的像素变化最小阈值（%）。\n两帧之间变化低于此值视为静态画面，跳过YOLO直接复用上一帧结果。\n值越低越敏感（5=轻微光影变化也会触发），值越高越宽松（8=画面明显变化才触发）。")

    def _build_params_section(self, parent):
        """分析参数折叠区"""
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row1, text="采样间隔(秒):").pack(side=tk.LEFT, padx=4)
        self.s1c_analysis_step_var = tk.StringVar(value=str(self._gv("vision.analysis_step", 5.0)))
        ttk.Entry(row1, textvariable=self.s1c_analysis_step_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="最大采样帧数:").pack(side=tk.LEFT, padx=8)
        self.s1c_max_sample_var = tk.StringVar(value=str(self._gv("vision.max_sample_frames", 50)))
        ttk.Entry(row1, textvariable=self.s1c_max_sample_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="VLM帧数:").pack(side=tk.LEFT, padx=8)
        self.s1c_vlm_frames_var = tk.StringVar(value=str(self._gv("vision.vlm_frames", 10)))
        ttk.Entry(row1, textvariable=self.s1c_vlm_frames_var, width=6).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row2, text="YOLO置信度:").pack(side=tk.LEFT, padx=4)
        self.s1c_yolo_conf_var = tk.StringVar(value=str(self._gv("yolo.confidence", 0.5)))
        ttk.Entry(row2, textvariable=self.s1c_yolo_conf_var, width=6).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(parent)
        row3.pack(fill=tk.X, padx=4, pady=2)
        self.s1c_all_var = tk.BooleanVar()
        ttk.Checkbutton(row3, text="处理所有未识别文件", variable=self.s1c_all_var).pack(side=tk.LEFT, padx=4)
        self.s1c_debug_var = tk.BooleanVar()
        ttk.Checkbutton(row3, text="启用调试", variable=self.s1c_debug_var).pack(side=tk.LEFT, padx=12)

        # 场景检测
        row4 = ttk.Frame(parent)
        row4.pack(fill=tk.X, padx=4, pady=2)
        self.s1c_scene_detection_var = tk.BooleanVar(value=self._gv("scene_detection.enabled", True))
        ttk.Checkbutton(row4, text="场景分段分析（长视频自动分段）", variable=self.s1c_scene_detection_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row4, text="最大段数:").pack(side=tk.LEFT, padx=(8, 2))
        self.s1c_max_scenes_var = tk.StringVar(value=str(self._gv("scene_detection.max_scenes", 10)))
        ttk.Entry(row4, textvariable=self.s1c_max_scenes_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(row4, text="每段发送帧数:").pack(side=tk.LEFT, padx=(4, 2))
        self.s1c_frames_per_scene_var = tk.StringVar(value=str(self._gv("scene_detection.frames_per_scene", 10)))
        fps_entry = ttk.Entry(row4, textvariable=self.s1c_frames_per_scene_var, width=4)
        fps_entry.pack(side=tk.LEFT, padx=2)
        ToolTip(fps_entry, "每个场景段发送给VLM分析的帧数，默认10帧。\n实际推理帧数按 min(每段帧数, 时长/0.5) 动态计算，可能多于发送帧数。")

    def _build_mux_section(self, parent):
        """字幕封装折叠区"""
        ttk.Label(parent, text="需要先运行音频识别产出字幕文件", foreground="blue", font=("Microsoft YaHei", 8)).pack(fill=tk.X, padx=4, pady=2)

        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, padx=4, pady=2)
        self.s1c_mux_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="启用字幕封装", variable=self.s1c_mux_enabled_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="输出格式:").pack(side=tk.LEFT, padx=8)
        self.s1c_mux_format_var = tk.StringVar(value="auto")
        ttk.Combobox(row1, textvariable=self.s1c_mux_format_var, values=["auto", "mkv", "mp4"], state="readonly", width=8).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row2, text="文件处理:").pack(side=tk.LEFT, padx=4)
        self.s1c_mux_handling_var = tk.StringVar(value="new")
        ttk.Combobox(row2, textvariable=self.s1c_mux_handling_var, values=["new", "overwrite"], state="readonly", width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="字幕处理:").pack(side=tk.LEFT, padx=8)
        self.s1c_mux_processing_var = tk.StringVar(value="direct")
        ttk.Combobox(row2, textvariable=self.s1c_mux_processing_var, values=["direct", "convert"], state="readonly", width=8).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(parent)
        row3.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(row3, text="封装字幕", command=self._run_mux_subtitle).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="重试失败", command=self._retry_failed_mux).pack(side=tk.LEFT, padx=4)

        self.s1c_mux_progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(parent, variable=self.s1c_mux_progress_var, maximum=100).pack(fill=tk.X, padx=4, pady=2)
        self.s1c_mux_status_var = tk.StringVar(value="就绪")
        ttk.Label(parent, textvariable=self.s1c_mux_status_var).pack(fill=tk.X, padx=4, pady=2)

    # ==================== 文件浏览 ====================

    def _browse_csv(self):
        """浏览CSV文件"""
        initial = Path(PROJECT_DIR) / "data" / "output"
        file_path = filedialog.askopenfilename(title="选择CSV文件", initialdir=str(initial), filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.ctx.csv_var.set(file_path)

    # ==================== 视觉识别 ====================

    def _run_vision(self):
        """运行视觉识别"""
        csv = self.ctx.csv_var.get()
        from ..utils.config import load_merged_config, get_config_value
        cfg = load_merged_config()
        gv = lambda key, fallback: get_config_value(cfg, key, fallback)
        provider = get_config_value(cfg, "providers.stage_providers.vision", "gcli")
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
            if threshold and threshold != str(gv("vision.motion_threshold", 8.0)):
                cmd.extend(["--motion-threshold", threshold])

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        max_sample = self.s1c_max_sample_var.get()
        if max_sample and max_sample != str(gv("vision.max_sample_frames", 50)):
            cmd.extend(["--max-sample-frames", max_sample])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        # YOLO置信度（校验范围）
        yolo_conf = self.s1c_yolo_conf_var.get()
        if yolo_conf and yolo_conf != str(gv("yolo.confidence", 0.5)):
            try:
                conf_val = float(yolo_conf)
                if not (0.1 <= conf_val <= 0.9):
                    messagebox.showwarning("警告", "YOLO置信度应在 0.1-0.9 之间")
                    return
            except ValueError:
                messagebox.showwarning("警告", "YOLO置信度必须是数字")
                return
            cmd.extend(["--yolo-conf", yolo_conf])

        # 场景检测
        if not self.s1c_scene_detection_var.get():
            cmd.append("--no-scene-detection")
        else:
            max_scenes = self.s1c_max_scenes_var.get()
            if max_scenes and max_scenes != str(gv("scene_detection.max_scenes", 10)):
                cmd.extend(["--max-scenes", max_scenes])
            frames_per = self.s1c_frames_per_scene_var.get()
            if frames_per and frames_per != str(gv("scene_detection.frames_per_scene", 10)):
                cmd.extend(["--frames-per-scene", frames_per])

        if self.s1c_all_var.get():
            cmd.append("--all")

        # 调试模式
        if self.s1c_debug_var.get():
            cmd.append("--debug")
            self._debug_enabled = True
        else:
            self._debug_enabled = False

        # 定义完成回调，用于同步数据库
        def on_vision_complete(returncode=None):
            if getattr(self, '_debug_enabled', False):
                print('[调试] 调试数据已保存，点击"查看调试结果"按钮可查看')
            # 同步视觉识别结果到数据库
            self._sync_csv_to_db(csv)

        self._run_command(cmd, callback=on_vision_complete)

    def _run_vision_retry(self):
        """重试失败的视觉识别行"""
        csv = self.ctx.csv_var.get()
        from ..utils.config import load_merged_config, get_config_value
        cfg = load_merged_config()
        gv = lambda key, fallback: get_config_value(cfg, key, fallback)
        provider = get_config_value(cfg, "providers.stage_providers.vision", "gcli")
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
            if threshold and threshold != str(gv("vision.motion_threshold", 8.0)):
                cmd.extend(["--motion-threshold", threshold])

        # 场景检测
        if not self.s1c_scene_detection_var.get():
            cmd.append("--no-scene-detection")
        else:
            max_scenes = self.s1c_max_scenes_var.get()
            if max_scenes and max_scenes != str(gv("scene_detection.max_scenes", 10)):
                cmd.extend(["--max-scenes", max_scenes])
            frames_per = self.s1c_frames_per_scene_var.get()
            if frames_per and frames_per != str(gv("scene_detection.frames_per_scene", 10)):
                cmd.extend(["--frames-per-scene", frames_per])

        # 添加分析参数
        analysis_step = self.s1c_analysis_step_var.get()
        if analysis_step:
            cmd.extend(["--analysis-step", analysis_step])

        max_sample = self.s1c_max_sample_var.get()
        if max_sample and max_sample != str(gv("vision.max_sample_frames", 50)):
            cmd.extend(["--max-sample-frames", max_sample])

        vlm_frames = self.s1c_vlm_frames_var.get()
        if vlm_frames:
            cmd.extend(["--vlm-frames", vlm_frames])

        # YOLO置信度（校验范围）
        yolo_conf = self.s1c_yolo_conf_var.get()
        if yolo_conf and yolo_conf != str(gv("yolo.confidence", 0.5)):
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
        def on_retry_complete(returncode=None):
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
                        path=original_path,
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
