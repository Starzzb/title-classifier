"""调试窗口 - 三栏布局：缩略图网格 | 大图预览 | 检测结果面板"""

import json
import os
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
except ImportError:
    import tkinter.ttk as ttk

    # Fallback constants
    VERTICAL = tk.VERTICAL
    HORIZONTAL = tk.HORIZONTAL
    BOTH = tk.BOTH
    LEFT = tk.LEFT
    RIGHT = tk.RIGHT
    NW = tk.NW
    X = tk.X
    Y = tk.Y
    W = tk.W
    E = tk.E
    NSEW = "nsew"
    NS = "ns"
    EW = "ew"
    END = tk.END
    WORD = tk.WORD
    DISABLED = "disabled"
    NORMAL = "normal"


class CollapsibleFrame(ttk.Frame):
    """可折叠的Frame组件"""

    def __init__(self, parent, text="", expanded=False, **kwargs):
        super().__init__(parent, **kwargs)
        self._expanded = expanded
        self._text = text

        # 标题栏
        self._header = ttk.Frame(self)
        self._header.pack(fill=X)

        self._toggle_btn = ttk.Label(
            self._header,
            text=f"▼ {text}" if expanded else f"▶ {text}",
            font=("Microsoft YaHei", 9, "bold"),
            cursor="hand2",
        )
        self._toggle_btn.pack(side=LEFT, padx=2, pady=2)
        self._toggle_btn.bind("<Button-1>", lambda e: self.toggle())

        # 内容区
        self._content = ttk.Frame(self)
        if expanded:
            self._content.pack(fill=BOTH, expand=True)

        # 绑定鼠标滚轮事件到所有子组件
        self._bind_mousewheel_recursive(self)

    def _bind_mousewheel_recursive(self, widget):
        """递归绑定鼠标滚轮事件"""
        widget.bind("<MouseWheel>", self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        # 查找父级的 right_scroll_canvas
        parent = self.master
        while parent:
            if hasattr(parent, '_right_mousewheel'):
                parent._right_mousewheel(event)
                break
            parent = parent.master if hasattr(parent, 'master') else None

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self._content.pack(fill=BOTH, expand=True)
            self._toggle_btn.configure(text=f"▼ {self._text}")
            # 重新绑定鼠标滚轮
            self._bind_mousewheel_recursive(self)
        else:
            self._content.pack_forget()
            self._toggle_btn.configure(text=f"▶ {self._text}")

    @property
    def content(self):
        return self._content


class DebugWindow(ttk.Toplevel):
    """调试窗口 - 三栏布局"""

    def __init__(self, parent, debug_dir: str):
        super().__init__(parent)
        self.title(f"调试窗口 - {Path(debug_dir).name}")
        self.geometry("1600x900")
        self.minsize(1200, 700)

        # 设置窗口图标
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            pass

        self.debug_dir = Path(debug_dir)
        self.current_frame_idx = 0
        self.frame_data = []
        self.vlm_frames = []
        self.photo_images = []
        self.vlm_photo_images = []
        self.thumb_photo_images = []

        self.is_scene_mode = False
        self.scenes: Dict[int, dict] = {}
        self.current_scene = 0
        self.scene_selector = None

        self._load_data()
        self._build_ui()
        self._display_first()

    def _load_scene_data(self, scene_dir: Path) -> dict:
        """加载单个场景的调试数据"""
        data = {"frame_data": [], "vlm_frames": [], "vlm_prompt": "", "vlm_response": ""}

        detection_dir = scene_dir / "detection"
        if detection_dir.exists():
            json_files = sorted(detection_dir.glob("*_result.json"))
            for jf in json_files:
                stem = jf.stem.replace("_result", "")
                original = detection_dir / f"{stem}_original.jpg"
                annotated = detection_dir / f"{stem}_annotated.jpg"
                with open(jf, "r", encoding="utf-8") as f:
                    frame_result = json.load(f)
                data["frame_data"].append({
                    "stem": stem,
                    "original": str(original) if original.exists() else None,
                    "annotated": str(annotated) if annotated.exists() else None,
                    "result": frame_result,
                })

        vlm_dir = scene_dir / "vlm_frames"
        if vlm_dir.exists():
            data["vlm_frames"] = sorted([str(f) for f in vlm_dir.glob("*.jpg")])

        prompt_file = scene_dir / "vlm_prompt.txt"
        response_file = scene_dir / "vlm_response.txt"
        if prompt_file.exists():
            data["vlm_prompt"] = prompt_file.read_text(encoding="utf-8")
        if response_file.exists():
            data["vlm_response"] = response_file.read_text(encoding="utf-8")

        return data

    def _apply_scene(self, scene_idx: int):
        """将指定场景的数据应用到当前显示变量"""
        scene = self.scenes.get(scene_idx)
        if not scene:
            return
        self.current_scene = scene_idx
        self.frame_data = scene["frame_data"]
        self.vlm_frames = scene["vlm_frames"]
        self.vlm_prompt = scene["vlm_prompt"]
        self.vlm_response = scene["vlm_response"]

    def _load_data(self):
        """加载调试数据"""
        # 检测是否场景模式
        first_scene = self.debug_dir / "scene_0" / "detection"
        if first_scene.exists():
            self.is_scene_mode = True
            scene_dirs = sorted(
                [d for d in self.debug_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")],
                key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
            )
            for sd in scene_dirs:
                scene_idx = int(sd.name.split("_")[1]) if sd.name.split("_")[1].isdigit() else 0
                self.scenes[scene_idx] = self._load_scene_data(sd)
            if self.scenes:
                self._apply_scene(0)
        else:
            # 平面模式（原有逻辑）
            detection_dir = self.debug_dir / "detection"
            if detection_dir.exists():
                json_files = sorted(detection_dir.glob("*_result.json"))
                for jf in json_files:
                    stem = jf.stem.replace("_result", "")
                    original = detection_dir / f"{stem}_original.jpg"
                    annotated = detection_dir / f"{stem}_annotated.jpg"
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.frame_data.append({
                        "stem": stem,
                        "original": str(original) if original.exists() else None,
                        "annotated": str(annotated) if annotated.exists() else None,
                        "result": data,
                    })

            vlm_dir = self.debug_dir / "vlm_frames"
            if vlm_dir.exists():
                self.vlm_frames = sorted([str(f) for f in vlm_dir.glob("*.jpg")])

            self.vlm_prompt = ""
            self.vlm_response = ""
            prompt_file = self.debug_dir / "vlm_prompt.txt"
            response_file = self.debug_dir / "vlm_response.txt"
            if prompt_file.exists():
                self.vlm_prompt = prompt_file.read_text(encoding="utf-8")
            if response_file.exists():
                self.vlm_response = response_file.read_text(encoding="utf-8")

        # 共同部分：汇总
        self.summary = {}
        summary_file = self.debug_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                self.summary = json.load(f)

        self.frame_timestamps = self.summary.get("frame_timestamps", [])

    def _build_ui(self):
        """构建三栏UI"""
        # 主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=5, pady=(5, 0))

        # 场景选择器（场景模式专用）
        if self.is_scene_mode:
            scene_top = ttk.Frame(main_frame)
            scene_top.pack(fill=X, pady=(0, 4))
            ttk.Label(scene_top, text="场景选择:", font=("Microsoft YaHei", 9, "bold")).pack(side=LEFT, padx=(0, 4))
            scene_keys = sorted(self.scenes.keys())
            scene_names = [f"场景 {k+1}" for k in scene_keys]
            self.scene_var = tk.StringVar(value=scene_names[0] if scene_names else "")
            self.scene_selector = ttk.Combobox(
                scene_top, textvariable=self.scene_var, values=scene_names, state="readonly", width=12
            )
            self.scene_selector.pack(side=LEFT, padx=4)
            self.scene_selector.bind("<<ComboboxSelected>>", lambda e: self._switch_scene())

        # 三栏 PanedWindow
        pane = ttk.Panedwindow(main_frame, orient=HORIZONTAL)
        pane.pack(fill=BOTH, expand=True)

        # ── 左栏：缩略图网格 ──
        left_frame = ttk.Frame(pane)
        pane.add(left_frame, weight=1)

        ttk.Label(
            left_frame, text="帧列表", font=("Microsoft YaHei", 10, "bold")
        ).pack(fill=X, padx=2, pady=(0, 4))

        thumb_container = ttk.Frame(left_frame)
        thumb_container.pack(fill=BOTH, expand=True)

        self.thumb_canvas = tk.Canvas(thumb_container, bg="#1e1e1e", highlightthickness=0)
        thumb_scrollbar = ttk.Scrollbar(thumb_container, orient=VERTICAL, command=self.thumb_canvas.yview)
        self.thumb_inner = ttk.Frame(self.thumb_canvas)

        self.thumb_inner.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")),
        )
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor=NW)
        self.thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)

        self.thumb_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        thumb_scrollbar.pack(side=RIGHT, fill=Y)

        # 鼠标滚轮（只在悬停缩略图区时生效）
        def _thumb_mousewheel(event):
            try:
                self.thumb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass

        self.thumb_canvas.bind("<MouseWheel>", _thumb_mousewheel)
        self.thumb_inner.bind("<MouseWheel>", lambda e: _thumb_mousewheel(e))

        self.thumb_labels = []

        # ── 中栏：大图预览 ──
        center_frame = ttk.Frame(pane)
        pane.add(center_frame, weight=3)

        self.img_title = ttk.Label(
            center_frame, text="检测结果", font=("Microsoft YaHei", 10, "bold")
        )
        self.img_title.pack(fill=X, pady=(0, 4))

        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b")
        h_scroll = ttk.Scrollbar(canvas_frame, orient=HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky=NSEW)
        v_scroll.grid(row=0, column=1, sticky=NS)
        h_scroll.grid(row=1, column=0, sticky=EW)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # 中栏大图滚轮
        def _center_mousewheel(event):
            try:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        self.canvas.bind("<MouseWheel>", _center_mousewheel)

        # ── 右栏：检测结果面板 ──
        right_frame = ttk.Frame(pane)
        pane.add(right_frame, weight=2)

        right_scroll_canvas = tk.Canvas(right_frame, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_frame, orient=VERTICAL, command=right_scroll_canvas.yview)
        right_inner = ttk.Frame(right_scroll_canvas)

        right_inner.bind(
            "<Configure>",
            lambda e: right_scroll_canvas.configure(scrollregion=right_scroll_canvas.bbox("all")),
        )
        right_scroll_canvas.create_window((0, 0), window=right_inner, anchor=NW)
        right_scroll_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_scroll_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        right_scrollbar.pack(side=RIGHT, fill=Y)

        # 右栏滚轮
        def _right_mousewheel(event):
            try:
                right_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass
        right_scroll_canvas.bind("<MouseWheel>", _right_mousewheel)
        right_inner.bind("<MouseWheel>", lambda e: _right_mousewheel(e))

        # 保存右栏滚动函数供子组件使用
        self._right_mousewheel = _right_mousewheel

        # 右栏内容：检测结果（展开）
        sec_detection = CollapsibleFrame(right_inner, text="检测结果", expanded=True)
        sec_detection.pack(fill=X, padx=2, pady=2)

        self.model_notebook = ttk.Notebook(sec_detection.content)
        self.model_notebook.pack(fill=BOTH, expand=True)

        # 综合概览子标签页
        self.overview_frame = ttk.Frame(self.model_notebook)
        self.model_notebook.add(self.overview_frame, text="综合概览")
        self.overview_text = scrolledtext.ScrolledText(self.overview_frame, wrap=WORD, font=("Consolas", 9))
        self.overview_text.pack(fill=BOTH, expand=True)

        # Detect模型子标签页
        self.detect_frame = ttk.Frame(self.model_notebook)
        self.model_notebook.add(self.detect_frame, text="Detect 检测")
        self.detect_text = scrolledtext.ScrolledText(self.detect_frame, wrap=WORD, font=("Consolas", 9))
        self.detect_text.pack(fill=BOTH, expand=True)

        # Pose模型子标签页
        self.pose_frame = ttk.Frame(self.model_notebook)
        self.model_notebook.add(self.pose_frame, text="Pose 姿态")
        self.pose_text = scrolledtext.ScrolledText(self.pose_frame, wrap=WORD, font=("Consolas", 9))
        self.pose_text.pack(fill=BOTH, expand=True)

        # Segment模型子标签页
        self.segment_frame = ttk.Frame(self.model_notebook)
        self.model_notebook.add(self.segment_frame, text="Segment 分割")
        self.segment_text = scrolledtext.ScrolledText(self.segment_frame, wrap=WORD, font=("Consolas", 9))
        self.segment_text.pack(fill=BOTH, expand=True)

        # 投票决策子标签页
        self.vote_frame = ttk.Frame(self.model_notebook)
        self.model_notebook.add(self.vote_frame, text="投票决策")
        self.vote_text = scrolledtext.ScrolledText(self.vote_frame, wrap=WORD, font=("Consolas", 9))
        self.vote_text.pack(fill=BOTH, expand=True)

        # 右栏内容：VLM 输入输出（折叠）
        sec_vlm = CollapsibleFrame(right_inner, text="VLM 输入输出", expanded=False)
        sec_vlm.pack(fill=X, padx=2, pady=2)

        # VLM Prompt
        ttk.Label(sec_vlm.content, text="Prompt:", font=("Microsoft YaHei", 9, "bold")).pack(
            fill=X, padx=2, pady=(2, 0)
        )
        self.prompt_text = scrolledtext.ScrolledText(sec_vlm.content, wrap=WORD, font=("Consolas", 9), height=6)
        self.prompt_text.pack(fill=BOTH, expand=True, padx=2, pady=2)

        # VLM响应
        ttk.Label(sec_vlm.content, text="Response:", font=("Microsoft YaHei", 9, "bold")).pack(
            fill=X, padx=2, pady=(2, 0)
        )
        self.response_text = scrolledtext.ScrolledText(sec_vlm.content, wrap=WORD, font=("Consolas", 9), height=6)
        self.response_text.pack(fill=BOTH, expand=True, padx=2, pady=2)

        # VLM帧
        ttk.Label(sec_vlm.content, text=f"VLM帧 ({len(self.vlm_frames)}):", font=("Microsoft YaHei", 9, "bold")).pack(
            fill=X, padx=2, pady=(2, 0)
        )
        self.vlm_frame_tab = sec_vlm.content
        self._build_vlm_frames_tab()

        # 右栏内容：汇总（折叠）
        sec_summary = CollapsibleFrame(right_inner, text="汇总", expanded=False)
        sec_summary.pack(fill=X, padx=2, pady=2)

        self.summary_text = scrolledtext.ScrolledText(sec_summary.content, wrap=WORD, font=("Consolas", 9), height=8)
        self.summary_text.pack(fill=BOTH, expand=True, padx=2, pady=2)

        # 填充静态文本
        if self.vlm_prompt:
            self.prompt_text.insert(END, self.vlm_prompt)
            self.prompt_text.configure(state=DISABLED)
        if self.vlm_response:
            self.response_text.insert(END, self.vlm_response)
            self.response_text.configure(state=DISABLED)
        if self.summary:
            self.summary_text.insert(END, json.dumps(self.summary, ensure_ascii=False, indent=2))
            self.summary_text.configure(state=DISABLED)

        # ── 底部导航栏 ──
        bottom = ttk.Frame(self)
        bottom.pack(fill=X, padx=5, pady=5)

        self.btn_prev = ttk.Button(bottom, text="< 上一帧", command=self._prev_frame)
        self.btn_prev.pack(side=LEFT, padx=5)

        self.frame_label = ttk.Label(bottom, text="帧 0/0")
        self.frame_label.pack(side=LEFT, padx=20)

        self.btn_next = ttk.Button(bottom, text="下一帧 >", command=self._next_frame)
        self.btn_next.pack(side=LEFT, padx=5)

        ttk.Separator(bottom, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)

        self.btn_save = ttk.Button(bottom, text="保存调试报告", command=self._save_report)
        self.btn_save.pack(side=LEFT, padx=5)

        ttk.Separator(bottom, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)

        self.show_var = tk.StringVar(value="annotated")
        ttk.Radiobutton(bottom, text="原始帧", variable=self.show_var, value="original", command=self._refresh_image).pack(side=LEFT)
        ttk.Radiobutton(bottom, text="标注帧", variable=self.show_var, value="annotated", command=self._refresh_image).pack(side=LEFT)

    def _build_vlm_frames_tab(self):
        """构建VLM帧缩略图网格"""
        if not self.vlm_frames:
            ttk.Label(self.vlm_frame_tab, text="无VLM帧数据").pack(pady=10)
            return

        vlm_grid_frame = ttk.Frame(self.vlm_frame_tab)
        vlm_grid_frame.pack(fill=BOTH, expand=True, padx=2, pady=2)

        cols = 4
        for i, frame_path in enumerate(self.vlm_frames):
            row, col = divmod(i, cols)
            try:
                img = Image.open(frame_path)
                img.thumbnail((120, 90))
                photo = ImageTk.PhotoImage(img)
                self.vlm_photo_images.append(photo)

                frame_container = ttk.Frame(vlm_grid_frame, relief="solid", borderwidth=1)
                frame_container.grid(row=row, column=col, padx=2, pady=2, sticky=NSEW)

                img_label = ttk.Label(frame_container, image=photo, cursor="hand2")
                img_label.pack(padx=1, pady=1)
                img_label.bind("<Button-1>", lambda e, p=frame_path, idx=i: self._preview_vlm_frame(p, idx))

                ts_str = ""
                if i < len(self.frame_timestamps):
                    ts_str = f" @{self.frame_timestamps[i]:.1f}s"
                name_label = ttk.Label(
                    frame_container,
                    text=f"#{i}{ts_str}",
                    font=("Consolas", 7),
                    cursor="hand2",
                )
                name_label.pack(padx=1, pady=(0, 1))
                name_label.bind("<Button-1>", lambda e, p=frame_path, idx=i: self._preview_vlm_frame(p, idx))

            except Exception as e:
                ttk.Label(vlm_grid_frame, text=f"#{i} 失败: {e}").grid(row=row, column=col)

    def _preview_vlm_frame(self, frame_path: str, idx: int):
        """在主画布中预览VLM帧"""
        if not Path(frame_path).exists():
            return

        try:
            img = Image.open(frame_path)
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                ratio = min(canvas_w / img.width, canvas_h / img.height, 1.0)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.photo_images.clear()
            self.photo_images.append(photo)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            self.img_title.configure(text=f"VLM帧 #{idx}: {Path(frame_path).name}")
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor=NW, text=f"图片加载失败: {e}", fill="red")

    def _switch_scene(self):
        """切换当前显示的场景"""
        if not self.scene_selector:
            return
        selection = self.scene_var.get()
        scene_keys = sorted(self.scenes.keys())
        for idx, name in [(k, f"场景 {k+1}") for k in scene_keys]:
            if name == selection:
                self._apply_scene(idx)
                break

        # 更新场景描述信息
        scene_info = self.scenes[self.current_scene]
        scene_result = self.summary.get("video_summary", {}).get("scene_results", [])
        for sr in scene_result:
            if sr.get("index") == self.current_scene:
                desc = sr.get("description", "")
                kw = sr.get("keywords", "")
                self.img_title.configure(text=f"场景 {self.current_scene+1}: {desc[:60]}...")
                break

        self._rebuild_display()

    def _rebuild_display(self):
        """重建显示（切换场景后刷新所有面板）"""
        # 清除旧内容
        for widget in self.thumb_inner.winfo_children():
            widget.destroy()
        for tab in [self.overview_text, self.detect_text, self.pose_text,
                     self.segment_text, self.vote_text]:
            tab.configure(state=NORMAL)
            tab.delete("1.0", END)
        self.canvas.delete("all")
        self.vlm_photo_images.clear()

        # 刷新
        self._build_thumbnails()
        if self.frame_data:
            self.current_frame_idx = 0
            self._display_frame(0)
        else:
            self.overview_text.insert(END, "无检测帧数据")

        # 刷新右栏 VLM 面板
        for widget in self.vlm_frame_tab.winfo_children():
            widget.destroy()
        self._build_vlm_frames_tab()
        if self.vlm_prompt:
            self.prompt_text.configure(state=NORMAL)
            self.prompt_text.delete("1.0", END)
            self.prompt_text.insert(END, self.vlm_prompt)
            self.prompt_text.configure(state=DISABLED)
        if self.vlm_response:
            self.response_text.configure(state=NORMAL)
            self.response_text.delete("1.0", END)
            self.response_text.insert(END, self.vlm_response)
            self.response_text.configure(state=DISABLED)

    def _build_thumbnails(self):
        """构建/重建左栏缩略图网格"""
        # 清除旧缩略图
        for lbl in self.thumb_labels:
            lbl.destroy()
        self.thumb_labels.clear()
        self.thumb_photo_images.clear()

        cols = 2
        for i, data in enumerate(self.frame_data):
            row, col = divmod(i, cols)
            img_path = data.get("annotated") or data.get("original")
            try:
                if img_path and Path(img_path).exists():
                    img = Image.open(img_path)
                    img.thumbnail((140, 100))
                    photo = ImageTk.PhotoImage(img)
                    self.thumb_photo_images.append(photo)

                    frame_container = ttk.Frame(self.thumb_inner, relief="solid", borderwidth=1)
                    frame_container.grid(row=row, column=col, padx=2, pady=2, sticky=NSEW)

                    img_label = ttk.Label(frame_container, image=photo, cursor="hand2")
                    img_label.pack(padx=1, pady=1)
                    img_label.bind("<Button-1>", lambda e, idx=i: self._display_frame(idx))

                    ts_str = ""
                    if i < len(self.frame_timestamps):
                        ts_str = f" @{self.frame_timestamps[i]:.1f}s"
                    name_label = ttk.Label(
                        frame_container,
                        text=f"#{i}{ts_str} {data['stem']}",
                        font=("Consolas", 7),
                        cursor="hand2",
                    )
                    name_label.pack(padx=1, pady=(0, 1))
                    name_label.bind("<Button-1>", lambda e, idx=i: self._display_frame(idx))

                    self.thumb_labels.append(frame_container)
                else:
                    placeholder = ttk.Label(self.thumb_inner, text=f"#{i} 无图片")
                    placeholder.grid(row=row, column=col, padx=2, pady=2)
                    self.thumb_labels.append(placeholder)
            except Exception as e:
                err_lbl = ttk.Label(self.thumb_inner, text=f"#{i} 失败")
                err_lbl.grid(row=row, column=col, padx=2, pady=2)
                self.thumb_labels.append(err_lbl)

    def _display_first(self):
        """显示第一帧"""
        if self.frame_data:
            self._build_thumbnails()
            self.current_frame_idx = 0
            self._display_frame(0)
        else:
            self.overview_text.insert(END, "无检测帧数据")

    def _display_frame(self, idx: int):
        """显示指定帧"""
        if idx < 0 or idx >= len(self.frame_data):
            return

        self.current_frame_idx = idx
        data = self.frame_data[idx]
        result = data["result"]

        self.frame_label.configure(text=f"帧 {idx + 1}/{len(self.frame_data)}")

        self._update_overview(result)
        self._update_detect_tab(result)
        self._update_pose_tab(result)
        self._update_segment_tab(result)
        self._update_vote_tab(result)
        self._refresh_image()

        # 高亮当前缩略图边框
        for i, lbl in enumerate(self.thumb_labels):
            try:
                if isinstance(lbl, ttk.Frame):
                    lbl.configure(relief="solid", borderwidth=2 if i == idx else 1)
            except Exception:
                pass

    def _update_overview(self, result: dict):
        """更新综合概览"""
        self.overview_text.configure(state=NORMAL)
        self.overview_text.delete("1.0", END)

        overview = {
            "timestamp": result.get("timestamp"),
            "has_person": result.get("has_person"),
            "confidence": result.get("confidence"),
            "models_used": result.get("models_used", []),
            "vote_count": result.get("vote_count"),
        }
        self.overview_text.insert(END, json.dumps(overview, ensure_ascii=False, indent=2))
        self.overview_text.configure(state=DISABLED)

    def _update_detect_tab(self, result: dict):
        """更新Detect模型标签页"""
        self.detect_text.configure(state=NORMAL)
        self.detect_text.delete("1.0", END)

        detection_details = result.get("detection_details", [])
        if detection_details:
            self.detect_text.insert(END, "【Detect 检测模型结果】\n")
            self.detect_text.insert(END, f"检测到 {len(detection_details)} 个人体\n\n")
            for i, person in enumerate(detection_details):
                self.detect_text.insert(END, f"--- 人体 {i + 1} ---\n")
                self.detect_text.insert(END, json.dumps(person, ensure_ascii=False, indent=2))
                self.detect_text.insert(END, "\n\n")
        else:
            self.detect_text.insert(END, "【Detect 检测模型结果】\n")
            self.detect_text.insert(END, "未检测到人体\n")
            self.detect_text.insert(
                END,
                f"\n原始数据:\n{json.dumps(result.get('detection_details', []), ensure_ascii=False, indent=2)}",
            )

        self.detect_text.configure(state=DISABLED)

    def _update_pose_tab(self, result: dict):
        """更新Pose模型标签页"""
        self.pose_text.configure(state=NORMAL)
        self.pose_text.delete("1.0", END)

        pose_analysis = result.get("pose_analysis", [])
        visible_keypoints = result.get("visible_keypoints", 0)
        keypoints = result.get("keypoints", {})

        self.pose_text.insert(END, "【Pose 姿态模型结果】\n\n")
        self.pose_text.insert(END, f"姿态分析: {', '.join(pose_analysis) if pose_analysis else '无'}\n")
        self.pose_text.insert(END, f"可见关键点: {visible_keypoints}/17\n\n")

        if keypoints:
            self.pose_text.insert(END, "【关键点详情】\n")
            self.pose_text.insert(END, f"{'名称':<20} {'X':>8} {'Y':>8} {'置信度':>8} {'状态':<6}\n")
            self.pose_text.insert(END, "-" * 60 + "\n")

            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle",
            ]

            for name in keypoint_names:
                if name in keypoints:
                    kpt = keypoints[name]
                    x = kpt.get("x", 0)
                    y = kpt.get("y", 0)
                    conf = kpt.get("conf", 0)
                    status = "可见" if conf > 0.5 else "遮挡"
                    self.pose_text.insert(END, f"{name:<20} {x:>8.1f} {y:>8.1f} {conf:>8.3f} {status:<6}\n")
                else:
                    self.pose_text.insert(END, f"{name:<20} {'--':>8} {'--':>8} {'--':>8} {'缺失':<6}\n")
        else:
            self.pose_text.insert(END, "无关键点数据\n")

        self.pose_text.configure(state=DISABLED)

    def _update_segment_tab(self, result: dict):
        """更新Segment模型标签页"""
        self.segment_text.configure(state=NORMAL)
        self.segment_text.delete("1.0", END)

        segment_details = result.get("segment_details", [])
        wearing_analysis = result.get("wearing_analysis", {})

        self.segment_text.insert(END, "【Segment 分割模型结果】\n\n")

        if segment_details:
            self.segment_text.insert(END, f"检测到 {len(segment_details)} 个分割区域\n\n")
            for i, seg in enumerate(segment_details):
                self.segment_text.insert(END, f"--- 分割区域 {i + 1} ---\n")
                self.segment_text.insert(END, json.dumps(seg, ensure_ascii=False, indent=2))
                self.segment_text.insert(END, "\n\n")
        else:
            self.segment_text.insert(END, "未检测到分割区域\n\n")

        self.segment_text.insert(END, "【穿着分析】\n")
        if wearing_analysis.get("has_wearing"):
            self.segment_text.insert(END, f"检测到穿着: 是\n")
            self.segment_text.insert(END, f"颜色方差: {wearing_analysis.get('color_variance', 0):.2f}\n")
            avg_color = wearing_analysis.get("avg_color_bgr", [])
            if avg_color:
                self.segment_text.insert(
                    END, f"平均颜色 (BGR): [{avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f}]\n"
                )
            self.segment_text.insert(END, f"\n完整数据:\n{json.dumps(wearing_analysis, ensure_ascii=False, indent=2)}\n")
        else:
            self.segment_text.insert(END, "未检测到穿着\n")

        self.segment_text.configure(state=DISABLED)

    def _update_vote_tab(self, result: dict):
        """更新投票决策标签页"""
        self.vote_text.configure(state=NORMAL)
        self.vote_text.delete("1.0", END)

        self.vote_text.insert(END, "【三模型投票决策】\n\n")

        models_used = result.get("models_used", [])
        self.vote_text.insert(END, f"使用的模型: {', '.join(models_used) if models_used else '无'}\n")
        self.vote_text.insert(END, f"投票数: {result.get('vote_count', 0)}/{len(models_used)}\n")
        self.vote_text.insert(END, f"最终结果: {'有人体' if result.get('has_person') else '无人体'}\n")
        self.vote_text.insert(END, f"加权置信度: {result.get('confidence', 0):.3f}\n\n")

        self.vote_text.insert(END, "【各模型投票详情】\n")
        self.vote_text.insert(END, f"{'模型':<15} {'检测结果':<10} {'置信度':<10} {'投票':<6}\n")
        self.vote_text.insert(END, "-" * 50 + "\n")

        detection_details = result.get("detection_details", [])
        det_has = len(detection_details) > 0
        det_conf = max([d.get("confidence", 0) for d in detection_details], default=0)
        self.vote_text.insert(
            END, f"{'Detect':<15} {'有人体' if det_has else '无人体':<10} {det_conf:<10.3f} {'+1' if det_has else ' 0':<6}\n"
        )

        pose_analysis = result.get("pose_analysis", [])
        visible_keypoints = result.get("visible_keypoints", 0)
        pose_has = len(pose_analysis) > 0 and "站立/正常姿态" not in pose_analysis
        pose_conf = visible_keypoints / 17 if visible_keypoints > 0 else 0
        self.vote_text.insert(
            END, f"{'Pose':<15} {'有人体' if pose_has else '无人体':<10} {pose_conf:<10.3f} {'+1' if pose_has else ' 0':<6}\n"
        )

        segment_details = result.get("segment_details", [])
        seg_has = len(segment_details) > 0
        seg_conf = max([s.get("confidence", 0) for s in segment_details], default=0)
        self.vote_text.insert(
            END, f"{'Segment':<15} {'有人体' if seg_has else '无人体':<10} {seg_conf:<10.3f} {'+1' if seg_has else ' 0':<6}\n"
        )

        self.vote_text.insert(END, "-" * 50 + "\n")
        vote_count = sum([1 if det_has else 0, 1 if pose_has else 0, 1 if seg_has else 0])
        self.vote_text.insert(END, f"{'总计':<15} {'':10} {'':10} {vote_count}/3\n\n")

        self.vote_text.insert(END, "【投票规则】\n")
        self.vote_text.insert(END, "- 至少 2 个模型检测到人体 → 最终结果为有人体\n")
        self.vote_text.insert(END, "- 加权置信度 = 各模型置信度的加权平均\n")
        self.vote_text.insert(END, "- 权重根据置信度动态调整\n")

        self.vote_text.configure(state=DISABLED)

    def _refresh_image(self):
        """刷新当前图片显示"""
        if not self.frame_data:
            return

        data = self.frame_data[self.current_frame_idx]
        mode = self.show_var.get()
        img_path = data.get(mode) or data.get("original")

        if not img_path or not Path(img_path).exists():
            return

        try:
            img = Image.open(img_path)
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                ratio = min(canvas_w / img.width, canvas_h / img.height, 1.0)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.photo_images.clear()
            self.photo_images.append(photo)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            self.img_title.configure(text="检测结果")
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor=NW, text=f"图片加载失败: {e}", fill="red")

    def _prev_frame(self):
        """上一帧"""
        if self.current_frame_idx > 0:
            self._display_frame(self.current_frame_idx - 1)

    def _next_frame(self):
        """下一帧"""
        if self.current_frame_idx < len(self.frame_data) - 1:
            self._display_frame(self.current_frame_idx + 1)

    def _save_report(self):
        """保存调试报告"""
        from tkinter import filedialog, messagebox

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialfile=f"debug_report_{self.debug_dir.name}.txt",
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"调试报告 - {self.debug_dir.name}\n")
                f.write("=" * 60 + "\n\n")

                f.write("【汇总信息】\n")
                f.write(json.dumps(self.summary, ensure_ascii=False, indent=2))
                f.write("\n\n")

                f.write("【检测结果】\n")
                for i, data in enumerate(self.frame_data):
                    f.write(f"\n--- 帧 {i + 1}: {data['stem']} ---\n")
                    f.write(json.dumps(data["result"], ensure_ascii=False, indent=2))
                    f.write("\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("【VLM Prompt】\n")
                f.write(self.vlm_prompt)
                f.write("\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("【VLM响应】\n")
                f.write(self.vlm_response)

            messagebox.showinfo("保存成功", f"报告已保存至:\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


def open_debug_window(parent, debug_dir: str):
    """打开调试窗口的便捷函数"""
    if not Path(debug_dir).exists():
        from tkinter import messagebox

        messagebox.showerror("错误", f"调试目录不存在: {debug_dir}")
        return
    DebugWindow(parent, debug_dir)
