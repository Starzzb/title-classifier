"""调试窗口 - 显示模型输入输出、检测结果、VLM帧"""

import json
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageTk
import cv2
import numpy as np


class DebugWindow(tk.Toplevel):
    """调试窗口"""

    def __init__(self, parent, debug_dir: str):
        super().__init__(parent)
        self.title(f"调试窗口 - {Path(debug_dir).name}")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        self.debug_dir = Path(debug_dir)
        self.current_frame_idx = 0
        self.frame_data = []  # 检测帧数据列表
        self.vlm_frames = []  # VLM输入帧列表
        self.photo_images = []  # 防止GC回收

        self._load_data()
        self._build_ui()
        self._display_first()

    def _load_data(self):
        """加载调试数据"""
        # 加载检测帧数据
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

        # 加载VLM帧列表
        vlm_dir = self.debug_dir / "vlm_frames"
        if vlm_dir.exists():
            self.vlm_frames = sorted([str(f) for f in vlm_dir.glob("*.jpg")])

        # 加载VLM prompt和response
        self.vlm_prompt = ""
        self.vlm_response = ""
        prompt_file = self.debug_dir / "vlm_prompt.txt"
        response_file = self.debug_dir / "vlm_response.txt"
        if prompt_file.exists():
            self.vlm_prompt = prompt_file.read_text(encoding="utf-8")
        if response_file.exists():
            self.vlm_response = response_file.read_text(encoding="utf-8")

        # 加载汇总信息
        self.summary = {}
        summary_file = self.debug_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                self.summary = json.load(f)

    def _build_ui(self):
        """构建UI"""
        # 主分割
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：图片显示区
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        # 图片标题
        self.img_title = ttk.Label(left_frame, text="检测结果", font=("Microsoft YaHei", 10, "bold"))
        self.img_title.pack(fill=tk.X, pady=(0, 5))

        # 图片Canvas（带滚动条）
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b")
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # 右侧：信息区
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)

        # 信息区Notebook
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 模型输出标签页
        self.json_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.json_frame, text="模型输出")
        self.json_text = scrolledtext.ScrolledText(self.json_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.json_text.pack(fill=tk.BOTH, expand=True)

        # VLM Prompt标签页
        self.prompt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prompt_frame, text="VLM Prompt")
        self.prompt_text = scrolledtext.ScrolledText(self.prompt_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # VLM响应标签页
        self.response_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.response_frame, text="VLM响应")
        self.response_text = scrolledtext.ScrolledText(self.response_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # VLM帧标签页
        self.vlm_frame_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vlm_frame_tab, text=f"VLM帧 ({len(self.vlm_frames)})")
        self._build_vlm_frames_tab()

        # 汇总标签页
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="汇总")
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # 底部导航
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=5, pady=5)

        self.btn_prev = ttk.Button(bottom, text="< 上一帧", command=self._prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.frame_label = ttk.Label(bottom, text="帧 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=20)

        self.btn_next = ttk.Button(bottom, text="下一帧 >", command=self._next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        ttk.Separator(bottom, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.btn_save = ttk.Button(bottom, text="保存调试报告", command=self._save_report)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # 显示切换
        ttk.Separator(bottom, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.show_var = tk.StringVar(value="annotated")
        ttk.Radiobutton(bottom, text="原始帧", variable=self.show_var, value="original", command=self._refresh_image).pack(side=tk.LEFT)
        ttk.Radiobutton(bottom, text="标注帧", variable=self.show_var, value="annotated", command=self._refresh_image).pack(side=tk.LEFT)

        # 填充VLM Prompt和Response
        if self.vlm_prompt:
            self.prompt_text.insert(tk.END, self.vlm_prompt)
            self.prompt_text.configure(state="disabled")
        if self.vlm_response:
            self.response_text.insert(tk.END, self.vlm_response)
            self.response_text.configure(state="disabled")

        # 填充汇总
        if self.summary:
            self.summary_text.insert(tk.END, json.dumps(self.summary, ensure_ascii=False, indent=2))
            self.summary_text.configure(state="disabled")

    def _build_vlm_frames_tab(self):
        """构建VLM帧预览标签页"""
        if not self.vlm_frames:
            ttk.Label(self.vlm_frame_tab, text="无VLM帧数据").pack(pady=20)
            return

        # 滚动区域
        canvas = tk.Canvas(self.vlm_frame_tab)
        scrollbar = ttk.Scrollbar(self.vlm_frame_tab, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 显示VLM帧缩略图
        cols = 3
        for i, frame_path in enumerate(self.vlm_frames):
            row, col = divmod(i, cols)
            try:
                img = Image.open(frame_path)
                img.thumbnail((200, 150))
                photo = ImageTk.PhotoImage(img)
                self.photo_images.append(photo)

                label = ttk.Label(scroll_frame, image=photo)
                label.grid(row=row * 2, column=col, padx=5, pady=2)

                name_label = ttk.Label(scroll_frame, text=Path(frame_path).name, font=("Consolas", 8))
                name_label.grid(row=row * 2 + 1, column=col, padx=5, pady=(0, 5))
            except Exception as e:
                ttk.Label(scroll_frame, text=f"加载失败: {e}").grid(row=row * 2, column=col)

    def _display_first(self):
        """显示第一帧"""
        if self.frame_data:
            self.current_frame_idx = 0
            self._display_frame(0)
        else:
            self.json_text.insert(tk.END, "无检测帧数据")

    def _display_frame(self, idx: int):
        """显示指定帧"""
        if idx < 0 or idx >= len(self.frame_data):
            return

        self.current_frame_idx = idx
        data = self.frame_data[idx]

        # 更新帧标签
        self.frame_label.configure(text=f"帧 {idx + 1}/{len(self.frame_data)}")

        # 更新模型输出
        self.json_text.configure(state="normal")
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(data["result"], ensure_ascii=False, indent=2))
        self.json_text.configure(state="disabled")

        # 更新图片
        self._refresh_image()

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
            # 自适应缩放
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
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor=tk.NW, text=f"图片加载失败: {e}", fill="red")

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
            initialfile=f"debug_report_{self.debug_dir.name}.txt"
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"调试报告 - {self.debug_dir.name}\n")
                f.write("=" * 60 + "\n\n")

                # 汇总
                f.write("【汇总信息】\n")
                f.write(json.dumps(self.summary, ensure_ascii=False, indent=2))
                f.write("\n\n")

                # 各帧检测结果
                f.write("【检测结果】\n")
                for i, data in enumerate(self.frame_data):
                    f.write(f"\n--- 帧 {i + 1}: {data['stem']} ---\n")
                    f.write(json.dumps(data["result"], ensure_ascii=False, indent=2))
                    f.write("\n")

                # VLM Prompt
                f.write("\n" + "=" * 60 + "\n")
                f.write("【VLM Prompt】\n")
                f.write(self.vlm_prompt)
                f.write("\n")

                # VLM响应
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
