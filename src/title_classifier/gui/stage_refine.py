"""Stage1b AI优化 Tab - 标题AI优化与手动编辑（重构版）"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from pathlib import Path
import threading
from collections import deque

from .app import ToolTip
from ..providers import get_providers_for_gui
from ..core.refiner import Refiner
from ..utils.atomic_csv import atomic_write_csv, safe_read_csv

DEFAULT_CSV = "data/output/title_review.csv"


class StageRefineTab(ttk.Frame):
    """Stage1b: AI标题优化标签页"""

    def __init__(self, parent, ctx, run_command_callback=None, sync_csv_to_db=None):
        super().__init__(parent)
        self.ctx = ctx
        self._run_command = run_command_callback
        self._sync_csv_to_db = sync_csv_to_db

        # 实例变量
        self.s1b_csv_var = self.ctx.csv_var
        self.s1b_provider_var = tk.StringVar(value="gcli")
        self.s1b_progress_var = tk.DoubleVar(value=0.0)
        self.s1b_results = {}
        self.s1b_modified = set()
        self._refine_buttons = []

        # 过滤状态
        self._filter_needs_vision = tk.BooleanVar(value=False)
        self._filter_modified = tk.BooleanVar(value=False)
        self._filter_audio = tk.BooleanVar(value=False)
        self.s1b_search_var = tk.StringVar()

        # 撤销历史栈
        self._undo_stack = deque(maxlen=10)

        self._build_ui()

    def _build_ui(self):
        # ===== ① 文件栏 =====
        file_bar = ttk.Frame(self)
        file_bar.pack(fill=tk.X, padx=4, pady=(4, 2))

        ttk.Label(file_bar, text="CSV:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(file_bar, textvariable=self.s1b_csv_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_bar, text="浏览", width=5, command=self._browse_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_bar, text="加载", width=5, command=self._load_preview).pack(side=tk.LEFT, padx=2)

        ttk.Separator(file_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(file_bar, text="AI:").pack(side=tk.LEFT, padx=(0, 2))
        providers = get_providers_for_gui("1b")
        ttk.Combobox(file_bar, textvariable=self.s1b_provider_var, values=providers, state="readonly", width=8).pack(side=tk.LEFT, padx=2)

        self.s1b_modified_label = ttk.Label(file_bar, text="已修改 0/0", foreground="#888888")
        self.s1b_modified_label.pack(side=tk.RIGHT, padx=8)

        # ===== ② 操作栏（分组） =====
        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, padx=4, pady=(0, 2))

        # -- AI操作组 --
        ai_group = ttk.LabelFrame(btn_bar, text="AI操作")
        ai_group.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(ai_group, text="优化选中", command=self._run_refine_selected).pack(side=tk.LEFT, padx=1, pady=2)
        ttk.Button(ai_group, text="优化全部", command=self._run_refine_all).pack(side=tk.LEFT, padx=1, pady=2)
        ttk.Button(ai_group, text="填入原标题", command=self._fill_original_smart).pack(side=tk.LEFT, padx=1, pady=2)

        # -- 编辑组 --
        edit_group = ttk.LabelFrame(btn_bar, text="编辑")
        edit_group.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(edit_group, text="编辑", width=4, command=self._edit_selected).pack(side=tk.LEFT, padx=1, pady=2)
        ttk.Button(edit_group, text="删除", width=4, command=self._delete_selected).pack(side=tk.LEFT, padx=1, pady=2)

        # -- 过滤组 --
        filter_group = ttk.LabelFrame(btn_bar, text="过滤")
        filter_group.pack(side=tk.LEFT, padx=(0, 4))

        filter_btn = ttk.Menubutton(filter_group, text="过滤条件 ▾")
        filter_btn.pack(side=tk.LEFT, padx=1)
        filter_menu = tk.Menu(filter_btn, tearoff=0)
        filter_btn["menu"] = filter_menu
        filter_menu.add_checkbutton(label="只显示 needs_vision=FALSE", variable=self._filter_needs_vision, command=self._apply_filters)
        filter_menu.add_checkbutton(label="只显示 audio_recognized=TRUE", variable=self._filter_audio, command=self._apply_filters)
        filter_menu.add_checkbutton(label="只显示已修改", variable=self._filter_modified, command=self._apply_filters)

        # 搜索框 + 进度条（右侧）
        ttk.Label(btn_bar, text="搜索:").pack(side=tk.RIGHT, padx=(4, 2))
        search_entry = ttk.Entry(btn_bar, textvariable=self.s1b_search_var, width=18)
        search_entry.pack(side=tk.RIGHT, padx=2)
        search_entry.bind("<KeyRelease>", lambda e: self._apply_filters())

        self.s1b_progress_label = ttk.Label(btn_bar, text="", width=10)
        self.s1b_progress_label.pack(side=tk.RIGHT, padx=2)
        self.s1b_progress_bar = ttk.Progressbar(btn_bar, variable=self.s1b_progress_var, maximum=100, length=100)
        self.s1b_progress_bar.pack(side=tk.RIGHT, padx=2)

        # ===== ③ 表格 =====
        preview_frame = ttk.LabelFrame(self, text="优化结果预览（双击编辑，右键更多操作）")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        columns = ("original", "needs_vision", "audio_recognized", "refined", "preview")
        self.s1b_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", selectmode="extended")
        self.s1b_tree.heading("original", text="原始标题")
        self.s1b_tree.heading("needs_vision", text="视觉")
        self.s1b_tree.heading("audio_recognized", text="音频")
        self.s1b_tree.heading("refined", text="AI优化结果")
        self.s1b_tree.heading("preview", text="最终文件名预览")
        self.s1b_tree.column("original", width=180)
        self.s1b_tree.column("needs_vision", width=45, anchor="center")
        self.s1b_tree.column("audio_recognized", width=45, anchor="center")
        self.s1b_tree.column("refined", width=180)
        self.s1b_tree.column("preview", width=220)

        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.s1b_tree.yview)
        self.s1b_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.s1b_tree.pack(fill=tk.BOTH, expand=True)

        # 修改行高亮样式（加深）
        self.s1b_tree.tag_configure("modified", background="#c8e0ff")

        # 右键菜单
        self.s1b_context_menu = tk.Menu(self, tearoff=0)
        self.s1b_context_menu.add_command(label="编辑标题", command=self._edit_selected)
        self.s1b_context_menu.add_command(label="采用原标题", command=self._use_original)
        self.s1b_context_menu.add_command(label="重置为原标题（取消优化）", command=self._reset_to_original)
        self.s1b_context_menu.add_separator()
        self.s1b_ctx_vision_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="需要视觉识别: -", command=self._toggle_needs_vision)
        self.s1b_ctx_audio_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="音频已识别: -", command=self._toggle_audio_recognized)
        self.s1b_context_menu.add_separator()
        self.s1b_context_menu.add_command(label="批量→需要视觉", command=lambda: self._batch_needs_vision("TRUE"))
        self.s1b_context_menu.add_command(label="批量→不需要视觉", command=lambda: self._batch_needs_vision("FALSE"))
        self.s1b_context_menu.add_command(label="批量→反选", command=lambda: self._batch_needs_vision("INVERT"))
        self.s1b_context_menu.add_separator()
        self.s1b_context_menu.add_command(label="填入原标题(选中)", command=self._fill_original_selected)
        self.s1b_context_menu.add_command(label="填入原标题(全部)", command=self._fill_original_all)
        self.s1b_context_menu.add_separator()
        self.s1b_context_menu.add_command(label="删除选中行", command=self._delete_selected)

        self.s1b_tree.bind("<Button-3>", self._show_context_menu)
        self.s1b_tree.bind("<Double-1>", self._edit_selected)

        # ===== ④ 底部状态栏 =====
        bottom_bar = ttk.Frame(self)
        bottom_bar.pack(fill=tk.X, padx=4, pady=(0, 4))

        ttk.Button(bottom_bar, text="确认写入CSV", command=self._confirm_results, bootstyle="success").pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_bar, text="撤销", command=self._undo).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bottom_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.s1b_filter_label = ttk.Label(bottom_bar, text="过滤: 无", foreground="#888888")
        self.s1b_filter_label.pack(side=tk.LEFT, padx=4)

        # 全选/取消全选（右侧）
        ttk.Button(bottom_bar, text="全选", width=5, command=self._select_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bottom_bar, text="取消全选", width=6, command=self._deselect_all).pack(side=tk.RIGHT, padx=2)

    # ==================== 浏览 / 加载 ====================

    def _browse_csv(self):
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1b_csv_var.set(file_path)

    def _load_preview(self):
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            messagebox.showwarning("警告", "CSV文件不存在")
            return

        try:
            import csv
            for item in self.s1b_tree.get_children():
                self.s1b_tree.delete(item)
            self.s1b_results.clear()
            self.s1b_modified.clear()
            self._undo_stack.clear()

            filter_vision = self._filter_needs_vision.get()

            loaded_count = 0
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    needs_vision = row.get("needs_vision", "").strip().lower()

                    if filter_vision and needs_vision == "true":
                        continue

                    original = row.get("original_title", "")
                    final = row.get("final_name", "")
                    audio_recognized = row.get("audio_recognized", "").strip().lower()

                    preview = self._compute_preview(original, final)

                    item_id = self.s1b_tree.insert("", tk.END, values=(
                        original,
                        needs_vision.upper(),
                        audio_recognized.upper() if audio_recognized else "FALSE",
                        final,
                        preview,
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
            self._update_modified_counter()
            self._update_filter_label()

        except Exception as e:
            print(f"[错误] 加载CSV失败: {e}")

    # ==================== 预览计算 ====================

    def _compute_preview(self, original: str, final_name: str) -> str:
        """计算最终文件名预览：[关键词]_原标题"""
        if not final_name or final_name == original:
            return original
        if final_name.startswith("["):
            return f"{final_name}_{original}"
        return f"[{final_name}]_{original}"

    def _refresh_preview(self, item_id):
        """刷新单行的预览列"""
        values = self.s1b_tree.item(item_id, "values")
        original = values[0]
        refined = values[3]
        preview = self._compute_preview(original, refined)
        self.s1b_tree.item(item_id, values=(original, values[1], values[2], refined, preview))

    # ==================== 过滤 ====================

    def _apply_filters(self):
        """综合所有过滤条件"""
        query = self.s1b_search_var.get().lower().strip()
        show_vision_only = self._filter_needs_vision.get()
        show_audio_only = self._filter_audio.get()
        show_modified_only = self._filter_modified.get()

        for item in self.s1b_tree.get_children():
            values = self.s1b_tree.item(item, "values")
            visible = True

            # 搜索过滤
            if query:
                original = str(values[0]).lower() if values[0] else ""
                refined = str(values[3]).lower() if len(values) > 3 and values[3] else ""
                if query not in original and query not in refined:
                    visible = False

            # needs_vision=FALSE 过滤
            if visible and show_vision_only:
                if str(values[1]).upper() != "FALSE":
                    visible = False

            # audio_recognized=TRUE 过滤
            if visible and show_audio_only:
                if str(values[2]).upper() != "TRUE":
                    visible = False

            # 已修改过滤
            if visible and show_modified_only:
                if item not in self.s1b_modified:
                    visible = False

            if visible:
                self.s1b_tree.reattach(item, "", "end")
            else:
                self.s1b_tree.detach(item)

        self._update_filter_label()

    def _update_filter_label(self):
        """更新底部过滤条件显示"""
        parts = []
        if self._filter_needs_vision.get():
            parts.append("needs_vision=FALSE")
        if self._filter_audio.get():
            parts.append("audio_recognized=TRUE")
        if self._filter_modified.get():
            parts.append("已修改")
        if self.s1b_search_var.get().strip():
            parts.append(f'搜索="{self.s1b_search_var.get().strip()}"')
        text = "过滤: " + ", ".join(parts) if parts else "过滤: 无"
        self.s1b_filter_label.configure(text=text)

    def _update_modified_counter(self):
        total = len(self.s1b_tree.get_children())
        modified = len(self.s1b_modified)
        self.s1b_modified_label.configure(text=f"已修改 {modified}/{total}")

    # ==================== 标记修改 ====================

    def _mark_modified(self, item_id, push_undo=True):
        if push_undo:
            self._push_undo(item_id)
        self.s1b_modified.add(item_id)
        self.s1b_tree.item(item_id, tags=("modified",))
        self._update_modified_counter()

    def _push_undo(self, item_id):
        """记录撤销快照"""
        if item_id in self.s1b_results:
            result = self.s1b_results[item_id].copy()
            values = self.s1b_tree.item(item_id, "values")
            self._undo_stack.append((item_id, result, values))

    def _undo(self):
        """撤销最近一次修改"""
        if not self._undo_stack:
            print("[提示] 没有可撤销的操作")
            return

        item_id, result, values = self._undo_stack.pop()

        # 检查 item 是否还存在
        if item_id not in self.s1b_results:
            return

        self.s1b_results[item_id] = result
        self.s1b_tree.item(item_id, values=values)

        # 如果恢复到未修改状态
        if result["final_name"] == result["original_title"]:
            self.s1b_modified.discard(item_id)
            self.s1b_tree.item(item_id, tags=())
        else:
            self.s1b_tree.item(item_id, tags=("modified",))

        self._update_modified_counter()
        print("[完成] 已撤销上一步操作")

    # ==================== 智能操作 ====================

    def _fill_original_smart(self):
        selected = self.s1b_tree.selection()
        if selected:
            self._fill_original_selected()
        else:
            self._fill_original_all()

    # ==================== AI优化 ====================

    def _run_refine_selected(self):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要优化的行")
            return
        self._run_refine(list(selected))

    def _run_refine_all(self):
        all_items = self.s1b_tree.get_children()
        if not all_items:
            messagebox.showwarning("警告", "表格为空，请先加载CSV")
            return
        total = len(all_items)
        if not messagebox.askyesno("确认", f"确定要优化所有 {total} 条记录吗？"):
            return
        self._run_refine(list(all_items))

    def _run_refine(self, item_ids):
        provider = self.s1b_provider_var.get()

        items_to_refine = []
        for item_id in item_ids:
            values = self.s1b_tree.item(item_id, "values")
            original = values[0]
            needs_vision = values[1]
            stem = Path(original).stem
            items_to_refine.append((item_id, original, needs_vision, stem))

        total = len(items_to_refine)
        print(f"[开始] 正在优化 {total} 条记录...")

        self._set_refine_buttons_state("disabled")
        self.s1b_progress_var.set(0)
        self.s1b_progress_label.config(text="0%")

        def run():
            try:
                refiner = Refiner(provider=provider)

                def on_progress(current, total_count, title):
                    pct = current / total_count * 100
                    self.after(0, lambda: self.s1b_progress_var.set(pct))
                    self.after(0, lambda: self.s1b_progress_label.config(text=f"{pct:.0f}%"))

                stems = [t[3] for t in items_to_refine]
                refined_stems = refiner.refine_batch(stems, progress_callback=on_progress)

                def update_gui():
                    for (item_id, original, needs_vision, _), refined in zip(items_to_refine, refined_stems):
                        audio_recognized = self.s1b_tree.item(item_id, "values")[2]
                        preview = self._compute_preview(original, refined)
                        self.s1b_tree.item(item_id, values=(original, needs_vision, audio_recognized, refined, preview))
                        if item_id in self.s1b_results:
                            self.s1b_results[item_id]["final_name"] = refined
                        self._mark_modified(item_id)
                    print(f"[完成] 已优化 {total} 条记录")
                    self._set_refine_buttons_state("normal")
                    self.s1b_progress_var.set(100)
                    self.s1b_progress_label.config(text="100%")

                self.after(0, update_gui)

            except Exception as e:
                def show_error():
                    print(f"[错误] AI优化失败: {e}")
                    messagebox.showerror("错误", f"AI优化失败: {e}")
                    self._set_refine_buttons_state("normal")
                    self.s1b_progress_label.config(text="失败")
                self.after(0, show_error)

        threading.Thread(target=run, daemon=True).start()

    def _set_refine_buttons_state(self, state):
        for btn in self._refine_buttons:
            btn.configure(state=state)

    # ==================== 行操作 ====================

    def _show_context_menu(self, event):
        item = self.s1b_tree.identify_row(event.y)
        if not item:
            return
        self.s1b_tree.selection_set(item)
        values = self.s1b_tree.item(item, "values")

        nv = values[1]
        ar = values[2]
        nv_next = "FALSE" if nv.upper() == "TRUE" else "TRUE"
        ar_next = "FALSE" if ar.upper() == "TRUE" else "TRUE"
        self.s1b_context_menu.entryconfigure(self.s1b_ctx_vision_idx, label=f"需要视觉识别: {nv} → {nv_next}")
        self.s1b_context_menu.entryconfigure(self.s1b_ctx_audio_idx, label=f"音频已识别: {ar} → {ar_next}")

        self.s1b_context_menu.post(event.x_root, event.y_root)

    def _edit_selected(self, event=None):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")

        dialog = tk.Toplevel(self)
        dialog.title("编辑标题")
        dialog.geometry("500x150")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        ttk.Label(dialog, text="原始标题:").grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)
        ttk.Label(dialog, text=values[0][:50]).grid(row=0, column=1, padx=4, pady=4, sticky=tk.W)

        ttk.Label(dialog, text="优化结果:").grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)
        edit_var = tk.StringVar(value=values[3])
        ttk.Entry(dialog, textvariable=edit_var, width=50).grid(row=1, column=1, padx=4, pady=4)

        def confirm():
            new_value = edit_var.get()
            preview = self._compute_preview(values[0], new_value)
            self.s1b_tree.item(item, values=(values[0], values[1], values[2], new_value, preview))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = new_value
            self._mark_modified(item)
            dialog.destroy()

        ttk.Button(dialog, text="确认", command=confirm).grid(row=2, column=0, columnspan=2, pady=8)

    def _use_original(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        original = values[0]
        self.s1b_tree.item(item, values=(original, values[1], values[2], original, original))
        if item in self.s1b_results:
            self.s1b_results[item]["final_name"] = original
        self._mark_modified(item)

    def _reset_to_original(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        for item in selected:
            values = self.s1b_tree.item(item, "values")
            original = values[0]
            self.s1b_tree.item(item, values=(original, values[1], values[2], original, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
        self._update_modified_counter()

    # ==================== 状态切换 ====================

    def _toggle_needs_vision(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[1].upper()
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            preview = self._compute_preview(values[0], values[3])
            self.s1b_tree.item(item, values=(values[0], new_value, values[2], values[3], preview))
            if item in self.s1b_results:
                self.s1b_results[item]["needs_vision"] = new_value.lower()
            count += 1

        self._write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的needs_vision值")

    def _toggle_audio_recognized(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[2].upper()
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            preview = self._compute_preview(values[0], values[3])
            self.s1b_tree.item(item, values=(values[0], values[1], new_value, values[3], preview))
            if item in self.s1b_results:
                self.s1b_results[item]["audio_recognized"] = new_value.lower()
            count += 1

        self._write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的audio_recognized值")

    def _batch_needs_vision(self, mode: str):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[1].upper()
            if mode == "INVERT":
                new_value = "FALSE" if current == "TRUE" else "TRUE"
            else:
                new_value = mode
            preview = self._compute_preview(values[0], values[3])
            self.s1b_tree.item(item, values=(values[0], new_value, values[2], values[3], preview))
            if item in self.s1b_results:
                self.s1b_results[item]["needs_vision"] = new_value.lower()
            count += 1

        self._write_toggles_to_csv()
        label = {"TRUE": "需要视觉", "FALSE": "不需要视觉", "INVERT": "反选"}[mode]
        print(f"[完成] 已将 {count} 条记录设为 {label}")

    def _select_all(self):
        for item in self.s1b_tree.get_children():
            self.s1b_tree.selection_add(item)

    def _deselect_all(self):
        self.s1b_tree.selection_remove(*self.s1b_tree.get_children())

    # ==================== CSV写入 ====================

    def _write_toggles_to_csv(self):
        csv_path = self.s1b_csv_var.get()
        if not Path(csv_path).exists():
            return

        try:
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

    def _fill_original_selected(self):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要填入的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            original = values[0]
            self.s1b_tree.item(item, values=(original, values[1], values[2], original, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        self._update_modified_counter()
        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _fill_original_all(self):
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
            self.s1b_tree.item(item, values=(original, values[1], values[2], original, original))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = original
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        self._update_modified_counter()
        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _delete_selected(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        for item in selected:
            self.s1b_tree.delete(item)
            if item in self.s1b_results:
                del self.s1b_results[item]
            self.s1b_modified.discard(item)
        self._update_modified_counter()

    # ==================== 确认写入 ====================

    def _confirm_results(self):
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
            fieldnames, rows = safe_read_csv(csv_path)
            if not rows:
                return

            updated = 0
            for item_id in self.s1b_modified:
                if item_id not in self.s1b_results:
                    continue
                result = self.s1b_results[item_id]
                row_index = result["row_index"]
                if row_index < len(rows):
                    final_name = result["final_name"]
                    original_title = result.get("original_title", "")
                    if final_name and final_name != original_title:
                        if not final_name.startswith("["):
                            final_name = f"[{final_name}]"
                        final_name = f"{final_name}_{original_title}"
                    rows[row_index]["final_name"] = final_name
                    updated += 1

            atomic_write_csv(csv_path, rows, fieldnames)

            self.s1b_modified.clear()
            self._undo_stack.clear()
            for item_id in self.s1b_tree.get_children():
                self.s1b_tree.item(item_id, tags=())
            self._update_modified_counter()

            print(f"[完成] 已更新 {updated} 条记录")
            messagebox.showinfo("完成", f"已更新 {updated} 条记录")

        except Exception as e:
            print(f"[错误] 写入CSV失败: {e}")
            messagebox.showerror("错误", str(e))

    # ==================== 数据库同步 ====================

    def sync_csv_to_db(self, csv_path: str):
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
            print("[数据库] 扫描结果已同步")
        except Exception as e:
            print(f"[警告] 数据库同步失败: {e}")
