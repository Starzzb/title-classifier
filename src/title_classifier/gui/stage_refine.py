"""Stage1b AI优化 Tab - 标题AI优化与手动编辑"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from pathlib import Path
import threading

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
        self.s1b_filter_vision_var = tk.BooleanVar(value=True)
        self.s1b_progress_var = tk.DoubleVar(value=0.0)
        self.s1b_results = {}
        self.s1b_modified = set()
        self._refine_buttons = []

        self._build_ui()

    def _build_ui(self):
        # 工具栏
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 2))

        # CSV 文件选择
        ttk.Label(toolbar, text="CSV:").pack(side=tk.LEFT, padx=(0, 2))
        csv_entry = ttk.Entry(toolbar, textvariable=self.s1b_csv_var, width=40)
        csv_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="浏览", width=5, command=self._browse_csv_s1b).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="加载", width=5, command=self._load_s1b_preview).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Provider
        ttk.Label(toolbar, text="AI:").pack(side=tk.LEFT, padx=(0, 2))
        providers = get_providers_for_gui("1b")
        ttk.Combobox(toolbar, textvariable=self.s1b_provider_var, values=providers, state="readonly", width=8).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 核心操作按钮
        ttk.Button(toolbar, text="AI优化选中", command=self._run_refine_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="AI优化全部", command=self._run_refine_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="确认写入", command=self._confirm_s1b_results).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # 更多操作下拉
        more_btn = ttk.Menubutton(toolbar, text="更多 ▾")
        more_btn.pack(side=tk.LEFT, padx=2)
        more_menu = tk.Menu(more_btn, tearoff=0)
        more_btn["menu"] = more_menu
        more_menu.add_command(label="填入原标题(选中)", command=self._s1b_fill_original_selected)
        more_menu.add_command(label="填入原标题(全部)", command=self._s1b_fill_original_all)
        more_menu.add_separator()
        more_menu.add_command(label="选中行→需要视觉", command=lambda: self._s1b_batch_needs_vision("TRUE"))
        more_menu.add_command(label="选中行→不需要视觉", command=lambda: self._s1b_batch_needs_vision("FALSE"))
        more_menu.add_command(label="选中行→反选", command=lambda: self._s1b_batch_needs_vision("INVERT"))
        more_menu.add_separator()
        more_menu.add_command(label="全选", command=self._s1b_select_all)
        more_menu.add_command(label="取消全选", command=self._s1b_deselect_all)

        # 搜索框（右侧）
        ttk.Label(toolbar, text="搜索:").pack(side=tk.RIGHT, padx=(4, 2))
        self.s1b_search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.s1b_search_var, width=20)
        search_entry.pack(side=tk.RIGHT, padx=2)
        search_entry.bind("<KeyRelease>", self._on_search_changed)

        # 过滤选项 + 进度条
        filter_bar = ttk.Frame(self)
        filter_bar.pack(fill=tk.X, padx=4, pady=(0, 2))
        ttk.Checkbutton(filter_bar, text="只加载 needs_vision=FALSE", variable=self.s1b_filter_vision_var).pack(side=tk.LEFT)
        self.s1b_progress_bar = ttk.Progressbar(filter_bar, variable=self.s1b_progress_var, maximum=100, length=120)
        self.s1b_progress_bar.pack(side=tk.RIGHT, padx=4)
        self.s1b_progress_label = ttk.Label(filter_bar, text="", width=12)
        self.s1b_progress_label.pack(side=tk.RIGHT)

        # 预览表格
        preview_frame = ttk.LabelFrame(self, text="优化结果预览（右键菜单可编辑）")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

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

        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.s1b_tree.yview)
        self.s1b_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.s1b_tree.pack(fill=tk.BOTH, expand=True)

        # 右键菜单
        self.s1b_context_menu = tk.Menu(self, tearoff=0)

        self.s1b_context_menu.add_command(label="编辑标题", command=self._s1b_edit)
        self.s1b_context_menu.add_command(label="采用原标题", command=self._s1b_use_original)
        self.s1b_context_menu.add_command(label="重置为原标题（取消优化）", command=self._s1b_reset_to_original)

        self.s1b_context_menu.add_separator()

        self.s1b_ctx_vision_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="需要视觉识别: -", command=self._s1b_toggle_needs_vision)
        self.s1b_ctx_audio_idx = self.s1b_context_menu.index("end") + 1
        self.s1b_context_menu.add_command(label="音频已识别: -", command=self._s1b_toggle_audio_recognized)

        self.s1b_context_menu.add_separator()

        self.s1b_context_menu.add_command(label="删除选中行", command=self._s1b_delete)

        self.s1b_tree.bind("<Button-3>", self._s1b_show_context_menu)
        self.s1b_tree.bind("<Double-1>", self._s1b_edit)

        # 修改行高亮样式
        self.s1b_tree.tag_configure("modified", background="#e6f3ff")

    # ==================== 浏览 ====================

    # ==================== 搜索 ====================

    def _on_search_changed(self, event=None):
        """搜索框内容变化时过滤表格"""
        query = self.s1b_search_var.get().lower().strip()
        for item in self.s1b_tree.get_children():
            values = self.s1b_tree.item(item, "values")
            if not query:
                # 显示所有行
                self.s1b_tree.reattach(item, "", "end")
            else:
                # 搜索 original_title 和 final_name
                original = str(values[0]).lower() if values[0] else ""
                refined = str(values[3]).lower() if len(values) > 3 and values[3] else ""
                if query in original or query in refined:
                    self.s1b_tree.reattach(item, "", "end")
                else:
                    self.s1b_tree.detach(item)

    # ==================== 浏览 ====================

    def _browse_csv_s1b(self):
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            self.s1b_csv_var.set(file_path)

    # ==================== 加载 ====================

    def _load_s1b_preview(self):
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

            filter_vision = self.s1b_filter_vision_var.get()

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

    # ==================== AI优化 ====================

    def _run_refine_selected(self):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要优化的行")
            return

        provider = self.s1b_provider_var.get()

        items_to_refine = []
        for item_id in selected:
            values = self.s1b_tree.item(item_id, "values")
            original = values[0]
            needs_vision = values[1]
            stem = Path(original).stem
            items_to_refine.append((item_id, original, needs_vision, stem))

        total = len(items_to_refine)
        print(f"[开始] 正在优化 {total} 条记录（并发批次）...")

        self._set_refine_buttons_state("disabled")
        self.s1b_progress_var.set(0)
        self.s1b_progress_label.config(text="0%")

        def run_refine():
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
                        self.s1b_tree.item(item_id, values=(original, needs_vision, audio_recognized, refined))
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

        threading.Thread(target=run_refine, daemon=True).start()

    def _run_refine_all(self):
        all_items = self.s1b_tree.get_children()
        if not all_items:
            messagebox.showwarning("警告", "表格为空，请先加载CSV")
            return

        total = len(all_items)
        if not messagebox.askyesno("确认", f"确定要优化所有 {total} 条记录吗？"):
            return

        provider = self.s1b_provider_var.get()

        items_to_refine = []
        for item_id in all_items:
            values = self.s1b_tree.item(item_id, "values")
            original = values[0]
            needs_vision = values[1]
            stem = Path(original).stem
            items_to_refine.append((item_id, original, needs_vision, stem))

        print(f"[开始] 正在优化 {total} 条记录（并发批次）...")

        self._set_refine_buttons_state("disabled")
        self.s1b_progress_var.set(0)
        self.s1b_progress_label.config(text="0%")

        def run_refine():
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
                        self.s1b_tree.item(item_id, values=(original, needs_vision, audio_recognized, refined))
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

        threading.Thread(target=run_refine, daemon=True).start()

    def _set_refine_buttons_state(self, state):
        for btn in self._refine_buttons:
            btn.configure(state=state)

    # ==================== 行操作 ====================

    def _mark_modified(self, item_id):
        self.s1b_modified.add(item_id)
        self.s1b_tree.item(item_id, tags=("modified",))

    def _s1b_show_context_menu(self, event):
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

    def _s1b_edit(self, event=None):
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
            self.s1b_tree.item(item, values=(values[0], values[1], values[2], new_value))
            if item in self.s1b_results:
                self.s1b_results[item]["final_name"] = new_value
            self._mark_modified(item)
            dialog.destroy()

        ttk.Button(dialog, text="确认", command=confirm).grid(row=2, column=0, columnspan=2, pady=8)

    def _s1b_use_original(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.s1b_tree.item(item, "values")
        original = values[0]
        needs_vision = values[1]
        audio_recognized = values[2]

        self.s1b_tree.item(item, values=(original, needs_vision, audio_recognized, original))
        if item in self.s1b_results:
            self.s1b_results[item]["final_name"] = original
        self._mark_modified(item)

    def _s1b_reset_to_original(self):
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
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())

    # ==================== 状态切换 ====================

    def _s1b_toggle_needs_vision(self):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[1].upper()
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            self.s1b_tree.item(item, values=(values[0], new_value, values[2], values[3]))
            if item in self.s1b_results:
                self.s1b_results[item]["needs_vision"] = new_value.lower()
            count += 1

        self._s1b_write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的needs_vision值（已写入CSV）")

    def _s1b_toggle_audio_recognized(self):
        selected = self.s1b_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的行")
            return

        count = 0
        for item in selected:
            values = self.s1b_tree.item(item, "values")
            current = values[2].upper()
            new_value = "FALSE" if current == "TRUE" else "TRUE"
            self.s1b_tree.item(item, values=(values[0], values[1], new_value, values[3]))
            if item in self.s1b_results:
                self.s1b_results[item]["audio_recognized"] = new_value.lower()
            count += 1

        self._s1b_write_toggles_to_csv()
        print(f"[完成] 已切换 {count} 条记录的audio_recognized值（已写入CSV）")

    def _s1b_batch_needs_vision(self, mode: str):
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

            self.s1b_tree.item(item, values=(values[0], new_value, values[2], values[3]))
            if item in self.s1b_results:
                self.s1b_results[item]["needs_vision"] = new_value.lower()
            count += 1

        self._s1b_write_toggles_to_csv()
        label = {"TRUE": "需要视觉", "FALSE": "不需要视觉", "INVERT": "反选"}[mode]
        print(f"[完成] 已将 {count} 条记录设为 {label}（已写入CSV）")

    def _s1b_select_all(self):
        for item in self.s1b_tree.get_children():
            self.s1b_tree.selection_add(item)

    def _s1b_deselect_all(self):
        self.s1b_tree.selection_remove(*self.s1b_tree.get_children())

    # ==================== CSV写入 ====================

    def _s1b_write_toggles_to_csv(self):
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

    def _s1b_fill_original_selected(self):
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
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _s1b_fill_original_all(self):
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
            self.s1b_modified.discard(item)
            self.s1b_tree.item(item, tags=())
            count += 1

        print(f"[完成] 已将 {count} 条记录的final_name设为原标题")

    def _s1b_delete(self):
        selected = self.s1b_tree.selection()
        if not selected:
            return

        item = selected[0]
        self.s1b_tree.delete(item)
        if item in self.s1b_results:
            del self.s1b_results[item]
        self.s1b_modified.discard(item)

    # ==================== 确认写入 ====================

    def _confirm_s1b_results(self):
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
            for item_id in self.s1b_tree.get_children():
                self.s1b_tree.item(item_id, tags=())

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
