import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np

def _app_base():
    """Return the directory containing the EXE."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _asset_path(name):
    return os.path.join(getattr(sys, "_MEIPASS", _app_base()), name)


from recognizer import ImageRecognizer


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
APP_BG = "#F7F7F6"
SURFACE = "#FFFFFF"
SIDEBAR = "#242526"
INK = "#292B2B"
MUTED = "#757A7A"
BORDER = "#E4E5E3"
ACCENT = "#303232"


def _first_image_path(paths):
    for path in paths:
        candidate = os.fspath(path)
        if os.path.splitext(candidate)[1].lower() in IMAGE_SUFFIXES and os.path.isfile(candidate):
            return candidate
    raise ValueError("没有找到可读取的图片文件")


def clipboard_to_input(content):
    """Convert a clipboard bitmap or Explorer file list to a BGR image."""
    if isinstance(content, Image.Image):
        rgb = np.asarray(content.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "clipboard"
    if isinstance(content, (list, tuple)):
        path = _first_image_path(content)
        return ImageRecognizer.load_image(path), os.path.basename(path)
    raise ValueError("剪贴板中没有图片或图片文件")


def dropped_image_path(data, splitlist):
    """Parse Tk's file list so paths containing spaces remain intact."""
    return _first_image_path(splitlist(data))


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("DPMOCR")
        self.geometry("920x700")
        self.minsize(650, 500)
        self.configure(bg=APP_BG)
        self.icon_path = _asset_path("图标文件.ico")
        window_icon_path = _asset_path("DPMOCR.ico")
        if os.path.isfile(window_icon_path):
            self.iconbitmap(window_icon_path)
        self.recognizer = ImageRecognizer(debug=True)
        self.current_image_bgr = None
        self.current_results = None
        self.display_photo = None
        self._busy = False
        self.history = []
        self._next_history_id = 1
        self._selected_history_id = None
        self._configure_style()
        self._build_ui()
        self._setup_shortcuts()
        self._setup_drop_targets()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Title.TLabel", background=SURFACE, foreground=INK,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("EmptyTitle.TLabel", background=SURFACE, foreground=INK,
                        font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("EmptyHelp.TLabel", background=SURFACE, foreground=MUTED,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Section.TLabel", background=SURFACE, foreground=INK,
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Status.TLabel", background=SURFACE, foreground=MUTED,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Primary.TButton", background=ACCENT, foreground=SURFACE,
                        borderwidth=0, padding=(14, 7),
                        font=("Microsoft YaHei UI", 9))
        style.map("Primary.TButton", background=[("active", "#444747")])
        style.configure("Quiet.TButton", background=SURFACE, foreground=INK,
                        borderwidth=1, bordercolor=BORDER, padding=(12, 7),
                        font=("Microsoft YaHei UI", 9))
        style.map("Quiet.TButton", background=[("active", "#F2F3F2")])

    def _build_ui(self):
        shell = ttk.Frame(self, style="App.TFrame")
        shell.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(shell, bg=SIDEBAR, width=208)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        brand = tk.Frame(self.sidebar, bg=SIDEBAR)
        brand.pack(fill="x", padx=19, pady=(22, 0))
        self.brand_icon = None
        if os.path.isfile(self.icon_path):
            with Image.open(self.icon_path) as icon_image:
                icon_image = icon_image.convert("RGBA").resize(
                    (28, 28), Image.Resampling.LANCZOS)
            self.brand_icon = ImageTk.PhotoImage(icon_image)
            tk.Label(brand, image=self.brand_icon, bg=SIDEBAR,
                     borderwidth=0).pack(side="left")
        tk.Label(brand, text="DPMOCR", bg=SIDEBAR, fg="#F1F1EF",
                 font=("Segoe UI Semibold", 12)).pack(
                     side="left", padx=(9 if self.brand_icon else 0, 0))
        tk.Frame(self.sidebar, bg="#3B3D3D", height=1).pack(
            fill="x", padx=19, pady=(22, 18))
        tk.Label(self.sidebar, text="本次记录", bg=SIDEBAR, fg="#B8BCBA",
                 anchor="w", font=("Microsoft YaHei UI", 9)).pack(
                     fill="x", padx=20, pady=(0, 10))
        self.history_empty = tk.Label(
            self.sidebar, text="处理图片后会显示在这里", bg=SIDEBAR,
            fg="#8F9492", anchor="w", font=("Microsoft YaHei UI", 9))
        self.history_empty.pack(fill="x", padx=20)
        self.history_list = tk.Listbox(
            self.sidebar, bg=SIDEBAR, fg="#D9DCDA", selectbackground="#414444",
            selectforeground="#FFFFFF", activestyle="none", highlightthickness=0,
            borderwidth=0, relief="flat", exportselection=False,
            font=("Microsoft YaHei UI", 9), cursor="hand2")
        self.history_list.bind("<<ListboxSelect>>", self._on_history_select)
        self.history_list.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        workspace = ttk.Frame(shell, style="Surface.TFrame")
        workspace.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(workspace, style="Surface.TFrame")
        header.pack(fill="x")
        toolbar = ttk.Frame(header, style="Surface.TFrame")
        toolbar.pack(side="right", padx=18, pady=15)
        ttk.Button(toolbar, text="打开", style="Primary.TButton",
                   command=self._browse).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="粘贴", style="Quiet.TButton",
                   command=self._paste).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="清空", style="Quiet.TButton",
                   command=self._clear).pack(side="left")
        tk.Frame(workspace, height=1, bg=BORDER).pack(fill="x")

        self.status_var = tk.StringVar(value="等待图片")
        status_bar = ttk.Frame(workspace, style="Surface.TFrame")
        status_bar.pack(side="bottom", fill="x")
        tk.Frame(status_bar, height=1, bg=BORDER).pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status_var,
                  style="Status.TLabel").pack(anchor="w", padx=24, pady=9)

        content = ttk.Frame(workspace, style="Surface.TFrame")
        content.pack(fill="both", expand=True)
        self.hint_frame = ttk.Frame(content, style="Surface.TFrame")
        self.hint_frame.pack(fill="both", expand=True)
        self.hint_card = ttk.Frame(self.hint_frame, style="Surface.TFrame")
        self.hint_card.pack(fill="both", expand=True)
        hint_content = ttk.Frame(self.hint_card, style="Surface.TFrame")
        hint_content.place(relx=0.5, rely=0.5, anchor="center")
        self.hint_label = ttk.Label(hint_content, text="拖入图片",
                                    style="EmptyTitle.TLabel", anchor="center")
        self.hint_label.pack()
        self.hint_help = ttk.Label(
            hint_content, text="或打开文件、按 Ctrl+V 粘贴",
            style="EmptyHelp.TLabel", anchor="center",
        )
        self.hint_help.pack(pady=(8, 0))

        self.main_frame = ttk.Frame(content, style="Surface.TFrame")
        self.main_frame.pack_forget()
        paned = ttk.PanedWindow(self.main_frame, orient="vertical")
        paned.pack(fill="both", expand=True, padx=18, pady=16)
        image_frame = ttk.Frame(paned, style="Surface.TFrame")
        self.image_canvas = tk.Canvas(image_frame, bg="#F1F2F0", highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True)
        paned.add(image_frame, weight=3)
        results_frame = ttk.Frame(paned, style="Surface.TFrame")
        paned.add(results_frame, weight=1)
        ttk.Label(results_frame, text="识别结果", style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(12, 4))
        result_body = ttk.Frame(results_frame, style="Surface.TFrame")
        result_body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.results_text = tk.Text(result_body, height=6, wrap="word",
                                    font=("Consolas", 11), bg=SURFACE, fg=INK,
                                    relief="flat", borderwidth=0, padx=2, pady=4,
                                    insertbackground=INK, selectbackground="#DADDDC")
        sb = ttk.Scrollbar(result_body, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=sb.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _setup_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._browse())
        self.bind("<Control-v>", lambda e: self._paste())

    def _setup_drop_targets(self):
        for widget in (self, self.sidebar, self.history_list, self.hint_card,
                       self.hint_label, self.hint_help,
                       self.image_canvas, self.results_text):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._drop)

    def _drop(self, event):
        try:
            path = dropped_image_path(event.data, self.tk.splitlist)
        except ValueError as e:
            messagebox.showerror("拖入失败", str(e))
            return
        self._process_image(path)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("所有文件", "*.*")],
        )
        if path:
            self._process_image(path)

    def _paste(self):
        try:
            from PIL import ImageGrab
            image_bgr, label = clipboard_to_input(ImageGrab.grabclipboard())
            self._process_image_array(image_bgr, label)
        except ValueError as e:
            messagebox.showinfo("粘贴", str(e))
        except Exception as e:
            messagebox.showerror("粘贴失败", str(e))

    def _clear(self):
        self._selected_history_id = None
        self.history_list.selection_clear(0, "end")
        self.current_image_bgr = None
        self.current_results = None
        self.display_photo = None
        self.image_canvas.delete("all")
        self.results_text.delete("1.0", "end")
        self.main_frame.pack_forget()
        self.hint_frame.pack(fill="both", expand=True)
        self.status_var.set("等待图片")

    def _remember_image(self, image_bgr, label):
        entry = {"id": self._next_history_id, "image": image_bgr,
                 "label": label, "results": None, "error": False}
        self._next_history_id += 1
        self.history.insert(0, entry)
        del self.history[30:]
        self._refresh_history()
        self._activate_history(entry)
        return entry

    def _refresh_history(self):
        self.history_list.delete(0, "end")
        for entry in self.history:
            name = entry["label"]
            if len(name) > 20:
                name = f"{name[:10]}…{name[-9:]}"
            self.history_list.insert("end", f"  {name}")
        if self.history:
            self.history_empty.pack_forget()
        else:
            self.history_empty.pack(fill="x", padx=20)
        for index, entry in enumerate(self.history):
            if entry["id"] == self._selected_history_id:
                self.history_list.selection_set(index)
                break

    def _on_history_select(self, _event):
        indices = self.history_list.curselection()
        if indices:
            self._activate_history(self.history[indices[0]])

    def _activate_history(self, entry):
        self._selected_history_id = entry["id"]
        self.current_image_bgr = entry["image"]
        self.current_results = entry["results"]
        self._display_image(entry["image"])
        self.results_text.delete("1.0", "end")
        if entry["results"] is not None:
            self.results_text.insert("end", self._format_results(entry["results"]))
            h, w = entry["results"]["image_shape"][:2]
            qc = len(entry["results"]["qr_codes"])
            tc = len(entry["results"]["texts"])
            self.status_var.set(f"图片: {w}x{h}  |  二维码: {qc}个  |  文字块: {tc}个")
        elif entry["error"]:
            self.results_text.insert("end", "识别失败")
            self.status_var.set(f"{entry['label']} · 识别失败")
        else:
            self.results_text.insert("end", "正在识别...")
            self.status_var.set(f"{entry['label']} · 正在识别...")
        self.history_list.selection_clear(0, "end")
        for index, item in enumerate(self.history):
            if item["id"] == entry["id"]:
                self.history_list.selection_set(index)
                break

    def _process_image(self, path):
        self.status_var.set("正在加载...")
        self.update_idletasks()
        try:
            image_bgr = ImageRecognizer.load_image(path)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))
            self.status_var.set("加载失败")
            return
        self._process_image_array(image_bgr, os.path.basename(path))

    def _process_image_array(self, image_bgr, label="image"):
        if self._busy:
            self.status_var.set("正在识别，请稍候")
            return
        self._busy = True
        entry = self._remember_image(image_bgr, label)
        threading.Thread(target=self._run_recognition,
                         args=(image_bgr.copy(), label, entry["id"]), daemon=True).start()

    def _run_recognition(self, image, label, history_id):
        try:
            results = self.recognizer.recognize_all(image, label)
            self.after(0, self._show_recognition_results, results, history_id)
        except Exception as e:
            import traceback
            log_dir = os.path.join(_app_base(), "debug_preprocess")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "error_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
                f.write(f"\nError: {e}")
            self.after(0, self._show_recognition_error, history_id, log_path)
        finally:
            self.after(0, self._finish_busy)

    def _finish_busy(self):
        self._busy = False

    def _show_recognition_results(self, results, history_id):
        entry = next((item for item in self.history if item["id"] == history_id), None)
        if entry is None:
            return
        entry["results"] = results
        self._refresh_history()
        if self._selected_history_id == history_id:
            self._activate_history(entry)

    def _show_recognition_error(self, history_id, log_path):
        entry = next((item for item in self.history if item["id"] == history_id), None)
        if entry is not None:
            entry["error"] = True
            self._refresh_history()
            if self._selected_history_id == history_id:
                self._activate_history(entry)
        messagebox.showerror("识别失败", f"错误已保存到:\n{log_path}")

    @staticmethod
    def _format_results(results):
        lines = []
        if results["qr_codes"]:
            lines.append("二维码内容")
            for qr in results["qr_codes"]:
                lines.append(qr["text"])
            lines.append("")
        if results["texts"]:
            if results["qr_codes"]:
                lines.append("文字内容")
            for txt in results["texts"]:
                lines.append(txt["text"])
        if not results["qr_codes"] and not results["texts"]:
            lines.append("未识别到内容")
        return "\n".join(lines)

    def _display_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        self.hint_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
        self.update_idletasks()
        cw = self.image_canvas.winfo_width() or 860
        ch = self.image_canvas.winfo_height() or 500
        if cw < 50: cw = 860
        if ch < 50: ch = 500
        pil_img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=self.display_photo, anchor="center")


if __name__ == "__main__":
    app = App()
    app.mainloop()
