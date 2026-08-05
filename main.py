import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
from PIL import Image, ImageTk
import cv2
import numpy as np

def _app_base():
    """Return the directory containing the EXE."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


from recognizer import ImageRecognizer


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("图像识别工具")
        self.geometry("900x700")
        self.minsize(640, 480)
        self.recognizer = ImageRecognizer(debug=True)
        self.current_image_bgr = None
        self.current_results = None
        self.display_photo = None
        self._build_ui()
        self._setup_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=8, pady=6)
        ttk.Button(toolbar, text="打开图片 (Ctrl+O)", command=self._browse).pack(side="left", padx=2)
        ttk.Button(toolbar, text="粘贴 (Ctrl+V)", command=self._paste).pack(side="left", padx=2)
        ttk.Button(toolbar, text="清空", command=self._clear).pack(side="left", padx=2)
        self.status_var = tk.StringVar(value="打开图片或按 Ctrl+V 粘贴")
        ttk.Label(toolbar, textvariable=self.status_var, foreground="gray").pack(side="right", padx=8)
        self.hint_frame = ttk.Frame(self)
        self.hint_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.hint_label = ttk.Label(
            self.hint_frame, text="拖入图片到此窗口\n\n或打开图片或按 Ctrl+V 粘贴",
            font=("Microsoft YaHei", 14), foreground="#aaa", anchor="center", justify="center", background="#f5f5f5",
        )
        self.hint_label.pack(fill="both", expand=True)
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack_forget()
        paned = ttk.PanedWindow(self.main_frame, orient="vertical")
        paned.pack(fill="both", expand=True)
        self.image_canvas = tk.Canvas(paned, bg="#eee", highlightthickness=0)
        paned.add(self.image_canvas, weight=3)
        results_frame = ttk.LabelFrame(paned, text="识别结果")
        paned.add(results_frame, weight=1)
        self.results_text = tk.Text(results_frame, height=6, wrap="word", font=("Consolas", 10))
        sb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=sb.set)
        self.results_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y", pady=4)

    def _setup_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._browse())
        self.bind("<Control-v>", lambda e: self._paste())

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
            pil_img = ImageGrab.grabclipboard()
            if pil_img is None:
                messagebox.showinfo("粘贴", "剪贴板中没有图片")
                return
            img_array = np.array(pil_img)
            if len(img_array.shape) == 3:
                image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            self._process_image_array(image_bgr, "clipboard")
        except Exception as e:
            messagebox.showerror("粘贴失败", str(e))

    def _clear(self):
        self.current_image_bgr = None
        self.current_results = None
        self.display_photo = None
        self.image_canvas.delete("all")
        self.results_text.delete("1.0", "end")
        self.main_frame.pack_forget()
        self.hint_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.status_var.set("打开图片或按 Ctrl+V 粘贴")

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
        self.current_image_bgr = image_bgr
        self._display_image(image_bgr)
        self.status_var.set(f"[{label}] 正在识别...")
        self.current_results = None
        threading.Thread(target=self._run_recognition, args=(image_bgr.copy(), label), daemon=True).start()

    def _run_recognition(self, image, label="image"):
        try:
            results = self.recognizer.recognize_all(image, label)
            self.after(0, self._show_recognition_results, results)
        except Exception as e:
            import traceback
            log_dir = os.path.join(_app_base(), "debug_preprocess")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "error_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
                f.write(f"\nError: {e}")
            self.after(0, lambda: messagebox.showerror("识别失败", f"错误已保存到:\n{log_path}"))
            self.after(0, lambda: self.status_var.set("识别失败"))

    def _show_recognition_results(self, results):
        self.current_results = results
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", self._format_results(results))
        h, w = results["image_shape"][:2]
        qc = len(results["qr_codes"])
        tc = len(results["texts"])
        self.status_var.set(f"图片: {w}x{h}  |  二维码: {qc}个  |  文字块: {tc}个")

    @staticmethod
    def _format_results(results):
        lines = []
        if results["qr_codes"]:
            lines.append("=== QR Code ===")
            for qr in results["qr_codes"]:
                lines.append(qr["text"])
            lines.append("")
        if results["texts"]:
            if results["qr_codes"]:
                lines.append("=== OCR ===")
            for txt in results["texts"]:
                lines.append(txt["text"])
        if not results["qr_codes"] and not results["texts"]:
            lines.append("未识别到内容")
        return "\n".join(lines)

    def _display_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        self.update_idletasks()
        cw = self.image_canvas.winfo_width() or 860
        ch = self.image_canvas.winfo_height() or 500
        if cw < 50: cw = 860
        if ch < 50: ch = 500
        pil_img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(pil_img)
        self.hint_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=self.display_photo, anchor="center")


if __name__ == "__main__":
    app = App()
    app.mainloop()
