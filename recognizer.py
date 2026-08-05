import os
import sys
from pathlib import Path


def _bundle_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


_bundle = _bundle_root()
for _dll_dir in (_bundle, _bundle / "paddle" / "libs", _bundle / "onnxruntime" / "capi"):
    _dll_path = str(_dll_dir)
    if os.path.isdir(_dll_path):
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_dll_path)
        except Exception:
            pass


import cv2
import numpy as np
import math
import paddle

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"


def _resource_dir():
    return _bundle_root()


CHAR_MODEL_PATH = str(_resource_dir() / "model" / "char_cnn" / "best_model.pdparams")
CHAR_CLASSES_PATH = str(_resource_dir() / "model" / "char_cnn" / "classes.json")
PADDLECLS_MODEL_PATH = str(_resource_dir() / "model" / "paddlex_v6" / "inference")
PADDLECLS_CLASSES_PATH = str(_resource_dir() / "model" / "paddlex_v6" / "classes.json")

# 切字/ROI 调参入口
SPLIT_CLAHE_CLIP = 2.5
SPLIT_ADAPTIVE_BLOCK = 41
SPLIT_ADAPTIVE_C = 15
SPLIT_GAP = 5
SPLIT_MAX_W_FACTOR = 1.3
SPLIT_MIN_WIDE = 38
OUTPUT_UPPERCASE = False
CLS_INPUT_SIZE = 96
PPOCRV6_REC_MODEL_PATH = str(_resource_dir() / "model" / "ppocrv6_rec_v3")
PPOCRV6_REC_MIN_HEIGHT = 24
WECHAT_QR_MODEL_DIR = str(_resource_dir() / "model" / "wechat_qr")
PPOCRV4_DET_MODEL_PATH = str(_resource_dir() / "model" / "ppocrv4_det")
PPOCRV4_DET_SCORE = 0.1

SMALL_IMG_MIN_H = 250
SMALL_IMG_MIN_W = 500
SMALL_CANVAS_H = 600
SMALL_CANVAS_W = 1000
SMALL_LINE_ASPECT = 4.5


def _app_base():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


class CharCNN(paddle.nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(3,32,3,padding=1), paddle.nn.BatchNorm2D(32), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2),
            paddle.nn.Conv2D(32,64,3,padding=1), paddle.nn.BatchNorm2D(64), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2),
            paddle.nn.Conv2D(64,128,3,padding=1), paddle.nn.BatchNorm2D(128), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2),
            paddle.nn.Conv2D(128,256,3,padding=1), paddle.nn.BatchNorm2D(256), paddle.nn.ReLU(),
        )
        self.pool = paddle.nn.AdaptiveAvgPool2D(1)
        self.fc = paddle.nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.features(x); x = self.pool(x).flatten(1)
        return self.fc(x)


class ImageRecognizer:
    def __init__(self, debug=False):
        self._d = debug
        self._rt = _app_base() / "debug_preprocess"
        self._rn = None
        # Load char classifier
        self._cm = None; self._cc = None
        self._load_char_model()
        self._rec6 = None
        self._wq = None
        self._det4 = None

    def _load_wechat_qr(self):
        if getattr(self, "_wq", None) is not None:
            return
        self._wq = None
        try:
            import cv2
            if not hasattr(cv2, "wechat_qrcode"):
                return
            self._wq = cv2.wechat_qrcode_WeChatQRCode(
                os.path.join(WECHAT_QR_MODEL_DIR, "detect.prototxt"),
                os.path.join(WECHAT_QR_MODEL_DIR, "detect.caffemodel"),
                os.path.join(WECHAT_QR_MODEL_DIR, "sr.prototxt"),
                os.path.join(WECHAT_QR_MODEL_DIR, "sr.caffemodel"),
            )
            print("[WeChatQR] loaded")
        except Exception as e:
            print(f"[WeChatQR] Not available: {e}")

    def _load_det4(self):
        if getattr(self, "_det4", None) is not None:
            return
        self._det4 = None
        try:
            import paddlex
            self._det4 = paddlex.create_model(
                "PP-OCRv4_mobile_det", model_dir=PPOCRV4_DET_MODEL_PATH, device="cpu"
            )
            print("[PP-OCRv4-det] loaded")
        except Exception as e:
            print(f"[PP-OCRv4-det] Not available: {e}")

    def _det_lines4(self, rgb):
        """微调后的 PP-OCRv4 检测文字行，返回 [y0,y1,x0,x1,score,text] 列表。"""
        if getattr(self, "_det4", None) is None:
            self._load_det4()
        if self._det4 is None:
            return []
        rows = []
        try:
            res = list(self._det4.predict(input=rgb, batch_size=1))[0]
            polys = res.get("dt_polys")
            scores = res.get("dt_scores")
            if polys is None:
                return rows
            for poly, sc in zip(polys, scores):
                if float(sc) < PPOCRV4_DET_SCORE:
                    continue
                pts = np.asarray(poly, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                rows.append([
                    int(pts[:, 1].min()), int(pts[:, 1].max()),
                    int(pts[:, 0].min()), int(pts[:, 0].max()),
                    float(sc), "",
                ])
        except Exception:
            pass
        return rows

    @staticmethod
    def _merge_det_lines(rows):
        """把检测出的词级碎片按 y 中心距离合并成整行。"""
        if len(rows) <= 1:
            return rows
        hs = [r[1] - r[0] for r in rows]
        med_h = float(max(hs))
        rows = sorted(rows, key=lambda r: (r[0] + r[1]) / 2)
        lines = []
        for r in rows:
            yc = (r[0] + r[1]) / 2
            placed = False
            for ln in lines:
                lyc = (ln[0] + ln[1]) / 2
                if abs(yc - lyc) <= 0.45 * med_h:
                    ln[0] = min(ln[0], r[0]); ln[1] = max(ln[1], r[1])
                    ln[2] = min(ln[2], r[2]); ln[3] = max(ln[3], r[3])
                    ln[4] = max(ln[4], r[4])
                    placed = True
                    break
            if not placed:
                lines.append(list(r))
        return lines

    @staticmethod
    def _project_line_bands(gray):
        """行投影找文字行带（整行宽度），排除二维码类方块。"""
        h, w = gray.shape
        enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        bw = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        proj = bw.sum(axis=1).astype(np.float32)
        th = max(proj.max() * 0.02, 2.0)
        runs = []
        on = False
        s = 0
        for y, v in enumerate(proj):
            if v > th:
                if not on:
                    s = y; on = True
            else:
                if on:
                    if y - s >= 6:
                        runs.append([s, y])
                    on = False
        if on and len(proj) - s >= 6:
            runs.append([s, len(proj)])
        if not runs:
            return []
        hs = [b - a for a, b in runs]
        med_h = float(sorted(hs)[len(hs) // 2])
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= max(3, int(med_h * 0.45)):
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out = []
        for a, b in merged:
            band_h = b - a
            if band_h < 6 or band_h > h * 0.35:
                continue
            if w / band_h < 1.5:
                continue
            out.append([a, b, 0, w - 1, 1.0, ""])
        return out

    def _load_rec6(self):
        if getattr(self, "_rec6", None) is not None:
            return
        self._rec6 = None
        try:
            import paddlex
            self._rec6 = paddlex.create_model(
                "PP-OCRv6_medium_rec", model_dir=PPOCRV6_REC_MODEL_PATH, device="cpu"
            )
            print("[PP-OCRv6] loaded")
        except Exception as e:
            print(f"[PP-OCRv6] Not available: {e}")

    def _recognize_line_rec6(self, line):
        if getattr(self, "_rec6", None) is None:
            self._load_rec6()
        if self._rec6 is None or line is None or line.size == 0:
            return "", 0.0
        try:
            gray = line
            if gray.shape[0] < PPOCRV6_REC_MIN_HEIGHT:
                scale = max(1, int(np.ceil(PPOCRV6_REC_MIN_HEIGHT / gray.shape[0])))
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            res = list(self._rec6.predict(input=rgb, batch_size=1))[0]
            return str(res.get("rec_text", "")), float(res.get("rec_score", 0.0))
        except Exception:
            return "", 0.0

    def _load_char_model(self):
        try:
            import paddlex, json
            self._pmx = paddlex.create_model(
                "ResNet18", model_dir=PADDLECLS_MODEL_PATH, device="cpu"
            )
            with open(PADDLECLS_CLASSES_PATH, "r", encoding="utf-8") as f:
                self._cc = json.load(f)
            print(f"[PaddleX-ResNet18] loaded, {len(self._cc)} classes")
            return
        except Exception as e:
            print(f"[PaddleX-ResNet18] Not available: {e}")
            self._pmx = None
        try:
            import paddle, json
            with open(CHAR_CLASSES_PATH, "r", encoding="utf-8") as f:
                self._cc = json.load(f)
            self._cm = CharCNN(len(self._cc))
            self._cm.set_state_dict(paddle.load(CHAR_MODEL_PATH))
            self._cm.eval()
            print(f"[CharCNN] loaded, {len(self._cc)} classes")
        except Exception as e:
            print(f"[CharCNN] Not available: {e}")
            self._cm = None

    def _recognize_char(self, roi):
        """单字分类：48x48 RGB /255。"""
        if getattr(self, "_pmx", None) is not None:
            try:
                inp = self._char_to_cls_input(roi)
                if inp is not None:
                    rgb = cv2.cvtColor(inp, cv2.COLOR_GRAY2RGB)
                    res = list(self._pmx.predict(input=rgb, batch_size=1))[0]
                    ids = [int(i) for i in res["class_ids"][0]]
                    ink = float(np.count_nonzero(inp < 200)) / inp.size
                    ch = None
                    for i, idx in enumerate(ids):
                        cand = self._cc[idx]
                        if cand == "." and (i > 0 or ink > 0.02):
                            continue
                        ch = cand
                        break
                    if ch is None:
                        ch = self._cc[ids[0]]
                    return ch.upper() if OUTPUT_UPPERCASE else ch
            except Exception:
                pass
        if self._cm is None or self._cc is None:
            return "?"
        try:
            import paddle
            img = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            arr = rgb.astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[None]
            with paddle.no_grad():
                logits = self._cm(paddle.to_tensor(arr))
            return self._cc[int(paddle.argmax(logits, axis=1).numpy()[0])]
        except:
            return "?"

    def _char_to_cls_input(self, roi, size=None, pad=4):
        if size is None:
            size = CLS_INPUT_SIZE
        """把单字 ROI 归一化为白底黑字的 64x64 输入，和训练数据一致。"""
        if roi is None or roi.size == 0:
            return None
        norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_fg = self._pick_foreground(bw)
        coords = cv2.findNonZero(white_fg)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(white_fg.shape[1], x + w + pad), min(white_fg.shape[0], y + h + pad)
        crop = white_fg[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        scale = min((size - 4) / crop.shape[1], (size - 4) / crop.shape[0])
        nw, nh = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size), dtype=np.uint8)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return 255 - canvas

    def _split_chars(self, line, gap=None):
        """行内字符分割：点阵连通域按质心间距聚类。返回 [(x0,x1,y0,y1)]。"""
        if gap is None:
            gap = SPLIT_GAP
        enh = cv2.createCLAHE(clipLimit=SPLIT_CLAHE_CLIP, tileGridSize=(8, 8)).apply(line)
        bw = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, SPLIT_ADAPTIVE_BLOCK, SPLIT_ADAPTIVE_C)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        proj_y = bw.sum(axis=1).astype(np.float32)
        th_y = max(proj_y.max() * 0.04, 1.0)
        ys = np.where(proj_y > th_y)[0]
        band_y0 = 0
        band_h = line.shape[0]
        if ys.size:
            band_y0 = int(ys.min())
            band_h = int(ys.max()) - int(ys.min()) + 1
            bw = bw[int(ys.min()):int(ys.max()) + 1, :]
        n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        comps = []
        for i in range(1, n):
            x = stats[i, cv2.CC_STAT_LEFT]; w = stats[i, cv2.CC_STAT_WIDTH]
            y = stats[i, cv2.CC_STAT_TOP]; h = stats[i, cv2.CC_STAT_HEIGHT]
            if stats[i, cv2.CC_STAT_AREA] >= 3 and w >= 2 and h >= 2:
                comps.append([x, x + w, y + band_y0, y + h + band_y0])
        if not comps:
            return []
        comps.sort(key=lambda c: c[0])
        groups = [[comps[0]]]
        for c in comps[1:]:
            if c[0] - groups[-1][-1][1] > gap:
                groups.append([c])
            else:
                groups[-1].append(c)
        widths = []
        for g in groups:
            gw = max(c[1] for c in g) - min(c[0] for c in g)
            if 6 <= gw <= max(20, line.shape[0] * 1.2):
                widths.append(gw)
        exp_w = float(np.median(widths)) if widths else max(8, int(band_h * 0.6))
        exp_w = max(8, int(exp_w))
        out = []
        for g in groups:
            x0 = min(c[0] for c in g); x1 = max(c[1] for c in g)
            y0 = min(c[2] for c in g); y1 = max(c[3] for c in g)
            w = x1 - x0
            if w < 6:
                continue
            if w > max(SPLIT_MAX_W_FACTOR * exp_w, SPLIT_MIN_WIDE):
                pieces = self._split_wide_group(bw, x0, x1, y0 - band_y0, y1 - band_y0, exp_w)
                out.extend((a, b, c + band_y0, d + band_y0) for a, b, c, d in pieces)
            else:
                out.append((x0, x1, y0, y1))
        out.sort(key=lambda b: (b[0], b[1]))
        proj = bw.sum(axis=0).astype(np.float32)
        res = []
        for b in out:
            if res and b[0] < res[-1][1]:
                prev = res[-1]
                z0, z1 = b[0], min(prev[1], b[1])
                seg = proj[z0:z1 + 1]
                ref = proj[max(0, prev[0]):prev[1] + 1].mean()
                if seg.size and z1 > z0 and seg.min() < max(ref, 1):
                    cut = z0 + int(np.argmin(seg))
                    if prev[0] < cut < prev[1]:
                        res[-1] = (prev[0], cut, prev[2], prev[3])
                        b = (cut, b[1], b[2], b[3])
            if b[1] > b[0]:
                res.append(list(b))
        out = [tuple(x) for x in res]
        return out

    def _split_wide_group(self, bw, x0, x1, y0, y1, exp_w):
        """过宽粘连块按列投影谷点细分，避免把多个字符切成一个。"""
        ry0, ry1 = max(0, y0 - 2), min(bw.shape[0], y1 + 3)
        rx0, rx1 = max(0, x0 - 2), min(bw.shape[1], x1 + 3)
        roi = bw[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            return [(x0, x1, y0, y1)]
        n, labels, stats, cents = cv2.connectedComponentsWithStats(roi, 8)
        comps = []
        for i in range(1, n):
            w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
            if stats[i, cv2.CC_STAT_AREA] >= 3 and w >= 2 and h >= 2:
                comps.append([stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_LEFT] + w])
        if not comps:
            return [(x0, x1, y0, y1)]
        comps.sort(key=lambda c: c[0])
        max_w = max(exp_w * 1.25, 30)
        min_w = max(int(exp_w * 0.4), 6)
        pieces = self._cut_by_gaps(comps, max_w, min_w)
        final = []
        for a, b in pieces:
            if b - a <= max_w:
                final.append((a, b))
                continue
            sub = self._project_split(roi[:, a:b], min_w)
            if len(sub) >= 2:
                final.extend((a + s, a + e) for s, e in sub)
                continue
            w2 = b - a
            n2 = max(2, int(round(w2 / (exp_w * 0.9))))
            width2 = w2 / n2
            for k in range(n2):
                aa = int(round(k * width2))
                bb = int(round((k + 1) * width2))
                if bb - aa >= min_w and np.count_nonzero(roi[:, a + aa:a + bb]) > 0:
                    final.append((a + aa, a + bb))
        if len(final) >= 2:
            return [(rx0 + a, rx0 + b, y0, y1) for a, b in final]
        proj_pieces = self._project_split(roi, min_w)
        if len(proj_pieces) >= 2:
            return [(rx0 + a, rx0 + b, y0, y1) for a, b in proj_pieces]
        if roi.shape[1] > max_w * 1.5:
            n = max(2, int(round(roi.shape[1] / (exp_w * 0.9))))
            width = roi.shape[1] / n
            eq_pieces = []
            for k in range(n):
                a = int(round(k * width))
                b = int(round((k + 1) * width))
                if b - a >= min_w and np.count_nonzero(roi[:, a:b]) > 0:
                    eq_pieces.append((rx0 + a, rx0 + b, y0, y1))
            if len(eq_pieces) >= 2:
                return eq_pieces
        return [(x0, x1, y0, y1)]

    @staticmethod
    def _cut_by_gaps(comps, max_w, min_w):
        """在最大组件间隙处反复拆分，直到每块宽度不超过 max_w。"""
        groups = [comps]
        changed = True
        while changed:
            changed = False
            new_groups = []
            for g in groups:
                if len(g) < 2 or (g[-1][1] - g[0][0]) <= max_w:
                    new_groups.append(g)
                    continue
                best_i, best_gap = -1, -1
                for i in range(len(g) - 1):
                    gap = g[i + 1][0] - g[i][1]
                    if gap > best_gap:
                        best_gap, best_i = gap, i
                if best_gap < 1:
                    new_groups.append(g)
                    continue
                left, right = g[:best_i + 1], g[best_i + 1:]
                if (left[-1][1] - left[0][0]) >= min_w and (right[-1][1] - right[0][0]) >= min_w:
                    new_groups.extend([left, right])
                    changed = True
                else:
                    new_groups.append(g)
            groups = new_groups
        return [(g[0][0], g[-1][1]) for g in groups]

    @staticmethod
    def _project_split(roi, min_w):
        """按列投影低谷切分，返回相对 x 区间列表。"""
        proj = roi.sum(axis=0).astype(np.float32)
        if proj.size == 0:
            return []
        th = max(proj.max() * 0.04, 0.5)
        runs = []
        on = False
        s = 0
        for i, v in enumerate(proj):
            if v <= th:
                if not on:
                    s = i
                    on = True
            else:
                if on:
                    if i - s >= 1:
                        runs.append((s, i))
                    on = False
        if on and len(proj) - s >= 1:
            runs.append((s, len(proj)))
        cuts = [0] + [s for s, e in runs] + [roi.shape[1]]
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < min_w or proj[a:b].max() <= 0:
                continue
            pieces.append((a, b))
        return pieces

    def _sv(self, d, n, i):
        if self._d and i is not None and self._rn:
            p=self._rn/d; os.makedirs(p,exist_ok=True); cv2.imwrite(str(p/n),i)

    def _save_result(self, texts):
        if not self._rn: return
        p = self._rn / "result.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

    def _bg(self, n="img"):
        if not self._d: return
        self._rt.mkdir(exist_ok=True)
        c=len([x for x in os.listdir(self._rt) if os.path.isdir(self._rt/x) and x.startswith(n)])
        self._rn=self._rt/f"{n}_{c+1:03d}"; os.makedirs(self._rn,exist_ok=True)

    def recognize_qr(self, img):
        if getattr(self, "_wq", None) is None:
            self._load_wechat_qr()
        results = []
        if self._wq is not None:
            for scale in (1, 2, 3):
                v = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                try:
                    texts, points = self._wq.detectAndDecode(v)
                except Exception:
                    continue
                if not texts:
                    continue
                for i, t in enumerate(texts):
                    bbox = None
                    if points is not None and i < len(points):
                        bbox = [(int(x / scale), int(y / scale)) for x, y in points[i].tolist()]
                    results.append({"text": t, "bbox": bbox})
                return results
        d = cv2.QRCodeDetector()
        for scale in (1, 2):
            v = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            try:
                ok, data, pts, _ = d.detectAndDecodeMulti(v)
            except Exception:
                continue
            if ok and data:
                for i, t in enumerate(data):
                    bbox = None
                    if pts is not None and i < len(pts):
                        bbox = [(int(x / scale), int(y / scale)) for x, y in pts[i].tolist()]
                    results.append({"text": t, "bbox": bbox})
                return results
        # crop refinement: detected quad but no decode yet
        try:
            ret, pts = cv2.QRCodeDetector().detect(img)
            if ret:
                for quad in pts:
                    q = quad.reshape(4, 2).astype(np.float32)
                    tl = q[np.argmin(q.sum(axis=1))]
                    br = q[np.argmax(q.sum(axis=1))]
                    tr = q[np.argmin(q[:, 0] - q[:, 1])]
                    bl = q[np.argmax(q[:, 0] - q[:, 1])]
                    side = int(max(np.linalg.norm(tr - tl), np.linalg.norm(bl - tl))) + 20
                    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
                    warp = cv2.warpPerspective(img, M, (side, side))
                    for scale in (4, 8):
                        big = cv2.resize(warp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        if self._wq is not None:
                            try:
                                texts, _ = self._wq.detectAndDecode(big)
                                if texts:
                                    bbox = [(int(x), int(y)) for x, y in q.tolist()]
                                    return [{"text": t, "bbox": bbox} for t in texts]
                            except Exception:
                                pass
                        try:
                            ok, data, dpts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(big)
                            if ok and data:
                                bbox = [(int(x), int(y)) for x, y in q.tolist()]
                                return [{"text": t, "bbox": bbox} for t in data]
                        except Exception:
                            pass
        except Exception:
            pass
        return []

    def _split(self, b):
        p=np.sum(b>128,axis=0); t=max(np.max(p)*0.15,2)
        cs=[]; s=0; on=False
        for x in range(len(p)):
            if p[x]>t:
                if not on: s=x; on=True
            else:
                if on and x-s>2: cs.append((s,x)); on=False
        if on and len(p)-s>2: cs.append((s,len(p)))
        cs=[c for c in cs if c[1]-c[0]>=4]
        if not cs: return []
        m=[cs[0]]
        for c in cs[1:]:
            if c[0]-m[-1][1]<6: m[-1]=(m[-1][0],c[1])
            else: m.append(c)
        r=[]
        for a,b in m:
            w=b-a
            if w>50:
                n=round(w/35)
                if n>1:
                    for i in range(n): r.append((a+int(i*w/n),a+int((i+1)*w/n)))
                else: r.append((a,b))
            else: r.append((a,b))
        return r

    @staticmethod
    def _pick_foreground(bin_img):
        """判断文字极性：返回白字黑底的二值图。"""
        def count_small(mask):
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            total = mask.size
            cnt = 0
            for i in range(1, n):
                a = stats[i, cv2.CC_STAT_AREA]
                if 6 <= a <= total * 0.2:
                    cnt += 1
            return cnt
        if count_small(bin_img) >= count_small(255 - bin_img):
            return bin_img
        return cv2.bitwise_not(bin_img)

    @staticmethod
    def _long_edge_angle(rect):
        """用外接矩形长边方向算文字倾斜角，不依赖 minAreaRect 的 angle 语义。"""
        box = cv2.boxPoints(rect).astype(np.float32)
        e1 = box[1] - box[0]
        e2 = box[2] - box[1]
        l1 = math.hypot(e1[0], e1[1]); l2 = math.hypot(e2[0], e2[1])
        e = e1 if l1 >= l2 else e2
        ang = math.degrees(math.atan2(e[1], e[0]))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        return ang

    @staticmethod
    def _block_score(area, ratio, ang):
        """文本块评分：面积大、宽高比像文字行、角度接近水平。"""
        s = float(area)
        if 1.8 <= ratio <= 8.0:
            s *= 1.0
        elif ratio > 8.0:
            s *= 0.2   # 长横条/划痕
        else:
            s *= 0.3   # 方形，多半是二维码
        a = abs(ang)
        if a <= 12.0:
            s *= 1.0
        elif a <= 25.0:
            s *= 0.5
        else:
            s *= 0.15  # 45 度对角，多半是二维码
        return s

    def _locate_block(self, fg):
        """点->字符->文本块，返回文本块二值图、旋转矩形、校正角度。"""
        # 点 -> 字符级连通域，过滤噪点、QR 大块
        join = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(join, 8)
        comp = np.zeros_like(join)
        img_area = join.shape[0] * join.shape[1]
        max_char = img_area * 0.03
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
            if area < 15 or w < 3 or h < 3:
                continue
            if area > max_char:
                continue
            comp[labels == i] = 255

        # 字符 -> 候选文本块（小核，让分散区域分开）
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
        text_block = cv2.dilate(comp, dk, iterations=2)
        contours, _ = cv2.findContours(text_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None; best_score = 0.0; best_rect = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            x, y, w2, h2 = cv2.boundingRect(c)
            if h2 <= 0:
                continue
            ratio = w2 / h2
            rect = cv2.minAreaRect(c)
            ang = self._long_edge_angle(rect)
            s = self._block_score(area, ratio, ang)
            if s > best_score:
                best_score = s; best = c; best_rect = rect
        if best is None:
            return comp, None, 0.0
        corr = self._long_edge_angle(best_rect)
        return comp, best_rect, corr

    @staticmethod
    def _text_ok(t):
        """行识别文本是否像真文字（过滤二维码/噪声乱码）。"""
        if t is None: return False
        chars = [c for c in t if c.isprintable()]
        if not chars: return False
        valid = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,-/<>.:;=@[]^_`{}~+*'\"!$%&()")
        return sum(1 for c in chars if c in valid) / len(chars) >= 0.5

    @staticmethod
    def _filter_merge_rows(rows, H):
        """过滤二维码/噪声框，合并重叠行框。"""
        keep = []
        for y0, y1, x0, x1, c, t in rows:
            h = y1 - y0; w = x1 - x0
            if h < 10 or h > H * 0.35: continue
            if h > H * 0.08 and w / h < 1.5: continue  # 大方形=二维码
            keep.append([y0, y1, x0, x1, c, t])
        keep.sort(key=lambda r: (r[0] + r[1]) / 2)
        out = []
        for r in keep:
            merged = False
            for m in out:
                y0, y1, x0, x1, c, t = r
                my0, my1, mx0, mx1, mc, mt = m
                yo = min(my1, y1) - max(my0, y0)
                xo = min(mx1, x1) - max(mx0, x0)
                if yo > 0.5 * min(my1 - my0, y1 - y0) and xo > 0.15 * min(mx1 - mx0, x1 - x0):
                    m[0] = min(my0, y0); m[1] = max(my1, y1)
                    m[2] = min(mx0, x0); m[3] = max(mx1, x1)
                    if c > mc: m[4] = c; m[5] = mt
                    merged = True
                    break
            if not merged:
                out.append(list(r))
        return out

    @staticmethod
    def _filter_edge_rows(rows, W, H):
        """排除贴边的细小干扰文字（角标/顶栏）。"""
        if len(rows) < 2:
            return rows
        hs = [r[1] - r[0] for r in rows]
        med_h = float(sorted(hs)[len(hs) // 2])
        mx, my = 0.10 * W, 0.08 * H
        out = []
        for r in rows:
            y0, y1, x0, x1, c, t = r
            h = y1 - y0
            at_edge = x1 < mx or x0 > W - mx or y1 < my or y0 > H - my
            small = h <= med_h * 0.55 or len(str(t).strip()) <= 4
            if at_edge and small:
                continue
            out.append(r)
        return out

    @staticmethod
    def _filter_main_rows(rows):
        """按行高/行宽中位数去掉碎片和低置信度框，保留主文字块。"""
        if len(rows) <= 3:
            return rows
        hs = [r[1] - r[0] for r in rows]
        ws = [r[3] - r[2] for r in rows]
        med_h = float(sorted(hs)[len(hs) // 2])
        med_w = float(sorted(ws)[len(ws) // 2])
        kept = [
            r for r in rows
            if (r[1] - r[0]) >= 0.5 * med_h
            and (r[3] - r[2]) >= 0.5 * med_w
            and r[4] >= 0.5
        ]
        if not kept:
            kept = rows
        kept.sort(key=lambda r: (r[0] + r[1]) / 2)
        out = []
        for r in kept:
            if out:
                pr = out[-1]
                yo = min(pr[1], r[1]) - max(pr[0], r[0])
                if yo > 0.5 * min(pr[1] - pr[0], r[1] - r[0]) and r[2] < pr[3] and r[3] > pr[2]:
                    if r[4] > pr[4]:
                        out[-1] = r
                    continue
            out.append(r)
        return out

    @staticmethod
    def _postprocess_text(txt):
        """把点阵字中常见的 . : 误识别纠正为逗号，保留小数点和真实冒号之外的情况。"""
        if not txt:
            return txt
        out = list(txt)
        n = len(out)
        for i in range(n):
            c = out[i]
            if c == ".":
                if i == 0 or i == n - 1:
                    continue
                prev_ok = out[i - 1].isalnum()
                next_ok = out[i + 1].isalnum()
                if not (prev_ok and next_ok):
                    continue
                window = txt[max(0, i - 4):i]
                if "," in window[-3:]:
                    continue
                k = 0
                j = i - 1
                while j >= 0 and out[j].isdigit():
                    k += 1
                    j -= 1
                if out[i - 1].isalpha() or k >= 3:
                    out[i] = ","
            elif c == ":":
                if i > 0 and i < n - 1 and out[i - 1].isalnum() and out[i + 1].isalnum():
                    out[i] = ","
        return "".join(out)

    @staticmethod
    def _fine_segments(gray, y0, y1, x0, x1, min_h=5, gap=1):
        """高框内部按投影拆行。"""
        roi = gray[max(0, y0 - 4):min(gray.shape[0], y1 + 4), max(0, x0 - 6):min(gray.shape[1], x1 + 6)]
        bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 15)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        proj = bw.sum(axis=1)
        th = max(proj.max() * 0.04, 3)
        segs = []; on = False
        for y in range(len(proj)):
            if proj[y] > th:
                if not on: s = y; on = True
            else:
                if on:
                    if y - s >= min_h: segs.append((s, y))
                    on = False
        if on and len(proj) - s >= min_h: segs.append((s, len(proj)))
        m = []
        for s, e in segs:
            if m and s - m[-1][1] <= gap: m[-1] = (m[-1][0], e)
            else: m.append((s, e))
        return [(max(0, y0 - 4 + s), min(gray.shape[0], y0 - 4 + e)) for s, e in m]

    def _split_tall_rows(self, rows, gray, H):
        out = []
        for y0, y1, x0, x1, c, t in rows:
            if y1 - y0 > 55:
                segs = self._fine_segments(gray, y0, y1, x0, x1)
                good = [s for s in segs if 8 <= s[1] - s[0] <= 70]
                if len(good) >= 2:
                    out.extend([[s, e, x0, x1, c, t] for s, e in good])
                    continue
            out.append([y0, y1, x0, x1, c, t])
        return out

    @staticmethod
    def _rectify_line(gray, box):
        """按检测框把文字行摆平并裁剪。"""
        rect = cv2.minAreaRect(np.array(box, dtype=np.float32))
        w, h = int(round(rect[1][0])), int(round(rect[1][1]))
        if w < h:
            w, h = h, w
        if w <= 0 or h <= 0:
            return None, None
        src = cv2.boxPoints(rect).astype(np.float32)
        # 排序为 左上、右上、右下、左下
        top = src[np.argsort(src[:, 1])[:2]]
        bottom = src[np.argsort(src[:, 1])[2:]]
        tl = top[np.argmin(top[:, 0])]; tr = top[np.argmax(top[:, 0])]
        bl = bottom[np.argmin(bottom[:, 0])]; br = bottom[np.argmax(bottom[:, 0])]
        src_order = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_order, dst)
        line = cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return line, rect

    def recognize_text(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self._sv("filter","01_gray.png",gray)

        self._load_det4()
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        H,W=gray.shape

        rows=self._det_lines4(rgb)
        rows=self._merge_det_lines(rows)
        rows=self._filter_merge_rows(rows,H)
        rows=self._split_tall_rows(rows,gray,H)
        rows=self._filter_merge_rows(rows,H)

        # 不足 3 行时用放大图补充检测
        if len(rows)<3:
            for scale in (2.0, 1.5, 2.5):
                big = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for r in self._det_lines4(big):
                    r[0] = int(r[0] / scale); r[1] = int(r[1] / scale)
                    r[2] = int(r[2] / scale); r[3] = int(r[3] / scale)
                    rows.append(r)
                rows = self._merge_det_lines(rows)
                rows = self._filter_merge_rows(rows, H)
                rows = self._split_tall_rows(rows, gray, H)
                rows = self._filter_merge_rows(rows, H)
                if len(rows) >= 3:
                    break
            if len(rows) == 0:
                enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
                enh_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
                big2 = cv2.resize(enh_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                for r in self._det_lines4(big2):
                    r[0] = int(r[0] / 2); r[1] = int(r[1] / 2)
                    r[2] = int(r[2] / 2); r[3] = int(r[3] / 2)
                    rows.append(r)
                rows = self._merge_det_lines(rows)
                rows = self._filter_merge_rows(rows, H)
                rows = self._split_tall_rows(rows, gray, H)
                rows = self._filter_merge_rows(rows, H)
        rows = self._filter_edge_rows(rows, W, H)
        rows = self._filter_main_rows(rows)
        rows.sort(key=lambda r:r[0])
        if not rows:
            return []

        if self._d:
            v=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            for y0,y1,x0,x1,c,t in rows:
                cv2.rectangle(v,(x0,y0),(x1,y1),(0,255,0),2)
            self._sv("lines","07_lines.png",v)

        results=[]
        all_texts=[]
        for idx,(y0,y1,x0,x1,c,t) in enumerate(rows):
            box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],dtype=np.float32)
            line, _ = self._rectify_line(gray, box)
            if line is None or line.size==0:
                continue
            self._sv("deskew",f"line{idx}.png",line)

            txt, conf = self._recognize_line_rec6(line)
            if not txt:
                cc = self._split_chars(line)
                if self._d:
                    lv = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
                    for a, b, _, _ in cc:
                        cv2.rectangle(lv, (a, 0), (b, line.shape[0]), (0, 255, 0), 1)
                    self._sv("split", f"line{idx}_chars.png", lv)
                txt = ""
                for ci, (a, b, cy0, cy1) in enumerate(cc):
                    roi = line[cy0:cy1 + 1, a:b + 1]
                    if roi.size == 0: continue
                    self._sv("chars", f"l{idx}c{ci}.png", roi)
                    txt += self._recognize_char(roi)
                conf = 1.0
            txt = self._postprocess_text(txt)
            if txt:
                results.append({"text":txt,"confidence":float(conf),"bbox":[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]})
                all_texts.append(txt)

        self._save_result(all_texts)
        return results

    def _small_mask(self, gray):
        enh = cv2.createCLAHE(clipLimit=SPLIT_CLAHE_CLIP, tileGridSize=(8, 8)).apply(gray)
        bw = cv2.adaptiveThreshold(
            enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            SPLIT_ADAPTIVE_BLOCK, SPLIT_ADAPTIVE_C,
        )
        return cv2.morphologyEx(
            bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )

    def _small_rect_line(self, gray):
        bw = self._small_mask(gray)
        coords = cv2.findNonZero(bw)
        if coords is None or len(coords) < 5:
            return None
        rect = cv2.minAreaRect(coords)
        (cx, cy), (rw, rh), angle = rect
        if rw < rh:
            angle -= 90.0
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        coords2 = cv2.findNonZero(self._small_mask(rotated))
        if coords2 is None:
            return None
        x, y, w2, h2 = cv2.boundingRect(coords2)
        if w2 <= 0 or h2 <= 0:
            return None
        return rotated[y:y + h2, x:x + w2]

    def _small_component_rows(self, gray):
        bw = self._small_mask(gray)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = (
                stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                stats[i, cv2.CC_STAT_AREA],
            )
            if area < 4 or w < 2 or h < 2:
                continue
            if w > 4 * h or h > 4 * w:
                continue
            comps.append([x, x + w, y, y + h, area])
        if not comps:
            return []
        comps.sort(key=lambda c: (c[2] + c[3]) / 2.0)
        groups = []
        for c in comps:
            placed = False
            for g in groups:
                gy0 = min(x[2] for x in g)
                gy1 = max(x[3] for x in g)
                overlap = min(gy1, c[3]) - max(gy0, c[2])
                if overlap > 0.30 * min(gy1 - gy0, c[3] - c[2]):
                    g.append(c)
                    placed = True
                    break
            if not placed:
                groups.append([c])
        out = []
        for g in groups:
            y0 = max(0, min(x[2] for x in g) - 2)
            y1 = min(gray.shape[0], max(x[3] for x in g) + 2)
            x0 = max(0, min(x[0] for x in g) - 2)
            x1 = min(gray.shape[1], max(x[1] for x in g) + 2)
            if y1 - y0 < 6 or x1 - x0 < 10:
                continue
            line = self._small_rect_line(gray[y0:y1, x0:x1])
            if line is not None:
                out.append(line)
        return out

    @staticmethod
    def _small_merge_boxes(boxes):
        if len(boxes) <= 1:
            return boxes
        med_h = float(max(b[1] - b[0] for b in boxes))
        boxes = sorted(boxes, key=lambda b: (b[0] + b[1]) / 2)
        out = []
        for b in boxes:
            yc = (b[0] + b[1]) / 2
            placed = False
            for m in out:
                myc = (m[0] + m[1]) / 2
                if abs(yc - myc) <= 0.45 * med_h:
                    m[0] = min(m[0], b[0]); m[1] = max(m[1], b[1])
                    m[2] = min(m[2], b[2]); m[3] = max(m[3], b[3])
                    m[4] = max(m[4], b[4])
                    placed = True
                    break
            if not placed:
                out.append(list(b))
        return out

    def _small_det_lines(self, image):
        self._load_det4()
        if self._det4 is None:
            return []
        h, w = image.shape[:2]
        canvas = np.full(
            (max(h, SMALL_CANVAS_H), max(w, SMALL_CANVAS_W), 3), 255, dtype=np.uint8
        )
        oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
        canvas[oy:oy + h, ox:ox + w] = image
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res = list(self._det4.predict(input=rgb, batch_size=1))[0]
        polys = res.get("dt_polys")
        scores = res.get("dt_scores")
        boxes = []
        if polys is not None and len(polys) > 0:
            for poly, sc in zip(polys, scores):
                pts = np.asarray(poly, dtype=np.float32)
                boxes.append([
                    int(pts[:, 1].min()), int(pts[:, 1].max()),
                    int(pts[:, 0].min()), int(pts[:, 0].max()), float(sc), "",
                ])
        boxes = self._filter_merge_rows(self._small_merge_boxes(boxes), canvas.shape[0])
        lines = []
        for a, b, c, d, s, t in sorted(boxes, key=lambda r: r[0]):
            pad_x = max(12, int((d - c) * 0.08))
            seg = gray[
                max(0, a - 3):min(gray.shape[0], b + 3),
                max(0, c - pad_x):min(gray.shape[1], d + pad_x),
            ]
            line = self._small_rect_line(seg)
            if line is not None:
                lines.append(line)
        return lines

    def _recognize_small_text(self, image):
        self._load_det4()
        self._load_rec6()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._sv("small", "01_gray.png", gray)
        h, w = image.shape[:2]
        lines = []
        if w / h >= SMALL_LINE_ASPECT:
            single = self._small_rect_line(gray)
            lines = [single] if single is not None else []
        if not lines:
            lines = self._small_det_lines(image)
        if not lines:
            lines = self._small_component_rows(gray)
        results = []
        for idx, line in enumerate(lines):
            self._sv("small", f"line{idx}.png", line)
            txt, conf = self._recognize_line_rec6(line)
            txt = self._postprocess_text(txt)
            if txt:
                results.append({"text": txt, "confidence": float(conf), "bbox": None})
        return results

    def recognize_all(self,image,name="img"):
        self._bg(name)
        qr=self.recognize_qr(image)
        if qr: return {"qr_codes":qr,"texts":[],"image_shape":image.shape}
        if image.shape[0] < SMALL_IMG_MIN_H or image.shape[1] < SMALL_IMG_MIN_W:
            return {"qr_codes":[],"texts":self._recognize_small_text(image),"image_shape":image.shape}
        return {"qr_codes":[],"texts":self.recognize_text(image),"image_shape":image.shape}

    @staticmethod
    def load_image(path):
        with open(str(path),"rb") as f: b=np.frombuffer(f.read(),dtype=np.uint8)
        img=cv2.imdecode(b,cv2.IMREAD_COLOR)
        if img is None: raise ValueError(f"无法加载: {path}")
        return img
