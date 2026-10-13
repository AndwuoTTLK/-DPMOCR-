"""Recover an isolated, reddish dot-peen QR code from a green steel photo.

The lattice fit only locates and samples candidate QR modules. OpenCV's QR
decoder must accept the rendered code before this module returns any text.
"""

import cv2
import numpy as np
from scipy.optimize import differential_evolution


def _finder_samples(module_count):
    rows, cols, expected = [], [], []
    anchors = ((0, 0), (0, module_count - 7), (module_count - 7, 0))
    for row0, col0 in anchors:
        for row in range(7):
            for col in range(7):
                dark = (row in (0, 6) or col in (0, 6)
                        or (2 <= row <= 4 and 2 <= col <= 4))
                rows.append(row0 + row)
                cols.append(col0 + col)
                expected.append(dark)
    for index in range(8, module_count - 8):
        rows.extend((6, index))
        cols.extend((index, 6))
        expected.extend((index % 2 == 0,) * 2)
    return np.asarray(rows), np.asarray(cols), np.asarray(expected, dtype=bool)


def _sample(channel, rows, cols, params):
    x0, y0, step_x, shear_x, shear_y, step_y = params
    xs = x0 + cols * step_x + rows * shear_x
    ys = y0 + cols * shear_y + rows * step_y
    return cv2.remap(
        channel, xs.astype(np.float32).reshape(1, -1),
        ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=100,
    ).reshape(rows.shape)


def decode_dot_peen_qr(image):
    """Return (decoded text, four corners) or None for a supported image."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    height, width = image.shape[:2]
    if not (180 <= min(height, width) <= 1200 and 0.72 <= width / height <= 1.4):
        return None
    image_scale = 340 / max(height, width)
    if abs(image_scale - 1) > 0.01:
        image = cv2.resize(
            image, None, fx=image_scale, fy=image_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = image.shape[:2]

    color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
    background = float(np.median(color))
    if background >= 121 or np.percentile(color, 90) - background < 11:
        return None
    ys, xs = np.where(color > max(background + 8, 120))
    if len(xs) < height * width * 0.025:
        return None
    left, right = np.percentile(xs, (2, 98))
    top, bottom = np.percentile(ys, (2, 98))
    span_x, span_y = right - left, bottom - top
    if not (width * 0.35 <= span_x <= width * 0.95
            and height * 0.35 <= span_y <= height * 0.95):
        return None

    smooth = cv2.GaussianBlur(color, (3, 3), 0.8)
    decoder = cv2.QRCodeDetector()
    for count in (29, 25, 33, 21, 37):
        rows, cols, expected = _finder_samples(count)
        step_x = span_x / (count - 1)
        step_y = span_y / (count - 1)
        bounds = (
            (left - step_x, left + step_x),
            (top - step_y, top + 1.5 * step_y),
            (0.85 * step_x, 1.15 * step_x),
            (-0.05 * step_x, 0.05 * step_x),
            (-0.05 * step_y, 0.05 * step_y),
            (0.85 * step_y, 1.15 * step_y),
        )

        def loss(params):
            values = _sample(smooth, rows, cols, params)
            return float(values[~expected].mean() - values[expected].mean())

        fit = differential_evolution(
            loss, bounds, maxiter=90, popsize=9, seed=4, polish=False,
        )
        if -fit.fun < 12:
            continue

        grid_rows, grid_cols = np.mgrid[:count, :count]
        center_x = fit.x[0] + grid_cols * fit.x[2] + grid_rows * fit.x[3]
        center_y = fit.x[1] + grid_cols * fit.x[4] + grid_rows * fit.x[5]
        finder_values = _sample(smooth, rows, cols, fit.x)
        midpoint = (finder_values[expected].mean()
                    + finder_values[~expected].mean()) / 2
        for center_shift_x in (0, -0.5, 0.5):
            values = np.stack([
                _sample(color, grid_rows, grid_cols,
                        (fit.x[0] + center_shift_x + offset_x,
                         fit.x[1] + offset_y, *fit.x[2:]))
                for offset_y in (-1, 0, 1)
                for offset_x in (-1, 0, 1)
            ]).max(axis=0)
            for threshold in range(int(midpoint) - 5, int(midpoint) + 6):
                modules = values > threshold
                rendered = np.pad(
                    np.kron((~modules).astype(np.uint8) * 255,
                            np.ones((10, 10), dtype=np.uint8)),
                    40, constant_values=255,
                )
                text, _, _ = decoder.detectAndDecode(rendered)
                if text:
                    corners = [
                        (int(center_x[0, 0]), int(center_y[0, 0])),
                        (int(center_x[0, -1]), int(center_y[0, -1])),
                        (int(center_x[-1, -1]), int(center_y[-1, -1])),
                        (int(center_x[-1, 0]), int(center_y[-1, 0])),
                    ]
                    return text, [
                        (int(x / image_scale), int(y / image_scale))
                        for x, y in corners
                    ]
    return None
