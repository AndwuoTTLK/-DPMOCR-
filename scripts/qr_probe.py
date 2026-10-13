"""Compare QR decoding on the bundled sample images without changing them."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from recognizer import ImageRecognizer


def variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    width = image.shape[1]
    candidates = {
        "original": image,
        "gray": gray,
        "clahe": clahe,
        "left_half": image[:, : width // 2],
        "left_half_clahe": clahe[:, : width // 2],
    }
    for label, candidate in candidates.items():
        for scale in (1, 2, 3):
            scaled = candidate if scale == 1 else cv2.resize(
                candidate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            yield f"{label}_x{scale}", scaled


def main():
    recognizer = ImageRecognizer()
    recognizer._load_wechat_qr()
    if recognizer._wq is None:
        raise RuntimeError("WeChat QR model did not load")
    counts = {}
    for path in sorted((ROOT / "Test").glob("*.png")):
        image = ImageRecognizer.load_image(path)
        hits = []
        for label, candidate in variants(image):
            texts, _ = recognizer._wq.detectAndDecode(candidate)
            valid = [text for text in texts if text]
            if valid:
                hits.append((label, valid))
                counts[label] = counts.get(label, 0) + 1
        print(path.name, hits, flush=True)
    print("variant_counts", counts)


if __name__ == "__main__":
    main()
