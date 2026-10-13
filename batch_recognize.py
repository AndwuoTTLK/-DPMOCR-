"""Batch image recognition without starting the Tkinter interface."""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = (
    "source", "status", "image_width", "image_height", "kind", "index",
    "text", "confidence", "bbox", "error",
)


def collect_images(inputs):
    """Expand files, directories and shell-independent wildcard patterns."""
    found = set()
    for item in inputs:
        candidates = glob.glob(str(item), recursive=True) if glob.has_magic(str(item)) else [item]
        matched = False
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_dir():
                files = path.rglob("*")
            else:
                files = (path,)
            for file in files:
                if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES:
                    found.add(file.resolve())
                    matched = True
        if not matched:
            raise ValueError(f"找不到输入路径中的图片: {item}")
    return sorted(found)


def _output_paths(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        stem = "results" if number == 1 else f"results_{number:03d}"
        jsonl_path = directory / f"{stem}.jsonl"
        csv_path = directory / f"{stem}.csv"
        if not jsonl_path.exists() and not csv_path.exists():
            return jsonl_path, csv_path
    raise RuntimeError("输出目录中的结果文件过多")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"无法写入 JSON: {type(value).__name__}")


def _write_csv_record(writer, record):
    base = {
        "source": record["source"], "status": record["status"],
        "image_width": record["image_width"],
        "image_height": record["image_height"], "error": record["error"],
    }
    items = [("qr", item) for item in record["qr_codes"]]
    items += [("text", item) for item in record["texts"]]
    if not items:
        writer.writerow(base)
    for index, (kind, item) in enumerate(items, 1):
        writer.writerow({
            **base, "kind": kind, "index": index, "text": item.get("text", ""),
            "confidence": item.get("confidence", ""),
            "bbox": json.dumps(item.get("bbox"), ensure_ascii=False,
                               default=_json_default),
        })


def run_batch(inputs, output_dir, recognizer=None, stream=None):
    """Recognize each image and save both per-image JSONL and flat CSV rows."""
    images = collect_images(inputs)
    if not images:
        raise ValueError("输入路径中没有找到支持的图片")
    if recognizer is None:
        from recognizer import ImageRecognizer
        recognizer = ImageRecognizer(debug=False)
    if stream is None:
        stream = sys.stdout
    jsonl_path, csv_path = _output_paths(Path(output_dir).expanduser())
    failed = 0
    unrecognized = 0

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_file, \
            csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for position, path in enumerate(images, 1):
            print(f"[{position}/{len(images)}] {path}", file=stream)
            record = {
                "source": str(path), "status": "ok", "image_width": None,
                "image_height": None, "qr_codes": [], "texts": [], "error": None,
            }
            try:
                image = recognizer.load_image(path)
                result = recognizer.recognize_all(image, path.name)
                height, width = result["image_shape"][:2]
                record["image_width"] = int(width)
                record["image_height"] = int(height)
                record["qr_codes"] = result["qr_codes"]
                record["texts"] = result["texts"]
                if not record["qr_codes"] and not record["texts"]:
                    record["status"] = "no_result"
                    unrecognized += 1
                    print("  未识别到内容", file=stream)
                for kind, items in (("二维码", record["qr_codes"]),
                                    ("文字", record["texts"])):
                    for index, item in enumerate(items, 1):
                        print(f"  {kind} {index}: {item['text']}", file=stream)
            except Exception as exc:
                failed += 1
                record["status"] = "error"
                record["qr_codes"] = []
                record["texts"] = []
                record["error"] = str(exc)
                print(f"  失败: {exc}", file=stream)
            jsonl_file.write(json.dumps(record, ensure_ascii=False,
                                        default=_json_default) + "\n")
            _write_csv_record(writer, record)
            jsonl_file.flush()
            csv_file.flush()

    print(f"完成: {len(images)} 张，未识别 {unrecognized} 张，失败 {failed} 张", file=stream)
    print(f"JSONL: {jsonl_path.resolve()}", file=stream)
    print(f"CSV:   {csv_path.resolve()}", file=stream)
    return {"total": len(images), "failed": failed, "unrecognized": unrecognized,
            "jsonl": jsonl_path, "csv": csv_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description="批量识别图片中的钢板二维码和文字")
    parser.add_argument("inputs", nargs="+", metavar="图片或文件夹",
                        help="可传多个图片、文件夹或通配符；文件夹会递归扫描")
    parser.add_argument("-o", "--output", default="batch_results", metavar="输出文件夹",
                        help="保存 JSONL 和 CSV 的文件夹，默认 batch_results")
    args = parser.parse_args(argv)
    try:
        summary = run_batch(args.inputs, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
