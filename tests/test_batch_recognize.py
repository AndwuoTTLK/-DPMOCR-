import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_recognize import collect_images, run_batch


class FakeRecognizer:
    def load_image(self, path):
        return Path(path).name

    def recognize_all(self, image, name):
        if name == "bad.png":
            raise RuntimeError("decode failed")
        return {
            "image_shape": (10, 20, 3),
            "qr_codes": [{"text": "钢板-A01", "bbox": [[1, 2], [3, 4]]}],
            "texts": [],
        }


class BatchRecognizeTests(unittest.TestCase):
    def test_collects_directories_globs_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "子目录").mkdir()
            first = root / "a.png"
            second = root / "子目录" / "b.jpg"
            for path in (first, second, root / "note.txt"):
                path.write_bytes(b"x")
            found = collect_images([str(root), str(first), str(root / "*.png")])
            self.assertEqual(found, sorted([first.resolve(), second.resolve()]))

    def test_missing_input_is_reported_instead_of_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "找不到输入路径"):
                collect_images([str(Path(temp) / "missing.png")])

    def test_exports_machine_readable_files_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "输入 图片"
            source.mkdir()
            (source / "good.png").write_bytes(b"x")
            (source / "bad.png").write_bytes(b"x")
            stream = io.StringIO()

            summary = run_batch([str(source)], root / "结果", FakeRecognizer(), stream)

            records = [json.loads(line) for line in summary["jsonl"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual([record["status"] for record in records], ["error", "ok"])
            self.assertEqual(records[1]["qr_codes"][0]["text"], "钢板-A01")
            self.assertEqual(records[0]["error"], "decode failed")
            with summary["csv"].open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["text"], "钢板-A01")
            self.assertEqual(rows[0]["status"], "error")
            self.assertIn("钢板-A01", stream.getvalue())
            self.assertIn("decode failed", stream.getvalue())

    def test_repeated_run_keeps_previous_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "a.png"
            source.write_bytes(b"x")
            first = run_batch([str(source)], root / "结果", FakeRecognizer(), io.StringIO())
            second = run_batch([str(source)], root / "结果", FakeRecognizer(), io.StringIO())
            self.assertNotEqual(first["jsonl"], second["jsonl"])
            self.assertTrue(first["jsonl"].exists())


if __name__ == "__main__":
    unittest.main()
