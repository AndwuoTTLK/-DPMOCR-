import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import App


class HistoryUITests(unittest.TestCase):
    def setUp(self):
        self.app = App()
        self.app.withdraw()
        self.image = np.zeros((24, 32, 3), dtype=np.uint8)

    def tearDown(self):
        self.app.destroy()

    def test_clear_keeps_history_and_selection_restores_image_and_result(self):
        entry = self.app._remember_image(self.image, "sample.png")
        result = {"image_shape": self.image.shape,
                  "qr_codes": [{"text": "SAMPLE-123"}], "texts": []}
        self.app._show_recognition_results(result, entry["id"])

        self.app._clear()
        self.assertEqual(len(self.app.history), 1)
        self.assertIsNone(self.app.current_image_bgr)

        self.app._activate_history(entry)
        self.assertIs(self.app.current_image_bgr, self.image)
        self.assertIn("SAMPLE-123", self.app.results_text.get("1.0", "end"))

    def test_late_result_does_not_replace_selected_older_image(self):
        older = self.app._remember_image(self.image, "older.png")
        older_result = {"image_shape": self.image.shape,
                        "qr_codes": [{"text": "OLDER"}], "texts": []}
        self.app._show_recognition_results(older_result, older["id"])
        newer = self.app._remember_image(self.image.copy(), "newer.png")

        self.app._activate_history(older)
        newer_result = {"image_shape": self.image.shape,
                        "qr_codes": [{"text": "NEWER"}], "texts": []}
        self.app._show_recognition_results(newer_result, newer["id"])

        self.assertIn("OLDER", self.app.results_text.get("1.0", "end"))
        self.assertNotIn("NEWER", self.app.results_text.get("1.0", "end"))
        self.app._activate_history(newer)
        self.assertIn("NEWER", self.app.results_text.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
