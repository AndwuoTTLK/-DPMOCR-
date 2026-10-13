import sys
import tkinter as tk
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import clipboard_to_input, dropped_image_path


class ImageInputTests(unittest.TestCase):
    def test_pasted_rgb_image_becomes_bgr_pixels(self):
        image = Image.new("RGB", (2, 1), (10, 20, 30))
        pixels, label = clipboard_to_input(image)
        self.assertEqual(label, "clipboard")
        self.assertEqual(pixels.shape, (1, 2, 3))
        np.testing.assert_array_equal(pixels[0, 0], [30, 20, 10])

    def test_pasted_file_list_loads_image(self):
        path = ROOT / "Test" / "1 (2).png"
        pixels, label = clipboard_to_input([str(path)])
        self.assertEqual(label, path.name)
        self.assertEqual(pixels.ndim, 3)
        self.assertEqual(pixels.shape[2], 3)

    def test_dropped_path_preserves_spaces_and_braces(self):
        path = ROOT / "Test" / "1 (2).png"
        data = tk.Tcl().call("list", str(path), str(ROOT / "README.md"))
        self.assertEqual(dropped_image_path(data, tk.Tcl().splitlist), str(path))

    def test_non_image_file_is_rejected(self):
        with self.assertRaises(ValueError):
            clipboard_to_input([str(ROOT / "README.md")])


if __name__ == "__main__":
    unittest.main()
