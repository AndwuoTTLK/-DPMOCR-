import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recognizer import ImageRecognizer


class QRRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = ImageRecognizer()

    def test_low_contrast_steel_qr_2(self):
        image = ImageRecognizer.load_image(ROOT / 'Test' / '1 (2).png')
        decoded = self.recognizer.recognize_qr(image)
        self.assertEqual([item['text'] for item in decoded], [
            '91=H2543=204=GR0A-WR1=R/CG=11A==H4'
        ])

    def test_dot_peen_steel_qr_3(self):
        image = ImageRecognizer.load_image(ROOT / 'Test' / '1 (3).png')
        decoded = self.recognizer.recognize_qr(image)
        self.assertEqual([item['text'] for item in decoded], [
            '12=H2543=234=IB1A-S13=/PG=11A==L4'
        ])

    def test_clear_qr_still_decodes(self):
        image = ImageRecognizer.load_image(ROOT / 'Test' / '1 (11).png')
        decoded = self.recognizer.recognize_qr(image)
        self.assertEqual([item['text'] for item in decoded], [
            '35=H2543=233=01L-FR61E-S3=/C1L=11A==H4'
        ])

    def test_image_without_complete_qr_has_no_result(self):
        image = ImageRecognizer.load_image(ROOT / 'Test' / '1 (10).png')
        self.assertEqual(self.recognizer.recognize_qr(image), [])

    def test_isolated_steel_dot_qr_reconstructs(self):
        image = ImageRecognizer.load_image(
            ROOT / 'research_artifacts' / 'steel_dot_qr_source_2026-10-13.png'
        )
        decoded = self.recognizer.recognize_qr(image)
        self.assertEqual([item['text'] for item in decoded], [
            '8=H2543=234=01L-C2=/LG=12AH36==S4'
        ])

    def test_plain_square_does_not_create_a_qr(self):
        import numpy as np
        image = np.full((331, 340, 3), (150, 170, 130), dtype=np.uint8)
        self.assertEqual(self.recognizer.recognize_qr(image), [])

    def test_dot_grid_recovery_survives_image_resizing(self):
        import cv2
        from dot_peen_qr import decode_dot_peen_qr
        image = ImageRecognizer.load_image(
            ROOT / 'research_artifacts' / 'steel_dot_qr_source_2026-10-13.png'
        )
        for scale in (0.75, 1.25):
            with self.subTest(scale=scale):
                resized = cv2.resize(
                    image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                recovered = decode_dot_peen_qr(resized)
                self.assertIsNotNone(recovered)
                self.assertEqual(recovered[0], '8=H2543=234=01L-C2=/LG=12AH36==S4')


if __name__ == '__main__':
    unittest.main()
