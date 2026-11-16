"""On-device OCR (Apple Vision): reads large rendered text, orders biggest first, says 'no text' on a blank frame."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ocr  # noqa: E402


class TestOCR(unittest.TestCase):
    def test_reads_sign_text_biggest_first(self):
        img = np.full((480, 640, 3), 240, np.uint8)
        cv2.putText(img, "Room 204", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(img, "EXIT", (190, 220), cv2.FONT_HERSHEY_DUPLEX, 3.4, (15, 15, 15), 7, cv2.LINE_AA)
        lines = ocr.read_text(img)
        self.assertEqual([l["text"] for l in lines][:2], ["EXIT", "Room 204"])
        self.assertTrue(ocr.sentence(lines).startswith("Text: EXIT. Room 204"))

    def test_blank_frame(self):
        lines = ocr.read_text(np.full((480, 640, 3), 128, np.uint8))
        self.assertEqual(lines, [])
        self.assertIn("No readable text", ocr.sentence(lines))


if __name__ == "__main__":
    unittest.main()
