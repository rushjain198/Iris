import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import depth as D


class TestObstacles(unittest.TestCase):
    def test_left_blob(self):
        d = np.full((240, 300), 0.1, np.float32)
        d[100:240, 0:90] = 0.95
        o = D.obstacles(d)
        self.assertEqual(o["zones"]["left"]["level"], "stop")
        self.assertEqual(o["zones"]["center"]["level"], "clear")
        self.assertEqual(o["zones"]["right"]["level"], "clear")
        self.assertEqual(o["clear"], "center")
        self.assertAlmostEqual(o["nearest"], 0.95, places=3)

    def test_uniform_far_prefers_center(self):
        o = D.obstacles(np.full((120, 150), 0.05, np.float32))
        self.assertTrue(all(z["level"] == "clear" for z in o["zones"].values()))
        self.assertEqual(o["clear"], "center")

    def test_caution_level(self):
        d = np.full((240, 300), 0.1, np.float32)
        d[100:240, 200:300] = 0.6
        self.assertEqual(D.obstacles(d)["zones"]["right"]["level"], "caution")

    def test_all_blocked_none(self):
        o = D.obstacles(np.full((120, 150), 0.9, np.float32))
        self.assertIsNone(o["clear"])

    def test_upper_third_ignored(self):
        d = np.full((240, 300), 0.1, np.float32)
        d[:70] = 1.0
        self.assertTrue(all(z["level"] == "clear" for z in D.obstacles(d)["zones"].values()))

    def test_least_near_when_center_blocked(self):
        d = np.full((240, 300), 0.1, np.float32)
        d[100:, 100:200] = 0.9
        d[100:, 0:100] = 0.3
        self.assertEqual(D.obstacles(d)["clear"], "right")


class TestEstimator(unittest.TestCase):
    def test_real_model(self):
        p = os.path.join(D.MODEL_DIR, D.DEFAULT_MODEL)
        if not os.path.exists(p):
            self.skipTest("model missing")
        import cv2
        rng = np.random.default_rng(0)
        f = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 8)
        d = D.DepthEstimator(p).infer(f)
        self.assertEqual(d.shape, (480, 640))
        self.assertEqual(d.dtype, np.float32)
        self.assertGreaterEqual(float(d.min()), 0.0)
        self.assertLessEqual(float(d.max()), 1.0)
        self.assertGreater(float(d.max() - d.min()), 0.5)


if __name__ == "__main__":
    unittest.main()
