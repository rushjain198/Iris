"""Color naming, SOS raise/clear over HTTP, and the camera fall check (lurch then stillness -> 'are you okay?' -> SOS)."""
import json, os, sys, time, unittest, urllib.request
import cv2, numpy as np
os.environ["IRIS_NO_ALARM"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import app, color  # noqa: E402

PORT = 8898


class TestColor(unittest.TestCase):
    def test_named_colors(self):
        for name, bgr in (("red", (30, 30, 210)), ("blue", (200, 80, 20)), ("green", (40, 170, 40)), ("white", (245, 245, 245)), ("black", (15, 15, 15))):
            self.assertEqual(color.dominant_color(np.full((480, 640, 3), bgr, np.uint8))["name"], name)

    def test_centre_wins_over_edges(self):
        img = np.full((480, 640, 3), (200, 80, 20), np.uint8); img[150:330, 200:440] = (30, 30, 210)   # red object, blue room
        self.assertEqual(color.dominant_color(img)["name"], "red")


class TestSOS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = np.full((240, 320, 3), 120, np.uint8)
        cls.path = os.path.join(HERE, "_sos.png"); cv2.imwrite(cls.path, img)
        cls.eng = app.Engine(cls.path, stub=True, speech=False); cls.eng.start()
        cls.srv = app.serve(cls.eng, PORT); time.sleep(2.0)

    @classmethod
    def tearDownClass(cls):
        cls.eng.stop = True; cls.srv.shutdown(); os.remove(cls.path)

    def post(self, p):
        return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{p}", method="POST"), timeout=5).read())

    def state(self):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=5).read())

    def test_sos_raise_and_clear(self):
        self.assertIn("reason", self.post("/api/sos")["sos"])
        self.assertTrue(self.state()["sos"])
        self.assertTrue(self.post("/api/okay")["cleared"])
        self.assertIsNone(self.state()["sos"])

    def test_color_endpoint(self):
        self.assertIn("gray", self.post("/api/color")["text"])

    def test_fall_check_asks_then_sos(self):
        e = self.eng
        e.watch, e.sos, e.check_t, e._prev_small, e._lurch_t, e._still_since = True, None, None, None, None, None
        dark, bright = np.zeros((240, 320, 3), np.uint8), np.full((240, 320, 3), 255, np.uint8)
        t = 1000.0
        e._fall_check(dark, t); e._fall_check(bright, t + 0.1)            # lurch: whole view changes
        for k in range(1, 40):                                            # then 3.9 s of a still view
            e._fall_check(bright, t + 0.1 + k * 0.1)
        self.assertIsNotNone(e.check_t, "should ask 'are you okay?'")
        e._fall_check(bright, e.check_t + 10.5)                           # no answer for 10 s
        self.assertTrue(e.sos)
        e.clear_sos(); e.watch = False


if __name__ == "__main__":
    unittest.main()
