import json, os, sys, threading, time, unittest, urllib.request
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import app

PORT = 8899


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = np.zeros((240, 320, 3), np.uint8)
        img[:, :] = np.linspace(0, 255, 320).astype(np.uint8)[None, :, None]
        cls.path = os.path.join(HERE, "_synthetic.png")
        cv2.imwrite(cls.path, img)
        cls.eng = app.Engine(cls.path, stub=True, speech=False)
        cls.eng.start()
        cls.srv = app.serve(cls.eng, PORT)
        time.sleep(2.5)

    @classmethod
    def tearDownClass(cls):
        cls.eng.stop = True
        cls.srv.shutdown()
        os.remove(cls.path)

    def get(self, p):
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{p}", timeout=5)

    def test_state(self):
        s = json.loads(self.get("/api/state").read())
        self.assertEqual(set(s["zones"]), {"left", "center", "right"})
        for z in s["zones"].values():
            self.assertIn(z["level"], ("clear", "caution", "stop"))
        self.assertGreater(s["fps"], 0)
        self.assertIn("Prototype", s["safety"])

    def test_stream(self):
        r = self.get("/stream.mjpg")
        data = r.read(4000)
        self.assertIn(b"--frame", data)
        self.assertIn(b"image/jpeg", data)

    def test_pages(self):
        for p, needle in (("/", b"Iris"), ("/caregiver", b"Caregiver"), ("/app.js", b"describe")):
            self.assertIn(needle, self.get(p).read())

    def test_mute_describe(self):
        post = lambda p: json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{p}", method="POST"), timeout=10).read())
        self.assertTrue(post("/api/mute")["muted"] is False)
        self.assertTrue(post("/api/describe")["text"])
        self.assertTrue(post("/api/mute")["muted"] is True)


if __name__ == "__main__":
    unittest.main()
