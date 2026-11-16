import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guide  # noqa: E402


def obs(l="clear", c="clear", r="clear", clear=None):
    z = lambda lv: {"near": {"clear": 0.1, "caution": 0.5, "stop": 0.9}[lv], "level": lv}
    return {"zones": {"left": z(l), "center": z(c), "right": z(r)}, "clear": clear, "nearest": 0.9}


def det(label, zone, score=0.8):
    return {"label": label, "score": score, "box": [0, 0, 10, 10], "zone": zone}


class GuideTest(unittest.TestCase):
    def test_warmup_is_silent(self):
        g = guide.Guide()   # default 2 s warm-up while the depth range settles
        self.assertEqual(g.update(obs(c="stop"), [], 0.0) + g.update(obs(c="stop"), [], 1.0) + g.update(obs(c="stop"), [], 1.9), [])
        self.assertEqual(g.update(obs(c="stop"), [], 2.1) + g.update(obs(c="stop"), [], 2.2) != [], True)

    def test_stop_does_not_chatter_when_label_flips(self):
        g = guide.Guide(warmup_s=0, cooldown_s=2.5)
        texts = []
        for k in range(20):   # 2 s at 10 fps, the detected label alternates every frame (seen live on the glasses)
            lab = "clock" if k % 2 else "chair"
            texts += [e["text"] for e in g.update(obs(c="stop"), [{"label": lab, "score": 0.8, "box": [280, 200, 80, 80], "zone": "center"}], k * 0.1)]
        self.assertEqual(len(texts), 1, texts)

    def test_hysteresis(self):
        g = guide.Guide(warmup_s=0)
        self.assertEqual(g.update(obs(c="stop"), [], 0.0), [])  # 1 frame: nothing
        self.assertEqual(g.update(obs(), [], 0.1), [])  # flicker back: reset
        self.assertEqual(g.update(obs(c="stop"), [], 0.2), [])
        ev = g.update(obs(c="stop"), [], 0.3)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["level"], "stop")

    def test_stop_phrasing_uses_label(self):
        g = guide.Guide(warmup_s=0)
        g.update(obs(c="stop"), [det("chair", "center")], 0)
        ev = g.update(obs(c="stop"), [det("chair", "center")], 0.1)
        self.assertEqual(ev[0]["text"], "Stop. Chair ahead.")

    def test_caution_side_and_clear_direction(self):
        g = guide.Guide(warmup_s=0)
        o = obs(l="caution", clear="right")
        g.update(o, [], 0)
        ev = g.update(o, [], 0.1)
        self.assertEqual(ev[0]["text"], "Obstacle on your left. Path clear on the right.")
        self.assertEqual(ev[0]["level"], "caution")

    def test_cooldown(self):
        g = guide.Guide(cooldown_s=2.5, warmup_s=0)
        o = obs(c="stop")
        g.update(o, [], 0)
        self.assertEqual(len(g.update(o, [], 0.1)), 1)
        for t in (0.5, 1.0, 2.0):
            self.assertEqual(g.update(o, [], t), [])
        self.assertEqual(len(g.update(o, [], 6.0)), 1)

    def test_stop_interrupts_caution(self):
        g = guide.Guide(warmup_s=0)
        c = obs(c="caution")
        g.update(c, [], 0)
        self.assertEqual(g.update(c, [], 0.1)[0]["level"], "caution")
        s = obs(c="stop")
        g.update(s, [], 0.2)
        ev = g.update(s, [], 0.3)  # well inside cooldown
        self.assertEqual(ev[0]["level"], "stop")

    def test_clear_spoken_rarely(self):
        g = guide.Guide(warmup_s=0)
        c = obs(c="caution")
        g.update(c, [], 0); g.update(c, [], 0.1)
        g.update(obs(), [], 5.0)
        ev = g.update(obs(), [], 5.1)
        self.assertEqual([e["text"] for e in ev], ["Path clear ahead."])
        self.assertEqual(g.update(obs(), [], 6.0), [])  # silent while clear

    def test_zone_of(self):
        self.assertEqual(guide.zone_of([0, 0, 100, 10], 640), "left")
        self.assertEqual(guide.zone_of([270, 0, 100, 10], 640), "center")
        self.assertEqual(guide.zone_of([500, 0, 100, 10], 640), "right")


class SpeakerTest(unittest.TestCase):
    def test_dry_run_latest_wins_and_mute(self):
        import time
        s = guide.Speaker(dry_run=True)
        s.say("a"); s.say("b"); s.say("c", urgent=True)
        time.sleep(0.3)
        self.assertEqual(s.spoken[-1], "c")
        n = len(s.spoken)
        s.set_mute(True); s.say("d")
        time.sleep(0.2)
        self.assertEqual(len(s.spoken), n)
        s.close()


class DescribeTest(unittest.TestCase):
    def test_local(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            t = guide.describe(None, [det("chair", "left")], obs(l="caution", clear="right"))
        self.assertIn("chair on your left", t)
        self.assertIn("right", t)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            self.assertIn("clear", guide.describe(None, [], obs()).lower())


class DetectorTest(unittest.TestCase):
    def test_blank_frame(self):
        d = guide.Detector()
        out = d.detect(np.zeros((480, 640, 3), np.uint8), now=100.0)
        self.assertIsInstance(out, list)
        self.assertIs(d.detect(np.zeros((480, 640, 3), np.uint8), now=100.1), d._last)


if __name__ == "__main__":
    unittest.main()
