"""Iris text reading: on-device OCR with Apple's Vision framework (the engine behind Live Text). No network, no key.

read_text(frame_bgr) -> [{"text", "conf", "box": [x, y, w, h] (image px, top-left origin), "area"}], biggest and most
central first. sentence(lines) -> what to say. Glasses frames are 640x480, so only large or close text is readable;
the frame is upscaled 2x before recognition, which helps small print a little.
"""
import cv2
import numpy as np

_MIN_CONF = 0.4


def read_text(frame_bgr, upscale=2.0, min_conf=_MIN_CONF):
    import Quartz
    import Vision
    from Foundation import NSData

    img = cv2.resize(frame_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC) if upscale != 1 else frame_bgr
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    h, w = frame_bgr.shape[:2]
    out = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        c = cands[0]
        text, conf = str(c.string()).strip(), float(c.confidence())
        if not text or conf < min_conf:
            continue
        bb = obs.boundingBox()                      # normalized, origin bottom-left
        x, y = bb.origin.x * w, (1 - bb.origin.y - bb.size.height) * h
        bw, bh = bb.size.width * w, bb.size.height * h
        out.append({"text": text, "conf": round(conf, 2), "box": [round(x), round(y), round(bw), round(bh)], "area": bw * bh})
    cx, cy = w / 2, h / 2
    # biggest text first, ties broken by closeness to the centre of view (where the wearer is looking)
    out.sort(key=lambda r: (-r["area"], abs(r["box"][0] + r["box"][2] / 2 - cx) + abs(r["box"][1] + r["box"][3] / 2 - cy)))
    return out


def sentence(lines, max_lines=4):
    if not lines:
        return "No readable text. Move closer or point straight at it."
    said = [l["text"] for l in lines[:max_lines]]
    more = len(lines) - len(said)
    return "Text: " + ". ".join(said) + (f". And {more} more lines." if more > 0 else ".")


if __name__ == "__main__":
    img = np.full((480, 640, 3), 245, np.uint8)
    cv2.putText(img, "EXIT", (200, 220), cv2.FONT_HERSHEY_DUPLEX, 3.2, (20, 20, 20), 6, cv2.LINE_AA)
    cv2.putText(img, "Room 204", (220, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    import time
    t = time.time(); r = read_text(img); dt = (time.time() - t) * 1000
    print(f"{dt:.0f} ms", [(x["text"], x["conf"]) for x in r]); print(sentence(r))
