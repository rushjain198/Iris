# Iris — sight for the path ahead

Iris turns a pair of camera glasses and a laptop into a navigation aid for blind and low-vision people.
One small camera on the glasses, no LiDAR, everything processed on the laptop.

**What it does**

| Feature | How |
|---|---|
| Obstacle guidance — "Stop. Chair ahead." / "Obstacle on your left. Path clear on the right." | Depth Anything V2 (single-camera depth) → left / center / right walking zones, with hysteresis so it doesn't chatter |
| Names what is in the way | MediaPipe EfficientDet-Lite0 (80 everyday object classes) |
| Reads signs and labels | Apple Vision text recognition, on-device |
| "What color is this?" | Dominant color of the centre of view |
| Describe the scene | One-sentence summary (Gemini if `GOOGLE_API_KEY` is set, otherwise local) |
| Voice commands — "Iris, read this", "what's ahead", "what color", "help", "I'm okay" | Chrome speech recognition in the dashboard tab |
| SOS + fall check | "Help" → spoken alert, alarm, red banner on the caregiver page. Optional fall check: sudden view change then stillness → "Are you okay?" → SOS if no answer in 10 s |
| Caregiver view | Live picture, path state and everything the wearer heard |
| HUD | Full-screen heads-up display drawn over the live camera (`/hud`), with freeze-frame for screenshots |

> **Prototype. Not a mobility aid replacement: keep using your cane or guide dog.**

## Run

```bash
./setup.sh                 # once: Python venv + dependencies (models download on first run, ~41 MB)
./go.sh glasses            # glasses on USB-C   (or: ./go.sh webcam   ·   ./go.sh file clip.mp4)
```

Open http://127.0.0.1:8810 in Chrome. Pages: `/` dashboard · `/hud` heads-up display · `/caregiver` caregiver view.
Keys on the dashboard: D describe · R read text · C color · H SOS · O I'm okay · V voice commands · F fall check · M mute.
HUD: Space freeze frame · F fullscreen · H hide depth view.

The glasses camera re-connects by itself if the cable is unplugged. If the image is upside down, pass `--rotate 0`
(the default for glasses is 180, for a camera mounted upside down).

Tests: `./go.sh test`

## Repository layout

```
app.py            camera → depth → zones → guidance; web server (dashboard, HUD, caregiver, JSON API)
camera.py         frame sources: glasses (USB serial), webcam, image or video file
usbcam.py         reads JPEG frames streamed by the glasses firmware over USB-C
depth.py          Depth Anything V2 Small (ONNX) + obstacle zones
guide.py          what to say and when (hysteresis, cooldown), speech, object detector, scene description
ocr.py            text reading (Apple Vision)
color.py          color identification
web/              dashboard, HUD and caregiver pages
tests/            unit + HTTP tests
firmware/iris_cam_usb/   ESP32-S3 camera firmware: streams JPEG frames over the USB-C cable
cad/              assembles the printed glasses-frame parts into one positioned model
demo/             builds the split-screen demo video (walk + HUD + rotating LiDAR scan)
```

### Glasses firmware

`firmware/iris_cam_usb/iris_cam_usb.ino` — Arduino IDE, board **XIAO_ESP32S3**, PSRAM = OPI PSRAM,
USB Mode = Hardware CDC and JTAG, USB CDC On Boot = Enabled. Frames are sent as
`0xA5 0x5A 'J' 'P' | uint32 length | JPEG bytes`.

### Glasses frame CAD

```bash
.venv/bin/python cad/assemble_glasses.py PARTS_DIR OUT_DIR
```
Positions the flat, print-ready frame parts (front, back frame, temples, covers) by geometry fits and writes
`OUT_DIR/glasses_assembly.stl` plus each part in place.

### Demo video

```bash
demo/compose_demo.sh WALK.mp4 HUD.mov SCAN.stl OUT.mp4
```
45 s, 1920×1080: the walk on the left, the HUD recording (with its audio) top right, the LiDAR scan rotating bottom
right. See the header of `compose_demo.sh` for the sync and cut settings.

## How it works

1. The glasses stream 640×480 JPEG frames over USB-C (~10–16 fps on an M-series MacBook).
2. Depth Anything V2 Small (int8 ONNX, ~43 ms per frame on CPU) gives relative depth; it is normalised over time so
   a far, empty scene stays "far".
3. The lower two thirds of the image (the walking band) is split into left / center / right; each zone is
   clear / caution / stop by its nearest 5%.
4. The guide speaks only on a level change that holds for 2 frames, never repeats a line inside its cooldown, and
   lets "Stop" interrupt anything else.

Depth from one camera is relative, not metric: Iris knows *nearer* and *farther*, not meters.

## Credits

Depth Anything V2 (Apache-2.0) · MediaPipe (Apache-2.0) · Apple Vision · OpenCV.
