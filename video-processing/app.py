import os
import uuid
import cv2
import torch
import logging
import subprocess
from flask import Flask, request, send_file, abort
from tempfile import TemporaryDirectory
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv, C2f, C3, Detect

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────
UPLOAD_DIR = "uploads"
MODEL_PATH = "yolov8n-face.pt"
MAX_VIDEO_DURATION = 900  # 15 minutes

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__)

# ── Safe Model Load ─────────────────────────────────────
add_safe_globals([
    DetectionModel,
    Sequential,
    Conv,
    C2f,
    C3,
    Detect,
    torch.nn.Module
])

model = YOLO(MODEL_PATH)

# ── Helpers ──────────────────────────────────────────────
def blur_faces(src_path: str, dst_path: str, blur_scene: bool = False) -> None:
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps or 1)
    if duration > MAX_VIDEO_DURATION:
        raise ValueError("Video duration exceeds 15-minute limit.")

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_factor = 0.5
    downscaled_w, downscaled_h = int(w * scale_factor), int(h * scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    intermediate = dst_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(intermediate, fourcc, fps, (downscaled_w, downscaled_h))

    if not out.isOpened():
        raise RuntimeError(f"Cannot write intermediate video: {intermediate}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (downscaled_w, downscaled_h), interpolation=cv2.INTER_AREA)

        if blur_scene:
            frame = cv2.GaussianBlur(frame, (25, 25), 0)
        else:
            results = model(frame, verbose=False)
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    k = max(15, ((x2 - x1) // 2) | 1)
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face, (k, k), 0)

        out.write(frame)

    cap.release()
    out.release()

    # Upscale to original resolution
    cap = cv2.VideoCapture(intermediate)
    out_final = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        out_final.write(frame)
    cap.release()
    out_final.release()
    os.remove(intermediate)

def merge_audio(video_path: str, audio_src: str, anonymize: bool) -> str:
    output_path = video_path.replace(".mp4", "_with_audio.mp4")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_src,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy"]

    if anonymize:
        cmd += ["-af", "asetrate=44100*0.8,atempo=1.25"]

    cmd += ["-c:a", "aac", "-shortest", "-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True)
    return output_path

# ── Routes ───────────────────────────────────────────────
@app.post("/blur_video")
def handle_blur():
    raw_path = None
    try:
        if "file" not in request.files:
            abort(400, "No file part 'file' found.")
        file = request.files["file"]
        if file.filename == "":
            abort(400, "Empty filename")

        blur_scene = request.form.get("blur_scene", "false").lower() == "true"
        anonymize_voice = request.form.get("anonymize_voice", "false").lower() == "true"

        uid = uuid.uuid4().hex
        raw_path = os.path.join(UPLOAD_DIR, f"{uid}_raw.mp4")
        file.save(raw_path)

        with TemporaryDirectory() as td:
            blurred_path = os.path.join(td, "blurred.mp4")

            # No rotation correction applied here
            blur_faces(raw_path, blurred_path, blur_scene=blur_scene)

            final_path = merge_audio(blurred_path, raw_path, anonymize=anonymize_voice)

            return send_file(final_path, mimetype="video/mp4", as_attachment=True, download_name="blurred_video.mp4")

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        abort(413, str(ve))
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        abort(500, "Failed to process video.")
    finally:
        if raw_path and os.path.exists(raw_path):
            os.remove(raw_path)

@app.get("/health")
def health():
    return {"status": "ok"}
