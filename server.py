import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
from vidgear.gears import CamGear

# ---------------------------------------------------------------------------
# Paths (all relative to this file's directory)
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_PATH = BASE.parent / "best.pt"
YOLO11_BASE = "yolo11m.pt"   # downloaded by ultralytics on first retrain
CLASSES_FILE = BASE / "classes.json"
ANNOTATIONS_DIR = BASE / "annotations"
IMAGES_DIR = ANNOTATIONS_DIR / "images"
LABELS_DIR = ANNOTATIONS_DIR / "labels"
DATA_YAML = BASE / "data.yaml"
STREAM_URL = "https://youtu.be/7i8ARjIeM2k"

# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------
def load_classes() -> list[str]:
    if CLASSES_FILE.exists():
        return json.loads(CLASSES_FILE.read_text())
    default = ["fish", "sergeant-major"]
    save_classes(default)
    return default

def save_classes(names: list[str]):
    CLASSES_FILE.write_text(json.dumps(names, indent=2))

# ---------------------------------------------------------------------------
# Global state (protected by locks where needed)
# ---------------------------------------------------------------------------
classes: list[str] = load_classes()
model = YOLO(str(MODEL_PATH))

_frame_lock = threading.Lock()
_raw_frame: np.ndarray | None = None
_annotated_frame: np.ndarray | None = None
_pending_frame: np.ndarray | None = None  # frame shown to user for annotation

_retrain_lock = threading.Lock()
_retrain_status = "idle"   # "idle" | "running" | "done"
_annotation_count = len(list(LABELS_DIR.glob("*.txt")))
_inferencing = True  # set to False during retraining to free MPS

# ---------------------------------------------------------------------------
# Background frame-capture thread
# ---------------------------------------------------------------------------
def _resolve_live_url(youtube_url: str) -> str:
    """Resolve a YouTube live URL to a direct stream URL at the live edge."""
    ydl_opts = {
        "format": "best[height<=1080]",
        "quiet": True,
        "no_warnings": True,
        "live_from_start": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def _capture_loop():
    global _raw_frame, _annotated_frame
    stream = None
    null_streak = 0
    NULL_RECONNECT = 50  # reconnect after this many consecutive None frames

    while True:
        if stream is None:
            try:
                print("Connecting to stream...")
                direct_url = _resolve_live_url(STREAM_URL)
                stream = CamGear(source=direct_url).start()
                null_streak = 0
                print("Stream connected.")
            except Exception as e:
                print(f"Stream connect failed: {e} — retrying in 10s")
                time.sleep(10)
                continue

        frame = stream.read()
        if frame is None:
            null_streak += 1
            if null_streak >= NULL_RECONNECT:
                print("Stream appears dead — reconnecting...")
                stream.stop()
                stream = None
            else:
                time.sleep(0.1)
            continue

        null_streak = 0
        if not _inferencing:
            with _frame_lock:
                _raw_frame = frame.copy()
            time.sleep(0.1)
            continue
        results = model.predict(frame, conf=0.6, device="mps", verbose=False)
        annotated = results[0].plot(pil=False, conf=False, font_size=18)
        with _frame_lock:
            _raw_frame = frame.copy()
            _annotated_frame = annotated.copy()

threading.Thread(target=_capture_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
def _write_data_yaml():
    DATA_YAML.write_text(
        f"train: {IMAGES_DIR}\n"
        f"val: {IMAGES_DIR}\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )

def _retrain_worker():
    global _retrain_status, model, _inferencing
    _inferencing = False
    try:
        print("Retrain started")
        _write_data_yaml()
        train_model = YOLO(YOLO11_BASE)
        result = train_model.train(
            data=str(DATA_YAML),
            epochs=20,
            imgsz=640,
            device="cpu",
            verbose=True,
        )
        weights = Path(result.save_dir) / "weights" / "best.pt"
        if weights.exists():
            shutil.copy(weights, MODEL_PATH)
            model = YOLO(str(MODEL_PATH))
        _retrain_status = "done"
        print("Retrain complete — model hot-swapped")
    except Exception as e:
        print(f"Retrain failed: {e}")
        _retrain_status = "idle"
    finally:
        _inferencing = True
        try:
            _retrain_lock.release()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()

def _encode_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

def _mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _annotated_frame
        if frame is None:
            time.sleep(0.05)
            continue
        data = _encode_jpeg(frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(1 / 24)  # ~24 fps cap

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/frame")
def get_frame():
    global _pending_frame
    with _frame_lock:
        frame = _raw_frame
    if frame is None:
        raise HTTPException(503, "No frame available yet")
    _pending_frame = frame.copy()
    data = _encode_jpeg(frame)
    return StreamingResponse(iter([data]), media_type="image/jpeg")

@app.get("/classes")
def get_classes():
    return JSONResponse(classes)

# ---------------------------------------------------------------------------
# Review endpoints
# ---------------------------------------------------------------------------
@app.get("/review/annotations")
def list_annotations():
    result = []
    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        img_path = IMAGES_DIR / f"{label_path.stem}.jpg"
        if not img_path.exists():
            continue
        boxes = []
        text = label_path.read_text().strip()
        if text:
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    label = classes[cls_id] if cls_id < len(classes) else f"unknown_{cls_id}"
                    boxes.append({"x": float(parts[1]), "y": float(parts[2]),
                                  "w": float(parts[3]), "h": float(parts[4]), "label": label})
        result.append({"id": label_path.stem, "boxes": boxes})
    return result

@app.get("/annotations/{id}/image")
def get_annotation_image(id: str):
    img_path = IMAGES_DIR / f"{id}.jpg"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(img_path), media_type="image/jpeg")

class Box(BaseModel):
    x: float  # x_center, normalized 0-1
    y: float  # y_center, normalized 0-1
    w: float  # width, normalized 0-1
    h: float  # height, normalized 0-1
    label: str

class AnnotationPayload(BaseModel):
    boxes: list[Box]
    pending_id: str | None = None  # if set, annotate an already-saved frame

@app.put("/review/annotations/{id}")
def update_annotation(id: str, payload: AnnotationPayload):
    global classes
    label_path = LABELS_DIR / f"{id}.txt"
    if not label_path.exists():
        raise HTTPException(404, "Annotation not found")
    changed = False
    for box in payload.boxes:
        if box.label and box.label not in classes:
            classes.append(box.label)
            changed = True
    if changed:
        save_classes(classes)
    lines = []
    for box in payload.boxes:
        if not box.label:
            continue
        cls_id = classes.index(box.label)
        lines.append(f"{cls_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}")
    label_path.write_text("\n".join(lines))
    return {"ok": True, "classes": classes}

# ---------------------------------------------------------------------------
# Annotation submission
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Save frame for later annotation
# ---------------------------------------------------------------------------
@app.post("/save_frame")
def save_frame():
    frame = _pending_frame
    if frame is None:
        raise HTTPException(503, "No frame available yet")
    ts = str(time.time())
    img_path = IMAGES_DIR / f"{ts}.jpg"
    cv2.imwrite(str(img_path), frame)
    return {"id": ts}

@app.get("/pending")
def list_pending():
    result = []
    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        if not (LABELS_DIR / f"{img_path.stem}.txt").exists():
            result.append({"id": img_path.stem})
    return result

@app.delete("/pending/{id}")
def delete_pending(id: str):
    img_path = IMAGES_DIR / f"{id}.jpg"
    if not img_path.exists():
        raise HTTPException(404, "Frame not found")
    img_path.unlink()
    return {"ok": True}

# ---------------------------------------------------------------------------
# Annotation submission
# ---------------------------------------------------------------------------

@app.post("/annotate")
def annotate(payload: AnnotationPayload):
    global classes, _annotation_count

    if payload.pending_id:
        # Annotating a previously saved frame
        ts = payload.pending_id
        img_path = IMAGES_DIR / f"{ts}.jpg"
        if not img_path.exists():
            raise HTTPException(404, "Saved frame not found")
    else:
        # Use the frame that was shown to the user when they paused
        frame = _pending_frame
        if frame is None:
            raise HTTPException(503, "No frame available yet")
        ts = str(time.time())
        img_path = IMAGES_DIR / f"{ts}.jpg"
        cv2.imwrite(str(img_path), frame)

    label_path = LABELS_DIR / f"{ts}.txt"

    # Register any new labels
    changed = False
    for box in payload.boxes:
        if box.label and box.label not in classes:
            classes.append(box.label)
            changed = True
    if changed:
        save_classes(classes)

    # Write YOLO labels
    lines = []
    for box in payload.boxes:
        if not box.label:
            continue
        cls_id = classes.index(box.label)
        lines.append(f"{cls_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}")
    label_path.write_text("\n".join(lines))

    _annotation_count += 1

    return {"saved": ts, "classes": classes, "total_annotations": _annotation_count}

@app.get("/retrain/status")
def retrain_status():
    return {
        "status": _retrain_status,
        "total_annotations": _annotation_count,
    }

@app.post("/retrain")
def trigger_retrain():
    global _retrain_status, _new_since_retrain
    if _retrain_status == "running":
        raise HTTPException(409, "Retrain already running")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "Retrain already running")
    _new_since_retrain = 0
    _retrain_status = "running"
    threading.Thread(target=_retrain_worker, daemon=True).start()
    return {"status": "started"}

# ---------------------------------------------------------------------------
# Export annotations as zip (for Colab upload)
# ---------------------------------------------------------------------------
@app.get("/export")
def export_annotations():
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in IMAGES_DIR.glob("*.jpg"):
            zf.write(img, f"images/{img.name}")
        for lbl in LABELS_DIR.glob("*.txt"):
            zf.write(lbl, f"labels/{lbl.name}")
        if DATA_YAML.exists():
            zf.write(DATA_YAML, "data.yaml")
        zf.writestr("classes.json", json.dumps(classes, indent=2))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=annotations.zip"},
    )

# Serve static files (index.html, app.js, style.css)
app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
