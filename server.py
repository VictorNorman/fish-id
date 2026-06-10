import json
import os
import shutil
import threading
import time
from pathlib import Path

# nvm installs node outside the default PATH; find and add it so yt-dlp
# can use it to solve YouTube's JS challenge.
_nvm_node_dir = Path.home() / ".nvm" / "versions" / "node"
if _nvm_node_dir.exists():
    _versions = sorted(_nvm_node_dir.iterdir(), reverse=True)
    if _versions:
        os.environ["PATH"] = str(_versions[0] / "bin") + ":" + os.environ.get("PATH", "")

import cv2
import numpy as np
import torch
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
MODEL_PATH = BASE / "fish_id.pt"
YOLO11_BASE = "yolo11m.pt"   # downloaded by ultralytics on first retrain
CLASSES_FILE = BASE / "classes.json"
ANNOTATIONS_DIR = BASE / "annotations"
IMAGES_DIR = ANNOTATIONS_DIR / "images"
LABELS_DIR = ANNOTATIONS_DIR / "labels"
DATA_YAML = BASE / "data.yaml"
LAST_RETRAIN_FILE = BASE / "last_retrain.json"
STREAM_URL = "https://youtu.be/7i8ARjIeM2k"

# ---------------------------------------------------------------------------
# Device selection (auto-detected)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    INFERENCE_DEVICE = "cuda"
    TRAIN_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    INFERENCE_DEVICE = "mps"
    TRAIN_DEVICE = "cpu"   # avoid MPS memory conflicts during training
else:
    INFERENCE_DEVICE = "cpu"
    TRAIN_DEVICE = "cpu"

print(f"Devices — inference: {INFERENCE_DEVICE}, training: {TRAIN_DEVICE}")

# Cookie auth: set one of these env vars (COOKIES_FILE takes precedence).
# COOKIES_FILE=/path/to/cookies.txt  — Netscape-format file exported via yt-dlp
# COOKIES_BROWSER=chrome             — read live from a browser (firefox, chrome, etc.)
# Set COOKIES_BROWSER= to disable cookie auth entirely.
COOKIES_FILE = os.environ.get("COOKIES_FILE", "")
COOKIES_BROWSER = os.environ.get("COOKIES_BROWSER", "firefox")

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

def _load_last_retrain_time() -> float:
    if LAST_RETRAIN_FILE.exists():
        return json.loads(LAST_RETRAIN_FILE.read_text()).get("time", 0.0)
    return 0.0

def _save_last_retrain_time(annotations_in_model: int):
    LAST_RETRAIN_FILE.write_text(json.dumps({
        "time": time.time(),
        "annotations_in_model": annotations_in_model,
    }))

def _load_annotations_in_model() -> int:
    if LAST_RETRAIN_FILE.exists():
        return json.loads(LAST_RETRAIN_FILE.read_text()).get("annotations_in_model", 0)
    return 0

def _get_new_annotation_ids() -> list[str]:
    last_time = _load_last_retrain_time()
    result = []
    for lbl_path in LABELS_DIR.glob("*.txt"):
        try:
            if float(lbl_path.stem) > last_time:
                result.append(lbl_path.stem)
        except ValueError:
            pass
    return result

# ---------------------------------------------------------------------------
# Global state (protected by locks where needed)
# ---------------------------------------------------------------------------
classes: list[str] = load_classes()
model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else YOLO11_BASE)

_frame_lock = threading.Lock()
_raw_frame: np.ndarray | None = None
_annotated_frame: np.ndarray | None = None
_pending_frame: np.ndarray | None = None  # frame shown to user for annotation

_retrain_lock = threading.Lock()
_retrain_status = "idle"   # "idle" | "running" | "done"
_retrain_type = "full"     # "full" | "quick"
_annotation_count = len(list(LABELS_DIR.glob("*.txt")))
_inferencing = True  # set to False during retraining to free MPS

# ---------------------------------------------------------------------------
# Background frame-capture thread
# ---------------------------------------------------------------------------
def _capture_loop():
    global _raw_frame, _annotated_frame
    stream = None
    null_streak = 0
    NULL_RECONNECT = 50  # reconnect after this many consecutive None frames

    while True:
        if stream is None:
            try:
                print("Connecting to stream...")
                # Resolve the node executable path (installed via nvm)
                _nvm_node_dir = Path.home() / ".nvm" / "versions" / "node"
                _node_bin = "node"
                if _nvm_node_dir.exists():
                    _node_bin = next(
                        (str(p / "bin" / "node") for p in sorted(_nvm_node_dir.iterdir(), reverse=True)
                         if (p / "bin" / "node").exists()),
                        "node",
                    )
                _stream_params: dict = {
                    "live_from_start": False,
                    "js_runtimes": {"node": {"path": _node_bin}},
                }
                if COOKIES_FILE:
                    _stream_params["cookiefile"] = COOKIES_FILE
                elif COOKIES_BROWSER:
                    _stream_params["cookiesfrombrowser"] = (COOKIES_BROWSER,)
                stream = CamGear(  # type: ignore[call-arg]
                    source=STREAM_URL,
                    stream_mode=True,
                    STREAM_PARAMS=_stream_params,
                ).start()
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
        results = model.predict(frame, conf=0.6, device=INFERENCE_DEVICE, verbose=False)
        annotated = results[0].plot(pil=False, conf=False, font_size=18, line_width=1)
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
    _inferencing = INFERENCE_DEVICE == "mps"
    try:
        print("Full retrain started")
        _write_data_yaml()
        train_model = YOLO(YOLO11_BASE)
        result = train_model.train(
            data=str(DATA_YAML),
            epochs=20,
            imgsz=640,
            device=TRAIN_DEVICE,
            verbose=True,
        )
        weights = Path(result.save_dir) / "weights" / "best.pt"
        if weights.exists():
            shutil.copy(weights, MODEL_PATH)
            model = YOLO(str(MODEL_PATH))
        _save_last_retrain_time(len(list(LABELS_DIR.glob("*.txt"))))
        _retrain_status = "done"
        print("Full retrain complete — model hot-swapped")
    except Exception as e:
        print(f"Retrain failed: {e}")
        _retrain_status = "idle"
    finally:
        _inferencing = True
        try:
            _retrain_lock.release()
        except RuntimeError:
            pass


def _quick_retrain_worker(new_ids: list[str]):
    global _retrain_status, model, _inferencing
    _inferencing = INFERENCE_DEVICE == "mps"
    try:
        import tempfile
        print(f"Quick retrain started on {len(new_ids)} new annotations")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "images").mkdir()
            (tmp / "labels").mkdir()
            for stem in new_ids:
                img_src = IMAGES_DIR / f"{stem}.jpg"
                lbl_src = LABELS_DIR / f"{stem}.txt"
                if img_src.exists() and lbl_src.exists():
                    shutil.copy(img_src, tmp / "images" / img_src.name)
                    shutil.copy(lbl_src, tmp / "labels" / lbl_src.name)
            tmp_yaml = tmp / "data.yaml"
            tmp_yaml.write_text(
                f"train: {tmp / 'images'}\n"
                f"val: {tmp / 'images'}\n"
                f"nc: {len(classes)}\n"
                f"names: {classes}\n"
            )
            base = str(MODEL_PATH) if MODEL_PATH.exists() else YOLO11_BASE
            train_model = YOLO(base)
            result = train_model.train(
                data=str(tmp_yaml),
                epochs=5,
                imgsz=640,
                device=TRAIN_DEVICE,
                verbose=True,
            )
            weights = Path(result.save_dir) / "weights" / "best.pt"
            if weights.exists():
                shutil.copy(weights, MODEL_PATH)
                model = YOLO(str(MODEL_PATH))
        _save_last_retrain_time(_load_annotations_in_model() + len(new_ids))
        _retrain_status = "done"
        print("Quick retrain complete — model hot-swapped")
    except Exception as e:
        print(f"Quick retrain failed: {e}")
        _retrain_status = "idle"
    finally:
        _inferencing = True
        try:
            _retrain_lock.release()
        except RuntimeError:
            pass


def _nightly_retrain_scheduler():
    while True:
        now = time.localtime()
        seconds_until_2am = ((2 - now.tm_hour) * 3600 - now.tm_min * 60 - now.tm_sec) % 86400
        if seconds_until_2am == 0:
            seconds_until_2am = 86400
        time.sleep(seconds_until_2am)
        if _annotation_count > 0 and _retrain_lock.acquire(blocking=False):
            global _retrain_status, _retrain_type
            _retrain_type = "full"
            _retrain_status = "running"
            print("Nightly full retrain triggered")
            threading.Thread(target=_retrain_worker, daemon=True).start()

threading.Thread(target=_nightly_retrain_scheduler, daemon=True).start()


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
    new_count = len(_get_new_annotation_ids())
    return {
        "status": _retrain_status,
        "type": _retrain_type,
        "total_annotations": _annotation_count,
        "annotations_in_model": _load_annotations_in_model(),
        "new_annotations": new_count,
    }

@app.post("/retrain")
def trigger_retrain():
    global _retrain_status, _retrain_type
    if _retrain_status == "running":
        raise HTTPException(409, "Retrain already running")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "Retrain already running")
    _retrain_type = "full"
    _retrain_status = "running"
    threading.Thread(target=_retrain_worker, daemon=True).start()
    return {"status": "started"}

@app.post("/retrain/quick")
def trigger_quick_retrain():
    global _retrain_status, _retrain_type
    if _retrain_status == "running":
        raise HTTPException(409, "Retrain already running")
    new_ids = _get_new_annotation_ids()
    if not new_ids:
        raise HTTPException(400, "No new annotations since last retrain")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "Retrain already running")
    _retrain_type = "quick"
    _retrain_status = "running"
    threading.Thread(target=_quick_retrain_worker, args=(new_ids,), daemon=True).start()
    return {"status": "started", "new_count": len(new_ids)}

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

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fish ID utilities")
    parser.add_argument("--retrain", action="store_true",
                        help="Run a full retrain from scratch and exit")
    args = parser.parse_args()

    if args.retrain:
        count = len(list(LABELS_DIR.glob("*.txt")))
        if count == 0:
            print("No annotations found — nothing to train on.")
            raise SystemExit(1)
        print(f"Starting full retrain on {count} annotations...")
        _write_data_yaml()
        train_model = YOLO(YOLO11_BASE)
        result = train_model.train(
            data=str(DATA_YAML),
            epochs=20,
            imgsz=640,
            device=TRAIN_DEVICE,
            verbose=True,
        )
        weights = Path(result.save_dir) / "weights" / "best.pt"
        if weights.exists():
            shutil.copy(weights, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        _save_last_retrain_time(len(list(LABELS_DIR.glob("*.txt"))))
        print("Retrain complete.")
    else:
        parser.print_help()
