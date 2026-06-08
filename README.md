# Fish ID

A web app for identifying and labeling fish species from a YouTube live stream. You pause the stream, draw bounding boxes around fish, label each species, and submit. Over time the labeled frames are used to retrain a YOLO model that runs inference on the live feed in real time.

---

## What it does

- Streams a live YouTube video through a local web server and runs YOLO object detection on every frame
- Lets you pause the stream, draw bounding boxes, and label species with a point-and-click UI
- Saves annotated frames for model retraining
- Lets you save frames without annotating them immediately ("Save for Later") and annotate them later from the Review page
- Provides a Review page for browsing, filtering, and correcting saved annotations
- Triggers an in-process YOLO retrain on demand; hot-swaps the model when training finishes

---

## Installation

**Requirements:** Python 3.11+, a GPU or Apple Silicon Mac (MPS) recommended for inference.

```bash
# 1. Clone the repo
git clone https://github.com/VictorNorman/fish-id.git
cd fish-id

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Provide a starting model
#    Place a YOLO model file named best.pt one directory above claude/:
#      fish-id/best.pt
#
#    If you don't have one yet, copy any YOLO11 checkpoint — the first
#    retrain will replace it with a model trained on your annotations.
#    You can download the base model with:
python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
cp yolo11m.pt ../best.pt   # use the base model as a placeholder
```

---

## Usage

### Start the server

```bash
cd fish-id/claude
source .venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in a browser.

### Live view

The home page streams the YouTube live feed with YOLO detection boxes overlaid. The header shows how many annotations have been collected and the status of any running retrain.

### Annotating a frame

1. Click **Pause & Annotate** to freeze the current frame.
2. Draw a bounding box by clicking and dragging over a fish.
3. Click the box in the list (or on the canvas) and select a species label. Add new species with **+ New species**.
4. Repeat for other fish in the frame.
5. Click **Submit Annotations** to save the frame and its labels.

If you want to annotate the frame later, click **Save for Later** (visible when no boxes have been drawn). The frame is saved without labels and can be annotated from the Review page.

### Reviewing annotations

Click **Review Annotations** in the header. The review page shows all saved annotated frames.

- **Filter bar** — click a species label to see only frames containing that species. **All** shows everything.
- **Saved (N)** — shows frames that have been saved but not yet annotated. Click **Annotate this Frame** to open it in the annotation UI.
- **Slider** — drag to jump quickly through many frames.
- Click a bounding box on the canvas or in the box list to select it, then pick a new label to correct a mislabeled box.
- Click **Save Changes** to write corrections to disk.

### Retraining

Click **Retrain Now** in the header. The server trains a fresh YOLO11m model from scratch on all saved annotations, then hot-swaps it into the running inference loop. Training runs on CPU to avoid conflicting with MPS inference; progress is printed to the server console. The header shows "Retraining…" while it runs and "Model updated" when it finishes.

---

## How it works

### Server (`server.py`)

Built with **FastAPI**. Three background concerns run concurrently:

**Stream capture loop** — a daemon thread resolves the YouTube URL to a direct HLS stream URL via `yt-dlp` (forcing the live edge with `live_from_start=False`), then reads frames with `CamGear`. If the stream dies (expired HLS URL or network drop), it automatically re-resolves and reconnects. Each frame is run through the YOLO model and the annotated result is kept in memory for the MJPEG feed.

**MJPEG feed** — `/video_feed` streams annotated frames to the browser as a multipart JPEG sequence (~24 fps).

**Retrain worker** — `/retrain` spawns a thread that writes a `data.yaml` pointing at the saved images and labels, trains a new `YOLO11m` model for 20 epochs, then copies the best checkpoint to `best.pt` and reloads it in the inference loop.

### Annotation storage

Images are saved as JPEGs under `annotations/images/`. Labels are saved in YOLO format (normalized center x, y, width, height per line) under `annotations/labels/`. A frame with a saved image but no label file is treated as "pending" (saved for later).

When you pause the stream, the server caches that exact frame. When you submit, the server writes the label file using the cached frame — not whatever the live stream has advanced to by then.

### Front end

Three pages served as static files:

- **`index.html` / `app.js`** — live view and annotation UI. Drawing uses an HTML5 canvas overlaid on the frozen frame; mouse coordinates are scaled from CSS pixels to natural image coordinates so boxes are stored accurately regardless of window size.
- **`review.html` / `review.js`** — annotation browser. Filtering and navigation are handled client-side; edits are written back to the server with a `PUT` request.
- **`style.css`** — shared styles.

### Model

YOLO11m from Ultralytics. Inference runs on MPS (Apple Silicon GPU). Retraining runs on CPU to avoid memory conflicts. The model is hot-swapped after retraining with no server restart.
