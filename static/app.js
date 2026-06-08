// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const COLORS = [
  "#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
  "#a855f7", "#06b6d4", "#f97316", "#ec4899",
];

let boxes = [];           // [{x, y, w, h, label}] — pixel coords on canvas
let selectedIdx = -1;     // which box is awaiting a label
let drawing = false;
let startX = 0, startY = 0;
let currentRect = null;
let knownClasses = [];

let canvas, ctx, frozenFrame, stream, annotationLayer, annotationPanel;
let pendingId = null;  // set when annotating a previously-saved frame

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  canvas = document.getElementById("canvas");
  ctx = canvas.getContext("2d");
  frozenFrame = document.getElementById("frozen-frame");
  stream = document.getElementById("stream");
  annotationLayer = document.getElementById("annotation-layer");
  annotationPanel = document.getElementById("annotation-panel");

  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("mouseleave", onMouseUp);

  loadClasses();
  pollStatus();

  // If opened with ?pending=ID, load that saved frame for annotation
  const params = new URLSearchParams(location.search);
  const pid = params.get("pending");
  if (pid) {
    pendingId = pid;
    frozenFrame.src = `/annotations/${pid}/image`;
    frozenFrame.onload = () => {
      canvas.width = frozenFrame.naturalWidth;
      canvas.height = frozenFrame.naturalHeight;
      canvas.style.width = "100%";
      canvas.style.height = "auto";
    };
    stream.style.display = "none";
    stream.src = "";
    annotationLayer.classList.remove("hidden");
    annotationPanel.classList.remove("hidden");
    document.getElementById("btn-pause").classList.add("hidden");
    document.getElementById("btn-resume").classList.remove("hidden");
    setStatus(`Annotating saved frame — draw boxes then submit.`, "#60a5fa");
    renderAll();
  }
});

// ---------------------------------------------------------------------------
// Pause / resume
// ---------------------------------------------------------------------------
function pause() {
  pendingId = null;
  frozenFrame.src = `/frame?t=${Date.now()}`;
  frozenFrame.onload = () => {
    canvas.width = frozenFrame.naturalWidth;
    canvas.height = frozenFrame.naturalHeight;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
  };
  stream.style.display = "none";
  stream.src = "";
  annotationLayer.classList.remove("hidden");
  annotationPanel.classList.remove("hidden");
  document.getElementById("btn-pause").classList.add("hidden");
  document.getElementById("btn-resume").classList.remove("hidden");
  boxes = [];
  selectedIdx = -1;
  renderAll();
}

function resume() {
  stream.style.display = "block";
  stream.src = "/video_feed";
  annotationLayer.classList.add("hidden");
  annotationPanel.classList.add("hidden");
  document.getElementById("btn-pause").classList.remove("hidden");
  document.getElementById("btn-resume").classList.add("hidden");
  boxes = [];
  selectedIdx = -1;
  redraw();
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------
function canvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function onMouseDown(e) {
  const p = canvasCoords(e);
  drawing = true;
  startX = p.x;
  startY = p.y;
  currentRect = null;
}

function onMouseMove(e) {
  if (!drawing) return;
  const p = canvasCoords(e);
  currentRect = {
    x: Math.min(startX, p.x),
    y: Math.min(startY, p.y),
    w: Math.abs(p.x - startX),
    h: Math.abs(p.y - startY),
  };
  redraw();
}

function onMouseUp(e) {
  if (!drawing) return;
  drawing = false;
  if (currentRect && currentRect.w > 5 && currentRect.h > 5) {
    boxes.push({ ...currentRect, label: "" });
    selectedIdx = boxes.length - 1;
    renderAll();
  }
  currentRect = null;
  redraw();
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  boxes.forEach((b, i) => {
    const isSelected = i === selectedIdx;
    const color = COLORS[i % COLORS.length];

    // Highlight selected box
    if (isSelected) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 4;
      ctx.strokeRect(b.x - 1, b.y - 1, b.w + 2, b.h + 2);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 2.5 : 1.5;
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    // Label badge
    const label = b.label || "?";
    ctx.font = "bold 13px system-ui";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(b.x, b.y - 22, tw + 10, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, b.x + 5, b.y - 6);
  });

  // In-progress rect
  if (currentRect) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
    ctx.setLineDash([]);
  }
}

// ---------------------------------------------------------------------------
// Label grid
// ---------------------------------------------------------------------------
function renderLabelGrid() {
  const grid = document.getElementById("label-grid");
  const picker = document.getElementById("label-picker");

  if (selectedIdx < 0) {
    picker.classList.add("hidden");
    return;
  }

  picker.classList.remove("hidden");
  const currentLabel = boxes[selectedIdx]?.label;

  grid.innerHTML = knownClasses.map(name => `
    <button class="label-btn ${name === currentLabel ? "active" : ""}"
            onclick="assignLabel('${name.replace(/'/g, "\\'")}')">
      ${name}
    </button>
  `).join("");
}

function assignLabel(name) {
  if (selectedIdx < 0) return;
  boxes[selectedIdx].label = name;

  // Advance to next unlabeled box, or deselect if all labeled
  const nextUnlabeled = boxes.findIndex((b, i) => i > selectedIdx && !b.label);
  if (nextUnlabeled >= 0) {
    selectedIdx = nextUnlabeled;
  } else {
    const anyUnlabeled = boxes.findIndex(b => !b.label);
    selectedIdx = anyUnlabeled;  // -1 if all done
  }

  renderAll();
}

// ---------------------------------------------------------------------------
// Box list
// ---------------------------------------------------------------------------
function renderBoxList() {
  const ul = document.getElementById("box-list");
  ul.innerHTML = "";
  const W = canvas.width, H = canvas.height;
  boxes.forEach((b, i) => {
    const color = COLORS[i % COLORS.length];
    const li = document.createElement("li");
    li.className = i === selectedIdx ? "selected" : "";
    const cx = ((b.x + b.w / 2) / W).toFixed(3);
    const cy = ((b.y + b.h / 2) / H).toFixed(3);
    const nw = (b.w / W).toFixed(3);
    const nh = (b.h / H).toFixed(3);
    li.innerHTML = `
      <span class="box-swatch" style="background:${color}"></span>
      <span class="box-label ${b.label ? "" : "unlabeled"}"
            onclick="selectBox(${i})">
        ${b.label || "unlabeled — click to label"}
        <span class="box-coords">cx=${cx} cy=${cy} w=${nw} h=${nh}</span>
      </span>
      <button class="del" onclick="removeBox(${i})" title="Remove">✕</button>
    `;
    ul.appendChild(li);
  });
}

function selectBox(i) {
  selectedIdx = i;
  renderAll();
}

function removeBox(i) {
  boxes.splice(i, 1);
  if (selectedIdx >= boxes.length) selectedIdx = boxes.length - 1;
  renderAll();
}

function renderAll() {
  renderLabelGrid();
  renderBoxList();
  redraw();
  const btn = document.getElementById("btn-save-later");
  if (btn) btn.style.display = (boxes.length === 0 && !pendingId) ? "" : "none";
}

// ---------------------------------------------------------------------------
// Add new species
// ---------------------------------------------------------------------------
function addNewSpecies() {
  const name = prompt("New species name:")?.trim();
  if (!name) return;
  if (!knownClasses.includes(name)) {
    knownClasses.push(name);
    renderLabelGrid();
  }
  assignLabel(name);
}

// ---------------------------------------------------------------------------
// Load classes from server
// ---------------------------------------------------------------------------
async function loadClasses() {
  const res = await fetch("/classes");
  knownClasses = await res.json();
  renderLabelGrid();
}


// ---------------------------------------------------------------------------
// Save for later
// ---------------------------------------------------------------------------
async function saveForLater() {
  try {
    await fetch("/save_frame", { method: "POST" });
    const flash = document.getElementById("save-flash");
    flash.textContent = "Frame saved for later annotation.";
    setTimeout(() => { flash.textContent = ""; }, 2000);
    resume();
  } catch {
    const flash = document.getElementById("save-flash");
    flash.style.color = "#ef4444";
    flash.textContent = "Error saving frame.";
    setTimeout(() => { flash.textContent = ""; flash.style.color = "#22c55e"; }, 2000);
  }
}

// ---------------------------------------------------------------------------
// Submit annotations
// ---------------------------------------------------------------------------
async function submitAnnotations() {
  const W = canvas.width;
  const H = canvas.height;

  const payload = {
    boxes: boxes
      .filter(b => b.label)
      .map(b => ({
        x: (b.x + b.w / 2) / W,
        y: (b.y + b.h / 2) / H,
        w: b.w / W,
        h: b.h / H,
        label: b.label,
      })),
    ...(pendingId ? { pending_id: pendingId } : {}),
  };

  if (payload.boxes.length === 0) {
    setStatus("No labeled boxes to submit.", "orange");
    return;
  }

  try {
    const res = await fetch("/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setStatus(`Saved! ${data.total_annotations} annotations total.`, "#22c55e");
    knownClasses = data.classes;
    boxes = [];
    selectedIdx = -1;
    if (pendingId) {
      pendingId = null;
      setTimeout(() => { location.href = "/review.html"; }, 800);
    } else {
      renderAll();
      pollStatus();
    }
  } catch {
    setStatus("Error submitting. Try again.", "#ef4444");
  }
}

function setStatus(msg, color) {
  const el = document.getElementById("submit-status");
  el.textContent = msg;
  el.style.color = color;
}

// ---------------------------------------------------------------------------
// Retrain
// ---------------------------------------------------------------------------
async function triggerRetrain() {
  const res = await fetch("/retrain", { method: "POST" });
  if (res.ok) pollStatus();
  else {
    const d = await res.json();
    alert(d.detail || "Could not start retrain.");
  }
}

function pollStatus() {
  fetch("/retrain/status")
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById("retrain-status");
      if (data.status === "running") {
        el.textContent = "⏳ Retraining…";
        setTimeout(pollStatus, 3000);
      } else if (data.status === "done") {
        el.textContent = "✅ Model updated";
        setTimeout(() => { el.textContent = statusLine(data); }, 4000);
      } else {
        el.textContent = statusLine(data);
      }
    })
    .catch(() => {});
}

function statusLine(data) {
  return `${data.total_annotations} annotations`;
}
