const COLORS = [
  "#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
  "#a855f7", "#06b6d4", "#f97316", "#ec4899",
];

let annotations = [];
let filteredAnnotations = [];
let activeFilter = "";
let currentIdx = 0;
let selectedBox = -1;
let knownClasses = [];
let dirty = false;

let pendingFrames = [];
let pendingMode = false;
let pendingIdx = 0;

let img, canvas, ctx, slider;

window.addEventListener("DOMContentLoaded", async () => {
  img = document.getElementById("review-img");
  canvas = document.getElementById("review-canvas");
  ctx = canvas.getContext("2d");
  slider = document.getElementById("nav-slider");

  slider.addEventListener("input", () => {
    const idx = parseInt(slider.value, 10) - 1;
    if (pendingMode) {
      if (idx !== pendingIdx) showPending(idx);
    } else if (idx !== currentIdx) {
      if (dirty && !confirm("You have unsaved changes. Leave anyway?")) {
        slider.value = currentIdx + 1;
        return;
      }
      show(idx);
    }
  });

  canvas.addEventListener("click", onCanvasClick);
  img.addEventListener("load", () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    redraw();
  });

  const [anns, cls, pending] = await Promise.all([
    fetch("/review/annotations").then(r => r.json()),
    fetch("/classes").then(r => r.json()),
    fetch("/pending").then(r => r.json()),
  ]);
  annotations = anns;
  knownClasses = cls;
  pendingFrames = pending;

  if (annotations.length === 0 && pendingFrames.length === 0) {
    document.getElementById("empty-state").classList.remove("hidden");
  } else {
    document.getElementById("review-ui").classList.remove("hidden");
    applyFilter("");
  }
});

function applyFilter(label) {
  pendingMode = false;
  activeFilter = label;
  filteredAnnotations = label === ""
    ? annotations
    : annotations.filter(a => a.boxes.some(b => b.label === label));
  document.getElementById("pending-actions").classList.add("hidden");
  renderFilterBar();
  if (filteredAnnotations.length === 0) {
    document.getElementById("empty-state").classList.remove("hidden");
    document.getElementById("review-ui").style.visibility = "hidden";
  } else {
    document.getElementById("empty-state").classList.add("hidden");
    document.getElementById("review-ui").style.visibility = "";
    slider.max = filteredAnnotations.length;
    show(0);
  }
}

function renderFilterBar() {
  const bar = document.getElementById("filter-bar");
  const labels = ["", ...knownClasses];
  const annotationBtns = labels.map(l => {
    const active = (!pendingMode && l === activeFilter) ? "active" : "";
    const text = l === "" ? "All" : l;
    const count = l === ""
      ? annotations.length
      : annotations.filter(a => a.boxes.some(b => b.label === l)).length;
    return `<button class="filter-btn ${active}" onclick="applyFilter('${l.replace(/'/g, "\\'")}')">${text} <span class="filter-count">${count}</span></button>`;
  }).join("");
  const pendingBtn = `<button class="filter-btn pending-btn ${pendingMode ? "active" : ""}" onclick="enterPendingMode()">Saved <span class="filter-count">${pendingFrames.length}</span></button>`;
  bar.innerHTML = annotationBtns + "<span class='filter-sep'>|</span>" + pendingBtn;
}

function enterPendingMode() {
  if (dirty && !confirm("You have unsaved changes. Leave anyway?")) return;
  pendingMode = true;
  pendingIdx = 0;
  renderFilterBar();
  showPending(0);
}

function showPending(idx) {
  pendingIdx = idx;
  const p = pendingFrames[pendingIdx];
  img.src = `/annotations/${p.id}/image`;
  canvas.width = 0;
  canvas.height = 0;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("nav-count").textContent =
    `${pendingIdx + 1} of ${pendingFrames.length} saved`;
  slider.max = pendingFrames.length;
  slider.value = pendingIdx + 1;
  document.getElementById("box-list").innerHTML = "";
  document.getElementById("label-picker").classList.add("hidden");
  document.getElementById("save-status").textContent = "";
  document.getElementById("pending-actions").classList.remove("hidden");
  document.getElementById("annotate-link").href = `/?pending=${p.id}`;
}

function navigatePending(dir) {
  const next = pendingIdx + dir;
  if (next < 0 || next >= pendingFrames.length) return;
  showPending(next);
}

async function deletePending() {
  const p = pendingFrames[pendingIdx];
  if (!confirm("Delete this saved frame?")) return;
  await fetch(`/pending/${p.id}`, { method: "DELETE" });
  pendingFrames.splice(pendingIdx, 1);
  renderFilterBar();
  if (pendingFrames.length === 0) {
    pendingMode = false;
    applyFilter("");
  } else {
    showPending(Math.min(pendingIdx, pendingFrames.length - 1));
  }
}

function show(idx) {
  currentIdx = idx;
  selectedBox = -1;
  dirty = false;
  setSaveStatus("");

  const ann = filteredAnnotations[currentIdx];
  img.src = `/annotations/${ann.id}/image`;
  document.getElementById("nav-count").textContent =
    `${currentIdx + 1} of ${filteredAnnotations.length}`;
  slider.value = currentIdx + 1;
  renderAll();
}

function navigate(dir) {
  if (pendingMode) { navigatePending(dir); return; }
  if (dirty && !confirm("You have unsaved changes. Leave anyway?")) return;
  const next = currentIdx + dir;
  if (next < 0 || next >= filteredAnnotations.length) return;
  show(next);
}

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------
function boxToPixels(b) {
  const W = canvas.width, H = canvas.height;
  return {
    x: (b.x - b.w / 2) * W,
    y: (b.y - b.h / 2) * H,
    w: b.w * W,
    h: b.h * H,
  };
}

function onCanvasClick(e) {
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const cy = (e.clientY - rect.top) * (canvas.height / rect.height);

  const boxes = filteredAnnotations[currentIdx].boxes;
  for (let i = boxes.length - 1; i >= 0; i--) {
    const p = boxToPixels(boxes[i]);
    if (cx >= p.x && cx <= p.x + p.w && cy >= p.y && cy <= p.y + p.h) {
      selectedBox = i;
      renderAll();
      return;
    }
  }
  selectedBox = -1;
  renderAll();
}

function redraw() {
  if (!canvas.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const boxes = filteredAnnotations[currentIdx]?.boxes ?? [];
  boxes.forEach((b, i) => {
    const p = boxToPixels(b);
    const color = COLORS[i % COLORS.length];
    const selected = i === selectedBox;

    if (selected) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 4;
      ctx.strokeRect(p.x - 1, p.y - 1, p.w + 2, p.h + 2);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 2.5 : 1.5;
    ctx.strokeRect(p.x, p.y, p.w, p.h);

    const label = b.label || "?";
    ctx.font = "bold 13px system-ui";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(p.x, p.y - 22, tw + 10, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, p.x + 5, p.y - 6);
  });
}

// ---------------------------------------------------------------------------
// Box list + label picker
// ---------------------------------------------------------------------------
function renderBoxList() {
  const ul = document.getElementById("box-list");
  const boxes = filteredAnnotations[currentIdx]?.boxes ?? [];
  ul.innerHTML = "";
  boxes.forEach((b, i) => {
    const color = COLORS[i % COLORS.length];
    const li = document.createElement("li");
    li.className = i === selectedBox ? "selected" : "";
    li.innerHTML = `
      <span class="box-swatch" style="background:${color}"></span>
      <span class="box-label ${b.label ? "" : "unlabeled"}" onclick="selectBox(${i})">
        ${b.label || "unlabeled"}
      </span>
    `;
    ul.appendChild(li);
  });
}

function renderLabelGrid() {
  const picker = document.getElementById("label-picker");
  const grid = document.getElementById("label-grid");
  if (selectedBox < 0) { picker.classList.add("hidden"); return; }
  picker.classList.remove("hidden");

  const currentLabel = filteredAnnotations[currentIdx].boxes[selectedBox]?.label;
  grid.innerHTML = knownClasses.map(name => `
    <button class="label-btn ${name === currentLabel ? "active" : ""}"
            onclick="assignLabel('${name.replace(/'/g, "\\'")}')">
      ${name}
    </button>
  `).join("");
}

function renderAll() {
  renderBoxList();
  renderLabelGrid();
  redraw();
}

function selectBox(i) {
  selectedBox = i;
  renderAll();
}

function assignLabel(name) {
  if (selectedBox < 0) return;
  filteredAnnotations[currentIdx].boxes[selectedBox].label = name;
  dirty = true;
  setSaveStatus("Unsaved changes", "#f59e0b");
  renderAll();
}

function addNewSpecies() {
  const name = prompt("New species name:")?.trim();
  if (!name) return;
  if (!knownClasses.includes(name)) knownClasses.push(name);
  assignLabel(name);
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------
async function saveChanges() {
  const ann = filteredAnnotations[currentIdx];
  try {
    const res = await fetch(`/review/annotations/${ann.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: ann.boxes }),
    });
    const data = await res.json();
    knownClasses = data.classes;
    dirty = false;
    setSaveStatus("Saved!", "#22c55e");
    setTimeout(() => setSaveStatus(""), 3000);
  } catch {
    setSaveStatus("Save failed.", "#ef4444");
  }
}

function setSaveStatus(msg, color = "#94a3b8") {
  const el = document.getElementById("save-status");
  el.textContent = msg;
  el.style.color = color;
}
