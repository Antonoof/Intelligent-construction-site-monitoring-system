import { slotColor } from "../core/dom.js";

const HANDLE = 8;
const MIN_SIZE = 0.004;
const EDGE_GRAB = 6;
const MAX_ZOOM = 16;
const ANIM_MS = 180;
const HISTORY_LIMIT = 60;

/** Handle order matters: corners win over edges when they overlap. */
const HANDLES = [
  ["nw", 0, 0, "nwse-resize"],
  ["ne", 1, 0, "nesw-resize"],
  ["se", 1, 1, "nwse-resize"],
  ["sw", 0, 1, "nesw-resize"],
  ["n", 0.5, 0, "ns-resize"],
  ["s", 0.5, 1, "ns-resize"],
  ["w", 0, 0.5, "ew-resize"],
  ["e", 1, 0.5, "ew-resize"],
];

const easeOut = (t) => 1 - (1 - t) ** 3;

/**
 * CVAT-style canvas editor for YOLO boxes.
 *
 * Left drag moves the picture, boxes are grabbed and resized directly, and the
 * view animates instead of jumping. Ground truth is solid, model predictions are
 * dashed — double click promotes a prediction into the ground truth.
 */
export class BoxEditor {
  constructor(canvas, { onChange = null, onSelect = null, readonly = false, onMode = null, onHistory = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onChange = onChange;
    this.onSelect = onSelect;
    this.onMode = onMode;
    this.readonly = readonly;

    this.image = null;
    this.boxes = [];
    this.predictions = [];
    this.selected = -1;
    this.hovered = -1;
    this.activeClass = 0;
    this.showPredictions = true;
    this.showGroundTruth = true;
    this.confMin = 0.25;
    this.mode = "cursor"; // cursor | draw

    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.drag = null;
    this.anim = null;
    this.frame = null;

    this.past = [];
    this.future = [];
    this.onHistory = onHistory;

    this._bind();
    this._resize();
    this._observer = new ResizeObserver(() => this._resize());
    this._observer.observe(canvas);
  }

  /* ------------------------------------------------------------- loading */
  async load({ imageUrl, gt = [], pred = [] }) {
    this.boxes = gt.map((box) => ({ ...box }));
    this.predictions = pred.map((box) => ({ ...box }));
    this.past.length = 0;
    this.future.length = 0;
    if (this.onHistory) this.onHistory(false, false);
    this.selected = -1;
    this.hovered = -1;
    this.anim = null;
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };

    this.image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Не удалось загрузить изображение"));
      img.src = imageUrl;
    });

    this.invalidate();
    return this.image;
  }

  getBoxes() {
    return this.boxes.map(({ cls, x, y, w, h }) => ({ cls, x, y, w, h }));
  }

  /* ------------------------------------------------------------- history */
  /** Snapshot taken before a change, so undo restores what was there. */
  _remember() {
    this.past.push(this.getBoxes());
    if (this.past.length > HISTORY_LIMIT) this.past.shift();
    this.future.length = 0;
    if (this.onHistory) this.onHistory(this.canUndo, this.canRedo);
  }

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }

  undo() {
    if (!this.past.length) return;
    this.future.push(this.getBoxes());
    this.boxes = this.past.pop();
    this.selected = -1;
    this._changed();
    if (this.onHistory) this.onHistory(this.canUndo, this.canRedo);
    this.invalidate();
  }

  redo() {
    if (!this.future.length) return;
    this.past.push(this.getBoxes());
    this.boxes = this.future.pop();
    this.selected = -1;
    this._changed();
    if (this.onHistory) this.onHistory(this.canUndo, this.canRedo);
    this.invalidate();
  }

  setMode(mode) {
    this.mode = mode;
    this._cursor();
    if (this.onMode) this.onMode(mode);
    this.invalidate();
  }

  setClass(cls) {
    this.activeClass = cls;
    if (this.selected >= 0) {
      this._remember();
      this.boxes[this.selected].cls = cls;
      this._changed();
    }
    this.invalidate();
  }

  deleteSelected() {
    if (this.selected < 0) return;
    this._remember();
    this.boxes.splice(this.selected, 1);
    this.selected = -1;
    this._changed();
    this.invalidate();
  }

  acceptAllPredictions() {
    this._remember();
    this.predictions
      .filter((box) => box.conf >= this.confMin)
      .forEach((box) => this.boxes.push({ cls: box.cls, x: box.x, y: box.y, w: box.w, h: box.h }));
    this._changed();
    this.invalidate();
  }

  clearBoxes() {
    this._remember();
    this.boxes = [];
    this.selected = -1;
    this._changed();
    this.invalidate();
  }

  /** Заменяет разметку целиком — перенос боксов с соседнего кадра. */
  setBoxes(boxes, { append = false } = {}) {
    this._remember();
    const incoming = boxes.map(({ cls, x, y, w, h }) => ({ cls, x, y, w, h }));
    this.boxes = append ? this.boxes.concat(incoming) : incoming;
    this.selected = -1;
    this._changed();
    this.invalidate();
  }

  duplicateSelected(offset = 0.01) {
    if (this.selected < 0) return;
    this._remember();
    const box = this.boxes[this.selected];
    this.boxes.push({ ...box, x: box.x + offset, y: box.y + offset });
    this.selected = this.boxes.length - 1;
    this._changed();
    this.invalidate();
  }

  /**
   * Клавиатурная доводка выделенного бокса.
   *
   * Мышью попасть в край техники на краю кадра тяжело, а стрелками — ровно
   * столько нажатий, сколько нужно пикселей. Шаг считается в долях кадра, чтобы
   * не зависеть от текущего зума.
   */
  nudgeSelected(dx, dy, { resize = false, step = 0.002 } = {}) {
    if (this.selected < 0) return;
    this._remember();
    const box = this.boxes[this.selected];

    if (resize) {
      box.w = Math.max(0.002, box.w + dx * step * 2);
      box.h = Math.max(0.002, box.h + dy * step * 2);
    } else {
      box.x = Math.min(1, Math.max(0, box.x + dx * step));
      box.y = Math.min(1, Math.max(0, box.y + dy * step));
    }
    this._changed();
    this.invalidate();
  }

  /* ---------------------------------------------------------------- view */
  fit() {
    this._animate(1, { x: 0, y: 0 });
  }

  /** Button zoom pivots on the viewport centre, so the pan scales with it. */
  zoomBy(factor) {
    const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * factor));
    if (next <= 1.001) {
      this.fit();
      return;
    }
    const ratio = next / this.zoom;
    this._animate(next, this._clampPan({ x: this.pan.x * ratio, y: this.pan.y * ratio }, next));
  }

  _animate(zoom, pan) {
    this.anim = {
      from: { zoom: this.zoom, pan: { ...this.pan } },
      to: { zoom, pan },
      start: performance.now(),
    };
    this.invalidate();
  }

  _baseScale() {
    if (!this.image || !this.view) return 1;
    return Math.min(this.view.width / this.image.width, this.view.height / this.image.height);
  }

  _offset(zoom = this.zoom, pan = this.pan) {
    const scale = this._baseScale() * zoom;
    return {
      scale,
      dx: (this.view.width - this.image.width * scale) / 2 + pan.x,
      dy: (this.view.height - this.image.height * scale) / 2 + pan.y,
    };
  }

  /** Zoomed-out views stay centred; zoomed-in views cannot be dragged off screen. */
  _clampPan(pan, zoom = this.zoom) {
    if (!this.image) return { x: 0, y: 0 };
    const scale = this._baseScale() * zoom;
    const overflowX = Math.max(0, (this.image.width * scale - this.view.width) / 2);
    const overflowY = Math.max(0, (this.image.height * scale - this.view.height) / 2);
    return {
      x: Math.min(overflowX, Math.max(-overflowX, pan.x)),
      y: Math.min(overflowY, Math.max(-overflowY, pan.y)),
    };
  }

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.view = { width: rect.width, height: rect.height };
    this.pan = this._clampPan(this.pan);
    this.invalidate();
  }

  /* ------------------------------------------------------------ geometry */
  _toCanvas(box) {
    const { scale, dx, dy } = this._offset();
    const w = box.w * this.image.width * scale;
    const h = box.h * this.image.height * scale;
    return {
      x: box.x * this.image.width * scale + dx - w / 2,
      y: box.y * this.image.height * scale + dy - h / 2,
      w,
      h,
    };
  }

  _toImage(point) {
    const { scale, dx, dy } = this._offset();
    return {
      x: (point.x - dx) / (this.image.width * scale),
      y: (point.y - dy) / (this.image.height * scale),
    };
  }

  _pointer(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  _handleAt(point, index) {
    if (index < 0 || !this.boxes[index]) return null;
    const rect = this._toCanvas(this.boxes[index]);
    for (const [name, fx, fy, cursor] of HANDLES) {
      const hx = rect.x + rect.w * fx;
      const hy = rect.y + rect.h * fy;
      const reach = name.length === 2 ? HANDLE : EDGE_GRAB;
      const withinX = name === "n" || name === "s" ? Math.abs(point.x - hx) <= rect.w / 2 : Math.abs(point.x - hx) <= reach;
      const withinY = name === "w" || name === "e" ? Math.abs(point.y - hy) <= rect.h / 2 : Math.abs(point.y - hy) <= reach;
      if (withinX && withinY && (Math.abs(point.x - hx) <= reach || Math.abs(point.y - hy) <= reach)) {
        return { name, cursor };
      }
    }
    return null;
  }

  _boxAt(point, list = this.boxes) {
    let best = -1;
    let bestArea = Infinity;
    list.forEach((box, index) => {
      const rect = this._toCanvas(box);
      if (point.x < rect.x || point.x > rect.x + rect.w) return;
      if (point.y < rect.y || point.y > rect.y + rect.h) return;
      const area = rect.w * rect.h;
      if (area < bestArea) {
        bestArea = area;
        best = index;
      }
    });
    return best;
  }

  _visiblePredictions() {
    return this.predictions.filter((box) => box.conf >= this.confMin);
  }

  _cursor(point = null) {
    if (this.drag) {
      this.canvas.style.cursor = this.drag.mode === "pan" ? "grabbing" : this.drag.cursor || "grabbing";
      return;
    }
    if (this.mode === "draw") {
      this.canvas.style.cursor = "crosshair";
      return;
    }
    if (point && !this.readonly && this.showGroundTruth) {
      const handle = this._handleAt(point, this.selected);
      if (handle) {
        this.canvas.style.cursor = handle.cursor;
        return;
      }
      if (this._boxAt(point) >= 0) {
        this.canvas.style.cursor = "move";
        return;
      }
    }
    this.canvas.style.cursor = "grab";
  }

  /* --------------------------------------------------------- interaction */
  _bind() {
    const canvas = this.canvas;

    canvas.addEventListener("mousedown", (event) => {
      if (!this.image) return;
      this.anim = null;
      const point = this._pointer(event);

      const wantsPan =
        event.button === 1 || event.button === 2 || event.altKey || this.readonly ||
        (this.mode === "cursor" && event.shiftKey);

      if (wantsPan) {
        this.drag = { mode: "pan", start: point, origin: { ...this.pan } };
        this._cursor();
        return;
      }
      if (event.button !== 0) return;

      if (this.mode === "draw") {
        this._remember();
        const start = this._toImage(point);
        this.boxes.push({ cls: this.activeClass, x: start.x, y: start.y, w: 0, h: 0 });
        this.selected = this.boxes.length - 1;
        this.drag = { mode: "create", index: this.selected, start, cursor: "crosshair" };
        this.invalidate();
        return;
      }

      const handle = this._handleAt(point, this.selected);
      if (handle) {
        this._remember();
        this.drag = {
          mode: "resize",
          handle: handle.name,
          index: this.selected,
          cursor: handle.cursor,
        };
        this._cursor();
        return;
      }

      const hit = this.showGroundTruth ? this._boxAt(point) : -1;
      if (hit >= 0) {
        this.selected = hit;
        this.activeClass = this.boxes[hit].cls;
        if (this.onSelect) this.onSelect(this.boxes[hit], hit);
        this._remember();
        this.drag = {
          mode: "move",
          index: hit,
          start: this._toImage(point),
          origin: { ...this.boxes[hit] },
          cursor: "move",
        };
        this.invalidate();
        return;
      }

      // Empty space belongs to the picture, exactly like CVAT.
      if (this.selected >= 0) {
        this.selected = -1;
        this.invalidate();
      }
      this.drag = { mode: "pan", start: point, origin: { ...this.pan } };
      this._cursor();
    });

    canvas.addEventListener("dblclick", (event) => {
      if (!this.image || this.readonly) return;
      const point = this._pointer(event);
      const visible = this._visiblePredictions();
      const hit = this.showPredictions ? this._boxAt(point, visible) : -1;

      if (hit >= 0) {
        this._remember();
        const box = visible[hit];
        this.boxes.push({ cls: box.cls, x: box.x, y: box.y, w: box.w, h: box.h });
        this.selected = this.boxes.length - 1;
        this._changed();
        this.invalidate();
        return;
      }
      this.fit();
    });

    window.addEventListener("mousemove", (event) => {
      if (!this.image || !this.view) return;
      const point = this._pointer(event);

      if (!this.drag) {
        const hovered = this.mode === "cursor" && this.showGroundTruth ? this._boxAt(point) : -1;
        if (hovered !== this.hovered) {
          this.hovered = hovered;
          this.invalidate();
        }
        this._cursor(point);
        return;
      }

      if (this.drag.mode === "pan") {
        this.pan = this._clampPan({
          x: this.drag.origin.x + (point.x - this.drag.start.x),
          y: this.drag.origin.y + (point.y - this.drag.start.y),
        });
      } else if (this.drag.mode === "create") {
        const now = this._toImage(point);
        const box = this.boxes[this.drag.index];
        box.x = (this.drag.start.x + now.x) / 2;
        box.y = (this.drag.start.y + now.y) / 2;
        box.w = Math.abs(now.x - this.drag.start.x);
        box.h = Math.abs(now.y - this.drag.start.y);
      } else if (this.drag.mode === "move") {
        const now = this._toImage(point);
        const box = this.boxes[this.drag.index];
        box.x = this.drag.origin.x + (now.x - this.drag.start.x);
        box.y = this.drag.origin.y + (now.y - this.drag.start.y);
      } else if (this.drag.mode === "resize") {
        this._resizeBox(this._toImage(point));
      }
      this.invalidate();
    });

    window.addEventListener("mouseup", () => {
      if (!this.drag) return;
      const mode = this.drag.mode;

      if (mode === "create") {
        const box = this.boxes[this.drag.index];
        if (box.w < MIN_SIZE || box.h < MIN_SIZE) {
          this.boxes.splice(this.drag.index, 1);
          this.selected = -1;
        }
      }
      this.drag = null;

      if (mode !== "pan") {
        this._clampBoxes();
        this._changed();
      }
      this._cursor();
      this.invalidate();
    });

    canvas.addEventListener(
      "wheel",
      (event) => {
        if (!this.image) return;
        event.preventDefault();
        this.anim = null;

        const point = this._pointer(event);
        const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
        if (next === this.zoom) return;

        // Zooming back out lands on the original framing, never a shifted one.
        if (next <= 1.001) {
          this._animate(1, { x: 0, y: 0 });
          return;
        }

        const before = this._toImage(point);
        this.zoom = next;
        const after = this._toImage(point);
        const { scale } = this._offset();
        this.pan = this._clampPan({
          x: this.pan.x + (after.x - before.x) * this.image.width * scale,
          y: this.pan.y + (after.y - before.y) * this.image.height * scale,
        });
        this.invalidate();
      },
      { passive: false }
    );

    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("mouseleave", () => {
      if (this.hovered !== -1) {
        this.hovered = -1;
        this.invalidate();
      }
    });
  }

  _resizeBox(now) {
    const box = this.boxes[this.drag.index];
    let left = box.x - box.w / 2;
    let top = box.y - box.h / 2;
    let right = box.x + box.w / 2;
    let bottom = box.y + box.h / 2;

    const handle = this.drag.handle;
    if (handle.includes("w")) left = now.x;
    if (handle.includes("e")) right = now.x;
    if (handle.includes("n")) top = now.y;
    if (handle.includes("s")) bottom = now.y;

    box.x = (Math.min(left, right) + Math.max(left, right)) / 2;
    box.y = (Math.min(top, bottom) + Math.max(top, bottom)) / 2;
    box.w = Math.abs(right - left);
    box.h = Math.abs(bottom - top);
  }

  _clampBoxes() {
    this.boxes.forEach((box) => {
      box.w = Math.min(Math.max(box.w, MIN_SIZE), 1);
      box.h = Math.min(Math.max(box.h, MIN_SIZE), 1);
      box.x = Math.min(Math.max(box.x, box.w / 2), 1 - box.w / 2);
      box.y = Math.min(Math.max(box.y, box.h / 2), 1 - box.h / 2);
    });
  }

  _changed() {
    if (this.onChange) this.onChange(this.getBoxes());
  }

  /* -------------------------------------------------------------- render */
  invalidate() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this._step();
    });
  }

  _step() {
    if (this.anim) {
      const t = Math.min(1, (performance.now() - this.anim.start) / ANIM_MS);
      const k = easeOut(t);
      this.zoom = this.anim.from.zoom + (this.anim.to.zoom - this.anim.from.zoom) * k;
      this.pan = {
        x: this.anim.from.pan.x + (this.anim.to.pan.x - this.anim.from.pan.x) * k,
        y: this.anim.from.pan.y + (this.anim.to.pan.y - this.anim.from.pan.y) * k,
      };
      if (t >= 1) this.anim = null;
      else this.invalidate();
    }
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    if (!this.view) return;
    ctx.clearRect(0, 0, this.view.width, this.view.height);
    if (!this.image) return;

    const { scale, dx, dy } = this._offset();
    ctx.imageSmoothingEnabled = this.zoom < 3;
    ctx.drawImage(this.image, dx, dy, this.image.width * scale, this.image.height * scale);

    if (this.showPredictions) {
      // Same class palette as the ground truth: the dash is what says "predicted",
      // so colour stays free to carry the class.
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.8;
      this._visiblePredictions().forEach((box) => {
        const rect = this._toCanvas(box);
        const color = slotColor(box.cls);
        ctx.strokeStyle = color;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
        this._tag(`${box.cls} · ${box.conf.toFixed(2)}`, rect.x, rect.y + rect.h + 13, color, true);
      });
      ctx.setLineDash([]);
    }

    if (!this.showGroundTruth) {
      this._drawZoomBadge();
      return;
    }

    this.boxes.forEach((box, index) => {
      const rect = this._toCanvas(box);
      const color = slotColor(box.cls);
      const active = index === this.selected;
      const hover = index === this.hovered;

      ctx.lineWidth = active ? 2.4 : hover ? 2 : 1.6;
      ctx.strokeStyle = color;
      ctx.fillStyle = `${color}${active ? "2e" : hover ? "24" : "16"}`;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      this._tag(String(box.cls), rect.x, rect.y - 5, color, false);

      if (active) {
        ctx.fillStyle = color;
        ctx.strokeStyle = "#0a0e14";
        ctx.lineWidth = 1;
        HANDLES.forEach(([, fx, fy]) => {
          const hx = rect.x + rect.w * fx - HANDLE / 2;
          const hy = rect.y + rect.h * fy - HANDLE / 2;
          ctx.fillRect(hx, hy, HANDLE, HANDLE);
          ctx.strokeRect(hx, hy, HANDLE, HANDLE);
        });
      }
    });

    this._drawZoomBadge();
  }

  _drawZoomBadge() {
    if (this.zoom <= 1.01) return;
    const ctx = this.ctx;
    ctx.fillStyle = "rgba(10,14,20,.7)";
    ctx.fillRect(10, this.view.height - 26, 54, 18);
    ctx.fillStyle = "#9aa9bd";
    ctx.font = "600 10px ui-monospace, monospace";
    ctx.fillText(`${this.zoom.toFixed(1)}×`, 18, this.view.height - 13);
  }

  _tag(text, x, y, color, dashed) {
    const ctx = this.ctx;
    ctx.font = "600 10px ui-monospace, monospace";
    const width = ctx.measureText(text).width + 8;
    ctx.fillStyle = dashed ? "rgba(10,14,20,.7)" : color;
    ctx.fillRect(x, y - 11, width, 13);
    ctx.fillStyle = dashed ? color : "#0a0e14";
    ctx.fillText(text, x + 4, y - 1.5);
  }
}
