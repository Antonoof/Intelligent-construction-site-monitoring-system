import { clusterName } from "../core/cluster-names.js";

const GRID_W = 260;
const GRID_H = 170;
const PAD = 26;
const MAX_ZOOM = 60;
const ANIM_MS = 200;
const BLUR_PASSES = 3;
const BLUR_RADIUS = 3;

/** Magma-like ramp: hue moves, lightness only ever rises — readable as one scale. */
const RAMP = [
  [0.0, [24, 16, 54], 0.0],
  [0.12, [59, 15, 112], 0.45],
  [0.35, [140, 41, 129], 0.72],
  [0.58, [222, 73, 104], 0.85],
  [0.8, [254, 159, 109], 0.92],
  [1.0, [252, 253, 191], 0.96],
];

const easeOut = (t) => 1 - (1 - t) ** 3;

export function rampColor(value) {
  const v = Math.max(0, Math.min(1, value));
  for (let i = 1; i < RAMP.length; i += 1) {
    if (v <= RAMP[i][0]) {
      const [p0, c0] = RAMP[i - 1];
      const [p1, c1] = RAMP[i];
      const t = (v - p0) / (p1 - p0 || 1);
      const rgb = c0.map((c, k) => Math.round(c + (c1[k] - c) * t));
      return `rgb(${rgb.join(",")})`;
    }
  }
  return "rgb(252,253,191)";
}

/**
 * Density heat map over a 2D embedding.
 *
 * Points are splatted into a grid and blurred once into an offscreen image, so
 * panning and zooming only blit that image — the field, the dots and the cluster
 * labels are produced in the same frame and can never drift apart.
 */
export class HeatField {
  constructor(canvas, { onPick = null, onHover = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.onPick = onPick;
    this.onHover = onHover;

    this.x = new Float32Array(0);
    this.y = new Float32Array(0);
    this.weight = new Float32Array(0);
    this.cluster = [];
    this.clusters = [];
    this.pointColors = null;

    this.threshold = 0;
    this.showPoints = true;
    this.showLabels = true;
    this.selected = -1;
    this.hovered = -1;

    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.drag = null;
    this.anim = null;
    this.frame = null;

    this.heat = document.createElement("canvas");
    this.heat.width = GRID_W;
    this.heat.height = GRID_H;

    this._bind();
    this.resize();
    this._observer = new ResizeObserver(() => this.resize());
    this._observer.observe(canvas);
  }

  /* ---------------------------------------------------------------- data */
  setData({ x, y, weight, denominator = null, sum = false, cluster = [], clusters = [], colors = null }) {
    this.x = Float32Array.from(x);
    this.y = Float32Array.from(y);
    this.cluster = cluster;
    this.clusters = clusters;
    this.pointColors = colors;
    this.setWeight(weight, { denominator, sum });
  }

  /**
   * `sum` counts, anything else divides.
   *
   * With no denominator each point weighs one and the cell shows a mean; with a
   * denominator the cell shows a rate — errors over objects, not errors over
   * images — which is the only form that survives a dense scene.
   */
  setWeight(weight, { denominator = null, sum = false } = {}) {
    this.weight = Float32Array.from(weight);
    this.denominator = denominator ? Float32Array.from(denominator) : null;
    this.sum = sum;
    this._buildField();
    this.invalidate();
  }

  /**
   * Per-point value the threshold compares against.
   *
   * In rate modes the raw weight is a count (five misses), while the map shows a
   * share — so the dots must be judged by the same share, or a busy image looks
   * worse than a genuinely failing one.
   */
  setPointValues(values) {
    this.pointValues = Float32Array.from(values);
    let top = 0;
    for (let i = 0; i < this.pointValues.length; i += 1) {
      if (this.pointValues[i] > top) top = this.pointValues[i];
    }
    this._maxPoint = top || 1;
    this._weightQ = HeatField._quantiles(
      Array.from(this.pointValues, (v) => v / this._maxPoint)
    );
    this.invalidate();
  }

  _relative(index) {
    const values = this.pointValues || this.weight;
    const top = this.pointValues ? this._maxPoint : this._maxWeight;
    return values[index] / (top || 1);
  }

  /**
   * The slider works in quantiles, not raw values.
   *
   * Density is heavy tailed — a linear cut at 0.5 would wipe out almost the whole
   * map in one step. On quantiles "0.5" means "hide the cooler half of the heat",
   * which is what the reader expects from the handle position.
   */
  static _quantiles(values) {
    const sorted = Float64Array.from(values).sort();
    const table = new Float32Array(101);
    for (let i = 0; i <= 100; i += 1) {
      table[i] = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((i / 100) * sorted.length))] : 0;
    }
    return table;
  }

  _fieldCut() {
    if (!this.threshold || !this._fieldQ) return 0;
    return this._fieldQ[Math.round(this.threshold * 100)];
  }

  _pointCut() {
    if (!this.threshold || !this._weightQ) return 0;
    return this._weightQ[Math.round(this.threshold * 100)];
  }

  setColors(colors) {
    this.pointColors = colors;
    this.invalidate();
  }

  setThreshold(value) {
    this.threshold = value;
    this._paintField();
    this.invalidate();
  }

  setShowPoints(value) {
    this.showPoints = value;
    this.invalidate();
  }

  setShowLabels(value) {
    this.showLabels = value;
    this.invalidate();
  }

  /**
   * Grid accumulation + separable box blur ≈ a Gaussian kernel, done once.
   *
   * Two grids are filled: the weights and the plain point count. Density mode
   * reads the first; value modes divide one by the other, because a sum over 45k
   * points just redraws the density map no matter what the weights say.
   */
  _buildField() {
    const cells = GRID_W * GRID_H;
    const sums = new Float32Array(cells);
    const hits = new Float32Array(cells);

    let top = 0;
    for (let i = 0; i < this.weight.length; i += 1) {
      if (this.weight[i] > top) top = this.weight[i];
    }
    this._maxWeight = top || 1;

    for (let i = 0; i < this.x.length; i += 1) {
      const fx = Math.max(0, Math.min(GRID_W - 1.001, this.x[i] * (GRID_W - 1)));
      const fy = Math.max(0, Math.min(GRID_H - 1.001, (1 - this.y[i]) * (GRID_H - 1)));
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const tx = fx - x0;
      const ty = fy - y0;
      const w = this.weight[i] || 0;

      const a = (1 - tx) * (1 - ty);
      const b = tx * (1 - ty);
      const c = (1 - tx) * ty;
      const d = tx * ty;
      const i00 = y0 * GRID_W + x0;
      const i10 = (y0 + 1) * GRID_W + x0;
      const q = this.denominator ? this.denominator[i] || 0 : 1;

      sums[i00] += w * a;
      sums[i00 + 1] += w * b;
      sums[i10] += w * c;
      sums[i10 + 1] += w * d;
      hits[i00] += q * a;
      hits[i00 + 1] += q * b;
      hits[i10] += q * c;
      hits[i10 + 1] += q * d;
    }

    let blurredSums = sums;
    let blurredHits = hits;
    for (let pass = 0; pass < BLUR_PASSES; pass += 1) {
      blurredSums = this._blur(blurredSums);
      blurredHits = this._blur(blurredHits);
    }

    this.support = new Uint8Array(cells);
    const field = new Float32Array(cells);

    if (this.sum) {
      // Scale by a high percentile so a single hot cell cannot flatten the rest.
      const values = Array.from(blurredSums).filter((v) => v > 0).sort((a, b) => a - b);
      const peak = values.length ? values[Math.floor(values.length * 0.995)] : 1;
      const scale = peak > 0 ? 1 / peak : 1;
      for (let i = 0; i < cells; i += 1) {
        if (blurredSums[i] <= 0) continue;
        this.support[i] = 1;
        field[i] = Math.min(1, blurredSums[i] * scale);
      }
    } else {
      // A rate computed from two objects is noise, not a signal — require support.
      const occupancy = Array.from(blurredHits).filter((v) => v > 0).sort((a, b) => a - b);
      const median = occupancy.length ? occupancy[Math.floor(occupancy.length * 0.5)] : 0;
      const floor = Math.max(median * 0.15, 1e-4);

      const valid = [];
      for (let i = 0; i < cells; i += 1) {
        if (blurredHits[i] < floor) continue;
        this.support[i] = 1;
        field[i] = blurredSums[i] / blurredHits[i];
        valid.push(field[i]);
      }

      // Rates sit in a narrow band; stretch it or every mode looks identically flat.
      // The clip stays at 1%/99%: cutting deeper saturates the worst regions into
      // one flat colour, which is exactly where the gradations matter most.
      valid.sort((a, b) => a - b);
      const lo = valid.length ? valid[Math.floor(valid.length * 0.01)] : 0;
      const hi = valid.length ? valid[Math.floor(valid.length * 0.99)] : 1;
      const span = hi - lo;
      if (span > 1e-6) {
        for (let i = 0; i < cells; i += 1) {
          if (this.support[i]) field[i] = Math.max(0, Math.min(1, (field[i] - lo) / span));
        }
      }
    }

    this.field = field;
    this._fieldQ = HeatField._quantiles(field.filter((v, i) => this.support[i]));
    if (!this.pointValues) {
      this._weightQ = HeatField._quantiles(
        Array.from(this.weight, (v) => v / (this._maxWeight || 1))
      );
    }
    this._paintField();
  }

  _blur(source) {
    const tmp = new Float32Array(source.length);
    const out = new Float32Array(source.length);
    const span = BLUR_RADIUS * 2 + 1;

    for (let y = 0; y < GRID_H; y += 1) {
      for (let x = 0; x < GRID_W; x += 1) {
        let sum = 0;
        for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k += 1) {
          const xx = Math.min(GRID_W - 1, Math.max(0, x + k));
          sum += source[y * GRID_W + xx];
        }
        tmp[y * GRID_W + x] = sum / span;
      }
    }
    for (let x = 0; x < GRID_W; x += 1) {
      for (let y = 0; y < GRID_H; y += 1) {
        let sum = 0;
        for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k += 1) {
          const yy = Math.min(GRID_H - 1, Math.max(0, y + k));
          sum += tmp[yy * GRID_W + x];
        }
        out[y * GRID_W + x] = sum / span;
      }
    }
    return out;
  }

  _paintField() {
    if (!this.field) return;
    const ctx = this.heat.getContext("2d");
    const image = ctx.createImageData(GRID_W, GRID_H);
    const data = image.data;

    const cut = this._fieldCut();

    for (let i = 0; i < this.field.length; i += 1) {
      if (!this.support[i]) continue;
      const value = this.field[i];
      const offset = i * 4;
      if (value <= cut) continue;

      // Re-stretch above the cut so raising the threshold reveals structure.
      const shown = (value - cut) / (1 - cut || 1);
      let alpha = 0;
      let rgb = [0, 0, 0];
      for (let s = 1; s < RAMP.length; s += 1) {
        if (shown <= RAMP[s][0]) {
          const [p0, c0, a0] = RAMP[s - 1];
          const [p1, c1, a1] = RAMP[s];
          const t = (shown - p0) / (p1 - p0 || 1);
          rgb = c0.map((c, k) => Math.round(c + (c1[k] - c) * t));
          alpha = a0 + (a1 - a0) * t;
          break;
        }
      }
      if (shown > 1) {
        rgb = RAMP[RAMP.length - 1][1];
        alpha = RAMP[RAMP.length - 1][2];
      }

      data[offset] = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = Math.round(alpha * 255);
    }
    ctx.putImageData(image, 0, 0);
  }

  /* ---------------------------------------------------------------- view */
  resize() {
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

  reset() {
    this._animate(1, { x: 0, y: 0 });
  }

  zoomBy(factor) {
    const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * factor));
    if (next <= 1.001) {
      this.reset();
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

  /** The plot area in screen space for the current zoom and pan. */
  _rect(zoom = this.zoom, pan = this.pan) {
    const width = (this.view.width - PAD * 2) * zoom;
    const height = (this.view.height - PAD * 2) * zoom;
    return {
      x: PAD + (this.view.width - PAD * 2 - width) / 2 + pan.x,
      y: PAD + (this.view.height - PAD * 2 - height) / 2 + pan.y,
      width,
      height,
    };
  }

  _clampPan(pan, zoom = this.zoom) {
    const overflowX = Math.max(0, ((this.view?.width ?? 0) - PAD * 2) * (zoom - 1) / 2);
    const overflowY = Math.max(0, ((this.view?.height ?? 0) - PAD * 2) * (zoom - 1) / 2);
    return {
      x: Math.min(overflowX, Math.max(-overflowX, pan.x)),
      y: Math.min(overflowY, Math.max(-overflowY, pan.y)),
    };
  }

  _project(index, rect = this._rect()) {
    return {
      x: rect.x + this.x[index] * rect.width,
      y: rect.y + (1 - this.y[index]) * rect.height,
    };
  }

  _pointer(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  _nearest(point, radius = 14) {
    const rect = this._rect();
    let best = -1;
    let bestDistance = radius * radius;

    const cut = this._pointCut();

    for (let i = 0; i < this.x.length; i += 1) {
      if (cut > 0 && this._relative(i) < cut) continue;
      const px = rect.x + this.x[i] * rect.width;
      if (px < -radius || px > this.view.width + radius) continue;
      const py = rect.y + (1 - this.y[i]) * rect.height;
      if (py < -radius || py > this.view.height + radius) continue;

      const distance = (px - point.x) ** 2 + (py - point.y) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return best;
  }

  /* --------------------------------------------------------- interaction */
  _bind() {
    const canvas = this.canvas;

    canvas.addEventListener("mousedown", (event) => {
      this.anim = null;
      this.drag = { start: this._pointer(event), origin: { ...this.pan }, moved: false };
      canvas.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (event) => {
      if (!this.view) return;
      const point = this._pointer(event);

      if (this.drag) {
        const dx = point.x - this.drag.start.x;
        const dy = point.y - this.drag.start.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
        this.pan = this._clampPan({ x: this.drag.origin.x + dx, y: this.drag.origin.y + dy });
        this.invalidate();
        return;
      }

      const bounds = canvas.getBoundingClientRect();
      const inside =
        event.clientX >= bounds.left && event.clientX <= bounds.right &&
        event.clientY >= bounds.top && event.clientY <= bounds.bottom;

      const hit = inside ? this._nearest(point) : -1;
      if (hit !== this.hovered) {
        this.hovered = hit;
        canvas.style.cursor = hit >= 0 ? "pointer" : "grab";
        this.invalidate();
      }
      if (this.onHover) this.onHover(hit, point);
    });

    window.addEventListener("mouseup", (event) => {
      if (!this.drag) return;
      const moved = this.drag.moved;
      this.drag = null;
      canvas.style.cursor = "grab";
      if (moved) return;

      const hit = this._nearest(this._pointer(event));
      if (hit >= 0) {
        this.selected = hit;
        if (this.onPick) this.onPick(hit);
        this.invalidate();
      }
    });

    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.anim = null;

        const point = this._pointer(event);
        const next = Math.min(MAX_ZOOM, Math.max(1, this.zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
        if (next === this.zoom) return;

        if (next <= 1.001) {
          this._animate(1, { x: 0, y: 0 });
          return;
        }

        // Keep the point under the cursor pinned while zooming.
        const rect = this._rect();
        const ux = (point.x - rect.x) / rect.width;
        const uy = (point.y - rect.y) / rect.height;
        this.zoom = next;
        const after = this._rect(next, this.pan);
        this.pan = this._clampPan({
          x: this.pan.x + (point.x - (after.x + ux * after.width)),
          y: this.pan.y + (point.y - (after.y + uy * after.height)),
        });
        this.invalidate();
      },
      { passive: false }
    );

    canvas.addEventListener("dblclick", () => this.reset());
    canvas.addEventListener("mouseleave", () => {
      if (this.hovered !== -1) {
        this.hovered = -1;
        if (this.onHover) this.onHover(-1, { x: 0, y: 0 });
        this.invalidate();
      }
    });
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

    ctx.fillStyle = "#0a0e14";
    ctx.fillRect(0, 0, this.view.width, this.view.height);
    const rect = this._rect();

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.heat, rect.x, rect.y, rect.width, rect.height);

    if (this.showPoints) this._drawPoints(rect);
    if (this.showLabels) this._drawLabels(rect);

    if (this.hovered >= 0) this._ring(this.hovered, "#ffffff", 6, 1.5, rect);
    if (this.selected >= 0) this._ring(this.selected, "#4c8dff", 9, 2, rect);
  }

  _drawPoints(rect) {
    const ctx = this.ctx;
    const step = this.zoom > 1.5 ? 1 : Math.max(1, Math.floor(this.x.length / 45000));
    const size = Math.max(1.2, Math.min(5, 1.5 * Math.sqrt(this.zoom)));

    const cut = this._pointCut();

    for (let i = 0; i < this.x.length; i += step) {
      const dim = cut > 0 && this._relative(i) < cut;
      const px = rect.x + this.x[i] * rect.width;
      if (px < -4 || px > this.view.width + 4) continue;
      const py = rect.y + (1 - this.y[i]) * rect.height;
      if (py < -4 || py > this.view.height + 4) continue;

      ctx.globalAlpha = dim ? 0.16 : 0.85;
      ctx.fillStyle = this.pointColors ? this.pointColors[i] : "#cfe0ff";
      ctx.fillRect(px - size / 2, py - size / 2, size, size);
    }
    ctx.globalAlpha = 1;
  }

  _drawLabels(rect) {
    const ctx = this.ctx;
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";

    this.clusters.forEach((cluster) => {
      const x = rect.x + cluster.cx * rect.width;
      const y = rect.y + (1 - cluster.cy) * rect.height;
      if (x < 0 || x > this.view.width || y < 0 || y > this.view.height) return;

      const label = clusterName(cluster.cluster);
      const width = ctx.measureText(label).width + 14;

      ctx.fillStyle = "rgba(10,14,20,.74)";
      ctx.strokeStyle = "rgba(255,255,255,.18)";
      ctx.lineWidth = 1;
      const boxX = x - width / 2;
      const boxY = y - 9;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(boxX, boxY, width, 18, 9);
      else ctx.rect(boxX, boxY, width, 18);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#e8eef7";
      ctx.textAlign = "center";
      ctx.fillText(label, x, y + 0.5);
    });
    ctx.textAlign = "start";
  }

  _ring(index, color, radius, width, rect) {
    if (index < 0 || index >= this.x.length) return;
    const point = this._project(index, rect);
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}
