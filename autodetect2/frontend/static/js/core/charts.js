/**
 * Small SVG chart set for the statistics dashboard.
 * Marks stay thin, one hue per series, hover is part of the deliverable.
 */
const NS = "http://www.w3.org/2000/svg";
const SURFACE = "#10151e";

export const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
export const seriesColor = (index) =>
  getComputedStyle(document.documentElement).getPropertyValue(SERIES[index % SERIES.length]).trim();

function node(tag, attrs = {}, children = []) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    element.setAttribute(key, value);
  }
  [].concat(children).forEach((child) => child && element.append(child));
  return element;
}

function text(value, attrs = {}) {
  const element = node("text", { "font-size": 11, fill: "var(--text-dim)", ...attrs });
  element.textContent = value;
  return element;
}

/* --------------------------------------------------------------- tooltip */
let tip = null;

function tooltip() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "chart-tip hidden";
    document.body.append(tip);
  }
  return tip;
}

function showTip(event, rows, title) {
  const element = tooltip();
  element.replaceChildren();

  if (title) {
    const head = document.createElement("div");
    head.className = "chart-tip__head";
    head.textContent = title;
    element.append(head);
  }

  rows.forEach(([label, value, color]) => {
    const row = document.createElement("div");
    row.className = "chart-tip__row";

    if (color) {
      const key = document.createElement("i");
      key.style.background = color;
      row.append(key);
    }
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("b");
    number.textContent = value;
    row.append(name, number);
    element.append(row);
  });

  element.classList.remove("hidden");
  const rect = element.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - rect.width - 12);
  const y = Math.min(event.clientY + 14, window.innerHeight - rect.height - 12);
  element.style.transform = `translate(${x}px, ${y}px)`;
}

const hideTip = () => tooltip().classList.add("hidden");

/* --------------------------------------------------------------- helpers */
function mount(host, render) {
  const draw = () => {
    const width = host.clientWidth || 480;
    host.replaceChildren(render(width));
  };
  draw();
  if (!host.dataset.observed) {
    host.dataset.observed = "1";
    new ResizeObserver(() => draw()).observe(host);
  }
}

/** Labels are trimmed to the space they have — never clipped mid-glyph. */
const fit = (label, pixels) => {
  const budget = Math.max(4, Math.floor(pixels / 6.2));
  return label.length <= budget ? label : `${label.slice(0, budget - 1)}…`;
};

const niceTicks = (max, count = 4) => {
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw || 1));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((value) => value >= raw) || magnitude;
  return Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step);
};

/* ------------------------------------------------------------ bar chart */
export function barChart(host, { items, format = (v) => v.toFixed(3), color = 0, rowHeight = 26 }) {
  mount(host, (width) => {
    const labelWidth = Math.min(150, Math.max(70, width * 0.26));
    const valueWidth = 52;
    const height = items.length * rowHeight + 8;
    const trackWidth = Math.max(40, width - labelWidth - valueWidth - 12);
    const max = Math.max(...items.map((item) => item.value), 1e-9);
    const hue = `var(${SERIES[color % SERIES.length]})`;

    const svg = node("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" });

    items.forEach((item, index) => {
      const y = index * rowHeight + 4;
      const barHeight = Math.min(14, rowHeight - 12);
      const barWidth = Math.max(2, (item.value / max) * trackWidth);
      const barY = y + (rowHeight - 12 - barHeight) / 2 + 2;

      const group = node("g", { class: "chart-row", tabindex: "0" }, [
        text(fit(item.label, labelWidth - 12), {
          x: labelWidth - 8,
          y: barY + barHeight - 1,
          "text-anchor": "end",
        }),
        node("rect", {
          x: labelWidth,
          y: barY,
          width: trackWidth,
          height: barHeight,
          rx: 3,
          fill: "var(--grid)",
        }),
        node("rect", {
          x: labelWidth,
          y: barY,
          width: barWidth,
          height: barHeight,
          rx: 4,
          fill: item.color || hue,
        }),
        text(format(item.value), {
          x: labelWidth + trackWidth + 8,
          y: barY + barHeight - 1,
          fill: "var(--text)",
          "font-variant-numeric": "tabular-nums",
        }),
        node("rect", {
          x: 0,
          y,
          width,
          height: rowHeight,
          fill: "transparent",
          class: "chart-hit",
        }),
      ]);

      const rows = [[item.label, format(item.value), item.color || hue]];
      if (item.sub) rows.push([item.sub.label, item.sub.value, null]);
      group.addEventListener("pointermove", (event) => showTip(event, rows));
      group.addEventListener("pointerleave", hideTip);
      group.addEventListener("focus", (event) => showTip(event, rows));
      group.addEventListener("blur", hideTip);
      svg.append(group);
    });

    return svg;
  });
}

/* --------------------------------------------------------- column chart */
export function columnChart(host, { bins, color = 0, height = 180, format = (v) => String(v), xLabel = "" }) {
  mount(host, (width) => {
    const padding = { top: 12, right: 10, bottom: xLabel ? 30 : 20, left: 40 };
    const plotWidth = Math.max(20, width - padding.left - padding.right);
    const plotHeight = height - padding.top - padding.bottom;
    const max = Math.max(...bins.map((bin) => bin.value), 1);
    const ticks = niceTicks(max);
    const slot = plotWidth / bins.length;
    const barWidth = Math.min(24, Math.max(2, slot - 2));
    const hue = `var(${SERIES[color % SERIES.length]})`;

    const svg = node("svg", { width, height, viewBox: `0 0 ${width} ${height}` });

    ticks.forEach((tick) => {
      const y = padding.top + plotHeight - (tick / (ticks[ticks.length - 1] || 1)) * plotHeight;
      svg.append(
        node("line", {
          x1: padding.left,
          x2: width - padding.right,
          y1: y,
          y2: y,
          stroke: "var(--grid)",
          "stroke-width": 1,
        }),
        text(tick >= 1000 ? `${(tick / 1000).toFixed(0)}k` : tick, {
          x: padding.left - 8,
          y: y + 3.5,
          "text-anchor": "end",
          "font-size": 10,
        })
      );
    });

    bins.forEach((bin, index) => {
      const barHeight = (bin.value / (ticks[ticks.length - 1] || max)) * plotHeight;
      const x = padding.left + index * slot + (slot - barWidth) / 2;
      const y = padding.top + plotHeight - barHeight;

      const group = node("g", { class: "chart-row", tabindex: "0" }, [
        node("rect", {
          x,
          y,
          width: barWidth,
          height: Math.max(barHeight, 1),
          rx: Math.min(4, barWidth / 2),
          fill: hue,
        }),
        node("rect", {
          x: padding.left + index * slot,
          y: padding.top,
          width: slot,
          height: plotHeight,
          fill: "transparent",
          class: "chart-hit",
        }),
      ]);

      const rows = [[bin.label, format(bin.value), hue]];
      group.addEventListener("pointermove", (event) => showTip(event, rows));
      group.addEventListener("pointerleave", hideTip);
      group.addEventListener("focus", (event) => showTip(event, rows));
      group.addEventListener("blur", hideTip);
      svg.append(group);
    });

    svg.append(
      node("line", {
        x1: padding.left,
        x2: width - padding.right,
        y1: padding.top + plotHeight,
        y2: padding.top + plotHeight,
        stroke: "var(--axis)",
      })
    );

    [0, Math.floor(bins.length / 2), bins.length - 1].forEach((index) => {
      if (!bins[index]) return;
      svg.append(
        text(bins[index].tick ?? bins[index].label, {
          x: padding.left + index * slot + slot / 2,
          y: height - (xLabel ? 16 : 6),
          "text-anchor": "middle",
          "font-size": 10,
        })
      );
    });

    if (xLabel) {
      svg.append(
        text(xLabel, {
          x: padding.left + plotWidth / 2,
          y: height - 3,
          "text-anchor": "middle",
          "font-size": 10,
          fill: "var(--text-faint)",
        })
      );
    }

    return svg;
  });
}

/* ----------------------------------------------------------- line chart */
export function lineChart(host, {
  series,
  xTicks = [],
  height = 210,
  yFormat = (v) => v.toFixed(2),
  xFormat = (v) => String(v),
  area = false,
  yMax = null,
}) {
  mount(host, (width) => {
    const padding = { top: 14, right: 46, bottom: 26, left: 42 };
    const plotWidth = Math.max(30, width - padding.left - padding.right);
    const plotHeight = height - padding.top - padding.bottom;
    const points = series[0]?.points ?? [];
    const xs = points.map((point) => point.x);
    const xMin = Math.min(...xs, 0);
    const xSpan = Math.max(...xs, 1) - xMin || 1;
    const max = yMax ?? Math.max(...series.flatMap((s) => s.points.map((p) => p.y)), 1e-9);
    const ticks = niceTicks(max);
    const top = ticks[ticks.length - 1] || max;

    const projectX = (x) => padding.left + ((x - xMin) / xSpan) * plotWidth;
    const projectY = (y) => padding.top + plotHeight - (y / top) * plotHeight;

    const svg = node("svg", { width, height, viewBox: `0 0 ${width} ${height}` });

    ticks.forEach((tick) =>
      svg.append(
        node("line", {
          x1: padding.left,
          x2: width - padding.right,
          y1: projectY(tick),
          y2: projectY(tick),
          stroke: "var(--grid)",
        }),
        text(yFormat(tick), {
          x: padding.left - 8,
          y: projectY(tick) + 3.5,
          "text-anchor": "end",
          "font-size": 10,
        })
      )
    );

    (xTicks.length ? xTicks : xs.filter((_, i) => i % Math.ceil(xs.length / 5) === 0)).forEach((value) =>
      svg.append(
        text(xFormat(value), {
          x: projectX(value),
          y: height - 8,
          "text-anchor": "middle",
          "font-size": 10,
        })
      )
    );

    series.forEach((entry, index) => {
      const color = entry.color || `var(${SERIES[index % SERIES.length]})`;
      const path = entry.points.map((point, i) => `${i ? "L" : "M"}${projectX(point.x)},${projectY(point.y)}`).join(" ");

      if (area) {
        const base = `${path} L${projectX(entry.points.at(-1).x)},${projectY(0)} L${projectX(entry.points[0].x)},${projectY(0)} Z`;
        svg.append(node("path", { d: base, fill: color, opacity: 0.1 }));
      }

      svg.append(
        node("path", {
          d: path,
          fill: "none",
          stroke: color,
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );

      const last = entry.points.at(-1);
      if (last) {
        svg.append(
          node("circle", {
            cx: projectX(last.x),
            cy: projectY(last.y),
            r: 4,
            fill: color,
            stroke: SURFACE,
            "stroke-width": 2,
          })
        );
        if (series.length <= 4) {
          svg.append(
            text(yFormat(last.y), {
              x: projectX(last.x) + 9,
              y: projectY(last.y) + 3.5,
              "font-size": 10,
              fill: "var(--text-dim)",
              "font-variant-numeric": "tabular-nums",
            })
          );
        }
      }
    });

    const hairline = node("line", {
      y1: padding.top,
      y2: padding.top + plotHeight,
      stroke: "var(--text-faint)",
      "stroke-width": 1,
      opacity: 0,
    });
    svg.append(hairline);

    const overlay = node("rect", {
      x: padding.left,
      y: padding.top,
      width: plotWidth,
      height: plotHeight,
      fill: "transparent",
    });

    overlay.addEventListener("pointermove", (event) => {
      const bounds = svg.getBoundingClientRect();
      const local = event.clientX - bounds.left;
      const value = xMin + ((local - padding.left) / plotWidth) * xSpan;
      let nearest = 0;
      xs.forEach((x, i) => {
        if (Math.abs(x - value) < Math.abs(xs[nearest] - value)) nearest = i;
      });

      hairline.setAttribute("x1", projectX(xs[nearest]));
      hairline.setAttribute("x2", projectX(xs[nearest]));
      hairline.setAttribute("opacity", 0.6);

      showTip(
        event,
        series.map((entry, index) => [
          entry.name,
          yFormat(entry.points[nearest]?.y ?? 0),
          entry.color || `var(${SERIES[index % SERIES.length]})`,
        ]),
        xFormat(xs[nearest])
      );
    });

    overlay.addEventListener("pointerleave", () => {
      hairline.setAttribute("opacity", 0);
      hideTip();
    });

    svg.append(overlay);
    return svg;
  });
}

export function legend(host, entries) {
  host.replaceChildren();
  entries.forEach(([label, color, shape = "line"]) => {
    const item = document.createElement("span");
    const key = document.createElement("i");
    key.style.background = color;
    if (shape === "line") {
      key.style.height = "3px";
      key.style.width = "14px";
      key.style.borderRadius = "2px";
    }
    const name = document.createElement("span");
    name.textContent = label;
    item.append(key, name);
    host.append(item);
  });
}
