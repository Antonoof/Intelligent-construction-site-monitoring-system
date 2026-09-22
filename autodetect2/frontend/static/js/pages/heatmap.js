import { projectApi } from "../core/api.js";
import { clusterName } from "../core/cluster-names.js";
import { NEUTRAL, clear, el, fmt, qs, qsa, slotColor } from "../core/dom.js";
import { notifyError, runTask } from "../core/ui.js";
import { BoxEditor } from "../components/box-editor.js";
import { openEditor } from "../components/editor-modal.js";
import { HeatField, rampColor } from "../components/heatfield.js";

const root = qs(".heatmap");
const projectId = root.dataset.project;
const project = projectApi(projectId);

/* no-pred · ok · ghost · missed · mixed · low-conf */
const STATE_COLORS = [NEUTRAL, "#199e70", "#e66767", "#c98500", "#d55181", "#3987e5"];
const STATE_LABELS = {
  "no-pred": "нет предсказаний",
  ok: "совпало",
  ghost: "уверенный FP",
  missed: "пропуск GT",
  mixed: "FP + пропуск",
  "low-conf": "низкая уверенность",
};

/**
 * Every mode is a ratio: the numerator is what went wrong, the denominator is how
 * much could have gone wrong there.
 *
 * Counting errors per image just redraws object density — a street with sixty cars
 * loses more of them than an empty road, whatever the model does. Dividing by the
 * objects at stake is what makes a genuinely bad region stand out.
 *
 * "Ложные" counts only boxes drawn where nothing is, per real object: a prediction
 * that sits on an object but misses IoU 0.5 is a sloppy frame and gets its own mode.
 * Two thirds of all unmatched boxes are that, so mixing them in filled the map with
 * frames where the model had simply found almost nothing. Dividing by the ground
 * truth rather than by the predictions asks "how much did the model invent here",
 * instead of "how pure was the little it said".
 */
const HEAT_MODES = {
  errors: {
    title: "доля ошибок (пропуски + ложные)",
    hint: "ошибки / (объекты + предсказания)",
    values: (p, i) => p.fp[i] + p.fn[i],
    per: (p, i) => p.n_gt[i] + p.n_pred[i],
  },
  missed: {
    title: "доля пропущенных объектов",
    hint: "пропуски / размеченные объекты · 1 − recall",
    values: (p, i) => p.fn[i],
    per: (p, i) => p.n_gt[i],
  },
  ghost: {
    title: "лишних боксов на объект",
    hint: "боксы в пустом месте / размеченные объекты",
    values: (p, i) => p.fp[i] - p.loc[i],
    per: (p, i) => p.n_gt[i],
  },
  loose: {
    title: "доля сползших рамок",
    hint: "объект найден, но IoU < 0.5 / все предсказания",
    values: (p, i) => p.loc[i],
    per: (p, i) => p.n_pred[i],
  },
  interest: {
    title: "средний интерес",
    hint: "среднее по изображениям области",
    values: (p, i) => p.interest[i],
    per: null,
  },
  conf: {
    title: "средняя неуверенность",
    hint: "среднее 1 - conf по изображениям",
    values: (p, i) => (p.conf[i] > 0 ? Math.max(0, 1 - p.conf[i]) : 0),
    per: null,
  },
  density: {
    title: "плотность данных",
    hint: "сколько изображений попало в область",
    values: () => 1,
    per: null,
    sum: true,
  },
};

/** Pseudo-counts pulling a per-image rate towards the global one. */
const SMOOTHING = 5;

const state = { data: null, heat: "errors", points: "heat", threshold: 0 };
const field = new HeatField(qs("#field"), { onPick: showPoint, onHover: showTooltip });
const preview = new BoxEditor(qs("#preview"), { readonly: true });

/* ---------------------------------------------------------------- weights */
function signalFor(mode) {
  const points = state.data.points;
  const spec = HEAT_MODES[mode];
  const count = points.names.length;

  return {
    weight: Array.from({ length: count }, (_, i) => spec.values(points, i)),
    denominator: spec.per ? Array.from({ length: count }, (_, i) => spec.per(points, i)) : null,
    sum: Boolean(spec.sum),
  };
}

/**
 * Per-image value for the dots and the threshold, smoothed towards the global rate.
 *
 * One prediction that happens to be wrong is a 100% error rate on paper; without
 * smoothing those single-box frames own the top of every ranking and the real
 * offenders never surface.
 */
function pointValues(mode) {
  const points = state.data.points;
  const spec = HEAT_MODES[mode];
  const count = points.names.length;
  const raw = Array.from({ length: count }, (_, i) => spec.values(points, i));
  if (!spec.per) return raw;

  const per = Array.from({ length: count }, (_, i) => spec.per(points, i));
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < count; i += 1) {
    numerator += raw[i];
    denominator += per[i];
  }
  const global = denominator > 0 ? numerator / denominator : 0;

  return raw.map((value, i) => (value + SMOOTHING * global) / (per[i] + SMOOTHING));
}

/** Only the eight largest clusters get an identity hue; the tail stays neutral. */
function clusterSlots() {
  const ranked = [...state.data.clusters].sort((a, b) => b.size - a.size).slice(0, 8);
  return new Map(ranked.map((cluster, index) => [cluster.cluster, index]));
}

function colorsFor(mode, weights) {
  const points = state.data.points;
  if (mode === "cluster") {
    const slots = clusterSlots();
    return points.cluster.map((value) => (slots.has(value) ? slotColor(slots.get(value)) : NEUTRAL));
  }
  if (mode === "state") return points.state.map((value) => STATE_COLORS[value] || NEUTRAL);
  if (mode === "split") return points.split.map((value) => (value === "val" ? "#9085e9" : "#199e70"));
  const top = Math.max(...weights, 1e-6);
  return weights.map((value) => rampColor(0.3 + 0.7 * (value / top)));
}

function renderLegend() {
  const node = clear(qs("#legend"));
  const entry = (color, label) =>
    el("span", {}, [el("i", { style: { background: color } }), el("span", { text: label })]);

  if (state.points === "state") {
    state.data.states.forEach((name, index) =>
      node.append(entry(STATE_COLORS[index], STATE_LABELS[name] || name))
    );
  } else if (state.points === "cluster") {
    const slots = clusterSlots();
    slots.forEach((slot, cluster) => node.append(entry(slotColor(slot), clusterName(cluster))));
    if (state.data.clusters.length > slots.size) {
      node.append(entry(NEUTRAL, `прочие (${state.data.clusters.length - slots.size})`));
    }
  } else if (state.points === "split") {
    node.append(entry("#199e70", "train"), entry("#9085e9", "val"));
  }
  const mode = HEAT_MODES[state.heat];
  qs("#scale-title").textContent = mode.title;
  qs("#scale-hint").textContent = mode.hint;
}

function applyHeat() {
  const signal = signalFor(state.heat);
  const values = pointValues(state.heat);
  field.setWeight(signal.weight, { denominator: signal.denominator, sum: signal.sum });
  field.setPointValues(values);
  field.setColors(colorsFor(state.points, values));
  renderLegend();
}

/* ---------------------------------------------------------------- details */
function showTooltip(index, point) {
  const tooltip = qs("#tooltip");
  if (index < 0) {
    tooltip.classList.add("hidden");
    return;
  }

  const points = state.data.points;
  tooltip.classList.remove("hidden");
  tooltip.style.left = `${Math.min(point.x + 14, field.view.width - 210)}px`;
  tooltip.style.top = `${Math.min(point.y + 14, field.view.height - 90)}px`;

  clear(tooltip).append(
    el("b", { text: points.names[index] }),
    el("div", { text: `${clusterName(points.cluster[index])} · ${points.split[index]}` }),
    el("div", {
      text: `интерес ${points.interest[index].toFixed(3)} · conf ${points.conf[index].toFixed(2)}`,
    }),
    el("div", { text: STATE_LABELS[state.data.states[points.state[index]]] || "" })
  );
}

async function showPoint(index) {
  const points = state.data.points;
  const name = points.names[index];

  qs("#detail-name").textContent = name;
  qs("#detail-sub").textContent = `${clusterName(points.cluster[index])} · ${points.split[index]} · интерес ${points.interest[index].toFixed(3)}`;
  const button = qs("#detail-annotate");
  button.classList.remove("hidden");
  button.onclick = () =>
    openEditor(project, name, {
      meta: `${clusterName(points.cluster[index])} · ${points.split[index]}`,
      onSaved: () => showPoint(index),
    });

  try {
    const data = await project.boxes(name);
    await preview.load({ imageUrl: project.imageUrl(name), gt: data.gt, pred: data.pred });
    preview.confMin = 0.2;
    preview.invalidate();

    const stats = clear(qs("#detail-stats"));
    [
      ["GT", data.gt.length],
      ["Pred", data.pred.length],
      ["max conf", data.pred.length ? data.pred[0].conf.toFixed(2) : "—"],
    ].forEach(([label, value]) =>
      stats.append(el("div", {}, [el("b", { text: String(value) }), el("span", { text: label })]))
    );

    const boxes = clear(qs("#detail-boxes"));
    data.gt.forEach((box, i) =>
      boxes.append(
        el("div", { class: "boxrow" }, [
          el("i", { style: { background: slotColor(box.cls) } }),
          el("span", { text: `GT #${i} · c${box.cls}` }),
          el("span", { class: "spacer" }),
          el("span", { class: "faint", text: `${(box.w * 100).toFixed(0)}×${(box.h * 100).toFixed(0)}` }),
        ])
      )
    );
    data.pred.slice(0, 30).forEach((box, i) =>
      boxes.append(
        el("div", { class: "boxrow" }, [
          el("i", { style: { background: "var(--pink)" } }),
          el("span", { text: `PR #${i} · c${box.cls}` }),
          el("span", { class: "spacer" }),
          el("span", { class: "faint", text: box.conf.toFixed(3) }),
        ])
      )
    );
  } catch (error) {
    notifyError(error);
  }
}

/* ------------------------------------------------------------------- boot */
/** Interest, errors and confidence all come from submission.csv — without it only density is real. */
function lockModesWithoutSubmission() {
  state.heat = "density";
  state.points = "cluster";

  qsa("#heat-mode button").forEach((button) => {
    const locked = button.dataset.heat !== "density";
    button.disabled = locked;
    button.title = locked ? "Нужен submission.csv" : "";
    button.classList.toggle("is-active", !locked);
  });
  qsa("#point-mode button").forEach((button) => {
    const locked = button.dataset.mode === "state";
    button.disabled = locked;
    button.title = locked ? "Нужен submission.csv" : "";
    button.classList.toggle("is-active", button.dataset.mode === "cluster");
  });
  qs("#threshold").title = "Порог по плотности данных";
}

function apply(data) {
  state.data = data;
  if (!data.has_submission) lockModesWithoutSubmission();
  const signal = signalFor(state.heat);

  field.setData({
    x: data.points.x,
    y: data.points.y,
    weight: signal.weight,
    denominator: signal.denominator,
    sum: signal.sum,
    cluster: data.points.cluster,
    clusters: data.clusters,
    colors: colorsFor(state.points, pointValues(state.heat)),
  });
  field.setPointValues(pointValues(state.heat));
  renderLegend();

  qs("#map-meta").textContent = `${fmt.int(data.count)} точек · ${data.method} · ${data.backend}${
    data.has_submission ? "" : " · submission.csv не загружен"
  }`;
  qs("#map-loading").classList.add("hidden");
}

async function load(force = false) {
  const clusters = Number(qs("#clusters").value) || 24;
  qs("#map-loading").classList.remove("hidden");

  if (!force) {
    try {
      apply(await project.heatmap(clusters));
      return;
    } catch (error) {
      if (error.status !== 404 && error.status !== 409) notifyError(error);
    }
  }

  try {
    apply(await runTask(() => project.buildHeatmap(clusters, force), { title: "Строим карту данных" }));
  } catch (error) {
    qs("#map-loading").innerHTML = "";
    qs("#map-loading").append(el("span", { class: "text-red", text: error.message }));
    notifyError(error);
  }
}

qsa("#heat-mode button").forEach((button) => {
  button.addEventListener("click", () => {
    qsa("#heat-mode button").forEach((other) => other.classList.remove("is-active"));
    button.classList.add("is-active");
    state.heat = button.dataset.heat;
    applyHeat();
  });
});

qsa("#point-mode button").forEach((button) => {
  button.addEventListener("click", () => {
    qsa("#point-mode button").forEach((other) => other.classList.remove("is-active"));
    button.classList.add("is-active");
    state.points = button.dataset.mode;
    field.setColors(colorsFor(state.points, pointValues(state.heat)));
    renderLegend();
  });
});

qs("#threshold").addEventListener("input", (event) => {
  state.threshold = Number(event.target.value);
  qs("#threshold-label").textContent = state.threshold.toFixed(2);
  field.setThreshold(state.threshold);
});

qsa("#preview-layers button").forEach((button) => {
  button.addEventListener("click", () => {
    const layer = button.dataset.layer;
    const visible = !(layer === "gt" ? preview.showGroundTruth : preview.showPredictions);
    if (layer === "gt") preview.showGroundTruth = visible;
    else preview.showPredictions = visible;
    button.classList.toggle("is-active", visible);
    preview.invalidate();
  });
});

qs("#show-points").addEventListener("change", (event) => field.setShowPoints(event.target.checked));
qs("#show-labels").addEventListener("change", (event) => field.setShowLabels(event.target.checked));
qs("#rebuild").addEventListener("click", () => load(true));
qs("#zoom-in").addEventListener("click", () => field.zoomBy(1.6));
qs("#zoom-out").addEventListener("click", () => field.zoomBy(1 / 1.6));
qs("#zoom-reset").addEventListener("click", () => field.reset());

load();
