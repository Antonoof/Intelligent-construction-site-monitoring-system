import { projectApi } from "../core/api.js";
import { barChart, columnChart, legend, lineChart, seriesColor } from "../core/charts.js";
import { clusterName } from "../core/cluster-names.js";
import { clear, el, fmt, qs, qsa } from "../core/dom.js";
import { emptyState, notifyError, runTask, statCard } from "../core/ui.js";
import { submissionDialog } from "../components/submission.js";

const root = qs("[data-project]");
const project = projectApi(root.dataset.project);
const projectId = root.dataset.project;

let dashboard = null;
let apMode = "ap50";

/* ------------------------------------------------------------------ tiles */
function renderTiles() {
  const overall = dashboard.overall;
  const coverage = dashboard.coverage;

  qs("#hero-map").textContent = overall.map50.toFixed(3);
  qs("#hero-sub").textContent = `${fmt.int(coverage.gt_boxes)} GT · ${fmt.int(coverage.pred_boxes)} предсказаний · conf ≥ ${dashboard.conf_eval}`;

  const tiles = clear(qs("#tiles"));
  [
    { label: "mAP@50:95", value: overall.map.toFixed(3), sub: "усреднение по IoU 0.50…0.95", meter: overall.map },
    { label: "Precision", value: overall.precision.toFixed(3), sub: `${fmt.int(overall.fp)} ложных`, meter: overall.precision },
    { label: "Recall", value: overall.recall.toFixed(3), sub: `${fmt.int(overall.fn)} пропущено`, meter: overall.recall },
    { label: "F1", value: overall.f1.toFixed(3), sub: `${fmt.int(overall.tp)} совпадений`, meter: overall.f1 },
  ].forEach((tile) => tiles.append(statCard(tile)));
}

/* ----------------------------------------------------------------- charts */
function renderCurve() {
  const curve = dashboard.curve;
  const series = [
    { name: "Precision", points: curve.map((p) => ({ x: p.conf, y: p.precision })) },
    { name: "Recall", points: curve.map((p) => ({ x: p.conf, y: p.recall })) },
    { name: "F1", points: curve.map((p) => ({ x: p.conf, y: p.f1 })) },
  ];

  legend(
    qs("#curve-legend"),
    series.map((entry, index) => [entry.name, seriesColor(index)])
  );
  lineChart(qs("#curve-chart"), {
    series,
    height: 230,
    yMax: 1,
    yFormat: (v) => v.toFixed(2),
    xFormat: (v) => `conf ${Number(v).toFixed(2)}`,
  });
}

function renderClasses() {
  const names = dashboard.classes || [];
  const items = dashboard.overall.per_class.map((entry) => ({
    label: names[entry.class_id] || `class_${entry.class_id}`,
    value: entry[apMode],
    sub: { label: "GT боксов", value: fmt.int(entry.n_gt) },
  }));

  barChart(qs("#class-chart"), { items, format: (v) => v.toFixed(3) });

  const table = clear(qs("#class-table"));
  table.append(
    el("thead", {}, [
      el("tr", {}, ["Класс", "AP@50", "AP@50:95", "GT", "Предсказано"].map((title) => el("th", { text: title }))),
    ]),
    el(
      "tbody",
      {},
      dashboard.overall.per_class.map((entry) =>
        el("tr", {}, [
          el("td", { text: names[entry.class_id] || `class_${entry.class_id}` }),
          el("td", { text: entry.ap50.toFixed(3) }),
          el("td", { text: entry.ap.toFixed(3) }),
          el("td", { text: fmt.int(entry.n_gt) }),
          el("td", { text: fmt.int(entry.n_pred) }),
        ])
      )
    )
  );
}

function renderClusters() {
  const clusters = dashboard.clusters || [];
  if (!clusters.length) {
    qs("#cluster-chart").replaceChildren(
      el("p", { class: "muted tiny", text: "Кластеры не рассчитаны — пересчитайте статистику." })
    );
    return;
  }

  const sorted = [...clusters].sort((a, b) => a.map50 - b.map50);
  barChart(qs("#cluster-chart"), {
    items: sorted.map((entry) => ({
      label: clusterName(entry.cluster),
      value: entry.map50,
      sub: { label: "изображений", value: fmt.int(entry.images) },
    })),
    format: (v) => v.toFixed(3),
    color: 2,
    rowHeight: 22,
  });

  const table = clear(qs("#cluster-table"));
  table.append(
    el("thead", {}, [
      el("tr", {}, ["Кластер", "Изображений", "mAP@50", "P", "R", "FP", "Пропуски"].map((t) => el("th", { text: t }))),
    ]),
    el(
      "tbody",
      {},
      sorted.map((entry) =>
        el("tr", {}, [
          el("td", { text: clusterName(entry.cluster), title: `k${entry.cluster}` }),
          el("td", { text: fmt.int(entry.images) }),
          el("td", { text: entry.map50.toFixed(3) }),
          el("td", { text: entry.precision.toFixed(3) }),
          el("td", { text: entry.recall.toFixed(3) }),
          el("td", { text: fmt.int(entry.ghost) }),
          el("td", { text: fmt.int(entry.missed) }),
        ])
      )
    )
  );
}

function renderErrors() {
  const errors = dashboard.errors;
  barChart(qs("#errors-chart"), {
    items: [
      { label: "Уверенные FP", value: errors.ghost },
      { label: "Пропущенные GT", value: errors.missed },
      { label: "Плохая локализация", value: errors.loc },
      { label: "Полоса неуверенности", value: errors.ambiguous },
    ],
    format: (v) => fmt.int(v),
    color: 1,
    rowHeight: 30,
  });

  const notes = clear(qs("#error-notes"));
  const total = dashboard.coverage.images || 1;
  [
    [`${fmt.int(errors.clean_images)}`, `изображений без ошибок (${fmt.pct(errors.clean_images / total, 0)})`],
    [`${fmt.int(errors.images_with_errors)}`, "изображений содержат FP или пропуск"],
    [`${fmt.int(dashboard.coverage.with_predictions)}`, `из ${fmt.int(dashboard.coverage.images)} получили предсказания`],
  ].forEach(([value, label]) =>
    notes.append(el("span", {}, [el("b", { text: value }), " ", label]))
  );
}

function binsFrom(histogram, labelFormat) {
  return histogram.counts.map((count, index) => ({
    label: labelFormat(histogram.edges[index], histogram.edges[index + 1]),
    tick: histogram.edges[index].toFixed(2),
    value: count,
  }));
}

function renderDistributions() {
  const dist = dashboard.distributions;

  columnChart(qs("#conf-chart"), {
    bins: binsFrom(dist.confidence, (a, b) => `conf ${a.toFixed(2)}–${b.toFixed(2)}`),
    format: (v) => fmt.int(v),
    xLabel: "confidence",
    color: 0,
  });

  const gt = binsFrom(dist.gt_area, (a, b) => `√площадь ${a.toFixed(2)}–${b.toFixed(2)}`);
  const pred = binsFrom(dist.pred_area, (a, b) => `√площадь ${a.toFixed(2)}–${b.toFixed(2)}`);
  legend(qs("#area-legend"), [
    ["Разметка", seriesColor(2)],
    ["Предсказания", seriesColor(4)],
  ]);
  lineChart(qs("#area-chart"), {
    series: [
      { name: "Разметка", color: seriesColor(2), points: gt.map((bin, i) => ({ x: i, y: bin.value })) },
      { name: "Предсказания", color: seriesColor(4), points: pred.map((bin, i) => ({ x: i, y: bin.value })) },
    ],
    height: 180,
    area: true,
    yFormat: (v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(Math.round(v))),
    xFormat: (i) => `√S ≈ ${(dist.gt_area.edges[i] ?? 0).toFixed(2)}`,
  });

  columnChart(qs("#density-chart"), {
    bins: binsFrom(dist.boxes_per_image, (a, b) => `${a.toFixed(0)}–${b.toFixed(0)} объектов`),
    format: (v) => fmt.int(v),
    xLabel: "объектов в разметке",
    color: 2,
  });
}

function renderTables() {
  const worst = clear(qs("#worst-table"));
  worst.append(
    el("thead", {}, [
      el("tr", {}, ["Изображение", "GT", "Pred", "FP", "Пропуски", "IoU"].map((t) => el("th", { text: t }))),
    ]),
    el(
      "tbody",
      {},
      dashboard.worst.map((row) =>
        el(
          "tr",
          {
            onClick: () => window.open(`/p/${projectId}/heatmap`, "_self"),
            title: "Открыть карту данных",
          },
          [
            el("td", { class: "mono truncate", style: { maxWidth: "220px" }, text: row.name }),
            el("td", { text: row.n_gt }),
            el("td", { text: row.n_pred }),
            el("td", { class: row.ghost ? "text-red" : "", text: row.ghost }),
            el("td", { class: row.missed ? "text-amber" : "", text: row.missed }),
            el("td", { text: row.iou_mean.toFixed(2) }),
          ]
        )
      )
    )
  );

  const split = clear(qs("#split-table"));
  split.append(
    el("thead", {}, [
      el("tr", {}, ["Split", "Изображений", "mAP@50", "Precision", "Recall", "F1"].map((t) => el("th", { text: t }))),
    ]),
    el(
      "tbody",
      {},
      Object.entries(dashboard.splits).map(([name, value]) =>
        el("tr", {}, [
          el("td", { text: name }),
          el("td", { text: fmt.int(value.images) }),
          el("td", { text: value.map50.toFixed(3) }),
          el("td", { text: value.precision.toFixed(3) }),
          el("td", { text: value.recall.toFixed(3) }),
          el("td", { text: value.f1.toFixed(3) }),
        ])
      )
    )
  );
}

function render(data) {
  dashboard = data;
  qs("#dashboard").classList.remove("hidden");
  qs("#placeholder").replaceChildren();

  renderTiles();
  renderCurve();
  renderClasses();
  renderClusters();
  renderErrors();
  renderDistributions();
  renderTables();
}

/* ------------------------------------------------------------------- boot */
async function recalc() {
  const params = {
    conf: Number(qs("#conf").value) || 0.25,
    iou: Number(qs("#iou").value) || 0.5,
    clusters: Number(qs("#clusters").value) || 12,
  };

  try {
    render(await runTask(() => project.buildStats(params), { title: "Считаем метрики" }));
  } catch (error) {
    notifyError(error);
  }
}

qs("#recalc").addEventListener("click", recalc);

qsa("#ap-mode button").forEach((button) => {
  button.addEventListener("click", () => {
    qsa("#ap-mode button").forEach((other) => other.classList.remove("is-active"));
    button.classList.add("is-active");
    apMode = button.dataset.ap;
    renderClasses();
  });
});

async function boot() {
  if (root.dataset.hasSubmission !== "1") {
    qs("#placeholder").replaceChildren(
      emptyState(
        "Для статистики нужен submission.csv",
        el("button", {
          class: "btn btn--primary",
          text: "Загрузить результаты",
          onClick: async () => {
            if (await submissionDialog(project)) window.location.reload();
          },
        })
      )
    );
    return;
  }

  try {
    const cached = await project.stats();
    qs("#conf").value = cached.conf_eval;
    qs("#iou").value = cached.iou_thr;
    render(cached);
  } catch (error) {
    if (error.status === 404) recalc();
    else notifyError(error);
  }
}

boot();
