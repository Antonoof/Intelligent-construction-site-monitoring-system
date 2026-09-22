import { projectApi } from "../core/api.js";
import { clusterName } from "../core/cluster-names.js";
import { clear, slotColor, el, fmt, qs, qsa } from "../core/dom.js";
import { confirmDialog, modal, notifyError, runTask, toast } from "../core/ui.js";
import { BoxEditor } from "../components/box-editor.js";

const root = qs(".annotate");
const projectId = root.dataset.project;
const project = projectApi(projectId);
const classNames = JSON.parse(root.dataset.classes || "[]");

const state = {
  items: [],
  index: -1,
  done: new Set(),
  filter: "all",
  source: "selection",
  signals: null,
  clipboard: null, // разметка предыдущего кадра — для «как в предыдущем»
  dirty: false,
  saved: 0,
  startedAt: null,
};

const editor = new BoxEditor(qs("#editor"), {
  onChange: () => {
    state.dirty = true;
    renderBoxList();
  },
  onSelect: () => renderBoxList(),
  onHistory: (canUndo, canRedo) => {
    qs("#undo").disabled = !canUndo;
    qs("#redo").disabled = !canRedo;
  },
  onMode: (mode) => {
    qsa("#mode button").forEach((button) =>
      button.classList.toggle("is-active", button.dataset.mode === mode)
    );
    flashHint(mode === "draw" ? "Режим рисования — тяните рамку, Esc для выхода" : "Курсор");
  },
});

let hintTimer = null;

function flashHint(text) {
  const node = qs("#mode-hint");
  node.textContent = text;
  node.classList.add("is-on");
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => node.classList.remove("is-on"), 1400);
}

const REASON_TONE = {
  ghost: "badge--red",
  missed: "badge--amber",
  ambiguous: "badge--violet",
  count: "badge--accent",
  diversity: "badge--green",
  medoid: "badge",
  assignment: "badge--violet",
  issue: "badge--red",
  empty: "badge--amber",
  plain: "badge",
};

/* ------------------------------------------------------------- selection */
function budgetDialog() {
  const count = el("input", { type: "number", min: "1", max: "100000", value: "100" });
  const clusters = el("input", { type: "number", min: "2", max: "2000", placeholder: "авто" });
  const alpha = el("input", { type: "range", min: "0", max: "1", step: "0.05", value: "0.5" });
  const alphaValue = el("b", { text: "0.50" });
  const split = el("select", {}, [
    el("option", { value: "", text: "train + val" }),
    el("option", { value: "train", text: "только train" }),
    el("option", { value: "val", text: "только val" }),
  ]);
  const scope = el("select", {}, [
    el("option", { value: "all", text: "любые кадры" }),
    el("option", { value: "unlabeled", text: "только без разметки" }),
    el("option", { value: "labeled", text: "только размеченные (проверка)" }),
    el("option", { value: "issues", text: "только с замечаниями" }),
  ]);

  alpha.addEventListener("input", () => (alphaValue.textContent = Number(alpha.value).toFixed(2)));

  const presets = el(
    "div",
    { class: "budget__presets" },
    [50, 100, 250, 500, 1000].map((value) =>
      el("button", {
        class: "btn btn--sm",
        text: String(value),
        onClick: () => (count.value = String(value)),
      })
    )
  );

  const body = el("div", { class: "budget" }, [
    el("div", {}, [
      el("div", { class: "label", text: "Сколько изображений вы готовы разметить?" }),
      el("div", { class: "budget__count" }, [count, presets]),
    ]),
    el("div", { class: "budget__algo" }, [
      el("div", {}, [
        el("b", { text: "1 · Кластеры" }),
        el("span", {
          text: "DINOv2-эмбеддинги, PCA-whitening и k-means: датасет режется на k групп, бюджет делится между ними по размеру и «массе ошибок».",
        }),
      ]),
      el("div", {}, [
        el("b", { text: "2 · Интересы" }),
        el("span", {
          text: "Из submission.csv: уверенный бокс без GT, пропущенный объект, боксы в полосе неуверенности и расхождение по количеству.",
        }),
      ]),
      el("div", {}, [
        el("b", { text: "3 · Разнообразие" }),
        el("span", {
          text: "Внутри кластера жадный отбор: максимум информативности при максимальном удалении от уже выбранного.",
        }),
      ]),
    ]),
    el("div", { class: "grid grid--2" }, [
      el("div", { class: "field" }, [
        el("label", { text: "Число кластеров" }),
        clusters,
        el("span", { class: "field__hint", text: "по умолчанию — равно бюджету" }),
      ]),
      el("div", { class: "field" }, [el("label", { text: "Данные" }), split]),
    ]),
    el("div", { class: "field" }, [
      el("label", { text: "Какие кадры брать" }),
      scope,
      el("span", { class: "field__hint", text: "кадры, отданные в задание напарнику, исключаются всегда" }),
    ]),
    el("div", { class: "field" }, [
      el("label", {}, ["Баланс: ошибки ← → разнообразие  (α = ", alphaValue, ")"]),
      el("div", { class: "budget__slider" }, [
        el("span", { class: "tiny faint", text: "ошибки" }),
        alpha,
        el("span", { class: "tiny faint", text: "разнообразие" }),
      ]),
    ]),
  ]);

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      handle.close();
      resolve(value);
    };

    const handle = modal({
      title: "Сколько размечаем?",
      wide: true,
      body,
      actions: [
        el("button", { class: "btn", text: "Отмена", onClick: () => finish(null) }),
        el("button", {
          class: "btn btn--primary",
          text: "Подобрать изображения",
          onClick: () =>
            finish({
              budget: Math.max(1, Number(count.value) || 100),
              clusters: clusters.value ? Number(clusters.value) : null,
              alpha: Number(alpha.value),
              split: split.value || null,
              scope: scope.value,
            }),
        }),
      ],
      onClose: () => finish(null),
    });
  });
}

async function newSelection() {
  const params = await budgetDialog();
  if (!params) return;

  try {
    const selection = await runTask(() => project.selection(params), { title: "Подбираем изображения" });
    setSource("selection", false);
    applySelection(selection);
    toast(`Отобрано ${selection.items.length} изображений`, { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
}

function applySelection(selection) {
  const withSubmission = selection.signals?.with_submission;
  state.signals = selection.signals;
  setItems(
    selection.items,
    `${selection.items.length} изображений · ${selection.signals?.clusters ?? "?"} кластеров · ${
      withSubmission ? "с учётом ошибок модели" : "только разнообразие"
    }`
  );
}

function setItems(items, meta) {
  state.items = items;
  state.done = new Set();
  state.index = -1;
  state.saved = 0;
  state.startedAt = null;
  qs("#queue-meta").textContent = meta;

  renderQueue();
  renderProgress();
  if (state.items.length) open(0);
  else qs("#stage-empty").classList.remove("hidden");
}

/* --------------------------------------------------------- queue sources */
const plainItem = (sample, index, reason = "plain", label = "Из проекта") => ({
  name: sample.name,
  cluster: 0,
  interest: 0,
  reason,
  reason_label: label,
  split: sample.split,
  n_gt: sample.boxes || 0,
  n_pred: 0,
  ghost: 0,
  missed: 0,
  rank: index + 1,
});

async function loadSource(source) {
  try {
    if (source === "selection") {
      const { selections } = await project.selections();
      if (!selections.length) {
        toast("Выборки ещё нет — нажмите «Подобрать»", { kind: "error" });
        return false;
      }
      applySelection(await project.selectionDetail(selections[0].file));
      return true;
    }

    if (source === "issues") {
      const report = await project.quality(500);
      const items = report.images.map((row, index) => ({
        ...plainItem({ name: row.name, split: row.split, boxes: row.boxes }, index, "issue", row.issues[0]?.label || "Замечание"),
        interest: Math.min(row.score / 10, 1),
        issues: row.issues,
      }));
      setItems(items, `${items.length} кадров с замечаниями · сверху самые тяжёлые`);
      return true;
    }

    const params = { limit: 500 };
    if (source === "unlabeled") params.labeled = false;
    const { items: samples, total } = await project.samples(params);
    const items = samples.map((sample, index) =>
      plainItem(sample, index, source === "unlabeled" ? "empty" : "plain", source === "unlabeled" ? "Без разметки" : "Из проекта")
    );
    setItems(items, `${items.length} из ${fmt.int(total)} кадров`);
    return true;
  } catch (error) {
    notifyError(error);
    return false;
  }
}

function setSource(source, load = true) {
  state.source = source;
  qsa("#queue-source button").forEach((button) =>
    button.classList.toggle("is-active", button.dataset.source === source)
  );
  if (load) loadSource(source);
}

/* ----------------------------------------------------------------- queue */
function visibleItems() {
  if (state.filter === "todo") return state.items.filter((item) => !state.done.has(item.name));
  if (state.filter === "done") return state.items.filter((item) => state.done.has(item.name));
  return state.items;
}

function renderQueue() {
  const list = clear(qs("#queue-list"));

  visibleItems().forEach((item) => {
    const index = state.items.indexOf(item);
    const tags = [el("span", { class: `badge ${REASON_TONE[item.reason] || "badge"}`, text: item.reason_label })];
    if (item.ghost) tags.push(el("span", { class: "badge badge--red", text: `FP ${item.ghost}` }));
    if (item.missed) tags.push(el("span", { class: "badge badge--amber", text: `FN ${item.missed}` }));

    list.append(
      el(
        "button",
        {
          class: `qitem ${index === state.index ? "is-active" : ""} ${
            state.done.has(item.name) ? "is-done" : ""
          }`,
          onClick: () => open(index),
        },
        [
          el("img", { class: "qitem__thumb", src: project.imageUrl(item.name), loading: "lazy", alt: "" }),
          el("div", { class: "qitem__body" }, [
            el("div", { class: "row", style: { gap: "6px" } }, [
              el("span", { class: "tiny faint", text: `#${item.rank}` }),
              el("span", {
                class: "badge",
                style: { borderColor: slotColor(item.cluster), color: slotColor(item.cluster) },
                text: clusterName(item.cluster),
                title: `кластер k${item.cluster}`,
              }),
            ]),
            el("div", { class: "qitem__name truncate", text: item.name, title: item.name }),
            el("div", { class: "qitem__tags" }, tags),
          ]),
        ]
      )
    );
  });

  if (!list.children.length) {
    list.append(el("div", { class: "tiny faint center", style: { padding: "20px" }, text: "Пусто" }));
  }
}

/**
 * Темп работы. Разметка — это марафон на несколько часов, и единственный
 * честный ответ на «когда закончу» считается по тому, сколько уже сделано.
 */
function renderProgress() {
  const total = state.items.length;
  const done = state.done.size;
  qs("#progress-bar").style.width = `${total ? (done / total) * 100 : 0}%`;
  qs("#progress-text").textContent = `${done} из ${total}`;

  if (!state.startedAt || state.saved < 3) {
    qs("#rate").textContent = "темп: —";
    return;
  }
  const minutes = (Date.now() - state.startedAt) / 60000;
  const perHour = (state.saved / minutes) * 60;
  const left = Math.max(total - done, 0);
  const eta = perHour > 0 ? left / perHour : 0;
  qs("#rate").textContent = `темп: ${perHour.toFixed(0)} кадров/час · осталось ~${
    eta >= 1 ? `${eta.toFixed(1)} ч` : `${Math.round(eta * 60)} мин`
  }`;
}

/* ---------------------------------------------------------------- editor */
async function open(index) {
  const item = state.items[index];
  if (!item) return;

  if (state.index >= 0 && state.index !== index && state.dirty && qs("#autosave").checked) {
    await save({ quiet: true });
  }
  if (state.index >= 0 && state.index !== index) {
    state.clipboard = editor.getBoxes();
  }

  state.index = index;
  state.dirty = false;

  qs("#current-name").textContent = item.name;
  const reason = qs("#current-reason");
  reason.textContent = item.reason_label;
  reason.className = `badge ${REASON_TONE[item.reason] || ""}`;
  qs("#stage-empty").classList.add("hidden");

  try {
    const data = await project.boxes(item.name);
    await editor.load({ imageUrl: project.imageUrl(item.name), gt: data.gt, pred: data.pred });
    editor.draw();

    const status = qs("#current-status");
    status.classList.toggle("hidden", data.status === "new" || data.status === "done");
    status.textContent = data.status === "assigned" ? "в задании" : data.status === "review" ? "спорный" : "";

    renderWhy(item, data);
    renderBoxList();
    renderQueue();
  } catch (error) {
    notifyError(error);
  }
}

function renderWhy(item, data) {
  const node = clear(qs("#why"));
  const rows = [
    ["Интерес", item.interest, "var(--accent)"],
    ["Кластер", clusterName(item.cluster), slotColor(item.cluster)],
    ["GT боксов", data.gt.length, "var(--teal)"],
    ["Предсказаний", data.pred.length, "var(--pink)"],
    ["Уверенных FP", item.ghost, "var(--red)"],
    ["Пропущено GT", item.missed, "var(--amber)"],
  ];

  node.append(
    el("div", { class: "why__bar" }, [
      el("i", { style: { width: `${Math.round((item.interest || 0) * 100)}%` } }),
    ])
  );

  rows.forEach(([label, value, color]) =>
    node.append(
      el("div", { class: "why__row" }, [
        el("i", { class: "dot", style: { color, width: "7px", height: "7px" } }),
        el("span", { class: "muted", text: label }),
        el("b", { text: typeof value === "number" ? fmt.num(value, value < 1 && value > 0 ? 3 : 0) : value }),
      ])
    )
  );

  // Замечания из проверки разметки — прямо здесь, чтобы не гадать, что чинить.
  (item.issues || []).slice(0, 6).forEach((issue) =>
    node.append(
      el("div", { class: `why__issue why__issue--${issue.severity}` }, [
        el("b", { text: issue.label }),
        el("span", { class: "tiny faint", text: issue.box >= 0 ? `бокс №${issue.box + 1} · ${issue.detail}` : issue.detail }),
      ])
    )
  );

  if (data.edited) {
    node.append(el("span", { class: "badge badge--green", text: "уже редактировалось" }));
  }
}

function renderClasses() {
  const node = clear(qs("#classes"));
  const names = classNames.length ? classNames : ["class_0"];

  names.forEach((name, index) => {
    const hotkey = index < 9 ? String(index + 1) : index === 9 ? "0" : "";
    node.append(
      el(
        "button",
        {
          class: `class-chip ${editor.activeClass === index ? "is-active" : ""}`,
          style: { color: editor.activeClass === index ? slotColor(index) : "" },
          title: hotkey ? `клавиша ${hotkey}` : "",
          onClick: () => {
            editor.setClass(index);
            renderClasses();
            renderBoxList();
          },
        },
        [el("i", { style: { background: slotColor(index) } }), `${index} · ${name}`]
      )
    );
  });
}

function renderBoxList() {
  const node = clear(qs("#boxlist"));
  const boxes = editor.getBoxes();

  boxes.forEach((box, index) => {
    node.append(
      el(
        "div",
        {
          class: `boxrow ${editor.selected === index ? "is-active" : ""}`,
          onClick: () => {
            editor.selected = index;
            editor.activeClass = box.cls;
            editor.draw();
            renderBoxList();
            renderClasses();
          },
        },
        [
          el("i", { style: { background: slotColor(box.cls) } }),
          el("span", { text: classNames[box.cls] ? classNames[box.cls].slice(0, 12) : `c${box.cls}` }),
          el("span", { class: "faint", text: `${box.x.toFixed(2)} ${box.y.toFixed(2)}` }),
          el("span", { class: "spacer" }),
          el("span", { class: "faint", text: `${(box.w * 100).toFixed(0)}×${(box.h * 100).toFixed(0)}` }),
        ]
      )
    );
  });

  if (!boxes.length) {
    node.append(el("div", { class: "tiny faint center", style: { padding: "14px" }, text: "Боксов нет" }));
  }
}

/* ------------------------------------------------------------ сохранение */
async function save({ quiet = false, advance = false } = {}) {
  const item = state.items[state.index];
  if (!item) return;

  try {
    await project.saveLabels(item.name, {
      boxes: editor.getBoxes(),
      write_source: qs("#write-source").checked,
    });
    state.dirty = false;
    if (!state.done.has(item.name)) {
      state.done.add(item.name);
      state.saved += 1;
      state.startedAt = state.startedAt || Date.now();
    }
    renderQueue();
    renderProgress();
    if (!quiet) toast(`${item.name}: сохранено`, { kind: "success", timeout: 1200 });
    if (advance) open(Math.min(state.items.length - 1, state.index + 1));
  } catch (error) {
    notifyError(error);
  }
}

/** Пустой кадр — это тоже разметка: отрицательные примеры модели нужны. */
async function markEmpty() {
  if (!state.items[state.index]) return;
  editor.clearBoxes();
  await save({ quiet: true, advance: true });
  flashHint("Кадр помечен пустым");
}

function copyFromPrevious() {
  if (!state.clipboard || !state.clipboard.length) {
    flashHint("Нечего копировать — сначала откройте кадр с разметкой");
    return;
  }
  editor.setBoxes(state.clipboard);
  renderBoxList();
  flashHint(`Перенесено боксов: ${state.clipboard.length}`);
}

async function flagImage() {
  const item = state.items[state.index];
  if (!item) return;
  try {
    await project.flag(item.name, { status: "review", note: "" });
    qs("#current-status").classList.remove("hidden");
    qs("#current-status").textContent = "спорный";
    toast("Кадр помечен спорным", { kind: "success", timeout: 1600 });
  } catch (error) {
    notifyError(error);
  }
}

/**
 * Перенос разметки на похожие кадры.
 *
 * Камера на стройке стоит месяцами: соседний кадр в пространстве признаков —
 * это чаще всего та же сцена спустя десять минут, где техника не сдвинулась.
 * Список соседей показывается до применения: дёшево отказаться, дорого потом
 * вычищать сотню неверных боксов.
 */
async function propagate() {
  const item = state.items[state.index];
  if (!item) return;

  const boxes = editor.getBoxes();
  if (!boxes.length) {
    toast("На кадре нет боксов — переносить нечего", { kind: "error" });
    return;
  }

  let data;
  try {
    data = await project.neighbors(item.name, 24);
  } catch (error) {
    notifyError(error);
    return;
  }

  const free = data.neighbors.filter((n) => n.status !== "assigned");
  if (!free.length) {
    toast("Похожих свободных кадров не нашлось", { kind: "error" });
    return;
  }

  const picked = new Set(free.slice(0, 5).map((n) => n.name));
  const grid = el(
    "div",
    { class: "propagate" },
    free.map((neighbour) => {
      const card = el(
        "button",
        {
          class: `propagate__item ${picked.has(neighbour.name) ? "is-picked" : ""}`,
          onClick: () => {
            if (picked.has(neighbour.name)) picked.delete(neighbour.name);
            else picked.add(neighbour.name);
            card.classList.toggle("is-picked", picked.has(neighbour.name));
            counter.textContent = `выбрано ${picked.size}`;
          },
        },
        [
          el("img", { src: project.imageUrl(neighbour.name), loading: "lazy", alt: "" }),
          el("span", { class: "tiny truncate", text: neighbour.name, title: neighbour.name }),
          el("span", { class: "tiny faint", text: `d=${neighbour.distance} · ${neighbour.boxes} боксов` }),
        ]
      );
      return card;
    })
  );

  const counter = el("span", { class: "tiny faint", text: `выбрано ${picked.size}` });
  const body = el("div", { class: "col", style: { gap: "10px" } }, [
    el("p", {
      class: "muted tiny",
      text: `${boxes.length} боксов с «${item.name}» будут записаны на выбранные кадры целиком, поверх того, что там есть.`,
    }),
    counter,
    grid,
  ]);

  const handle = modal({
    title: "Перенести на похожие кадры",
    wide: true,
    body,
    actions: [
      el("button", { class: "btn", text: "Отмена", onClick: () => handle.close() }),
      el("button", {
        class: "btn btn--primary",
        text: "Перенести",
        onClick: async () => {
          if (!picked.size) return;
          try {
            const result = await project.saveLabelsBatch({
              names: [...picked],
              boxes,
              write_source: qs("#write-source").checked,
            });
            handle.close();
            toast(`Разметка перенесена на ${result.images} кадров`, { kind: "success" });
          } catch (error) {
            notifyError(error);
          }
        },
      }),
    ],
  });
}

function helpDialog() {
  const rows = [
    ["←  →", "предыдущий / следующий кадр"],
    ["1…9, 0", "класс объекта (0 — десятый)"],
    ["N", "режим рисования нового бокса"],
    ["V", "режим курсора"],
    ["D", "дубль выделенного бокса"],
    ["Стрелки", "двигать выделенный бокс"],
    ["Shift + стрелки", "менять размер выделенного бокса"],
    ["Del", "удалить выделенный бокс"],
    ["E", "кадр пустой — сохранить и идти дальше"],
    ["Ctrl + D", "скопировать разметку предыдущего кадра"],
    ["R", "пометить кадр спорным"],
    ["S", "сохранить"],
    ["G / P", "показать или скрыть разметку / предсказания"],
    ["F", "вписать изображение"],
    ["Ctrl+Z / Ctrl+Y", "отменить / вернуть"],
    ["Esc", "снять выделение или выйти из рисования"],
  ];

  modal({
    title: "Горячие клавиши",
    body: el(
      "div",
      { class: "hotkeys" },
      rows.map(([key, text]) =>
        el("div", { class: "hotkeys__row" }, [el("kbd", { text: key }), el("span", { text })])
      )
    ),
  });
}

/* ----------------------------------------------------------------- wiring */
qs("#new-selection").addEventListener("click", newSelection);
qs("#save").addEventListener("click", () => save());
qs("#fit").addEventListener("click", () => editor.fit());
qs("#undo").addEventListener("click", () => editor.undo());
qs("#redo").addEventListener("click", () => editor.redo());
qs("#zoom-in").addEventListener("click", () => editor.zoomBy(1.5));
qs("#zoom-out").addEventListener("click", () => editor.zoomBy(1 / 1.5));
qs("#help").addEventListener("click", helpDialog);
qs("#copy-prev").addEventListener("click", copyFromPrevious);
qs("#mark-empty").addEventListener("click", markEmpty);
qs("#propagate").addEventListener("click", propagate);
qs("#flag").addEventListener("click", flagImage);

qsa("#mode button").forEach((button) => {
  button.addEventListener("click", () => editor.setMode(button.dataset.mode));
});
qsa("#queue-source button").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});
qs("#prev").addEventListener("click", () => open(Math.max(0, state.index - 1)));
qs("#next").addEventListener("click", () => open(Math.min(state.items.length - 1, state.index + 1)));
qs("#accept-all").addEventListener("click", () => editor.acceptAllPredictions());
qs("#clear-boxes").addEventListener("click", () => editor.clearBoxes());

function setLayer(layer, visible) {
  if (layer === "gt") editor.showGroundTruth = visible;
  else editor.showPredictions = visible;

  qs(`#layers button[data-layer="${layer}"]`).classList.toggle("is-active", visible);
  editor.invalidate();
  flashHint(`${layer === "gt" ? "Разметка" : "Предсказания"}: ${visible ? "показаны" : "скрыты"}`);
}

const layerVisible = (layer) => (layer === "gt" ? editor.showGroundTruth : editor.showPredictions);

qsa("#layers button").forEach((button) => {
  button.addEventListener("click", () => {
    const layer = button.dataset.layer;
    setLayer(layer, !layerVisible(layer));
  });
});

qs("#conf-range").addEventListener("input", (event) => {
  editor.confMin = Number(event.target.value);
  qs("#conf-value").textContent = editor.confMin.toFixed(2);
  editor.draw();
});

qsa("#queue-filter button").forEach((button) => {
  button.addEventListener("click", () => {
    qsa("#queue-filter button").forEach((other) => other.classList.remove("is-active"));
    button.classList.add("is-active");
    state.filter = button.dataset.filter;
    renderQueue();
  });
});

const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  const key = event.key.toLowerCase();

  if (event.ctrlKey || event.metaKey) {
    if (key === "z" || key === "я") {
      event.preventDefault();
      if (event.shiftKey) editor.redo();
      else editor.undo();
    } else if (key === "y" || key === "н") {
      event.preventDefault();
      editor.redo();
    } else if (key === "d" || key === "в") {
      event.preventDefault();
      copyFromPrevious();
    } else if (key === "s" || key === "ы") {
      event.preventDefault();
      save();
    }
    return;
  }

  // Стрелки при выделенном боксе двигают его, иначе листают очередь.
  if (ARROWS[event.key]) {
    const [dx, dy] = ARROWS[event.key];
    if (editor.selected >= 0) {
      event.preventDefault();
      editor.nudgeSelected(dx, dy, { resize: event.shiftKey });
      renderBoxList();
    } else if (event.key === "ArrowRight") {
      open(Math.min(state.items.length - 1, state.index + 1));
    } else if (event.key === "ArrowLeft") {
      open(Math.max(0, state.index - 1));
    }
    return;
  }

  if (event.key === "Delete" || event.key === "Backspace") editor.deleteSelected();
  else if (key === "s" || key === "ы") save();
  else if (key === "n" || key === "т") editor.setMode(editor.mode === "draw" ? "cursor" : "draw");
  else if (key === "v" || key === "м") editor.setMode("cursor");
  else if (key === "f" || key === "а") editor.fit();
  else if (key === "d" || key === "в") editor.duplicateSelected();
  else if (key === "e" || key === "у") markEmpty();
  else if (key === "r" || key === "к") flagImage();
  else if (key === "g" || key === "п") setLayer("gt", !editor.showGroundTruth);
  else if (key === "p" || key === "з") setLayer("pred", !editor.showPredictions);
  else if (event.key === "?" || event.key === "/") helpDialog();
  else if (event.key === "Escape") {
    if (editor.mode === "draw") editor.setMode("cursor");
    else {
      editor.selected = -1;
      editor.invalidate();
      renderBoxList();
    }
  } else if (/^[0-9]$/.test(event.key)) {
    editor.setClass(event.key === "0" ? 9 : Number(event.key) - 1);
    renderClasses();
    renderBoxList();
  }
});

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty || qs("#autosave").checked) return;
  event.preventDefault();
  event.returnValue = "";
});

async function boot() {
  renderClasses();
  renderBoxList();
  renderProgress();

  try {
    const { selections } = await project.selections();
    if (selections.length) {
      applySelection(await project.selectionDetail(selections[0].file));
      return;
    }
  } catch (error) {
    notifyError(error);
  }

  // Выборки ещё нет — показываем то, что точно есть: кадры без разметки.
  setSource("unlabeled");
}

boot();
