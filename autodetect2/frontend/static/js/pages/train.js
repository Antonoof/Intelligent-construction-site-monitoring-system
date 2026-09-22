import { projectApi } from "../core/api.js";
import { clear, el, fmt, qs, qsa } from "../core/dom.js";
import { notifyError, setBusy, toast } from "../core/ui.js";

const root = qs("[data-project]");
const project = projectApi(root.dataset.project);

const GROUPS = [
  {
    key: "train",
    title: "Обучение",
    fields: [
      { name: "epochs", label: "epochs", type: "number", step: 1 },
      { name: "imgsz", label: "imgsz", type: "number", step: 32 },
      { name: "batch", label: "batch (-1 = auto)", type: "number", step: 1 },
      { name: "patience", label: "patience", type: "number", step: 1 },
      { name: "optimizer", label: "optimizer", type: "select", options: ["SGD", "Adam", "AdamW", "auto"] },
      { name: "lr0", label: "lr0", type: "number", step: 0.0001 },
      { name: "lrf", label: "lrf", type: "number", step: 0.0001 },
      { name: "weight_decay", label: "weight_decay", type: "number", step: 0.0001 },
      { name: "cos_lr", label: "cos_lr", type: "bool" },
      { name: "warmup_epochs", label: "warmup_epochs", type: "number", step: 1 },
      { name: "warmup_momentum", label: "warmup_momentum", type: "number", step: 0.01 },
      { name: "close_mosaic", label: "close_mosaic", type: "number", step: 1 },
      { name: "freeze", label: "freeze", type: "number", step: 1 },
      { name: "save_period", label: "save_period", type: "number", step: 1 },
      { name: "conf", label: "conf", type: "number", step: 0.0001 },
      { name: "iou", label: "iou", type: "number", step: 0.01 },
    ],
  },
  {
    key: "augment",
    title: "Аугментации",
    fields: [
      { name: "augment", label: "augment", type: "bool" },
      { name: "hsv_h", label: "hsv_h", type: "number", step: 0.001 },
      { name: "hsv_s", label: "hsv_s", type: "number", step: 0.001 },
      { name: "hsv_v", label: "hsv_v", type: "number", step: 0.001 },
      { name: "fliplr", label: "fliplr", type: "number", step: 0.05 },
      { name: "flipud", label: "flipud", type: "number", step: 0.05 },
      { name: "translate", label: "translate", type: "number", step: 0.001 },
      { name: "scale", label: "scale", type: "number", step: 0.01 },
      { name: "shear", label: "shear", type: "number", step: 0.01 },
      { name: "mixup", label: "mixup", type: "number", step: 0.01 },
      { name: "cutmix", label: "cutmix", type: "number", step: 0.01 },
    ],
  },
  {
    key: "runtime",
    title: "Среда",
    fields: [
      { name: "device", label: "device", type: "text" },
      { name: "workers", label: "workers", type: "number", step: 1 },
      { name: "seed", label: "seed", type: "number", step: 1 },
      { name: "deterministic", label: "deterministic", type: "bool" },
    ],
  },
  {
    key: "predict",
    title: "Сборка submission.csv",
    fields: [
      { name: "conf", label: "conf", type: "number", step: 0.0001 },
      { name: "iou", label: "iou", type: "number", step: 0.01 },
      { name: "max_det", label: "max_det", type: "number", step: 10 },
      { name: "submission", label: "имя файла", type: "text" },
    ],
  },
];

let config = null;
let families = {};
let defaults = null;

/* ------------------------------------------------------------------ форма */
function fieldNode(group, field) {
  const value = config[group.key]?.[field.name];

  if (field.type === "bool") {
    const input = el("input", { type: "checkbox", checked: Boolean(value) });
    input.addEventListener("change", () => update(group.key, field.name, input.checked));
    return el("label", { class: "check" }, [input, field.label]);
  }

  if (field.type === "select") {
    const select = el(
      "select",
      {},
      field.options.map((option) =>
        el("option", { value: option, text: option, selected: option === value })
      )
    );
    select.addEventListener("change", () => update(group.key, field.name, select.value));
    return el("div", { class: "field" }, [el("label", { text: field.label }), select]);
  }

  const input = el("input", {
    type: field.type,
    step: field.step,
    value: value ?? "",
  });
  input.addEventListener("change", () => {
    const next = field.type === "number" ? Number(input.value) : input.value;
    update(group.key, field.name, next);
  });
  return el("div", { class: "field" }, [el("label", { text: field.label }), input]);
}

function renderParams() {
  const node = clear(qs("#params"));
  node.className = "params";

  GROUPS.forEach((group) => {
    node.append(
      el("div", { class: "params__group" }, [
        el("div", { class: "params__title", text: group.title }),
        el(
          "div",
          { class: "params__grid" },
          group.fields.map((field) => fieldNode(group, field))
        ),
      ])
    );
  });
}

function update(section, name, value) {
  config[section] = config[section] || {};
  config[section][name] = value;
  renderYaml();
}

/* ------------------------------------------------------------- архитектура */
function enabledModels() {
  return (config.model.ensemble || []).filter((item) => item.enabled);
}

function renderModels() {
  const ensembleMode = qs("#ensemble-mode").checked;

  qsa(".model").forEach((button) => {
    const family = button.dataset.family;
    const entry = config.model.ensemble.find((item) => String(item.family) === family);
    button.classList.toggle("is-active", Boolean(entry?.enabled));

    button.onclick = () => {
      if (ensembleMode) {
        entry.enabled = !entry.enabled;
        if (!enabledModels().length) entry.enabled = true;
      } else {
        config.model.ensemble.forEach((item) => (item.enabled = String(item.family) === family));
        config.model.family = family;
        config.train.imgsz = families[family]?.imgsz ?? config.train.imgsz;
      }
      syncAll();
    };
  });

  qsa("#sizes button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.size === config.model.size);
    button.onclick = () => {
      config.model.size = button.dataset.size;
      config.model.ensemble.forEach((item) => (item.size = button.dataset.size));
      syncAll();
    };
  });

  const ensembleNode = qs("#ensemble");
  ensembleNode.classList.toggle("hidden", !ensembleMode);
  clear(ensembleNode);

  if (ensembleMode) {
    config.model.ensemble.forEach((item) => {
      const imgsz = el("input", { type: "number", step: 32, value: item.imgsz });
      imgsz.addEventListener("change", () => {
        item.imgsz = Number(imgsz.value);
        renderYaml();
      });

      const toggle = el("input", { type: "checkbox", checked: Boolean(item.enabled) });
      toggle.addEventListener("change", () => {
        item.enabled = toggle.checked;
        if (!enabledModels().length) {
          item.enabled = true;
          toggle.checked = true;
        }
        syncAll();
      });

      ensembleNode.append(
        el("div", { class: "ensemble__row" }, [
          el("label", { class: "check" }, [toggle]),
          el("span", { text: `${families[item.family]?.name || "yolo"}${item.size}` }),
          el("span", { class: "tiny faint", text: "imgsz" }),
          imgsz,
        ])
      );
    });
  }

  const count = enabledModels().length;
  qs("#ensemble-count").textContent = count === 1 ? "1 модель" : `${count} модели`;
}

function renderYaml() {
  const lines = [];
  const walk = (value, indent) => {
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (typeof item === "object") {
          const entries = Object.entries(item);
          lines.push(`${indent}- ${entries[0][0]}: ${entries[0][1]}`);
          entries.slice(1).forEach(([k, v]) => lines.push(`${indent}  ${k}: ${v}`));
        } else {
          lines.push(`${indent}- ${item}`);
        }
      });
      return;
    }
    Object.entries(value).forEach(([key, item]) => {
      if (item && typeof item === "object") {
        lines.push(`${indent}${key}:`);
        walk(item, `${indent}  `);
      } else {
        lines.push(`${indent}${key}: ${item}`);
      }
    });
  };

  walk(config, "");
  qs("#yaml-preview").textContent = lines.join("\n");
}

function syncAll() {
  renderModels();
  renderParams();
  renderYaml();
}

/* --------------------------------------------------------------- действия */
function randomParams() {
  const pick = (min, max, digits = 4) => Number((Math.random() * (max - min) + min).toFixed(digits));
  Object.assign(config.train, {
    lr0: pick(0.0008, 0.0012, 6),
    lrf: pick(0.0008, 0.0012, 6),
    weight_decay: pick(0.00021, 0.00039, 6),
    warmup_epochs: Math.round(pick(2, 5, 0)),
    warmup_momentum: pick(0.95, 0.999, 3),
    close_mosaic: Math.round(pick(4, 7, 0)),
    freeze: Math.round(pick(3, 6, 0)),
    iou: pick(0.25, 0.35, 2),
  });
  Object.assign(config.augment, {
    hsv_h: pick(0, 0.03),
    hsv_s: pick(0, 0.03),
    hsv_v: pick(0, 0.03),
    fliplr: pick(0.4, 0.6, 2),
    translate: pick(0.005, 0.015),
    scale: pick(0.2, 0.3, 3),
    shear: pick(0.03, 0.07, 3),
    mixup: pick(0.03, 0.07, 3),
    cutmix: pick(0.07, 0.13, 3),
  });
  syncAll();
  toast("Параметры пересобраны по логике random_params()", { kind: "info" });
}

qs("#randomize").addEventListener("click", randomParams);

qs("#reset-params").addEventListener("click", () => {
  config = structuredClone(defaults);
  syncAll();
});

qs("#ensemble-mode").addEventListener("change", renderModels);

qs("#save-config").addEventListener("click", async (event) => {
  setBusy(event.target, true, "Сохраняем…");
  try {
    await project.saveTrainConfig(config);
    toast("train.yaml обновлён", { kind: "success" });
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(event.target, false);
  }
});

qs("#build").addEventListener("click", async (event) => {
  setBusy(event.target, true, "Собираем…");
  try {
    const result = await project.trainBundle(config);
    const node = qs("#bundle-result");
    node.classList.remove("hidden");
    clear(node).append(
      el("span", { class: "text-green strong", text: "✓" }),
      el("div", {}, [
        el("div", { text: result.file }),
        el("div", { class: "tiny faint", text: fmt.bytes(result.size) }),
      ]),
      el("div", { class: "spacer" }),
      el("a", { class: "btn btn--sm btn--primary", href: result.url, text: "Скачать", download: true })
    );
    toast("AutoFile.zip готов", { kind: "success" });
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(event.target, false);
  }
});

async function boot() {
  try {
    const data = await project.trainConfig();
    config = data.config;
    families = data.families;
    defaults = structuredClone(data.config);
    syncAll();
  } catch (error) {
    notifyError(error);
  }
}

boot();
