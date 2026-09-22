import { projectApi } from "../core/api.js";
import { clear, el, fmt, qs } from "../core/dom.js";
import { notifyError, runTask, setBusy, toast } from "../core/ui.js";
import { pickFiles } from "../components/file-picker.js";

const root = qs("[data-project]");
const project = projectApi(root.dataset.project);

const state = { models: [], enabled: new Set(), params: null };

/* ---------------------------------------------------------------- модели */
function renderModels() {
  const node = clear(qs("#models"));

  if (!state.models.length) {
    node.append(
      el("div", { class: "empty" }, [
        el("p", { text: "Добавьте хотя бы одну модель *.pt" }),
        el("div", { class: "row" }, [
          el("button", { class: "btn btn--sm", text: "Поиск по диску", onClick: findModels }),
          el("button", {
            class: "btn btn--sm",
            text: "Загрузить файл",
            onClick: () => qs("#model-file").click(),
          }),
        ]),
      ])
    );
  }

  state.models.forEach((model) => {
    const toggle = el("input", { type: "checkbox", checked: state.enabled.has(model.name) });
    toggle.addEventListener("change", () => {
      if (toggle.checked) state.enabled.add(model.name);
      else state.enabled.delete(model.name);
      renderModels();
    });

    node.append(
      el("div", { class: `model-row ${state.enabled.has(model.name) ? "" : "is-off"}` }, [
        el("label", { class: "check" }, [toggle]),
        el("span", { class: "badge badge--violet", text: ".pt" }),
        el("div", { style: { minWidth: 0 } }, [
          el("div", { class: "model-row__name truncate", text: model.name, title: model.path }),
          el("div", { class: "model-row__meta", text: fmt.bytes(model.size) }),
        ]),
        el("div", { class: "model-row__weight" }),
        el("button", {
          class: "icon-btn",
          html: "&times;",
          title: "Убрать модель",
          onClick: async () => {
            await project.deleteModel(model.name);
            state.enabled.delete(model.name);
            loadModels();
          },
        }),
      ])
    );
  });

  qs("#model-count").textContent = String(state.enabled.size);
  qs("#run").disabled = state.enabled.size === 0;
}

async function loadModels() {
  try {
    const { models } = await project.models();
    state.models = models;
    models.forEach((model) => state.enabled.add(model.name));
    renderModels();
  } catch (error) {
    notifyError(error);
  }
}

async function findModels() {
  const paths = await pickFiles("*.pt", "Найдите модели YOLO");
  if (!paths || !paths.length) return;

  try {
    for (const path of paths) await project.linkModel(path);
    toast(`Добавлено моделей: ${paths.length}`, { kind: "success" });
    loadModels();
  } catch (error) {
    notifyError(error);
  }
}

qs("#find-models").addEventListener("click", findModels);
qs("#upload-model").addEventListener("click", () => qs("#model-file").click());

qs("#model-file").addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;

  try {
    for (const file of files) await project.uploadModel(file);
    toast(`Загружено моделей: ${files.length}`, { kind: "success" });
    loadModels();
  } catch (error) {
    notifyError(error);
  } finally {
    event.target.value = "";
  }
});

/* --------------------------------------------------------------- подбор */
function renderResult(params) {
  state.params = params;
  const node = clear(qs("#result"));
  const gain = params.score - (params.baseline_score ?? 0);

  node.append(
    el("div", { class: "score" }, [
      el("b", { text: params.score.toFixed(4) }),
      el("span", { text: "F1 на валидации" }),
      el("div", { class: "spacer" }),
      el("span", {
        class: gain >= 0 ? "badge badge--green" : "badge badge--red",
        text: `${gain >= 0 ? "+" : ""}${gain.toFixed(4)} к базовым порогам`,
      }),
    ])
  );

  const rows = [
    ["wbf_iou", params.wbf_iou],
    ["skip_box_thr", params.skip_box_thr],
    ["conf_type", params.conf_type],
    ["sampler", params.sampler],
    ["trials", params.trials],
    ["val изображений", params.val_images],
    ["device", params.device],
    ...Object.entries(params.conf).map(([name, value]) => [`conf · ${name}`, value]),
  ];

  node.append(
    el(
      "table",
      { class: "result-table" },
      rows.map(([label, value]) =>
        el("tr", {}, [el("td", { class: "muted", text: label }), el("td", { text: String(value) })])
      )
    )
  );

  qs("#build").disabled = false;
}

qs("#run").addEventListener("click", async () => {
  const body = {
    models: Array.from(state.enabled),
    k: Number(qs("#k").value) || 100,
    trials: Number(qs("#trials").value) || 40,
    conf_range: [Number(qs("#conf-min").value), Number(qs("#conf-max").value)],
    iou_range: [Number(qs("#iou-min").value), Number(qs("#iou-max").value)],
    skip_range: [Number(qs("#skip-min").value), Number(qs("#skip-max").value)],
    conf_type: qs("#conf-type").value,
    split: qs("#split").value || null,
    device: qs("#device").value,
  };

  try {
    const params = await runTask(() => project.tune(body), { title: "Optuna · подбор WBF" });
    renderResult(params);
    toast(`Готово: F1 = ${params.score.toFixed(4)}`, { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
});

qs("#build").addEventListener("click", async (event) => {
  setBusy(event.target, true, "Собираем…");
  try {
    const result = await project.inferBundle(qs("#include-weights").checked);
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
    toast("AutoFile_infer.zip готов", { kind: "success" });
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(event.target, false);
  }
});

async function boot() {
  await loadModels();
  try {
    renderResult(await project.wbf());
  } catch (error) {
    if (error.status !== 404) notifyError(error);
  }
}

boot();
