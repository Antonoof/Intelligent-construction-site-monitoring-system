import { api, fsApi } from "../core/api.js";
import { clear, el, fmt, icon, qs, qsa } from "../core/dom.js";
import { notifyError, runTask, toast } from "../core/ui.js";
import { classEditor } from "../components/class-editor.js";
import { pickFile, pickFolders } from "../components/file-picker.js";

const sources = [];
const listNode = qs("#sources");
const summaryNode = qs("#summary");
const createButton = qs("#create");
const nameInput = qs("#project-name");

// Классы проекта подставляются сразу: править готовый список быстрее, чем
// набирать десять имён руками, а порядок тут — это id класса.
const defaultClasses = JSON.parse(qs("[data-default-classes]")?.dataset.defaultClasses || "[]");

const classes = classEditor(qs("#classes"), {
  onChange: (names) => {
    qs("#class-count").textContent = String(names.length);
  },
});

if (defaultClasses.length) classes.setNames(defaultClasses);

function setStep(step) {
  qsa(".step").forEach((node) => {
    const index = Number(node.dataset.step);
    node.classList.toggle("is-active", index === step);
    node.classList.toggle("is-done", index < step);
  });
}

function suggestName() {
  if (nameInput.value.trim() || !sources.length) return;
  const parts = sources[0].path.split(/[\\/]/).filter(Boolean);
  const last = parts[parts.length - 1] || "project";
  nameInput.value = ["train", "val", "valid", "test", "images"].includes(last.toLowerCase())
    ? parts[parts.length - 2] || last
    : last;
}

function render() {
  clear(listNode);

  if (!sources.length) {
    listNode.append(
      el("div", { class: "empty" }, [
        el("div", { html: icon("folder"), style: { width: "32px", height: "32px" } }),
        el("p", { text: "Добавьте хотя бы одну папку с train-данными" }),
        el("div", { class: "row" }, [
          el("button", { class: "btn btn--primary", text: "Выбрать папки", onClick: addFolders }),
          el("button", { class: "btn", text: "Загрузить data.yaml", onClick: loadYamlFromDisk }),
        ]),
      ])
    );
  }

  sources.forEach((source, index) => {
    const meta = el("div", { class: "source__meta" });
    if (source.scanning) {
      meta.append(el("span", { class: "faint", text: "сканируем…" }));
    } else {
      const ratio = source.pairing?.ratio ?? 0;
      const paired =
        ratio >= 0.999
          ? "разметка найдена для всех"
          : ratio > 0
            ? `разметка у ${fmt.pct(ratio, 0)} изображений`
            : "разметка не найдена";

      meta.append(
        el("span", { text: `${fmt.int(source.images)} изображений` }),
        el("span", { class: "faint", text: "·" }),
        el("span", { class: ratio > 0.5 ? "" : "warn", text: paired, title: source.pairing?.label_example || "" })
      );
    }

    listNode.append(
      el("div", { class: "source" }, [
        el("div", { class: "source__icon", html: icon("folder") }),
        el("div", { style: { minWidth: 0 } }, [
          el("div", { class: "source__path truncate", text: source.path, title: source.path }),
          meta,
        ]),
        el("div", { class: "btn-group" }, [
          el("button", {
            class: source.split === "train" ? "is-active" : "",
            text: "train",
            onClick: () => {
              source.split = "train";
              render();
            },
          }),
          el("button", {
            class: source.split === "val" ? "is-active" : "",
            text: "val",
            onClick: () => {
              source.split = "val";
              render();
            },
          }),
        ]),
        el("button", {
          class: "icon-btn",
          html: icon("trash"),
          title: "Убрать папку",
          onClick: () => {
            sources.splice(index, 1);
            render();
          },
        }),
      ])
    );
  });

  const images = sources.reduce((sum, source) => sum + (source.images || 0), 0);
  const hasTrain = sources.some((source) => source.split === "train");
  const hasVal = sources.some((source) => source.split === "val");

  summaryNode.textContent = sources.length
    ? `${sources.length} папок · ${fmt.int(images)} изображений · ${hasVal ? "train + val" : "только train"}`
    : "Папки не выбраны";

  createButton.disabled = !hasTrain;
  setStep(!sources.length ? 1 : hasTrain ? 3 : 2);
  suggestName();
}

/* ---------------------------------------------------------------- sources */
function addSource(entry) {
  if (sources.some((source) => source.path === entry.path)) return null;
  const source = { images: 0, labels: 0, scanning: false, split: "train", ...entry };
  sources.push(source);
  return source;
}

/**
 * A picked folder may hold the images itself or split them across subfolders
 * (`images/train`, `images/val`). The deep scan decides which case it is.
 */
async function inspect(source) {
  try {
    const data = await fsApi.inspect(source.path);

    if (!data.direct_images && data.children.length) {
      sources.splice(sources.indexOf(source), 1);
      data.children.forEach((child) =>
        addSource({
          path: child.path,
          split: child.split,
          images: child.images,
          labels: child.labels,
          pairing: child.pairing,
        })
      );
      return;
    }

    source.images = data.images;
    source.labels = data.labels;
    source.pairing = data.pairing;
    source.split = source.explicitSplit || data.split;
  } catch (error) {
    source.images = 0;
    source.labels = 0;
    notifyError(error);
  } finally {
    source.scanning = false;
    render();
  }
}

async function addFolders() {
  const picked = await pickFolders();
  if (!picked || !picked.length) return;

  const added = picked
    .map((path) => addSource({ path, scanning: true }))
    .filter(Boolean);

  render();
  for (const source of added) await inspect(source);
}

/* ------------------------------------------------------------------- yaml */
function applyYaml(data) {
  if (data.names.length) classes.setNames(data.names);

  let added = 0;
  data.sources.forEach((entry) => {
    const source = addSource({
      path: entry.path,
      split: entry.split,
      images: entry.images,
      labels: entry.labels,
      pairing: entry.pairing,
      explicitSplit: entry.split,
    });
    if (source) added += 1;
  });
  render();

  const note = qs("#yaml-note");
  note.classList.remove("hidden");
  clear(note).append(
    el("div", { html: icon("check"), style: { width: "16px", height: "16px", flex: "none" } }),
    el("div", {}, [
      el("b", { text: `${data.names.length} классов` }),
      ` из ${data.file.split(/[\\/]/).pop()}`,
      added ? `, добавлено папок: ${added}` : "",
      data.missing.length
        ? el("div", { class: "text-amber", text: `не найдены на диске: ${data.missing.join(", ")}` })
        : null,
    ])
  );

  toast(
    added ? `Классы и ${added} папок из data.yaml` : `Загружено классов: ${data.names.length}`,
    { kind: "success" }
  );
}

async function loadYamlFromDisk() {
  const path = await pickFile("*.y*ml", "Выберите data.yaml");
  if (!path) return;
  try {
    applyYaml(await api.post("/api/fs/yaml", { path }));
  } catch (error) {
    notifyError(error);
  }
}

qs("#add-folders").addEventListener("click", addFolders);
qs("#load-yaml").addEventListener("click", loadYamlFromDisk);
qs("#upload-yaml").addEventListener("click", () => qs("#yaml-file").click());

qs("#yaml-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    applyYaml(await api.upload("/api/fs/yaml/upload", file));
  } catch (error) {
    notifyError(error);
  } finally {
    event.target.value = "";
  }
});

createButton.addEventListener("click", async () => {
  const name = nameInput.value.trim() || "project";
  const names = classes.getNames();

  try {
    const project = await runTask(
      () =>
        api.post("/api/projects", {
          name,
          sources: sources.map(({ path, split }) => ({ path, split })),
          classes: names.length ? names : null,
        }),
      { title: "Собираем проект" }
    );

    toast(`Проект «${project.name}» создан`, { kind: "success" });
    window.location.href = `/p/${project.id}`;
  } catch (error) {
    notifyError(error);
  }
});

render();
