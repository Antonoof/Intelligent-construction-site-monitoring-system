import { api, projectApi } from "../core/api.js";
import { clear, el, fmt, icon, qs } from "../core/dom.js";
import { emptyState, notifyError, setBusy, toast } from "../core/ui.js";
import { classEditor } from "../components/class-editor.js";
import { pickFile } from "../components/file-picker.js";
import { skipNotes, submissionDialog } from "../components/submission.js";

const projectId = qs("[data-project]").dataset.project;
const project = projectApi(projectId);
const artifactsNode = qs("#artifacts");

const classes = classEditor(qs("#classes"), {
  onChange: (names) => {
    qs("#class-count").textContent = String(names.length);
  },
});

function artifact({ name, meta, href = null, iconName = "box" }) {
  const body = el("div", { class: "artifact__body" }, [
    el("div", { class: "artifact__name truncate", text: name, title: name }),
    el("div", { class: "artifact__meta", text: meta }),
  ]);

  return el("div", { class: "artifact" }, [
    el("div", { class: "artifact__icon", html: icon(iconName) }),
    body,
    href ? el("a", { class: "btn btn--sm", href, text: "Скачать", download: true }) : null,
  ]);
}

async function loadArtifacts() {
  clear(artifactsNode);
  try {
    const [{ project: detail }, { models }] = await Promise.all([project.detail(), project.models()]);
    classes.setNames(detail.names || []);
    const items = [];

    items.push(
      artifact({ name: "data.yaml", meta: "конфигурация датасета", iconName: "layers" }),
      artifact({ name: "train.yaml", meta: "конфигурация обучения", iconName: "cpu" })
    );

    detail.exports.forEach((file) =>
      items.push(
        artifact({
          name: file,
          meta: "готовый бандл",
          href: `/api/projects/${projectId}/download/${file}`,
          iconName: "download",
        })
      )
    );

    models.forEach((model) =>
      items.push(artifact({ name: model.name, meta: fmt.bytes(model.size), iconName: "box" }))
    );

    if (detail.submission) {
      items.push(
        artifact({
          name: detail.submission.file,
          meta: `${fmt.int(detail.submission.boxes)} боксов · ${detail.submission.layout}`,
          iconName: "chart",
        })
      );
    }

    if (detail.last_selection) {
      items.push(
        artifact({
          name: detail.last_selection.file,
          meta: `выборка на ${detail.last_selection.budget} изображений`,
          iconName: "wand",
        })
      );
    }

    items.forEach((node) => artifactsNode.append(node));
  } catch (error) {
    artifactsNode.append(emptyState("Не удалось получить артефакты"));
    notifyError(error);
  }
}

qs("#load-results").addEventListener("click", async () => {
  const info = await submissionDialog(project);
  if (!info) return;

  const stat = qs("#submission-stat");
  const notes = skipNotes(info.skipped);
  stat.querySelector(".stat__value").textContent = "есть";
  stat.querySelector(".stat__sub").textContent = notes.length
    ? `${fmt.int(info.boxes)} боксов · пропущено ${notes.join(", ")}`
    : `${fmt.int(info.boxes)} боксов · ${info.layout}`;
  qs("#submission-meta").textContent = info.file;
  loadArtifacts();
});

qs("#refresh-artifacts").addEventListener("click", loadArtifacts);

qs("#save-classes").addEventListener("click", async (event) => {
  setBusy(event.target, true, "Сохраняем…");
  try {
    const { names } = await api.put(`/api/projects/${projectId}/classes`, { names: classes.getNames() });
    classes.setNames(names);
    toast("Классы обновлены в data.yaml", { kind: "success" });
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(event.target, false);
  }
});

qs("#load-yaml").addEventListener("click", async () => {
  const path = await pickFile("*.y*ml", "Выберите data.yaml с именами классов");
  if (!path) return;
  try {
    const data = await api.post("/api/fs/yaml", { path });
    if (!data.names.length) {
      toast("В файле нет ключа names", { kind: "error" });
      return;
    }
    classes.setNames(data.names);
    toast(`Загружено классов: ${data.names.length} — нажмите «Сохранить»`, { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
});

loadArtifacts();
