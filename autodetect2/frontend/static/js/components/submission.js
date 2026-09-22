import { el, fmt, icon } from "../core/dom.js";
import { modal, notifyError, toast } from "../core/ui.js";
import { pickFile } from "./file-picker.js";

const SKIP_LABELS = {
  ground_truth: "строк разметки (source=gt)",
  no_detection: "строк «ничего не найдено»",
  malformed: "битых строк",
};

/** Explains which rows the parser dropped, so the numbers are never a surprise. */
export function skipNotes(skipped) {
  return Object.entries(skipped || {})
    .filter(([, count]) => count)
    .map(([key, count]) => `${fmt.int(count)} ${SKIP_LABELS[key] || key}`);
}

/**
 * Dialog for attaching submission.csv — drag & drop, file input or a path on disk.
 * Resolves with the registered submission metadata, or null when cancelled.
 */
export function submissionDialog(project) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      handle.close();
      resolve(value);
    };

    const input = el("input", { type: "file", accept: ".csv,.txt", class: "hidden" });
    const status = el("div", { class: "tiny muted" });

    const drop = el("div", { class: "upload-drop" }, [
      el("div", { html: icon("upload") }),
      el("div", {}, [
        el("b", { text: "Перетащите submission.csv сюда" }),
        el("div", { class: "tiny faint", text: "или выберите файл вручную" }),
      ]),
      el("div", { class: "row" }, [
        el("button", { class: "btn btn--sm", text: "Выбрать файл", onClick: () => input.click() }),
        el("button", {
          class: "btn btn--sm",
          text: "Указать путь на диске",
          onClick: async () => {
            const path = await pickFile("*.csv", "Выберите submission.csv");
            if (path) send(() => project.linkSubmission(path));
          },
        }),
      ]),
      input,
    ]);

    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
      const file = event.dataTransfer.files[0];
      if (file) send(() => project.uploadSubmission(file));
    });

    input.addEventListener("change", () => {
      if (input.files[0]) send(() => project.uploadSubmission(input.files[0]));
    });

    async function send(action) {
      status.textContent = "Разбираем файл…";
      try {
        const data = await action();
        const info = data.submission;
        toast(`Загружено ${fmt.int(info.boxes)} боксов на ${fmt.int(info.images)} изображений`, {
          kind: "success",
        });

        const notes = skipNotes(info.skipped);
        if (notes.length) {
          toast(notes.join(" · "), { title: "Пропущено", kind: "info", timeout: 9000 });
        }
        if (data.unmatched?.length) {
          toast(`Не сопоставлено имён: ${data.unmatched.length} (например ${data.unmatched[0]})`, {
            kind: "error",
            timeout: 8000,
          });
        }
        finish(info);
      } catch (error) {
        status.textContent = "";
        notifyError(error);
      }
    }

    const handle = modal({
      title: "Загрузить результаты",
      body: el("div", { class: "col" }, [
        el("p", { class: "muted tiny" }, [
          "Строки с ",
          el("code", { text: "source=gt" }),
          " и без координат отбрасываются — считаются только предсказания. ",
          "Поддерживаются форматы: ",
          el("code", { text: "image_id,class_id,confidence,x_center,y_center,width,height" }),
          ", xyxy-колонки и упакованная колонка ",
          el("code", { text: "prediction" }),
          ".",
        ]),
        drop,
        status,
      ]),
      actions: [el("button", { class: "btn", text: "Закрыть", onClick: () => finish(null) })],
      onClose: () => finish(null),
    });
  });
}
