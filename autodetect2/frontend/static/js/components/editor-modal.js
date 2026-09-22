import { clear, el, slotColor } from "../core/dom.js";
import { modal, notifyError, toast } from "../core/ui.js";
import { BoxEditor } from "./box-editor.js";

/**
 * Full-size editor over the current page.
 *
 * Opened from the map so a suspicious point can be fixed on the spot: leaving the
 * map to annotate and coming back loses the zoom, the mode and the place you were
 * looking at. Closing with unsaved work asks first.
 */
export function openEditor(project, name, { onSaved = null, meta = "" } = {}) {
  const canvas = el("canvas");
  const boxList = el("div", { class: "editor-modal__boxes scroll-y" });
  const classRow = el("div", { class: "classes" });
  const counter = el("span", { class: "tiny faint" });

  let dirty = false;
  let classNames = [];
  let closing = false;

  const editor = new BoxEditor(canvas, {
    onChange: () => {
      dirty = true;
      render();
    },
    onSelect: () => render(),
    onHistory: (canUndo, canRedo) => {
      undoButton.disabled = !canUndo;
      redoButton.disabled = !canRedo;
    },
  });

  const undoButton = el("button", { title: "Отменить (Ctrl+Z)", disabled: true, html: "↶" });
  const redoButton = el("button", { title: "Вернуть (Ctrl+Y)", disabled: true, html: "↷" });
  undoButton.addEventListener("click", () => editor.undo());
  redoButton.addEventListener("click", () => editor.redo());

  const layerButton = (layer, label) => {
    const button = el("button", { class: "is-active", text: label });
    button.addEventListener("click", () => {
      const visible = !(layer === "gt" ? editor.showGroundTruth : editor.showPredictions);
      if (layer === "gt") editor.showGroundTruth = visible;
      else editor.showPredictions = visible;
      button.classList.toggle("is-active", visible);
      editor.invalidate();
    });
    return button;
  };

  const modeButton = (mode, label, title) => {
    const button = el("button", { class: mode === "cursor" ? "is-active" : "", text: label, title });
    button.addEventListener("click", () => editor.setMode(mode));
    return button;
  };
  const cursorButton = modeButton("cursor", "Курсор", "V");
  const drawButton = modeButton("draw", "Рисовать", "N");
  editor.onMode = (mode) => {
    cursorButton.classList.toggle("is-active", mode === "cursor");
    drawButton.classList.toggle("is-active", mode === "draw");
  };

  const confValue = el("b", { text: "0.25" });
  const confRange = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: "0.25" });
  confRange.addEventListener("input", () => {
    editor.confMin = Number(confRange.value);
    confValue.textContent = editor.confMin.toFixed(2);
    editor.invalidate();
  });

  const saveButton = el("button", { class: "btn btn--sm btn--primary", text: "Сохранить" });
  saveButton.addEventListener("click", () => save());

  function render() {
    const boxes = editor.getBoxes();
    counter.textContent = `${boxes.length} боксов${dirty ? " · есть изменения" : ""}`;

    clear(boxList);
    boxes.forEach((box, index) => {
      boxList.append(
        el(
          "div",
          {
            class: `boxrow ${editor.selected === index ? "is-active" : ""}`,
            onClick: () => {
              editor.selected = index;
              editor.activeClass = box.cls;
              editor.invalidate();
              render();
            },
          },
          [
            el("i", { style: { background: slotColor(box.cls) } }),
            el("span", { text: classNames[box.cls] || `class_${box.cls}` }),
            el("span", { class: "spacer" }),
            el("span", { class: "faint", text: `${(box.w * 100).toFixed(0)}×${(box.h * 100).toFixed(0)}` }),
          ]
        )
      );
    });

    clear(classRow);
    (classNames.length ? classNames : ["class_0"]).forEach((label, index) => {
      classRow.append(
        el(
          "button",
          {
            class: `class-chip ${editor.activeClass === index ? "is-active" : ""}`,
            style: { color: editor.activeClass === index ? slotColor(index) : "" },
            onClick: () => {
              editor.setClass(index);
              render();
            },
          },
          [el("i", { style: { background: slotColor(index) } }), `${index} · ${label}`]
        )
      );
    });
  }

  async function save() {
    try {
      await project.saveLabels(name, { boxes: editor.getBoxes(), write_source: false });
      dirty = false;
      render();
      toast(`${name}: сохранено`, { kind: "success", timeout: 1800 });
      if (onSaved) onSaved(editor.getBoxes());
      return true;
    } catch (error) {
      notifyError(error);
      return false;
    }
  }

  /** Three ways out, because "close" must not silently mean "discard". */
  function askToSave() {
    return new Promise((resolve) => {
      const finish = (value) => {
        ask.close();
        resolve(value);
      };
      const ask = modal({
        title: "Сохранить изменения?",
        body: el("p", { class: "muted", text: `Разметка ${name} была изменена.` }),
        actions: [
          el("button", { class: "btn", text: "Отмена", onClick: () => finish("cancel") }),
          el("button", { class: "btn btn--danger", text: "Не сохранять", onClick: () => finish("discard") }),
          el("button", { class: "btn btn--primary", text: "Сохранить", onClick: () => finish("save") }),
        ],
        onClose: () => resolve("cancel"),
      });
    });
  }

  const body = el("div", { class: "editor-modal" }, [
    el("div", { class: "editor-modal__bar" }, [
      el("div", { class: "btn-group" }, [undoButton, redoButton]),
      el("div", { class: "btn-group" }, [cursorButton, drawButton]),
      el("div", { class: "btn-group" }, [layerButton("gt", "GT"), layerButton("pred", "Предсказания")]),
      el("div", { class: "row", style: { gap: "8px" } }, [
        el("span", { class: "tiny faint" }, ["conf ≥ ", confValue]),
        confRange,
      ]),
      el("div", { class: "spacer" }),
      el("button", { class: "btn btn--sm", text: "Вписать", onClick: () => editor.fit() }),
      saveButton,
    ]),
    el("div", { class: "editor-modal__stage" }, [canvas]),
    el("div", { class: "editor-modal__side" }, [
      el("div", { class: "card__title", text: "Классы" }),
      classRow,
      el("div", { class: "card__title", style: { marginTop: "10px" }, text: "Боксы" }),
      boxList,
    ]),
    el("div", { class: "editor-modal__foot" }, [
      counter,
      el("div", { class: "spacer" }),
      el("span", {
        class: "tiny faint",
        text: "ЛКМ — двигать · по боксу — тащить · N — рисовать · двойной клик по пунктиру — принять · Del — удалить",
      }),
    ]),
  ]);

  const handle = modal({
    title: name,
    wide: true,
    body,
    beforeClose: async () => {
      if (closing || !dirty) return true;
      const answer = await askToSave();
      if (answer === "cancel") return false;
      if (answer === "save" && !(await save())) return false;
      return true;
    },
  });

  handle.root.querySelector(".modal").classList.add("modal--editor");
  if (meta) {
    handle.root.querySelector(".modal__head h2").after(el("span", { class: "badge", text: meta }));
  }

  const onKey = (event) => {
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
      } else if (key === "s" || key === "ы") {
        event.preventDefault();
        save();
      }
      return;
    }

    if (event.key === "Delete" || event.key === "Backspace") editor.deleteSelected();
    else if (key === "n" || key === "т") editor.setMode(editor.mode === "draw" ? "cursor" : "draw");
    else if (key === "v" || key === "м") editor.setMode("cursor");
    else if (key === "f" || key === "а") editor.fit();
    else if (key === "g" || key === "п") {
      editor.showGroundTruth = !editor.showGroundTruth;
      editor.invalidate();
    } else if (key === "p" || key === "з") {
      editor.showPredictions = !editor.showPredictions;
      editor.invalidate();
    } else if (/^[1-9]$/.test(event.key)) {
      editor.setClass(Number(event.key) - 1);
      render();
    }
  };
  document.addEventListener("keydown", onKey);

  const close = handle.close;
  handle.close = async (force = false) => {
    const done = await close(force);
    if (!document.body.contains(handle.root)) document.removeEventListener("keydown", onKey);
    return done;
  };

  (async () => {
    try {
      const data = await project.boxes(name);
      classNames = data.classes || [];
      await editor.load({ imageUrl: project.imageUrl(name), gt: data.gt, pred: data.pred });
      dirty = false;
      render();
    } catch (error) {
      notifyError(error);
      closing = true;
      handle.close(true);
    }
  })();

  return handle;
}
