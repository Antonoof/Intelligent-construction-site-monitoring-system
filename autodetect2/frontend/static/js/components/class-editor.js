import { NEUTRAL, clear, el, icon, slotColor } from "../core/dom.js";

/**
 * Editable list of class names. Order is the class id, so rows carry their index
 * and deleting a row renumbers everything below it.
 */
export function classEditor(host, { names = [], onChange = null } = {}) {
  let items = [...names];

  const emit = () => {
    if (onChange) onChange(getNames());
  };

  function getNames() {
    return items.map((name, index) => (name || "").trim() || `class_${index}`);
  }

  function render() {
    const node = clear(host);
    node.classList.add("classes-editor");

    items.forEach((name, index) => {
      const input = el("input", {
        type: "text",
        value: name,
        placeholder: `class_${index}`,
        onInput: (event) => {
          items[index] = event.target.value;
          emit();
        },
      });

      // Slots past the eighth fall back to the neutral chip, which needs light ink.
      const fill = slotColor(index);
      node.append(
        el("div", { class: "class-row" }, [
          el("span", {
            class: "class-row__id",
            style: { background: fill, color: fill === NEUTRAL ? "var(--text)" : "#0a0e14" },
            text: String(index),
          }),
          input,
          el("button", {
            class: "icon-btn",
            html: icon("trash"),
            title: "Удалить класс",
            onClick: () => {
              items.splice(index, 1);
              render();
              emit();
            },
          }),
        ])
      );
    });

    if (!items.length) {
      node.append(
        el("p", {
          class: "tiny faint",
          text: "Классы не заданы — имена возьмутся из разметки (class_0, class_1, …). Загрузите data.yaml или добавьте их вручную.",
        })
      );
    }

    node.append(
      el("button", {
        class: "btn btn--sm class-add",
        html: `${icon("plus")} Добавить класс`,
        onClick: () => {
          items.push("");
          render();
          emit();
        },
      })
    );
  }

  render();

  return {
    getNames,
    get count() {
      return items.length;
    },
    setNames(next) {
      items = [...next];
      render();
      emit();
    },
  };
}
