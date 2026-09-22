import { fsApi } from "../core/api.js";
import { clear, el, fmt, icon } from "../core/dom.js";
import { modal, notifyError } from "../core/ui.js";

/**
 * Filesystem browser. `mode: "dir"` returns folder paths, `mode: "file"` returns
 * file paths filtered by `pattern`. Multiple selection fills a basket first.
 */
const globMatcher = (pattern) =>
  new RegExp(
    `^${pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".")}$`,
    "i"
  );

export function browse({ mode = "dir", multiple = false, pattern = "*.pt", title = "Выбор папки" } = {}) {
  return new Promise((resolve) => {
    const basket = new Map();
    let current = "";

    const pathInput = el("input", {
      type: "text",
      placeholder: "Путь к папке — можно вставить вручную",
      onKeydown: (event) => {
        if (event.key === "Enter") load(event.target.value.trim());
      },
    });

    const list = el("div", { class: "picker__list" });
    const basketNode = el("div", { class: "picker__basket" });
    const searchRow = el("div", { class: "row", style: { display: mode === "file" ? "flex" : "none" } }, [
      el("input", { type: "text", value: pattern, id: "picker-pattern", placeholder: "*.pt" }),
      el("button", {
        class: "btn btn--sm",
        html: `${icon("search")} Найти рекурсивно`,
        onClick: () => search(),
      }),
    ]);

    const confirm = el("button", {
      class: "btn btn--primary",
      text: multiple ? "Добавить выбранное" : "Выбрать",
      onClick: () => finish(),
    });

    const handle = modal({
      title,
      wide: true,
      body: el("div", { class: "picker" }, [
        el("div", { class: "picker__bar" }, [
          el("button", { class: "btn btn--sm", html: icon("arrow", "flip"), title: "Вверх", onClick: up }),
          pathInput,
        ]),
        searchRow,
        list,
        basketNode,
      ]),
      actions: [
        el("button", { class: "btn", text: "Отмена", onClick: () => finish(true) }),
        confirm,
      ],
      onClose: () => resolve(null),
    });

    function finish(cancelled = false) {
      const value = cancelled
        ? null
        : multiple
          ? Array.from(basket.keys())
          : mode === "dir"
            ? current
            : (basket.keys().next().value ?? null);
      handle.close();
      resolve(value);
    }

    function renderBasket() {
      clear(basketNode);
      if (!multiple || !basket.size) return;
      basketNode.append(
        el("span", { class: "tiny faint", text: `Выбрано: ${basket.size}` }),
        ...Array.from(basket.entries()).map(([path, label]) =>
          el("span", { class: "badge badge--accent" }, [
            el("span", { text: label, title: path }),
            el("button", {
              class: "picker__chip-x",
              html: "×",
              onClick: () => {
                basket.delete(path);
                renderBasket();
                markRows();
              },
            }),
          ])
        )
      );
    }

    function markRows() {
      list.querySelectorAll("[data-path]").forEach((row) => {
        row.classList.toggle("is-picked", basket.has(row.dataset.path));
      });
    }

    function pick(path, label) {
      if (multiple) {
        if (basket.has(path)) basket.delete(path);
        else basket.set(path, label);
        renderBasket();
        markRows();
      } else {
        basket.clear();
        basket.set(path, label);
        finish();
      }
    }

    function countLabel(value, cap = 4000) {
      if (value === null || value === undefined) return null;
      return value >= cap ? `${fmt.int(cap)}+` : fmt.int(value);
    }

    function rowFor(entry, isDir) {
      const meta = [];
      if (isDir) {
        const images = countLabel(entry.images);
        if (images && entry.images) {
          meta.push(el("span", { class: "badge badge--green", text: `${images} изображений` }));
        }
        if (entry.labels) {
          meta.push(el("span", { class: "badge badge--violet", text: `${countLabel(entry.labels)} .txt` }));
        }
        if (!entry.images && entry.subdirs) {
          meta.push(el("span", { class: "tiny faint", text: `${entry.subdirs} папок` }));
        }
      } else if (entry.size !== undefined) {
        meta.push(el("span", { class: "tiny faint", text: fmt.bytes(entry.size) }));
      }

      return el("div", { class: "picker__row", dataset: { path: entry.path } }, [
        el("span", { class: "picker__icon", html: icon(isDir ? "folder" : "box") }),
        el("button", {
          class: "picker__name truncate",
          text: entry.name,
          title: entry.path,
          onClick: () => (isDir && mode === "dir" ? load(entry.path) : pick(entry.path, entry.name)),
        }),
        ...meta,
        isDir && mode === "dir"
          ? el("button", {
              class: "btn btn--sm",
              text: multiple ? (basket.has(entry.path) ? "Убрать" : "Выбрать") : "Открыть",
              onClick: () =>
                multiple ? pick(entry.path, entry.name) : load(entry.path),
            })
          : null,
        isDir && mode === "file"
          ? el("button", { class: "btn btn--sm", text: "Открыть", onClick: () => load(entry.path) })
          : null,
      ]);
    }

    async function load(path) {
      try {
        const data = path
          ? await fsApi.list(path, mode === "file")
          : { dirs: [], files: [], path: "", parent: null };
        current = data.path || "";
        pathInput.value = current;
        clear(list);

        if (!current) {
          const roots = await fsApi.roots();
          list.append(
            el("div", { class: "picker__hint", text: "Диски и быстрые ссылки" }),
            ...roots.roots.map((root) =>
              el("div", { class: "picker__row" }, [
                el("span", { class: "picker__icon", html: icon("folder") }),
                el("button", { class: "picker__name", text: root.path, onClick: () => load(root.path) }),
              ])
            )
          );
          return;
        }

        if (mode === "dir" && current) {
          const deep = data.deep_images ?? data.images;
          const found = deep
            ? `${fmt.int(deep)}${data.deep_truncated ? "+" : ""} изображений${
                data.images ? "" : " во вложенных папках"
              }`
            : "изображений не найдено";

          list.append(
            el("div", { class: "picker__current" }, [
              el("div", { style: { minWidth: 0 } }, [
                el("div", { class: "truncate mono", text: current, title: current }),
                el("div", { class: "tiny faint", text: found }),
              ]),
              el("div", { class: "spacer" }),
              el("button", {
                class: "btn btn--sm btn--primary",
                text: multiple ? "Добавить эту папку" : "Выбрать эту папку",
                onClick: () => pick(current, current.split(/[\\/]/).filter(Boolean).pop() || current),
              }),
            ])
          );

          if (data.preview?.length) {
            list.append(
              el(
                "div",
                { class: "picker__preview" },
                data.preview.map((image) =>
                  el("img", {
                    src: `/api/fs/thumb?path=${encodeURIComponent(image)}`,
                    loading: "lazy",
                    alt: "",
                    title: image,
                  })
                )
              )
            );
          }
        }

        data.dirs.forEach((entry) => list.append(rowFor(entry, true)));

        if (mode === "file") {
          const active = document.getElementById("picker-pattern")?.value || pattern;
          const matcher = globMatcher(active);
          data.files
            .filter((entry) => matcher.test(entry.name))
            .forEach((entry) => list.append(rowFor(entry, false)));
        }

        if (!data.dirs.length && !data.files.length) {
          list.append(el("div", { class: "picker__hint", text: "Пусто" }));
        }
        markRows();
      } catch (error) {
        notifyError(error);
      }
    }

    async function search() {
      if (!current) return;
      const value = document.getElementById("picker-pattern").value || pattern;
      try {
        const data = await fsApi.search(current, value);
        clear(list);
        list.append(
          el("div", { class: "picker__hint", text: `Найдено ${data.matches.length} файлов по «${value}»` })
        );
        data.matches.forEach((entry) => list.append(rowFor(entry, false)));
        markRows();
      } catch (error) {
        notifyError(error);
      }
    }

    function up() {
      if (!current) return;
      const parts = current.split(/[\\/]/).filter(Boolean);
      parts.pop();
      const next = parts.length > 1 ? parts.join("\\") : parts.length === 1 ? `${parts[0]}\\` : "";
      load(next);
    }

    load("");
  });
}

export const pickFolders = (title = "Выберите папки с данными") =>
  browse({ mode: "dir", multiple: true, title });

export const pickFolder = (title = "Выберите папку") => browse({ mode: "dir", multiple: false, title });

export const pickFiles = (pattern = "*.pt", title = "Выберите файлы") =>
  browse({ mode: "file", multiple: true, pattern, title });

export const pickFile = (pattern = "*.csv", title = "Выберите файл") =>
  browse({ mode: "file", multiple: false, pattern, title });
