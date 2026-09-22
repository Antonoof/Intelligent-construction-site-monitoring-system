import { clear, el, icon } from "./dom.js";
import { taskApi } from "./api.js";

/* --------------------------------------------------------------- toasts */
export function toast(message, { title = "", kind = "info", timeout = 4200 } = {}) {
  const root = document.getElementById("toast-root");
  if (!root) return;

  const node = el("div", { class: `toast toast--${kind}` }, [
    el("i", { class: "toast__bar" }),
    el("div", {}, [title ? el("strong", { text: title }) : null, el("span", { text: message })]),
  ]);

  root.append(node);
  setTimeout(() => {
    node.style.transition = "opacity 200ms, transform 200ms";
    node.style.opacity = "0";
    node.style.transform = "translateX(20px)";
    setTimeout(() => node.remove(), 220);
  }, timeout);
}

export const notifyError = (error) =>
  toast(error?.message || String(error), { title: "Ошибка", kind: "error", timeout: 7000 });

/* ---------------------------------------------------------------- modal */
export function modal({ title = "", body, actions = [], wide = false, onClose = null, beforeClose = null } = {}) {
  const root = el("div", { class: "modal-root" });
  const bodyNode = el("div", { class: "modal__body" });
  if (body) bodyNode.append(body);

  /**
   * `beforeClose` may veto: it runs while the dialog is still on screen, so a
   * "save your work?" prompt appears over the thing being closed instead of after
   * it has already vanished.
   */
  const close = async (force = false) => {
    if (!force && beforeClose && (await beforeClose()) === false) return;
    root.remove();
    document.removeEventListener("keydown", onKey);
    if (onClose) onClose();
  };

  const onKey = (event) => {
    if (event.key === "Escape") close();
  };

  const footer = actions.length
    ? el("div", { class: "modal__foot" }, [el("div", { class: "spacer" }), ...actions])
    : null;

  root.append(
    el("div", { class: `modal ${wide ? "modal--wide" : ""}` }, [
      el("div", { class: "modal__head" }, [
        el("h2", { text: title }),
        el("div", { class: "spacer" }),
        el("button", { class: "icon-btn", html: icon("close"), onClick: close, title: "Закрыть" }),
      ]),
      bodyNode,
      footer,
    ])
  );

  root.addEventListener("mousedown", (event) => {
    if (event.target === root) close();
  });
  document.addEventListener("keydown", onKey);
  document.body.append(root);

  return { root, body: bodyNode, close };
}

export function confirmDialog(message, { title = "Подтвердите", danger = true } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      handle.close();
      resolve(value);
    };

    const handle = modal({
      title,
      body: el("p", { class: "muted", text: message }),
      actions: [
        el("button", { class: "btn", text: "Отмена", onClick: () => finish(false) }),
        el("button", {
          class: `btn ${danger ? "btn--danger" : "btn--primary"}`,
          text: "Продолжить",
          onClick: () => finish(true),
        }),
      ],
      onClose: () => finish(false),
    });
  });
}

/* ----------------------------------------------------------------- tasks */
export function taskPanel() {
  const bar = el("i");
  const barWrap = el("div", { class: "task__bar" }, [bar]);
  const status = el("div", { class: "row", style: { gap: "8px" } }, [
    el("div", { class: "spinner" }),
    el("span", { class: "muted", text: "Запуск…" }),
  ]);
  const log = el("div", { class: "task__log" });
  const node = el("div", { class: "task" }, [status, barWrap, log]);

  return {
    node,
    update(task) {
      const percent = Math.round((task.progress || 0) * 100);
      bar.style.width = `${percent}%`;
      barWrap.classList.toggle("is-indeterminate", percent === 0 && task.status === "running");
      status.lastChild.textContent = task.message || task.status;
      if (task.status !== "running") {
        status.firstChild.replaceWith(
          el("span", {
            class: task.status === "done" ? "text-green" : "text-red",
            html: icon(task.status === "done" ? "check" : "close"),
            style: { width: "15px", height: "15px", display: "grid" },
          })
        );
      }
      if (task.logs) {
        log.textContent = task.logs.slice(-120).join("\n");
        log.scrollTop = log.scrollHeight;
      }
    },
  };
}

export function pollTask(taskId, { onUpdate = null, interval = 600 } = {}) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const task = await taskApi.get(taskId);
        if (onUpdate) onUpdate(task);
        if (task.status === "running") {
          setTimeout(tick, interval);
        } else if (task.status === "done") {
          resolve(task.result);
        } else {
          reject(new Error(task.error || "Задача завершилась с ошибкой"));
        }
      } catch (error) {
        reject(error);
      }
    };
    tick();
  });
}

/**
 * Runs a background task and shows a live progress dialog.
 * `starter` must resolve to `{ task: {...} }`.
 */
export async function runTask(starter, { title = "Выполняется" } = {}) {
  const panel = taskPanel();
  const handle = modal({ title, body: panel.node });

  try {
    const { task } = await starter();
    const result = await pollTask(task.id, { onUpdate: (state) => panel.update(state) });
    setTimeout(() => handle.close(), 350);
    return result;
  } catch (error) {
    panel.update({ status: "error", message: error.message, progress: 0 });
    handle.root.querySelector(".modal").append(
      el("div", { class: "modal__foot" }, [
        el("div", { class: "spacer" }),
        el("button", { class: "btn", text: "Закрыть", onClick: handle.close }),
      ])
    );
    throw error;
  }
}

/* ------------------------------------------------------------- fragments */
export function emptyState(message, actionNode = null) {
  return el("div", { class: "empty" }, [
    el("div", { html: icon("info"), style: { width: "32px", height: "32px" } }),
    el("p", { text: message }),
    actionNode,
  ]);
}

export function statCard({ label, value, sub = "", meter = null, tone = "" }) {
  return el("div", { class: "stat" }, [
    el("span", { class: "stat__label", text: label }),
    el("span", { class: `stat__value ${tone}`, text: value }),
    sub ? el("span", { class: "stat__sub", text: sub }) : null,
    meter !== null
      ? el("div", { class: "meter" }, [el("i", { style: { width: `${Math.round(meter * 100)}%` } })])
      : null,
  ]);
}

export function setBusy(button, busy, labelWhenBusy = "Работаем…") {
  if (busy) {
    button.dataset.label = button.textContent;
    button.disabled = true;
    button.textContent = labelWhenBusy;
  } else {
    button.disabled = false;
    if (button.dataset.label) button.textContent = button.dataset.label;
  }
}

export { clear, el, icon };
