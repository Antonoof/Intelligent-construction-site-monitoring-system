export const qs = (selector, scope = document) => scope.querySelector(selector);
export const qsa = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }

  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function icon(name, className = "") {
  return `<svg class="${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><use href="#i-${name}"/></svg>`;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export const fmt = {
  int: (value) => new Intl.NumberFormat("ru-RU").format(Math.round(value ?? 0)),
  pct: (value, digits = 1) => `${((value ?? 0) * 100).toFixed(digits)}%`,
  num: (value, digits = 3) => (value ?? 0).toFixed(digits),
  bytes: (value) => {
    const units = ["Б", "КБ", "МБ", "ГБ"];
    let size = value ?? 0;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
      size /= 1024;
      unit += 1;
    }
    return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
  },
  path: (value, max = 46) => {
    if (!value || value.length <= max) return value ?? "";
    return `…${value.slice(-(max - 1))}`;
  },
};

/** Categorical slots, fixed order, validated against the dark chart surface. */
export const SERIES_COLORS = [
  "#3987e5", "#d95926", "#199e70", "#c98500",
  "#d55181", "#008300", "#9085e9", "#e66767",
];

export const NEUTRAL = "#3a4657";

/** Identity colour for a class or slot; anything past slot 8 is "other". */
export const slotColor = (index) =>
  index >= 0 && index < SERIES_COLORS.length ? SERIES_COLORS[index] : NEUTRAL;

export function debounce(fn, delay = 250) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
