import { projectApi } from "../core/api.js";
import { clear, el, fmt, qs, slotColor } from "../core/dom.js";
import { modal, notifyError, runTask, statCard, toast } from "../core/ui.js";
import { openEditor } from "../components/editor-modal.js";

const root = qs(".quality");
const projectId = root.dataset.project;
const project = projectApi(projectId);

const state = { report: null, kind: null, needle: "" };

const SEVERITY_TONE = { high: "badge--red", medium: "badge--amber", low: "badge" };

/* ------------------------------------------------------------- отрисовка */
function renderSummary(report) {
  const node = clear(qs("#summary"));
  const s = report.summary;
  const dirty = s.labeled ? s.images_with_issues / s.labeled : 0;

  node.append(
    statCard({
      label: "Кадров с проблемами",
      value: fmt.int(s.images_with_issues),
      sub: `из ${fmt.int(s.labeled)} размеченных`,
      meter: Math.min(dirty, 1),
      tone: dirty > 0.15 ? "text-red" : dirty > 0.05 ? "text-amber" : "text-green",
    }),
    statCard({
      label: "Замечаний",
      value: fmt.int(s.issues),
      sub: `${s.high} критичных · ${s.medium} средних · ${s.low} мелких`,
    }),
    statCard({
      label: "Без разметки",
      value: fmt.int(s.unlabeled),
      sub: `${fmt.int(s.empty)} пустых файлов · ${fmt.int(s.boxes)} боксов всего`,
    }),
    statCard({
      label: "Как считали",
      value: report.backend.startsWith("geometry+") ? "геометрия + вид" : "геометрия",
      sub: report.created_at?.slice(0, 16).replace("T", " ") || "",
    })
  );
}

function renderKinds(report) {
  const node = clear(qs("#kinds"));
  if (!report.by_kind.length) {
    node.append(el("p", { class: "tiny faint", text: "Ни одного замечания — разметка чистая." }));
    return;
  }

  const peak = Math.max(...report.by_kind.map((item) => item.count));
  report.by_kind.forEach((item) => {
    node.append(
      el(
        "button",
        {
          class: `kind ${state.kind === item.kind ? "is-active" : ""}`,
          onClick: () => {
            state.kind = state.kind === item.kind ? null : item.kind;
            renderKinds(report);
            renderRows();
          },
        },
        [
          el("div", { class: "row", style: { gap: "8px" } }, [
            el("span", { class: `badge ${SEVERITY_TONE[item.severity]}`, text: item.severity === "high" ? "!" : "·" }),
            el("span", { class: "kind__label truncate", text: item.label, title: item.label }),
            el("span", { class: "spacer" }),
            el("b", { text: fmt.int(item.count) }),
          ]),
          el("div", { class: "meter" }, [el("i", { style: { width: `${(item.count / peak) * 100}%` } })]),
          el("span", { class: "tiny faint", text: `${fmt.int(item.images)} кадров` }),
        ]
      )
    );
  });
}

function renderClasses(report) {
  const node = clear(qs("#classes"));
  const peak = Math.max(1, ...report.classes.map((item) => item.boxes));

  report.classes.forEach((item) => {
    node.append(
      el("div", { class: `classbar ${item.known ? "" : "is-unknown"}` }, [
        el("div", { class: "row", style: { gap: "8px" } }, [
          el("i", { class: "dot", style: { background: slotColor(item.id) } }),
          el("span", { class: "truncate", text: `${item.id} · ${item.name}`, title: item.name }),
          el("span", { class: "spacer" }),
          el("b", { text: fmt.int(item.boxes) }),
        ]),
        el("div", { class: "meter" }, [
          el("i", { style: { width: `${(item.boxes / peak) * 100}%`, background: slotColor(item.id) } }),
        ]),
        el("span", {
          class: "tiny faint",
          text: item.known
            ? `${fmt.int(item.images)} кадров · доля ${fmt.pct(item.share, 1)}`
            : "класса нет в data.yaml",
        }),
      ])
    );
  });
}

function visibleRows() {
  if (!state.report) return [];
  return state.report.images.filter((row) => {
    if (state.kind && !row.kinds.includes(state.kind)) return false;
    if (state.needle && !row.name.toLowerCase().includes(state.needle)) return false;
    return true;
  });
}

function renderRows() {
  const node = clear(qs("#rows"));
  const rows = visibleRows();
  qs("#rows-count").textContent = fmt.int(rows.length);
  qs("#clear-kind").hidden = !state.kind;

  if (!rows.length) {
    node.append(el("div", { class: "tiny faint center", style: { padding: "28px" }, text: "Ничего не найдено" }));
    return;
  }

  rows.slice(0, 600).forEach((row) => {
    const shown = state.kind ? row.issues.filter((issue) => issue.kind === state.kind) : row.issues;
    node.append(
      el("button", { class: "qrow", onClick: () => open(row) }, [
        el("img", { class: "qrow__thumb", src: project.imageUrl(row.name), loading: "lazy", alt: "" }),
        el("div", { class: "qrow__body" }, [
          el("div", { class: "row", style: { gap: "6px" } }, [
            el("span", { class: "qrow__name truncate", text: row.name, title: row.name }),
            el("span", { class: "spacer" }),
            el("span", { class: "badge", text: row.split }),
            el("span", { class: "badge badge--accent", text: `${row.boxes} боксов` }),
            el("span", { class: "badge badge--red", text: `риск ${row.score.toFixed(1)}` }),
          ]),
          el(
            "div",
            { class: "qrow__issues" },
            shown.slice(0, 5).map((issue) =>
              el("span", { class: `chip chip--${issue.severity}` }, [
                el("b", { text: issue.label }),
                issue.box >= 0 ? el("span", { class: "faint", text: `бокс №${issue.box + 1}` }) : null,
                issue.detail ? el("span", { class: "faint", text: issue.detail }) : null,
              ])
            )
          ),
        ]),
      ])
    );
  });

  if (rows.length > 600) {
    node.append(
      el("div", {
        class: "tiny faint center",
        style: { padding: "14px" },
        text: `показаны первые 600 из ${fmt.int(rows.length)} — сузьте фильтр`,
      })
    );
  }
}

function open(row) {
  const summary = row.issues
    .slice(0, 3)
    .map((issue) => issue.label)
    .join(" · ");
  openEditor(project, row.name, {
    meta: summary,
    onSaved: () => toast(`${row.name}: сохранено`, { kind: "success", timeout: 1600 }),
  });
}

/* ---------------------------------------------------------------- запуск */
async function run() {
  try {
    await runTask(() => project.buildQuality({ check_classes: qs("#check-classes").checked }), {
      title: "Проверяем разметку",
    });
    await load();
    toast("Проверка завершена", { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
}

async function load(quiet = false) {
  try {
    const report = await project.quality(2000);
    state.report = report;
    renderSummary(report);
    renderKinds(report);
    renderClasses(report);
    renderRows();
    return true;
  } catch (error) {
    if (!quiet) notifyError(error);
    return false;
  }
}

/* ---------------------------------------------------- передача напарнику */
async function toAssignment() {
  const rows = visibleRows();
  if (!rows.length) {
    toast("Сначала отфильтруйте кадры", { kind: "error" });
    return;
  }

  const count = el("input", { type: "number", min: "1", max: String(rows.length), value: String(Math.min(rows.length, 200)) });
  const owner = el("input", { type: "text", placeholder: "кому отдаём" });
  const note = el("input", { type: "text", placeholder: "комментарий (необязательно)" });

  const body = el("div", { class: "col", style: { gap: "12px" } }, [
    el("p", { class: "muted", text: `В выборке ${fmt.int(rows.length)} кадров с замечаниями. Они уйдут в задание и пропадут из вашей очереди, пока не вернётся результат.` }),
    el("div", { class: "field" }, [el("label", { text: "Сколько кадров" }), count]),
    el("div", { class: "field" }, [el("label", { text: "Напарник" }), owner]),
    el("div", { class: "field" }, [el("label", { text: "Заметка" }), note]),
  ]);

  const handle = modal({
    title: "Задание напарнику",
    body,
    actions: [
      el("button", { class: "btn", text: "Отмена", onClick: () => handle.close() }),
      el("button", {
        class: "btn btn--primary",
        text: "Создать",
        onClick: async () => {
          const names = rows.slice(0, Math.max(1, Number(count.value) || 1)).map((row) => row.name);
          try {
            const { assignment } = await project.createAssignment({
              images: names,
              owner: owner.value.trim(),
              note: note.value.trim(),
              title: state.kind ? `Проблема: ${state.kind}` : "Проблемная разметка",
            });
            handle.close();
            toast(`Задание на ${assignment.count} кадров создано`, { kind: "success" });
            window.location.href = `/p/${projectId}/dataset`;
          } catch (error) {
            notifyError(error);
          }
        },
      }),
    ],
  });
}

/* --------------------------------------------------------------- wiring */
qs("#run").addEventListener("click", run);
qs("#to-assignment").addEventListener("click", toAssignment);
qs("#clear-kind").addEventListener("click", () => {
  state.kind = null;
  renderKinds(state.report);
  renderRows();
});
qs("#filter").addEventListener("input", (event) => {
  state.needle = event.target.value.trim().toLowerCase();
  renderRows();
});

load(true).then((ok) => {
  if (!ok) {
    clear(qs("#summary")).append(
      el("div", { class: "card card--pad-sm", style: { gridColumn: "1 / -1" } }, [
        el("p", { class: "muted", text: "Проверка ещё не запускалась. Нажмите «Проверить» — первый прогон по большому датасету занимает минуту." }),
      ])
    );
  }
});
