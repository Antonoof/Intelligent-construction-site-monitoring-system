import { projectApi } from "../core/api.js";
import { clear, el, fmt, qs, qsa, slotColor } from "../core/dom.js";
import { confirmDialog, modal, notifyError, runTask, setBusy, statCard, toast } from "../core/ui.js";
import { pickFile, pickFolder } from "../components/file-picker.js";

const root = qs(".dataset");
const projectId = root.dataset.project;
const project = projectApi(projectId);

const state = { mode: "ratio", plan: null, models: [] };

const splitParams = () => ({
  val_ratio: state.mode === "ratio" ? Number(qs("#ratio").value) : null,
  val_count: state.mode === "count" ? Number(qs("#count").value) : null,
  scope: qs("#scope").value,
  group_by: qs("#group-by").value,
  stratify: qs("#stratify").checked,
  seed: Number(qs("#seed").value) || 42,
});

/* ----------------------------------------------------------------- статус */
async function loadStatus() {
  try {
    const data = await project.status();
    const node = clear(qs("#status"));
    const total = data.counts.total || 0;

    node.append(
      statCard({
        label: "Всего кадров",
        value: fmt.int(total),
        sub: `${fmt.int(data.labeled)} с разметкой · ${fmt.int(data.unlabeled)} без`,
        meter: total ? data.labeled / total : 0,
      }),
      statCard({
        label: "В ожидании у напарников",
        value: fmt.int(data.counts.assigned || 0),
        sub: data.pending.length ? `последнее: ${data.pending[0].owner || "без имени"}` : "заданий нет",
        tone: data.counts.assigned ? "text-amber" : "",
      }),
      statCard({
        label: "Закрыто в AutoDetect2",
        value: fmt.int(data.done || 0),
        sub: "проверено и сохранено вручную",
        tone: "text-green",
      }),
      statCard({
        label: "Помечено спорными",
        value: fmt.int(data.counts.review || 0),
        sub: "клавиша R в редакторе",
        tone: data.counts.review ? "text-amber" : "",
      })
    );
  } catch (error) {
    notifyError(error);
  }
}

/* -------------------------------------------------------------- train/val */
function renderPlan(plan) {
  const node = clear(qs("#plan-result"));
  const total = plan.train + plan.val;
  if (!total) return;

  node.append(
    el("div", { class: "split-bar" }, [
      el("i", { class: "split-bar__train", style: { width: `${(plan.train / total) * 100}%` } }, [
        `train ${fmt.int(plan.train)}`,
      ]),
      el("i", { class: "split-bar__val", style: { width: `${(plan.val / total) * 100}%` } }, [
        `val ${fmt.int(plan.val)}`,
      ]),
    ]),
    el("div", { class: "row tiny faint", style: { gap: "12px", marginTop: "8px" } }, [
      el("span", { text: `${fmt.int(plan.groups)} групп` }),
      el("span", { text: `доля val ${fmt.pct(plan.params.val_ratio, 1)}` }),
      el("span", { text: plan.params.stratify ? "классы выровнены" : "без выравнивания" }),
      plan.leak_risk !== "none"
        ? el("span", { class: "text-amber", text: "похожие кадры по обе стороны — включите группировку" })
        : null,
    ])
  );

  if (!plan.balance.length) return;
  const table = el("table", { class: "table" }, [
    el("thead", {}, [
      el("tr", {}, [
        el("th", { text: "Класс" }),
        el("th", { text: "train" }),
        el("th", { text: "val" }),
        el("th", { text: "доля val" }),
      ]),
    ]),
  ]);
  const body = el("tbody");
  plan.balance.forEach((row) => {
    const off = row.train + row.val > 0 && Math.abs(row.val_share - plan.params.val_ratio) > 0.12;
    body.append(
      el("tr", {}, [
        el("td", {}, [
          el("i", { class: "dot", style: { background: slotColor(row.id), marginRight: "6px" } }),
          `${row.id} · ${row.name}`,
        ]),
        el("td", { text: fmt.int(row.train) }),
        el("td", { text: fmt.int(row.val) }),
        el("td", { class: off ? "text-amber" : "", text: fmt.pct(row.val_share, 1) }),
      ])
    );
  });
  table.append(body);
  node.append(table);
}

async function plan() {
  const button = qs("#plan");
  setBusy(button, true, "Считаем…");
  try {
    const { plan: result } = await project.planSplit(splitParams());
    state.plan = result;
    renderPlan(result);
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(button, false);
  }
}

async function apply() {
  const ok = await confirmDialog(
    "Split будет переписан у всех кадров проекта, а data.yaml переключится на списки splits/train.txt и splits/val.txt. Исходные папки не меняются.",
    { title: "Применить деление?", danger: false }
  );
  if (!ok) return;

  const button = qs("#apply");
  setBusy(button, true, "Применяем…");
  try {
    const { plan: result, split } = await project.applySplit(splitParams());
    state.plan = result;
    renderPlan(result);
    qs("#split-current").textContent = `train ${fmt.int(split.train)} · val ${fmt.int(split.val)}`;
    toast("Деление применено", { kind: "success" });
    loadStatus();
  } catch (error) {
    notifyError(error);
  } finally {
    setBusy(button, false);
  }
}

/* --------------------------------------------------------------- выгрузка */
async function exportDataset() {
  try {
    const result = await runTask(
      () =>
        project.exportDataset({
          link_mode: qs("#link-mode").value,
          include_unlabeled: qs("#include-unlabeled").value === "true",
        }),
      { title: "Выгружаем датасет" }
    );
    const node = clear(qs("#export-result"));
    node.append(
      el("div", {}, [el("span", { text: "Папка" }), el("b", { class: "mono tiny", text: result.path })]),
      el("div", {}, [
        el("span", { text: "Файлов" }),
        el("b", { text: `train ${fmt.int(result.train)} · val ${fmt.int(result.val)} · labels ${fmt.int(result.labels)}` }),
      ]),
      result.skipped
        ? el("div", {}, [el("span", { text: "Пропущено" }), el("b", { text: fmt.int(result.skipped) })])
        : null
    );
    toast("Датасет выгружен", { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
}

/* ----------------------------------------------------------- предразметка */
async function loadModels() {
  try {
    const { models } = await project.models();
    state.models = models;
    const select = clear(qs("#prelabel-model"));
    if (!models.length) {
      select.append(el("option", { value: "", text: "— загрузите *.pt в разделе «Инференс» —" }));
      return;
    }
    models.forEach((model) => select.append(el("option", { value: model.name, text: model.name })));
  } catch (error) {
    notifyError(error);
  }
}

async function loadProposals() {
  try {
    const info = await project.proposals();
    qs("#proposals-state").textContent = `${fmt.int(info.images)} кадров · ${fmt.int(info.boxes)} боксов`;
    qs("#proposals-state").className = "badge badge--green";
  } catch {
    qs("#proposals-state").textContent = "нет";
    qs("#proposals-state").className = "badge";
  }
}

async function prelabel() {
  const model = qs("#prelabel-model").value;
  if (!model) {
    toast("Сначала подключите модель на странице «Инференс»", { kind: "error" });
    return;
  }
  try {
    await runTask(
      () =>
        project.prelabel({
          model,
          conf: Number(qs("#prelabel-conf").value),
          limit: Number(qs("#prelabel-limit").value) || 1000,
          scope: "unlabeled",
        }),
      { title: "Предразметка" }
    );
    await loadProposals();
    toast("Черновая разметка готова — откройте раздел «Разметка»", { kind: "success" });
  } catch (error) {
    notifyError(error);
  }
}

async function applyProposals() {
  const ok = await confirmDialog(
    "Черновые боксы станут разметкой на кадрах, где её ещё нет. Ручная работа не перезаписывается.",
    { title: "Принять предразметку?", danger: false }
  );
  if (!ok) return;
  try {
    const result = await project.applyProposals({ conf: Number(qs("#prelabel-conf").value), only_unlabeled: true });
    toast(`Размечено ${result.images} кадров, ${result.boxes} боксов`, { kind: "success" });
    loadStatus();
  } catch (error) {
    notifyError(error);
  }
}

/* ----------------------------------------------------------------- задания */
const STATUS_TONE = { open: "badge--amber", accepted: "badge--accent", imported: "badge--green", cancelled: "badge" };
const STATUS_LABEL = {
  open: "в работе у напарника",
  accepted: "принято мной",
  imported: "результат влит",
  cancelled: "отозвано",
};

async function loadAssignments() {
  try {
    const { assignments } = await project.assignments();
    qs("#assignment-count").textContent = String(assignments.length);
    const node = clear(qs("#assignments"));

    if (!assignments.length) {
      node.append(
        el("p", { class: "tiny faint", style: { padding: "12px 0" }, text: "Заданий пока нет." })
      );
      return;
    }

    assignments.forEach((item) => {
      const actions = el("div", { class: "row", style: { gap: "6px" } });

      if (item.status === "open") {
        actions.append(
          el("button", { class: "btn btn--sm", text: "Манифест .json", onClick: () => exportAssignment(item, "manifest") }),
          el("button", { class: "btn btn--sm", text: "Архив .zip", onClick: () => exportAssignment(item, "pack") }),
          el("button", { class: "btn btn--sm btn--danger", text: "Отозвать", onClick: () => cancelAssignment(item) })
        );
      } else if (item.status === "accepted") {
        actions.append(
          el("button", {
            class: "btn btn--sm btn--primary",
            text: "Собрать результат",
            onClick: () => buildResult(item),
          }),
          el("a", { class: "btn btn--sm", href: `/p/${projectId}/annotate`, text: "Разметить" })
        );
      }

      node.append(
        el("div", { class: "assignment" }, [
          el("div", { class: "assignment__main" }, [
            el("div", { class: "row", style: { gap: "8px" } }, [
              el("b", { class: "truncate", text: item.title || item.id }),
              el("span", { class: `badge ${STATUS_TONE[item.status] || "badge"}`, text: STATUS_LABEL[item.status] || item.status }),
              item.owner ? el("span", { class: "badge badge--violet", text: item.owner }) : null,
            ]),
            el("div", { class: "tiny faint" }, [
              `${fmt.int(item.count)} кадров · ${(item.created_at || "").slice(0, 16).replace("T", " ")}`,
              item.result ? ` · влито ${fmt.int(item.result.images)} кадров, ${fmt.int(item.result.boxes)} боксов` : "",
              item.note ? ` · ${item.note}` : "",
            ]),
            item.exports?.length
              ? el(
                  "div",
                  { class: "row tiny", style: { gap: "8px", flexWrap: "wrap" } },
                  item.exports.map((file) =>
                    el("a", { class: "mono", href: `/api/projects/${projectId}/download/${file}`, text: file })
                  )
                )
              : null,
          ]),
          actions,
        ])
      );
    });
  } catch (error) {
    notifyError(error);
  }
}

async function exportAssignment(item, mode) {
  try {
    const result = await runTask(() => project.exportAssignment(item.id, mode), {
      title: mode === "pack" ? "Собираем архив" : "Собираем манифест",
    });
    toast(`${result.file} · ${fmt.bytes(result.size)}`, { kind: "success" });
    window.location.href = result.url;
    loadAssignments();
  } catch (error) {
    notifyError(error);
  }
}

async function cancelAssignment(item) {
  const ok = await confirmDialog(
    `Кадры задания «${item.title}» вернутся в вашу очередь. Разметка, которую уже сделал напарник, не пропадёт — её можно влить позже.`,
    { title: "Отозвать задание?" }
  );
  if (!ok) return;
  try {
    await project.cancelAssignment(item.id);
    toast("Задание отозвано", { kind: "success" });
    loadAssignments();
    loadStatus();
  } catch (error) {
    notifyError(error);
  }
}

async function buildResult(item) {
  try {
    const result = await project.assignmentResult(item.id);
    toast(`${result.file} — отправьте этот файл обратно`, { kind: "success", timeout: 6000 });
    window.location.href = result.url;
  } catch (error) {
    notifyError(error);
  }
}

function createDialog() {
  const source = el("select", {}, [
    el("option", { value: "unlabeled", text: "неразмеченные кадры" }),
    el("option", { value: "issues", text: "кадры с замечаниями (нужна проверка разметки)" }),
    el("option", { value: "selection", text: "последняя подобранная выборка" }),
    el("option", { value: "all", text: "любые свободные кадры" }),
  ]);
  const count = el("input", { type: "number", min: "1", value: "300" });
  const owner = el("input", { type: "text", placeholder: "кому отдаём" });
  const title = el("input", { type: "text", placeholder: "название задания" });
  const note = el("input", { type: "text", placeholder: "комментарий" });
  const preview = el("div", { class: "tiny faint" });

  const refresh = async () => {
    try {
      const result = await project.candidates({
        count: Number(count.value) || 1,
        source: source.value,
      });
      preview.textContent = `найдено свободных: ${result.count}`;
      preview.className = "tiny faint";
    } catch (error) {
      preview.textContent = error.message;
      preview.className = "tiny text-amber";
    }
  };
  source.addEventListener("change", refresh);
  count.addEventListener("change", refresh);
  refresh();

  const body = el("div", { class: "col", style: { gap: "12px" } }, [
    el("p", { class: "muted tiny", text: "Из выборки автоматически исключается всё, что уже отдано в другое задание или закрыто." }),
    el("div", { class: "field" }, [el("label", { text: "Откуда брать кадры" }), source]),
    el("div", { class: "grid grid--2", style: { gap: "12px" } }, [
      el("div", { class: "field" }, [el("label", { text: "Сколько кадров" }), count]),
      el("div", { class: "field" }, [el("label", { text: "Напарник" }), owner]),
    ]),
    el("div", { class: "field" }, [el("label", { text: "Название" }), title]),
    el("div", { class: "field" }, [el("label", { text: "Заметка" }), note]),
    preview,
  ]);

  const handle = modal({
    title: "Новое задание",
    body,
    actions: [
      el("button", { class: "btn", text: "Отмена", onClick: () => handle.close() }),
      el("button", {
        class: "btn btn--primary",
        text: "Создать",
        onClick: async () => {
          try {
            const { assignment } = await project.createAssignment({
              source: source.value,
              count: Number(count.value) || 1,
              owner: owner.value.trim(),
              title: title.value.trim(),
              note: note.value.trim(),
            });
            handle.close();
            toast(`Задание на ${assignment.count} кадров создано`, { kind: "success" });
            loadAssignments();
            loadStatus();
          } catch (error) {
            notifyError(error);
          }
        },
      }),
    ],
  });
}

/** Результат приходит файлом, а из чужого инструмента — папкой labels. */
function pickSource(title) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = async (picker) => {
      if (settled) return;
      settled = true;
      handle.close();
      resolve(picker ? await picker() : null);
    };

    const handle = modal({
      title,
      body: el("p", { class: "muted", text: "Файл — это манифест задания или результат. Папка — обычный labels/ из другого инструмента." }),
      actions: [
        el("button", { class: "btn", text: "Отмена", onClick: () => finish(null) }),
        el("button", {
          class: "btn",
          text: "Папка labels",
          onClick: () => finish(() => pickFolder("Папка с *.txt")),
        }),
        el("button", {
          class: "btn btn--primary",
          text: "Файл",
          onClick: () => finish(() => pickFile("*.json", "Файл задания или результата")),
        }),
      ],
      onClose: () => finish(null),
    });
  });
}

async function accept() {
  const path = await pickFile("*.json", "Файл задания — *.adtask.json или *.zip");
  if (!path) return;
  try {
    const result = await project.acceptAssignment(path);
    if (result.matched) {
      toast(`Принято ${result.matched} кадров — они уже в очереди разметки`, { kind: "success", timeout: 6000 });
    } else if (result.extracted_to) {
      toast(`Архив распакован в ${result.extracted_to} — создайте проект из папки images`, {
        kind: "info",
        timeout: 9000,
      });
    } else {
      toast("Ни один кадр задания не найден в этом проекте", { kind: "error", timeout: 7000 });
    }
    loadAssignments();
    loadStatus();
  } catch (error) {
    notifyError(error);
  }
}

async function importResult() {
  const path = await pickSource("Импорт результата");
  if (!path) return;
  try {
    const result = await project.importResult(path);
    toast(`Влито ${result.images} кадров, ${result.boxes} боксов`, { kind: "success", timeout: 6000 });
    if (result.unknown) {
      toast(`${result.unknown} файлов не совпали с проектом по имени`, { kind: "error", timeout: 7000 });
    }
    loadAssignments();
    loadStatus();
  } catch (error) {
    notifyError(error);
  }
}

/* ---------------------------------------------------------------- wiring */
qsa("#split-mode button").forEach((button) => {
  button.addEventListener("click", () => {
    qsa("#split-mode button").forEach((other) => other.classList.remove("is-active"));
    button.classList.add("is-active");
    state.mode = button.dataset.mode;
    qs("#ratio-field").classList.toggle("hidden", state.mode !== "ratio");
    qs("#count-field").classList.toggle("hidden", state.mode !== "count");
  });
});

qs("#ratio").addEventListener("input", (event) => {
  qs("#ratio-value").textContent = fmt.pct(Number(event.target.value), 0);
  qsa("[data-ratio]").forEach((button) =>
    button.classList.toggle("is-active", button.dataset.ratio === event.target.value)
  );
});

qsa("[data-ratio]").forEach((button) => {
  button.addEventListener("click", () => {
    qs("#ratio").value = button.dataset.ratio;
    qs("#ratio").dispatchEvent(new Event("input"));
  });
});

qs("#group-by").addEventListener("change", (event) => {
  const hints = {
    series: "кадры одной съёмки не разъезжаются между train и val",
    folder: "кадры одной папки целиком уходят в один сплит",
    none: "чистая случайность — похожие кадры могут попасть в обе части",
  };
  qs("#group-hint").textContent = hints[event.target.value];
  qs("#group-hint").className = event.target.value === "none" ? "field__hint text-amber" : "field__hint";
});

qs("#prelabel-conf").addEventListener("input", (event) => {
  qs("#prelabel-conf-value").textContent = Number(event.target.value).toFixed(2);
});

qs("#plan").addEventListener("click", plan);
qs("#apply").addEventListener("click", apply);
qs("#export").addEventListener("click", exportDataset);
qs("#prelabel").addEventListener("click", prelabel);
qs("#apply-proposals").addEventListener("click", applyProposals);
qs("#create").addEventListener("click", createDialog);
qs("#accept").addEventListener("click", accept);
qs("#import").addEventListener("click", importResult);

loadStatus();
loadAssignments();
loadModels();
loadProposals();
plan();
