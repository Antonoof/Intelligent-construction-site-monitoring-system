// ОКО — интерфейс прототипа: сводка площадки, снимки с разметкой, отклонения с доказательствами,
// календарный график, методика «этап → техника», быстрая проверка снимка.
// Без сборки и зависимостей: страница обслуживается бэкендом (FastAPI) и ходит в его REST API.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const SEV = { critical: 'Критично', warning: 'Внимание', info: 'К сведению' };
const STATUS = { preliminary: 'предварительно', confirmed: 'подтверждено' };
// основание подтверждения: серия снимков или одна очень уверенная рамка
const statusText = (d) => d.status === 'confirmed' && d.metrics?.confirmed_by === 'confidence'
  ? 'подтверждено: уверенная рамка' : STATUS[d.status];
const REVIEW = { accepted: 'нарушение зарегистрировано', rejected: 'ложное срабатывание', force_majeure: 'форс-мажор' };
const OBS = { high: 'высокая', medium: 'средняя', low: 'низкая', none: 'не видно с камер' };
const ZSTATUS = { ok: 'по графику', critical: 'критично', warning: 'внимание', info: 'к сведению', no_data: 'нет данных',
  no_tasks: 'работ нет', no_snapshots: 'нет снимков' };
const STAGES = { S1: 'S1 подготовка и котлован', S2: 'S2 фундамент', S3: 'S3 каркас', S4: 'S4 фасад и сети', S5: 'S5 отделка, благоустройство' };

const state = { projects: [], project: null, pid: null, day: null, tab: 'summary', eq: {}, rules: null, workTypes: null,
  profiles: null, filters: { sev: '', status: '', zone: '' }, camFilter: '', ai: null };

// ------------------------------------------------------------------ API
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) { /* не JSON */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r.text();
}

function toast(text, ms = 3500) {
  const t = $('#toast');
  t.textContent = text; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, ms);
}

// ------------------------------------------------------------------ форматирование
const fDay = (iso) => { if (!iso) return ''; const [y, m, d] = iso.slice(0, 10).split('-'); return `${d}.${m}.${y}`; };
const fTime = (iso) => iso ? iso.slice(11, 16) : '';
const clsName = (k) => state.eq[k]?.name || k;
const clsColor = (k) => state.eq[k]?.color || '#888';
const sevChip = (s) => `<span class="chip s-${s}">${SEV[s] || s}</span>`;
const zoneChip = (z) => z ? `<span class="zchip" style="--zc:${esc(z.color)}"><i></i>${esc(z.name)}</span>` : '';
function eqChip(cls, n, mod = '') {
  return `<span class="eq ${mod}" title="${esc(clsName(cls))}"><span class="dot" style="background:${esc(clsColor(cls))}"></span>${esc(clsName(cls))}${n ? ` <b>×${n}</b>` : ''}</span>`;
}
function plural(n, one, few, many) { const a = Math.abs(n) % 100, b = a % 10; return a > 10 && a < 20 ? many : b === 1 ? one : b >= 2 && b <= 4 ? few : many; }

// ------------------------------------------------------------------ кадр с разметкой (SVG поверх снимка)
function frameHTML(snap, { boxes = [], zones = [], highlight = null, thumb = false, onlyZone = null, labels = true,
  attention = null, aiBoxes = [], objBoxes = [] } = {}) {
  const W = snap.width, H = snap.height, fs = Math.round(W / 70);
  let svg = `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">`;
  if (attention) {             // карта внимания модели готовности: сетка 9×16, ярче — важнее для вывода
    const gh = attention.length, gw = attention[0].length, cw = W / gw, ch = H / gh;
    attention.forEach((row, i) => row.forEach((v, j) => {
      if (v > 0.05) svg += `<rect x="${(j * cw).toFixed(1)}" y="${(i * ch).toFixed(1)}" width="${cw.toFixed(1)}" height="${ch.toFixed(1)}" fill="#FF6A00" fill-opacity="${(v * 0.55).toFixed(2)}"/>`;
    }));
  }
  for (const z of zones) {
    if (onlyZone && z.key !== onlyZone) continue;
    const pts = z.polygon.map(([x, y]) => `${(x * W).toFixed(1)},${(y * H).toFixed(1)}`).join(' ');
    svg += `<polygon class="ov-zone" points="${pts}" fill="${esc(z.color)}" stroke="${esc(z.color)}"/>`;
    if (labels && !thumb) {
      const [x0, y0] = z.polygon.reduce((a, p) => [Math.min(a[0], p[0]), Math.min(a[1], p[1])], [1, 1]);
      svg += `<text x="${x0 * W + 8}" y="${y0 * H + fs + 4}" class="ov-label" style="font-size:${fs}px" fill="#fff" stroke="${esc(z.color)}" stroke-width="4" paint-order="stroke">${esc(z.name)}</text>`;
    }
  }
  for (const b of boxes) {
    if (b.rejected) continue;
    if (onlyZone && b.zone && b.zone !== onlyZone) continue;
    const [x1, y1, x2, y2] = b.xyxy, hl = highlight && highlight.has(b.id);
    svg += `<rect class="ov-box${b.strong ? '' : ' weak'}${hl ? ' hl' : ''}" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" stroke="${esc(b.color)}" rx="3"/>`;
    if (labels) {
      const t = `${b.label} ${b.conf.toFixed(2)}`, tw = t.length * fs * 0.56 + 10;
      svg += `<rect x="${x1}" y="${Math.max(0, y1 - fs - 8)}" width="${tw}" height="${fs + 7}" fill="${esc(b.color)}" rx="3"/>`
        + `<text x="${x1 + 5}" y="${Math.max(fs, y1 - 6)}" class="ov-label" style="font-size:${fs}px" fill="#fff">${esc(t)}</text>`;
    }
  }
  for (const b of objBoxes) {  // Grounding DINO: опалубка, леса, рабочие… и техника, которой нет у детектора
    const [x1, y1, x2, y2] = b.xyxy, t = `${b.label} ${b.conf.toFixed(2)}`, tw = t.length * fs * 0.52 + 10;
    svg += `<rect class="ov-box obj" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" rx="2"/>`
      + `<rect x="${x1}" y="${Math.max(0, y1 - fs - 6)}" width="${tw}" height="${fs + 5}" fill="#0B7F86" rx="2"/>`
      + `<text x="${x1 + 5}" y="${Math.max(fs - 2, y1 - 5)}" class="ov-label" style="font-size:${fs * 0.92}px" fill="#fff">${esc(t)}</text>`;
  }
  for (const b of aiBoxes) {   // техника, которую нашёл ИИ, а детектор пропустил (ещё не применено)
    const [x1, y1, x2, y2] = b.xyxy, t = `ИИ: ${b.name} ${b.conf.toFixed(2)}`, tw = t.length * fs * 0.56 + 10;
    svg += `<rect class="ov-box ai" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" rx="3"/>`
      + `<rect x="${x1}" y="${Math.min(H - fs - 7, y2 + 2)}" width="${tw}" height="${fs + 7}" fill="#C0198C" rx="3"/>`
      + `<text x="${x1 + 5}" y="${Math.min(H - 5, y2 + fs + 2)}" class="ov-label" style="font-size:${fs}px" fill="#fff">${esc(t)}</text>`;
  }
  svg += '</svg>';
  const src = thumb ? snap.thumb || `/api/snapshots/${snap.id}/image?w=480` : snap.image || `/api/snapshots/${snap.id}/image`;
  return `<div class="frame${snap.quality_ok === false ? ' nodata' : ''}" data-snap="${snap.id}"><img src="${src}" alt="${esc(snap.camera)} ${esc(snap.taken_at)}" loading="lazy">${svg}`
    + `<span class="badge">${esc(snap.camera)} · ${fDay(snap.taken_at)} ${fTime(snap.taken_at)}</span></div>`;
}

// ------------------------------------------------------------------ инициализация
async function init() {
  const [projects, eq, rules, ai] = await Promise.all([api('/api/projects'), api('/api/reference/equipment'), api('/api/reference/rules'),
    api('/api/ai/status').catch(() => null)]);
  state.projects = projects;
  state.ai = ai;
  eq.forEach((e) => { state.eq[e.key] = e; });
  state.rules = rules;
  const h = new URLSearchParams(location.hash.slice(1));
  state.pid = Number(h.get('p')) || projects[0]?.id || null;
  state.tab = h.get('t') || 'summary';
  state.day = h.get('d') || null;
  $('#project').innerHTML = projects.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
  $('#project').value = state.pid;
  $('#project').onchange = () => { state.pid = Number($('#project').value); state.day = null; loadProject(); };
  $('#day').onchange = () => { state.day = $('#day').value; render(); };
  $('#dayPrev').onclick = () => shiftDay(-1);
  $('#dayNext').onclick = () => shiftDay(1);
  $('#uploadBtn').onclick = openUpload;
  $$('.tabs button').forEach((b) => { b.onclick = () => { state.tab = b.dataset.tab; render(); }; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && $('#dlg').open) $('#dlg').close(); });
  window.addEventListener('hashchange', () => {            // ссылки вида #p=1&d=2026-09-24&t=deviations
    const q = new URLSearchParams(location.hash.slice(1));
    const p = Number(q.get('p')) || state.pid, d = q.get('d') || state.day, t = q.get('t') || state.tab;
    if (p === state.pid && d === state.day && t === state.tab) return;
    const reload = p !== state.pid;
    state.pid = p; state.day = d; state.tab = t;
    if (reload) { $('#project').value = p; loadProject(); } else { $('#day').value = d; render(); }
  });
  api('/api/health').then((hh) => {
    const a = state.ai;
    const layers = a ? [a.readiness?.enabled && 'модель готовности', a.open_vocab?.enabled && 'Grounding DINO', a.vlm?.enabled && `VLM ${a.vlm.model || '—'}`,
      a.llm?.provider !== 'off' && `${a.llm.name} ${a.llm.model}`].filter(Boolean) : [];
    $('#status').innerHTML = `<span>Детектор: <b>${esc(hh.detector)}</b> (${hh.detector_classes.length} классов)</span>`
      + `<span>ИИ-анализ: ${layers.length ? esc(layers.join(' · ')) : 'выключен'}</span>`
      + `<span>Методика ${esc(hh.methodology_version)} · правила ${esc(hh.rules_version)}</span><span>БД: ${esc(hh.database)}</span>`;
  });
  if (state.pid) await loadProject(); else render();
}

async function loadProject() {
  state.project = await api(`/api/projects/${state.pid}`);
  const days = state.project.days.map((d) => d.day);
  if (!state.day || !days.includes(state.day)) state.day = days[days.length - 1] || new Date().toISOString().slice(0, 10);
  $('#day').innerHTML = (days.length ? days : [state.day]).map((d) => {
    const x = state.project.days.find((q) => q.day === d);
    const n = x ? Object.values(x.deviations).reduce((a, b) => a + b, 0) : 0;
    return `<option value="${d}">${fDay(d)}${x ? ` · ${x.snapshots} сн., ${n} откл.` : ''}</option>`;
  }).join('');
  $('#day').value = state.day;
  render();
}

function shiftDay(dir) {
  const opts = $$('#day option').map((o) => o.value);
  const i = opts.indexOf(state.day) + dir;
  if (i >= 0 && i < opts.length) { state.day = opts[i]; $('#day').value = state.day; render(); }
}

function render() {
  const h = `p=${state.pid}&d=${state.day}&t=${state.tab}`;
  if (location.hash.slice(1) !== h) history.replaceState(null, '', `#${h}`);
  $$('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === state.tab));
  const main = $('#main');
  main.innerHTML = '<p class="muted"><span class="spinner"></span> Загрузка…</p>';
  const views = { summary: vSummary, snapshots: vSnapshots, deviations: vDeviations, schedule: vSchedule,
    methodology: vMethodology, analyze: vAnalyze };
  (views[state.tab] || vSummary)(main).catch((e) => { main.innerHTML = `<div class="card">Ошибка: ${esc(e.message)}</div>`; });
}

// ------------------------------------------------------------------ сводка
async function vSummary(main) {
  if (!state.pid) { main.innerHTML = '<div class="empty">Нет проектов</div>'; return; }
  const s = await api(`/api/projects/${state.pid}/summary?day=${state.day}`);
  const k = s.kpi;
  $('#nDevs').textContent = s.deviations.length || '';
  $('#nSnaps').textContent = k.snapshots || '';
  const kpi = (l, v, sub, cls = '') => `<div class="kpi ${cls}"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`;
  let html = `<div class="kpis">`
    + kpi('Снимков за день', k.snapshots, `годных ${k.valid} · камер ${k.cameras}`)
    + kpi('Критично', k.critical, 'срыв графика, нет работ', 'k-critical')
    + kpi('Внимание', k.warning, 'неполный комплект, лишняя техника', 'k-warning')
    + kpi('К сведению', k.info, 'опережение, нет данных', 'k-info')
    + kpi('Нарушений открыто', k.violations, 'подтверждены инженером')
    + kpi('Обработка снимка', `${k.avg_processing_ms} мс`, `этапов в работе: ${k.active_tasks}`) + `</div>`;
  html += `<section class="card ai-day" id="aiDay"></section>`;
  html += `<div class="grid2"><section><h2>Зоны площадки · ${fDay(s.day)}</h2><div class="zones">`;
  for (const z of s.board) html += zoneCard(z);
  html += `</div></section><section><h2>Отклонения · ${s.deviations.length}</h2><div class="devlist">`;
  html += s.deviations.length ? s.deviations.map((d) => devCard(d, { compact: true })).join('')
    : '<div class="card empty">Отклонений нет — техника на площадке соответствует графику.</div>';
  html += `</div></section></div>`;
  main.innerHTML = html;
  bindCommon(main);
  loadDayAI(state.day);
}

function zoneCard(z) {
  const st = z.status;
  const last = z.last_snapshot;
  const seen = z.seen || {};
  let h = `<article class="zone"><header>${zoneChip(z.zone)}<span class="chip s-${st} status-chip">${esc(ZSTATUS[st] || st)}</span></header>`;
  if (last) {
    h += `<div class="zone-frame" data-open-snap="${last.id}" data-zone="${esc(z.zone.key)}"><div class="frame${last.quality_ok ? '' : ' nodata'}"><img src="/api/snapshots/${last.id}/image?w=640&annotate=1" loading="lazy" alt="">`
      + `<span class="badge">${esc(last.camera)} · ${fTime(last.taken_at)}${last.quality_ok ? '' : ' · нет данных'}</span></div></div>`;
  }
  h += `<div class="body">`;
  if (!z.tasks.length) h += `<div class="hint">По графику на этот день работ в зоне нет — любая тяжёлая техника здесь отклонение.</div>`;
  for (const t of z.tasks) {
    const obs = OBS[t.observability] || '';
    h += `<div class="task"><div><span class="code">${esc(t.wbs || t.work_type_id || '')}</span> ${esc(t.name)}</div>`;
    if (t.summary) { h += `<div class="hint">сводный этап — проверяются подэтапы</div></div>`; continue; }
    h += `<div class="hint">${esc(t.profile || 'не сопоставлен со справочником')} · наблюдаемость: ${obs}</div>`;
    const groups = t.day_check?.groups || [];
    if (groups.length && (t.observability === 'high' || t.observability === 'medium')) {
      // состояние групп за день — тот же расчёт, что у отклонений
      const MARK = { ok: ['ok', '✓'], site_equipment: ['ok', '✓'], missing: ['miss', '✗'], pending: ['', '…'], not_detectable: ['nd', '?'] };
      h += `<div class="row"><span class="lbl">нужна:</span>` + groups.map((g) => {
        const [cls, mk] = MARK[g.state] || ['', '·'];
        return `<span class="eq ${cls}" title="${esc(g.role)}">${mk} ${esc(g.expected)}${g.window === 'shift' ? ' <span class="muted">за смену</span>' : ''}</span>`;
      }).join('') + `</div>`;
      const bad = (t.day_check.companions || []).filter((c) => c.state === 'violated');
      if (bad.length) h += `<div class="row"><span class="lbl">пара:</span>${bad.map((c) => `<span class="eq miss">✗ ${esc(c.lead)} → ${esc(c.partner)}</span>`).join('')}</div>`;
    }
    h += `</div>`;
  }
  const seenList = Object.entries(seen);
  h += `<div class="row"><span class="lbl">последний снимок:</span>${seenList.length ? seenList.map(([c, n]) => eqChip(c, n)).join('') : '<span class="hint">техники нет</span>'}</div>`;
  const dayList = Object.entries(z.seen_day || {});
  if (dayList.length) h += `<div class="row"><span class="lbl">за день (макс.):</span>${dayList.map(([c, n]) => eqChip(c, n)).join('')}</div>`;
  h += `<div class="row hint">снимков: ${z.snapshots} (годных ${z.valid}) · камеры: ${esc(z.cameras.join(', ') || '—')}${z.deviations ? ` · <a href="#" data-goto-dev="${esc(z.zone.key)}">отклонений: ${z.deviations}</a>` : ''}</div>`;
  return h + `</div></article>`;
}

// ------------------------------------------------------------------ карточка отклонения
function devCard(d, { compact = false } = {}) {
  const t = d.task;
  const span = d.first_seen && d.last_seen ? (fTime(d.first_seen) === fTime(d.last_seen) ? fTime(d.first_seen) : `${fTime(d.first_seen)}–${fTime(d.last_seen)}`) : '';
  let h = `<article class="dev ${d.severity}" id="dev-${d.id}"><div class="head"><div><div class="title">${esc(d.title)}</div>`
    + `<div class="row">${sevChip(d.severity)}<span class="chip plain s-${d.status === 'confirmed' ? 'ok' : 'pending'}">${statusText(d)} · ${d.snapshots_count} ${plural(d.snapshots_count, 'снимок', 'снимка', 'снимков')}</span>`
    + `${d.review_status ? `<span class="chip plain s-${d.review_status === 'accepted' ? 'critical' : 'no_tasks'} reviewed">${REVIEW[d.review_status]}${d.violation ? ` · ${esc(d.violation.number)}` : ''}</span>` : ''}</div></div>`
    + `<div style="text-align:right">${zoneChip(d.zone)}<div class="hint">${fDay(d.day)} ${span}</div></div></div>`;
  h += `<div class="msg">${esc(d.message)}</div>`;
  if (t) h += `<div class="hint">Этап графика: <span class="mono">${esc(t.wbs || '')}</span> ${esc(t.name)} · ${fDay(t.start)}–${fDay(t.end)}${t.work_type_id ? ` · вид работ ${esc(t.work_type_id)}` : ''}</div>`;
  if (d.evidence?.length) {
    h += `<div class="ev${compact ? '' : ' big'}">` + d.evidence.map((e) => `<figure data-open-snap="${e.snapshot_id}" data-hl="${(e.detection_ids || []).join(',')}"><img src="${e.thumb}" alt="снимок ${e.snapshot_id}" loading="lazy">`
      + `<figcaption>${esc(e.camera)} · ${fTime(e.taken_at)}${e.note ? ' · ' + esc(e.note) : ''}</figcaption></figure>`).join('') + `</div>`;
  }
  if (!compact) {
    h += `<div class="rec">Что сделать: ${esc(d.recommendation)}</div>`;
    h += `<details class="calc"><summary>Расчёт и правило ${esc(d.rule)} (правила ${esc(d.rules_version)})</summary><pre>${esc(JSON.stringify(d.metrics, null, 1))}</pre></details>`;
  }
  if (!d.review_status) {
    h += `<div class="row">`
      + `<button class="btn small bad" data-review="${d.id}" data-v="accepted">Подтвердить → нарушение</button>`
      + `<button class="btn small" data-review="${d.id}" data-v="rejected">Ложное срабатывание</button>`
      + `<button class="btn small" data-review="${d.id}" data-v="force_majeure">Форс-мажор</button></div>`;
  }
  return h + `</article>`;
}

function bindCommon(root) {
  $$('[data-open-snap]', root).forEach((el) => {
    el.onclick = (e) => { e.preventDefault(); openSnapshot(Number(el.dataset.openSnap), el.dataset.hl || '', el.dataset.zone || ''); };
  });
  $$('[data-review]', root).forEach((b) => { b.onclick = () => review(Number(b.dataset.review), b.dataset.v); });
  $$('[data-goto-dev]', root).forEach((a) => { a.onclick = (e) => { e.preventDefault(); state.filters.zone = a.dataset.gotoDev; state.tab = 'deviations'; render(); }; });
}

async function review(id, verdict) {
  const labels = { accepted: 'Подтвердить отклонение и зарегистрировать нарушение?', rejected: 'Отметить как ложное срабатывание? Рамки исключатся из пересчёта и будут помечены для дообучения детектора.', force_majeure: 'Отметить форс-мажор (погода, поломка, поставки)?' };
  const comment = prompt(`${labels[verdict]}\nКомментарий (необязательно):`, '');
  if (comment === null) return;
  const d = await api(`/api/deviations/${id}/review`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ verdict, comment }) });
  toast(verdict === 'accepted' ? `Нарушение ${d.violation?.number || ''} зарегистрировано` : `Отклонение: ${REVIEW[verdict]}`);
  render();
}

// ------------------------------------------------------------------ снимки
async function vSnapshots(main) {
  const cams = state.project.cameras;
  const list = await api(`/api/projects/${state.pid}/snapshots?day=${state.day}${state.camFilter ? `&camera=${encodeURIComponent(state.camFilter)}` : ''}`);
  let h = `<div class="filters"><label class="sel"><span>Камера</span><select id="camF"><option value="">все камеры</option>`
    + cams.map((c) => `<option value="${esc(c.key)}"${c.key === state.camFilter ? ' selected' : ''}>${esc(c.key)} · ${esc(c.name)}</option>`).join('')
    + `</select></label><span class="hint">${list.length} ${plural(list.length, 'снимок', 'снимка', 'снимков')} за ${fDay(state.day)}. Нажмите на снимок — откроется разметка и сопоставление с графиком.</span></div>`;
  h += `<div class="legend"><span>сплошная рамка — техника уверенно распознана</span><span>пунктир — уверенность ниже порога (в правила «лишней техники» не идёт)</span></div>`;
  h += `<div class="thumbs">` + list.map((s) => {
    const f = s.findings || {};
    const badges = Object.entries(f).map(([k, n]) => `<span class="chip s-${k}">${n}</span>`).join('');
    return `<div class="thumb" data-open-snap="${s.id}"><div class="frame${s.quality_ok ? '' : ' nodata'}"><img src="${s.thumb}" loading="lazy" alt=""><span class="badge">${esc(s.camera)} · ${fTime(s.taken_at)}</span></div>`
      + `<div class="meta"><span class="row">${Object.entries(s.counts).map(([c, n]) => eqChip(c, n)).join('') || `<span class="hint">${s.quality_ok ? 'техники нет' : esc(s.quality_reason)}</span>`}</span><span class="row">${badges || (s.quality_ok ? '<span class="chip s-ok">ок</span>' : '')}</span></div></div>`;
  }).join('') + `</div>`;
  if (!list.length) h += `<div class="card empty">За этот день снимков нет. Загрузите снимки кнопкой «Загрузить снимки».</div>`;
  main.innerHTML = h;
  $('#camF').onchange = () => { state.camFilter = $('#camF').value; render(); };
  bindCommon(main);
}

async function openSnapshot(id, hl = '', zone = '') {
  const d = await api(`/api/snapshots/${id}`);
  const highlight = hl ? new Set(hl.split(',').filter(Boolean).map(Number)) : null;
  const body = $('#dlgBody');
  const q = d.quality_ok ? '<span class="chip s-ok">годен</span>' : `<span class="chip s-no_data">нет данных: ${esc(d.quality_reason)}</span>`;
  const TS = { form: 'указано при загрузке', exif: 'EXIF снимка', filename: 'имя файла', ocr: 'дата на кадре (OCR)', upload: 'время загрузки' };
  const SRC = { demo: 'демо-разметка', model: 'детектор', llm: 'добавлено ИИ', manual: 'вручную' };
  const a = d.assessment;
  let h = `<button class="btn small close" id="dlgClose">Закрыть ✕</button><h2>${esc(d.camera)} · ${esc(d.camera_name)} · ${fDay(d.taken_at)} ${fTime(d.taken_at)}</h2>`;
  h += `<div class="snapview"><div><div class="row" style="margin-bottom:8px"><label><input type="checkbox" id="tgZ" checked> зоны</label><label><input type="checkbox" id="tgB" checked> рамки</label>`
    + `<label id="tgAIl" hidden><input type="checkbox" id="tgAI" checked> находки ИИ</label>`
    + `<label id="tgOVl" hidden title="Grounding DINO: объекты по текстовым подсказкам"><input type="checkbox" id="tgOV" checked> объекты</label>`
    + (a?.attention ? `<label title="Куда смотрела модель готовности, оценивая стадию"><input type="checkbox" id="tgA"> внимание модели</label>` : '')
    + `<a class="btn small ghost" href="/api/snapshots/${d.id}/image?annotate=1" target="_blank" rel="noopener">разметка JPEG</a><a class="btn small ghost" href="/api/snapshots/${d.id}/image" target="_blank" rel="noopener">оригинал</a></div>`
    + `<div id="bigFrame"></div>`;
  h += `<h3 style="margin-top:12px">Техника на снимке</h3>`;
  h += d.boxes.length ? `<table><thead><tr><th>#</th><th>Класс</th><th>Уверенность</th><th>Зона</th><th>Источник</th></tr></thead><tbody>`
    + d.boxes.map((b) => `<tr${b.rejected ? ' class="muted"' : ''}><td class="mono">${b.id}</td><td>${eqChip(b.cls)}${b.rejected ? ' <span class="hint">исключена</span>' : ''}</td><td>${b.conf.toFixed(2)}${b.strong ? '' : ' <span class="hint">ниже порога</span>'}</td><td>${esc(b.zone ? ((d.project_zones || d.zones).find((z) => z.key === b.zone)?.name || b.zone) : (state.eq[b.cls]?.site_wide ? 'вся площадка' : '—'))}</td><td>${esc(SRC[b.source] || b.source)}</td></tr>`).join('')
    + `</tbody></table>` : `<div class="hint">${d.quality_ok ? 'Техника не обнаружена' : 'Детекция не выполнялась: снимок не прошёл контроль качества'}</div>`;
  h += `</div><div><section class="ai" id="aiBox" data-sid="${d.id}"><h3>Анализ ИИ</h3><div id="aiBody"><span class="spinner"></span></div></section>`
    + `<dl class="kv"><dt>Качество</dt><dd>${q}</dd><dt>Яркость / контраст / резкость</dt><dd>${d.brightness} / ${d.contrast} / ${d.sharpness}</dd>`
    + `<dt>Время съёмки</dt><dd>${fDay(d.taken_at)} ${d.taken_at.slice(11, 19)} <span class="hint">(${TS[d.time_source] || d.time_source})</span></dd>`
    + `<dt>Детектор</dt><dd class="mono">${esc(d.detector)}</dd><dt>Обработка</dt><dd>${d.processing_ms} мс</dd><dt>Файл</dt><dd class="mono">${esc(d.original_name)}</dd>`
    + `<dt>Зоны на кадре</dt><dd><select id="fzSel" class="mini"><option value="">по разметке камеры</option>${(d.project_zones || []).map((z) => `<option value="${esc(z.key)}"${d.frame_zone?.key === z.key ? ' selected' : ''}>весь кадр → ${esc(z.name)}</option>`).join('')}</select>`
    + `<div class="hint">${d.frame_zone ? 'полигоны камеры не применяются: вся техника кадра относится к зоне' : 'зона техники — по точке опоры рамки в полигоне камеры; рамки вне полигонов в проверки зон не идут'}</div></dd>`
    + (a ? `<dt>Стадия по кадру</dt><dd>${assessmentHTML(a)}</dd>` : '')
    + `</dl><h3 style="margin-top:14px">Сопоставление с графиком</h3>`;
  const checks = [...d.checks].sort((x, y) => (x.zone === zone ? -1 : y.zone === zone ? 1 : 0));
  h += checks.length ? checks.map(checkHTML).join('') : '<div class="hint">Камера не привязана к зонам.</div>';
  if (d.deviations.length) h += `<h3>Отклонения, где снимок — доказательство</h3><div class="devlist">${d.deviations.map((x) => devCard(x, { compact: true })).join('')}</div>`;
  h += `</div></div>`;
  body.innerHTML = h;
  const dlg = $('#dlg');
  if (!dlg.open) dlg.showModal();
  $('#dlgClose').onclick = () => dlg.close();
  const view = { aiBoxes: [] };
  const redraw = () => {
    $('#bigFrame').innerHTML = frameHTML(d, { boxes: $('#tgB').checked ? d.boxes : [], zones: $('#tgZ').checked ? d.zones : [], highlight,
      attention: $('#tgA')?.checked ? a.attention : null, aiBoxes: $('#tgAI').checked ? view.aiBoxes : [],
      objBoxes: $('#tgOV').checked ? view.objBoxes || [] : [] });
  };
  ['#tgZ', '#tgB', '#tgA', '#tgAI', '#tgOV'].forEach((s) => { if ($(s)) $(s).onchange = redraw; });
  redraw();
  $('#fzSel').onchange = async () => {
    try {
      await api(`/api/snapshots/${d.id}`, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ frame_zone: $('#fzSel').value || null }) });
      toast('Зоны снимка изменены, выводы пересчитаны');
      openSnapshot(d.id);
      if (state.tab !== 'analyze') render();
    } catch (e) { toast(e.message); }
  };
  bindCommon(body);
  loadAI(d, view, redraw);
}

function assessmentHTML(a) {
  const plan = a.expected != null
    ? `<div>план по графику ${a.expected}% · <b class="${a.status === 'late' || a.status === 'risk' ? 'bad-mark' : 'ok-mark'}">${esc(a.status_ru)}</b> (${a.delta_pp > 0 ? '+' : ''}${a.delta_pp} п.п., ≈ ${a.delta_days > 0 ? '+' : ''}${a.delta_days} дн.)</div>` : '';
  const extra = [a.stage_prob != null && `уверенность ${Math.round(a.stage_prob * 100)}%`,
    a.neighbors_readiness != null && `похожие кадры обучения ≈ ${a.neighbors_readiness}%`].filter(Boolean).join(' · ');
  return `${esc(STAGES[a.stage] || a.stage)}, готовность ${a.readiness}%${plan}${extra ? `<div class="hint">${esc(extra)}</div>` : ''}`
    + `${a.note ? `<div class="hint">${esc(a.note)}</div>` : ''}<div class="hint">${esc(a.model || '')}</div>`;
}

// ------------------------------------------------------------------ ИИ-анализ снимка
const VERDICT = { confirmed: ['ok', '✓ подтверждена'], false_positive: ['bad', '✗ ложная рамка'], wrong_class: ['warn', '↻ другой класс'],
  uncertain: ['muted', '? не разобрать'], not_checked: ['muted', 'не проверялась'] };
const DVERDICT = { confirmed: ['bad', 'подтверждено'], doubtful: ['warn', 'сомнительно'], rejected: ['ok', 'не подтверждено'], not_checked: ['muted', 'не проверялось'] };
const vchip = ([cls, text]) => `<span class="vchip ${cls}">${esc(text)}</span>`;

// рамки Grounding DINO для кадра: прочие объекты и техника, которой нет у детектора
function ovBoxes(ov) {
  if (!ov) return [];
  return [...(ov.objects || []).map((o) => ({ xyxy: o.xyxy, label: o.label, conf: o.conf })),
    ...(ov.only_open_vocab || []).map((o) => ({ xyxy: o.xyxy, label: `GDINO: ${clsName(o.cls)}`, conf: o.conf }))];
}

function aiLayersHint() {
  const a = state.ai;
  if (!a) return '';
  const l = [];
  l.push(a.readiness?.enabled ? 'модель готовности (DINOv2 + голова)' : null);
  l.push(a.open_vocab?.enabled ? 'Grounding DINO (опалубка, леса, рабочие, техника по подсказкам)' : null);
  l.push(a.vlm?.enabled ? `локальная VLM ${a.vlm.model || '(не помещается в память)'}` : null);
  l.push(a.llm?.provider !== 'off' ? `${a.llm.name} ${a.llm.model}${a.llm.vision ? '' : ' (текст: судит по VLM и рамкам)'}` : null);
  return l.filter(Boolean).join(' → ');
}

async function loadAI(d, view, redraw) {
  const box = $('#aiBody');
  if (!box || $('#aiBox').dataset.sid !== String(d.id)) return;
  if (!state.ai?.enabled) {
    box.innerHTML = `<div class="hint">ИИ-анализ выключен. Включите слои: модель готовности — веса в <span class="mono">weights/readiness/</span>, локальная VLM — <span class="mono">OKO_VLM=auto</span>, YandexGPT — <span class="mono">OKO_LLM_PROVIDER=yandex</span> и <span class="mono">OKO_YC_FOLDER_ID</span>.</div>`;
    return;
  }
  let r;
  try { r = await api(`/api/snapshots/${d.id}/ai-review`); } catch (e) { box.innerHTML = `<div class="hint">Ошибка: ${esc(e.message)}</div>`; return; }
  renderAI(d, r, view, redraw);
  if (r && (r.status === 'queued' || r.status === 'running')) {
    setTimeout(() => { if ($('#dlg').open) loadAI(d, view, redraw); }, 2500);
  }
}

function renderAI(d, r, view, redraw) {
  const box = $('#aiBody');
  const start = (label) => `<button class="btn small primary" id="aiRun">${label}</button>`;
  view.aiBoxes = [];
  view.objBoxes = [];
  if (!r) {
    box.innerHTML = `<p class="hint">Слои: ${esc(aiLayersHint())}. Модели сверяют друг друга, LLM исправляет ошибки детектора и правил, прогнозирует и даёт рекомендации.</p>${start('Запустить анализ ИИ')}`;
  } else if (r.status === 'queued' || r.status === 'running') {
    box.innerHTML = `<p><span class="spinner"></span> ${esc(r.step || 'в очереди')}…</p><p class="hint">На CPU модель готовности и VLM работают до минуты-двух; окно можно закрыть — анализ продолжится.</p>`;
  } else if (r.status === 'error' && !r.final) {
    box.innerHTML = `<p class="bad-mark">Ошибка анализа</p><p class="hint">${esc(r.error)}</p>${start('Повторить')}`;
  } else {
    box.innerHTML = aiResultHTML(d, r) + `<div class="row" style="margin-top:8px">${start('Повторить анализ')}</div>`;
    const f = r.final;
    view.aiBoxes = r.applied && !r.applied.reverted ? [] : (f.missed || []);
    view.objBoxes = ovBoxes(f.open_vocab);
  }
  $('#tgAIl').hidden = !view.aiBoxes.length;
  $('#tgOVl').hidden = !view.objBoxes.length;
  redraw();
  const run = $('#aiRun');
  if (run) run.onclick = async () => {
    run.disabled = true;
    try { await api(`/api/snapshots/${d.id}/ai-review`, { method: 'POST' }); loadAI(d, view, redraw); } catch (e) { toast(e.message); run.disabled = false; }
  };
  const act = async (url, msg) => {
    try { await api(url, { method: 'POST' }); toast(msg); openSnapshot(d.id); if (state.tab === 'summary' || state.tab === 'deviations') render(); } catch (e) { toast(e.message); }
  };
  if ($('#aiApply')) $('#aiApply').onclick = () => act(`/api/ai/reviews/${r.id}/apply`, 'Исправления применены, отклонения пересчитаны');
  if ($('#aiRevert')) $('#aiRevert').onclick = () => act(`/api/ai/reviews/${r.id}/revert`, 'Исправления отменены');
}

function aiResultHTML(d, r, { quick = false } = {}) {
  const f = r.final, L = f.layers || {};
  const src = [L.readiness_model && 'модель готовности', L.open_vocab_model && 'Grounding DINO', L.vlm_model && `VLM ${L.vlm_model.split('/').pop()}`,
    L.llm_model && `${L.llm_name || L.llm_provider} ${L.llm_model}`].filter(Boolean);
  let h = `<div class="ai-head">${src.map((x) => `<span class="chip plain s-info">${esc(x)}</span>`).join('')}`
    + `${f.confidence != null ? `<span class="hint">уверенность ${Math.round(f.confidence * 100)}%</span>` : ''}`
    + `<span class="hint">${(r.duration_ms / 1000).toFixed(1)} с${r.llm?.tokens_in ? ` · токенов ${r.llm.tokens_in}+${r.llm.tokens_out}` : ''}</span></div>`;
  const TL = { readiness: 'модель готовности', open_vocab: 'Grounding DINO', vlm: 'VLM', llm: L.llm_name || 'LLM' };
  const hit = new Set(L.cached || []);
  const tline = Object.entries(L.timings || {}).map(([k, v]) => `${TL[k] || k} ${hit.has(k) ? 'из кеша' : `${v} с`}`).join(' · ');
  if (tline) h += `<div class="hint">время слоёв: ${esc(tline)}</div>`;
  if (f.summary) h += `<p class="ai-sum">${esc(f.summary)}</p>`;
  if (r.error) h += `<p class="hint">Не все слои отработали: ${esc(r.error)}</p>`;
  // техника: детектор / VLM / LLM
  if (f.equipment?.length) {
    const gd = f.equipment.some((e) => e.open_vocab != null);
    h += `<table class="eqtab"><thead><tr><th>Техника</th><th>Детектор</th>${gd ? '<th title="Grounding DINO по текстовым подсказкам">GDINO</th>' : ''}<th>VLM</th><th>ИИ-итог</th></tr></thead><tbody>`
      + f.equipment.map((e) => `<tr class="${e.agree ? '' : 'dis'}"><td>${eqChip(e.cls)}</td><td>${e.detector}</td>${gd ? `<td>${e.open_vocab ?? '—'}</td>` : ''}<td>${e.vlm ?? '—'}</td><td><b>${e.final}</b>${e.agree ? '' : ' <span class="vchip warn">расхождение</span>'}</td></tr>`).join('')
      + `</tbody></table>`;
  }
  // стадия по всем слоям
  const st = f.stage || {};
  const parts = [st.planned?.length && `график ${st.planned.join('/')}`,
    st.model && `модель готовности ${st.model}${st.model_readiness != null ? ` (${st.model_readiness}%${st.model_expected != null ? `, план ${st.model_expected}%` : ''})` : ''}`,
    st.vlm && `VLM ${st.vlm}`, st.llm && `${L.llm_name || 'LLM'} ${st.llm}`].filter(Boolean);
  if (parts.length) {
    h += `<div class="ai-row"><span class="lbl">стадия:</span> ${esc(parts.join(' · '))}${st.final ? ` → <b>${esc(STAGES[st.final] || st.final)}</b>` : ''}`
      + `${st.final ? (st.agree ? ' <span class="vchip ok">слои согласны</span>' : ' <span class="vchip warn">слои расходятся</span>') : ''}`
      + `${st.matches_plan === false ? ' <span class="vchip bad">не совпадает с графиком</span>' : ''}${st.comment ? `<div class="hint">${esc(st.comment)}</div>` : ''}</div>`;
  }
  const ov = f.open_vocab;
  if (ov) {
    const objs = Object.entries(ov.counts || {});
    const only = (ov.only_open_vocab || []).reduce((a, o) => ({ ...a, [o.cls]: (a[o.cls] || 0) + 1 }), {});
    h += `<div class="ai-row"><span class="lbl">Grounding DINO:</span> ${objs.length ? objs.map(([k, n]) => `<span class="vchip obj">${esc(k)} ×${n}</span>`).join(' ') : '<span class="hint">прочих объектов нет</span>'}`
      + `<div class="hint">техника: подтвердил рамок детектора ${ov.detector_confirmed.length}`
      + `${Object.keys(only).length ? ` · только у Grounding DINO: ${Object.entries(only).map(([c, n]) => `${esc(clsName(c))} ×${n}`).join(', ')}` : ''}`
      + `${ov.class_conflict?.length ? ` · спорный класс: ${ov.class_conflict.map((c) => `#${c.detection_id} ${esc(clsName(c.detector_cls))} → ${esc(clsName(c.open_vocab_cls))}`).join(', ')}` : ''}`
      + ` · ${ov.seconds ?? '—'} с</div></div>`;
  }
  const cc = f.cross_check;
  if (cc) {
    h += `<div class="ai-row"><span class="lbl">детектор ↔ VLM:</span> совпали ${cc.agree.length} · спорный класс ${cc.class_conflict.length} · только VLM ${cc.vlm_only.length} · только детектор ${cc.detector_only.length}</div>`;
  }
  // вердикты по рамкам детектора
  const checked = (f.detections || []).filter((x) => x.verdict !== 'not_checked');
  if (checked.length) {
    h += `<div class="ai-list"><div class="lbl">рамки детектора:</div>` + checked.map((x) => `<div><span class="mono">#${x.id}</span> ${esc(x.name)} ${x.conf.toFixed(2)} ${vchip(VERDICT[x.verdict] || ['muted', x.verdict])}`
      + `${x.correct_name ? ` → <b>${esc(x.correct_name)}</b>` : ''}${x.comment ? ` <span class="hint">${esc(x.comment)}</span>` : ''}</div>`).join('') + `</div>`;
  }
  if (f.missed?.length) {
    h += `<div class="ai-list"><div class="lbl">пропустил детектор:</div>` + f.missed.map((x) => `<div><span class="vchip ai">+ ${esc(x.name)} ${x.conf.toFixed(2)}</span>${x.comment ? ` <span class="hint">${esc(x.comment)}</span>` : ''}</div>`).join('') + `</div>`;
  }
  const devs = (f.deviations || []).filter((x) => x.verdict !== 'not_checked');
  if (devs.length) {
    h += `<div class="ai-list"><div class="lbl">отклонения правил:</div>` + devs.map((x) => `<div>${esc(x.title)} ${vchip(DVERDICT[x.verdict] || ['muted', x.verdict])}${x.comment ? ` <span class="hint">${esc(x.comment)}</span>` : ''}</div>`).join('') + `</div>`;
  }
  if (f.new_findings?.length) {
    h += `<div class="ai-list"><div class="lbl">новые находки:</div>` + f.new_findings.map((x) => `<div>${sevChip(x.severity)} <b>${esc(x.title)}</b>${x.zone ? ` · ${esc(x.zone)}` : ''} <span class="hint">${esc(x.reason || '')}</span></div>`).join('') + `</div>`;
  }
  if (f.forecast) h += `<div class="ai-row"><span class="lbl">прогноз:</span> ${esc(f.forecast)}</div>`;
  if (f.recommendations?.length) h += `<div class="ai-list"><div class="lbl">рекомендации:</div><ol>${f.recommendations.map((x) => `<li>${esc(x)}</li>`).join('')}</ol></div>`;
  if (f.scene?.vlm_scene) h += `<div class="ai-row"><span class="lbl">VLM видит:</span> ${esc(f.scene.vlm_scene)}${f.scene.vlm_notes ? ` <span class="hint">${esc(f.scene.vlm_notes)}</span>` : ''}</div>`;
  // исправления
  const corr = f.corrections || [];
  if (corr.length) {
    const ap = r.applied && !r.applied.reverted;
    const n = corr.filter((c) => c.apply).length;
    h += `<div class="ai-fix"><div class="lbl">${quick ? 'исправления ИИ' : `исправления данных${ap ? ' — применены' : ''}`}:</div>`
      + corr.map((c) => `<div>${c.apply ? '<span class="ok-mark">●</span>' : '<span class="wait-mark">○</span>'} ${esc(c.text)}${c.apply ? '' : ` <span class="hint">уверенность ниже ${f.min_conf}</span>`}</div>`).join('')
      + (quick ? `<div class="hint" style="margin-top:4px">В быстрой проверке снимок не сохраняется — исправления только показываются. Чтобы применить их, загрузите снимок в проект.</div>`
        : `<div class="row" style="margin-top:6px">${ap ? `<button class="btn small" id="aiRevert">Отменить исправления</button><span class="hint">применены ${esc(r.applied.at.replace('T', ' '))} UTC</span>`
          : n ? `<button class="btn small good" id="aiApply">Применить исправления (${n})</button><span class="hint">рамки изменятся, отклонения пересчитаются; можно отменить</span>` : ''}</div>`) + `</div>`;
  }
  h += `<details class="calc"><summary>Что видели и ответили модели</summary><pre>${esc(JSON.stringify({ vlm: r.vlm?.output, stage: f.stage, cross_check: f.cross_check }, null, 1))}</pre>`
    + `<a href="${quick ? `/api/analyze/ai/${r.id}` : `/api/ai/reviews/${r.id}`}?full=1" target="_blank" rel="noopener">все данные, отправленные в LLM, и её ответ (JSON)</a></details>`;
  return h;
}

// ------------------------------------------------------------------ ИИ-анализ площадки за день
async function loadDayAI(dayAtStart) {
  const box = $('#aiDay');
  if (!box || state.tab !== 'summary' || state.day !== dayAtStart) return;
  if (!state.ai?.llm || state.ai.llm.provider === 'off') {
    box.innerHTML = `<div class="ai-day-h"><h2>Анализ ИИ за день</h2></div><p class="hint">Итоговый анализ площадки делает LLM: включите YandexGPT — <span class="mono">OKO_LLM_PROVIDER=yandex</span>, <span class="mono">OKO_YC_FOLDER_ID</span> и API-ключ или сервисный аккаунт ВМ.</p>`;
    return;
  }
  let r = null;
  try { r = await api(`/api/projects/${state.pid}/ai-summary?day=${state.day}`); } catch (e) { /* нет анализа */ }
  const busy = r && (r.status === 'queued' || r.status === 'running');
  let h = `<div class="ai-day-h"><h2>Анализ ИИ за день · ${esc(state.ai.llm.name)}</h2>`
    + (busy ? '' : `<button class="btn small primary" id="aiDayRun">${r ? 'Обновить анализ' : 'Проанализировать день'}</button>`) + `</div>`;
  if (!r) h += `<p class="hint">${esc(state.ai.llm.name)} получит доску зон, отклонения правил, оценки модели готовности, итоги ИИ-анализа снимков и историю по дням — и вернёт статус, риски срыва сроков, прогноз задержек, ошибки моделей и рекомендации.</p>`;
  else if (busy) h += `<p><span class="spinner"></span> ${esc(r.step || 'в очереди')}…</p>`;
  else if (r.status === 'error') h += `<p class="bad-mark">Ошибка</p><p class="hint">${esc(r.error)}</p>`;
  else h += dayAIHTML(r);
  box.innerHTML = h;
  if ($('#aiDayRun')) $('#aiDayRun').onclick = async () => {
    try { await api(`/api/projects/${state.pid}/ai-summary?day=${state.day}`, { method: 'POST' }); loadDayAI(dayAtStart); } catch (e) { toast(e.message); }
  };
  if (busy) setTimeout(() => loadDayAI(dayAtStart), 3000);
}

function dayAIHTML(r) {
  const f = r.final, L = f.layers || {};
  const ST = { 'по графику': 's-ok', 'есть риски': 's-warning', 'отставание': 's-critical' };
  const IMP = { 'высокое': 's-critical', 'среднее': 's-warning', 'низкое': 's-info' };
  let h = `<div class="row"><span class="chip ${ST[f.status] || 's-info'}">${esc(f.status || '')}</span>`
    + `<span class="hint">${esc(L.llm_model || '')} · снимков с ИИ-анализом ${L.snapshots_reviewed ?? 0} · оценок модели готовности ${L.readiness_frames ?? 0} · ${(r.duration_ms / 1000).toFixed(1)} с · уверенность ${Math.round((f.confidence || 0) * 100)}%</span></div>`;
  h += `<p class="ai-sum">${esc(f.summary || '')}</p><div class="cols2">`;
  h += `<div>${f.risks?.length ? `<div class="lbl">риски срыва сроков</div><table class="eqtab"><thead><tr><th>Риск</th><th>Вероятность</th><th>Влияние</th><th>Что сделать</th></tr></thead><tbody>`
    + f.risks.map((x) => `<tr><td><b>${esc(x.title)}</b>${x.zone || x.task ? `<div class="hint">${esc([x.zone, x.task].filter(Boolean).join(' · '))}</div>` : ''}<div class="hint">${esc(x.reason || '')}</div></td><td>${Math.round((x.probability || 0) * 100)}%</td><td><span class="chip plain ${IMP[x.impact] || ''}">${esc(x.impact || '')}</span></td><td>${esc(x.mitigation || '')}</td></tr>`).join('')
    + `</tbody></table>` : '<p class="hint">Рисков не выявлено.</p>'}</div>`;
  h += `<div>${f.forecast?.length ? `<div class="lbl">прогноз по этапам</div>` + f.forecast.map((x) => `<div class="ai-row">${esc(x.task)}: <b>${x.expected_delay_days > 0 ? `+${x.expected_delay_days} дн.` : 'в срок'}</b> <span class="hint">${esc(x.reason || '')}</span></div>`).join('') : ''}`
    + `${f.model_errors?.length ? `<div class="lbl" style="margin-top:8px">где модели и правила, вероятно, ошиблись</div><ul>${f.model_errors.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}`
    + `${f.recommendations?.length ? `<div class="lbl" style="margin-top:8px">рекомендации</div><ol>${f.recommendations.map((x) => `<li>${esc(x)}</li>`).join('')}</ol>` : ''}</div></div>`;
  return h;
}

function checkHTML(c) {
  const zname = c.zone === '__site__' ? 'Вся площадка' : c.zone_name;
  const zc = state.project.zones.find((z) => z.key === c.zone);
  let h = `<div class="check"><div class="row" style="justify-content:space-between">${zc ? zoneChip(zc) : `<b>${esc(zname)}</b>`}<span class="chip s-${c.status === 'deviation' ? 'warning' : c.status}">${c.status === 'deviation' ? 'есть отклонения' : esc(ZSTATUS[c.status] || c.status)}</span></div>`;
  const seen = Object.entries(c.observed || {});
  h += `<div class="g"><span class="lbl">видно:</span>${seen.length ? seen.map(([k, n]) => eqChip(k, n)).join('') : '<span class="hint">техники нет</span>'}</div>`;
  if (!c.tasks.length) h += `<div class="hint">Идущих этапов нет.</div>`;
  for (const t of c.tasks) {
    h += `<div style="margin-top:6px"><b class="mono">${esc(t.wbs || '')}</b> ${esc(t.name)} <span class="hint">· ${esc(t.profile_name)} · наблюдаемость ${OBS[t.observability] || ''}</span></div>`;
    if (t.status === 'summary') { h += `<div class="hint">сводный этап: проверяются подэтапы</div>`; continue; }
    if (t.status === 'not_checked') { h += `<div class="hint">${esc(t.reason || 'техника этапа не проверяется')}</div>`; }
    for (const g of t.groups) {
      const mark = { ok: '<span class="ok-mark">✓</span>', missing: '<span class="bad-mark">✗</span>', pending: '<span class="wait-mark">…</span>',
        not_detectable: '<span class="wait-mark">?</span>', site_equipment: '<span class="ok-mark">✓</span>', reference: '<span class="wait-mark">·</span>' }[g.state] || '';
      const note = { pending: 'проверяется за смену', not_detectable: 'модель пока не распознаёт', site_equipment: 'заявлена на площадке', reference: 'справочно' }[g.state] || '';
      h += `<div class="g">${mark} <span>${esc(g.role)}:</span> <b>${esc(g.expected)}</b>${g.window === 'shift' ? ' <span class="hint">(за смену)</span>' : ''}${note ? ` <span class="hint">${note}</span>` : ''}</div>`;
    }
    for (const p of t.companions) {
      const mark = p.state === 'violated' ? '<span class="bad-mark">✗</span>' : p.state === 'ok' ? '<span class="ok-mark">✓</span>' : '<span class="wait-mark">·</span>';
      h += `<div class="g">${mark} пара: ${esc(p.lead)} → ${esc(p.partner)}${p.state === 'no_lead' ? ' <span class="hint">ведущей техники нет</span>' : ''}</div>`;
    }
    for (const p of t.planned || []) {
      if (p.state === 'not_checked') continue;
      h += `<div class="g">${p.state === 'ok' ? '<span class="ok-mark">✓</span>' : '<span class="bad-mark">✗</span>'} по графику ${p.planned} × ${esc(clsName(p.cls))}, видно ${p.observed}</div>`;
    }
  }
  for (const u of c.unexpected || []) {
    const V = { EARLY_START: `этап «${u.task}» начнётся ${fDay(u.start)}`, STAGE_OVERRUN: `этап «${u.task}» закончился ${fDay(u.end)}`,
      UNEXPECTED_EQUIPMENT: 'не нужен идущим этапам', NO_ACTIVE_STAGE: 'в зоне нет работ по графику' };
    h += `<div class="g"><span class="bad-mark">!</span> ${eqChip(u.cls, u.count)} ${esc(V[u.verdict] || u.verdict)}</div>`;
  }
  for (const f of c.findings || []) h += `<div class="g">${sevChip(f.severity)} ${esc(f.title)}</div>`;
  return h + `</div>`;
}

// ------------------------------------------------------------------ отклонения и нарушения
async function vDeviations(main) {
  const F = state.filters;
  const qs = new URLSearchParams({ day: state.day, include_rejected: 'true' });
  if (F.sev) qs.set('severity', F.sev);
  if (F.status) qs.set('status', F.status);
  if (F.zone) qs.set('zone', F.zone);
  const [list, vio] = await Promise.all([api(`/api/projects/${state.pid}/deviations?${qs}`), api(`/api/projects/${state.pid}/violations`)]);
  const opt = (v, t, cur) => `<option value="${v}"${v === cur ? ' selected' : ''}>${t}</option>`;
  let h = `<div class="filters">`
    + `<label class="sel"><span>Важность</span><select id="fSev">${opt('', 'любая', F.sev)}${opt('critical', 'критично', F.sev)}${opt('warning', 'внимание', F.sev)}${opt('info', 'к сведению', F.sev)}</select></label>`
    + `<label class="sel"><span>Статус</span><select id="fSt">${opt('', 'любой', F.status)}${opt('confirmed', 'подтверждено', F.status)}${opt('preliminary', 'предварительно', F.status)}</select></label>`
    + `<label class="sel"><span>Зона</span><select id="fZone">${opt('', 'все зоны', F.zone)}${state.project.zones.map((z) => opt(z.key, esc(z.name), F.zone)).join('')}</select></label>`
    + `<a class="btn ghost" href="/api/projects/${state.pid}/report?day=${state.day}&format=csv">Отчёт CSV</a>`
    + `<a class="btn ghost" href="/api/projects/${state.pid}/report?day=${state.day}" target="_blank" rel="noopener">JSON</a></div>`;
  h += `<p class="hint">Отклонение по одному снимку — «предварительно»; серия снимков за день (или очень уверенная рамка лишней техники) делает его «подтверждённым». Инженер подтверждает отклонение — оно становится нарушением в реестре; ложные рамки исключаются из пересчёта и помечаются для дообучения детектора.</p>`;
  h += `<div class="devlist">${list.length ? list.map((d) => devCard(d)).join('') : '<div class="card empty">Отклонений нет.</div>'}</div>`;
  h += `<h2 style="margin-top:24px">Реестр нарушений · ${vio.length}</h2>`;
  h += vio.length ? `<div class="card"><table><thead><tr><th>№</th><th>Нарушение</th><th>Зона</th><th>Этап</th><th>Подрядчик</th><th>Важность</th><th>Срок</th><th>Статус</th></tr></thead><tbody>`
    + vio.map((v) => `<tr><td class="mono">${esc(v.number)}</td><td>${esc(v.title)}<div class="hint">${esc(v.description)}</div></td><td>${esc(v.zone)}</td><td>${esc(v.task || '—')}</td><td>${esc(v.contractor || '—')}</td><td>${sevChip(v.severity)}</td><td>${fDay(v.due_date)}</td><td>${v.status === 'open' ? 'открыто' : 'закрыто'}</td></tr>`).join('')
    + `</tbody></table></div>` : `<div class="card empty">Нарушений пока нет: подтвердите отклонение кнопкой «Подтвердить → нарушение».</div>`;
  main.innerHTML = h;
  $('#fSev').onchange = (e) => { F.sev = e.target.value; render(); };
  $('#fSt').onchange = (e) => { F.status = e.target.value; render(); };
  $('#fZone').onchange = (e) => { F.zone = e.target.value; render(); };
  bindCommon(main);
}

// ------------------------------------------------------------------ график
async function vSchedule(main) {
  const tasks = await api(`/api/projects/${state.pid}/schedule?day=${state.day}`);
  if (!tasks.length) {
    main.innerHTML = `<div class="card empty">График не загружен.<br><br><input type="file" id="schedFile" accept=".xlsx,.csv"> <button class="btn primary" id="schedUp">Загрузить график</button></div>`;
    $('#schedUp').onclick = uploadSchedule;
    return;
  }
  const t0 = Math.min(...tasks.map((t) => Date.parse(t.start))), t1 = Math.max(...tasks.map((t) => Date.parse(t.end)));
  const pos = (d) => ((Date.parse(d) - t0) / Math.max(1, t1 - t0) * 100).toFixed(2);
  const sch = state.project.schedule;
  let h = `<div class="filters"><span class="hint">Версия графика ${sch.version} · ${esc(sch.file)} · строк ${sch.rows}, сопоставлено со справочником ${sch.matched}. Строка графика → вид работ справочника → профиль техники → проверки на снимках.</span>`
    + `<span style="margin-left:auto"></span><input type="file" id="schedFile" accept=".xlsx,.csv"><button class="btn" id="schedUp">Загрузить новую версию</button></div>`;
  h += `<div class="card"><table class="sched"><thead><tr><th style="width:80px">Код</th><th>Этап графика → вид работ справочника</th><th style="width:170px">Зона</th><th style="width:260px">Сроки · красная линия — ${fDay(state.day)}</th><th>Техника: по методике и в графике</th></tr></thead><tbody>`;
  for (const t of tasks) {
    const prof = t.profile;
    const req = prof && !t.summary ? prof.required.map((g) => `<span class="eq" title="${esc(g.role)}">${esc(g.any_of.map(clsName).join(' / '))}${g.window === 'shift' ? ' <span class="muted">за смену</span>' : ''}</span>`).join(' ') : '';
    const comp = prof && !t.summary ? prof.companions.map((c) => `<div class="hint">пара: ${esc(c.lead.map(clsName).join('/'))} → ${esc(c.partner.map(clsName).join('/'))}</div>`).join('') : '';
    const how = t.match.method === 'code' ? 'по коду' : t.match.method === 'name' ? `по названию, ${Math.round(t.match.score * 100)}%` : 'вручную';
    const wt = t.work_type
      ? `<div class="hint">→ <span class="mono">${esc(t.work_type.id)}</span> ${[esc(t.work_type.name), esc(prof?.name || ''), esc(t.work_type.stage || ''), `наблюдаемость ${OBS[t.work_type.observability]}`, how].filter(Boolean).join(' · ')}</div>${t.work_type.expert_note ? `<div class="note">Замечание экспертов: ${esc(t.work_type.expert_note)}</div>` : ''}`
      : `<div class="row" style="margin-top:3px"><span class="chip s-warning">нет в справочнике</span><button class="btn small" data-map="${t.id}">Указать вид работ</button></div>`;
    const planned = Object.entries(t.planned || {});
    h += `<tr class="${t.active ? 'active' : ''}${t.summary ? ' summary' : ''}"><td class="mono">${esc(t.wbs)}</td>`
      + `<td><div class="tname">${esc(t.name)}${t.summary ? ' <span class="hint">· сводный</span>' : ''}</div>${wt}${t.contractor ? `<div class="hint">${esc(t.contractor)}</div>` : ''}</td>`
      + `<td>${t.zone ? zoneChip(t.zone) : '<span class="hint">вся площадка</span>'}</td>`
      + `<td><div class="gantt"><div class="bar${t.active ? ' act' : ''}" style="left:${pos(t.start)}%;width:${Math.max(0.8, pos(t.end) - pos(t.start))}%"></div><div class="today" style="left:${pos(state.day)}%"></div></div><div class="hint">${fDay(t.start)} – ${fDay(t.end)}</div></td>`
      + `<td>${t.summary ? '<span class="hint">проверяются подэтапы</span>' : `<div class="pill-list">${req || '<span class="hint">обязательной техники нет</span>'}</div>${comp}`}`
      + `${planned.length ? `<div class="pill-list" style="margin-top:4px"><span class="lbl">в графике:</span>${planned.map(([c, n]) => eqChip(c, n, state.eq[c] && !state.eq[c].detected_now ? 'nd' : '')).join(' ')}</div>` : ''}</td></tr>`;
  }
  h += `</tbody></table></div><p class="hint">Пунктирная рамка у техники — класс пока не распознаётся текущей моделью (например, автобетононасос): требование видно, но в проверку не входит.</p>`;
  main.innerHTML = h;
  $('#schedUp').onclick = uploadSchedule;
  $$('[data-map]', main).forEach((b) => { b.onclick = () => mapTask(Number(b.dataset.map)); });
}

async function uploadSchedule() {
  const f = $('#schedFile').files[0];
  if (!f) { toast('Выберите файл графика (.xlsx или .csv)'); return; }
  const fd = new FormData(); fd.append('file', f);
  const r = await api(`/api/projects/${state.pid}/schedule`, { method: 'POST', body: fd });
  toast(`График v${r.version}: строк ${r.rows}, сопоставлено ${r.matched}${r.warnings.length ? `, замечаний ${r.warnings.length}` : ''}`, 6000);
  await loadProject();
}

async function mapTask(id) {
  const q = prompt('Вид работ справочника: введите код (например 12.3.1 или 12.3.7-1) — полный список на вкладке «Методика»');
  if (!q) return;
  try {
    await api(`/api/tasks/${id}`, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ work_type_id: q.trim() }) });
    toast('Этап сопоставлен, выводы пересчитаны'); render();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ методика
async function vMethodology(main) {
  if (!state.workTypes) {
    [state.workTypes, state.profiles] = await Promise.all([api('/api/reference/work-types?limit=1000'), api('/api/reference/profiles')]);
  }
  const wts = state.workTypes;
  const cnt = (k) => wts.filter((w) => w.observability === k).length;
  const OT = Object.fromEntries(state.rules.object_types.map((o) => [o.key, o.name]));
  let h = `<div class="kpis">`
    + [['Видов работ', wts.length, 'справочник организаторов'], ['Профилей техники', state.profiles.length, '+ составные для групп'],
      ['Высокая наблюдаемость', cnt('high'), 'нет техники — нет работ'], ['Средняя', cnt('medium'), 'техника хотя бы раз за смену'],
      ['Низкая', cnt('low'), 'только лишняя техника'], ['Не видно с камер', cnt('none'), 'внутри здания, документы']]
      .map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('') + `</div>`;
  h += `<div class="card" style="margin-bottom:14px"><b>Как читать методику.</b> Каждый вид работ справочника ссылается на профиль техники. Профиль задаёт обязательную технику (роль и классы, «на каждом снимке» или «за смену»), парные правила (ведущая → парная: экскаватор → самосвал) и допустимую технику. Всё остальное в зоне — техника, не соответствующая этапу. Наблюдаемость определяет, проверяется ли отсутствие техники.</div>`;
  h += `<div class="filters"><label class="sel"><span>Поиск</span><input id="mQ" placeholder="код или название" value=""></label>`
    + `<label class="sel"><span>Тип объекта</span><select id="mOT"><option value="">все</option>${state.rules.object_types.map((o) => `<option value="${o.key}">${esc(o.name)}</option>`).join('')}</select></label>`
    + `<label class="sel"><span>Стадия</span><select id="mSt"><option value="">все</option>${Object.entries(STAGES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></label>`
    + `<label class="sel"><span>Наблюдаемость</span><select id="mObs"><option value="">любая</option>${Object.entries(OBS).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></label>`
    + `<a class="btn ghost" href="/api/reference/methodology.xlsx">Скачать методику (XLSX)</a></div>`;
  h += `<div class="card scroll" id="mTable"></div>`;
  h += `<h2 style="margin-top:22px">Правила выявления отклонений · версия ${esc(state.rules.version)}</h2><div class="card"><table><thead><tr><th>Код</th><th>Отклонение</th><th>1 снимок</th><th>Серия</th><th>Шаблон предупреждения</th></tr></thead><tbody>`
    + state.rules.rules.map((r) => `<tr><td class="mono">${r.code}</td><td><b>${esc(r.title)}</b><div class="hint">${esc(r.recommendation)}</div></td><td>${sevChip(r.severity)}</td><td>${sevChip(r.confirmed_severity)}</td><td>${esc(r.message)}</td></tr>`).join('')
    + `</tbody></table><p class="hint">Пороги: уверенность для «лишней техники» ≥ ${state.rules.config.detection.presence_threshold}, для правил отсутствия засчитывается рамка ≥ ${state.rules.config.detection.weak_presence_threshold}; подтверждение — ${state.rules.config.windows.confirm_min_snapshots} снимка; парное правило — доля снимков ≥ ${state.rules.config.windows.incomplete_share}; опережение — этап начнётся в пределах ${state.rules.config.windows.early_start_horizon_days} дн.; «этап не завершён» — закончился не раньше ${state.rules.config.windows.overrun_lookback_days} дн. назад.</p></div>`;
  h += `<h2 style="margin-top:22px">Классы техники</h2><div class="card"><table><thead><tr><th>Класс</th><th>В перечне ТЗ</th><th>Версии модели</th><th>Распознаёт текущий детектор</th></tr></thead><tbody>`
    + Object.values(state.eq).map((e) => `<tr><td>${eqChip(e.key)}</td><td>${e.in_tz ? 'да' : '—'}</td><td class="mono">${esc(e.models.join(', '))}</td><td>${e.detected_now ? '<span class="chip s-ok">да</span>' : '<span class="chip s-no_tasks">нет</span>'}</td></tr>`).join('') + `</tbody></table></div>`;
  h += `<h2 style="margin-top:22px">Профили техники · ${state.profiles.length}</h2><div class="profiles">`
    + state.profiles.map((p) => `<details class="profile"><summary>${esc(p.name)} <span class="hint mono">${p.id}</span></summary><p class="hint">${esc(p.description)} · наблюдаемость ${OBS[p.observability]}</p>`
      + (p.required.length ? `<div class="g"><span class="lbl">нужна:</span> ${p.required.map((g) => `<div>${esc(g.role)}: ${g.any_of.map((c) => eqChip(c)).join(' ')}${g.window === 'shift' ? ' <span class="hint">за смену</span>' : ''}</div>`).join('')}</div>` : '<div class="hint">обязательной техники нет</div>')
      + p.companions.map((c) => `<div class="hint">пара: ${esc(c.lead.map(clsName).join('/'))} → ${esc(c.partner.map(clsName).join('/'))} — ${esc(c.message)}</div>`).join('')
      + `<div class="pill-list" style="margin-top:6px"><span class="lbl">допустимо:</span>${p.allowed.map((c) => eqChip(c)).join('') || '—'}</div></details>`).join('') + `</div>`;
  main.innerHTML = h;
  const draw = () => {
    const q = $('#mQ').value.trim().toLowerCase(), ot = $('#mOT').value, st = $('#mSt').value, ob = $('#mObs').value;
    const rows = wts.filter((w) => (!q || w.name.toLowerCase().includes(q) || w.id.startsWith(q)) && (!ot || w.object_types.includes(ot)) && (!st || w.stage === st) && (!ob || w.observability === ob));
    $('#mTable').innerHTML = `<table><thead><tr><th>ID</th><th>Вид работ</th><th>Стадия</th><th>Профиль</th><th>Наблюдаемость</th><th>Обязательная техника</th><th>Парные правила</th></tr></thead><tbody>`
      + rows.map((w) => `<tr><td class="mono">${esc(w.id)}</td><td style="padding-left:${8 + (w.level - 1) * 14}px">${w.has_children ? '<b>' : ''}${esc(w.name)}${w.has_children ? '</b>' : ''}<div class="hint">${w.object_types.length === 9 ? 'все типы объектов' : esc(w.object_types.map((k) => OT[k]).join(', '))}</div>${w.expert_note ? `<div class="note">${esc(w.expert_note)}</div>` : ''}</td>`
        + `<td>${esc(w.stage || '—')}</td><td>${esc(w.profile_name)}</td><td><span class="chip plain s-${w.observability === 'high' ? 'ok' : w.observability === 'medium' ? 'info' : 'no_tasks'}">${OBS[w.observability]}</span></td>`
        + `<td>${w.required.map((g) => `<div>${esc(g.any_of.map(clsName).join(' / '))}${g.window === 'shift' ? ' <span class="hint">за смену</span>' : ''}</div>`).join('') || '—'}</td>`
        + `<td>${w.companions.map((c) => `<div class="hint">${esc(c.lead.map(clsName).join('/'))} → ${esc(c.partner.map(clsName).join('/'))}</div>`).join('')}</td></tr>`).join('')
      + `</tbody></table>${rows.length ? '' : '<div class="empty">Ничего не найдено</div>'}`;
  };
  ['mQ', 'mOT', 'mSt', 'mObs'].forEach((id) => { $('#' + id).oninput = draw; });
  draw();
}

// ------------------------------------------------------------------ быстрая проверка снимка
async function vAnalyze(main) {
  if (!state.workTypes) state.workTypes = await api('/api/reference/work-types?limit=1000');
  const leaf = state.workTypes;
  main.innerHTML = `<div class="grid2"><section class="card"><h2>Проверить снимок против этапа работ</h2>
    <p class="hint">Любой снимок стройплощадки + вид работ из справочника → техника на снимке, проверки методики и предупреждения. Проект не нужен.</p>
    <div class="dropzone" id="aDrop">Перетащите снимок сюда или нажмите, чтобы выбрать<input type="file" id="aFile" accept="image/*" hidden></div>
    <div id="aFileName" class="hint" style="margin:6px 0"></div>
    <label class="sel" style="margin:8px 0"><span>Вид работ (можно несколько через запятую)</span><input id="aWT" list="wtList" value="12.3.1" style="max-width:none"></label>
    <div id="aWTname" class="hint"></div>
    <datalist id="wtList">${leaf.map((w) => `<option value="${esc(w.id)}">${esc(w.name)}</option>`).join('')}</datalist>
    <label class="sel" style="margin:8px 0"><span>Техника по графику (необязательно)</span><input id="aPlan" placeholder="Экскаватор ×1; Самосвал ×3" style="max-width:none"></label>
    <label class="row" style="margin:8px 0"><input type="checkbox" id="aTiles"> общий план с высоты: детекция по фрагментам 3×3</label>
    <button class="btn primary" id="aGo">Проверить</button>
    <p class="hint" style="margin-top:12px">Демо-детектор знает только снимки из data/demo. Для своих фото подключите веса RF-DETR из ML-части: <span class="mono">OKO_DETECTOR=rfdetr</span>.</p></section>
    <section id="aOut"><div class="card empty">Здесь появятся снимок с рамками техники, проверки методики для выбранного вида работ и предупреждения.<br><br>Пример: снимок <span class="mono">data/demo/housing/snapshots/CAM-01_2026-09-24_10-30.jpg</span> и вид работ <span class="mono">12.3.1</span> — пример из ТЗ.</div></section></div>`;
  let file = null;
  const byId = Object.fromEntries(leaf.map((w) => [w.id, w]));
  const showWT = () => {
    $('#aWTname').innerHTML = $('#aWT').value.split(',').map((x) => x.trim()).filter(Boolean)
      .map((id) => byId[id] ? `<div>${esc(id)} — ${esc(byId[id].name)} · ${esc(byId[id].profile_name)} · наблюдаемость ${OBS[byId[id].observability]}</div>` : `<div class="bad-mark">${esc(id)} — нет в справочнике</div>`).join('');
  };
  $('#aWT').oninput = showWT; showWT();
  const drop = $('#aDrop');
  const setFile = (f) => { file = f; $('#aFileName').textContent = f ? `${f.name} · ${(f.size / 1024).toFixed(0)} КБ` : ''; };
  drop.onclick = () => $('#aFile').click();
  $('#aFile').onchange = (e) => setFile(e.target.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); setFile(e.dataTransfer.files[0]); };
  $('#aGo').onclick = async () => {
    if (!file) { toast('Выберите снимок'); return; }
    const fd = new FormData();
    fd.append('file', file); fd.append('work_types', $('#aWT').value); fd.append('planned', $('#aPlan').value);
    fd.append('tiles', $('#aTiles').checked ? '3' : '0');
    $('#aOut').innerHTML = '<p class="muted"><span class="spinner"></span> Распознаём…</p>';
    try {
      const r = await api('/api/analyze', { method: 'POST', body: fd });
      const url = URL.createObjectURL(file);
      const snap = { id: 0, camera: 'снимок', taken_at: r.taken_at, width: r.width, height: r.height, image: url, quality_ok: r.quality.ok };
      let h = `<div class="card"><div class="row" id="qTg" style="margin-bottom:6px" hidden><label id="qTgAIl" hidden><input type="checkbox" id="qTgAI" checked> находки ИИ</label><label id="qTgOVl" hidden><input type="checkbox" id="qTgOV" checked> объекты</label><label id="qTgAl" hidden><input type="checkbox" id="qTgA"> внимание модели</label></div>`
        + `<div id="qFrame">${frameHTML(snap, { boxes: r.boxes })}</div><div class="hint" style="margin-top:6px">детектор ${esc(r.detector)} · ${r.timing_ms.total} мс${r.detector_note ? ` · ${esc(r.detector_note)}` : ''}</div>`;
      h += `<section class="ai" id="qAi" style="margin-top:10px"><h3>Анализ ИИ</h3><div id="qAiBody">${state.ai?.enabled
        ? `<p class="hint">Слои: ${esc(aiLayersHint())}. Модели сверяют друг друга, LLM проверяет рамки детектора и предупреждения, находит пропущенную технику и даёт рекомендации. На CPU — 1–2 минуты.</p><button class="btn small primary" id="qAiRun">Запустить анализ ИИ</button>`
        : '<div class="hint">ИИ-анализ выключен: модель готовности — веса в <span class="mono">weights/readiness/</span>, VLM — <span class="mono">OKO_VLM=auto</span>, YandexGPT — <span class="mono">OKO_LLM_PROVIDER=yandex</span>.</div>'}</div></section>`;
      h += `<h3 style="margin-top:10px">Предупреждения</h3>` + (r.findings.length ? r.findings.map((f) => `<div class="dev ${f.severity}"><div class="title">${esc(f.title)}</div><div class="row">${sevChip(f.severity)}</div><div class="msg">${esc(f.message)}</div><div class="rec">${esc(f.recommendation)}</div></div>`).join('') : '<div class="hint">Отклонений нет.</div>');
      h += `<h3 style="margin-top:10px">Проверки методики</h3>${checkHTML({ ...r.check, zone: 'FRAME', zone_name: 'Кадр' })}</div>`;
      $('#aOut').innerHTML = h;
      bindQuickAI(snap, r, () => {
        const q = new FormData();
        q.append('file', file); q.append('work_types', $('#aWT').value); q.append('planned', $('#aPlan').value);
        q.append('tiles', $('#aTiles').checked ? '3' : '0');
        return q;
      });
    } catch (e) { $('#aOut').innerHTML = `<div class="card">Ошибка: ${esc(e.message)}</div>`; }
  };
}

// ИИ-анализ снимка из «Проверить снимок»: тот же конвейер, результат хранится в памяти сервиса
function bindQuickAI(snap, r, formData, view = { aiBoxes: [], attention: null, objBoxes: [] }) {
  const redraw = () => {
    $('#qFrame').innerHTML = frameHTML(snap, { boxes: r.boxes, aiBoxes: $('#qTgAI').checked ? view.aiBoxes : [],
      attention: $('#qTgA').checked ? view.attention : null, objBoxes: $('#qTgOV').checked ? view.objBoxes || [] : [] });
  };
  $('#qTgAI').onchange = redraw; $('#qTgA').onchange = redraw; $('#qTgOV').onchange = redraw;
  const run = $('#qAiRun');
  if (!run) return;
  run.onclick = async () => {
    run.disabled = true;
    try {
      const j = await api('/api/analyze/ai', { method: 'POST', body: formData() });
      $('#qAi').dataset.jid = j.id;
      pollQuickAI(j.id, view, redraw, formData, snap, r);
    } catch (e) { toast(e.message); run.disabled = false; }
  };
}

async function pollQuickAI(jid, view, redraw, formData, snap, r) {
  const box = $('#qAiBody');
  if (!box || $('#qAi').dataset.jid !== jid) return;
  let j;
  try { j = await api(`/api/analyze/ai/${jid}`); } catch (e) { box.innerHTML = `<div class="hint">Ошибка: ${esc(e.message)}</div>`; return; }
  if (j.status === 'queued' || j.status === 'running') {
    box.innerHTML = `<p><span class="spinner"></span> ${esc(j.step || 'в очереди')}…</p><p class="hint">На CPU модель готовности и VLM работают до минуты-двух.</p>`;
    setTimeout(() => pollQuickAI(jid, view, redraw, formData, snap, r), 2500);
    return;
  }
  const f = j.final;
  let h = '';
  if (!f) h = `<p class="bad-mark">Ошибка анализа</p><p class="hint">${esc(j.error)}</p>`;
  else {
    if (f.assessment) h += `<div class="ai-row"><span class="lbl">модель готовности:</span> ${assessmentHTML(f.assessment)}</div>`;
    h += aiResultHTML(null, j, { quick: true });
  }
  box.innerHTML = h + `<div class="row" style="margin-top:8px"><button class="btn small primary" id="qAiRun">Повторить анализ</button></div>`;
  view.aiBoxes = f?.missed || [];
  view.attention = f?.assessment?.attention || null;
  view.objBoxes = ovBoxes(f?.open_vocab);
  $('#qTgAIl').hidden = !view.aiBoxes.length;
  $('#qTgAl').hidden = !view.attention;
  $('#qTgOVl').hidden = !view.objBoxes.length;
  $('#qTg').hidden = !view.aiBoxes.length && !view.attention && !view.objBoxes.length;
  redraw();
  bindQuickAI(snap, r, formData, view);
}

// ------------------------------------------------------------------ загрузка снимков в проект
function openUpload() {
  if (!state.project) return;
  const cams = state.project.cameras;
  $('#dlgBody').innerHTML = `<button class="btn small close" id="dlgClose">Закрыть ✕</button><h2>Загрузка снимков · ${esc(state.project.name)}</h2>
    <div class="cols2"><div><label class="sel"><span>Камера</span><select id="uCam">${cams.map((c) => `<option value="${esc(c.key)}">${esc(c.key)} · ${esc(c.name)}</option>`).join('')}</select></label>
    <label class="sel" style="margin-top:10px"><span>Зоны на кадре</span><select id="uZone"></select></label>
    <p class="hint" id="uZoneHint"></p>
    <label class="row" style="margin:6px 0"><input type="checkbox" id="uTiles"> детекция по фрагментам 3×3 — для общих планов, где техника мелкая</label>
    <label class="sel" style="margin-top:10px"><span>Время съёмки (важнее EXIF и имени файла)</span><input type="datetime-local" id="uTime"></label>
    <p class="hint">Без поля время берётся из EXIF, затем из имени файла (CAM-01_2026-09-24_10-30.jpg), затем — время загрузки. Выводы строятся по этапам графика на дату съёмки: для демо-проекта укажите дату в пределах его графика (например, 24.09.2026). Повторная загрузка того же снимка не создаёт дубликат.</p></div>
    <div><div class="dropzone" id="uDrop">Перетащите снимки или нажмите для выбора<input type="file" id="uFiles" accept="image/*" multiple hidden></div><div id="uList" class="hint" style="margin-top:8px"></div></div></div>
    <div class="row" style="margin-top:12px"><button class="btn primary" id="uGo">Загрузить и проверить</button></div><div id="uOut" style="margin-top:12px"></div>`;
  const dlg = $('#dlg'); dlg.showModal();
  $('#dlgClose').onclick = () => dlg.close();
  const zoneName = (k) => state.project.zones.find((z) => z.key === k)?.name || k;
  const fillZones = () => {
    const cam = cams.find((c) => c.key === $('#uCam').value);
    const keys = [...new Set([...Object.keys(cam?.zones || {}), ...(cam?.default_zone ? [cam.default_zone] : [])])];
    // одна зона у камеры — по умолчанию весь кадр относится к ней: свой снимок может быть снят с другого ракурса
    $('#uZone').innerHTML = `<option value="">по разметке камеры (полигоны зон)</option>`
      + state.project.zones.map((z) => `<option value="${esc(z.key)}"${keys.length === 1 && keys[0] === z.key ? ' selected' : ''}>весь кадр → ${esc(z.name)}</option>`).join('');
    $('#uZoneHint').textContent = keys.length
      ? `Разметка камеры: ${keys.map(zoneName).join(', ')}. Полигоны подходят только снимкам с того же ракурса; для других фото выберите «весь кадр → зона».`
      : 'У камеры нет разметки зон: выберите зону для всего кадра.';
  };
  $('#uCam').onchange = fillZones; fillZones();
  let files = [];
  const setFiles = (fl) => { files = Array.from(fl); $('#uList').textContent = files.map((f) => f.name).join(', '); };
  const drop = $('#uDrop');
  drop.onclick = () => $('#uFiles').click();
  $('#uFiles').onchange = (e) => setFiles(e.target.files);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); setFiles(e.dataTransfer.files); };
  $('#uGo').onclick = async () => {
    if (!files.length) { toast('Выберите снимки'); return; }
    const fd = new FormData();
    fd.append('camera', $('#uCam').value);
    if ($('#uTime').value) fd.append('taken_at', $('#uTime').value);
    if ($('#uZone').value) fd.append('frame_zone', $('#uZone').value);
    if ($('#uTiles').checked) fd.append('tiles', '3');
    files.forEach((f) => fd.append('files', f));
    $('#uOut').innerHTML = '<span class="spinner"></span> Обработка…';
    try {
      const res = await api(`/api/projects/${state.pid}/snapshots`, { method: 'POST', body: fd });
      $('#uOut').innerHTML = `<table><thead><tr><th>Снимок</th><th>Время</th><th>Техника</th><th>Обработка</th><th>Отклонения</th></tr></thead><tbody>`
        + res.map((r) => `<tr><td><a href="#" data-open-snap="${r.id}">${esc(r.camera)} #${r.id}</a>${r.duplicate ? ' <span class="hint">уже был загружен</span>' : ''}</td><td>${fDay(r.taken_at)} ${fTime(r.taken_at)}</td>`
          + `<td>${Object.entries(r.counts).map(([c, n]) => eqChip(c, n)).join(' ') || (r.quality_ok ? 'техники нет' : esc(r.quality_reason))}${r.detector_note ? `<div class="hint">${esc(r.detector_note)}</div>` : ''}</td><td>${r.processing_ms} мс</td><td>${r.deviation_ids.length}</td></tr>`).join('') + `</tbody></table>`;
      bindCommon($('#uOut'));
      const day = res[res.length - 1]?.taken_at?.slice(0, 10);
      if (day) state.day = day;
      await loadProject();
    } catch (e) { $('#uOut').innerHTML = `Ошибка: ${esc(e.message)}`; }
  };
}

init().catch((e) => { $('#main').innerHTML = `<div class="card">Не удалось загрузить данные: ${esc(e.message)}</div>`; });
