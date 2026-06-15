"use strict";

// ---- state ----
let DATA = null;            // current /api/data payload
let SELECTED = null;        // highlighted flower_id
let TRUES = [];             // sorted distinct true values (the discrete class levels)
let CLASS_GAP = 0;          // spacing between adjacent true values (for ±1-class metric)
let BAND_HALF = 0;          // half the class spacing — band x-extent + fork slot range
let TRUE_OF = new Map();    // class -> true value
let FLOWER_CLASS = new Map();   // flower_id -> original class
let SLOT_KEY = null;        // cache key (split + change version) for the packed slots
let CHANGES_VERSION = 0;    // bumps whenever CHANGES mutates (invalidates slots)
const FORK_X = new Map();   // flower_id -> fixed x slot (a flower is always one column)
const CHANGES = new Map();  // flower_id -> new_class (label corrections, in-memory)
const FORK_INC = new Set(); // included fork numbers (empty => all)
const FORK_EXC = new Set(); // excluded fork numbers
const HIDDEN = new Set();   // classes toggled off via the legend
let DRAGGING = null;        // flower_id being dragged to a new class
let DRAG_TARGET = null;     // class band under the cursor while dragging
const COLLAPSED = new Set();   // round ids collapsed in the detail panel (current flower)
const CYCLE = { ctx: null, item: null, n: 0 };   // legend/views click cycle: toggle → only → all

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];
const classColor = (klass) => PALETTE[Math.max(0, DATA.classes.indexOf(klass)) % PALETTE.length];

const prettyView = (v) => v.split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
const prettyAgg = (n) => n.split("_").map((w) => (w === "mil" ? "MIL" : w[0].toUpperCase() + w.slice(1))).join(" ");

// ---- flower identity / effective (corrected) class ----
const forkOf = (fid) => fid.slice(fid.lastIndexOf("_") + 1);
const origClass = (fid) => FLOWER_CLASS.get(fid);
const effClass = (fid) => CHANGES.get(fid) ?? FLOWER_CLASS.get(fid);
const trueOfClass = (k) => TRUE_OF.get(k);

// ---- small math ----
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN);
function std(a) {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1));
}
const rms = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length);
function pearson(x, y) {
  const n = x.length;
  if (n < 2) return null;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  const d = Math.sqrt(sxx * syy);
  return d ? sxy / d : null;
}

init();

async function init() {
  const { runs } = await fetch("/api/runs").then((r) => r.json());
  const runSel = $("run");
  runSel.innerHTML = "";
  runs.forEach((run) => {
    const o = document.createElement("option");
    o.value = run.name; o.textContent = run.name;
    o.dataset.aggregators = JSON.stringify(run.aggregators);
    runSel.appendChild(o);
  });
  if (!runs.length) { $("status").textContent = "No trained runs found under output/."; return; }
  runSel.onchange = onRunChange;
  $("aggregator").onchange = loadData;
  $("split").onchange = render;
  $("agg-views").onchange = render;
  $("agg-forks").onchange = render;
  $("forks-clear").onclick = () => { FORK_INC.clear(); FORK_EXC.clear(); renderForkTags(); render(); };
  $("fork-input").addEventListener("keydown", onForkInput);
  $("fork-input").addEventListener("focus", showForkDrop);
  $("fork-input").addEventListener("input", showForkDrop);
  $("fork-input").addEventListener("blur", () => setTimeout(hideForkDrop, 120));
  $("tol-num").oninput = render;
  $("outliers-only").onchange = render;
  $("cl-save").onclick = saveChanges;
  document.addEventListener("keydown", onKey);
  new MutationObserver(() => { if (DATA) render(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  onRunChange();
}

function onRunChange() {
  const opt = $("run").selectedOptions[0];
  const aggs = JSON.parse(opt.dataset.aggregators || "[]");
  const aggSel = $("aggregator");
  aggSel.innerHTML = "";
  aggs.forEach((a) => {
    const o = document.createElement("option");
    o.value = a; o.textContent = prettyAgg(a);
    aggSel.appendChild(o);
  });
  if (aggs.includes("mil_mean")) aggSel.value = "mil_mean";
  loadData();
}

async function loadData() {
  const run = $("run").value, agg = $("aggregator").value;
  if (!run || !agg) return;
  $("status").textContent = "Loading…";
  let d;
  try {
    d = await fetch(`/api/data?run=${encodeURIComponent(run)}&aggregator=${encodeURIComponent(agg)}`).then((r) => r.json());
  } catch (e) { d = { error: String(e) }; }
  if (!d || d.error) {
    DATA = null; Plotly.purge($("plot"));
    $("status").textContent = d ? d.error : "Failed to load.";
    $("views").innerHTML = ""; $("stats").innerHTML = ""; $("legend").innerHTML = "";
    return;
  }
  $("status").textContent = "";
  DATA = d; SELECTED = null;
  $("agg-views").checked = true;
  HIDDEN.clear(); COLLAPSED.clear();
  FLOWER_CLASS = new Map();
  DATA.records.forEach((r) => { if (!FLOWER_CLASS.has(r.flower_id)) FLOWER_CLASS.set(r.flower_id, r.klass); });
  buildClassLevels();
  await loadChanges(run);
  SLOT_KEY = null;
  FORK_INC.clear(); FORK_EXC.clear(); renderForkTags(); hideForkDrop();
  buildViewPills(DATA.views);
  buildRoundPills(DATA.rounds);
  $("details").innerHTML = "<p class='hint'>Click a point to inspect a flower. Then relabel it: drag it onto another class band, press 1–6, or use the class buttons.</p>";
  render();
}

// Class levels (true values + spacing) are split-independent.
function buildClassLevels() {
  TRUE_OF = new Map();
  DATA.records.forEach((r) => { if (!TRUE_OF.has(r.klass)) TRUE_OF.set(r.klass, r.true); });
  TRUES = [...TRUE_OF.values()].sort((a, b) => a - b);
  CLASS_GAP = 0;
  for (let i = 1; i < TRUES.length; i++) CLASS_GAP = i === 1 ? TRUES[i] - TRUES[i - 1] : Math.min(CLASS_GAP, TRUES[i] - TRUES[i - 1]);
  BAND_HALF = TRUES.length > 1 ? CLASS_GAP / 2 : 0.1;
}

// Give each flower a fixed x slot in its (effective) class region, ordered by fork
// number and evenly spaced across the flowers PRESENT IN THE CURRENT SPLIT — so columns
// pack evenly and a relabeled flower jumps into its new class's band. Rebuilt when the
// split or the set of corrections changes.
function ensureForkSlots() {
  const split = $("split").value;
  const key = split + "#" + CHANGES_VERSION;
  if (SLOT_KEY === key) return;
  SLOT_KEY = key;
  const byClass = new Map();
  DATA.records.forEach((r) => {
    if (split !== "all" && r.split !== split) return;
    const ec = effClass(r.flower_id);
    (byClass.get(ec) || byClass.set(ec, new Set()).get(ec)).add(r.flower_id);
  });
  FORK_X.clear();
  byClass.forEach((set, klass) => {
    const t = TRUE_OF.get(klass);
    const flowers = [...set].sort((a, b) => (parseInt(forkOf(a)) - parseInt(forkOf(b))) || (a < b ? -1 : 1));
    const k = flowers.length;
    flowers.forEach((fid, i) => {
      FORK_X.set(fid, k === 1 ? t : t - BAND_HALF + ((i + 0.5) / k) * (2 * BAND_HALF));
    });
  });
}
const slotX = (fid) => {
  const x = FORK_X.get(fid);
  return x == null ? (TRUE_OF.get(effClass(fid)) ?? 0) : x;
};

// ---- view toggle pills ----
function buildViewPills(views) {
  const box = $("views");
  box.innerHTML = "";
  views.forEach((v) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "toggle is-on"; b.dataset.view = v;
    b.textContent = prettyView(v);
    b.onclick = () => onViewClick(v, b);
    box.appendChild(b);
  });
}
// click cycle shared by the legend and the view pills: 1st = toggle this one,
// 2nd = show only this one, 3rd = show all (then repeats).
function cyclePhase(ctxName, item) {
  if (CYCLE.ctx === ctxName && CYCLE.item === item) CYCLE.n++;
  else { CYCLE.ctx = ctxName; CYCLE.item = item; CYCLE.n = 1; }
  if (CYCLE.n >= 3) { CYCLE.n = 0; CYCLE.item = null; return "all"; }
  return CYCLE.n === 1 ? "toggle" : "only";
}
function onViewClick(v, btn) {
  const a = cyclePhase("views", v);
  const pills = [...$("views").querySelectorAll(".toggle")];
  if (a === "toggle") btn.classList.toggle("is-on");
  else if (a === "only") pills.forEach((p) => p.classList.toggle("is-on", p.dataset.view === v));
  else pills.forEach((p) => p.classList.add("is-on"));
  render(); refreshDetail();
}
const checkedViews = () => new Set([...$("views").querySelectorAll(".toggle.is-on")].map((b) => b.dataset.view));

// ---- round toggle pills (mirror the view pills) ----
function buildRoundPills(rounds) {
  const box = $("rounds");
  box.innerHTML = "";
  rounds.forEach((r) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "toggle is-on"; b.dataset.round = String(r);
    b.textContent = "R" + r;
    b.onclick = () => onRoundClick(String(r), b);
    box.appendChild(b);
  });
}
function onRoundClick(r, btn) {
  const a = cyclePhase("rounds", r);
  const pills = [...$("rounds").querySelectorAll(".toggle")];
  if (a === "toggle") btn.classList.toggle("is-on");
  else if (a === "only") pills.forEach((p) => p.classList.toggle("is-on", p.dataset.round === r));
  else pills.forEach((p) => p.classList.add("is-on"));
  render(); refreshDetail();
}
const checkedRounds = () => new Set([...$("rounds").querySelectorAll(".toggle.is-on")].map((b) => b.dataset.round));

// ---- fork autocomplete dropdown ----
function showForkDrop() {
  if (!DATA) return;
  const q = $("fork-input").value.trim();
  const opts = DATA.forks.filter((f) => !FORK_INC.has(f) && String(f).startsWith(q));
  const drop = $("fork-drop");
  if (!opts.length) { drop.style.display = "none"; return; }
  drop.innerHTML = opts.slice(0, 300).map((f) => `<div class="opt" data-f="${f}">fork ${f}</div>`).join("");
  drop.style.display = "block";
  drop.querySelectorAll(".opt").forEach((el) => {
    el.onmousedown = (e) => {
      e.preventDefault();
      FORK_INC.add(parseInt(el.dataset.f));
      $("fork-input").value = "";
      hideForkDrop(); renderForkTags(); render();
    };
  });
}
function hideForkDrop() { $("fork-drop").style.display = "none"; }

// ---- fork tag filter ----
function onForkInput(e) {
  if (e.key !== "Enter") return;
  const raw = e.target.value.trim();
  e.target.value = "";
  if (!raw) return;
  if (raw.toLowerCase() === "all") { FORK_INC.clear(); FORK_EXC.clear(); }
  else {
    raw.split(/[\s,]+/).forEach((tok) => {
      if (tok.startsWith("-")) { const n = parseInt(tok.slice(1)); if (!isNaN(n) && DATA.forks.includes(n)) FORK_EXC.add(n); }
      else { const n = parseInt(tok); if (!isNaN(n) && DATA.forks.includes(n)) FORK_INC.add(n); }
    });
  }
  renderForkTags(); render();
}
function renderForkTags() {
  const box = $("fork-tags");
  box.innerHTML = "";
  const chip = (n, excl) => {
    const t = document.createElement("span");
    t.className = "tag" + (excl ? " tag-excl" : "");
    t.textContent = (excl ? "−" : "") + n;
    const x = document.createElement("button");
    x.className = "tagx"; x.textContent = "×";
    x.onclick = () => { (excl ? FORK_EXC : FORK_INC).delete(n); renderForkTags(); render(); };
    t.appendChild(x); box.appendChild(t);
  };
  [...FORK_INC].sort((a, b) => a - b).forEach((n) => chip(n, false));
  [...FORK_EXC].sort((a, b) => a - b).forEach((n) => chip(n, true));
}
function forkAllowed(f) {
  if (FORK_EXC.has(f)) return false;
  return FORK_INC.size ? FORK_INC.has(f) : true;
}

// ---- tolerance ----
const tol = () => {
  let v = parseFloat($("tol-num").value);
  if (isNaN(v)) v = 0.1;
  return Math.max(0, Math.min(0.5, v));
};

// ---- aggregation ----
function currentPoints() {
  const split = $("split").value;
  const views = checkedViews();
  const rounds = checkedRounds();
  const meanViews = $("agg-views").checked, meanForks = $("agg-forks").checked;
  const t = tol();

  const recs = DATA.records.filter(
    (r) => (split === "all" || r.split === split) && views.has(r.view_type) && rounds.has(r.round) && forkAllowed(parseInt(r.fork))
  );

  const groups = new Map();
  for (const r of recs) {
    const parts = [r.flower_id];
    if (!meanForks) parts.push("r" + r.round);
    if (!meanViews) parts.push(r.view_type);
    const key = parts.join("|");
    (groups.get(key) || groups.set(key, []).get(key)).push(r);
  }
  return [...groups.values()].map((g) => {
    const y = mean(g.map((r) => r.pred));
    const r0 = g[0];
    const ec = effClass(r0.flower_id);
    const x = trueOfClass(ec);
    let label = `fork ${r0.fork}`;
    if (!meanForks) label += ` · round ${r0.round}`;
    label += meanViews ? ` · ${g.length} views (mean)` : ` · ${prettyView(r0.view_type)}`;
    return { x, xplot: slotX(r0.flower_id), y, flower: r0.flower_id, klass: ec, label, inBand: Math.abs(y - x) <= t, recs: g };
  });
}

// ---- global stats (reflect exactly what is plotted) ----
function computeStats(pts) {
  const recs = pts.flatMap((p) => p.recs);
  const byFlower = new Map();
  recs.forEach((r) => (byFlower.get(r.flower_id) || byFlower.set(r.flower_id, []).get(r.flower_id)).push(r));
  const viewStds = [], milStds = [];
  byFlower.forEach((fr) => {
    viewStds.push(std(fr.map((r) => r.pred)));
    const byRound = new Map();
    fr.forEach((r) => (byRound.get(r.round) || byRound.set(r.round, []).get(r.round)).push(r));
    const mils = [...byRound.values()].map((vs) => mean(vs.map((r) => r.pred)));
    if (mils.length > 1) milStds.push(std(mils));
  });

  const t = tol();
  const errs = pts.map((p) => p.y - p.x);
  const within = (thr) => (pts.length ? pts.filter((p) => Math.abs(p.y - p.x) <= thr).length / pts.length * 100 : 0);
  return {
    views: recs.length,
    flowers: new Set(recs.map((r) => r.flower_id + "|" + r.round)).size,
    uniqueFlowers: byFlower.size,
    rmse: pts.length ? rms(errs) : null,
    mae: pts.length ? mean(errs.map(Math.abs)) : null,
    bias: pts.length ? mean(errs) : null,
    inBandPct: within(t),
    within1: CLASS_GAP ? within(CLASS_GAP) : null,
    corr: pearson(pts.map((p) => p.x), pts.map((p) => p.y)),
    viewStd: viewStds.length ? mean(viewStds) : null,
    milStd: milStds.length ? mean(milStds) : null,
  };
}

function renderStats(s) {
  const f = (x, d = 3) => (x == null || isNaN(x) ? "—" : x.toFixed(d));
  const cell = (val, label, title) => `<div class="stat" title="${title}"><span class="sv">${val}</span><span class="sl">${label}</span></div>`;
  const grp = (name, cells) => `<div class="stat-grp"><div class="stat-gh">${name}</div><div class="stat-cells">${cells}</div></div>`;
  $("stats").innerHTML =
    grp("Counts",
      cell(s.views, "views", "view images shown") +
      cell(s.flowers, "flowers", "captured flowers — one fork at one round") +
      cell(s.uniqueFlowers, "unique flowers", "distinct forks (rounds collapsed to one)")) +
    grp("Accuracy",
      cell(f(s.rmse), "RMSE", "root mean squared error — typical error size, penalises big misses") +
      cell(f(s.mae), "MAE", "mean absolute error — average error size") +
      cell(s.bias == null ? "—" : (s.bias >= 0 ? "+" : "") + f(s.bias), "bias", "mean(pred − true): + over-predicts, − under-predicts") +
      cell(f(s.inBandPct, 1) + "%", "in band", "share within ±tolerance of true") +
      cell(s.within1 == null ? "—" : f(s.within1, 1) + "%", "within 1 class", "share within one ripeness class of true") +
      cell(f(s.corr), "correlation", "how well predicted tracks true, −1…1 (1 = perfect)")) +
    grp("Consistency",
      cell(f(s.viewStd), "spread · views", "avg per-flower std across views — lower = views agree") +
      cell(f(s.milStd), "spread · rounds", "avg per-flower std across rounds — lower = repeat captures agree"));
}

// ---- class legend (boxed, click to toggle) ----
function renderLegend() {
  const box = $("legend");
  box.innerHTML = "";
  DATA.classes.forEach((c, i) => {
    const color = PALETTE[i % PALETTE.length];
    const b = document.createElement("div");
    b.className = "legend-item" + (HIDDEN.has(c) ? " is-off" : "");
    b.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>class ${c}`;
    b.onclick = () => onLegendClick(c);
    box.appendChild(b);
  });
}
function onLegendClick(c) {
  const a = cyclePhase("legend", c);
  if (a === "toggle") { HIDDEN.has(c) ? HIDDEN.delete(c) : HIDDEN.add(c); }
  else if (a === "only") { HIDDEN.clear(); DATA.classes.forEach((k) => { if (k !== c) HIDDEN.add(k); }); }
  else HIDDEN.clear();
  render();
}

// ---- plot ----
function render() {
  if (!DATA) return;
  ensureForkSlots();
  const all = currentPoints();
  let pts = all.filter((p) => !HIDDEN.has(p.klass));
  if ($("outliers-only").checked) pts = pts.filter((p) => !p.inBand);
  renderStats(computeStats(pts));
  renderLegend();

  const traces = [];
  DATA.classes.forEach((c, i) => {
    const cp = pts.filter((p) => p.klass === c);
    if (!cp.length) return;
    traces.push({
      type: "scattergl", mode: "markers", name: `class ${c}`,
      x: cp.map((p) => p.xplot), y: cp.map((p) => p.y),
      customdata: cp.map((p) => [p.flower, p.x]), text: cp.map((p) => p.label),
      hovertemplate: "%{text}<br>true ripeness %{customdata[1]:.3f} · pred %{y:.3f}<extra></extra>",
      marker: { size: 8, color: PALETTE[i % PALETTE.length], opacity: 0.6 },
    });
  });
  if (SELECTED) {
    const hp = pts.filter((p) => p.flower === SELECTED);
    traces.push({
      type: "scattergl", mode: "markers", name: "selected", showlegend: false,
      x: hp.map((p) => p.xplot), y: hp.map((p) => p.y),
      customdata: hp.map((p) => [p.flower, p.x]), text: hp.map((p) => p.label),
      hovertemplate: "%{text}<br>true ripeness %{customdata[1]:.3f} · pred %{y:.3f}<extra></extra>",
      marker: { size: 11, color: css("--text-strong"), line: { color: css("--panel-strong"), width: 1.4 } },
    });
  }

  const layout = {
    title: { text: `${DATA.task} · ${prettyAgg(DATA.aggregator)} — predicted vs true (${$("split").value})`, font: { color: css("--text") } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: css("--muted"), family: css("--font-family-base") },
    xaxis: { title: { text: "True Ripeness", standoff: 8 }, automargin: true, range: [-0.15, 1.15], zeroline: false, gridcolor: css("--line") },
    yaxis: { title: { text: "Predicted", standoff: 8 }, automargin: true, range: [-0.05, 1.05], zeroline: false, gridcolor: css("--line") },
    margin: { t: 40, r: 12, b: 40, l: 40 }, showlegend: false, dragmode: false,
    hovermode: "closest", shapes: stairBands(tol()),
  };
  const gd = $("plot");
  Plotly.react(gd, traces, layout, { displayModeBar: false, responsive: true });
  if (!gd._bound) {
    gd._bound = true;
    gd.on("plotly_click", onPointClick);
    gd.addEventListener("mousedown", onPlotMouseDown);
    window.addEventListener("mousemove", onPlotMouseMove);
    window.addEventListener("mouseup", onPlotMouseUp);
  }
}

// staircase tolerance band: one horizontal ±tol step per class, plus a dashed
// centre line at the true value. `highlight` (a class) brightens its band while
// a flower is being dragged onto it.
function stairBands(t, highlight) {
  const line = css("--muted"), accent = css("--accent");
  const shapes = [];
  DATA.classes.forEach((c) => {
    const v = TRUE_OF.get(c);
    if (v == null) return;
    const x0 = v - BAND_HALF, x1 = v + BAND_HALF, hot = c === highlight;
    shapes.push({
      type: "rect", xref: "x", yref: "y", layer: "below", x0, x1, y0: v - t, y1: v + t,
      fillcolor: hot ? "rgba(42,92,151,0.25)" : "rgba(120,120,120,0.18)",
      line: hot ? { color: accent, width: 1.5 } : { width: 0 },
    });
    shapes.push({ type: "line", xref: "x", yref: "y", layer: "below", x0, x1, y0: v, y1: v, line: { color: line, width: 1, dash: "dash" } });
  });
  return shapes;
}

// ---- relabel ----
function reassign(fid, newKlass) {
  if (!fid) return;
  if (newKlass === origClass(fid)) CHANGES.delete(fid);
  else CHANGES.set(fid, newKlass);
  CHANGES_VERSION++;
  render(); renderChangeLog(); refreshDetail();
}
function removeChange(fid) {
  if (!CHANGES.delete(fid)) return;
  CHANGES_VERSION++;
  render(); renderChangeLog(); refreshDetail();
}
function onKey(e) {
  if (!SELECTED || !DATA) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  const n = parseInt(e.key);
  if (n >= 1 && n <= DATA.classes.length) { reassign(SELECTED, DATA.classes[n - 1]); e.preventDefault(); }
  else if (e.key === "u" || e.key === "Backspace") { removeChange(SELECTED); e.preventDefault(); }
}

function renderChangeLog() {
  const wrap = $("changelog"), list = $("cl-list");
  if (!CHANGES.size) { wrap.hidden = true; list.innerHTML = ""; $("cl-summary").textContent = ""; return; }
  wrap.hidden = false;
  const counts = {};
  CHANGES.forEach((nc, fid) => { const k = `${origClass(fid)}→${nc}`; counts[k] = (counts[k] || 0) + 1; });
  $("cl-summary").textContent = `${CHANGES.size} flower${CHANGES.size > 1 ? "s" : ""} changed · ` +
    Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(", ");
  list.innerHTML = "";
  [...CHANGES.entries()].sort((a, b) => parseInt(forkOf(a[0])) - parseInt(forkOf(b[0]))).forEach(([fid, nc]) => {
    const row = document.createElement("div"); row.className = "cl-row";
    row.innerHTML = `<span class="cl-swatch" style="background:${classColor(nc)}"></span>fork ${forkOf(fid)} · class ${origClass(fid)} → <b>${nc}</b>`;
    const undo = document.createElement("button"); undo.className = "btn btn--sm"; undo.textContent = "undo";
    undo.onclick = (e) => { e.stopPropagation(); removeChange(fid); };
    row.appendChild(undo);
    row.onclick = () => selectFlower(fid);
    list.appendChild(row);
  });
}

async function loadChanges(run) {
  CHANGES.clear();
  let map = {};
  try { map = (await fetch(`/api/changes?run=${encodeURIComponent(run)}`).then((r) => r.json())).changes || {}; }
  catch (e) { /* none */ }
  Object.entries(map).forEach(([fid, nc]) => { if (FLOWER_CLASS.has(fid)) CHANGES.set(fid, String(nc)); });
  CHANGES_VERSION++;
  renderChangeLog();
}

async function saveChanges() {
  const run = $("run").value;
  const changes = [...CHANGES.entries()].map(([flower_id, new_class]) => ({ flower_id, new_class }));
  const btn = $("cl-save"); btn.disabled = true; btn.textContent = "Saving…";
  try {
    const res = await fetch("/api/changes", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run, changes }),
    }).then((r) => r.json());
    $("cl-summary").textContent = `Saved ${res.flowers} flower${res.flowers === 1 ? "" : "s"} (${res.rows} rows) → ripeness_changes.csv`;
  } catch (e) {
    $("cl-summary").textContent = "Save failed: " + e;
  } finally { btn.disabled = false; btn.textContent = "Save CSV"; }
}

// ---- click → inspect / select a flower ----
function selectFlower(fid) {
  if (fid !== SELECTED) COLLAPSED.clear();   // fresh collapse state per flower
  SELECTED = fid;
  render();
  showFlower(fid);
}
function onPointClick(ev) {
  const p = ev.points[0];
  if (!p || p.customdata == null) return;
  selectFlower(String(p.customdata[0]));
}
function refreshDetail() { if (SELECTED) showFlower(SELECTED); }

// ---- drag the selected flower onto another class band ----
function selectedPlotPoints() {
  return SELECTED ? currentPoints().filter((p) => p.flower === SELECTED && !HIDDEN.has(p.klass)) : [];
}
function onPlotMouseDown(e) {
  if (!SELECTED || !DATA) return;
  const gd = $("plot"), xa = gd._fullLayout && gd._fullLayout.xaxis, ya = gd._fullLayout && gd._fullLayout.yaxis;
  if (!xa || !ya) return;
  const rect = gd.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const near = selectedPlotPoints().some((p) => {
    try { return Math.hypot((xa.d2p(p.xplot) + xa._offset) - mx, (ya.d2p(p.y) + ya._offset) - my) <= 14; }
    catch (_) { return false; }
  });
  if (!near) return;
  DRAGGING = SELECTED; DRAG_TARGET = null;
  gd.style.cursor = "grabbing";
  e.preventDefault();
}
function onPlotMouseMove(e) {
  if (!DRAGGING) return;
  const gd = $("plot"), xa = gd._fullLayout && gd._fullLayout.xaxis;
  if (!xa) return;
  let dataX;
  try { dataX = xa.p2d(e.clientX - gd.getBoundingClientRect().left - xa._offset); }
  catch (_) { return; }
  let target = null, best = Infinity;
  DATA.classes.forEach((c) => {
    const tv = TRUE_OF.get(c);
    if (tv == null) return;
    const d = Math.abs(tv - dataX);
    if (d <= BAND_HALF + 1e-9 && d < best) { best = d; target = c; }
  });
  if (target !== DRAG_TARGET) { DRAG_TARGET = target; Plotly.relayout(gd, { shapes: stairBands(tol(), target) }); }
}
function onPlotMouseUp() {
  if (!DRAGGING) return;
  const fid = DRAGGING, target = DRAG_TARGET;
  DRAGGING = null; DRAG_TARGET = null;
  $("plot").style.cursor = "";
  if (target && target !== effClass(fid)) reassign(fid, target);   // render() resets the shapes
  else Plotly.relayout($("plot"), { shapes: stairBands(tol()) });
}

function showFlower(flower) {
  const run = $("run").value;
  const sel = checkedViews();
  const recs = DATA.records.filter((r) => r.flower_id === flower);
  if (!recs.length) return;
  const fork = recs[0].fork;
  const oc = origClass(flower), ec = effClass(flower), changed = ec !== oc;
  const trueVal = trueOfClass(ec);            // effective (asserted) true value
  const t = tol();

  const byRound = new Map();
  recs.forEach((r) => (byRound.get(r.round) || byRound.set(r.round, []).get(r.round)).push(r));
  const roundMils = [...byRound.values()]
    .map((vs) => mean(vs.filter((r) => sel.has(r.view_type)).map((r) => r.pred)))
    .filter((m) => !isNaN(m));
  const flowerMean = mean(roundMils), flowerStd = std(roundMils);

  const reassignBtns = DATA.classes.map((c) => {
    const active = c === ec;
    return `<button class="reassign-btn${active ? " is-active" : ""}"${active ? ` style="border-bottom-color:${classColor(c)}"` : ""} data-c="${c}"><span class="rb-sw" style="background:${classColor(c)}"></span>${c}</button>`;
  }).join("");

  let html = `<div class="dhead">
      <span class="swatch" style="background:${classColor(ec)}"></span>
      <h2 class="fork-title" title="collapse / expand all rounds">Fork ${fork}</h2>
      <span class="dmeta">Class <b>${oc}</b>${changed ? ` → <b>${ec}</b>` : ""}</span>
      <span class="dmeta">True Ripeness <b>${trueVal.toFixed(3)}</b></span>
      <span class="dmeta">Flower Mean <b>${isNaN(flowerMean) ? "—" : flowerMean.toFixed(3)}</b> ± ${flowerStd.toFixed(3)}</span>
      <div class="reassign"><span class="reassign-label">Set Class</span>${reassignBtns}</div>
    </div>`;

  [...byRound.keys()].sort((a, b) => a - b).forEach((rnd) => {
    const views = byRound.get(rnd).sort((a, b) => a.view_id - b.view_id);
    const selPreds = views.filter((r) => sel.has(r.view_type)).map((r) => r.pred);
    const milMean = mean(selPreds), roundStd = std(selPreds);
    const milCls = !isNaN(milMean) && Math.abs(milMean - trueVal) <= t ? "in" : "out";
    html += `<div class="round${COLLAPSED.has(String(rnd)) ? " is-collapsed" : ""}"><div class="round-head" data-round="${rnd}">Round ${rnd}</div><div class="thumbs">`;
    views.forEach((r) => {
      const src = `/api/image?run=${encodeURIComponent(run)}&file=${encodeURIComponent(r.file_name)}`;
      const on = sel.has(r.view_type);
      const off = Math.abs(r.pred - trueVal) <= t ? "in" : "out";
      html += `<figure class="thumb${on ? "" : " faded"}">
          <img loading="lazy" src="${src}" alt="${r.view_type}" />
          <figcaption>${prettyView(r.view_type)}<br><span class="${off}">${r.pred.toFixed(3)}</span></figcaption>
        </figure>`;
    });
    html += `<div class="round-summary">
        <div class="rs-row"><span class="rs-label">Mean Views</span><span class="rs-val ${milCls}">${isNaN(milMean) ? "—" : milMean.toFixed(3)}</span></div>
        <div class="rs-row"><span class="rs-label">Std Dev</span><span class="rs-val">± ${roundStd.toFixed(3)}</span></div>
      </div>`;
    html += "</div></div>";
  });
  $("details").innerHTML = html;
  $("details").querySelectorAll(".reassign-btn").forEach((b) => { b.onclick = () => reassign(flower, b.dataset.c); });
  $("details").querySelectorAll(".round-head").forEach((h) => {
    h.onclick = () => {
      const r = h.dataset.round;
      COLLAPSED.has(r) ? COLLAPSED.delete(r) : COLLAPSED.add(r);
      showFlower(flower);
    };
  });
  const ft = $("details").querySelector(".fork-title");
  if (ft) ft.onclick = () => {                            // collapse all rounds, or expand all if already collapsed
    const rounds = [...byRound.keys()].map(String);
    if (rounds.every((r) => COLLAPSED.has(r))) COLLAPSED.clear();
    else rounds.forEach((r) => COLLAPSED.add(r));
    showFlower(flower);
  };
}
