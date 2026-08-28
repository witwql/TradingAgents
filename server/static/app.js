/* TradingAgents Dashboard frontend — vanilla JS, no build step. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ANALYSTS = [
  ["market", "市场分析", "K线 / 技术指标"],
  ["social", "情绪分析", "新闻+社媒情绪"],
  ["news", "新闻分析", "宏观与公司事件"],
  ["fundamentals", "基本面分析", "估值与财报"],
  ["macro", "全球宏观", "金/油/美债/美股四因子"],
];
const STATUS_TEXT = {
  pending: "排队中", running: "运行中", completed: "已完成",
  failed: "失败", cancelled: "已取消",
};

const STAGE_TEXT = {
  market: "市场分析", sentiment: "情绪分析", news: "新闻分析", fundamentals: "基本面分析",
  macro: "全球宏观分析",
  bull_researcher: "多方研究", bear_researcher: "空方研究", research_manager: "研究裁决",
  trader: "交易决策", risk_debate: "风险辩论", portfolio_manager: "终审",
};
const state = { filterStatus: "", drawerTaskId: null, drawerES: null, cache: {}, feedCount: 0, feedScroll: true, lastScreenRun: null };

/* ---------------- router ---------------- */
const VIEWS = ["dashboard", "new", "tasks", "picks", "reports", "favorites", "settings"];
function route() {
  const name = (location.hash || "#dashboard").slice(1) || "dashboard";
  const target = VIEWS.includes(name) ? name : "dashboard";
  $$(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === target));
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === target));
  loadView(target);
}
function go(view) { location.hash = "#" + view; }

function buildNav() {
  $("#nav").innerHTML = VIEWS.map((v) =>
    `<a href="#${v}" data-view="${v}">${{
      dashboard: "工作台", new: "新建分析", tasks: "任务队列",
      reports: "报告中心", favorites: "自选股", settings: "设置",
    }[v]}</a>`).join("");
}

async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${(await res.json().catch(() => ({}))).detail || ""}`);
  return res.json();
}

/* ---------------- dashboard ---------------- */
async function loadDashboard() {
  const { tasks } = await api("/tasks?limit=200");
  renderStats(tasks);
  renderRecent(tasks.slice(0, 10));
}
function renderStats(tasks) {
  const today = new Date().toDateString();
  const todayTasks = tasks.filter((t) => new Date(t.created_at * 1000).toDateString() === today);
  const running = tasks.filter((t) => t.status === "running").length;
  const done = tasks.filter((t) => t.status === "completed").length;
  const finished = tasks.filter((t) => t.status !== "pending" && t.status !== "running").length;
  const failedRate = finished ? tasks.filter((t) => t.status === "failed").length / finished : 0;
  $("#stats").innerHTML = [
    ["今日分析", todayTasks.length],
    ["进行中", running],
    ["累计完成", done],
    ["失败率", (failedRate * 100).toFixed(0) + "%"],
  ].map(([l, n]) => `<div class="stat"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");
}
function renderRecent(tasks) {
  $("#recent-tasks").innerHTML = `
    <tr><th>标的</th><th>日期</th><th>状态</th><th>评级</th><th>创建时间</th></tr>` +
    (tasks.map(taskRow).join("") || `<tr><td colspan="5" class="muted">暂无任务，去「新建分析」提交第一份研究</td></tr>`);
}

/* ---------------- new analysis ---------------- */
function buildAnalystPicker() {
  $("#f-analysts").innerHTML = ANALYSTS.map(([k, label, hint], i) =>
    `<label class="analyst-pick"><input type="checkbox" value="${k}" checked/>${label} <span class="muted">${hint}</span></label>`).join("");
}
async function submitAnalysis(ev) {
  ev.preventDefault();
  const tickers = $("#f-tickers").value.split(/[\s,，;；]+/).filter(Boolean);
  if (!tickers.length) return;
  const payload = {
    tickers,
    trade_date: $("#f-date").value,
    analysts: $$('#f-analysts input:checked').map((i) => i.value),
    debate_rounds: +$("#f-debate").value,
    risk_rounds: +$("#f-risk").value,
    output_language: $("#f-lang").value,
  };
  const btn = $("#new-form button[type=submit]");
  btn.disabled = true;
  btn.textContent = "提交中…";
  try {
    const { task_ids } = await api("/tasks", { method: "POST", body: payload });
    $("#new-msg").textContent = `已提交 ${task_ids.length} 个任务，正在打开实时面板…`;
    $("#f-tickers").value = "";
    loadView("tasks");
    openDrawer(task_ids[0]);
  } catch (e) { $("#new-msg").textContent = `提交失败：${e.message}`; }
  finally { btn.disabled = false; btn.textContent = "提交分析任务"; }
}

/* ---------------- tasks ---------------- */
const STAGE_ORDER = ["market","sentiment","news","fundamentals","macro","bull_researcher",
  "bear_researcher","research_manager","trader","risk_debate","portfolio_manager"];
function stageProgress(stage) {
  const i = STAGE_ORDER.indexOf(stage);
  return i < 0 ? 0 : Math.round(((i + 1) / STAGE_ORDER.length) * 100);
}
function elapsedText(startedAt) {
  if (!startedAt) return "";
  const secs = Math.max(0, (Date.now() - startedAt * 1000) / 1000);
  return `${Math.floor(secs / 60)}分${String(Math.floor(secs % 60)).padStart(2, "0")}秒`;
}
function taskRow(t) {
  const ratingClass = /buy/i.test(t.rating) ? "rating-buy" : /sell|underweight/i.test(t.rating)
    ? "rating-sell" : /hold|overweight/i.test(t.rating) ? "rating-hold" : "";
  const running = t.status === "running";
  const pct = running ? stageProgress(t.current_stage) : (t.status === "completed" ? 100 : 0);
  return `<tr onclick="openDrawer('${t.id}')">
    <td><strong>${esc(t.ticker)}</strong></td>
    <td>${esc(t.trade_date)}</td>
    <td><span class="badge ${t.status}">${STATUS_TEXT[t.status] || t.status}</span>
        ${running ? `<span class="muted" style="font-size:12px"> · ${STAGE_TEXT[t.current_stage] || "启动中"}…</span>
        <div class="mini-bar"><i style="width:${pct}%"></i></div>
        <span class="muted" style="font-size:11px">已运行 ${elapsedText(t.started_at)} · 约完成 ${pct}%</span>` : ""}</td>
    <td class="${ratingClass}">${t.status === "completed" ? esc(ratingText(t.rating)) : "—"}</td>
    <td class="muted">${fmtTime(t.created_at)}</td></tr>`;
}
function ratingText(raw) {
  if (/^buy/i.test(raw)) return "买入 Buy";
  if (/overweight/i.test(raw)) return "增持 Overweight";
  if (/hold/i.test(raw)) return "持有 Hold";
  if (/underweight/i.test(raw)) return "减持 Underweight";
  if (/sell/i.test(raw)) return "卖出 Sell";
  return raw.replace(/^Final Decision:\s*/i, "").split(/[.\n]/)[0].slice(0, 24);
}
async function loadTasks() {
  const { tasks } = await api(`/tasks?limit=200`);
  renderTaskTable(tasks);
}
function renderTaskTable(tasks) {
  // 分段筛选器计数
  const counts = { "": tasks.length };
  for (const st of ["running", "pending", "completed", "failed"]) {
    counts[st] = tasks.filter((t) => t.status === st).length;
  }
  $$("#task-filters .seg-btn").forEach((b) => {
    const c = counts[b.dataset.status] ?? 0;
    b.querySelector("i").textContent = c;
    b.classList.toggle("active", b.dataset.status === state.filterStatus);
  });

  const filtered = state.filterStatus ? tasks.filter((t) => t.status === state.filterStatus) : tasks;
  const wrap = $("#task-cards");
  if (!filtered.length) {
    wrap.innerHTML = `<div class="task-empty">${
      state.filterStatus ? "没有符合筛选条件的任务" :
      "还没有任务 —— 去「新建分析」提交第一份研究，或把常看的标的加入自选股快速分析"
    }</div>`;
    return;
  }
  wrap.innerHTML = filtered.map(taskCard).join("");
}

function taskCard(t) {
  const running = t.status === "running";
  const pct = running ? stageProgress(t.current_stage) : (t.status === "completed" ? 100 : 0);
  const stage = STAGE_TEXT[t.current_stage] || "启动中";
  const pill = ratingPill(t);
  const summary = (t.summary || "").split("\n").find((l) => l.trim()) || "";
  const teamDesc = `${(t.analysts || []).length} 分析师 · 辩论${t.debate_rounds}轮 · 风险${t.risk_rounds}轮`;
  return `<div class="task-card st-${t.status}" onclick="openDrawer('${t.id}')">
    <div class="tc-main">
      <div class="tc-title">
        <span class="tc-ticker">${esc(t.ticker)}</span>
        <span class="badge ${t.status}">${STATUS_TEXT[t.status] || t.status}</span>
        ${running ? `<span class="muted" style="font-size:12px">${stage}…</span>` : ""}
      </div>
      <div class="tc-meta">${esc(t.trade_date)} · ${teamDesc} · ${fmtTime(t.created_at)}${
        running ? ` · 已运行 ${elapsedText(t.started_at)}` :
        t.finished_at && t.started_at ? ` · 耗时 ${Math.round(t.finished_at - t.started_at)}秒` : ""
      }</div>
      ${running ? `<div class="tc-progress">
        <div class="row"><span>${stage}</span><span>${pct}%</span></div>
        <div class="mini-bar" style="max-width:none"><i style="width:${pct}%"></i></div>
      </div>` : ""}
      ${t.status === "completed" && summary ? `<div class="tc-summary">${esc(summary)}</div>` : ""}
    </div>
    <div class="tc-side">
      ${pill}
      <button class="tc-del" title="${t.status === "running" ? "运行中不可删除" : "删除"}"
        onclick="event.stopPropagation();removeTask('${t.id}','${t.status}')">×</button>
    </div>
  </div>`;
}

function ratingPill(t) {
  if (t.status !== "completed") return "";
  const raw = (t.rating || "").toLowerCase();
  let cls = "hold", label = ratingText(t.rating);
  if (/^buy/.test(raw)) cls = "buy";
  else if (/overweight/.test(raw)) cls = "buy";
  else if (/underweight|sell/.test(raw)) cls = "sell";
  return `<span class="rating-pill ${cls}">${esc(label)}</span>`;
}
async function removeTask(id, status) {
  if (status === "running") { alert("运行中的任务无法删除"); return; }
  await api(`/tasks/${id}`, { method: "DELETE" });
  loadView("tasks");
}

/* ---------------- drawer (workflow visualization) ---------------- */
async function openDrawer(id) {
  closeDrawer(false);
  state.drawerTaskId = id;
  const detail = await api(`/tasks/${id}`).catch(() => null);
  if (!detail) return;
  $("#drawer-title").textContent = `${detail.ticker} · ${detail.trade_date}`;
  renderDrawer(detail);
  $("#drawer-mask").classList.remove("hidden");
  $("#drawer").classList.add("open");

  if (["pending", "running"].includes(detail.status)) listenEvents(id);
}
function renderDrawer(detail) {
  const stages = detail.stages || [];
  const body = $("#drawer-body");
  body.innerHTML = `
    <div id="drawer-hint" class="muted hidden" style="font-size:12px;margin-bottom:8px"></div>
    <div class="kv">
      <span class="k">状态</span><span><span class="badge ${detail.status}">${STATUS_TEXT[detail.status]}</span></span>
      <span class="k">团队</span><span>${detail.analysts.map(esc).join(" / ")}</span>
      <span class="k">辩论深度</span><span>投资辩论 ${detail.debate_rounds} 轮 · 风险讨论 ${detail.risk_rounds} 轮</span>
      ${detail.error ? `<span class="k">错误</span><span class="muted" style="color:var(--red);word-break:break-all">${esc(detail.error)}</span>` : ""}
    </div>
    <h3 style="margin:10px 0 4px">智能体工作流</h3>
    <div id="wf-stages">${stages.map(stageRow).join("")}</div>
    <div class="feed-head">
      <h3>实时输出流 <span class="muted" id="feed-count"></span></h3>
      <span class="feed-tools">
        <select id="feed-filter">
          <option value="">全部</option>
          <option value="llm">思考/结果</option>
          <option value="tool">工具调用</option>
          <option value="err">错误</option>
        </select>
        <button class="btn small" id="feed-scroll-btn" onclick="toggleFeedScroll()">⬇ 自动滚动：开</button>
      </span>
    </div>
    <div id="agent-feed"></div>
    <div id="drawer-result"></div>`;
  state.feedCount = 0;
  $("#feed-filter").value = "";
  $("#feed-filter").onchange = () => applyFeedFilter();
  fillResult(detail);
}
function stageRow(s) {
  return `<div class="stage-row ${s.status}" data-stage="${esc(s.name)}">
    <span class="dot"></span>
    <span class="muted" style="font-size:11px;width:30px">#${s.seq + 1}</span>
    <span>${STAGE_TEXT[s.name] || esc(s.name)}</span>
    <span class="muted" style="margin-left:auto;font-size:12px">${s.finished_at && s.started_at ? ((s.finished_at - s.started_at)).toFixed(1) + "s" : s.status === "pending" ? "" : "…"}</span>
  </div>`;
}
function fillResult(detail) {
  const box = $("#drawer-result");
  if (!box) return;
  if (detail.status === "completed") {
    const cls = /buy/i.test(detail.rating) ? "rating-buy" : /sell|underweight/i.test(detail.rating) ? "rating-sell" : "rating-hold";
    box.innerHTML = `<h3 style="margin-top:14px">最终评级：<span class="${cls}">${esc(ratingText(detail.rating))}</span></h3>
      <pre class="hint" style="white-space:pre-wrap">${esc(detail.summary)}</pre>
      <button class="btn primary" style="margin-top:12px" onclick="go('#reports');setTimeout(()=>openReportPicker('${detail.id}'),150)">查看完整报告 →</button>`;
  }
}
function listenEvents(id) {
  if (state.drawerES) state.drawerES.close();
  const es = new EventSource(`/api/tasks/${id}/events`);
  state.drawerES = es;
  let currentRunning = "";
  setDrawerHint("");
  es.onmessage = async (msg) => {
    if (msg.data === undefined) return; // heartbeat comment
    const ev = JSON.parse(msg.data);
    const rows = $$("#wf-stages .stage-row");
    if (["llm_start","llm_end","llm_error","tool_start","tool_end","tool_error"].includes(ev.type)) {
      appendAgentEvent(ev);
      return;
    }
    if (ev.type === "node" && ev.stage) currentRunning = ev.stage;
    if (ev.type === "stage") {
      currentRunning = ev.started || "";
      if (ev.completed) rows.forEach((r) => r.dataset.stage === ev.completed && (r.className = "stage-row done"));
      rows.forEach((r) => r.classList.toggle("running", r.dataset.stage === currentRunning));
    }
    if (ev.type === "status") {
      es.close();
      state.drawerES = null;
      setTimeout(async () => {
        if (state.drawerTaskId !== id) return;
        const fresh = await api(`/tasks/${id}`);
        renderDrawer(fresh);
        pollTick();
      }, 400);
    }
  };
  es.onerror = () => {
    es.close();
    if (state.drawerES !== es) return;
    state.drawerES = null;
    if (state.drawerTaskId === id) {
      setDrawerHint("⚠ 连接中断，正在自动重连…");
      setTimeout(() => { if (state.drawerTaskId === id) listenEvents(id); }, 2000);
    }
  };
}
function setDrawerHint(text) {
  let el = $("#drawer-hint");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("hidden", !text);
}
/* ---------------- 实时输出流 ---------------- */
const NODE_DISPLAY = {
  "Market Analyst": "市场分析师", "Sentiment Analyst": "情绪分析师", "News Analyst": "新闻分析师",
  "Global Macro Analyst": "全球宏观分析师",
  "Fundamentals Analyst": "基本面分析师", "Bull Researcher": "多方研究员", "Bear Researcher": "空方研究员",
  "Research Manager": "研究主管", "Trader": "交易员", "Aggressive Analyst": "激进派",
  "Conservative Analyst": "保守派", "Neutral Analyst": "中立派", "Portfolio Manager": "组合经理",
};
const EV_META = {
  llm_start:   ["🧠", "开始推理", "llm", false],
  llm_end:     ["📝", "输出结果", "llm-end", true],
  tool_start:  ["⚙️", "调用工具", "tool-start", false],
  tool_end:    ["✅", "工具返回", "tool-end", false],
  llm_error:   ["💥", "推理出错", "err", true],
  tool_error:  ["💥", "工具出错", "err", true],
};
function who(ev) { return NODE_DISPLAY[ev.node] || ev.node || "…"; }
function stageText(ev) { return STAGE_TEXT[ev.stage] || ""; }
function fmtSize(n) { return n == null ? "" : n > 1024 ? (n / 1024).toFixed(1) + "KB" : n + "B"; }

function appendAgentEvent(ev) {
  const [icon, title, cls, open] = EV_META[ev.type] || [];
  if (!icon) return;
  let body = "";
  if (ev.type === "llm_start") {
    body = `<span class="size-hint">${ev.msgs ?? "?"} 条上下文 · 模型 ${esc(ev.model || "")}</span>
            <pre>${esc(ev.input || "")}</pre>`;
  } else if (ev.type === "llm_end") {
    const chips = (ev.tool_calls || []).map((t) => `<span class="tc-chip">→ ${esc(t)}</span>`).join("");
    body = `${chips}${ev.reasoning ? `<pre class="muted" style="font-size:11px">${esc(ev.reasoning)}</pre>` : ""}
            <pre>${esc((ev.text || "").slice(-800))}</pre>
            <span class="size-hint">${fmtSize(ev.text_len)}${ev.tokens ? ` · ${ev.tokens} tokens` : ""}</span>`;
  } else if (ev.type === "tool_start") {
    body = `<pre>${esc(ev.args || "")}</pre>`;
  } else if (ev.type === "tool_end") {
    body = `<pre>${esc((ev.result || "").slice(-600))}</pre><span class="size-hint">完整输出 ${fmtSize(ev.size)}</span>`;
  } else {
    body = `<pre>${esc(ev.error || "")}</pre>`;
  }
  const el = document.createElement("details");
  el.className = `ev ${cls}`;
  el.dataset.kind = ev.type;
  if (open) el.open = true;
  el.innerHTML = `<summary>${icon}<span class="who">${esc(who(ev))}</span>${title}
      <span class="tag">${esc(stageText(ev))}</span></summary>${body}`;
  const feed = $("#agent-feed");
  if (!feed) return;
  // 过滤模式下新事件可能被隐藏
  applyFilterTo(el);
  feed.appendChild(el);
  state.feedCount++;
  const counter = $("#feed-count");
  if (counter) counter.textContent = `(${state.feedCount})`;
  pruneFeed(feed);
  if (state.feedScroll) feed.scrollTop = feed.scrollHeight;
}
function pruneFeed(feed) {
  while (feed.children.length > 400) feed.removeChild(feed.firstChild);
}
function applyFilterTo(el) {
  const mode = ($("#feed-filter") || {}).value || "";
  const kinds = { llm: ["llm_start", "llm_end"], tool: ["tool_start", "tool_end"], err: ["llm_error", "tool_error"] };
  el.style.display = (!mode || (kinds[mode] || []).includes(el.dataset.kind)) ? "" : "none";
}
function applyFeedFilter() { $$("#agent-feed .ev").forEach(applyFilterTo); }
function toggleFeedScroll() {
  state.feedScroll = !state.feedScroll;
  $("#feed-scroll-btn").textContent = `⬇ 自动滚动：${state.feedScroll ? "开" : "关"}`;
}

/* ---------------- 明日精选 ---------------- */
const screenPoll = { timer: null, fails: 0 };
let screenPrevStatus = null;

function loadPicks() {
  clearTimeout(screenPoll.timer);
  screenPoll.fails = 0;
  pollScreening();
}

async function deepResearch(code) {
  if (deepResearch._pending) return;               // 防连点
  deepResearch._pending = code;
  try {
    const { task_ids } = await api("/tasks", {
      method: "POST",
      body: { tickers: [code], output_language: "Chinese" },
    });
    showToast("深度研究已排队", `${code} · 全分析师团队`);
    go("#tasks");
    openDrawer(task_ids[0]);
  } catch (e) {
    showToast("排队失败", e.message, true);
  } finally { deepResearch._pending = null; }
}

function setRunButton(mode, extra = "") {
  const btn = $("#screen-run");
  if (mode === "running") { btn.disabled = true; btn.textContent = `筛选中… ${extra}`; }
  else if (mode === "starting") { btn.disabled = true; btn.textContent = "启动中…"; }
  else { btn.disabled = false; btn.textContent = "▶ 运行筛选"; }
}

async function startScreening() {
  setRunButton("starting");
  try {
    const resp = await api("/screen", { method: "POST" });
    showToast(
      resp.already_running ? "已有一轮筛选在运行，直接展示进度" : "筛选已启动",
      resp.already_running ? "轮询实时进度中" : "股票池拉取 → 逐股评估 → 概率合成",
    );
    screenPrevStatus = null;
    screenPoll.fails = 0;
    pollScreening();
  } catch (e) {
    setRunButton("idle");
    showToast("启动失败", e.message, true);
  }
}

async function cancelScreening() {
  const r = state.lastScreenRun;
  if (!r || r.status !== "running") return;
  try {
    await api("/screen/cancel", { method: "POST", body: { run_id: r.id } });
    showToast("已请求停止", "完成当前个股后即终止，已评估数据保留");
    $("#ss-stop").disabled = true;
  } catch (e) {
    showToast("停止失败", e.message, true);
  }
}

const STAGE_HINTS = {
  universe: { text: "拉取主板股票池（新浪全市场快照）", hint: "此阶段约需 15-20 秒，无逐只进度" },
  analyzing: { text: "逐股评估（拉历史 → 5+1因子 → 概率合成）", hint: "历史数据已缓存时会快很多" },
};

async function pollScreening() {
  clearTimeout(screenPoll.timer);
  let r = null;
  try {
    const data = await api("/screen/latest");
    r = data.run;
    screenPoll.fails = 0;
  } catch (e) {
    screenPoll.fails++;
  }

  if (r === null) {
    // 轮询失败：有界退避重试，绝不死锁按钮
    if (screenPoll.watching || screenPoll.fails <= 3) {
      $("#screen-meta").innerHTML =
        `<span style="color:var(--amber)">连接波动，${3 * screenPoll.fails}s 后自动重试（第 ${screenPoll.fails} 次）…</span>`;
      screenPoll.timer = setTimeout(pollScreening, 3000 * screenPoll.fails);
    } else {
      setRunButton("idle");
      $("#screen-status").classList.add("hidden");
      $("#screen-meta").innerHTML =
        `<span style="color:var(--red)">与服务的连接失败，请确认服务仍在运行后刷新本页。</span>`;
    }
    return;
  }

  state.lastScreenRun = r;
  renderScreenStatus(r);

  if (r.status === "running") {
    screenPoll.watching = true;
    setRunButton("running", `${r.processed ?? 0}/${r.total ?? "?"}`);
    screenPoll.timer = setTimeout(pollScreening, 2000);
    return;
  }

  // ── 终态 ──
  screenPoll.watching = false;
  setRunButton("idle");
  $("#screen-status").classList.add("hidden");
  renderScreenHistory();

  const sawRunning = screenPrevStatus === "running";
  if (r.status === "failed") {
    $("#screen-meta").innerHTML =
      `运行失败：<span style="color:var(--red)">${esc((r.error || "").slice(0, 160))}</span> · ${fmtTime(r.created_at)}`;
    $("#picks-list").innerHTML = "";
    if (sawRunning) showToast("筛选失败", (r.error || "").slice(0, 80), true);
    screenPrevStatus = r.status;
    return;
  }
  if (r.status === "cancelled") {
    $("#screen-meta").textContent =
      `已停止 · 中止前评估 ${r.evaluated ?? r.processed ?? 0} 只 · ${fmtTime(r.created_at)}`;
    const wl = r.watchlist || [];
    $("#picks-list").innerHTML =
      `<div class="task-empty">筛选已停止（已评估个股的结果已缓存，重新运行会快很多）。</div>` +
      (wl.length ? `<h3 style="margin:12px 0 8px">👁 中止前观察名单</h3>` +
        wl.map((p, i) => pickCard(p, i, true)).join("") : "");
    if (sawRunning) showToast("筛选已停止", "已评估数据保留在缓存中");
    screenPrevStatus = r.status;
    return;
  }

  const picks = r.picks || [], watchlist = r.watchlist || [];
  $("#screen-meta").textContent =
    `最近完成 ${fmtTime(r.created_at)} · 股票池 ${r.universe ?? "?"} · 评估 ${r.evaluated ?? "?"} 只 · 达标(≥80%) ${r.qualifying ?? 0} 只`;
  let html = "";
  if ((r.qualifying ?? 0) > 0) {
    html += `<h3 style="margin:4px 0 10px">✅ 达标精选（P≥80%）</h3>` +
      picks.map((p, i) => pickCard(p, i)).join("");
  } else {
    html += `<div class="task-empty">今日无标的达到 80% 概率阈值 —— 空仓等待也是模型的建议。以下观察名单仅跟踪参考。</div>`;
  }
  html += `<h3 style="margin:16px 0 10px">👁 观察名单 Top5（未达 80%）</h3>` +
    (watchlist.map((p, i) => pickCard(p, i, true)).join("") || `<div class="muted">无</div>`);
  $("#picks-list").innerHTML = html;

  if (sawRunning) {
    showToast(
      `筛选完成：达标 ${r.qualifying ?? 0} 只`,
      (r.qualifying ?? 0) > 0
        ? `最高概率 ${(Math.max(...picks.map(p => p.probability)) * 100).toFixed(0)}%`
        : "今日无达标标的，已展示观察名单",
    );
    setTimeout(() => $("#picks-list").scrollIntoView({ behavior: "smooth", block: "start" }), 150);
  }
  screenPrevStatus = r.status;
}

function renderScreenStatus(r) {
  const card = $("#screen-status");
  if (r.status !== "running") { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  const st = STAGE_HINTS[r.stage] || STAGE_HINTS.analyzing;
  $("#ss-stage").textContent = st.text;
  $("#ss-hint").textContent = st.hint;
  const done = r.processed ?? 0, total = r.total ?? 0;
  $("#ss-count").textContent = total ? `${done}/${total} 只 · 达标 ${r.qualifying ?? 0}` : "";
  $("#ss-bar-wrap").classList.toggle("indeterminate", !total);
  $("#ss-bar").style.width = total ? Math.round((done / total) * 100) + "%" : "30%";
  $("#ss-stop").disabled = false;
  const started = r.created_at ? (Date.now() - r.created_at * 1000) / 1000 : 0;
  $("#ss-elapsed").textContent = started
    ? `已运行 ${Math.floor(started / 60)}分${String(Math.floor(started % 60)).padStart(2, "0")}秒`
    : "";
}

async function renderScreenHistory() {
  const data = await api("/screen/history?limit=8").catch(() => null);
  const box = $("#screen-history");
  if (!box) return;
  const runs = (data && data.runs) || [];
  if (!runs.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<h3>历史运行</h3><table class="sh-table">
    <tr><th>时间</th><th>状态</th><th>股票池</th><th>评估</th><th>达标</th><th>最高概率</th></tr>` +
    runs.map((h) => `<tr>
      <td>${fmtTime(h.created_at)}</td>
      <td><span class="badge ${h.status}">${STATUS_TEXT[h.status] || h.status}</span></td>
      <td>${h.universe ?? "—"}</td><td>${h.evaluated ?? "—"}</td>
      <td>${h.qualifying ?? "—"}</td>
      <td>${h.top_probability != null ? (h.top_probability * 100).toFixed(0) + "%" : "—"}</td>
    </tr>`).join("") + `</table>`;
}

function showToast(title, sub = "", isErr = false) {
  let wrap = $(".toast-wrap");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "toast-wrap";
    document.body.appendChild(wrap);
  }
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.innerHTML = `<div class="t-title">${esc(title)}</div>${sub ? `<div class="t-sub">${esc(sub)}</div>` : ""}`;
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

/* ---------------- 实时状态：全局轮询 + 常驻状态栏 ---------------- */
function curView() { return (location.hash || "#dashboard").slice(1) || "dashboard"; }

function renderActiveStrip(tasks) {
  const strip = $("#active-strip");
  const running = tasks.filter((t) => t.status === "running");
  const pendingCount = tasks.filter((t) => t.status === "pending").length;
  if (!running.length && !pendingCount) { strip.classList.add("hidden"); return; }
  strip.classList.remove("hidden");
  const now = Date.now();
  strip.innerHTML =
    running.map((t) => {
      const secs = t.started_at ? Math.max(0, (now - t.started_at * 1000) / 1000) : 0;
      const mm = `${Math.floor(secs / 60)}分${String(Math.floor(secs % 60)).padStart(2, "0")}秒`;
      return `<span class="run-item" onclick="openDrawer('${t.id}')">
        <span class="dot"></span>${esc(t.ticker)} · ${STAGE_TEXT[t.current_stage] || "启动中"} ·
        <span class="elapsed">${mm}</span></span>`;
    }).join("") +
    (pendingCount ? `<span class="queue-chip">排队中 ×${pendingCount}</span>` : "");
}

async function pollTick() {
  if (document.hidden) return;
  let tasks = [];
  try { ({ tasks } = await api("/tasks?limit=200")); } catch (_) { return; }
  renderActiveStrip(tasks);
  const v = curView();
  // 只静默刷新数据型视图，避免重建表单/设置页打字内容
  if (v === "dashboard") { renderStats(tasks); renderRecent(tasks.slice(0, 10)); }
  else if (v === "tasks") renderTaskTable(tasks);
}
setInterval(pollTick, 4000);

window.addEventListener("hashchange", route);

document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  route();
  loadHealthBadge();

  $("#f-date").valueAsDate = new Date();
  $("#new-form").addEventListener("submit", submitAnalysis);
  $("#fav-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await api("/favorites", { method: "POST", body: { code: $("#fav-code").value.trim(), name: $("#fav-name").value.trim() } });
    $("#fav-code").value = ""; $("#fav-name").value = ""; loadFavorites();
  });
  $("#settings-form").addEventListener("submit", saveSettings);
  $$("#task-filters .chip-btn").forEach((b) => b.addEventListener("click", () => {
    $$("#task-filters .chip-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.filterStatus = b.dataset.status;
    loadTasks();
  }));
  $("#drawer-mask").addEventListener("click", () => closeDrawer());
  document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());
});
