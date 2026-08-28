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
const state = { filterStatus: "", drawerTaskId: null, drawerES: null, cache: {}, feedCount: 0, feedScroll: true };

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
    <div id="drawer-result"></div>`;
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
function loadPicks() { pollScreening(false); }

async function deepResearch(code) {
  try {
    const { task_ids } = await api("/tasks", {
      method: "POST",
      body: { tickers: [code], output_language: "Chinese" },
    });
    showToast("深度研究已排队", `${code} · 全分析师团队 · 打开任务面板查看进度`);
    go("#tasks");
    openDrawer(task_ids[0]);
  } catch (e) {
    showToast("排队失败", e.message, true);
  }
}

async function startScreening() {
  const btn = $("#screen-run");
  btn.disabled = true; btn.textContent = "筛选中…(约1-3分钟)";
  try {
    await api("/screen", { method: "POST" });
    pollScreening(true);
  } catch (e) {
    btn.disabled = false; btn.textContent = "▶ 运行筛选";
    $("#screen-meta").textContent = `启动失败：${e.message}`;
  }
}

let screeningTimer = null;
async function pollScreening(active) {
  const data = await api("/screen/latest").catch(() => null);
  if (!data || !data.run) {
    if (!active) $("#screen-meta").textContent = "尚未运行过筛选";
    return;
  }
  const r = data.run;
  if (r.status === "running") {
    $("#screen-meta").textContent = `筛选运行中…（股票池约 ${r.universe || "…"} 只，后台拉取历史+计算因子）`;
    clearTimeout(screeningTimer);
    screeningTimer = setTimeout(() => pollScreening(true), 5000);
    return;
  }
  const btn = $("#screen-run");
  btn.disabled = false; btn.textContent = "▶ 运行筛选";
  if (r.status === "failed") {
    $("#screen-meta").innerHTML = `运行失败：<span style="color:var(--red)">${esc(r.error || "")}</span>`;
    return;
  }
  const达标 = r.qualifying > 0;
  $("#screen-meta").textContent =
    `运行于 ${fmtTime(r.created_at)} · 评估 ${r.evaluated} 只 · 达标(≥80%) ${r.qualifying} 只`;
  let html = "";
  if (达标) {
    html += `<h3 style="margin:4px 0 10px">✅ 达标精选</h3>`;
    html += r.picks.map((p, i) => pickCard(p, i)).join("");
  } else {
    html += `<div class="task-empty">今日无标的达到 80% 概率阈值 —— 空仓等待也是模型的建议。<br/>以下为概率最高的观察名单（未达标）</div>`;
  }
  html += `<h3 style="margin:16px 0 10px">👁 观察名单 Top5（未达 80%，仅跟踪参考）</h3>`;
  html += (r.watchlist || []).map((p, i) => pickCard(p, i, true)).join("")
    || `<div class="muted">无</div>`;
  $("#picks-list").innerHTML = html;
}

function pickCard(p, idx, muted = false) {
  const chips = (p.contributions || [])
    .filter((c) => c.fired)
    .map((c) => c.used
      ? `<span class="factor-chip">${esc(c.factor)} <b>+${((c.p - 0.5) * 100).toFixed(0)}pp</b> <span class="n">n=${c.n}</span></span>`
      : `<span class="factor-chip" style="opacity:.55">${esc(c.factor)} <span class="n">样本不足</span></span>`)
    .join("");
  const calib = p.resonance_hit_rate != null
    ? `共振日历史命中率 <b style="color:var(--text)">${(p.resonance_hit_rate * 100).toFixed(1)}%</b>（${p.resonance_samples} 个样本）`
    : `共振日历史样本不足（${p.resonance_samples}），无校准数据`;
  const pct = Math.round(p.probability * 100);
  const probColor = pct >= 80 ? "var(--green)" : "var(--amber)";
  return `<div class="pick-card"${muted ? ' style="opacity:.82"' : ""}>
    <div class="pick-head">
      <div class="pick-rank">${idx + 1}</div>
      <span class="pick-ticker">${esc(p.code)}</span>
      <span>${esc(p.name)}</span>
      <span class="pick-price">¥${(p.close ?? 0).toFixed(2)}</span>
      <span class="muted" style="font-size:12px">命中因子 ${p.factors_fired}/5</span>
      <div class="pick-prob">
        <div class="num" style="color:${probColor}">${pct}%</div>
        <div class="lbl">P(次日上涨)</div>
      </div>
      <button class="btn small primary" onclick="event.stopPropagation();deepResearch('${p.code}')">🔬 深度研究</button>
    </div>
    <div class="prob-bar"><i style="width:${pct}%"></i></div>
    <div class="factor-chips">${chips}</div>
    <div class="calib">📏 ${calib} · 统计窗口 ${p.history_days} 交易日</div>
  </div>`;
}

function closeDrawer(reload = true) {
  if (state.drawerES) { state.drawerES.close(); state.drawerES = null; }
  $("#drawer-mask").classList.add("hidden");
  $("#drawer").classList.remove("open");
  state.drawerTaskId = null;
  if (reload) loadView(location.hash.slice(1) || "dashboard");
}

/* ---------------- reports ---------------- */
async function loadReports(presetId) {
  const { tasks } = await api("/tasks?status=completed&limit=50");
  const sel = $("#report-task");
  sel.innerHTML = tasks
    .map((t) => `<option value="${t.id}">${esc(t.ticker)} · ${esc(t.trade_date)} · ${esc(fmtTime(t.created_at))}${presetId === t.id ? " selected" : ""}</option>`)
    .join("") || `<option disabled>暂无已完成的分析（确认端口=8000 并强刷；完整报告文件始终保留在 ~/.tradingagents/logs/reports/）</option>`;
  sel.onchange = () => loadReportFiles(sel.value);
  const first = presetId || tasks[0]?.id;
  if (first) { sel.value = first; loadReportFiles(first); }
}
async function loadReportFiles(id) {
  if (!id) return;
  const manifest = await api(`/tasks/${id}/reports`);
  state.cache.reportManifest = manifest.files;
  const picker = $("#report-task");
  // second column: use the aside select's sibling file list injected as another select
  let files = $("#report-files");
  if (!files) {
    files = document.createElement("select");
    files.id = "report-files";
    files.size = 14;
    files.style.marginTop = "10px";
    picker.after(files);
    files.onchange = () => showReport(id, files.value);
  }
  files.innerHTML = manifest.files
    .map((f) => `<option value="${esc(f.path)}">${esc(f.path)}</option>`).join("")
    || `<option disabled>无报告文件</option>`;
  const prefer = ["complete_report.md", "5_portfolio/decision.md"].find((p) => manifest.files.some((f) => f.path === p));
  if (prefer) { files.value = prefer; showReport(id, prefer); }
}
async function showReport(id, path) {
  const data = await api(`/tasks/${id}/report?path=${encodeURIComponent(path)}`);
  $("#report-title").textContent = path;
  const raw = $("#open-raw");
  raw.classList.remove("hidden");
  raw.href = "#";
  raw.onclick = (e) => { e.preventDefault(); download(path, data.content); };
  $("#report-body").innerHTML = markdown(data.content);
}
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  a.download = name.split("/").pop();
  a.click();
}
function openReportPicker(id) { route(); }

/* ---------------- favorites ---------------- */
async function loadFavorites() {
  const data = await api("/favorites");
  const { favorites, quote_ts: ts, refresh_seconds: cycle } = data;
  const ready = data.quotes_ready !== false;  // 旧后端无此字段时视为就绪
  const note = ready
    ? (ts ? `行情 ${fmtTime(ts)} 更新` + (cycle ? ` · 每${Math.round(cycle / 60)}分钟自动刷新` : "") : "")
    : "行情首次加载中…数秒后自动重试";
  $("#fav-table").innerHTML = `<tr><th>代码</th><th>名称</th><th>最新价</th><th>涨跌幅</th><th></th><th>
      <span class="muted" style="font-weight:400;font-size:11px">${note}</span></th></tr>` +
    (favorites.map((f) => `<tr>
      <td><strong>${esc(f.code)}</strong></td><td>${esc(f.name || f.info?.name || "")}</td>
      <td>${f.price ?? "—"}</td>
      <td class="${pctTrend(f.pct)}">${fmtPct(f.pct)}</td>
      <td><button class="btn small" onclick='quickAnalyze("${f.code}")'>快速分析</button></td>
      <td><button class="btn danger small" onclick="removeFav('${f.code}')">×</button></td></tr>`).join("")
      || `<tr><td colspan="6" class="muted">自选股为空。若你之前添加过，请确认地址栏端口（默认 8000）并强制刷新 —— 数据库路径见 <a href="#" onclick="fetch('/api/health').then(h=>h.json()).then(h=>alert('当前数据库：'+h.db_path));return false">/api/health</a></td></tr>`);
  if (!ready && !loadFavorites._retried) {
    loadFavorites._retried = true;
    setTimeout(() => { if (curView() === "favorites") loadFavorites(); }, 4000);
  }
}
/* 行情涨跌遵循 A 股配色：涨红跌绿 */
function pctTrend(pct) { return (pct ?? 0) >= 0 ? "up" : "down"; }
function fmtPct(pct) {
  if (pct == null || isNaN(pct)) return "—";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}% ${pct >= 0 ? "▲" : "▼"}`;
}
async function removeFav(code) { await api(`/favorites/${encodeURIComponent(code)}`, { method: "DELETE" }); loadFavorites(); }
async function quickAnalyze(code) {
  const { task_ids } = await api("/tasks", { method: "POST", body: { tickers: [code] } });
  go("#tasks"); openDrawer(task_ids[0]);
}

/* ---------------- settings ---------------- */
async function loadSettings() {
  const [settings, health] = await Promise.all([api("/settings"), api("/health")]);
  $("#s-region").value = settings.glm_region || "glm-cn";
  $("#s-deep").value = settings.deep_model || settings.glm_model || "glm-5.2";
  $("#s-quick").value = settings.quick_model || "";
  $("#s-temp").value = settings.temperature || "";
  $("#s-autoscreen").value = settings.auto_screen_time || "15:30";
  const hasKey = settings.glm_region === "glm" ? health.has_zhipu_intl_key : health.has_zhipu_cn_key;
  $("#key-state").textContent = hasKey
    ? `✅ 已检测到 API Key（区域：${settings.glm_region === "glm" ? "Z.AI 国际站" : "BigModel 中国区"}），可以直接提交分析。`
    : `⚠️ 尚未检测到 API Key —— 提交任务会失败。按下方指引配置 .env 后重启服务。`;
  const badge = $("#key-badge");
  badge.textContent = hasKey ? "Key ✓" : "Key ✗";
  badge.className = `chip ${hasKey ? "ok" : "warn"}`;
  return { settings };
}
async function saveSettings(ev) {
  ev.preventDefault();
  const body = {};
  const deep = $("#s-deep").value.trim(), quick = $("#s-quick").value.trim(),
    temp = $("#s-temp").value.trim();
  body.glm_region = $("#s-region").value;
  body.deep_model = deep; body.glm_model = deep;
  if (quick) body.quick_model = quick;
  if (temp && !isNaN(+temp)) body.temperature = temp;
  const autoscreen = $("#s-autoscreen").value.trim();
  body.auto_screen_time = autoscreen || "off";
  try {
    await api("/settings", { method: "PUT", body });
    $("#settings-msg").textContent = "已保存 ✓";
    loadHealthBadge();
  } catch (e) { $("#settings-msg").textContent = `保存失败：${e.message}`; }
}
async function loadHealthBadge() {
  try {
    const h = await api("/health");
    const settings = await api("/settings");
    const hasKey = settings.glm_region === "glm" ? h.has_zhipu_intl_key : h.has_zhipu_cn_key;
    $("#llm-badge").textContent = (settings.deep_model || settings.glm_model || "GLM-5.2").toUpperCase();
    const badge = $("#key-badge");
    badge.textContent = hasKey ? "Key ✓" : "Key ✗";
    badge.className = `chip ${hasKey ? "ok" : "warn"}`;
  } catch (_) {}
}

/* ---------------- tiny markdown renderer ---------------- */
function markdown(src) {
  const lines = esc(src).replace(/\r/g, "").split("\n");
  let html = "", inCode = false, listMode = "", tableBuf = [];
  const flushTable = () => {
    if (!tableBuf.length) return;
    html += "<table>" + tableBuf.map((cells, i) =>
      "<tr>" + cells.map((c) => `<${i === 0 ? "th" : "td"}>${inline(c)}</${i === 0 ? "th" : "td"}>`).join("") + "</tr>").join("") + "</table>";
    tableBuf = [];
  };
  const inline = (t) => t
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^```/.test(line)) { flushTable(); html += inCode ? "</pre>" : "<pre>"; inCode = !inCode; continue; }
    if (inCode) { html += line + "\n"; continue; }
    if (/^\|(.+)\|$/.test(line.trim())) {
      const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue;
      tableBuf.push(cells); continue;
    } else flushTable();
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)/))) { const l = m[1].length; html += `<h${l}>${inline(m[2])}</h${l}>`; }
    else if (/^(---+|\*\*\*+)$/.test(line.trim())) html += "<hr/>";
    else if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
      if (listMode !== "ul") { html += "<ul>"; listMode = "ul"; } html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (listMode !== "ol") { html += "<ol>"; listMode = "ol"; } html += `<li>${inline(m[1])}</li>`;
    } else if ((m = line.match(/^>\s?(.*)/))) html += `<blockquote>${inline(m[1])}</blockquote>`;
    else if (!line.trim()) { if (listMode) { html += `</${listMode}>`; listMode = ""; } }
    else { if (listMode) { html += `</${listMode}>`; listMode = ""; } html += `<p>${inline(line)}</p>`; }
  }
  flushTable();
  if (listMode) html += `</${listMode}>`;
  if (inCode) html += "</pre>";
  return html;
}

/* ---------------- plumbing ---------------- */
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
function loadView(name) {
  try {
    ({ dashboard: loadDashboard, new: buildAnalystPicker, tasks: loadTasks,
       picks: loadPicks, reports: loadReports, favorites: loadFavorites,
       settings: loadSettings }[name] || (() => {}) )();
  } catch (e) {
    console.error(`view ${name} failed:`, e);
    const box = $(`section[data-view="${name}"]`);
    if (box) box.insertAdjacentHTML("afterbegin",
      `<div class="task-empty" style="border-color:var(--red)">视图加载出错：${esc(e.message)}（可尝试强制刷新 Cmd+Shift+R）</div>`);
  }
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
