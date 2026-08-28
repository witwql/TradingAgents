/**
 * Frontend replay harness: runs the real app.js against a minimal DOM shim
 * and the live backend, replaying a user flow to surface runtime errors that
 * API-level tests cannot see (the retrospective class of bugs).
 *
 * Usage: node tools/frontend_replay.js <base_url> [flow]
 *   flow: picks (default) | favs
 */
const http = require("http");
const fs = require("fs");
const vm = require("vm");

const BASE = process.argv[2] || "http://127.0.0.1:8000";
const FLOW = process.argv[3] || "picks";

// ---------- minimal DOM ----------
function el(id) {
  const e = {
    id, innerHTML: "", textContent: "", value: "", disabled: false,
    href: "#", className: "", dataset: {}, size: 0,
    style: {}, children: [],
    classList: {
      _s: new Set(),
      add(...c) { c.forEach((x) => this._s.add(x)); },
      remove(...c) { c.forEach((x) => this._s.delete(x)); },
      toggle(c, force) { force ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    appendChild(ch) { this.children.push(ch); return ch; },
    removeChild(ch) { this.children = this.children.filter((x) => x !== ch); return ch; },
    after() {}, remove() {}, click() {}, focus() {},
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    scrollIntoView() {},
    insertAdjacentHTML(pos, html) { this.innerHTML += html; },
    onchange: null, onerror: null, onmessage: null, onload: null,
  };
  if (id === "screen-meta") {
    let _t = "";
    Object.defineProperty(e, "textContent", {
      get() { return _t; },
      set(v) { _t = String(v); console.log("[trace] screen-meta.textContent 被赋值:", _t.slice(0, 60)); },
    });
  }
  return e;
}

const elements = {};
const listeners = {};
const documentShim = {
  body: el("body"),
  querySelector(sel) {
    const m = sel.match(/^#([\w-]+)$/);
    if (m) { elements[m[1]] = elements[m[1]] || el(m[1]); return elements[m[1]]; }
    return null; // class-based selectors return nothing
  },
  querySelectorAll(sel) {
    // 处理任务筛选按钮等 class 选择器：返回空集不影响主流程
    return [];
  },
  createElement(tag) { return el("dyn-" + tag + "-" + Math.random().toString(36).slice(2, 7)); },
  addEventListener(ev, fn) { listeners[ev] = fn; },
  hidden: false,
};
elements["drawer-mask"] = el("drawer-mask");

function fetchShim(path, opts = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(BASE + path);
    const req = http.request({
      hostname: url.hostname, port: url.port || 80, path: url.pathname + url.search,
      method: opts.method || "GET",
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve({
        ok: res.statusCode < 400,
        status: res.statusCode,
        json: async () => JSON.parse(body),
        text: async () => body,
      }));
    });
    req.on("error", reject);
    if (opts.body) req.write(typeof opts.body === "string" ? opts.body : JSON.stringify(opts.body));
    req.end();
  });
}

const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: fetchShim,
  document: documentShim,
  location: { hash: "#picks" },
  window: { addEventListener() {} },
  EventSource: class { close() {} set onmessage(_) {} set onerror(_) {} },
  URL, Blob: class {}, alert() {},
};
sandbox.globalThis = sandbox;

const code = fs.readFileSync("server/static/app.js", "utf8");
try {
  vm.runInNewContext(code, sandbox, { filename: "app.js" });
} catch (e) {
  console.log("LOAD ERROR:", e.message);
  process.exit(1);
}
console.log("app.js loaded ✓");

process.on("unhandledRejection", (e) => {
  console.log("!! 未处理的 Promise 拒绝（页面会静默吞掉的错误）:", e && e.message);
  console.log((e && e.stack || "").split("\n").slice(0, 4).join("\n"));
  process.exitCode = 1;
});

(async () => {
  try {
    // 触发初始路由（模拟 DOMContentLoaded）
    if (listeners.DOMContentLoaded) listeners.DOMContentLoaded();
    sandbox.location.hash = "#picks";
    if (listeners.hashchange) listeners.hashchange();

    // 等待异步链就绪（meta 出现内容或超时）
    let meta = null;
    for (let i = 0; i < 25; i++) {
      await new Promise((r) => setTimeout(r, 200));
      meta = elements["screen-meta"];
      if (meta && meta.textContent && meta.textContent.length > 0) break;
    }

    const list = elements["picks-list"];
    const hist = elements["screen-history"];
    console.log("---- 重放结果 ----");
    console.log("screen-meta:", (meta ? meta.textContent : "<元素未创建>").slice(0, 90));
    console.log("picks-list html 长度:", list ? list.innerHTML.length : 0,
      "| 含 pick-card:", list ? list.innerHTML.includes("pick-card") : false);
    console.log("screen-history 长度:", hist ? hist.innerHTML.length : 0,
      "| 含表头:", hist ? hist.innerHTML.includes("历史运行") : false);
    const btn = elements["screen-run"];
    console.log("run 按钮: disabled =", btn ? btn.disabled : "?", "| text:", btn ? btn.textContent : "?");
    const card = elements["screen-status"];
    console.log("状态卡 hidden:", card ? card.classList.contains("hidden") : "?");

    // 连续三次再轮询，验证终态稳定（用户报告的“突然消失”）
    for (let i = 0; i < 3; i++) {
      await sandbox.pollScreening ? sandbox.pollScreening() : null;
      await new Promise((r) => setTimeout(r, 400));
    }
    console.log("二次轮询后 meta:", (meta ? meta.textContent : "").slice(0, 90));
    console.log("[state] lastScreenRun.status =", sandbox.state.lastScreenRun && sandbox.state.lastScreenRun.status);
    console.log("[state] lastScreenRun.watchlist =", (sandbox.state.lastScreenRun && sandbox.state.lastScreenRun.watchlist || []).length);
    console.log("二次轮询后 picks-list 长度:", list ? list.innerHTML.length : 0);
    console.log("REPLAY DONE");
    process.exit(process.exitCode || 0);
  } catch (e) {
    console.log("!! 运行时异常（这就是用户看到的崩溃点）:", e.message);
    console.log(e.stack ? e.stack.split("\n").slice(0, 4).join("\n") : "");
    process.exit(1);
  }
})();
