/* VisualAgents dashboard - vanilla JS, no build step. */
(() => {
  "use strict";

  // ---------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs = {}) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const now = () => Date.now() / 1000;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const ICONS = {
    thought: '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.9 10.6c.6.5.9 1.3.9 2.4h6c0-1.1.3-1.9.9-2.4A6 6 0 0 0 12 3z"/>',
    tool: '<path d="M14.7 6.3a4 4 0 0 0-5.3 5.3L3.5 17.5a2 2 0 0 0 3 3l5.9-5.9a4 4 0 0 0 5.3-5.3l-2.6 2.6-2.3-.6-.6-2.3z"/>',
    tool_input: '<path d="M4 7h16M4 12h10M4 17h7"/>',
    observation: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    output: '<path d="M4 4v16M4 4h11l-2 4 2 4H4"/>',
    handoff: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    error: '<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17.5v.01"/>',
    task: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>',
    message: '<path d="M4 5h16v11H8l-4 4z"/>',
    narration: '<path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7-4.7-1.8 4.7-1.8z"/><path d="M19 15l.7 1.8 1.8.7-1.8.7L19 20l-.7-1.8-1.8-.7 1.8-.7z"/>',
    status: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16v.01"/>',
    log: '<path d="M4 6l5 6-5 6M12 18h8"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    wait: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    visits: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    auto: '<circle cx="12" cy="12" r="9"/><path d="M12 3v18" /><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/>',
  };
  const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.log}</svg>`;
  const KIND_LABEL = {
    thought: "Thought", tool: "Tool", tool_input: "Input", observation: "Result", output: "Answer",
    handoff: "Handoff", error: "Error", task: "Task", message: "Message", narration: "Now", status: "", log: "",
  };
  const MERGEABLE = new Set(["log", "thought", "output", "observation", "tool_input", "message", "status"]);
  const LABEL_PREFIX = /^#*\s*(?:Thought|Thinking|Final Answer|Using Tool|Tool Input|Tool Output|Tool Result|Action Input|Action|Observation|Task|Answer|Output)\s*:\s*/i;
  const START = "__start__", END = "__end__";
  const NODE_W = 240, NODE_H = 96, TERM_W = 76, TERM_H = 36, GAP_X = 92, GAP_Y = 64, GAP_TERM = 56;

  // ---------------------------------------------------------------- state
  const S = {
    bp: null, st: null, meta: {}, agents: [], byId: {}, slot: {},
    events: [], seqMax: 0,
    layout: null, layoutSig: "", nodeEls: {}, edgeEls: {}, termEls: {},
    view: { x: 0, y: 0, k: 1 }, userMoved: false, follow: true,
    filter: null, hideLogs: false, autoScroll: true, lastItem: null,
    modalAuto: false, lastActive: undefined, pendingAnims: [], raf: 0,
    drawerId: null, drawerTimer: 0, narrated: false, wasDisconnected: false, es: null,
    bubbleText: "",
  };

  const wrap = $("#canvas-wrap"), canvas = $("#canvas"), edgesSvg = $("#edges"), nodesEl = $("#nodes");
  const feedEl = $("#feed");

  // ---------------------------------------------------------------- data
  async function loadState() {
    const res = await fetch("/api/state", { cache: "no-store" });
    const data = await res.json();
    S.bp = data.blueprint;
    S.meta = data.meta || {};
    S.st = data.state;
    S.seqMax = 0;
    S.events = [];
    S.lastItem = null;
    S.layoutSig = "";
    feedEl.textContent = "";
    S.narrated = false;
    syncAgents();
    for (const ev of data.events) addEvent(ev, false);
    S.seqMax = Math.max(S.seqMax, data.state.seq || 0);
    renderBlueprintModal();
    renderFilters();
    renderState();
    if (!S.userMoved) fit(false);
    maybeAutoOpenBlueprint();
  }

  function connect() {
    if (S.es) S.es.close();
    const es = new EventSource(`/api/stream?since=${S.seqMax}`);
    S.es = es;
    es.addEventListener("ev", (e) => {
      const ev = JSON.parse(e.data);
      if (ev.seq <= S.seqMax) return;
      S.seqMax = ev.seq;
      addEvent(ev, true);
    });
    es.addEventListener("state", (e) => {
      S.st = JSON.parse(e.data);
      scheduleRender();
    });
    es.onopen = () => {
      if (S.wasDisconnected) {
        S.wasDisconnected = false;
        es.close();
        loadState().then(connect).catch(() => setTimeout(connect, 1500));
        return;
      }
      setRunStatus();
    };
    es.onerror = () => {
      S.wasDisconnected = true;
      setRunStatus(true);
    };
  }

  function syncAgents() {
    const extra = (S.st && S.st.extra_agents) || [];
    const ids = new Set(S.bp.agents.map((a) => a.id));
    S.agents = [...S.bp.agents, ...extra.filter((a) => !ids.has(a.id))];
    S.byId = {};
    for (const a of S.agents) {
      S.byId[a.id] = a;
      if (!(a.id in S.slot)) S.slot[a.id] = Object.keys(S.slot).length;
    }
  }

  const colorVar = (id) => {
    const s = S.slot[id];
    return s == null || s >= 8 ? "var(--c-other)" : `var(--c${s + 1})`;
  };
  const nameOf = (id) => (id === START ? "Start" : id === END ? "End" : (S.byId[id] && S.byId[id].name) || id || "system");

  // ---------------------------------------------------------------- graph model
  function graphEdges() {
    const out = [];
    const seen = new Set();
    const push = (e) => {
      const key = `${e.from}->${e.to}`;
      if (seen.has(key) || e.from === e.to) return;
      seen.add(key);
      out.push({ ...e, key });
    };
    for (const id of S.bp.entry) push({ from: START, to: id, type: "terminal", label: "" });
    for (const e of S.bp.edges) push({ ...e, type: e.conditional ? "conditional" : "solid" });
    for (const id of S.bp.exits) push({ from: id, to: END, type: "terminal", label: "" });
    const counts = (S.st && S.st.edge_counts) || {};
    for (const [f, t] of (S.st && S.st.dynamic_edges) || []) {
      if (S.byId[f] && S.byId[t]) push({ from: f, to: t, type: "dynamic", label: "" });
    }
    for (const key of Object.keys(counts)) {
      const [f, t] = key.split("->");
      if ((f === START && S.byId[t]) || (t === END && S.byId[f])) push({ from: f, to: t, type: "terminal", label: "" });
    }
    return out;
  }

  function computeOrder() {
    const ids = [START, ...S.agents.map((a) => a.id), END];
    const edges = graphEdges();
    const out = {};
    ids.forEach((id) => (out[id] = []));
    edges.forEach((e) => out[e.from] && out[e.to] && out[e.from].push(e));

    // break cycles: DFS from START marks back edges
    const mark = {};
    const back = new Set();
    const dfs = (u) => {
      mark[u] = 1;
      for (const e of out[u]) {
        if (mark[e.to] === 1) back.add(e.key);
        else if (!mark[e.to]) dfs(e.to);
      }
      mark[u] = 2;
    };
    dfs(START);
    ids.forEach((id) => { if (!mark[id] && id !== END) dfs(id); });

    // longest-path layering over forward edges
    const layer = {};
    const indeg = {};
    ids.forEach((id) => (indeg[id] = 0));
    const fwd = edges.filter((e) => !back.has(e.key) && out[e.from] && out[e.to]);
    fwd.forEach((e) => indeg[e.to]++);
    const queue = ids.filter((id) => indeg[id] === 0);
    ids.forEach((id) => (layer[id] = id === START ? 0 : 1));
    while (queue.length) {
      const u = queue.shift();
      for (const e of fwd) {
        if (e.from !== u) continue;
        layer[e.to] = Math.max(layer[e.to], layer[u] + 1);
        if (--indeg[e.to] === 0) queue.push(e.to);
      }
    }
    layer[START] = 0;
    layer[END] = Math.max(1, ...S.agents.map((a) => layer[a.id])) + 1;

    // order within layers: barycenter sweeps
    const layers = [];
    ids.forEach((id) => (layers[layer[id]] = layers[layer[id]] || []).push(id));
    for (let i = 0; i < layers.length; i++) layers[i] = layers[i] || [];
    const pos = {};
    const reindex = () => layers.forEach((L) => L.forEach((id, i) => (pos[id] = i)));
    reindex();
    for (let sweep = 0; sweep < 6; sweep++) {
      const down = sweep % 2 === 0;
      const order = down ? layers.map((_, i) => i) : layers.map((_, i) => layers.length - 1 - i);
      for (const li of order) {
        const bary = {};
        for (const id of layers[li]) {
          const nbrs = edges
            .filter((e) => (down ? e.to === id && layer[e.from] < li : e.from === id && layer[e.to] > li))
            .map((e) => pos[down ? e.from : e.to]);
          bary[id] = nbrs.length ? nbrs.reduce((a, b) => a + b, 0) / nbrs.length : pos[id];
        }
        layers[li].sort((a, b) => bary[a] - bary[b] || pos[a] - pos[b]);
        reindex();
      }
    }
    return { layers, layer, edges, back };
  }

  // Place nodes left-to-right ("LR") or top-to-bottom ("TB") and route the edges.
  function place(order, orient) {
    const { layers, layer, edges, back } = order;
    const lr = orient === "LR";
    const gapLayer = lr ? GAP_X : 150; // TB leaves room under each node for the thought bubble
    const gapCross = lr ? GAP_Y : 48;
    const along = (id) => (lr ? sizeOf(id).w : sizeOf(id).h);
    const across = (id) => (lr ? sizeOf(id).h : sizeOf(id).w);
    const crossLen = layers.map((L) => L.reduce((s, id) => s + across(id), 0) + Math.max(0, L.length - 1) * gapCross);
    const widest = Math.max(...crossLen, lr ? NODE_H : NODE_W);
    const nodes = {};
    let a = 0, end = 0;
    layers.forEach((L, li) => {
      const thick = Math.max(...L.map(along), lr ? TERM_W : TERM_H);
      let c = (widest - crossLen[li]) / 2;
      for (const id of L) {
        const { w, h } = sizeOf(id);
        nodes[id] = lr
          ? { x: a + (thick - w) / 2, y: c, w, h, layer: li }
          : { x: c, y: a + (thick - h) / 2, w, h, layer: li };
        c += across(id) + gapCross;
      }
      end = a + thick;
      a += thick + (li === 0 || (lr && li === layers.length - 2) ? GAP_TERM : gapLayer);
    });
    const all = Object.values(nodes);
    const right = Math.max(...all.map((n) => n.x + n.w));
    const bottom = Math.max(...all.map((n) => n.y + n.h));
    let backCount = 0;
    const routed = edges.map((e) => {
      const p = nodes[e.from], q = nodes[e.to];
      const same = layer[e.from] === layer[e.to];
      const isBack = back.has(e.key) || layer[e.to] < layer[e.from];
      const idx = isBack || same ? backCount++ : 0;
      return { ...e, back: isBack, d: lr ? pathLR(p, q, isBack, same, idx, bottom) : pathTB(p, q, isBack, same, idx, right) };
    });
    const extra = backCount ? 70 + backCount * 18 : 0;
    const w = (lr ? end : right + extra) + 40;
    const h = (lr ? bottom + extra : end) + 130;
    return { orient, nodes, edges: routed, bounds: { x: -20, y: -20, w: w + 20, h: h + 20 } };
  }

  function pathLR(a, b, isBack, same, index, bottom) {
    if (same) {
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x + b.w, y2 = b.y + b.h / 2, bulge = 70 + index * 14;
      return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`;
    }
    if (!isBack) {
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2, dx = Math.max(40, (x2 - x1) / 2);
      return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
    }
    const x1 = a.x + a.w / 2 - 24, y1 = a.y + a.h, x2 = b.x + b.w / 2 + 24, y2 = b.y + b.h;
    const Y = (bottom + 46 + index * 18 - (y1 + y2) / 8) / 0.75;
    return `M${x1},${y1} C${x1},${Y} ${x2},${Y} ${x2},${y2}`;
  }

  function pathTB(a, b, isBack, same, index, right) {
    if (same) {
      const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y + b.h, bulge = 60 + index * 14;
      return `M${x1},${y1} C${x1},${y1 + bulge} ${x2},${y2 + bulge} ${x2},${y2}`;
    }
    if (!isBack) {
      const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y, dy = Math.max(40, (y2 - y1) / 2);
      return `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
    }
    const x1 = a.x + a.w, y1 = a.y + a.h / 2 - 12, x2 = b.x + b.w, y2 = b.y + b.h / 2 + 12;
    const X = (right + 46 + index * 18 - (x1 + x2) / 8) / 0.75;
    return `M${x1},${y1} C${X},${y1} ${X},${y2} ${x2},${y2}`;
  }

  function fitScale(b) {
    const W = wrap.clientWidth || 1000, H = wrap.clientHeight || 700;
    return Math.min((W - 80) / b.w, (H - 80) / b.h);
  }

  function computeLayout() {
    const order = computeOrder();
    const lr = place(order, "LR"), tb = place(order, "TB");
    return fitScale(tb.bounds) > fitScale(lr.bounds) * 1.1 ? tb : lr;
  }

  function sizeOf(id) {
    return id === START || id === END ? { w: TERM_W, h: TERM_H } : { w: NODE_W, h: NODE_H };
  }

  // ---------------------------------------------------------------- static render (layout changes)
  function renderGraph() {
    S.layout = computeLayout();
    const L = S.layout;
    const shiftX = -L.bounds.x, shiftY = -L.bounds.y;
    canvas.style.width = `${L.bounds.w}px`;
    canvas.style.height = `${L.bounds.h}px`;
    edgesSvg.setAttribute("width", L.bounds.w);
    edgesSvg.setAttribute("height", L.bounds.h);
    edgesSvg.textContent = "";
    const defs = svgEl("defs");
    for (const [id, cls] of [["arrow", "arrow"], ["arrow-traversed", "arrow traversed"], ["arrow-dynamic", "arrow dynamic"]]) {
      const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
      m.appendChild(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: cls }));
      defs.appendChild(m);
    }
    edgesSvg.appendChild(defs);
    const g = svgEl("g", { transform: `translate(${shiftX},${shiftY})` });
    edgesSvg.appendChild(g);
    S.edgeGroup = g;
    S.edgeEls = {};
    for (const e of L.edges) {
      const path = svgEl("path", { d: e.d, class: `edge ${e.type}`, "marker-end": `url(#${e.type === "dynamic" ? "arrow-dynamic" : "arrow"})` });
      path.style.setProperty("--hot", colorVar(e.to));
      g.appendChild(path);
      const label = svgEl("text", { class: "edge-label", "text-anchor": "middle" });
      g.appendChild(label);
      S.edgeEls[e.key] = { path, label, edge: e, mid: null };
    }
    // labels after paths exist
    requestAnimationFrame(() => {
      for (const key in S.edgeEls) {
        const ref = S.edgeEls[key];
        try {
          const len = ref.path.getTotalLength();
          ref.mid = ref.path.getPointAtLength(len / 2);
          ref.label.setAttribute("x", ref.mid.x);
          ref.label.setAttribute("y", ref.mid.y - 7);
        } catch { /* path not rendered */ }
      }
      updateEdges();
    });

    nodesEl.textContent = "";
    nodesEl.style.transform = `translate(${shiftX}px,${shiftY}px)`;
    S.nodeEls = {};
    S.termEls = {};
    for (const id of [START, END]) {
      const n = L.nodes[id];
      const t = el("div", `terminal-node ${id === START ? "start" : "end"}`, id === START ? "START" : "END");
      Object.assign(t.style, { left: `${n.x}px`, top: `${n.y}px` });
      nodesEl.appendChild(t);
      S.termEls[id] = t;
    }
    for (const a of S.agents) {
      const n = L.nodes[a.id];
      const node = el("div", "node s-idle" + (a.dynamic ? " dynamic" : ""));
      node.style.left = `${n.x}px`;
      node.style.top = `${n.y}px`;
      node.style.setProperty("--agent", colorVar(a.id));
      node.dataset.id = a.id;
      node.tabIndex = 0;
      node.setAttribute("role", "button");
      node.setAttribute("aria-label", `${a.name}: ${a.role || "agent"}`);
      node.innerHTML = `
        <div class="node-top"><span class="node-dot"></span><span class="node-name"></span><span class="status-tag s-idle"></span></div>
        <div class="node-role"></div>
        <div class="node-activity"></div>
        <div class="node-meta"></div>`;
      $(".node-name", node).textContent = a.name;
      $(".node-role", node).textContent = a.role || (a.dynamic ? "discovered at runtime" : "");
      node.title = a.description || a.role || a.name;
      node.addEventListener("click", () => openDrawer(a.id));
      node.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openDrawer(a.id); } });
      nodesEl.appendChild(node);
      S.nodeEls[a.id] = node;
    }
    const bubble = el("div", "bubble");
    bubble.hidden = true;
    bubble.innerHTML = '<div class="bubble-kind"></div><div class="bubble-text"></div>';
    nodesEl.appendChild(bubble);
    S.bubble = bubble;
    S.bubbleText = "";
  }

  // ---------------------------------------------------------------- dynamic render
  function scheduleRender() {
    if (S.raf) return;
    S.raf = requestAnimationFrame(() => {
      S.raf = 0;
      renderState();
    });
  }

  function renderState() {
    if (!S.bp || !S.st) return;
    syncAgents();
    const sig = S.agents.map((a) => a.id).join(",") + "|" + graphEdges().map((e) => e.key).join(",");
    if (sig !== S.layoutSig) {
      S.layoutSig = sig;
      renderGraph();
      renderFilters();
      renderBlueprintModal();
      if (!S.userMoved) fit(false);
    }
    updateNodes();
    updateEdges();
    renderNow();
    renderStepper();
    renderPath();
    setRunStatus();
    renderEmptyHint();
    const active = S.st.active;
    if (active !== S.lastActive) {
      S.lastActive = active;
      if (active && S.follow) ensureVisible(active);
      document.title = active ? `● ${nameOf(active)} · VisualAgents` : `VisualAgents · ${S.bp.project}`;
    }
    while (S.pendingAnims.length) animateHandoff(S.pendingAnims.shift());
    if (S.drawerId) refreshDrawerSoon();
  }

  const STATUS_TAG = {
    active: () => `<span class="typing"><i></i><i></i><i></i></span>Working`,
    waiting: () => `${icon("wait")}Waiting`,
    done: () => `${icon("check")}Done`,
    error: () => `${icon("x")}Error`,
    idle: () => "",
  };

  function updateNodes() {
    const st = S.st;
    for (const a of S.agents) {
      const node = S.nodeEls[a.id];
      if (!node) continue;
      const s = st.agents[a.id] || { status: "idle", activity: "", visits: 0, tools: 0 };
      const status = s.status || "idle";
      node.className = `node s-${status}${a.dynamic ? " dynamic" : ""}`;
      const tag = $(".status-tag", node);
      tag.className = `status-tag s-${status}`;
      tag.innerHTML = STATUS_TAG[status] ? STATUS_TAG[status]() : "";
      $(".node-activity", node).textContent = status === "active" ? "" : s.activity || "";
      const meta = $(".node-meta", node);
      const parts = [];
      if (s.visits > 1) parts.push(`<span title="times this agent took over">${icon("visits")}×${s.visits}</span>`);
      if (s.tools) parts.push(`<span title="tool calls">${icon("tool")}${s.tools}</span>`);
      if (s.errors) parts.push(`<span title="errors">${icon("error")}${s.errors}</span>`);
      meta.innerHTML = parts.join("");
    }
    const started = (st.path_total || 0) > 0;
    S.termEls[START] && S.termEls[START].classList.toggle("lit", started);
    S.termEls[END] && S.termEls[END].classList.toggle("lit", st.run.status === "finished");

    // thought bubble under the active agent
    const b = S.bubble;
    const act = st.active && S.layout && S.layout.nodes[st.active];
    if (!act) {
      b.hidden = true;
      return;
    }
    const s = st.agents[st.active] || {};
    const text = s.activity || "Working…";
    const kind = s.activity_kind || "log";
    b.hidden = false;
    b.style.setProperty("--agent", colorVar(st.active));
    b.style.left = `${act.x - 20}px`;
    b.style.top = `${act.y + act.h + 14}px`;
    if (text !== S.bubbleText) {
      S.bubbleText = text;
      $(".bubble-kind", b).innerHTML = `${icon(kind)}${KIND_LABEL[kind] || "Working"}`;
      $(".bubble-text", b).textContent = text;
      b.classList.remove("fresh");
      void b.offsetWidth;
      b.classList.add("fresh");
    }
  }

  function updateEdges() {
    const st = S.st;
    const counts = st.edge_counts || {};
    const last = st.path && st.path.length ? st.path[st.path.length - 1] : null;
    const hotKey = st.active && last ? `${last.from || START}->${last.agent}` : null;
    for (const key in S.edgeEls) {
      const { path, label, edge } = S.edgeEls[key];
      const n = counts[key] || 0;
      path.classList.toggle("traversed", n > 0);
      path.classList.toggle("hot", key === hotKey);
      if (edge.type !== "dynamic") path.setAttribute("marker-end", `url(#${n > 0 ? "arrow-traversed" : "arrow"})`);
      const text = [edge.label, n > 1 ? `×${n}` : ""].filter(Boolean).join("  ");
      label.textContent = text;
      label.setAttribute("class", `edge-label${n ? " traversed" : ""}`);
    }
  }

  function renderNow() {
    const box = $("#now");
    const st = S.st;
    const steps = S.bp.steps || [];
    const stepTxt = st.step_index >= 0 && steps[st.step_index] ? `Step ${st.step_index + 1} of ${steps.length} · ${steps[st.step_index].title}` : "";
    const id = st.active;
    if (id && S.byId[id]) {
      const a = S.byId[id];
      const s = st.agents[id] || {};
      const last = st.path[st.path.length - 1];
      const since = last ? fmtDur(now() - last.start) : "";
      box.style.setProperty("--agent", colorVar(id));
      box.innerHTML = `
        <div class="now-eyebrow"><span>Working now</span><span>${esc(stepTxt)}</span></div>
        <div class="now-agent"><span class="node-dot"></span><div><div class="now-name">${esc(a.name)}</div><div class="now-role">${esc(a.role || "")}</div></div></div>
        <div class="now-activity">${esc(s.activity || "Working…")}</div>
        <div class="now-stats"><span>in this agent <b>${since}</b></span><span>visit <b>#${s.visits || 1}</b></span><span>tool calls <b>${s.tools || 0}</b></span><span>handoffs <b>${Math.max(0, (st.path_total || 1) - 1)}</b></span></div>`;
      return;
    }
    box.style.removeProperty("--agent");
    const run = st.run;
    if (run.status === "finished" || run.status === "failed") {
      const dur = run.started && run.ended ? fmtDur(run.ended - run.started) : "";
      const ok = run.status === "finished";
      const lastAgent = st.path.length ? nameOf(st.path[st.path.length - 1].agent) : "-";
      box.innerHTML = `
        <div class="now-eyebrow"><span>${ok ? "Run finished" : "Run failed"}</span><span>${esc(dur)}</span></div>
        <div class="now-agent"><span class="status-tag ${ok ? "s-done" : "s-error"}">${icon(ok ? "check" : "x")}${ok ? "Complete" : `Exit code ${run.exit_code}`}</span></div>
        <div class="now-stats"><span>agents used <b>${Object.values(st.agents).filter((x) => x.visits).length}</b></span><span>handoffs <b>${Math.max(0, (st.path_total || 1) - 1)}</b></span><span>final agent <b>${esc(lastAgent)}</b></span></div>`;
      return;
    }
    const waiting = run.status === "running" ? "Output is flowing - waiting for the first agent to speak up…" : S.meta.mode === "blueprint" ? "Blueprint view - no run attached." : "Waiting for the run to start…";
    box.innerHTML = `<div class="now-eyebrow"><span>Working now</span><span>${esc(stepTxt)}</span></div><div class="now-role" style="margin-top:8px">${esc(waiting)}</div>`;
  }

  function renderStepper() {
    const nav = $("#stepper");
    const steps = S.bp.steps || [];
    const st = S.st;
    const finished = st.run.status === "finished";
    nav.textContent = "";
    steps.forEach((s, i) => {
      const visited = s.agents.some((id) => st.agents[id] && st.agents[id].visits > 0);
      const current = i === st.step_index && !!st.active;
      const done = !current && visited && (i < st.step_index || finished || st.step_index > i);
      const item = el("div", `step${current ? " current" : ""}${done ? " done" : ""}`);
      const num = el("span", "step-num");
      if (done) num.innerHTML = icon("check");
      else num.textContent = String(i + 1);
      const body = el("div");
      body.appendChild(el("div", "step-title", s.title));
      const dots = el("div", "step-agents");
      for (const id of s.agents) {
        const d = el("i");
        d.style.setProperty("--agent", colorVar(id));
        d.title = nameOf(id);
        dots.appendChild(d);
      }
      body.appendChild(dots);
      item.title = s.description || s.title;
      item.append(num, body);
      item.addEventListener("mouseenter", () => highlight(s.agents));
      item.addEventListener("mouseleave", () => highlight(null));
      nav.appendChild(item);
    });
  }

  function highlight(ids) {
    for (const id in S.nodeEls) S.nodeEls[id].classList.toggle("dimmed", !!ids && !ids.includes(id));
  }

  function renderPath() {
    const box = $("#path");
    const st = S.st;
    const atEnd = box.scrollLeft + box.clientWidth >= box.scrollWidth - 30;
    box.textContent = "";
    const crumb = (id, cls, sub) => {
      const c = el("span", `crumb ${cls || ""}`);
      c.style.setProperty("--agent", colorVar(id));
      const dot = el("i");
      c.append(dot, document.createTextNode(nameOf(id)));
      if (sub) c.appendChild(el("small", "", sub));
      return c;
    };
    const sep = () => el("span", "crumb-sep", "›");
    if (!st.path.length) {
      box.appendChild(el("span", "muted small", "No handoffs yet."));
      return;
    }
    if ((st.path_total || 0) > st.path.length) box.append(el("span", "muted small", "…"), sep());
    else box.append(crumb(START, ""), sep());
    st.path.forEach((p, i) => {
      const isCur = i === st.path.length - 1 && st.active === p.agent;
      const dur = p.end ? fmtDur(p.end - p.start) : isCur ? fmtDur(now() - p.start) : "";
      box.appendChild(crumb(p.agent, isCur ? "current" : "", dur));
      if (i < st.path.length - 1) box.appendChild(sep());
    });
    if (st.run.status === "finished") box.append(sep(), crumb(END, ""));
    if (atEnd || S.autoScroll) box.scrollLeft = box.scrollWidth;
  }

  function setRunStatus(disconnected) {
    const pill = $("#run-status");
    const st = S.st;
    let status = st ? st.run.status : "connecting";
    if (disconnected || S.wasDisconnected) status = "disconnected";
    if (S.meta.mode === "blueprint" && status === "idle") status = "blueprint";
    const labels = { connecting: "Connecting", idle: "Waiting", running: "Running", finished: "Finished", failed: "Failed", disconnected: "Reconnecting…", blueprint: "Blueprint" };
    pill.dataset.status = status;
    $(".label", pill).textContent = labels[status] || status;
    tickTimer();
    $("#project-name").textContent = S.bp ? S.bp.project : "";
    $("#framework").textContent = S.bp ? S.bp.framework : "";
    const attr = !st ? "" : st.precise ? "instrumented" : S.narrated ? "logs + LLM" : st.run.status === "idle" ? "" : "from logs";
    $("#attribution").textContent = attr;
  }

  function tickTimer() {
    const t = $("#timer");
    const run = S.st && S.st.run;
    if (!run || !run.started) { t.textContent = ""; return; }
    const end = run.ended || now();
    t.textContent = fmtClock(end - run.started);
    if (run.status === "running" && S.st.active) {
      const cur = $(".crumb.current small");
      const last = S.st.path[S.st.path.length - 1];
      if (cur && last) cur.textContent = fmtDur(now() - last.start);
    }
  }

  function renderEmptyHint() {
    const box = $("#empty-hint");
    if (!S.agents.length) {
      box.hidden = false;
      box.innerHTML = `<b>No agents found yet.</b><br>Try an LLM scan for a deeper read of the code:<br><br><code>visualagents scan --llm</code>`;
    } else if (S.meta.mode === "blueprint" && S.st.run.status === "idle" && !S.st.path_total) {
      box.hidden = true;
    } else {
      box.hidden = true;
    }
  }

  // ---------------------------------------------------------------- feed
  function addEvent(ev, live) {
    S.events.push(ev);
    if (S.events.length > 8000) S.events.splice(0, 2000);
    if (ev.kind === "narration") S.narrated = true;
    if (live && ev.kind === "handoff") {
      S.pendingAnims.push(ev);
      if (S.modalAuto && !$("#blueprint-modal").hidden) setTimeout(() => S.modalAuto && closeModal(), 900);
    }
    if (passes(ev)) appendFeed(ev, feedEl, true);
    if (live) scheduleRender();
  }

  function passes(ev) {
    if (S.hideLogs && (ev.kind === "log" || ev.kind === "status")) return false;
    if (!S.filter) return true;
    if (ev.agent === S.filter) return true;
    return ev.kind === "handoff" && ev.data && (ev.data.frm === S.filter || ev.data.to === S.filter);
  }

  function displayText(ev) {
    let t = ev.text || "";
    if (ev.source === "log") t = t.replace(LABEL_PREFIX, "");
    if (ev.kind === "tool" && ev.data && ev.data.tool && t.trim() === ev.data.tool) return "";
    return t;
  }

  function appendFeed(ev, container, track) {
    const li = track ? S.lastItem : null;
    if (li && li.container === container && li.agent === ev.agent && li.kind === ev.kind && MERGEABLE.has(ev.kind)
        && ev.ts - li.ts < 4 && li.lines < 60 && ev.kind !== "handoff") {
      li.body.textContent += "\n" + displayText(ev);
      li.lines++;
      li.ts = ev.ts;
      checkLong(li.body);
      if (container === feedEl) stickToEnd();
      return;
    }
    const item = el("div", `fi k-${ev.kind}`);
    if (ev.agent) item.style.setProperty("--agent", colorVar(ev.agent));
    item.appendChild(el("span", "fi-time", fmtTime(ev.ts)));
    const head = el("div", "fi-head");
    if (ev.kind === "handoff") {
      const d = ev.data || {};
      const line = el("span", "handoff-line");
      line.innerHTML = `${agentChip(d.frm)}${icon("handoff")}${agentChip(d.to || ev.agent)}`;
      head.appendChild(line);
      if (ev.text && ev.text.includes("(returns)")) head.appendChild(el("span", "fi-kind", "returns"));
    } else {
      head.insertAdjacentHTML("beforeend", agentChip(ev.agent));
      if (KIND_LABEL[ev.kind]) {
        const k = el("span", "fi-kind");
        k.innerHTML = `${icon(ev.kind)}${KIND_LABEL[ev.kind]}`;
        head.appendChild(k);
      }
      if (ev.data && ev.data.tool) head.appendChild(el("span", "fi-tool", ev.data.tool));
    }
    item.appendChild(head);
    const body = el("div", "fi-body", displayText(ev));
    body.addEventListener("click", () => body.classList.toggle("open"));
    item.appendChild(body);
    container.appendChild(item);
    if (track) S.lastItem = { container, agent: ev.agent, kind: ev.kind, ts: ev.ts, body, lines: 1 };
    while (container.childElementCount > 1500) container.firstElementChild.remove();
    checkLong(body);
    if (container === feedEl) stickToEnd();
  }

  function stickToEnd() {
    if (!S.autoScroll || S.stickRaf) return;
    S.stickRaf = requestAnimationFrame(() => {
      S.stickRaf = 0;
      S.programmaticScroll = true;
      feedEl.scrollTop = feedEl.scrollHeight;
    });
  }

  function checkLong(body) {
    requestAnimationFrame(() => body.classList.toggle("long", body.scrollHeight > body.clientHeight + 2));
  }

  function agentChip(id) {
    if (!id) return `<span class="fi-agent muted">system</span>`;
    return `<span class="fi-agent" style="--agent:${colorVar(id)}"><i></i>${esc(nameOf(id))}</span>`;
  }

  function rebuildFeed() {
    feedEl.textContent = "";
    S.lastItem = null;
    const evs = S.events.filter(passes).slice(-1500);
    S.autoScroll = true;
    $("#jump").hidden = true;
    for (const ev of evs) appendFeed(ev, feedEl, true);
  }

  function renderFilters() {
    const bar = $("#filters");
    bar.textContent = "";
    const mk = (id, label) => {
      const c = el("button", "chip");
      c.type = "button";
      if (id) {
        c.style.setProperty("--agent", colorVar(id));
        c.appendChild(el("i"));
      }
      c.appendChild(document.createTextNode(label));
      c.setAttribute("aria-pressed", String(S.filter === id));
      c.addEventListener("click", () => {
        S.filter = S.filter === id ? null : id;
        renderFilters();
        rebuildFeed();
      });
      bar.appendChild(c);
    };
    mk(null, "All");
    for (const a of S.agents) mk(a.id, a.name);
  }

  // ---------------------------------------------------------------- animation & viewport
  function animateHandoff(ev) {
    const d = ev.data || {};
    const ref = S.edgeEls[`${d.frm}->${d.to}`];
    if (!ref || !S.edgeGroup) return;
    const path = ref.path;
    let len;
    try { len = path.getTotalLength(); } catch { return; }
    const dot = svgEl("circle", { r: "6", class: "particle" });
    dot.style.setProperty("--hot", colorVar(d.to));
    S.edgeGroup.appendChild(dot);
    const t0 = performance.now(), dur = 800;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      const pt = path.getPointAtLength(len * e);
      dot.setAttribute("cx", pt.x);
      dot.setAttribute("cy", pt.y);
      if (p < 1) requestAnimationFrame(step);
      else dot.remove();
    };
    requestAnimationFrame(step);
  }

  function applyView(animate) {
    canvas.classList.toggle("animate", !!animate);
    const v = S.view;
    canvas.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.k})`;
  }

  function fit(animate) {
    if (!S.layout) return;
    const b = S.layout.bounds;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (!W || !H) return;
    const pad = 40;
    const k = clamp(Math.min((W - pad * 2) / b.w, (H - pad * 2) / b.h), 0.3, 1.2);
    S.view = { k, x: (W - b.w * k) / 2, y: Math.max(pad / 2, (H - b.h * k) / 2) };
    applyView(animate);
  }

  function ensureVisible(id) {
    const n = S.layout && S.layout.nodes[id];
    if (!n) return;
    const v = S.view, W = wrap.clientWidth, H = wrap.clientHeight;
    const ox = -S.layout.bounds.x, oy = -S.layout.bounds.y;
    const x1 = (n.x + ox - 30) * v.k + v.x, y1 = (n.y + oy - 20) * v.k + v.y;
    const x2 = (n.x + ox + n.w + 60) * v.k + v.x, y2 = (n.y + oy + n.h + 120) * v.k + v.y;
    if (x1 >= 0 && y1 >= 0 && x2 <= W && y2 <= H) return;
    S.view.x = W / 2 - (n.x + ox + n.w / 2) * v.k;
    S.view.y = H / 2 - (n.y + oy + n.h / 2 + 40) * v.k;
    applyView(true);
  }

  function zoomAt(factor, mx, my) {
    const v = S.view;
    const k2 = clamp(v.k * factor, 0.2, 2.5);
    v.x = mx - (mx - v.x) * (k2 / v.k);
    v.y = my - (my - v.y) * (k2 / v.k);
    v.k = k2;
    S.userMoved = true;
    applyView(false);
  }

  function bindViewport() {
    wrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = wrap.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    let drag = null;
    wrap.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".node, button, .legend, .empty-hint")) return;
      drag = { x: e.clientX, y: e.clientY, vx: S.view.x, vy: S.view.y, id: e.pointerId };
      wrap.setPointerCapture(e.pointerId);
      wrap.classList.add("panning");
    });
    wrap.addEventListener("pointermove", (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      S.view.x = drag.vx + e.clientX - drag.x;
      S.view.y = drag.vy + e.clientY - drag.y;
      S.userMoved = true;
      applyView(false);
    });
    const end = () => { drag = null; wrap.classList.remove("panning"); };
    wrap.addEventListener("pointerup", end);
    wrap.addEventListener("pointercancel", end);
    new ResizeObserver(() => {
      if (S.layout && S.bp && computeLayout().orient !== S.layout.orient) {
        renderGraph();
        S.userMoved = false;
        updateNodes();
        updateEdges();
      }
      if (!S.userMoved) fit(false);
    }).observe(wrap);
    const center = () => [wrap.clientWidth / 2, wrap.clientHeight / 2];
    $("#zoom-in").onclick = () => zoomAt(1.2, ...center());
    $("#zoom-out").onclick = () => zoomAt(1 / 1.2, ...center());
    $("#fit").onclick = () => { S.userMoved = false; fit(true); };
    $("#follow").onclick = (e) => {
      S.follow = !S.follow;
      e.currentTarget.setAttribute("aria-pressed", String(S.follow));
      if (S.follow && S.st && S.st.active) ensureVisible(S.st.active);
    };
  }

  // ---------------------------------------------------------------- blueprint modal
  function renderBlueprintModal() {
    const bp = S.bp;
    $("#bp-title").textContent = bp.project;
    const body = $("#bp-body");
    const chips = (ids) => ids.map((id) => `<span class="chip" style="--agent:${colorVar(id)}"><i></i>${esc(nameOf(id))}</span>`).join("");
    const facts = [
      bp.framework && `<span class="badge">${esc(bp.framework)}</span>`,
      `<span class="badge">${S.agents.length} agent${S.agents.length === 1 ? "" : "s"}</span>`,
      `<span class="badge">${bp.edges.length} handoff${bp.edges.length === 1 ? "" : "s"}</span>`,
      bp.entry.length && `<span class="badge">starts at ${esc(bp.entry.map(nameOf).join(", "))}</span>`,
      bp.exits.length && `<span class="badge">answers from ${esc(bp.exits.map(nameOf).join(", "))}</span>`,
    ].filter(Boolean).join("");
    const steps = (bp.steps || []).map((s) => `
      <div class="bp-step">
        <div class="bp-step-num">STEP ${s.order}</div>
        <h4>${esc(s.title)}</h4>
        <div class="chips">${chips(s.agents)}</div>
        ${s.description ? `<p>${esc(s.description)}</p>` : ""}
      </div>`).join("");
    const agents = S.agents.map((a) => `
      <div class="bp-agent" style="--agent:${colorVar(a.id)}">
        <h4><span class="node-dot"></span>${esc(a.name)}${a.dynamic ? ' <span class="tag">runtime</span>' : ""}</h4>
        ${a.role ? `<div class="role">${esc(a.role)}</div>` : ""}
        ${a.description ? `<p>${esc(a.description)}</p>` : ""}
        ${a.tools && a.tools.length ? `<div class="tools">${a.tools.map((t) => `<span class="tool-chip">${esc(t)}</span>`).join("")}</div>` : ""}
        ${a.file ? `<div class="src">${esc(a.file)}${a.line ? ":" + a.line : ""}</div>` : ""}
      </div>`).join("");
    const edges = bp.edges.map((e) => `
      <div class="bp-edge">${chips([e.from])}<span class="arrow-txt">${e.conditional ? "⇢" : "→"}</span>${chips([e.to])}
        ${e.label ? `<span class="lbl">${esc(e.label)}</span>` : ""}${e.conditional ? '<span class="tag">conditional</span>' : ""}</div>`).join("");
    body.innerHTML = `
      ${bp.summary ? `<p class="bp-summary">${esc(bp.summary)}</p>` : ""}
      <div class="bp-facts">${facts}</div>
      ${steps ? `<div class="bp-section-title">How a query flows</div><div class="bp-steps">${steps}</div>` : ""}
      <div class="bp-section-title">Agents</div><div class="bp-agents">${agents || '<span class="muted">No agents detected.</span>'}</div>
      ${edges ? `<div class="bp-section-title">Handoffs</div><div class="bp-edges">${edges}</div>` : ""}`;
    const gen = bp.generated_by ? `Mapped by ${bp.generated_by}` : "";
    $("#bp-meta").textContent = [gen, bp.generated_at ? new Date(bp.generated_at).toLocaleString() : ""].filter(Boolean).join(" · ");
  }

  function openModal(auto) {
    S.modalAuto = !!auto;
    $("#blueprint-modal").hidden = false;
    $(".modal-card [data-close].icon-btn").focus({ preventScroll: true });
  }

  function closeModal() {
    S.modalAuto = false;
    $("#blueprint-modal").hidden = true;
  }

  function maybeAutoOpenBlueprint() {
    const key = `va-bp-seen:${S.bp.project}`;
    const mode = S.meta.mode;
    if (mode === "blueprint" || mode === "demo" || !store.get(key)) {
      store.set(key, "1");
      if (!(S.st.active && mode !== "blueprint")) openModal(mode !== "blueprint");
    }
  }

  // ---------------------------------------------------------------- drawer
  function openDrawer(id) {
    S.drawerId = id;
    $("#drawer").hidden = false;
    renderDrawer();
  }

  function closeDrawer() {
    S.drawerId = null;
    $("#drawer").hidden = true;
  }

  function refreshDrawerSoon() {
    if (S.drawerTimer) return;
    S.drawerTimer = setTimeout(() => { S.drawerTimer = 0; if (S.drawerId) renderDrawer(); }, 600);
  }

  function renderDrawer() {
    const id = S.drawerId;
    const a = S.byId[id];
    if (!a) return closeDrawer();
    const s = (S.st.agents && S.st.agents[id]) || {};
    const title = $("#drawer-title");
    title.style.setProperty("--agent", colorVar(id));
    title.innerHTML = `<div class="now-agent" style="margin:0"><span class="node-dot"></span><div><div class="now-name">${esc(a.name)}</div><div class="now-role">${esc(a.role || "")}</div></div></div>`;
    const body = $("#drawer-body");
    const keepScroll = body.scrollTop;
    body.style.setProperty("--agent", colorVar(id));
    body.innerHTML = `
      <span class="status-tag s-${s.status || "idle"}" style="display:inline-flex">${(STATUS_TAG[s.status] || STATUS_TAG.idle)() || "Not started"}</span>
      ${a.description ? `<p>${esc(a.description)}</p>` : ""}
      ${a.tools && a.tools.length ? `<div class="tools" style="display:flex;gap:4px;flex-wrap:wrap">${a.tools.map((t) => `<span class="tool-chip">${esc(t)}</span>`).join("")}</div>` : ""}
      ${a.file ? `<div class="src muted small" style="font-family:var(--mono);margin-top:8px">${esc(a.file)}${a.line ? ":" + a.line : ""}</div>` : ""}
      <div class="stat-grid">
        <div class="stat"><b>${s.visits || 0}</b><span>turns</span></div>
        <div class="stat"><b>${s.tools || 0}</b><span>tool calls</span></div>
        <div class="stat"><b>${s.events || 0}</b><span>lines</span></div>
        <div class="stat"><b>${s.errors || 0}</b><span>errors</span></div>
      </div>
      <div class="bp-section-title" style="margin-top:4px">Its chain of thought</div>
      <div class="feed" id="drawer-feed"></div>`;
    const feed = $("#drawer-feed");
    const evs = S.events.filter((ev) => ev.agent === id).slice(-400);
    if (!evs.length) feed.appendChild(el("div", "muted small", "Nothing yet."));
    const saved = S.lastItem;
    S.lastItem = null;
    for (const ev of evs) appendFeed(ev, feed, true);
    S.lastItem = saved;
    body.scrollTop = keepScroll;
  }

  // ---------------------------------------------------------------- theme
  const THEMES = ["auto", "light", "dark"];
  function applyTheme(mode) {
    const root = document.documentElement;
    if (mode === "auto") delete root.dataset.theme;
    else root.dataset.theme = mode;
    const btn = $("#btn-theme");
    btn.innerHTML = icon(mode === "auto" ? "auto" : mode === "light" ? "sun" : "moon");
    btn.title = `Theme: ${mode}`;
    store.set("va-theme", mode);
  }

  // ---------------------------------------------------------------- formatting
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function fmtClock(sec) {
    sec = Math.max(0, Math.floor(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return (h ? `${h}:${String(m).padStart(2, "0")}` : `${String(m).padStart(2, "0")}`) + `:${String(s).padStart(2, "0")}`;
  }
  function fmtDur(sec) {
    if (!(sec >= 0)) return "";
    if (sec < 1) return "<1s";
    if (sec < 60) return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
    return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
  }
  function fmtTime(ts) {
    const start = S.st && S.st.run && S.st.run.started;
    if (start && ts >= start) return fmtClock(ts - start);
    const d = new Date(ts * 1000);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  // ---------------------------------------------------------------- wiring
  function bindUI() {
    applyTheme(store.get("va-theme") || "auto");
    $("#btn-theme").onclick = () => {
      const cur = store.get("va-theme") || "auto";
      applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
    };
    $("#btn-blueprint").onclick = () => ($("#blueprint-modal").hidden ? openModal(false) : closeModal());
    document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeModal));
    $("#drawer-close").onclick = closeDrawer;
    $("#hide-logs").onchange = (e) => { S.hideLogs = e.target.checked; rebuildFeed(); };
    $("#clear-feed").onclick = () => { S.events = []; rebuildFeed(); };
    $("#reset-view").onclick = () => fetch("/api/reset", { method: "POST" });
    feedEl.addEventListener("scroll", () => {
      const atEnd = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 40;
      if (S.programmaticScroll) {
        S.programmaticScroll = false;
        if (!atEnd) feedEl.scrollTop = feedEl.scrollHeight;
        return;
      }
      S.autoScroll = atEnd;
      $("#jump").hidden = atEnd;
    });
    $("#jump").onclick = () => { feedEl.scrollTop = feedEl.scrollHeight; S.autoScroll = true; $("#jump").hidden = true; };
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input, textarea")) return;
      if (e.key === "Escape") { closeModal(); closeDrawer(); }
      else if (e.key === "b" || e.key === "B") $("#btn-blueprint").click();
      else if (e.key === "f" || e.key === "F") { S.userMoved = false; fit(true); }
    });
    bindViewport();
    setInterval(tickTimer, 1000);
  }

  bindUI();
  loadState()
    .then(connect)
    .catch((err) => {
      console.error(err);
      setRunStatus(true);
      setTimeout(() => loadState().then(connect), 2000);
    });
})();
