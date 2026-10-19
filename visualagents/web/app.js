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
    play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor"/>',
    pause: '<rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor"/><rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor"/>',
    layout: '<rect x="3" y="4" width="7" height="6" rx="1.5"/><rect x="14" y="14" width="7" height="6" rx="1.5"/><path d="M10 7h4a3 3 0 0 1 3 3v4"/>',
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
  const NODE_W = 240, NODE_H = 96, TERM_W = 76, TERM_H = 36, GAP_X = 92, GAP_Y = 112, GAP_TERM = 56;

  // ---------------------------------------------------------------- state
  const S = {
    bp: null, st: null, meta: {}, agents: [], byId: {}, slot: {},
    events: [], seqMax: 0,
    layout: null, layoutSig: "", nodeEls: {}, edgeEls: {}, termEls: {},
    view: { x: 0, y: 0, k: 1 }, userMoved: false, follow: true,
    filter: null, hideLogs: false, autoScroll: true, lastItem: null,
    modalAuto: false, lastActive: undefined, pendingAnims: [], raf: 0,
    drawerId: null, drawerTimer: 0, narrated: false, wasDisconnected: false, es: null,
    bubbleText: "", moved: {}, shift: { x: 0, y: 0 },
    scrubT: null, hist: null, playing: false, speed: 4, hoverItem: null,
    projectId: new URLSearchParams(location.search).get("project") || "", projects: [],
  };

  const q = (path) => (S.projectId ? `${path}${path.includes("?") ? "&" : "?"}project=${encodeURIComponent(S.projectId)}` : path);

  const wrap = $("#canvas-wrap"), canvas = $("#canvas"), edgesSvg = $("#edges"), nodesEl = $("#nodes");
  const feedEl = $("#feed");

  // ---------------------------------------------------------------- data
  async function loadState() {
    const res = await fetch(q("/api/state"), { cache: "no-store" });
    const data = await res.json();
    S.bp = data.blueprint;
    S.meta = data.meta || {};
    S.st = data.state;
    if (!S.projectId && S.meta.project_id) S.projectId = S.meta.project_id;
    S.seqMax = 0;
    S.hist = S.fullHist = null;
    S.scrubT = null;
    S.layoutSig = "";
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
    const es = new EventSource(q(`/api/stream?since=${S.seqMax}`));
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
    const keys = new Set(edges.map((e) => e.key));
    const routed = edges.map((e) => {
      const p = nodes[e.from], q = nodes[e.to];
      const lf = layer[e.from], lt = layer[e.to];
      const same = lf === lt;
      const isBack = back.has(e.key) || lt < lf;
      let d, labelPos;
      if (Math.abs(lt - lf) === 1 && keys.has(`${e.to}->${e.from}`)) {
        d = pathPair(p, q, lt > lf, lr); // a <-> b handoff in both directions: two parallel arrows
        labelPos = lr ? (lt > lf ? "above" : "below") : (lt > lf ? "left" : "right");
      } else if (isBack || same) {
        const idx = backCount++;
        d = lr ? pathLR(p, q, isBack, same, idx, bottom) : pathTB(p, q, isBack, same, idx, right);
      } else if (lt - lf > 1) {
        const mids = all.filter((n) => n.layer > lf && n.layer < lt);
        d = pathAround(p, q, mids, lr);
      } else {
        d = lr ? pathLR(p, q, false, false, 0, bottom) : pathTB(p, q, false, false, 0, right);
      }
      return { ...e, back: isBack, d, labelPos };
    });
    const extra = backCount ? 70 + backCount * 18 : 0;
    const w = (lr ? end : right + extra) + 40;
    const h = (lr ? bottom + extra : end) + 130;
    return { orient, nodes, edges: routed, bounds: { x: -20, y: -20, w: w + 20, h: h + 20 } };
  }

  function pathPair(a, b, forward, lr) {
    const off = 9;
    if (lr) {
      const x1 = forward ? a.x + a.w : a.x, x2 = forward ? b.x : b.x + b.w;
      const y1 = a.y + a.h / 2 + (forward ? -off : off), y2 = b.y + b.h / 2 + (forward ? -off : off);
      const dx = Math.max(40, Math.abs(x2 - x1) / 2) * (forward ? 1 : -1);
      return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
    }
    const y1 = forward ? a.y + a.h : a.y, y2 = forward ? b.y : b.y + b.h;
    const x1 = a.x + a.w / 2 + (forward ? -off : off), x2 = b.x + b.w / 2 + (forward ? -off : off);
    const dy = Math.max(40, Math.abs(y2 - y1) / 2) * (forward ? 1 : -1);
    return `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
  }

  // A forward edge that skips layers: arc over (LR) or beside (TB) any node standing in its way.
  function pathAround(a, b, mids, lr) {
    if (lr) {
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
      const dx = Math.max(40, (x2 - x1) / 4);
      const lo = Math.min(y1, y2) - 14, hi = Math.max(y1, y2) + 14;
      if (!mids.some((n) => n.y < hi && n.y + n.h > lo)) return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
      const peak = Math.min(...mids.map((n) => n.y)) - 34;
      const Y = (8 * peak - y1 - y2) / 6;
      return `M${x1},${y1} C${x1 + dx},${Y} ${x2 - dx},${Y} ${x2},${y2}`;
    }
    const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y;
    const dy = Math.max(40, (y2 - y1) / 4);
    const lo = Math.min(x1, x2) - 14, hi = Math.max(x1, x2) + 14;
    if (!mids.some((n) => n.x < hi && n.x + n.w > lo)) return `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
    const side = Math.max(...mids.map((n) => n.x + n.w)) + 34;
    const X = (8 * side - x1 - x2) / 6;
    return `M${x1},${y1} C${X},${y1 + dy} ${X},${y2 - dy} ${x2},${y2}`;
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
  // ---------------------------------------------------------------- dragging blocks
  const posKey = () => `va-pos:${S.bp.project}:${S.layout.orient}`;

  function loadPositions() {
    let saved = {};
    try { saved = JSON.parse(store.get(posKey()) || "{}"); } catch { saved = {}; }
    S.moved = {};
    for (const id in saved) {
      const n = S.layout.nodes[id];
      if (n && Number.isFinite(saved[id].x) && Number.isFinite(saved[id].y)) {
        n.x = saved[id].x;
        n.y = saved[id].y;
        S.moved[id] = saved[id];
      }
    }
    if (!Object.keys(S.moved).length) return;
    const keys = new Set(S.layout.edges.map((e) => e.key));
    for (const e of S.layout.edges) {
      if (S.moved[e.from] || S.moved[e.to]) {
        e.d = pathFree(S.layout.nodes[e.from], S.layout.nodes[e.to], keys.has(`${e.to}->${e.from}`) ? (e.from < e.to ? -9 : 9) : 0);
        e.labelPos = undefined;
      }
    }
  }

  function savePositions() {
    store.set(posKey(), JSON.stringify(S.moved));
  }

  // Route an edge between two freely placed blocks: leave and enter on the sides that face each other.
  function pathFree(a, b, offset) {
    const ax = a.x + a.w / 2, ay = a.y + a.h / 2, bx = b.x + b.w / 2, by = b.y + b.h / 2;
    const dx = bx - ax, dy = by - ay;
    if (Math.abs(dx) - (a.w + b.w) / 2 >= Math.abs(dy) - (a.h + b.h) / 2) {
      const sx = Math.sign(dx) || 1;
      const x1 = ax + (sx * a.w) / 2, x2 = bx - (sx * b.w) / 2, y1 = ay + offset, y2 = by + offset;
      const c = Math.max(30, Math.abs(x2 - x1) / 2) * sx;
      return `M${x1},${y1} C${x1 + c},${y1} ${x2 - c},${y2} ${x2},${y2}`;
    }
    const sy = Math.sign(dy) || 1;
    const y1 = ay + (sy * a.h) / 2, y2 = by - (sy * b.h) / 2, x1 = ax + offset, x2 = bx + offset;
    const c = Math.max(30, Math.abs(y2 - y1) / 2) * sy;
    return `M${x1},${y1} C${x1},${y1 + c} ${x2},${y2 - c} ${x2},${y2}`;
  }

  function placeLabel(ref) {
    try {
      const len = ref.path.getTotalLength();
      ref.mid = ref.path.getPointAtLength(len / 2);
    } catch { return; }
    const pos = ref.edge.labelPos;
    ref.label.setAttribute("x", ref.mid.x + (pos === "left" ? -10 : pos === "right" ? 10 : 0));
    ref.label.setAttribute("y", ref.mid.y + (pos === "below" ? 17 : pos === "left" || pos === "right" ? 4 : -7));
    ref.label.setAttribute("text-anchor", pos === "left" ? "end" : pos === "right" ? "start" : "middle");
  }

  function reroute(id) {
    for (const key in S.edgeEls) {
      const ref = S.edgeEls[key];
      const e = ref.edge;
      if (e.from !== id && e.to !== id) continue;
      const pair = !!S.edgeEls[`${e.to}->${e.from}`];
      e.d = pathFree(S.layout.nodes[e.from], S.layout.nodes[e.to], pair ? (e.from < e.to ? -9 : 9) : 0);
      e.labelPos = undefined;
      ref.path.setAttribute("d", e.d);
      placeLabel(ref);
    }
  }

  function makeDraggable(elm, id, onClick) {
    elm.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      const n = S.layout.nodes[id];
      const start = { x: e.clientX, y: e.clientY, nx: n.x, ny: n.y };
      let dragging = false;
      elm.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const dx = ev.clientX - start.x, dy = ev.clientY - start.y;
        if (!dragging && Math.hypot(dx, dy) < 4) return;
        if (!dragging) {
          dragging = true;
          elm.classList.add("dragging");
          document.body.classList.add("dragging-node");
        }
        n.x = start.nx + dx / S.view.k;
        n.y = start.ny + dy / S.view.k;
        elm.style.left = `${n.x}px`;
        elm.style.top = `${n.y}px`;
        S.moved[id] = { x: Math.round(n.x), y: Math.round(n.y) };
        reroute(id);
        positionBubble(viewSt());
      };
      const up = () => {
        elm.removeEventListener("pointermove", move);
        elm.removeEventListener("pointerup", up);
        elm.removeEventListener("pointercancel", up);
        elm.classList.remove("dragging");
        document.body.classList.remove("dragging-node");
        if (dragging) savePositions();
        else if (onClick) onClick();
      };
      elm.addEventListener("pointermove", move);
      elm.addEventListener("pointerup", up);
      elm.addEventListener("pointercancel", up);
    });
  }

  function resetLayout() {
    store.set(posKey(), "{}");
    S.moved = {};
    renderGraph();
    updateNodes();
    updateEdges();
    S.userMoved = false;
    fit(true);
  }

  function renderGraph() {
    S.layout = computeLayout();
    loadPositions();
    const L = S.layout;
    const shiftX = -L.bounds.x, shiftY = -L.bounds.y;
    S.shift = { x: shiftX, y: shiftY };
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
      for (const key in S.edgeEls) placeLabel(S.edgeEls[key]);
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
      makeDraggable(t, id);
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
      makeDraggable(node, a.id, () => openDrawer(a.id));
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

  // ---------------------------------------------------------------- history (time ruler)
  // The map at any moment is rebuilt from the event log, so the ruler can rewind and replay it.
  const INFORMATIVE = new Set(["thought", "tool", "output", "task", "narration", "message"]);

  function viewSt() {
    if (S.scrubT == null) return S.st;
    if (!S.hist || S.hist.T !== S.scrubT || S.hist.seq !== S.seqMax) {
      S.hist = { T: S.scrubT, seq: S.seqMax, st: buildHistory(S.scrubT) };
    }
    return S.hist.st;
  }

  const viewNow = () => (S.scrubT == null ? now() : S.scrubT);

  function activityLabel(ev) {
    if (ev.kind === "tool" && ev.data && ev.data.tool) return `Using ${ev.data.tool}`;
    return displayText(ev).replace(/\s+/g, " ").trim().slice(0, 160);
  }

  function buildHistory(T) {
    const live = S.st;
    const t0 = live.run.started || 0;
    const blank = () => ({ status: "idle", activity: "", activity_kind: "", visits: 0, tools: 0, errors: 0, events: 0 });
    const agents = {};
    for (const a of S.agents) agents[a.id] = blank();
    const path = [];
    const counts = {};
    let active = null;
    let step = -1;
    const marks = [];
    for (const ev of S.events) {
      if (ev.ts > T) break;
      if (ev.ts < t0) continue;
      if (ev.kind === "handoff" && ev.data && ev.data.to) {
        const to = ev.data.to, frm = ev.data.frm || START;
        if (active && agents[active] && agents[active].status === "active") agents[active].status = "done";
        if (path.length) path[path.length - 1].end = ev.ts;
        counts[`${frm}->${to}`] = (counts[`${frm}->${to}`] || 0) + 1;
        const st = (agents[to] = agents[to] || blank());
        st.status = "active";
        st.visits++;
        active = to;
        path.push({ agent: to, start: ev.ts, end: null, from: frm === START ? null : frm });
        if (ev.data.step != null && ev.data.step >= 0 && ev.data.step !== step) marks.push({ t: ev.ts, step: ev.data.step });
        if (ev.data.step != null) step = ev.data.step;
        continue;
      }
      const st = ev.agent && agents[ev.agent];
      if (!st) continue;
      st.events++;
      if (ev.kind === "tool") st.tools++;
      if (ev.kind === "error") st.errors++;
      if (INFORMATIVE.has(ev.kind) || (!st.activity && ev.kind === "log")) {
        st.activity = activityLabel(ev);
        st.activity_kind = ev.kind;
      }
    }
    const run = { ...live.run };
    if (live.run.ended && T >= live.run.ended) {
      if (active) {
        agents[active].status = live.run.status === "failed" ? "error" : "done";
        counts[`${active}->${END}`] = (counts[`${active}->${END}`] || 0) + 1;
        if (path.length) path[path.length - 1].end = live.run.ended;
        active = null;
      }
    } else {
      Object.assign(run, { status: "running", ended: null, exit_code: null });
    }
    return {
      active, agents, path, path_total: path.length, edge_counts: counts, step_index: step, run, marks,
      dynamic_edges: live.dynamic_edges, extra_agents: live.extra_agents, precise: live.precise, seq: live.seq,
    };
  }

  // ---------------------------------------------------------------- hover: feed -> map
  function setHover(item) {
    if (item === S.hoverItem) return;
    clearHover();
    if (!item) return;
    S.hoverItem = item;
    const { agent, frm, to, ts } = item.dataset;
    const ids = frm || to ? [frm, to].filter(Boolean) : agent ? [agent] : [];
    for (const id of ids) {
      const elm = S.nodeEls[id] || S.termEls[id];
      if (!elm) continue;
      elm.classList.add("hl");
      elm.classList.remove("bounce");
      void elm.offsetWidth; // restart the bounce
      elm.classList.add("bounce");
    }
    if (frm && to) {
      const ref = S.edgeEls[`${frm}->${to}`];
      if (ref) {
        ref.path.classList.add("hover-hl");
        animateHandoff({ data: { frm, to } });
      }
    }
    if (item.classList.contains("tl-seg")) item.classList.add("hl");
    else if (ts) {
      const seg = findSegment(+ts);
      if (seg) seg.classList.add("hl");
    }
  }

  function clearHover() {
    S.hoverItem = null;
    document.querySelectorAll(".node.hl, .terminal-node.hl, .tl-seg.hl").forEach((e) => e.classList.remove("hl"));
    document.querySelectorAll(".edge.hover-hl").forEach((e) => e.classList.remove("hover-hl"));
  }

  function bindHover(container) {
    container.addEventListener("mouseover", (e) => setHover(e.target.closest(".fi")));
    container.addEventListener("mouseleave", () => clearHover());
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
    renderTimeline();
    setRunStatus();
    renderEmptyHint();
    renderStartCard();
    const vst = viewSt();
    const active = vst.active;
    if (active !== S.lastActive) {
      S.lastActive = active;
      const last = vst.path[vst.path.length - 1];
      if (S.playing && active && last) animateHandoff({ data: { frm: last.from || START, to: active } });
      if (active && S.follow) ensureVisible(active);
      document.title = active ? `● ${nameOf(active)} · VisualAgents` : `VisualAgents · ${S.bp.project}`;
    }
    while (S.pendingAnims.length) {
      const ev = S.pendingAnims.shift();
      if (S.scrubT == null) animateHandoff(ev);
    }
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
    const st = viewSt();
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

    positionBubble(st);
  }

  // thought bubble under the active agent
  function positionBubble(st) {
    const b = S.bubble;
    if (!b) return;
    const act = st.active && S.layout && S.layout.nodes[st.active];
    if (!act) {
      b.hidden = true;
      return;
    }
    const s = st.agents[st.active] || {};
    const text = s.activity || "Working…";
    const kind = s.activity_kind || "log";
    b.hidden = false;
    b.classList.toggle("instant", document.body.classList.contains("dragging-node"));
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
    const st = viewSt();
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
    const st = viewSt();
    const at = S.scrubT != null && st.run.started ? `At ${fmtClock(S.scrubT - st.run.started)}` : "Working now";
    const steps = S.bp.steps || [];
    const stepTxt = st.step_index >= 0 && steps[st.step_index] ? `Step ${st.step_index + 1} of ${steps.length} · ${steps[st.step_index].title}` : "";
    const id = st.active;
    if (id && S.byId[id]) {
      const a = S.byId[id];
      const s = st.agents[id] || {};
      const last = st.path[st.path.length - 1];
      const since = last ? fmtDur(viewNow() - last.start) : "";
      box.style.setProperty("--agent", colorVar(id));
      box.innerHTML = `
        <div class="now-eyebrow"><span>${at}</span><span>${esc(stepTxt)}</span></div>
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
        <div class="now-stats"><span>agents used <b>${Object.values(st.agents).filter((x) => x.visits).length}</b></span><span>handoffs <b>${Math.max(0, (st.path_total || 1) - 1)}</b></span><span>final agent <b>${esc(lastAgent)}</b></span></div>
        ${S.scrubT == null && launchInfo() && launchInfo().command ? `<div class="run-again">${startControls("Run again")}</div>` : ""}`;
      bindStartControls(box);
      return;
    }
    const info = launchInfo();
    const waiting = run.status === "running" ? "Output is flowing - waiting for the first agent to speak up…"
      : S.starting ? "Starting - give it its task in your terminal."
      : info && info.command ? "Not running yet - press ▶ Start on the map, then give it a task."
      : S.meta.mode === "hook" ? "Waiting for it to run - start it the way you usually do."
      : "Waiting for the run to start…";
    box.innerHTML = `<div class="now-eyebrow"><span>${at}</span><span>${esc(stepTxt)}</span></div><div class="now-role" style="margin-top:8px">${esc(waiting)}</div>`;
  }

  function renderStepper() {
    const nav = $("#stepper");
    const steps = S.bp.steps || [];
    const st = viewSt();
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
    if (run.status === "running") renderTimeline();
  }

  // ---------------------------------------------------------------- making it live
  const launchInfo = () => S.meta && S.meta.launch;

  function startControls(label) {
    const info = launchInfo();
    return `
      ${info.takes_task ? `<input class="task" type="text" placeholder="What should it do?" aria-label="Task for the agent">` : ""}
      <div class="start-row">
        <button class="btn primary start-btn">${icon("play")}${label}</button>
        <span class="cmd"><code>${esc(info.command)}</code></span>
      </div>`;
  }

  function bindStartControls(root) {
    const btn = $(".start-btn", root);
    if (!btn) return;
    const input = $(".task", root);
    const go = async () => {
      btn.disabled = true;
      const task = input ? input.value.trim() : "";
      let out = { ok: false, message: "Couldn't reach VisualAgents." };
      try {
        out = await (await fetch(q("/api/start"), { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task }) })).json();
      } catch { /* shown below */ }
      if (out.launch) S.meta.launch = out.launch;
      S.startMessage = out.message;
      S.starting = !!out.ok;
      btn.disabled = false;
      renderStartCard();
      if (!out.ok && input) input.focus();
    };
    btn.onclick = go;
    if (input) input.onkeydown = (e) => { if (e.key === "Enter") go(); };
  }

  function renderStartCard() {
    const card = $("#start-card");
    const st = S.st;
    const running = st.run.status === "running";
    if (running) S.starting = false;
    const show = !running && !st.path_total && st.run.status === "idle" && S.scrubT == null
      && !["demo", "replay", "daemon-empty"].includes(S.meta.mode);
    if (!show) {
      card.hidden = true;
      return;
    }
    const name = esc(S.bp.project);
    const info = launchInfo();
    let html;
    if (S.starting) {
      html = `<h3><span class="dot"></span>Starting ${name}…</h3>
        <p>${esc(S.startMessage || "")} The map lights up as soon as its agents start working.</p>`;
    } else if (S.meta.mode === "hook") {
      html = `<h3><span class="dot"></span>Waiting for ${name}</h3>
        <p>Start it the way you usually do - your IDE, a script, a notebook. It appears here as soon as it calls an LLM.</p>`;
    } else if (S.meta.mode === "pipe" || S.meta.mode === "watch") {
      html = `<h3><span class="dot"></span>Waiting for output…</h3><p>The map comes alive as the agent's output arrives.</p>`;
    } else if (info && info.command) {
      html = `<h3><span class="dot"></span>Not running yet</h3>
        <p>The map comes alive while ${name} works on something. Start it, then give it a task the way it expects -
        answer its questions in your terminal, or use its own web page.</p>
        ${startControls("Start")}
        ${S.startMessage ? `<p class="small">${esc(S.startMessage)}</p>` : ""}`;
    } else if (info) {
      html = `<h3><span class="dot"></span>Not running yet</h3>
        <p>VisualAgents doesn't know how to start ${name} yet. Tell it once in your terminal:</p>
        <p><code>${esc(info.hint)}</code></p>`;
    } else {
      card.hidden = true;
      return;
    }
    card.classList.toggle("waiting", !!S.starting || S.meta.mode === "hook");
    if (card.dataset.html !== html) {
      card.dataset.html = html;
      card.innerHTML = html;
      bindStartControls(card);
    }
    card.hidden = false;
  }

  function renderEmptyHint() {
    const box = $("#empty-hint");
    if (!S.agents.length && S.meta.mode !== "daemon-empty") {
      box.hidden = false;
      box.innerHTML = `<b>No agents found yet.</b><br>Try an LLM scan for a deeper read of the code:<br><br><code>visualagents scan --llm</code>`;
    } else if (S.meta.mode === "daemon-empty") {
      box.hidden = false;
      box.innerHTML = `<b>No hooked projects yet.</b><br>Hook a folder once, then start your agent the way you always do:<br><br><code>visualagents hook ~/my-agent</code>`;
    } else {
      box.hidden = true;
    }
  }

  // ---------------------------------------------------------------- projects (background dashboard)
  async function refreshProjects() {
    let data;
    try {
      data = await (await fetch("/api/projects", { cache: "no-store" })).json();
    } catch { return; }
    S.projects = data.projects || [];
    const sel = $("#project-switch");
    sel.hidden = !(data.daemon && S.projects.length > 1);
    if (sel.hidden) return;
    const dot = { running: "●", finished: "✓", failed: "✕" };
    sel.innerHTML = S.projects
      .map((p) => `<option value="${esc(p.id)}">${dot[p.status] ? dot[p.status] + " " : ""}${esc(p.name)}</option>`)
      .join("");
    sel.value = S.projectId || (S.meta.project_id || "");
  }

  async function switchProject(id) {
    if (!id || id === S.projectId) return;
    S.projectId = id;
    const url = new URL(location.href);
    url.searchParams.set("project", id);
    history.replaceState(null, "", url);
    if (S.es) S.es.close();
    S.slot = {};
    S.filter = null;
    S.lastActive = undefined; // refresh the tab title for the new project
    closeDrawer();
    await loadState();
    connect();
  }

  // ---------------------------------------------------------------- feed
  function addEvent(ev, live) {
    S.events.push(ev);
    if (S.events.length > 20000) S.events.splice(0, 5000);
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
    item.dataset.ts = ev.ts;
    if (ev.agent) item.dataset.agent = ev.agent;
    if (ev.kind === "handoff" && ev.data) {
      item.dataset.frm = ev.data.frm || "";
      item.dataset.to = ev.data.to || ev.agent || "";
    }
    if (S.scrubT != null && ev.ts > S.scrubT) item.classList.add("future");
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
    if (!S.autoScroll || S.stickRaf || S.scrubT != null) return;
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


  // ---------------------------------------------------------------- time ruler
  const track = $("#tl-track");

  function tlRange() {
    const run = S.st && S.st.run;
    if (!run || !run.started) return null;
    return { t0: run.started, t1: Math.max(run.started + 1, run.ended || now()) };
  }

  function fullHistory() {
    const key = `${S.seqMax}|${S.st.run.ended}|${S.st.run.started}`;
    if (!S.fullHist || S.fullHist.key !== key) S.fullHist = { key, st: buildHistory(Infinity) };
    return S.fullHist.st;
  }

  function renderTimeline() {
    const r = tlRange();
    document.body.classList.toggle("scrubbing", S.scrubT != null);
    $("#tl-empty").hidden = !!r;
    $("#tl-head").hidden = !r;
    $("#tl-fade").hidden = !r || S.scrubT == null;
    const bar = $("#tl-bar"), ticks = $("#tl-ticks"), flags = $("#tl-steps");
    if (!r) {
      bar.textContent = ticks.textContent = flags.textContent = "";
      S.tlSig = "";
      return;
    }
    const W = track.clientWidth || 600;
    const running = S.st.run.status === "running";
    const sig = `${S.seqMax}|${Math.floor(r.t1 * 2)}|${W}|${S.st.run.status}`;
    if (sig !== S.tlSig) {
      S.tlSig = sig;
      const span = r.t1 - r.t0;
      const pct = (t) => `${clamp((t - r.t0) / span, 0, 1) * 100}%`;
      const full = fullHistory();
      bar.textContent = "";
      full.path.forEach((p, i) => {
        const end = p.end || r.t1;
        const live = !p.end && running && i === full.path.length - 1;
        const seg = el("div", `tl-seg${live ? " running" : ""}`);
        seg.style.left = pct(p.start);
        seg.style.width = `${Math.max(0.3, ((end - p.start) / span) * 100)}%`;
        seg.style.setProperty("--agent", colorVar(p.agent));
        seg.dataset.agent = p.agent;
        seg.dataset.start = p.start;
        seg.dataset.end = end;
        if (((end - p.start) / span) * W > 58) seg.textContent = nameOf(p.agent);
        seg.title = `${nameOf(p.agent)} · ${fmtClock(p.start - r.t0)}–${fmtClock(end - r.t0)} (${fmtDur(end - p.start)})`;
        bar.appendChild(seg);
      });
      const hov = S.hoverItem;
      if (hov && hov.isConnected && hov.dataset.ts) {
        const seg = findSegment(+hov.dataset.ts);
        if (seg) seg.classList.add("hl");
      }
      ticks.textContent = "";
      const target = span / Math.max(1, W / 90);
      const every = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200].find((x) => x >= target) || 7200;
      for (let t = 0; t <= span + 1e-6; t += every) {
        const tk = el("span", "tl-tick", fmtClock(t));
        tk.style.left = `${(t / span) * 100}%`;
        ticks.appendChild(tk);
      }
      flags.textContent = "";
      const steps = S.bp.steps || [];
      let lastLabelX = -1e9;
      for (const m of full.marks) {
        const x = ((m.t - r.t0) / span) * W;
        const s = steps[m.step];
        const f = el("span", "tl-flag");
        f.style.left = pct(m.t);
        const room = x - lastLabelX > 120 && W - x > 60;
        f.innerHTML = `<b>${m.step + 1}</b>${room && s ? esc(s.title) : ""}`;
        if (room) lastLabelX = x;
        f.title = s ? `Step ${m.step + 1}: ${s.title}` : `Step ${m.step + 1}`;
        flags.appendChild(f);
      }
    }
    positionPlayhead(r);
  }

  function positionPlayhead(r = tlRange()) {
    if (!r) return;
    const T = S.scrubT == null ? r.t1 : S.scrubT;
    const f = clamp((T - r.t0) / (r.t1 - r.t0), 0, 1);
    const head = $("#tl-head");
    head.style.left = `${f * 100}%`;
    $("#tl-fade").style.left = `${f * 100}%`;
    const st = viewSt();
    const who = st.active ? nameOf(st.active) : "";
    const when = S.scrubT == null ? (S.st.run.status === "running" ? "Live" : "End") : fmtClock(T - r.t0);
    const tip = $("#tl-tip");
    tip.textContent = who ? `${when} · ${who}` : when;
    tip.style.transform = f > 0.88 ? "translateX(-100%)" : f < 0.12 ? "translateX(0)" : "translateX(-50%)";
    track.setAttribute("aria-valuemax", String(Math.round(r.t1 - r.t0)));
    track.setAttribute("aria-valuenow", String(Math.round(T - r.t0)));
    track.setAttribute("aria-valuetext", tip.textContent);
    const liveBtn = $("#tl-live");
    liveBtn.classList.toggle("on", S.scrubT == null);
    liveBtn.textContent = S.scrubT == null ? "Live" : "Go live";
    $("#tl-play").innerHTML = icon(S.playing ? "pause" : "play");
    $("#tl-play").title = S.playing ? "Pause (Space)" : S.scrubT == null ? "Replay this run (Space)" : "Play from here (Space)";
  }

  function findSegment(ts) {
    let hit = null;
    for (const seg of $("#tl-bar").children) if (+seg.dataset.start <= ts) hit = seg;
    return hit;
  }

  function seek(t) {
    const r = tlRange();
    if (!r) return;
    S.seekTo = clamp(t, r.t0, r.t1);
    if (S.seekRaf) return;
    S.seekRaf = requestAnimationFrame(() => {
      S.seekRaf = 0;
      S.scrubT = S.seekTo;
      renderState();
      syncFeedToTime();
    });
  }

  function goLive() {
    S.scrubT = null;
    S.playing = false;
    S.hist = null;
    renderState();
    syncFeedToTime();
    S.autoScroll = true;
    $("#jump").hidden = true;
    stickToEnd();
  }

  function togglePlay() {
    const r = tlRange();
    if (!r) return;
    if (S.playing) {
      S.playing = false;
      positionPlayhead(r);
      return;
    }
    if (S.scrubT == null || S.scrubT >= r.t1 - 0.1) S.scrubT = r.t0; // replay from the start
    S.playing = true;
    S.lastFrame = performance.now();
    requestAnimationFrame(playStep);
    positionPlayhead(r);
  }

  function playStep(ts) {
    if (!S.playing) return;
    const r = tlRange();
    const dt = Math.min(0.25, (ts - S.lastFrame) / 1000);
    S.lastFrame = ts;
    const t = S.scrubT + dt * S.speed;
    if (t >= r.t1 - 0.05) {
      goLive();
      return;
    }
    S.scrubT = t;
    if (ts - (S.lastPlayRender || 0) > 60) {
      S.lastPlayRender = ts;
      renderState();
      syncFeedToTime();
    } else positionPlayhead(r);
    requestAnimationFrame(playStep);
  }

  function syncFeedToTime() {
    const T = S.scrubT;
    let lastPast = null;
    for (const it of feedEl.children) {
      const fut = T != null && +it.dataset.ts > T;
      it.classList.toggle("future", fut);
      it.classList.remove("at-time");
      if (!fut) lastPast = it;
    }
    if (T != null && lastPast) {
      lastPast.classList.add("at-time");
      S.programmaticScroll = true;
      feedEl.scrollTop = lastPast.offsetTop - feedEl.offsetTop - feedEl.clientHeight * 0.6;
    }
  }

  function bindTimeline() {
    let scrubbing = false;
    const xToTime = (cx) => {
      const r = tlRange();
      const b = track.getBoundingClientRect();
      return r.t0 + clamp((cx - b.left) / b.width, 0, 1) * (r.t1 - r.t0);
    };
    track.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || !tlRange()) return;
      scrubbing = true;
      S.playing = false;
      track.setPointerCapture(e.pointerId);
      track.classList.add("grabbing");
      seek(xToTime(e.clientX));
    });
    track.addEventListener("pointermove", (e) => {
      const r = tlRange();
      if (!r) return;
      if (scrubbing) {
        seek(xToTime(e.clientX));
        return;
      }
      const hover = $("#tl-hover");
      const b = track.getBoundingClientRect();
      hover.hidden = false;
      hover.style.left = `${clamp((e.clientX - b.left) / b.width, 0, 1) * 100}%`;
      hover.dataset.label = fmtClock(xToTime(e.clientX) - r.t0);
    });
    const end = (e) => {
      if (!scrubbing) return;
      scrubbing = false;
      track.classList.remove("grabbing");
      const r = tlRange();
      if (r && xToTime(e.clientX) >= r.t1 - (r.t1 - r.t0) * 0.005) goLive();
    };
    track.addEventListener("pointerup", end);
    track.addEventListener("pointercancel", end);
    track.addEventListener("pointerleave", () => { $("#tl-hover").hidden = true; });
    track.addEventListener("keydown", (e) => {
      const r = tlRange();
      if (!r) return;
      const T = S.scrubT == null ? r.t1 : S.scrubT;
      const stepS = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowLeft") seek(T - stepS);
      else if (e.key === "ArrowRight") (T + stepS >= r.t1 ? goLive() : seek(T + stepS));
      else if (e.key === "Home") seek(r.t0);
      else if (e.key === "End") goLive();
      else return;
      e.preventDefault();
    });
    const bar = $("#tl-bar");
    bar.addEventListener("mouseover", (e) => setHover(e.target.closest(".tl-seg")));
    bar.addEventListener("mouseleave", () => clearHover());
    $("#tl-live").onclick = goLive;
    $("#tl-play").onclick = togglePlay;
    const speeds = [1, 2, 4, 8, 16];
    $("#tl-play").innerHTML = icon("play");
    const speedBtn = $("#tl-speed");
    speedBtn.textContent = `${S.speed}×`;
    speedBtn.onclick = () => {
      S.speed = speeds[(speeds.indexOf(S.speed) + 1) % speeds.length];
      speedBtn.textContent = `${S.speed}×`;
    };
    new ResizeObserver(() => { S.tlSig = ""; if (S.st) renderTimeline(); }).observe(track);
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

  function contentBox() {
    const ns = Object.values(S.layout.nodes);
    let x1 = Math.min(...ns.map((n) => n.x)), y1 = Math.min(...ns.map((n) => n.y));
    let x2 = Math.max(...ns.map((n) => n.x + n.w)), y2 = Math.max(...ns.map((n) => n.y + n.h)) + 110; // bubble room
    try {
      const b = S.edgeGroup.getBBox();
      if (b.width || b.height) {
        x1 = Math.min(x1, b.x); y1 = Math.min(y1, b.y - 14);
        x2 = Math.max(x2, b.x + b.width); y2 = Math.max(y2, b.y + b.height + 10);
      }
    } catch { /* not rendered yet */ }
    return { x: x1 - 20, y: y1 - 20, w: x2 - x1 + 40, h: y2 - y1 + 40 };
  }

  function fit(animate) {
    if (!S.layout) return;
    const b = contentBox();
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (!W || !H) return;
    const pad = 32;
    const k = clamp(Math.min((W - pad * 2) / b.w, (H - pad * 2) / b.h), 0.3, 1.2);
    S.view = {
      k,
      x: (W - b.w * k) / 2 - (b.x + S.shift.x) * k,
      y: Math.max(pad / 2, (H - b.h * k) / 2) - (b.y + S.shift.y) * k,
    };
    applyView(animate);
  }

  function ensureVisible(id) {
    const n = S.layout && S.layout.nodes[id];
    if (!n) return;
    const v = S.view, W = wrap.clientWidth, H = wrap.clientHeight;
    const ox = S.shift.x, oy = S.shift.y;
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
      if (e.target.closest(".node, .terminal-node, button, .legend, .empty-hint")) return;
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
    $("#relayout").onclick = resetLayout;
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
    const chips = (ids) => ids.map((id) => `<span class="chip" style="--agent:${colorVar(id)}"><i></i>${esc(nameOf(id))}</span>`).join("");
    const facts = [
      bp.framework && ["Framework", esc(bp.framework)],
      ["Size", `${S.agents.length} agent${S.agents.length === 1 ? "" : "s"} · ${bp.edges.length} handoff${bp.edges.length === 1 ? "" : "s"}`],
      bp.entry.length && ["Starts at", chips(bp.entry)],
      bp.exits.length && ["Answer from", chips(bp.exits)],
    ].filter(Boolean).map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
    const steps = (bp.steps || []).map((s) => `
      <li>
        <span class="bp-num">${s.order}</span>
        <div class="bp-step-body">
          <h4>${esc(s.title)}</h4>
          ${s.agents.length ? `<div class="chips">${chips(s.agents)}</div>` : ""}
          ${s.description ? `<p>${esc(s.description)}</p>` : ""}
        </div>
      </li>`).join("");
    const agents = S.agents.map((a) => `
      <li style="--agent:${colorVar(a.id)}">
        <div class="bp-agent-head"><span class="node-dot"></span><b>${esc(a.name)}</b>
          ${a.role ? `<span class="role">${esc(a.role)}</span>` : ""}${a.dynamic ? '<span class="tag">runtime</span>' : ""}</div>
        ${a.description ? `<p>${esc(a.description)}</p>` : ""}
        ${a.tools && a.tools.length ? `<div class="tools">${a.tools.map((t) => `<span class="tool-chip">${esc(t)}</span>`).join("")}</div>` : ""}
        ${a.file ? `<div class="src">${esc(a.file)}${a.line ? ":" + a.line : ""}</div>` : ""}
      </li>`).join("");
    const edges = bp.edges.map((e) => `
      <li>${chips([e.from])}<span class="arrow-txt">${e.conditional ? "⇢" : "→"}</span>${chips([e.to])}
        ${e.label ? `<span class="lbl">${esc(e.label)}</span>` : ""}${e.conditional ? '<span class="tag">if needed</span>' : ""}</li>`).join("");
    $("#bp-body").innerHTML = `
      ${bp.summary ? `<p class="bp-summary">${esc(bp.summary)}</p>` : ""}
      <dl class="bp-facts">${facts}</dl>
      ${steps ? `<h3 class="bp-section-title">How a request flows</h3><ol class="bp-timeline">${steps}</ol>` : ""}
      <h3 class="bp-section-title">Agents</h3>
      ${agents ? `<ul class="bp-agent-list">${agents}</ul>` : '<p class="muted">No agents detected.</p>'}
      ${edges ? `<h3 class="bp-section-title">Handoffs</h3><ul class="bp-edge-list">${edges}</ul>` : ""}`;
    const gen = bp.generated_by ? `Mapped by ${bp.generated_by}` : "";
    $("#bp-meta").textContent = [gen, bp.generated_at ? new Date(bp.generated_at).toLocaleString() : ""].filter(Boolean).join(" · ");
  }

  function openModal(auto) {
    S.modalAuto = !!auto;
    const live = S.st && (S.st.run.status === "running" || S.st.path_total);
    $(".modal-foot .btn.primary").textContent = live ? "Watch it live" : "Show the map";
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
    if (mode === "blueprint" || mode === "demo" || mode === "setup" || !store.get(key)) {
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
    $("#reset-view").onclick = () => fetch(q("/api/reset"), { method: "POST" });
    feedEl.addEventListener("scroll", () => {
      const atEnd = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 40;
      if (S.programmaticScroll) {
        S.programmaticScroll = false;
        if (!atEnd && S.scrubT == null && S.autoScroll) feedEl.scrollTop = feedEl.scrollHeight;
        return;
      }
      S.autoScroll = atEnd;
      $("#jump").hidden = atEnd || S.scrubT != null;
    });
    $("#jump").onclick = () => { feedEl.scrollTop = feedEl.scrollHeight; S.autoScroll = true; $("#jump").hidden = true; };
    document.addEventListener("keydown", (e) => {
      if (e.target.closest("input, textarea")) return;
      if (e.key === " " && !e.target.closest("button, [role=slider]")) { e.preventDefault(); togglePlay(); return; }
      if (e.key === "Escape") { closeModal(); closeDrawer(); }
      else if (e.key === "l" || e.key === "L") goLive();
      else if (e.key === "b" || e.key === "B") $("#btn-blueprint").click();
      else if (e.key === "f" || e.key === "F") { S.userMoved = false; fit(true); }
    });
    bindViewport();
    bindTimeline();
    $("#project-switch").onchange = (e) => switchProject(e.target.value);
    setInterval(refreshProjects, 5000);
    bindHover(feedEl);
    bindHover($("#drawer-body"));
    setInterval(tickTimer, 1000);
  }

  bindUI();
  loadState()
    .then(() => { refreshProjects(); connect(); })
    .catch((err) => {
      console.error(err);
      setRunStatus(true);
      setTimeout(() => loadState().then(connect), 2000);
    });
})();
