/* DataAnalyst AI - dashboard logic (vanilla JS). */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const sessionId = localStorage.getItem("daa_session") || (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
  localStorage.setItem("daa_session", sessionId);

  const EXAMPLES = [
    "What was total revenue in August?",
    "Show monthly revenue trend",
    "Which product had the largest revenue decline and what explanation was given in management reports?",
    "What does the report say about inventory?",
  ];

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const rich = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
      throw new Error(detail);
    }
    return body;
  }

  /* ---------------------------------------------------------------- status */
  async function refreshStatus() {
    const el = $("status"), text = $("statusText");
    try {
      const s = await api("/api/model-status");
      el.dataset.state = s.reachable ? "online" : "offline";
      text.textContent = s.reachable ? `Local AI · ${s.model}` : "LM Studio offline";
      el.title = s.message;
    } catch (e) {
      el.dataset.state = "offline";
      text.textContent = "Backend unreachable";
      el.title = e.message;
    }
  }

  /* ------------------------------------------------------------- documents */
  function renderDocs(docs) {
    const list = $("docList");
    list.innerHTML = "";
    $("docEmpty").hidden = docs.length > 0;
    docs.forEach((d) => {
      const li = document.createElement("li");
      li.className = `doc-item ${d.kind}`;
      let meta = d.kind === "table" ? `${d.rows ?? "?"} rows` + (d.tables > 1 ? `, ${d.tables} sheets` : "")
        : `${d.pages ?? "?"} page${d.pages === 1 ? "" : "s"}, ${d.chunks ?? "?"} chunks`;
      li.innerHTML = `<span class="name">${esc(d.name)}</span><span class="meta">${esc(meta)}</span>` +
        `<button type="button" title="Remove ${esc(d.name)}" aria-label="Remove ${esc(d.name)}">×</button>`;
      li.querySelector("button").addEventListener("click", async () => {
        if (!confirm(`Remove ${d.name}?`)) return;
        try { await api(`/api/documents/${d.id}`, { method: "DELETE" }); } catch (e) { alert(e.message); }
        loadDocs();
      });
      list.appendChild(li);
    });
  }

  async function loadDocs() {
    try { renderDocs((await api("/api/documents")).documents); }
    catch (e) { $("docEmpty").hidden = false; $("docEmpty").textContent = "Could not load files: " + e.message; }
  }

  function uploadOne(file) {
    const li = document.createElement("li");
    li.innerHTML = `<div>${esc(file.name)}</div><div class="bar"><i></i></div><div class="msg muted small">Uploading…</div>`;
    $("uploadStatus").prepend(li);
    const bar = li.querySelector("i"), msg = li.querySelector(".msg");
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
        if (e.loaded === e.total) msg.textContent = "Processing…";
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) {
          bar.style.width = "100%";
          msg.className = "msg ok small";
          msg.textContent = body.message || "Done";
        } else {
          msg.className = "msg err small";
          msg.textContent = body.detail || `Upload failed (${xhr.status})`;
        }
        setTimeout(() => li.remove(), 8000);
        resolve();
      };
      xhr.onerror = () => { msg.className = "msg err small"; msg.textContent = "Network error."; resolve(); };
      const fd = new FormData();
      fd.append("file", file);
      xhr.send(fd);
    });
  }

  async function uploadFiles(files) {
    for (const f of files) await uploadOne(f);
    loadDocs();
  }

  /* ----------------------------------------------------------------- charts */
  function drawChart(el, spec) {
    if (!spec || !window.Plotly) {
      if (spec) el.innerHTML = '<p class="muted small">Chart library could not be loaded (internet needed for Plotly.js).</p>';
      return;
    }
    Plotly.newPlot(el, spec.data, spec.layout, { responsive: true, displaylogo: false });
  }

  /* ---------------------------------------------------------------- results */
  function renderAnswer(text) {
    const m = text.match(/^CALCULATED:\n([\s\S]*?)\n\nDOCUMENT EVIDENCE:\n([\s\S]*)$/);
    if (m) {
      return `<div class="ledger"><section class="calc"><h4>Calculated</h4>${rich(m[1])}</section>` +
             `<section class="doc"><h4>Document evidence</h4>${rich(m[2])}</section></div>`;
    }
    return `<div class="answer">${rich(text)}</div>`;
  }

  function renderTable(rows) {
    if (!Array.isArray(rows) || !rows.length) return "";
    const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const cell = (v) => typeof v === "number"
      ? `<td class="num">${Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>`
      : `<td>${esc(v ?? "")}</td>`;
    return `<div class="table-wrap"><table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      rows.slice(0, 50).map((r) => `<tr>${cols.map((c) => cell(r[c])).join("")}</tr>`).join("") + "</tbody></table></div>";
  }

  function renderResult(card, r) {
    const route = r.route || {};
    const rewritten = r.rewritten_question && r.rewritten_question !== r.question
      ? `<div class="q-sub">Interpreted as: ${esc(r.rewritten_question)}</div>` : "";
    let html = `<div class="q">${esc(r.question)}</div>${rewritten}` +
      `<div class="q-sub debug-only"><span class="badge ${esc(route.route)}">${esc(route.route)}</span>${esc(route.method)} — ${esc(route.reason)} · ${r.latency_ms} ms</div>`;
    html += renderAnswer(r.answer || "");
    (r.warnings || []).forEach((w) => { html += `<div class="warn">${esc(w)}</div>`; });
    if (r.sources && r.sources.length) {
      html += '<div class="sources"><span class="label">Sources</span>' + r.sources.map((s) =>
        s.type === "data" ? `<span class="chip data">${esc(s.file)}</span>`
          : `<span class="chip document">${esc(s.file)} — Page ${esc(s.page)}</span>`).join("") + "</div>";
    }
    if (r.chart) html += '<div class="chart"></div>';
    if (r.calculation || (r.data && r.data.length)) {
      html += `<details><summary>Calculation and data</summary>` +
        (r.calculation ? `<div class="calc-text">${esc(r.calculation)}</div>` : "") + renderTable(r.data) + "</details>";
    }
    if (r.evidence && r.evidence.length) {
      html += `<details><summary>Retrieved passages (${r.evidence.length})</summary>` + r.evidence.map((e) =>
        `<div class="passage"><b>${esc(e.file)} — Page ${esc(e.page)}${e.score != null ? ` · score ${esc(e.score)}` : ""}</b>${esc(e.text)}</div>`).join("") + "</details>";
    }
    html += `<details class="debug-only"><summary>Debug</summary><pre class="debug">${esc(JSON.stringify({ route, debug: r.debug }, null, 2))}</pre></details>`;
    card.innerHTML = html;
    if (r.chart) drawChart(card.querySelector(".chart"), r.chart);
  }

  async function ask(question) {
    const btn = $("askBtn");
    btn.disabled = true;
    $("empty").hidden = true;
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<div class="q">${esc(question)}</div><p class="loading">Analyzing</p>`;
    $("results").prepend(card);
    try {
      const r = await api("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
      });
      renderResult(card, r);
    } catch (e) {
      card.innerHTML = `<div class="q">${esc(question)}</div><div class="error">${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  /* ------------------------------------------------------------------- init */
  function init() {
    $("examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q.length > 60 ? q.slice(0, 57) + "…" : q)}</button>`).join("");
    $("examples").querySelectorAll("button").forEach((b, i) => b.addEventListener("click", () => { $("question").value = EXAMPLES[i]; $("question").focus(); }));

    $("askForm").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = $("question").value.trim();
      if (!q) return;
      $("question").value = "";
      ask(q);
    });
    $("question").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("askForm").requestSubmit(); }
    });

    const dz = $("dropzone"), input = $("fileInput");
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

    $("clearChat").addEventListener("click", async () => {
      await api("/api/conversation/clear", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "x", session_id: sessionId }) }).catch(() => {});
      $("results").innerHTML = "";
      $("empty").hidden = false;
    });

    const debug = $("debugToggle");
    const stored = localStorage.getItem("daa_debug");
    const apply = (on) => document.body.classList.toggle("debug", on);
    if (stored !== null) { debug.checked = stored === "1"; apply(debug.checked); }
    else api("/api/health").then((h) => { debug.checked = !!h.debug; apply(debug.checked); }).catch(() => {});
    debug.addEventListener("change", () => { localStorage.setItem("daa_debug", debug.checked ? "1" : "0"); apply(debug.checked); });

    loadDocs();
    refreshStatus();
    setInterval(refreshStatus, 20000);
  }

  init();
})();
