(function () {
  const feed = document.getElementById("feed");
  const sel = document.getElementById("case-filter");
  const total = document.getElementById("total");
  const dot = document.getElementById("live-dot");
  let seen = new Set();
  let first = true;
  let lastSig = null;

  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtReq = r => {
    if (!r || typeof r !== "object") return esc(r);
    return Object.entries(r).map(([k, v]) => {
      const val = typeof v === "string" ? v : JSON.stringify(v);
      return k === "prompt" ? `${k}:\n${val}` : `${k}: ${val}`;
    }).map(esc).join("\n");
  };
  const fmtUsage = u => {
    if (!u) return "";
    if (u.input_tokens !== undefined) return `${u.input_tokens.toLocaleString()} in · ${u.output_tokens.toLocaleString()} out tokens`;
    if (u.prompt_tokens !== undefined) return `${u.prompt_tokens.toLocaleString()} in · ${(u.completion_tokens || 0).toLocaleString()} out tokens`;
    if (u.audioDurationMs !== undefined) return `${(u.audioDurationMs / 1000).toFixed(1)} s audio · ${u.turns} turns`;
    if (u.frames !== undefined) return `${u.frames} frames`;
    return "";
  };
  const key = c => `${c.case}|${c.ts}|${c.step}|${c.mode}`;

  function card(c, fresh) {
    const el = document.createElement("article");
    el.className = `call${fresh ? " fresh" : ""}`;
    const ts = (c.ts || "").replace("T", " ").replace("Z", " UTC").slice(11);
    const usd = c.usd ? `$${Number(c.usd).toFixed(4)}` : (c.model && c.model.startsWith("meta-models/") ? "Baseten credits" : "$0");
    el.innerHTML = `
      <div class="call-top">
        <span class="ts">${esc(ts)}</span>
        <span class="badge ${c.mode === "replay" ? "replay" : "live"}">${c.mode === "replay" ? "REPLAY" : "LIVE"}</span>
        <span class="model">${esc(c.model_name || c.model)}</span>
        <span class="ep">${esc(c.endpoint)}</span>
        <span class="lat">${c.latency_s !== undefined ? esc(c.latency_s) + " s" : ""}</span>
        <span class="usd">${esc(usd)}</span>
        <span class="badge case">${esc(c.case)} · ${esc(c.step)}</span>
      </div>
      <div class="muted" style="font-size:14px;margin-top:4px">${esc(fmtUsage(c.usage))}${c.status ? " · status " + esc(c.status) : ""}${c.replay_changed ? " · request changed since recording" : ""}</div>
      <div class="io">
        <div><h4>Request</h4><pre>${fmtReq(c.request)}</pre></div>
        <div><h4>Response</h4><pre>${esc(c.response)}</pre></div>
      </div>`;
    return el;
  }

  async function tick() {
    try {
      const q = sel.value ? `?case=${encodeURIComponent(sel.value)}` : "";
      const r = await fetch(`/api/calls${q}`, { cache: "no-store" });
      const d = await r.json();
      const calls = d.calls || [];
      const sig = calls.map(key).join(";");
      if (sig === lastSig) { dot.classList.remove("paused"); setTimeout(tick, 2000); return; }
      lastSig = sig;
      feed.innerHTML = "";
      if (!calls.length) feed.innerHTML = `<div class="empty">No calls yet. Start a case and the requests appear here as they happen.</div>`;
      calls.forEach(c => {
        const k = key(c);
        feed.appendChild(card(c, !first && !seen.has(k)));
        seen.add(k);
      });
      total.textContent = `${calls.length} calls · $${Number(d.total_usd || 0).toFixed(4)}`;
      dot.classList.remove("paused");
      first = false;
    } catch (e) {
      dot.classList.add("paused");
    }
    setTimeout(tick, 2000);
  }
  sel.addEventListener("change", () => {
    const u = new URL(location.href);
    sel.value ? u.searchParams.set("case", sel.value) : u.searchParams.delete("case");
    history.replaceState(null, "", u);
    seen = new Set(); first = true; lastSig = null;
  });
  tick();
})();
