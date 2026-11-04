(function () {
  // progress page: poll the status and reload when the run is done
  const prog = document.getElementById("progress");
  if (prog) {
    const id = prog.dataset.case;
    const tick = async () => {
      try {
        const r = await fetch(`/api/cases/${encodeURIComponent(id)}/status`, { cache: "no-store" });
        const s = await r.json();
        if (s.state === "done") { location.reload(); return; }
        for (const [name, st] of Object.entries(s.steps || {})) {
          const li = prog.querySelector(`[data-step="${name}"]`);
          if (li) li.className = `step ${st.state || "pending"}`;
        }
        if (s.state === "failed") { location.reload(); return; }
      } catch (e) { /* keep polling */ }
      setTimeout(tick, 2000);
    };
    if (!document.querySelector(".err")) setTimeout(tick, 1500);
    return;
  }

  const dataEl = document.getElementById("case-data");
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent || "{}");
  const video = document.getElementById("video");
  const timeline = document.getElementById("timeline");
  const playhead = document.getElementById("playhead");
  const dur = () => (video && isFinite(video.duration) && video.duration > 0 ? video.duration : data.duration || 1);
  const cards = [...document.querySelectorAll(".claim")];
  const turns = [...document.querySelectorAll(".turn")];

  function seek(t, id) {
    cards.forEach(c => c.classList.toggle("active", c.dataset.id === id));
    document.querySelectorAll(".mk").forEach(m => m.classList.toggle("active", m.dataset.id === id));
    if (video && t !== "" && t !== null && !isNaN(t)) {
      video.currentTime = Math.max(0, Number(t) - 1);
      video.play().catch(() => {});
    }
  }

  function drawMarkers() {
    if (!timeline) return;
    timeline.querySelectorAll(".mk").forEach(m => m.remove());
    // overlapping claims go to separate lanes so every marker stays visible and clickable
    const d = dur();
    const placed = (data.claims || []).map((c, i) => ({ c, i }))
      .filter(x => x.c.start_s !== null && x.c.start_s !== undefined && x.c.status !== "not_found")
      .sort((a, b) => a.c.start_s - b.c.start_s);
    const laneEnd = [];
    const minSpan = d * 0.025;
    placed.forEach(x => {
      const end = Math.max(x.c.end_s, x.c.start_s + minSpan);
      let lane = laneEnd.findIndex(e => e <= x.c.start_s);
      if (lane < 0) { lane = laneEnd.length; laneEnd.push(0); }
      laneEnd[lane] = end;
      x.lane = lane;
    });
    const lanes = Math.max(1, laneEnd.length);
    timeline.style.height = `${8 + lanes * 20}px`;
    placed.forEach(({ c, i, lane }) => {
      const m = document.createElement("div");
      m.className = `mk ${c.status === "on_recording" ? "on" : "ck"}`;
      m.dataset.id = c.id;
      m.style.top = `${4 + lane * 20}px`;
      m.style.height = "17px";
      m.style.bottom = "auto";
      m.style.left = `${(c.start_s / d) * 100}%`;
      m.style.width = `${Math.max(2.5, ((c.end_s - c.start_s) / d) * 100)}%`;
      m.title = `${i + 1}. ${c.text}`;
      m.innerHTML = `<span>${i + 1}</span>`;
      m.addEventListener("click", ev => {
        ev.stopPropagation();
        seek(c.start_s, c.id);
        const card = cards.find(x => x.dataset.id === c.id);
        if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
      });
      timeline.appendChild(m);
    });
  }

  cards.forEach(card => {
    const go = () => seek(card.dataset.start, card.dataset.id);
    card.addEventListener("click", go);
    card.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  });
  turns.forEach(t => t.addEventListener("click", () => { if (video) { video.currentTime = Number(t.dataset.start); video.play().catch(() => {}); } }));
  if (timeline) timeline.addEventListener("click", e => {
    if (!video) return;
    const r = timeline.getBoundingClientRect();
    video.currentTime = ((e.clientX - r.left) / r.width) * dur();
  });
  if (video) {
    video.addEventListener("loadedmetadata", drawMarkers);
    video.addEventListener("timeupdate", () => {
      const t = video.currentTime;
      if (playhead) playhead.style.left = `${(t / dur()) * 100}%`;
      let cur = null;
      turns.forEach(el => {
        const on = t >= Number(el.dataset.start) && t <= Number(el.dataset.end) + 0.3;
        el.classList.toggle("now", on);
        if (on) cur = el;
      });
      if (cur) {
        const box = document.getElementById("turns");
        const top = cur.offsetTop - box.offsetTop;
        if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - 30) box.scrollTop = top - 60;
      }
    });
  }
  drawMarkers();

  // ---------------- SAM 3.1: follow a person (click in the video) and draw the box during playback
  const caseId = dataEl.dataset.case;
  const wrap = document.querySelector(".vwrap");
  const canvas = document.getElementById("overlay");
  const followBtn = document.getElementById("follow");
  const followStatus = document.getElementById("follow-status");
  let trackData = null;

  function contentRect() {
    const cw = video.clientWidth, ch = video.clientHeight, vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
    const s = Math.min(cw / vw, ch / vh);
    return { s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2, vw, vh };
  }
  function drawTrack() {
    if (!canvas || !video) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = video.clientWidth * dpr; canvas.height = video.clientHeight * dpr;
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, canvas.width, canvas.height);
    if (!trackData || !trackData.boxes || !trackData.boxes.length) return;
    const t = video.currentTime;
    let best = null;
    for (const b of trackData.boxes) if (!best || Math.abs(b.t - t) < Math.abs(best.t - t)) best = b;
    if (!best || Math.abs(best.t - t) > 0.15) return;
    const r = contentRect();
    const [x1, y1, x2, y2] = best.box;
    g.strokeStyle = "#2dd4bf"; g.lineWidth = 3;
    g.strokeRect(r.ox + x1 * r.s, r.oy + y1 * r.s, (x2 - x1) * r.s, (y2 - y1) * r.s);
    g.fillStyle = "#2dd4bf"; g.font = "bold 13px -apple-system, sans-serif";
    g.fillText("followed person · SAM 3.1", r.ox + x1 * r.s + 4, Math.max(14, r.oy + y1 * r.s - 6));
  }
  const fmt = s => new Date(s * 1000).toISOString().slice(11, 19);
  async function pollTrack(once) {
    try {
      const st = await (await fetch(`/api/cases/${encodeURIComponent(caseId)}/track`, { cache: "no-store" })).json();
      if (st.state === "done") {
        trackData = st;
        followStatus.textContent = `Following from ${fmt(st.window[0])} to ${fmt(st.window[1])} · ${st.boxes.length} boxes`;
        drawTrack();
        return;
      }
      if (st.state === "failed") { followStatus.textContent = `SAM 3.1 failed: ${st.error}`; return; }
      if (st.state === "running") { followStatus.textContent = "SAM 3.1 is following the person…"; if (!once || true) setTimeout(() => pollTrack(false), 2000); }
    } catch (e) { setTimeout(() => pollTrack(false), 3000); }
  }
  if (video && canvas && followBtn) {
    video.addEventListener("timeupdate", drawTrack);
    video.addEventListener("seeked", drawTrack);
    window.addEventListener("resize", drawTrack);
    followBtn.addEventListener("click", () => {
      const on = !wrap.classList.contains("picking");
      wrap.classList.toggle("picking", on);
      followBtn.classList.toggle("on", on);
      followStatus.textContent = on ? "Pause on a frame and click the person to follow" : "";
      if (on) video.pause();
    });
    canvas.addEventListener("click", async e => {
      if (!wrap.classList.contains("picking")) return;
      const rect = canvas.getBoundingClientRect(), r = contentRect();
      const x = Math.round((e.clientX - rect.left - r.ox) / r.s), y = Math.round((e.clientY - rect.top - r.oy) / r.s);
      if (x < 0 || y < 0 || x >= r.vw || y >= r.vh) return;
      wrap.classList.remove("picking"); followBtn.classList.remove("on");
      video.pause();
      followStatus.textContent = "SAM 3.1 is following the person…";
      const res = await fetch(`/api/cases/${encodeURIComponent(caseId)}/track`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ t: video.currentTime, x, y }) });
      if (!res.ok) { followStatus.textContent = `Could not start: ${await res.text()}`; return; }
      setTimeout(() => pollTrack(false), 1500);
    });
    pollTrack(true);
  }

  // ---------------- export a clip for a claim with bystander faces blurred (SAM 3.1 "face")
  document.querySelectorAll(".clip-btn").forEach(btn => {
    const status = btn.parentElement.querySelector(".clip-status");
    const url = `/api/cases/${encodeURIComponent(caseId)}/claims/${encodeURIComponent(btn.dataset.id)}/clip`;
    const show = st => {
      if (st.state === "done") {
        status.innerHTML = `<a href="${st.url}" target="_blank" rel="noopener">Open clip</a> · ${fmt(st.window[0])}–${fmt(st.window[1])} · ${st.face_boxes} face boxes blurred${st.kept_followed_person ? " · followed person kept" : ""}`;
        return true;
      }
      if (st.state === "failed") { status.textContent = `Failed: ${st.error}`; return true; }
      if (st.state === "running") { status.textContent = "SAM 3.1 is finding faces…"; return false; }
      return true;
    };
    const poll = async () => {
      try { if (!show(await (await fetch(url, { cache: "no-store" })).json())) setTimeout(poll, 2000); }
      catch (e) { setTimeout(poll, 3000); }
    };
    btn.addEventListener("click", async ev => {
      ev.stopPropagation();
      status.textContent = "Starting…";
      const r = await fetch(url, { method: "POST" });
      if (!r.ok) { status.textContent = `Could not start: ${await r.text()}`; return; }
      setTimeout(poll, 1500);
    });
    poll();
  });
})();
