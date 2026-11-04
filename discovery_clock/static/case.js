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
})();
