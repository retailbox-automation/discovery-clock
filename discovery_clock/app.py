"""Web interface (FastAPI + Jinja2, no front-end build, no CDN). Serve only on 127.0.0.1:
    python -m discovery_clock serve --port 8765
Pages: /  cases + upload form · /cases/{id}  claims next to the player · /calls  live Meta calls (auto-refresh)."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import RUNS, pipeline, track as track_mod
from .meta import SPARK_MODEL, SPARK_OPEN_MODEL, VOICE_MODEL, SAM_MODEL, MetaClient, read_calls

HERE = Path(__file__).resolve().parent
CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
MODEL_NAMES = {SPARK_MODEL: "Muse Spark 1.3", SPARK_OPEN_MODEL: "Muse Spark Open (open weights, Baseten)",
               VOICE_MODEL: "Muse Voice Transcribe", SAM_MODEL: "SAM 3.1", "code": "code", "-": "Files API"}

app = FastAPI(title="Discovery Clock", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def hms(s) -> str:
    if s is None:
        return ""
    s = int(round(float(s)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


templates.env.filters["hms"] = hms
templates.env.filters["model_name"] = lambda m: MODEL_NAMES.get(m, m)


def runs_dir() -> Path:
    return Path(app.state.runs_dir) if getattr(app.state, "runs_dir", None) else RUNS


def _case_dir(case_id: str) -> Path:
    if not CASE_RE.match(case_id):
        raise HTTPException(404, "no such case")
    d = runs_dir() / case_id
    if not d.is_dir():
        raise HTTPException(404, "no such case")
    return d


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_cases() -> list[dict]:
    out = []
    base = runs_dir()
    if not base.exists():
        return out
    for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or not CASE_RE.match(d.name):
            continue
        case, status, result = _read(d / "case.json"), _read(d / "status.json"), _read(d / "result.json")
        if not (case or status or result):
            continue
        out.append({"id": d.name, "case": case or {}, "status": status or {}, "result": result,
                    "title": ((result or {}).get("source") or (case or {}).get("source") or {}).get("title") or d.name})
    return out


def video_path(case_id: str) -> Path | None:
    d = _case_dir(case_id)
    case = _read(d / "case.json") or {}
    for p in (case.get("video_path"), d / "video.mp4"):
        if p and Path(p).is_file():
            return Path(p)
    return None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"cases": list_cases(), "page": "cases"})


def _start(case_id: str, video: Path, report: str, mode: str):
    def work():
        with pipeline.RUN_LOCK:
            try:
                pipeline.run(case_id, video, report, claims_mode=mode, runs_dir=runs_dir())
            except Exception:
                pass  # written to status.json by pipeline.run
    threading.Thread(target=work, daemon=True, name=f"run-{case_id}").start()


@app.post("/cases")
async def create_case(request: Request, report: str = Form(...), name: str = Form(""), mode: str = Form("extract"),
                      video_path_text: str = Form(""), video: UploadFile | None = File(None)):
    if mode not in ("extract", "lines"):
        raise HTTPException(400, "bad mode")
    if not report.strip():
        raise HTTPException(400, "the report is empty")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "case"
    case_id = f"{slug}-{time.strftime('%m%d-%H%M%S', time.gmtime())}"
    d = runs_dir() / case_id
    d.mkdir(parents=True, exist_ok=True)
    if video is not None and video.filename:
        dest = d / "video.mp4"
        with open(dest, "wb") as f:
            while chunk := await video.read(1 << 20):
                f.write(chunk)
        src = dest
    elif video_path_text.strip():
        src = Path(video_path_text.strip()).expanduser()
        if not src.is_file() or src.suffix.lower() not in (".mp4", ".mov", ".m4v", ".webm"):
            raise HTTPException(400, "video file not found")
    else:
        raise HTTPException(400, "add a video file")
    (d / "status.json").write_text(json.dumps({"state": "queued", "steps": {s: {"state": "pending"} for s in pipeline.STEPS}}))
    _start(case_id, src, report, mode)
    return RedirectResponse(f"/cases/{case_id}", status_code=303)


@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_page(request: Request, case_id: str):
    d = _case_dir(case_id)
    result, status, case = _read(d / "result.json"), _read(d / "status.json") or {}, _read(d / "case.json") or {}
    vp = video_path(case_id) if result else None
    return templates.TemplateResponse(request, "case.html", {
        "page": "cases", "case_id": case_id, "result": result, "status": status, "case": case,
        "has_video": vp is not None, "steps": pipeline.STEPS, "labels": pipeline.STEP_LABELS,
        "result_json": json.dumps({"claims": (result or {}).get("claims", []),
                                   "duration": ((result or {}).get("video") or {}).get("duration_s", 0)}).replace("</", "<\\/")})


@app.get("/api/cases/{case_id}/status")
def case_status(case_id: str):
    return JSONResponse(_read(_case_dir(case_id) / "status.json") or {})


@app.get("/api/cases/{case_id}/result")
def case_result(case_id: str):
    r = _read(_case_dir(case_id) / "result.json")
    if r is None:
        raise HTTPException(404, "no result yet")
    return JSONResponse(r)


@app.get("/media/{case_id}")
def media(case_id: str):
    p = video_path(case_id)
    if p is None:
        raise HTTPException(404, "video file is not on this computer")
    return FileResponse(p, media_type="video/mp4")


@app.get("/calls", response_class=HTMLResponse)
def calls_page(request: Request, case: str = ""):
    if case and not CASE_RE.match(case):
        case = ""
    return templates.TemplateResponse(request, "calls.html", {"page": "calls", "case": case,
                                                             "cases": [c["id"] for c in list_cases()]})


@app.get("/api/calls")
def api_calls(case: str = "", limit: int = 60):
    if case and not CASE_RE.match(case):
        raise HTTPException(400, "bad case")
    rows = read_calls(runs_dir(), case or None, limit=max(1, min(limit, 500)))
    for r in rows:
        r["model_name"] = MODEL_NAMES.get(r.get("model"), r.get("model"))
    return JSONResponse({"calls": rows[::-1], "total_usd": round(sum(r.get("usd", 0) or 0 for r in rows), 5)})


# ---------------------------------------------------------------- SAM 3.1: follow a person, export a clip with faces blurred
JOB_LOCK = threading.Lock()


def _job(status_path: Path, fn):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"state": "running", "started": time.time()}))

    def work():
        with JOB_LOCK:
            try:
                out = fn()
                status_path.write_text(json.dumps({"state": "done", **out}))
            except Exception as e:  # shown in the UI
                status_path.write_text(json.dumps({"state": "failed", "error": f"{type(e).__name__}: {str(e)[:400]}"}))
    threading.Thread(target=work, daemon=True).start()


def _case_media(case_id: str) -> tuple[Path, dict, dict]:
    d = _case_dir(case_id)
    result, case = _read(d / "result.json"), _read(d / "case.json") or {}
    if not result:
        raise HTTPException(409, "the case has no result yet")
    vp = video_path(case_id)
    if vp is None:
        raise HTTPException(404, "video file is not on this computer")
    return vp, result, case


@app.post("/api/cases/{case_id}/track")
async def start_track(case_id: str, request: Request):
    vp, result, case = _case_media(case_id)
    body = await request.json()
    try:
        t, x, y = float(body["t"]), int(body["x"]), int(body["y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "need t, x, y")
    dur = result["video"]["duration_s"]
    w, h = track_mod.video_size(vp)
    if not (0 <= t <= dur and 0 <= x < w and 0 <= y < h):
        raise HTTPException(400, "point outside the video")
    d = _case_dir(case_id)
    status = d / "tracks" / "latest.json"

    def run():
        client = MetaClient(case_id, runs_dir=runs_dir())
        out = track_mod.track(client, vp, result["video"]["sha256"], dur, t, x, y)
        track_mod.save_json(out, d / "tracks" / f"{t:.1f}-{x}-{y}.json")
        return out
    _job(status, run)
    return JSONResponse({"state": "running"})


@app.get("/api/cases/{case_id}/track")
def get_track(case_id: str):
    st = _read(_case_dir(case_id) / "tracks" / "latest.json")
    return JSONResponse(st or {"state": "none"})


@app.post("/api/cases/{case_id}/claims/{claim_id}/clip")
def start_clip(case_id: str, claim_id: str):
    vp, result, case = _case_media(case_id)
    c = next((x for x in result["claims"] if x["id"] == claim_id), None)
    if c is None or c.get("start_s") is None:
        raise HTTPException(404, "claim has no timecode")
    if not CASE_RE.match(claim_id):
        raise HTTPException(400, "bad claim id")
    d = _case_dir(case_id)
    dur = result["video"]["duration_s"]
    a, b = track_mod.clip_window(c, dur)

    def run():
        client = MetaClient(case_id, runs_dir=runs_dir())
        fc = track_mod.faces(client, vp, result["video"]["sha256"], dur, a, b)
        latest = _read(d / "tracks" / "latest.json") or {}
        keep = latest.get("boxes") if latest.get("state") == "done" else None
        out_path = d / "clips" / f"{claim_id}.mp4"
        track_mod.export_clip(vp, a, b, fc["boxes"], out_path, keep_boxes=keep)
        faces_n = len({(r["t"], r["id"]) for r in fc["boxes"]})
        return {"url": f"/clips/{case_id}/{claim_id}.mp4", "window": [a, b], "face_boxes": faces_n,
                "kept_followed_person": bool(keep)}
    _job(d / "clips" / f"{claim_id}.json", run)
    return JSONResponse({"state": "running", "window": [a, b]})


@app.get("/api/cases/{case_id}/claims/{claim_id}/clip")
def get_clip(case_id: str, claim_id: str):
    if not CASE_RE.match(claim_id):
        raise HTTPException(400, "bad claim id")
    return JSONResponse(_read(_case_dir(case_id) / "clips" / f"{claim_id}.json") or {"state": "none"})


@app.get("/clips/{case_id}/{claim_id}.mp4")
def clip_file(case_id: str, claim_id: str):
    if not CASE_RE.match(claim_id):
        raise HTTPException(400, "bad claim id")
    p = _case_dir(case_id) / "clips" / f"{claim_id}.mp4"
    if not p.is_file():
        raise HTTPException(404, "no clip")
    return FileResponse(p, media_type="video/mp4", filename=f"{case_id}-{claim_id}-faces-blurred.mp4")
