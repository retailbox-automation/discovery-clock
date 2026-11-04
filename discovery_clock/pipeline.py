"""video + report -> runs/<case>/result.json. Steps: claims, transcribe, upload, review, verify."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

from . import RECORDINGS, ROOT, RUNS, claims as claims_mod, review as review_mod, stamp as stamp_mod, verify as verify_mod
from .meta import SPARK_MODEL, VOICE_MODEL, MetaClient, sha256_file, utc_now

STEPS = ["claims", "transcribe", "upload", "review", "verify"]
STEP_LABELS = {"claims": "Split the report into claims", "transcribe": "Transcribe speech with speakers",
               "upload": "Stamp video time on each frame, upload", "review": "Place each claim on the recording",
               "verify": "Code checks the model's answer"}
RUN_LOCK = threading.Lock()


def probe_duration(video: str | Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def source_for(video: Path) -> dict | None:
    p = ROOT / "data" / "sources.json"
    if not p.exists():
        return None
    try:
        for s in json.loads(p.read_text()):
            if s.get("id") and s["id"] in video.stem:
                return s
    except (json.JSONDecodeError, TypeError):
        return None
    return None


class Status:
    def __init__(self, case_dir: Path):
        self.path = case_dir / "status.json"
        self.data = {"state": "running", "started_utc": utc_now(), "steps": {s: {"state": "pending"} for s in STEPS}}
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def step(self, name: str, state: str, **extra):
        self.data["steps"][name] = {"state": state, "utc": utc_now(), **extra}
        self.save()

    def finish(self, state: str, **extra):
        self.data.update({"state": state, "finished_utc": utc_now(), **extra})
        self.save()


def run(case_id: str, video: str | Path, report_text: str, *, claims_mode: str = "extract", use_open: bool = True,
        client: MetaClient | None = None, runs_dir: Path | None = None, duration_s: float | None = None) -> dict:
    video = Path(video).resolve()
    runs = Path(runs_dir or RUNS)
    case_dir = runs / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    client = client or MetaClient(case_id, runs_dir=runs)
    st = Status(case_dir)
    t_start = time.time()
    current = "claims"
    try:
        duration = duration_s if duration_s is not None else probe_duration(video)
        media_sha = sha256_file(video)
        src = source_for(video)
        case = {"case": case_id, "created_utc": utc_now(), "video_path": str(video), "video_name": video.name,
                "video_sha256": media_sha, "duration_s": round(duration, 3), "claims_mode": claims_mode,
                "report": report_text, "source": src, "replay": client.replay}
        (case_dir / "case.json").write_text(json.dumps(case, indent=1, ensure_ascii=False))

        st.step("claims", "running")
        cl = claims_mod.extract(client, report_text, claims_mode, use_open=use_open)
        st.step("claims", "done", count=len(cl["claims"]), model=cl["model"])

        current = "transcribe"
        st.step("transcribe", "running")
        tr = client.transcribe(video, duration, media_sha=media_sha)
        st.step("transcribe", "done", turns=len(tr["turns"]), pieces=tr["pieces"])

        current = "upload"
        st.step("upload", "running")
        media_key = f"{media_sha}:{stamp_mod.STAMP_VERSION}"
        if client.replay:  # the recorded upload is found by media_key; no need to rebuild the stamped copy
            fid = client.upload(video, media_key=media_key)
        else:
            stamped = case_dir / "upload-stamped.mp4"
            stamp_mod.make_stamped(video, stamped, duration)
            fid = client.upload(stamped, media_key=media_key)
            stamped.unlink(missing_ok=True)
        st.step("upload", "done")

        current = "review"
        st.step("review", "running")
        answer, raw = review_mod.review(client, fid, tr["turns"], cl["claims"], duration)
        (case_dir / "review-raw.json").write_text(json.dumps({"answer": answer, "usage": raw.get("usage"),
                                                              "status": raw.get("status")}, indent=1, ensure_ascii=False))
        st.step("review", "done", usage=raw.get("usage"))

        current = "verify"
        st.step("verify", "running")
        checked = verify_mod.verify(answer, cl["claims"], tr["turns"], duration)
        st.step("verify", "done", changed=checked["summary"]["changed_by_code"])

        result = {"case": case_id, "created_utc": utc_now(), "replay": client.replay,
                  "video": {"name": video.name, "sha256": media_sha, "duration_s": round(duration, 3)},
                  "source": src, "report": report_text, "claims_mode": claims_mode,
                  "models": {"claims": cl["model"], "transcribe": VOICE_MODEL, "review": SPARK_MODEL, "verify": "code"},
                  "claims_notes": cl["notes"], "turns": tr["turns"], "camera_clock": checked["camera_clock"],
                  "claims": checked["claims"], "summary": checked["summary"], "verify_notes": checked["notes"],
                  "spend_usd": round(client.case_spend, 5), "seconds": round(time.time() - t_start, 1)}
        (case_dir / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        if client.record:
            rec_dir = client.recordings / case_id
            rec_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(case_dir / "result.json", rec_dir / "result.json")
            (rec_dir / "case.json").write_text(json.dumps({k: v for k, v in case.items() if k != "video_path"},
                                                          indent=1, ensure_ascii=False))
        st.finish("done", summary=result["summary"], spend_usd=result["spend_usd"])
        return result
    except Exception as e:  # the UI shows the failure; the CLI re-raises it
        st.step(current, "failed", error=str(e)[:500])
        st.finish("failed", error=f"{type(e).__name__}: {str(e)[:500]}", trace=traceback.format_exc()[-1500:])
        raise


def recorded_case(case_id: str, recordings_dir: Path | None = None) -> dict:
    p = Path(recordings_dir or RECORDINGS) / case_id / "case.json"
    if not p.exists():
        raise FileNotFoundError(f"no recorded case {case_id} ({p})")
    return json.loads(p.read_text())
