"""Client for the Meta models used at runtime, with a call log, a spend ledger with a hard cap, and record/replay.

Every call (live or replayed) is appended to runs/<case>/calls.jsonl: UTC time, model, endpoint, a short request
summary (never the key), a response excerpt (<= 2000 chars), latency, usage and the estimated cost.
Live spend goes to the ledger (runs/ledger.jsonl by default); once the ledger reaches DC_CAP_USD the next live call
raises SpendCapReached.

Replay (DC_REPLAY=1): responses come from recordings/<case>/<step>-<hash>.json, written during live runs.
The hash covers the full request with uploaded files replaced by the sha256 of their bytes, so a replay proves the
same request was made. No network and no key are needed in replay mode.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import RECORDINGS, RUNS

VOICE_MODEL = "muse-voice-transcribe-1.0"
SPARK_MODEL = "muse-spark-1.3"
SPARK_OPEN_MODEL = "meta-models/muse-spark-open-1026"
SAM_MODEL = "sam-3.1"
BASETEN_URL = "https://inference.baseten.co/v1/chat/completions"

# USD. Spark 1.3: per 1M input / output tokens. Voice: per hour of audio. SAM video: per 1000 frames, minimum 50 frames.
PRICE_SPARK = (1.25, 4.25)
PRICE_VOICE_HOUR = 0.18
PRICE_SAM_1000_FRAMES = 0.20
SAM_MIN_FRAMES = 50

EXCERPT = 2000
CHUNK_MAX_S = 570.0  # Voice Transcribe takes at most 10 min per file; we cut at 9:30


class SpendCapReached(RuntimeError):
    pass


class ReplayMissing(RuntimeError):
    pass


class MetaAPIError(RuntimeError):
    pass


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spark_cost(usage: dict) -> float:
    pin, pout = PRICE_SPARK
    return usage.get("input_tokens", 0) / 1e6 * pin + usage.get("output_tokens", 0) / 1e6 * pout


def voice_cost(audio_ms: float) -> float:
    return audio_ms / 3.6e6 * PRICE_VOICE_HOUR


def sam_cost(frames: int) -> float:
    return max(frames, SAM_MIN_FRAMES) / 1000 * PRICE_SAM_1000_FRAMES


def plan_chunks(duration_s: float, max_s: float = CHUNK_MAX_S) -> list[tuple[float, float]]:
    """Cut [0, duration] into consecutive pieces no longer than max_s."""
    if duration_s <= 0:
        return []
    out, start = [], 0.0
    while start < duration_s - 1e-6:
        end = min(start + max_s, duration_s)
        out.append((round(start, 3), round(end, 3)))
        start = end
    return out


def merge_turns(pieces: list[tuple[float, dict]]) -> list[dict]:
    """pieces: [(offset_s, transcribe_response)] -> one list of turns with times shifted by the piece offset."""
    turns = []
    for offset_s, resp in pieces:
        shift = int(round(offset_s * 1000))
        for t in resp.get("turns", []):
            turns.append({"startMs": int(t.get("startMs", 0)) + shift, "endMs": int(t.get("endMs", 0)) + shift,
                          "speaker": t.get("speaker", "?"), "transcript": t.get("transcript", ""),
                          "piece_offset_s": offset_s})
    turns.sort(key=lambda t: (t["startMs"], t["endMs"]))
    return turns


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _clip(s: str, n: int = EXCERPT) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def response_text(raw: dict) -> str:
    """Join the message text of a /responses result."""
    return "".join(c.get("text", "") for o in raw.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if isinstance(c, dict))


def parse_json_text(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i: j + 1])
        raise


class MetaClient:
    def __init__(self, case_id: str, *, replay: bool | None = None, record: bool = True, cap_usd: float | None = None,
                 runs_dir: Path | None = None, recordings_dir: Path | None = None, ledger_path: Path | None = None):
        self.case_id = case_id
        self.replay = (os.environ.get("DC_REPLAY", "0") == "1") if replay is None else replay
        self.record = record and not self.replay
        self.cap = float(os.environ.get("DC_CAP_USD", "10")) if cap_usd is None else cap_usd
        self.runs = Path(runs_dir or RUNS)
        self.recordings = Path(recordings_dir or RECORDINGS)
        self.case_dir = self.runs / case_id
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.case_dir / "calls.jsonl"
        self.ledger_path = Path(ledger_path or os.environ.get("DC_LEDGER", self.runs / "ledger.jsonl"))
        self.base = os.environ.get("META_MODEL_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
        self.file_sha: dict[str, str] = {}
        self.case_spend = 0.0

    # ---------- keys (read from the environment only when a live call is made; never logged) ----------
    def _meta_key(self) -> str:
        k = os.environ.get("MODEL_API_KEY")
        if not k:
            raise MetaAPIError("MODEL_API_KEY is not set (live mode). Use DC_REPLAY=1 to replay recorded responses.")
        return k

    # ---------- ledger ----------
    def spent(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        for line in self.ledger_path.read_text().splitlines():
            if line.strip():
                total += float(json.loads(line).get("usd", 0))
        return total

    def guard(self, estimate: float = 0.0):
        if self.replay:
            return
        s = self.spent()
        if s >= self.cap or s + estimate > self.cap:
            raise SpendCapReached(f"spend cap ${self.cap:.2f} reached (spent ${s:.4f}, next call ~${estimate:.4f})")

    def _charge(self, usd: float, model: str, note: str):
        self.case_spend += usd
        if self.replay:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps({"ts": utc_now(), "case": self.case_id, "model": model, "usd": round(usd, 6),
                                "note": note}) + "\n")

    # ---------- call log ----------
    def _log(self, rec: dict):
        rec = {"ts": utc_now(), "case": self.case_id, "mode": "replay" if self.replay else "live", **rec}
        with open(self.calls_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- recordings ----------
    def _rec_path(self, step: str, h: str) -> Path:
        return self.recordings / self.case_id / f"{step}-{h}.json"

    def _save_recording(self, step: str, h: str, key: dict, response, meta: dict):
        if not self.record:
            return
        p = self._rec_path(step, h)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"step": step, "hash": h, "recorded_utc": utc_now(), "request": key,
                                 "response": response, **meta}, ensure_ascii=False, indent=1))

    def _load_recording(self, step: str, h: str) -> tuple[dict, bool]:
        """Exact hash in this case, then exact hash in any case, then same step in this case (flagged as changed)."""
        p = self._rec_path(step, h)
        if p.exists():
            return json.loads(p.read_text()), False
        for q in self.recordings.glob(f"*/{step}-{h}.json"):
            return json.loads(q.read_text()), False
        same_step = sorted((self.recordings / self.case_id).glob(f"{step}-*.json"))
        if same_step:
            return json.loads(same_step[-1].read_text()), True
        raise ReplayMissing(f"no recording for step '{step}' (hash {h}) in {self.recordings / self.case_id}")

    def _canon(self, obj):
        """Replace uploaded file ids with the sha256 of the uploaded bytes, so hashes survive re-uploads."""
        obj = copy.deepcopy(obj)

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k == "file_id" and isinstance(v, str):
                        x[k] = "sha256:" + self.file_sha.get(v, v)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(obj)
        return obj

    # ---------- HTTP ----------
    def _post_json(self, url: str, body: dict, key: str, timeout: int = 900) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise MetaAPIError(f"HTTP {e.code} from {url.split('?')[0]}: {e.read()[:800]!r}") from None
        except urllib.error.URLError as e:
            raise MetaAPIError(f"network error for {url.split('?')[0]}: {e.reason}") from None

    def _post_multipart(self, url: str, parts: list[str], key: str, timeout: int = 900) -> dict:
        # curl streams large files; the Authorization header goes through stdin so the key never shows in `ps`.
        cmd = ["curl", "-sS", "--fail-with-body", "-m", str(timeout), "--connect-timeout", "30",
               "--speed-limit", "2000", "--speed-time", "90",  # abort a stalled transfer instead of hanging for 15 min
               url, "-H", "@-", "-H", "Accept: application/json"]
        for p in parts:
            cmd += ["-F", p]
        for attempt in range(2):  # one retry for a stalled or dropped transfer (curl 28 timeout, 7/56 connection)
            out = subprocess.run(cmd, input=f"Authorization: Bearer {key}\n", capture_output=True, text=True)
            if out.returncode not in (7, 28, 56):
                break
        if out.returncode:
            raise MetaAPIError(f"curl rc={out.returncode} for {url.split('?')[0]}: {out.stdout[:600]} {out.stderr[:300]}")
        return json.loads(out.stdout)

    # ---------- Files API ----------
    def upload(self, path: str | os.PathLike, *, step: str = "upload", media_key: str | None = None) -> str:
        """media_key identifies the content for record/replay (default: sha256 of the file). The pipeline passes
        "<sha256 of the original recording>:<stamp version>" so a replay does not need to rebuild the stamped copy."""
        sha = media_key or sha256_file(path)
        size = os.path.getsize(path) if os.path.exists(path) else None
        key = {"endpoint": "/files", "purpose": "user_data", "media_sha256": sha}
        h = canonical_hash(key)
        t0 = time.time()
        changed = False
        if self.replay:
            rec, changed = self._load_recording(step, h)
            d = rec["response"]
        else:
            self.guard()
            d = self._post_multipart(f"{self.base}/files", ["purpose=user_data", f"file=@{path}"], self._meta_key())
        sec = time.time() - t0
        fid = d["id"]
        self.file_sha[fid] = sha
        self._save_recording(step, h, key, {"id": fid}, {"latency_s": round(sec, 2)})
        self._log({"step": step, "model": "-", "endpoint": "POST /files", "latency_s": round(sec, 2), "usd": 0.0,
                   "request": {"purpose": "user_data", "file": Path(path).name, "bytes": size, "content": sha[:16] + sha[64:]},
                   "response": _clip(json.dumps({"id": fid})), "replay_changed": changed})
        return fid

    # ---------- Muse Voice Transcribe ----------
    def transcribe(self, video: str | os.PathLike, duration_s: float, *, media_sha: str | None = None) -> dict:
        """Diarized speech for the whole recording, cut into pieces <= 9:30, times shifted back to the video clock."""
        media_sha = media_sha or sha256_file(video)
        pieces = []
        total_ms = 0
        with tempfile.TemporaryDirectory() as td:
            for i, (a, b) in enumerate(plan_chunks(duration_s)):
                step = f"transcribe-{i}"
                req = {"mode": "DIARIZATION", "model": VOICE_MODEL, "audioEncoding": "WAV"}
                key = {"endpoint": "/asr/transcribe", **req, "media_sha256": media_sha, "piece_s": [a, b],
                       "audio": "mono s16le 24000 Hz"}
                h = canonical_hash(key)
                t0 = time.time()
                changed = False
                if self.replay:
                    rec, changed = self._load_recording(step, h)
                    d = rec["response"]
                    usd = rec.get("usd", 0.0)
                else:
                    self.guard(voice_cost((b - a) * 1000))
                    wav = Path(td) / f"piece{i}.wav"
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", str(video),
                                    "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1",
                                    "-fflags", "+bitexact", str(wav)], check=True)
                    d = self._post_multipart(f"{self.base}/asr/transcribe?sessionId=dc-{uuid.uuid4().hex[:12]}",
                                             [f"request={json.dumps(req)};type=application/json", f"audio=@{wav}"],
                                             self._meta_key())
                    usd = voice_cost(d.get("audioDurationMs", (b - a) * 1000))
                sec = time.time() - t0
                self._charge(usd, VOICE_MODEL, f"piece {i} {a:.0f}-{b:.0f}s")
                self._save_recording(step, h, key, d, {"latency_s": round(sec, 2), "usd": round(usd, 6),
                                                       "model": VOICE_MODEL})
                turns = d.get("turns", [])
                self._log({"step": step, "model": VOICE_MODEL, "endpoint": "POST /asr/transcribe", "latency_s": round(sec, 2),
                           "usd": round(usd, 6),
                           "request": {"mode": "DIARIZATION", "audioEncoding": "WAV", "audio": f"{Path(video).name} {a:.1f}-{b:.1f}s, mono 24 kHz"},
                           "response": _clip("\n".join(f"[{t.get('startMs', 0)/1000+a:.1f}s] {t.get('speaker', '?')}: {t.get('transcript', '')}" for t in turns)
                                             or "(no speech)"),
                           "usage": {"audioDurationMs": d.get("audioDurationMs"), "turns": len(turns)}, "replay_changed": changed})
                pieces.append((a, d))
                total_ms += d.get("audioDurationMs", 0) or 0
        return {"model": VOICE_MODEL, "turns": merge_turns(pieces), "audioDurationMs": total_ms, "pieces": len(pieces)}

    # ---------- /responses (Muse Spark 1.3, SAM 3.1) ----------
    def responses(self, body: dict, *, step: str, summary: dict, estimate: float = 0.0, cost_fn=None,
                  usage_extra: dict | None = None) -> tuple[str, dict]:
        """POST /responses. Returns (message text, raw response). cost_fn(raw) -> usd (default: Spark token prices)."""
        key = {"endpoint": "/responses", "body": self._canon(body)}
        h = canonical_hash(key)
        t0 = time.time()
        changed = False
        if self.replay:
            rec, changed = self._load_recording(step, h)
            raw = rec["response"]
            usd = rec.get("usd", 0.0)
        else:
            self.guard(estimate)
            raw = self._post_json(f"{self.base}/responses", body, self._meta_key())
            usd = (cost_fn or (lambda r: spark_cost(r.get("usage", {}))))(raw)
        sec = time.time() - t0
        model = body.get("model", "?")
        self._charge(usd, model, step)
        self._save_recording(step, h, key, raw, {"latency_s": round(sec, 2), "usd": round(usd, 6), "model": model})
        text = response_text(raw)
        self._log({"step": step, "model": model, "endpoint": "POST /responses", "latency_s": round(sec, 2),
                   "usd": round(usd, 6), "request": summary, "response": _clip(text or json.dumps(raw)[:EXCERPT]),
                   "usage": {**(raw.get("usage") or {}), **(usage_extra or {})} or None, "status": raw.get("status"),
                   "replay_changed": changed})
        return text, raw

    # ---------- Muse Spark Open on Baseten (OpenAI-compatible) ----------
    def baseten_chat(self, messages: list[dict], *, step: str, summary: dict, schema: dict | None = None,
                     max_tokens: int = 8000) -> tuple[str, dict]:
        body = {"model": SPARK_OPEN_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": step, "schema": schema, "strict": True}}
        key = {"endpoint": BASETEN_URL, "body": body}
        h = canonical_hash(key)
        t0 = time.time()
        changed = False
        if self.replay:
            rec, changed = self._load_recording(step, h)
            raw = rec["response"]
        else:
            bkey = os.environ.get("BASETEN_API_KEY")
            if not bkey:
                raise MetaAPIError("BASETEN_API_KEY is not set")
            self.guard()
            try:
                raw = self._post_json(BASETEN_URL, body, bkey, timeout=300)
            except MetaAPIError as e:
                s = str(e)
                if schema and ("response_format" in s.lower() or "HTTP 400" in s or "HTTP 422" in s):
                    # Retry once without structured output; the prompt still asks for JSON and the caller validates it.
                    body2 = {k: v for k, v in body.items() if k != "response_format"}
                    raw = self._post_json(BASETEN_URL, body2, bkey, timeout=300)
                else:
                    raise
        sec = time.time() - t0
        msg = (raw.get("choices") or [{}])[0].get("message", {}) or {}
        text = msg.get("content") or ""
        self._charge(0.0, SPARK_OPEN_MODEL, f"{step} (Baseten credits)")
        self._save_recording(step, h, key, raw, {"latency_s": round(sec, 2), "usd": 0.0, "model": SPARK_OPEN_MODEL})
        self._log({"step": step, "model": SPARK_OPEN_MODEL, "endpoint": "POST baseten /v1/chat/completions",
                   "latency_s": round(sec, 2), "usd": 0.0, "request": summary, "response": _clip(text or json.dumps(raw)),
                   "usage": raw.get("usage"), "replay_changed": changed, "note": "Muse Spark Open, open weights on Baseten"})
        return text, raw


def read_calls(runs_dir: Path | None = None, case_id: str | None = None, limit: int = 200) -> list[dict]:
    runs = Path(runs_dir or RUNS)
    paths = [runs / case_id / "calls.jsonl"] if case_id else sorted(runs.glob("*/calls.jsonl"))
    out = []
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    out.sort(key=lambda r: r.get("ts", ""))
    return out[-limit:]
