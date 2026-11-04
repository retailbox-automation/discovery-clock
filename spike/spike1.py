"""Spike 1: can Muse Spark 1.3 place report claims on a body-camera video with usable timecodes?
Steps: ffmpeg audio -> Muse Voice Transcribe (DIARIZATION) -> Files API upload -> one Muse Spark 1.3 call with the video,
the diarized turns and the claims -> JSON per claim. Logs every call (request summary, latency, usage) to spike/calls.jsonl
and estimated cost to spike/ledger.jsonl; stops if the ledger passes CAP_USD.
Usage: MODEL_API_KEY=... python3 spike1.py media/<id>.mp4 claims.txt out.json"""
import json, os, subprocess, sys, time, urllib.error, urllib.request, uuid

BASE = os.environ.get("META_MODEL_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
KEY = os.environ["MODEL_API_KEY"]
HERE = os.path.dirname(os.path.abspath(__file__))
CAP_USD = float(os.environ.get("DC_CAP_USD", "10"))
PRICE = {"muse-spark-1.3": (1.25, 4.25)}  # USD per 1M input / output tokens (Standard), docs/RULES-OFFICIAL-2026-10-26.md


def spent():
    p = os.path.join(HERE, "ledger.jsonl")
    return sum(json.loads(l)["usd"] for l in open(p)) if os.path.exists(p) else 0.0


def ledger(kind, usd, note):
    with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "usd": round(usd, 5), "note": note}) + "\n")


def log_call(rec):
    with open(os.path.join(HERE, "calls.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def guard():
    if spent() >= CAP_USD:
        sys.exit(f"STOP: spend cap ${CAP_USD} reached")


def curl_multipart(url, parts, accept="application/json"):
    cmd = ["curl", "-sS", "--fail-with-body", "-m", "600", url, "-H", f"Authorization: Bearer {KEY}", "-H", f"Accept: {accept}"]
    for p in parts:
        cmd += ["-F", p]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"curl rc={out.returncode}: {out.stdout[:500]} {out.stderr[:300]}")
    return json.loads(out.stdout), time.time() - t0


def transcribe(mp4):
    guard()
    wav = mp4.rsplit(".", 1)[0] + ".wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1", wav], check=True)
    req = json.dumps({"mode": "DIARIZATION", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"})
    d, sec = curl_multipart(f"{BASE}/asr/transcribe?sessionId=dc-{uuid.uuid4().hex[:12]}", [f"request={req};type=application/json", f"audio=@{wav}"])
    hours = d.get("audioDurationMs", 0) / 3.6e6
    ledger("voice", hours * 0.18, os.path.basename(wav))
    log_call({"model": "muse-voice-transcribe-1.0", "endpoint": "/asr/transcribe", "sec": round(sec, 2), "turns": len(d.get("turns", [])), "audio_ms": d.get("audioDurationMs")})
    return d


def upload(mp4):
    guard()
    d, sec = curl_multipart(f"{BASE}/files", ["purpose=user_data", f"file=@{mp4}"])
    log_call({"endpoint": "/files", "sec": round(sec, 2), "file_id": d.get("id"), "bytes": os.path.getsize(mp4)})
    return d["id"]


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["id", "status", "start_s", "end_s", "evidence", "reason"],
    "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["on_recording", "not_found", "check"]},
                   "start_s": {"type": "number"}, "end_s": {"type": "number"}, "evidence": {"type": "string"}, "reason": {"type": "string"}}}}}}

PROMPT = """You review a body-worn camera recording against a written account of the same incident.
For each numbered CLAIM, find where the recording shows or says it.
status: on_recording = the video or the speech clearly supports the claim at start_s..end_s;
not_found = the claim is not visible or audible anywhere in this recording (start_s=end_s=-1);
check = the recording shows something that does not match the claim, or support is partial. Never guess: if unsure, use check.
evidence: what is visible or said at that time (quote speech from the TURNS when relevant). Times are seconds from the start of the video.
TURNS (speech with speaker labels and times, from a separate transcriber):
{turns}
CLAIMS:
{claims}"""


def spark(file_id, turns, claims, model="muse-spark-1.3"):
    guard()
    tt = "\n".join(f"[{t['startMs']/1000:.1f}-{t['endMs']/1000:.1f}s] {t.get('speaker','?')}: {t['transcript']}" for t in turns)
    cc = "\n".join(f"{c['id']}. {c['text']}" for c in claims)
    body = {"model": model, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": PROMPT.format(turns=tt or "(no speech detected)", claims=cc)},
        {"type": "input_file", "file_id": file_id}]}],
        "text": {"format": {"type": "json_schema", "name": "claims", "schema": SCHEMA, "strict": True}},
        "max_output_tokens": int(os.environ.get("DC_MAX_OUT", "30000")), "reasoning": {"effort": os.environ.get("DC_EFFORT", "medium")}}
    req = urllib.request.Request(f"{BASE}/responses", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        raw = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:800]}")
    sec = time.time() - t0
    u = raw.get("usage", {})
    pin, pout = PRICE[model]
    ledger("spark", u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout, f"{model} claims")
    log_call({"model": model, "endpoint": "/responses", "sec": round(sec, 2), "usage": u, "status": raw.get("status")})
    text = "".join(c.get("text", "") for o in raw.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
    if not text.strip():
        json.dump(raw, open(os.path.join(HERE, "last-raw.json"), "w"), indent=1)
        raise RuntimeError(f"empty output, status={raw.get('status')} usage={u}")
    return json.loads(text), raw


if __name__ == "__main__":
    mp4, claims_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    claims = [{"id": f"C{i}", "text": l.strip()} for i, l in enumerate(open(claims_path), 1) if l.strip()]
    tr = transcribe(mp4)
    fid = upload(mp4)
    res, raw = spark(fid, tr.get("turns", []), claims)
    json.dump({"claims": claims, "transcribe": tr, "file_id": fid, "result": res, "usage": raw.get("usage")}, open(out, "w"), indent=1)
    print("spent so far $%.4f" % spent())
    for c, r in zip(claims, res["claims"]):
        print(f"{r['id']:>4} {r['status']:12} {r['start_s']:>7}-{r['end_s']:<7} | {c['text'][:70]} | {r['evidence'][:90]}")
