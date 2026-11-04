"""One Muse Spark 1.3 call: the video + diarized speech (from Muse Voice Transcribe, as text) + the claims ->
for each claim: status, start_s, end_s, evidence, reason; plus the camera clock burned into the frames.
The answer is checked afterwards by verify.py; this module only asks and parses."""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .meta import SPARK_MODEL, parse_json_text

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims", "camera_clock"], "properties": {
    "claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "status", "start_s", "end_s", "evidence", "reason"],
        "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["on_recording", "not_found", "check"]},
                       "start_s": {"type": "number"}, "end_s": {"type": "number"},
                       "evidence": {"type": "string"}, "reason": {"type": "string"}}}},
    "camera_clock": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["video_s", "clock", "camera"],
        "properties": {"video_s": {"type": "number"}, "clock": {"type": "string"}, "camera": {"type": "string"}}}}}}

PROMPT = """You review a body-worn camera recording against a written account of the same incident.
For each numbered CLAIM, find where the recording shows or says it.
TIME: our software stamped every frame in the top-left corner with "t=N s" = seconds from the start of this video
(the stamp is not part of the original recording). Read start_s, end_s and every video_s from that stamp; do not estimate time.
status: on_recording = the video or the speech clearly supports the claim at start_s..end_s;
not_found = the recording neither shows nor contradicts the claim (start_s=end_s=-1);
check = the recording contradicts the claim, or shows only part of it in a way that matters (for example the claim says
a man was clinging to a pipe and the recording shows him floating free). If unsure between on_recording and check, use check.
A name, rank or job title that the recording cannot show does not by itself make a claim partial: if the event is shown,
use on_recording and say in reason that the name or rank is not shown.
evidence: what is visible or said at that time. When you quote speech, copy the words exactly from TURNS inside double quotes.
reason: one sentence why this status.
camera_clock: read the date and time the body camera burned into the frame (for example "2026-09-27 21:00:29 -0400  AXON BODY 4 D01A16937",
usually top-right) on the first frame, on the last frame, and wherever the camera id or the clock jumps (the video can join several cameras).
For each reading give video_s (from our stamp), clock exactly as printed (date and time) and camera (the id printed, or "").
If the camera prints no clock, return an empty list.
TURNS (speech with speaker labels and times, from a separate transcriber):
{turns}
CLAIMS (kind in brackets):
{claims}"""


def format_turns(turns: list[dict]) -> str:
    return "\n".join(f"[{t['startMs']/1000:.1f}-{t['endMs']/1000:.1f}s] {t.get('speaker', '?')}: {t.get('transcript', '')}"
                     for t in turns) or "(no speech detected)"


def build_body(file_id: str, turns: list[dict], claims: list[dict]) -> dict:
    cc = "\n".join(f"{c['id']}. [{c.get('kind', 'action')}] {c['text']}" for c in claims)
    return {"model": SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": PROMPT.format(turns=format_turns(turns), claims=cc)},
        {"type": "input_file", "file_id": file_id}]}],
        "text": {"format": {"type": "json_schema", "name": "claims_review", "schema": SCHEMA, "strict": True}},
        "max_output_tokens": int(os.environ.get("DC_MAX_OUT", "30000")),
        "reasoning": {"effort": os.environ.get("DC_EFFORT", "medium")}}


def review(client, file_id: str, turns: list[dict], claims: list[dict], duration_s: float) -> tuple[dict, dict]:
    """Returns (parsed model answer, raw response)."""
    body = build_body(file_id, turns, claims)
    # rough guard estimate: ~280 input tokens per video second + prompt, up to 8k output tokens
    est = (duration_s * 280 + 4000) / 1e6 * 1.25 + 8000 / 1e6 * 4.25
    summary = {"task": "place claims on the recording", "video": f"input_file {file_id} ({duration_s:.0f} s)",
               "turns": len(turns), "claims": len(claims), "effort": body["reasoning"]["effort"],
               "format": "json_schema claims_review (strict)",
               "prompt": body["input"][0]["content"][0]["text"][:700]}
    text, raw = client.responses(body, step="review", summary=summary, estimate=est)
    if not text.strip():
        raise RuntimeError(f"empty answer from {SPARK_MODEL}, status={raw.get('status')} usage={json.dumps(raw.get('usage'))}")
    return parse_json_text(text), raw


# ---------------------------------------------------------------- long recordings: one Muse Spark call per piece
# Tested on clips of up to ~4.3 min with one call per video. A recording longer than LONG_OVER_S is cut into pieces of
# at most PIECE_MAX_S (each piece after the first starts PIECE_OVERLAP_S earlier, so an event on a boundary is in both).
# The stamped copy carries the time of the FULL recording, so every piece answers in absolute seconds.
LONG_OVER_S = float(os.environ.get("DC_LONG_OVER_S", "720"))
PIECE_MAX_S = float(os.environ.get("DC_PIECE_MAX_S", "600"))
PIECE_OVERLAP_S = 15.0
PIECE_TURN_PAD_S = 5.0   # speech turns sent with a piece: those overlapping the piece +- this
PIECE_TOL_S = 5.0        # a timecode more than this outside the piece the model watched is not trusted
MERGE_RANK = {"on_recording": 0, "check": 1, "not_found": 2}

PIECE_TEXT = """PIECE: this file is piece {k} of {n} of a longer recording; it covers t={a} s to t={b} s of the FULL recording.
The stamp shows seconds of the FULL recording, so read start_s, end_s and video_s from the stamp as usual.
Answer every claim from this piece only: if this piece neither shows nor contradicts a claim, use not_found
(another piece may show it). For camera_clock, "first frame" and "last frame" mean the first and last frame of this piece.
"""
_PIECE_AT = "status: on_recording ="
assert _PIECE_AT in PROMPT


def needs_pieces(duration_s: float, over_s: float | None = None) -> bool:
    return duration_s > (LONG_OVER_S if over_s is None else over_s)


def plan_pieces(duration_s: float, max_s: float | None = None, overlap_s: float = PIECE_OVERLAP_S) -> list[tuple[float, float]]:
    """Equal pieces of at most max_s that cover [0, duration]; every piece after the first starts overlap_s earlier."""
    max_s = PIECE_MAX_S if max_s is None else max_s
    if duration_s <= 0:
        return []
    n = max(1, math.ceil(duration_s / max_s - 1e-9))
    step = duration_s / n
    out = []
    for i in range(n):
        a = 0.0 if i == 0 else max(0.0, i * step - overlap_s)
        b = duration_s if i == n - 1 else (i + 1) * step
        out.append((round(a, 3), round(b, 3)))
    return out


def turns_for_piece(turns: list[dict], a: float, b: float, pad: float = PIECE_TURN_PAD_S) -> list[dict]:
    lo, hi = (a - pad) * 1000, (b + pad) * 1000
    return [t for t in turns if t["endMs"] >= lo and t["startMs"] <= hi]


def piece_prompt_text(k: int, n: int, a: float, b: float) -> str:
    return PIECE_TEXT.format(k=k, n=n, a=int(a), b=int(math.ceil(b)))


def piece_max_out(n_claims: int) -> int:
    """Output budget of one piece call. Reasoning shares it, and one answer per claim is needed: 30k (tested on 6-9
    claims) plus 500 per claim above 20, at most 64k. A function of the claims, not of the run, so replay rebuilds the
    same request. DC_MAX_OUT overrides it."""
    if os.environ.get("DC_MAX_OUT"):
        return int(os.environ["DC_MAX_OUT"])
    return max(30000, min(64000, 30000 + 500 * (n_claims - 20)))


def build_piece_body(file_id: str, turns: list[dict], claims: list[dict], k: int, n: int, a: float, b: float) -> dict:
    body = build_body(file_id, turns, claims)
    body["max_output_tokens"] = piece_max_out(len(claims))
    text = body["input"][0]["content"][0]["text"]
    body["input"][0]["content"][0]["text"] = text.replace(_PIECE_AT, piece_prompt_text(k, n, a, b) + _PIECE_AT, 1)
    return body


def piece_estimate(a: float, b: float, max_out: int = 8000) -> float:
    """Worst case of one piece call for the spend guard: ~280 input tokens per video second plus the prompt, and the
    whole output budget (reasoning included)."""
    return ((b - a) * 280 + 4000) / 1e6 * 1.25 + max_out / 1e6 * 4.25


def _retryable(e: Exception) -> bool:
    s = str(e)
    return "HTTP 429" in s or "HTTP 5" in s or "network error" in s


def review_pieces(client, file_ids: list[str], turns: list[dict], claims: list[dict], pieces: list[tuple[float, float]],
                  *, workers: int | None = None, retry_wait_s: float = 20.0) -> list[tuple[dict, dict]]:
    """One Muse Spark 1.3 call per piece (in parallel), each with all claims and the speech turns of that piece.
    Returns [(parsed answer, raw response)] in piece order."""
    n = len(pieces)
    workers = workers or int(os.environ.get("DC_PIECE_WORKERS", "3"))
    max_out = piece_max_out(len(claims))
    # the pieces run in parallel, so the whole batch must fit under the cap, at its worst case, before it starts
    client.guard(sum(piece_estimate(a, b, max_out) for a, b in pieces))

    def one(i: int):
        a, b = pieces[i]
        tt = turns_for_piece(turns, a, b)
        body = build_piece_body(file_ids[i], tt, claims, i + 1, n, a, b)
        summary = {"task": f"place claims on the recording, piece {i + 1} of {n}",
                   "video": f"input_file {file_ids[i]} (piece t={a:.0f}-{b:.0f} s of {pieces[-1][1]:.0f} s)",
                   "turns": len(tt), "claims": len(claims), "effort": body["reasoning"]["effort"],
                   "format": "json_schema claims_review (strict)",
                   "prompt": body["input"][0]["content"][0]["text"][:900]}
        for attempt in range(2):
            try:
                text, raw = client.responses(body, step=f"review-p{i}", summary=summary, estimate=piece_estimate(a, b, max_out))
                break
            except Exception as e:  # noqa: BLE001 - one retry for rate limits, server and network errors only
                if attempt or client.replay or not _retryable(e):
                    raise
                time.sleep(retry_wait_s)
        if not text.strip():
            raise RuntimeError(f"empty answer from {SPARK_MODEL} for piece {i + 1}, status={raw.get('status')}")
        return parse_json_text(text), raw

    with ThreadPoolExecutor(max_workers=max(1, min(workers, n))) as ex:
        return list(ex.map(one, range(n)))


def _hms(s) -> str:
    s = int(round(float(s)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def merge_answers(answers: list[dict], pieces: list[tuple[float, float]], claim_ids: list[str],
                  tol: float = PIECE_TOL_S) -> tuple[dict, dict]:
    """Per-piece answers -> one answer for verify.py, plus what the merge did.
    Per claim the best status wins (on_recording > check > not_found); between equals the earlier timecode, then the
    earlier piece. Merge rule: on_recording with a timecode outside the piece the model watched (+- tol) -> check.
    Findings of the other pieces are kept as notes. Camera clock readings outside their piece are dropped."""
    notes: list[str] = []
    cands: dict[str, list[dict]] = {cid: [] for cid in claim_ids}
    clock: list[dict] = []
    n = len(pieces)
    for k, (ans, (a, b)) in enumerate(zip(answers, pieces)):
        seen = set()
        for x in (ans or {}).get("claims", []) or []:
            if not isinstance(x, dict) or x.get("id") is None:
                continue
            cid = str(x["id"]).strip()
            if cid not in cands:
                notes.append(f"piece {k + 1}: answer for unknown claim {cid} ignored")
                continue
            if cid in seen:
                notes.append(f"piece {k + 1}: duplicate answer for {cid} ignored")
                continue
            seen.add(cid)
            x = dict(x)
            orig = x.get("status")
            change = None
            s, e = x.get("start_s"), x.get("end_s")
            if x.get("status") == "on_recording":
                inside = _num(s) and _num(e) and a - tol <= s <= e <= b + tol
                if not inside:
                    x["status"] = "check"
                    change = (f"piece {k + 1} covers {_hms(a)}-{_hms(b)}, but its answer \"on recording\" points to "
                              f"{s}..{e} s, outside the piece the model watched")
            cands[cid].append({"piece": k, "answer": x, "model_status": orig, "change": change})
        for r in (ans or {}).get("camera_clock", []) or []:
            vs = r.get("video_s") if isinstance(r, dict) else None
            if _num(vs) and a - tol <= vs <= b + tol:
                clock.append(r)
            else:
                notes.append(f"piece {k + 1}: camera clock reading at video_s={vs} is outside the piece, ignored")

    def rank(c):
        st = c["answer"].get("status")
        s = c["answer"].get("start_s")
        return (MERGE_RANK.get(st, 1), s if st != "not_found" and _num(s) and s >= 0 else math.inf, c["piece"])

    merged, per_claim = [], {}
    for cid in claim_ids:
        cs = cands[cid]
        if not cs:
            per_claim[cid] = {"piece": None, "notes": [f"no piece answered this claim ({n} pieces)"], "change": None}
            continue
        best = min(cs, key=rank)
        merged.append(best["answer"])
        pa, pb = pieces[best["piece"]]
        cn = []
        if best["answer"].get("status") == "not_found":
            cn.append(f"not found in any of the {n} pieces")
        for c in sorted(cs, key=rank):
            if c is best or c["answer"].get("status") == "not_found":
                continue
            x = c["answer"]
            when = f"{_hms(x['start_s'])}-{_hms(x['end_s'])}" if _num(x.get("start_s")) and _num(x.get("end_s")) and x["start_s"] >= 0 else "no timecode"
            ev = str(x.get("evidence") or "")
            cn.append(f"piece {c['piece'] + 1} also answered {str(c['model_status']).replace('_', ' ')} at {when}"
                      + (f": {ev[:160]}{'…' if len(ev) > 160 else ''}" if ev else ""))
        found = best["answer"].get("status") != "not_found"  # a claim found nowhere belongs to no piece
        per_claim[cid] = {"piece": best["piece"] + 1 if found else None, "piece_window": [pa, pb] if found else None,
                          "notes": cn, "change": best["change"], "model_status": best["model_status"]}
    uniq = {}
    for r in clock:
        uniq.setdefault((round(float(r["video_s"]), 1), str(r.get("clock", "")).strip()), r)
    merged_answer = {"claims": merged, "camera_clock": [uniq[k] for k in sorted(uniq)]}
    info = {"pieces": [{"index": k + 1, "window": [a, b]} for k, (a, b) in enumerate(pieces)],
            "claims": per_claim, "notes": notes, "rule": f"best of on_recording > check > not_found; ties: earlier timecode; "
                                                         f"on_recording outside its piece (+-{tol:.0f} s) -> check"}
    return merged_answer, info


def apply_merge(checked: dict, info: dict | None) -> dict:
    """After verify.py: attach the piece of each claim and the merge notes; a merge change is written as a code note
    (rule "piece") with the model's original status kept in claim["model"]."""
    if not info:
        return checked
    for c in checked["claims"]:
        m = info["claims"].get(c["id"]) or {}
        c["piece"] = m.get("piece")
        c["merge_notes"] = list(m.get("notes") or [])
        if m.get("change"):
            if c.get("model"):
                c["model"]["status"] = m.get("model_status") or c["model"].get("status")
            c["verify_notes"].insert(0, {"rule": "piece", "detail": m["change"], "changed_status": True})
    s = checked["summary"]
    s["changed_by_code"] = sum(1 for c in checked["claims"] if any(n["changed_status"] for n in c["verify_notes"]))
    checked["notes"] = list(checked.get("notes") or []) + list(info.get("notes") or [])
    return checked


def remerge(raw: dict, claim_ids: list[str]) -> tuple[dict, dict | None]:
    """The model answer to re-check from a saved review-raw.json: for a recording reviewed in pieces, the per-piece
    answers merged again with the current rules; otherwise the saved answer as is."""
    pcs = raw.get("pieces")
    if not pcs:
        return raw["answer"], raw.get("merge")
    return merge_answers([p["answer"] for p in pcs], [tuple(p["window"]) for p in pcs], claim_ids)
