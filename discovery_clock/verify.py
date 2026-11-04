"""Code checks the model. The model's answer is never trusted as is: every rule below can only move a claim
towards "check" (never to "on_recording"), every change is written to verify_notes, and the original answer is kept
next to the final one.

Rules
1. timecode outside the recording (or end before start)          -> check
2. a speech quote in the evidence must be found (fuzzy, >= 0.8 by tokens) in the transcribed turns that overlap
   [start_s - 10, end_s + 10]                                     -> otherwise check
3. on_recording without visual or speech evidence                 -> check
4. a clock time stated in the claim vs the camera clock burned into the frames: more than 10 min apart -> check,
   with both times in the note
5. a claim the model did not answer                               -> check
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher

STATUSES = ("on_recording", "not_found", "check")
TOL_S = 1.0
QUOTE_PAD_S = 10.0
QUOTE_MIN_RATIO = 0.8
CLOCK_MAX_DIFF_MIN = 10.0
MIN_EVIDENCE_CHARS = 12

_DQ = re.compile(r"[\"“”]([^\"“”]+)[\"“”]")
_SQ = re.compile(r"(?:^|(?<=[\s:(\[]))['‘](.+?)['’](?=[\s.,;:!?)\]]|$)")
_TOKEN = re.compile(r"[a-z0-9]+")
_NEGATION = re.compile(r"\b(?:not (?:visible|shown|seen|audible|heard|found)|no (?:visual|audio|video|speech) (?:evidence|support)|"
                       r"no evidence|cannot be (?:seen|confirmed|verified)|never (?:shown|seen|said|heard))\b", re.I)


# ---------------------------------------------------------------- quotes
def norm_tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower().replace("’", "'").replace("'", ""))


def find_quotes(evidence: str) -> list[str]:
    quotes = [m.group(1) for m in _DQ.finditer(evidence or "")]
    if not quotes:
        quotes = [m.group(1) for m in _SQ.finditer(evidence or "")]
    out = []
    for q in quotes:
        for part in re.split(r"\.\.\.|…", q):  # "first part ... second part": each part must be found
            if norm_tokens(part):
                out.append(part.strip())
    return out


def quote_score(quote: str, text: str) -> float:
    q, t = norm_tokens(quote), norm_tokens(text)
    if not q:
        return 1.0
    if not t:
        return 0.0
    best = 0.0
    n = len(q)
    for w in sorted({max(1, n - 1), n, n + 1}):
        for i in range(0, max(1, len(t) - w + 1)):
            r = SequenceMatcher(None, q, t[i: i + w], autojunk=False).ratio()
            if r > best:
                best = r
                if best >= 0.999:
                    return best
    return best


def turns_near(turns: list[dict], start_s: float, end_s: float, pad: float = QUOTE_PAD_S) -> list[dict]:
    a, b = (start_s - pad) * 1000, (end_s + pad) * 1000
    return [t for t in turns if t["endMs"] >= a and t["startMs"] <= b]


def locate_quote(quote: str, turns: list[dict]) -> float | None:
    """Seconds where the quote best matches anywhere in the transcript (for the note), or None."""
    best, at = (0.0, 0.0), None
    for i, t in enumerate(turns):
        pair = quote_score(quote, " ".join(x.get("transcript", "") for x in turns[i: i + 2]))  # quote may span two turns
        key = (round(pair, 3), quote_score(quote, t.get("transcript", "")))
        if key > best:
            best, at = key, t["startMs"] / 1000
    return at if best[0] >= QUOTE_MIN_RATIO else None


# ---------------------------------------------------------------- clocks
_CLOCK_FULL = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T]+(\d{1,2}):(\d{2}):(\d{2})")
_CLOCK_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")


def parse_clock(s: str) -> datetime | None:
    """Camera overlay text -> datetime (date defaults to 2000-01-01 when only a time is printed)."""
    if not s:
        return None
    m = _CLOCK_FULL.search(s)
    try:
        if m:
            y, mo, d, h, mi, se = map(int, m.groups())
            return datetime(y, mo, d, h, mi, se)
        m = _CLOCK_TIME.search(s)
        if m:
            h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            return datetime(2000, 1, 1, h, mi, se)
    except ValueError:
        return None
    return None


_RT_HM = re.compile(r"\b(\d{1,2}):([0-5]\d)(?::[0-5]\d)?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?(?![\w:])", re.I)
_RT_H = re.compile(r"\b(\d{1,2})\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?!\w)", re.I)
_RT_MIL = re.compile(r"\b([01]\d|2[0-3])([0-5]\d)\s*(?:hours|hrs|hr)\b", re.I)


def report_times(text: str) -> list[tuple[str, list[float]]]:
    """Clock times stated in a claim -> [(as written, candidate minutes after midnight)].
    Without a.m./p.m. a 12-hour reading is ambiguous, so both candidates are returned."""
    out, used = [], []

    def add(span, label, cands):
        if any(span[0] < u[1] and u[0] < span[1] for u in used):
            return
        used.append(span)
        out.append((label, cands))

    for m in _RT_MIL.finditer(text):
        add(m.span(), m.group(0), [int(m.group(1)) * 60 + int(m.group(2))])
    for m in _RT_HM.finditer(text):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        if h > 23:
            continue
        if ap:
            if not 1 <= h <= 12:
                continue
            h = h % 12 + (12 if ap.startswith("p") else 0)
            add(m.span(), m.group(0).strip(), [h * 60 + mi])
        elif h > 12 or h == 0:
            add(m.span(), m.group(0).strip(), [h * 60 + mi])
        else:
            add(m.span(), m.group(0).strip(), [(h % 12) * 60 + mi, (h % 12 + 12) * 60 + mi])
    for m in _RT_H.finditer(text):
        h, ap = int(m.group(1)), m.group(2).lower()
        if 1 <= h <= 12:
            add(m.span(), m.group(0).strip(), [(h % 12 + (12 if ap.startswith("p") else 0)) * 60])
    return out


def minutes_apart(a: float, b: float) -> float:
    """Signed difference a - b in minutes on a 24 h dial, in (-720, 720]."""
    d = (a - b) % 1440
    return d - 1440 if d > 720 else d


def clean_readings(readings: list[dict], duration_s: float) -> tuple[list[dict], list[str]]:
    out, notes = [], []
    for r in readings or []:
        dt = parse_clock(str(r.get("clock", "")))
        vs = r.get("video_s")
        if dt is None or not isinstance(vs, (int, float)):
            notes.append(f"camera clock reading ignored (unreadable): {r}")
            continue
        if vs < -TOL_S or vs > duration_s + TOL_S:
            notes.append(f"camera clock reading ignored (video_s {vs} outside recording)")
            continue
        out.append({"video_s": float(vs), "clock": str(r.get("clock", "")).strip(), "camera": str(r.get("camera", "")).strip(), "dt": dt})
    out.sort(key=lambda r: r["video_s"])
    return out, notes


def clock_at(readings: list[dict], t_s: float) -> datetime | None:
    if not readings:
        return None
    ref = readings[0]
    for r in readings:
        if r["video_s"] <= t_s:
            ref = r
    return ref["dt"] + timedelta(seconds=t_s - ref["video_s"])


def _hm(minutes: float) -> str:
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------- main
def _note(c: dict, rule: str, detail: str, to_check: bool):
    c["verify_notes"].append({"rule": rule, "detail": detail, "changed_status": to_check and c["status"] != "check"})
    if to_check:
        c["status"] = "check"


def verify(answer: dict, claims: list[dict], turns: list[dict], duration_s: float) -> dict:
    by_id = {}
    extra_notes = []
    for a in (answer or {}).get("claims", []) or []:
        if isinstance(a, dict) and a.get("id") is not None:
            cid = str(a["id"]).strip()
            if cid in by_id:
                extra_notes.append(f"duplicate answer for {cid} ignored")
                continue
            by_id[cid] = a
    known = {c["id"] for c in claims}
    for cid in by_id:
        if cid not in known:
            extra_notes.append(f"answer for unknown claim {cid} ignored")

    readings, clock_notes = clean_readings((answer or {}).get("camera_clock", []), duration_s)
    extra_notes += clock_notes
    out = []
    for claim in claims:
        a = by_id.get(claim["id"])
        c = {"id": claim["id"], "text": claim["text"], "kind": claim.get("kind", "action"), "line": claim.get("line"),
             "status": "check", "start_s": None, "end_s": None, "evidence": "", "reason": "", "model": None, "verify_notes": []}
        if a is None:
            c["reason"] = "No answer from the model for this claim."
            c["status"] = None
            _note(c, "missing", "the model returned no answer for this claim", True)
            out.append(c)
            continue
        c["model"] = {k: a.get(k) for k in ("status", "start_s", "end_s", "evidence", "reason")}
        c["evidence"] = str(a.get("evidence") or "")
        c["reason"] = str(a.get("reason") or "")
        status = a.get("status")
        if status not in STATUSES:
            c["status"] = None
            _note(c, "status", f"invalid status {status!r}", True)
            status = "check"
        c["status"] = status
        s, e = a.get("start_s"), a.get("end_s")

        if status == "not_found":
            if isinstance(s, (int, float)) and s >= 0:
                c["verify_notes"].append({"rule": "timecode", "detail": f"not_found with a timecode ({s}) - timecode dropped",
                                          "changed_status": False})
        else:
            ok = isinstance(s, (int, float)) and isinstance(e, (int, float)) and -TOL_S <= s <= e <= duration_s + TOL_S
            if not ok:
                _note(c, "timecode", f"timecode outside recording: {s}..{e} s, recording is {duration_s:.1f} s", True)
            else:
                c["start_s"], c["end_s"] = max(0.0, float(s)), min(float(duration_s), float(e))

        # rule 3: on_recording needs evidence
        if c["status"] == "on_recording":
            if len(c["evidence"].strip()) < MIN_EVIDENCE_CHARS:
                _note(c, "evidence", "on_recording without visual or speech evidence", True)
            elif _NEGATION.search(c["evidence"]):
                _note(c, "evidence", f"evidence says the claim is not shown: \"{_NEGATION.search(c['evidence']).group(0)}\"", True)

        # rule 2: quotes must be in the transcript near the timecode
        if c["start_s"] is not None and c["status"] in ("on_recording", "check"):
            near = " ".join(t.get("transcript", "") for t in turns_near(turns, c["start_s"], c["end_s"]))
            for q in find_quotes(c["evidence"]):
                score = quote_score(q, near)
                if score < QUOTE_MIN_RATIO:
                    where = locate_quote(q, turns)
                    hint = f"; found at {where:.1f} s" if where is not None else "; not in the transcript"
                    _note(c, "quote", f"quote not found near timecode: \"{q[:80]}\" (best match {score:.2f} within "
                                      f"{c['start_s'] - QUOTE_PAD_S:.0f}..{c['end_s'] + QUOTE_PAD_S:.0f} s{hint})",
                          c["status"] == "on_recording")

        # rule 4: stated clock time vs camera clock
        times = report_times(claim["text"])
        if times and readings:
            if c["start_s"] is not None:
                cam = clock_at(readings, c["start_s"])
                cam_min = cam.hour * 60 + cam.minute + cam.second / 60
                for label, cands in times:
                    best = min(cands, key=lambda m: abs(minutes_apart(m, cam_min)))
                    d = minutes_apart(best, cam_min)
                    detail = f"report says {label} ({_hm(best)}), camera clock at {c['start_s']:.0f} s reads {cam.strftime('%H:%M:%S')} ({d:+.0f} min)"
                    if abs(d) > CLOCK_MAX_DIFF_MIN:
                        _note(c, "clock", detail, c["status"] == "on_recording")
                    else:
                        c["verify_notes"].append({"rule": "clock", "detail": detail + " - within 10 min", "changed_status": False})
            else:
                a0, a1 = readings[0]["dt"], clock_at(readings, duration_s)
                for label, cands in times:
                    c["verify_notes"].append({"rule": "clock", "detail": f"report says {label}; recording runs "
                                              f"{a0.strftime('%H:%M:%S')} -> {a1.strftime('%H:%M:%S')} by the camera clock",
                                              "changed_status": False})
        elif c["kind"] == "time" and not readings:
            c["verify_notes"].append({"rule": "clock", "detail": "time claim, but no camera clock was read from the frames",
                                      "changed_status": False})
        out.append(c)

    summary = {"total": len(out), **{s: sum(1 for c in out if c["status"] == s) for s in STATUSES},
               "changed_by_code": sum(1 for c in out if any(n["changed_status"] for n in c["verify_notes"]))}
    clock = None
    if readings:
        end_dt = clock_at(readings, duration_s)
        clock = {"start": readings[0]["dt"].strftime("%H:%M:%S"), "end": end_dt.strftime("%H:%M:%S"),
                 "date": readings[0]["dt"].strftime("%Y-%m-%d") if readings[0]["dt"].year != 2000 else "",
                 "readings": [{k: r[k] for k in ("video_s", "clock", "camera")} for r in readings]}
    return {"claims": out, "summary": summary, "camera_clock": clock, "notes": extra_notes}
