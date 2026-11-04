"""Score Discovery Clock runs against the sealed evaluation keys and write eval/RESULTS.md.

    python eval/run.py                  # reads runs/<case>/result.json (falls back to recordings/<case>/result.json)
    python eval/run.py --results runs   # explicit results folder

Labels in eval/cases/<case>/key.json: "account" = a sentence from the agency's official narrative published with the
video; "planted_contradiction" / "planted_invented" = written by us before any model run. Keys were sealed by hash
(eval/SEALED.txt, sha256 of json.dumps(key, sort_keys=True)) before the first model run; this script re-checks the seal.

Metrics
1. planted claims caught: a planted claim must never be marked on_recording (a miss = the tool vouched for a false claim)
2. account claims marked check = "disputed": listed with evidence and timecode for a human to look at
3. share of account claims that got a timecode (on_recording or check with start_s)
Secondary: expected status (account -> on_recording, planted_contradiction -> check, planted_invented -> not_found).
Scores use the current code checks re-run over the saved model answers (runs/<case>/review-raw.json); the
"as run" numbers are listed next to them when the checks changed after a run.
The keys carry no timecodes, so timecode accuracy is checked by eye from the disputed table, not scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"account": "on_recording", "planted_contradiction": "check", "planted_invented": "not_found"}
PART_TOL_S = 2.0  # packet of clips: a timecode this close to its own clip still counts as inside it


def hms(s) -> str:
    if s is None:
        return "-"
    s = int(round(float(s)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def seals(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^(\S+) key sha256 ([0-9a-f]{64})$", line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def load_result(case: str, results: Path | None) -> tuple[dict | None, str]:
    cands = [results / case / "result.json"] if results else [ROOT / "runs" / case / "result.json",
                                                              ROOT / "recordings" / case / "result.json"]
    for p in cands:
        if p.exists():
            return json.loads(p.read_text()), str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return None, ""


def saved_answer(case: str, result_src: Path) -> dict | None:
    """The model's saved review answer for a case: runs/<case>/review-raw.json, else the recorded review response."""
    p = result_src.parent / "review-raw.json"
    if p.exists():
        return json.loads(p.read_text()).get("answer")
    for q in sorted((ROOT / "recordings" / case).glob("review-*.json")):
        raw = json.loads(q.read_text())["response"]
        text = "".join(c.get("text", "") for o in raw.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def saved_raw(result_src: Path) -> dict | None:
    """review-raw.json next to the result (for a long recording it holds the per-piece answers)."""
    p = result_src.parent / "review-raw.json"
    return json.loads(p.read_text()) if p.exists() else None


def reverified(result: dict, answer: dict, raw: dict | None = None) -> dict:
    """Current code checks (and, for a long recording, the current merge of the per-piece answers) over the saved
    model answers (no model calls)."""
    sys.path.insert(0, str(ROOT))
    from discovery_clock import review as R, verify as V
    base = [{k: c.get(k) for k in ("id", "text", "kind", "line")} for c in result["claims"]]
    merge = None
    if raw and raw.get("pieces"):
        answer, merge = R.remerge(raw, [c["id"] for c in base])
    chk = R.apply_merge(V.verify(answer, base, result.get("turns", []), result["video"]["duration_s"]), merge)
    return {**result, "claims": chk["claims"], "summary": chk["summary"], "verify_version": chk["verify_version"]}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_case(case_dir: Path, result: dict) -> dict:
    key = json.loads((case_dir / "key.json").read_text())
    lines = [l.strip() for l in (case_dir / "claims.txt").read_text().splitlines() if l.strip()]
    labels = key["labels"]
    by_id = {c["id"]: c for c in result.get("claims", [])}
    problems = []
    rows = []
    for i, text in enumerate(lines, 1):
        cid = f"C{i}"
        c = by_id.get(cid)
        if c is None:
            problems.append(f"{cid} missing from the result")
            continue
        if norm(c["text"]) != norm(text):
            problems.append(f"{cid} text differs from claims.txt (run with --claims lines)")
        rows.append({"id": cid, "label": labels.get(cid, "?"), "text": text, "status": c["status"], "start_s": c.get("start_s"),
                     "end_s": c.get("end_s"), "evidence": c.get("evidence", ""), "reason": c.get("reason", ""),
                     "code": [n["detail"] for n in c.get("verify_notes", []) if n.get("changed_status")],
                     "part": (key.get("claim_part") or {}).get(cid), "piece": c.get("piece")})
    parts = {p["part"]: p for p in key.get("parts") or []}
    for r in rows:  # a packet of clips: is the claim placed inside its own clip?
        p = parts.get(r["part"])
        if p and r["status"] in ("on_recording", "check") and r["start_s"] is not None:
            r["in_part"] = p["start_s"] - PART_TOL_S <= r["start_s"] <= p["end_s"] + PART_TOL_S
            if not r["in_part"]:
                r["found_in"] = next((q["part"] for q in parts.values() if q["start_s"] <= r["start_s"] < q["end_s"]), "?")
    planted = [r for r in rows if r["label"].startswith("planted")]
    account = [r for r in rows if r["label"] == "account"]
    return {
        "case": case_dir.name, "rows": rows, "problems": problems, "replay": result.get("replay"),
        "planted": len(planted), "planted_caught": sum(r["status"] != "on_recording" for r in planted),
        "planted_missed": [r for r in planted if r["status"] == "on_recording"],
        "account": len(account),
        "account_on": sum(r["status"] == "on_recording" for r in account),
        "account_check": [r for r in account if r["status"] == "check"],
        "account_not_found": [r for r in account if r["status"] == "not_found"],
        "account_timecoded": sum(r["status"] in ("on_recording", "check") and r["start_s"] is not None for r in account),
        "expected_ok": sum(EXPECTED.get(r["label"]) == r["status"] for r in rows), "total": len(rows),
        "changed_by_code": sum(1 for r in rows if r["code"]),
    }


def long_sections(longs: list[dict], short_by_case: dict) -> list[str]:
    """Packet of clips: per clip, and whether each timecoded claim sits inside its own clip; the claims copied from the
    short evaluation cases are compared with their own separate runs."""
    out = []
    cell = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
    for s in longs:
        rows = [r for r in s["rows"] if r.get("part")]
        if not rows:
            continue
        tc = [r for r in rows if "in_part" in r]
        inside = [r for r in tc if r["in_part"]]
        out += [f"## Long input: {s['case']}", "",
                f"**Timecoded claims inside their own clip: {pct(len(inside), len(tc))}.** Every claim belongs to one clip of the "
                f"packet (`claim_part` in the sealed key); a timecode outside that clip (± {PART_TOL_S:.0f} s) means the model "
                "placed it on another clip.", "",
                "| clip | claims | account on recording | account with timecode | planted kept off on recording | timecoded inside own clip |",
                "|---|---|---|---|---|---|"]
        order = []
        for r in rows:
            if r["part"] not in order:
                order.append(r["part"])
        for part in order:
            rr = [r for r in rows if r["part"] == part]
            acc = [r for r in rr if r["label"] == "account"]
            pl = [r for r in rr if r["label"].startswith("planted")]
            t = [r for r in rr if "in_part" in r]
            out.append(f"| {part} | {len(rr)} | {pct(sum(r['status'] == 'on_recording' for r in acc), len(acc))} | "
                       f"{pct(sum(r['status'] in ('on_recording', 'check') and r['start_s'] is not None for r in acc), len(acc))} | "
                       f"{pct(sum(r['status'] != 'on_recording' for r in pl), len(pl))} | {pct(sum(r['in_part'] for r in t), len(t))} |")
        wrong = [r for r in tc if not r["in_part"]]
        out += ["", "Claims placed outside their own clip:", ""]
        out += [f"- {r['id']} ({r['part']}, {r['label']}, {r['status']}) at {hms(r['start_s'])} — in {r.get('found_in', '?')}: "
                f"{cell(r['text'])} — {cell(r['evidence'])[:200]}" for r in wrong] or ["- none"]
        same, diff = 0, []
        key = json.loads((ROOT / "eval" / "cases" / s["case"] / "key.json").read_text())
        for r in rows:
            m = re.match(r"^(\S+) (C\d+) \(sealed", (key.get("claim_sources") or {}).get(r["id"], ""))
            if not m or m.group(1) not in short_by_case:
                continue
            o = next((x for x in short_by_case[m.group(1)]["rows"] if x["id"] == m.group(2)), None)
            if o is None:
                continue
            if o["status"] == r["status"]:
                same += 1
            else:
                diff.append((r, o, m.group(1)))
        n = same + len(diff)
        if n:
            out += ["", f"**Claims of the five evaluation clips, packet run vs their own separate runs: same status {pct(same, n)}.**", "",
                    "| claim | clip | separate run | packet run |", "|---|---|---|---|"]
            out += [f"| {r['id']}. {cell(r['text'])} | {c} {o['id']} | {o['status']} {hms(o['start_s'])} | {r['status']} {hms(r['start_s'])} |"
                    for r, o, c in diff]
        out.append("")
    return out


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f} %)" if b else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=None, help="folder with <case>/result.json (default runs/, then recordings/)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "RESULTS.md")
    ap.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases")
    a = ap.parse_args(argv)

    sealed = seals(a.cases.parent / "SEALED.txt")
    scored, missing, seal_lines = [], [], []
    for case_dir in sorted(p for p in a.cases.iterdir() if (p / "key.json").exists()):
        key = json.loads((case_dir / "key.json").read_text())
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        seal = sealed.get(case_dir.name)
        seal_lines.append(f"| {case_dir.name} | {'ok' if seal == h else ('MISMATCH' if seal else 'not sealed')} | `{h[:16]}…` |")
        res, src = load_result(case_dir.name, a.results)
        if res is None:
            missing.append(case_dir.name)
            continue
        src_path = ROOT / src if not Path(src).is_absolute() else Path(src)
        first = sorted(src_path.parent.glob("result-v*.json"))  # result as first run, kept by `reverify`
        first_res = json.loads(first[0].read_text()) if first else res
        as_run = score_case(case_dir, first_res)
        answer = saved_answer(case_dir.name, src_path)
        s = as_run
        if answer is not None:
            s = score_case(case_dir, reverified(res, answer, saved_raw(src_path)))
        s["source"] = src
        s["long"] = bool(res.get("review_pieces"))
        s["duration_s"] = (res.get("video") or {}).get("duration_s")
        s["pieces"] = len(res.get("review_pieces") or [])
        s["spend_usd"], s["seconds"] = res.get("spend_usd"), res.get("seconds")
        s["as_run"] = as_run
        s["as_run_version"] = first_res.get("verify_version", 1)
        scored.append(s)

    short = [s for s in scored if not s["long"]]   # the totals stay over the short clips; long recordings get their own rows
    longs = [s for s in scored if s["long"]]
    T = lambda k, ss=None: sum(s[k] if isinstance(s[k], int) else len(s[k]) for s in (short if ss is None else ss))
    out = ["# Discovery Clock — evaluation results", "",
           "Generated by `python eval/run.py`. Claims: sentences from the agency's official narrative (*account*) plus claims we "
           "planted before any model run (*planted*). A planted claim must never be marked **on recording**.", "",
           "## Seals", "", "| case | seal | sha256(json.dumps(key, sort_keys=True)) |", "|---|---|---|", *seal_lines, ""]
    if missing:
        out += [f"Not run yet: {', '.join(missing)}", ""]
    if scored:
        out += ["## Summary", "",
                "| case | planted caught | account on recording | account to check (disputed) | account not found | account with timecode | expected status | changed by code | source |",
                "|---|---|---|---|---|---|---|---|---|"]
        row = lambda s, name: (f"| {name} | {pct(s['planted_caught'], s['planted'])} | {pct(s['account_on'], s['account'])} | "
                               f"{len(s['account_check'])} | {len(s['account_not_found'])} | {pct(s['account_timecoded'], s['account'])} | "
                               f"{pct(s['expected_ok'], s['total'])} | {s['changed_by_code']} | `{s['source']}`{' (replay)' if s['replay'] else ''} |")
        for s in short:
            out.append(row(s, s["case"]))
        if short:
            out.append(f"| **all** | **{pct(T('planted_caught'), T('planted'))}** | {pct(T('account_on'), T('account'))} | "
                       f"{T('account_check')} | {T('account_not_found')} | {pct(T('account_timecoded'), T('account'))} | "
                       f"{pct(T('expected_ok'), T('total'))} | {T('changed_by_code')} | |")
        for s in longs:
            out.append(row(s, f"{s['case']} (long recording, {hms(s['duration_s'])}, {s['pieces']} pieces)"))
        out += ["", "**Misses** (planted claim marked on recording) are the failure that matters most: "
                f"{T('planted_missed', scored)} in total.", ""]
        if longs:
            out += ["Long recordings are listed apart from the short clips; the **all** row covers the short clips only. "
                    "A long recording is reviewed in pieces of at most 10 minutes, one Muse Spark 1.3 call per piece, "
                    "and the per-piece answers are merged by code before the checks.", ""]
            for s in longs:
                out.append(f"- {s['case']}: {hms(s['duration_s'])} of video, {s['pieces']} pieces, run took {s['seconds']} s "
                           f"wall-clock, spend ${s['spend_usd']}")
            out.append("")
            out += long_sections(longs, {x["case"]: x for x in short})
        for s in scored:
            for r in s["planted_missed"]:
                out.append(f"- MISS {s['case']} {r['id']} ({r['label']}): {r['text']} — at {hms(r['start_s'])}: {r['evidence']}")
        changed_rows = []
        for s in scored:
            before = {r["id"]: r for r in s["as_run"]["rows"]}
            for r in s["rows"]:
                b = before.get(r["id"])
                if b and (b["status"] != r["status"] or b["start_s"] != r["start_s"]):
                    changed_rows.append((s, b, r))
        out += ["", "## Code-check rule change after run 1", "",
                "**quote rule changed after run 1 (housefire C3).** Run 1 moved a claim to *check* when a quoted line was not "
                "within 10 s of the claim's timecode, even if the line exists elsewhere in the transcript (housefire C3 quoted a "
                "real line at 52.9 s). Rule v2: a quote found anywhere in the transcript keeps the status and widens the claim's "
                "window to that line; only a quote found nowhere (invented) moves the claim to *check*. All numbers above are the "
                "v2 checks re-run over the saved model answers — no new model calls.", "",
                "| case | verify as run | planted caught (as run → v2) | account on recording (as run → v2) | disputed (as run → v2) |",
                "|---|---|---|---|---|"]
        for s in scored:
            ar = s["as_run"]
            out.append(f"| {s['case']} | v{s['as_run_version']} | {ar['planted_caught']}/{ar['planted']} → {s['planted_caught']}/{s['planted']} | "
                       f"{ar['account_on']}/{ar['account']} → {s['account_on']}/{s['account']} | "
                       f"{len(ar['account_check'])} → {len(s['account_check'])} |")
        out += ["", "Claims whose status or timecode changed under v2:", ""]
        out += [f"- {s['case']} {r['id']}: {b['status']} {hms(b['start_s'])} → {r['status']} {hms(r['start_s'])}–{hms(r['end_s'])}"
                for s, b, r in changed_rows] or ["- none"]
        out += ["", "## Disputed account claims (check by eye)", "",
                "Official-narrative sentences the tool marked **check this claim**. Each needs a human look at the timecode: "
                "either the recording really differs from the narrative, or the tool is wrong.", "",
                "| case | claim | timecode | evidence | reason | code check |", "|---|---|---|---|---|---|"]
        cell = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
        for s in scored:
            for r in s["account_check"]:
                out.append(f"| {s['case']} | {r['id']}. {cell(r['text'])} | {hms(r['start_s'])}–{hms(r['end_s'])} | "
                           f"{cell(r['evidence'])} | {cell(r['reason'])} | {cell('; '.join(r['code']))} |")
        out += ["", "## Account claims not found on the recording", "", "| case | claim | reason |", "|---|---|---|"]
        for s in scored:
            for r in s["account_not_found"]:
                out.append(f"| {s['case']} | {r['id']}. {cell(r['text'])} | {cell(r['reason'])} |")
        out += ["", "## All claims", ""]
        for s in scored:
            out += [f"### {s['case']}", "", "| id | label | status | timecode | claim |", "|---|---|---|---|---|"]
            for r in s["rows"]:
                out.append(f"| {r['id']} | {r['label']} | {r['status']} | {hms(r['start_s'])} | {cell(r['text'])} |")
            if s["problems"]:
                out += ["", "Problems: " + "; ".join(s["problems"])]
            out.append("")
    a.out.write_text("\n".join(out) + "\n")
    print(f"wrote {a.out} · cases scored {len(scored)} · not run {len(missing)}")
    if scored:
        print(f"planted caught {pct(T('planted_caught'), T('planted'))} · disputed account {T('account_check')} · "
              f"account with timecode {pct(T('account_timecoded'), T('account'))}")
        for s in longs:
            print(f"long {s['case']}: planted caught {pct(s['planted_caught'], s['planted'])} · disputed account "
                  f"{len(s['account_check'])} · account with timecode {pct(s['account_timecoded'], s['account'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
