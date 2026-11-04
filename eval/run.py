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
                     "code": [n["detail"] for n in c.get("verify_notes", []) if n.get("changed_status")]})
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
        s = score_case(case_dir, res)
        s["source"] = src
        scored.append(s)

    T = lambda k: sum(s[k] if isinstance(s[k], int) else len(s[k]) for s in scored)
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
        for s in scored:
            out.append(f"| {s['case']} | {pct(s['planted_caught'], s['planted'])} | {pct(s['account_on'], s['account'])} | "
                       f"{len(s['account_check'])} | {len(s['account_not_found'])} | {pct(s['account_timecoded'], s['account'])} | "
                       f"{pct(s['expected_ok'], s['total'])} | {s['changed_by_code']} | `{s['source']}`{' (replay)' if s['replay'] else ''} |")
        out.append(f"| **all** | **{pct(T('planted_caught'), T('planted'))}** | {pct(T('account_on'), T('account'))} | "
                   f"{T('account_check')} | {T('account_not_found')} | {pct(T('account_timecoded'), T('account'))} | "
                   f"{pct(T('expected_ok'), T('total'))} | {T('changed_by_code')} | |")
        out += ["", "**Misses** (planted claim marked on recording) are the failure that matters most: "
                f"{T('planted_missed')} in total.", ""]
        for s in scored:
            for r in s["planted_missed"]:
                out.append(f"- MISS {s['case']} {r['id']} ({r['label']}): {r['text']} — at {hms(r['start_s'])}: {r['evidence']}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
