import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_run", Path(__file__).resolve().parent.parent / "eval" / "run.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def make(tmp_path, statuses):
    cases = tmp_path / "eval" / "cases"
    d = cases / "x-abc"
    d.mkdir(parents=True)
    (d / "claims.txt").write_text("A happened.\nB happened.\nC happened.\nD happened.\n")
    key = {"labels": {"C1": "account", "C2": "account", "C3": "planted_contradiction", "C4": "planted_invented"}}
    (d / "key.json").write_text(json.dumps(key, indent=1))
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    (cases.parent / "SEALED.txt").write_text(f"x-abc key sha256 {h}\n")
    res = tmp_path / "res" / "x-abc"
    res.mkdir(parents=True)
    claims = [{"id": f"C{i}", "text": f"{ch} happened.", "status": st, "start_s": None if st == "not_found" else 10.0 * i,
               "end_s": None if st == "not_found" else 10.0 * i + 3, "evidence": "ev", "reason": "r", "verify_notes": []}
              for i, (ch, st) in enumerate(zip("ABCD", statuses), 1)]
    (res / "result.json").write_text(json.dumps({"claims": claims, "replay": False}))
    return cases, tmp_path / "res"


def test_scores_and_writes_results(tmp_path):
    cases, res = make(tmp_path, ["on_recording", "check", "on_recording", "not_found"])
    out = tmp_path / "RESULTS.md"
    E.main(["--cases", str(cases), "--results", str(res), "--out", str(out)])
    t = out.read_text()
    assert "| x-abc | ok |" in t                       # seal verified
    assert "1/2 (50 %)" in t                           # one planted claim caught, one missed
    assert "MISS x-abc C3" in t                        # planted contradiction marked on_recording
    assert "C2. B happened." in t                      # disputed account claim listed for review
    assert "| x-abc | 1/2 (50 %) | 1/2 (50 %) | 1 | 0 | 2/2 (100 %) |" in t


def test_seal_mismatch_and_text_mismatch(tmp_path):
    cases, res = make(tmp_path, ["on_recording", "on_recording", "check", "not_found"])
    (cases.parent / "SEALED.txt").write_text("x-abc key sha256 " + "0" * 64 + "\n")
    r = json.loads((res / "x-abc" / "result.json").read_text())
    r["claims"][0]["text"] = "Something else."
    (res / "x-abc" / "result.json").write_text(json.dumps(r))
    out = tmp_path / "RESULTS.md"
    E.main(["--cases", str(cases), "--results", str(res), "--out", str(out)])
    t = out.read_text()
    assert "MISMATCH" in t and "C1 text differs" in t and "0 in total" in t
