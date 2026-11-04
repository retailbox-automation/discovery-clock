import json

import pytest
from fastapi.testclient import TestClient

from discovery_clock import app as A, pipeline
from conftest import make_client


@pytest.fixture
def web(dirs, tiny_video):
    pipeline.run("case1", tiny_video, "The deputy got in the water.\nA passing boater pulled the man out.\nNo one was there.",
                 claims_mode="lines", client=make_client(dirs, case="case1"), runs_dir=dirs["runs"])
    A.app.state.runs_dir = dirs["runs"]
    yield TestClient(A.app)
    A.app.state.runs_dir = None


def test_index_lists_case(web):
    r = web.get("/")
    assert r.status_code == 200 and "case1" in r.text and "3 claims" in r.text


def test_case_page_shows_statuses_player_clock_and_code_notes(web):
    r = web.get("/cases/case1")
    assert r.status_code == 200
    t = r.text
    assert "On recording · 00:00:00" in t and "Check this claim" in t and "Not found on recording" in t
    assert 'src="/media/case1"' in t and "Camera clock 21:00:29" in t
    assert "Changed by code check</b> (the model said “on recording”)" in t and "quote not in the transcript" in t
    assert "Muse Voice Transcribe" in t and "Muse Spark 1.3" in t
    data = json.loads(t.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert len(data["claims"]) == 3


def test_case_json_cannot_break_out_of_script(web, dirs):
    p = dirs["runs"] / "case1" / "result.json"
    r = json.loads(p.read_text())
    r["claims"][0]["evidence"] = "</script><script>alert(1)</script>"
    p.write_text(json.dumps(r))
    t = web.get("/cases/case1").text
    assert "</script><script>alert(1)" not in t


def test_media_supports_range(web):
    r = web.get("/media/case1", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_calls_api_and_page(web):
    d = web.get("/api/calls").json()
    steps = [c["step"] for c in d["calls"]]
    assert {"review", "upload", "transcribe-0"} <= set(steps)
    assert d["calls"][0]["ts"] >= d["calls"][-1]["ts"]  # newest first
    rv = next(c for c in d["calls"] if c["step"] == "review")
    assert rv["model_name"] == "Muse Spark 1.3" and rv["request"]["prompt"] and rv["response"]
    assert d["total_usd"] > 0
    page = web.get("/calls?case=case1")
    assert page.status_code == 200 and "Live Meta model calls" in page.text and 'class="dark"' in page.text


def test_bad_case_ids(web):
    assert web.get("/cases/..%2F..%2Fetc").status_code == 404
    assert web.get("/cases/nope").status_code == 404
    assert web.get("/api/calls?case=../x").status_code == 400


def test_progress_page_while_running(web, dirs):
    d = dirs["runs"] / "busy"
    d.mkdir()
    (d / "status.json").write_text(json.dumps({"state": "running", "steps": {"claims": {"state": "done", "count": 4},
                                                                          "transcribe": {"state": "running"}}}))
    t = web.get("/cases/busy").text
    assert "Checking the report" in t and 'class="step running"' in t and "4 claims" in t


def test_create_case_rejects_missing_video(web):
    r = web.post("/cases", data={"report": "A man fell.", "mode": "lines"})
    assert r.status_code == 400


def _wait(web, url, tries=100):
    import time
    for _ in range(tries):
        st = web.get(url).json()
        if st.get("state") in ("done", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_follow_person_and_export_clip(web, dirs, monkeypatch):
    from test_track import SamClient
    monkeypatch.setattr(A, "MetaClient", lambda case_id, runs_dir=None: SamClient(
        case_id, runs_dir=dirs["runs"], recordings_dir=dirs["recordings"], ledger_path=dirs["ledger"]))
    assert web.post("/api/cases/case1/track", json={"t": 1.0, "x": 9999, "y": 1}).status_code == 400
    assert web.post("/api/cases/case1/track", json={"t": 1.0, "x": 80, "y": 45}).json()["state"] == "running"
    st = _wait(web, "/api/cases/case1/track")
    assert st["state"] == "done" and st["boxes"], st
    r = web.post("/api/cases/case1/claims/C1/clip")
    assert r.status_code == 200
    st = _wait(web, "/api/cases/case1/claims/C1/clip")
    assert st["state"] == "done", st
    clip = web.get(st["url"])
    assert clip.status_code == 200 and len(clip.content) > 1000
    assert web.post("/api/cases/case1/claims/C3/clip").status_code == 404  # not_found claim has no timecode
    t = web.get("/cases/case1").text
    assert "Follow a person · SAM 3.1" in t and "Export clip · bystander faces blurred" in t
