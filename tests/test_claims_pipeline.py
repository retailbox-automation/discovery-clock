import json

import pytest

from discovery_clock import claims as C, meta as M, pipeline
from conftest import FakeClient, make_client


def test_split_lines_strips_numbering():
    text = "1. Deputies responded.\n\n C2) The gate was locked.\n- A man fell.\nPlain line"
    out = C.split_lines(text)
    assert [c["text"] for c in out] == ["Deputies responded.", "The gate was locked.", "A man fell.", "Plain line"]
    assert [c["id"] for c in out] == ["C1", "C2", "C3", "C4"] and out[1]["line"] == 3


def test_guess_kind():
    assert C.guess_kind("At 21:05 deputies arrived") == "time"
    assert C.guess_kind('The driver responded "What crash?"') == "speech"
    assert C.guess_kind("The man told deputies he had been drinking") == "speech"
    assert C.guess_kind("A deputy kicked down the front door") == "action"


def test_lines_mode_uses_spark_open_and_keeps_text(dirs):
    c = make_client(dirs)
    os_env = pytest.MonkeyPatch()
    os_env.setenv("BASETEN_API_KEY", "x")
    try:
        r = C.extract(c, "A man fell.\nHe said hi.", "lines")
    finally:
        os_env.undo()
    assert r["model"] == M.SPARK_OPEN_MODEL and [x["text"] for x in r["claims"]] == ["A man fell.", "He said hi."]


def test_lines_mode_falls_back_to_spark_13_then_code(dirs):
    r = C.extract(make_client(dirs), "A man fell.\nHe said hi.", "lines")  # no Baseten key -> Spark 1.3
    assert r["model"] == M.SPARK_MODEL and all(x["kind"] == "speech" for x in r["claims"])
    assert "BASETEN_API_KEY" in r["notes"][0]

    class Broken(FakeClient):
        def _post_json(self, url, body, key, timeout=900):
            return {"output": [{"type": "message", "content": [{"text": '{"claims": [{"id": "X9", "kind": "action"}]}'}]}]}
    r = C.extract(make_client(dirs, case="t3", cls=Broken), "A man fell.\nHe said hi.", "lines")
    assert r["model"] == "code" and [x["kind"] for x in r["claims"]] == ["action", "speech"]
    assert any("changed the claim ids" in n for n in r["notes"])


def test_extract_mode_validates_model_output(dirs):
    class Ext(FakeClient):
        def _post_json(self, url, body, key, timeout=900):
            out = {"claims": [{"text": "Deputies responded", "kind": "action", "line": 1}, {"text": " ", "kind": "x", "line": 1},
                              {"text": "Around 8:55", "kind": "bogus", "line": 99}]}
            return {"output": [{"type": "message", "content": [{"text": json.dumps(out)}]}], "usage": {}}
    r = C.extract(make_client(dirs, cls=Ext), "Deputies responded around 8:55.", "extract")
    assert [x["id"] for x in r["claims"]] == ["C1", "C2"]
    assert r["claims"][1]["kind"] == "time" and r["claims"][1]["line"] is None


def test_pipeline_end_to_end_with_verify(dirs, tiny_video):
    client = make_client(dirs, case="case1")
    report = "The deputy got in the water.\nA passing boater pulled the man out.\nThe man said he had been drinking."
    res = pipeline.run("case1", tiny_video, report, claims_mode="lines", client=client, runs_dir=dirs["runs"])
    st = {c["id"]: c for c in res["claims"]}
    assert st["C1"]["status"] == "on_recording"
    assert st["C2"]["status"] == "check" and st["C2"]["verify_notes"][0]["rule"] == "quote"  # invented quote caught
    assert st["C3"]["status"] == "not_found"
    assert res["summary"]["changed_by_code"] == 1 and res["camera_clock"]["start"] == "21:00:29"
    status = json.loads((dirs["runs"] / "case1" / "status.json").read_text())
    assert status["state"] == "done" and all(s["state"] == "done" for s in status["steps"].values())
    assert (dirs["recordings"] / "case1" / "result.json").exists()
    assert "video_path" not in json.loads((dirs["recordings"] / "case1" / "case.json").read_text())

    # replay of the recorded case gives the same statuses without any HTTP
    rp = make_client(dirs, case="case1", replay=True)
    res2 = pipeline.run("case1", tiny_video, report, claims_mode="lines", client=rp, runs_dir=dirs["runs"])
    assert rp.http == [] and [c["status"] for c in res2["claims"]] == [c["status"] for c in res["claims"]]
    assert res2["replay"] is True


def test_pipeline_failure_is_written_to_status(dirs, tiny_video):
    class Fail(FakeClient):
        def _post_multipart(self, url, parts, key, timeout=900):
            raise M.MetaAPIError("HTTP 500 from transcribe")
    with pytest.raises(M.MetaAPIError):
        pipeline.run("bad", tiny_video, "A man fell.", claims_mode="lines", client=make_client(dirs, case="bad", cls=Fail),
                     runs_dir=dirs["runs"])
    st = json.loads((dirs["runs"] / "bad" / "status.json").read_text())
    assert st["state"] == "failed" and st["steps"]["transcribe"]["state"] == "failed"


def test_reverify_uses_saved_answer_without_calls(dirs, tiny_video):
    client = make_client(dirs, case="rv")
    pipeline.run("rv", tiny_video, "The deputy got in the water.\nA boater pulled him out.\nNo one.", claims_mode="lines",
                 client=client, runs_dir=dirs["runs"])
    p = dirs["runs"] / "rv" / "result.json"
    r = json.loads(p.read_text())
    r["verify_version"] = 1
    r["claims"][0]["status"] = "check"  # pretend the old rule had downgraded it
    p.write_text(json.dumps(r))
    n_http = len(client.http)
    res = pipeline.reverify("rv", runs_dir=dirs["runs"], recordings_dir=dirs["recordings"])
    assert len(client.http) == n_http
    assert res["claims"][0]["status"] == "on_recording" and res["verify_version"] == 2
    assert json.loads((dirs["runs"] / "rv" / "result-v1.json").read_text())["claims"][0]["status"] == "check"
    assert (dirs["recordings"] / "rv" / "result-v1.json").exists()
