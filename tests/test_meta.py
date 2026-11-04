import json
import os

import pytest

from discovery_clock import meta as M
from conftest import FakeClient, make_client


def test_plan_chunks():
    assert M.plan_chunks(222.35) == [(0.0, 222.35)]
    assert M.plan_chunks(570) == [(0.0, 570.0)]
    assert M.plan_chunks(1300) == [(0.0, 570.0), (570.0, 1140.0), (1140.0, 1300.0)]
    assert M.plan_chunks(0) == []
    assert all(b - a <= 570 for a, b in M.plan_chunks(7200))


def test_merge_turns_shifts_offsets():
    turns = M.merge_turns([(570.0, {"turns": [{"startMs": 1000, "endMs": 2000, "transcript": "b", "speaker": "A"}]}),
                           (0.0, {"turns": [{"startMs": 500, "endMs": 900, "transcript": "a", "speaker": "B"}]})])
    assert [(t["startMs"], t["endMs"], t["transcript"]) for t in turns] == [(500, 900, "a"), (571000, 572000, "b")]


def test_costs():
    assert M.spark_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(5.5)
    assert M.voice_cost(3.6e6) == pytest.approx(0.18)
    assert M.sam_cost(10) == pytest.approx(0.01)  # minimum 50 frames
    assert M.sam_cost(1000) == pytest.approx(0.20)


def test_parse_json_text_fences_and_noise():
    assert M.parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert M.parse_json_text('Here you go: {"a": [1, 2]} done') == {"a": [1, 2]}


def test_canon_replaces_file_ids_with_content_hash(dirs):
    c = make_client(dirs)
    c.file_sha["file-1"] = "abc"
    out = c._canon({"input": [{"content": [{"type": "input_file", "file_id": "file-1"}]}]})
    assert out["input"][0]["content"][0]["file_id"] == "sha256:abc"


def test_record_then_replay_same_answer_no_network(dirs, tiny_video):
    live = make_client(dirs)
    tr = live.transcribe(tiny_video, 3.0)
    fid = live.upload(tiny_video)
    body = {"model": M.SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "x"}, {"type": "input_file", "file_id": fid}]}]}
    text_live, _ = live.responses(body, step="review", summary={"t": 1})
    assert len(live.http) == 3
    recs = sorted(p.name.split("-")[0] for p in (dirs["recordings"] / "t1").glob("*.json"))
    assert recs == ["review", "transcribe", "upload"]

    rp = make_client(dirs, replay=True)
    tr2 = rp.transcribe(tiny_video, 3.0)
    fid2 = rp.upload(tiny_video)
    body2 = json.loads(json.dumps(body).replace(fid, fid2))
    text_rp, _ = rp.responses(body2, step="review", summary={"t": 1})
    assert rp.http == [] and tr2["turns"] == tr["turns"] and text_rp == text_live
    calls = M.read_calls(dirs["runs"], "t1")
    assert [c["mode"] for c in calls] == ["live"] * 3 + ["replay"] * 3
    assert not any(c["replay_changed"] for c in calls)


def test_replay_flags_changed_request(dirs, tiny_video):
    live = make_client(dirs)
    live.responses({"model": M.SPARK_MODEL, "input": "prompt v1"}, step="review", summary={})
    rp = make_client(dirs, replay=True)
    rp.responses({"model": M.SPARK_MODEL, "input": "prompt v2"}, step="review", summary={})
    assert M.read_calls(dirs["runs"], "t1")[-1]["replay_changed"] is True


def test_replay_missing_raises(dirs):
    rp = make_client(dirs, replay=True)
    with pytest.raises(M.ReplayMissing):
        rp.responses({"model": M.SPARK_MODEL, "input": "x"}, step="review", summary={})


def test_spend_cap_stops_live_calls(dirs):
    c = make_client(dirs, cap_usd=0.25)  # the guard runs before each call: 0 -> 0.092 -> 0.184 -> 0.276, then stop
    c.responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})  # 60k in + 4k out = $0.092
    c.responses({"model": M.SPARK_MODEL, "input": "b"}, step="s2", summary={})
    c.responses({"model": M.SPARK_MODEL, "input": "c"}, step="s3", summary={})
    assert c.spent() == pytest.approx(0.276, abs=1e-3)
    with pytest.raises(M.SpendCapReached):
        c.responses({"model": M.SPARK_MODEL, "input": "d"}, step="s4", summary={})
    with pytest.raises(M.SpendCapReached):  # estimate of the next call counts too
        make_client(dirs, cap_usd=0.30).guard(estimate=0.05)  # 0.276 + 0.05 > 0.30
    assert len(c.http) == 3


def test_replay_does_not_spend(dirs):
    make_client(dirs).responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})
    before = make_client(dirs).spent()
    make_client(dirs, replay=True, cap_usd=0.0).responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})
    assert make_client(dirs).spent() == before


def test_key_never_written(dirs, tiny_video):
    c = make_client(dirs)
    c.transcribe(tiny_video, 3.0)
    c.upload(tiny_video)
    c.responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})
    blobs = [p.read_text() for p in list(dirs["runs"].rglob("*.jsonl")) + list(dirs["recordings"].rglob("*.json"))]
    assert blobs and not any("test-key-not-real" in b for b in blobs)


def test_live_without_key_is_clear_error(dirs):
    c = M.MetaClient("t2", runs_dir=dirs["runs"], recordings_dir=dirs["recordings"], ledger_path=dirs["ledger"], replay=False)
    with pytest.raises(M.MetaAPIError, match="DC_REPLAY=1"):
        c.responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})


def test_call_log_excerpt_is_bounded(dirs):
    class Big(FakeClient):
        def _post_json(self, url, body, key, timeout=900):
            return {"status": "completed", "usage": {}, "output": [{"type": "message", "content": [{"text": "x" * 9000}]}]}
    make_client(dirs, cls=Big).responses({"model": M.SPARK_MODEL, "input": "a"}, step="s1", summary={})
    rec = M.read_calls(dirs["runs"], "t1")[-1]
    assert len(rec["response"]) <= M.EXCERPT
