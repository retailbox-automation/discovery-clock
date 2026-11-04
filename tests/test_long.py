"""Long recordings: pieces, absolute times, merge of the per-piece answers, end-to-end with record/replay. No network."""
import json
import re
import subprocess

import pytest

from discovery_clock import meta as M, pipeline, review as R, stamp
from conftest import FakeClient, make_client


# ---------------------------------------------------------------- planning the pieces
def test_plan_pieces_cover_the_recording_with_overlap():
    d = 2722.4  # 45:22
    p = R.plan_pieces(d)
    assert len(p) == 5 and p[0][0] == 0.0 and p[-1][1] == pytest.approx(d)
    for (a0, b0), (a1, b1) in zip(p, p[1:]):
        assert a1 == pytest.approx(b0 - R.PIECE_OVERLAP_S)  # every boundary is inside two pieces
    assert all(b - a <= R.PIECE_MAX_S + R.PIECE_OVERLAP_S + 1e-6 for a, b in p)
    assert R.plan_pieces(600) == [(0.0, 600)] and R.plan_pieces(0) == []
    assert len(R.plan_pieces(3600)) == 6


def test_needs_pieces_threshold():
    assert not R.needs_pieces(258) and not R.needs_pieces(720) and R.needs_pieces(721)
    assert R.needs_pieces(10, over_s=5)


def test_piece_prompt_names_the_window_and_keeps_the_rest():
    claims = [{"id": "C1", "text": "A man fell.", "kind": "action"}]
    turns = [{"startMs": 10_000, "endMs": 12_000, "speaker": "A", "transcript": "hello"},
             {"startMs": 900_000, "endMs": 901_000, "speaker": "B", "transcript": "far away"}]
    plain = R.build_body("f1", turns, claims)
    piece = R.build_piece_body("f1", R.turns_for_piece(turns, 525.0, 1080.0), claims, 2, 5, 525.0, 1080.0)
    text = piece["input"][0]["content"][0]["text"]
    assert "piece 2 of 5" in text and "t=525 s to t=1080 s of the FULL recording" in text
    assert "far away" in text and "hello" not in text  # only the speech of this piece
    # the single-video prompt is unchanged (recorded requests of the short cases must still match)
    assert "PIECE:" not in plain["input"][0]["content"][0]["text"]
    assert text.replace(R.piece_prompt_text(2, 5, 525.0, 1080.0), "").split("TURNS")[0] == \
        plain["input"][0]["content"][0]["text"].split("TURNS")[0]


def test_piece_max_out_grows_with_the_claims(monkeypatch):
    monkeypatch.delenv("DC_MAX_OUT", raising=False)
    assert R.piece_max_out(9) == 30000 and R.piece_max_out(20) == 30000
    assert R.piece_max_out(74) == 57000 and R.piece_max_out(500) == 64000
    claims = [{"id": f"C{i}", "text": "x", "kind": "action"} for i in range(1, 75)]
    assert R.build_piece_body("f", [], claims, 1, 4, 0, 475)["max_output_tokens"] == 57000
    monkeypatch.setenv("DC_MAX_OUT", "12000")
    assert R.piece_max_out(74) == 12000


def test_turns_for_piece_pad():
    turns = [{"startMs": 1000 * s, "endMs": 1000 * s + 500, "speaker": "A", "transcript": str(s)} for s in (90, 98, 120, 207, 300)]
    got = [t["transcript"] for t in R.turns_for_piece(turns, 100, 200)]
    assert got == ["98", "120"]
    assert [t["transcript"] for t in R.turns_for_piece(turns, 100, 200, pad=10)] == ["90", "98", "120", "207"]


# ---------------------------------------------------------------- merge
PIECES = [(0.0, 600.0), (585.0, 1200.0), (1185.0, 1800.0)]


def ans(*claims, clock=()):
    return {"claims": [dict(zip(("id", "status", "start_s", "end_s", "evidence", "reason"), c)) for c in claims],
            "camera_clock": list(clock)}


def test_merge_best_status_wins_and_ties_go_to_the_earlier_timecode():
    a = [ans(("C1", "not_found", -1, -1, "", "not here"), ("C2", "check", 100, 110, "partly", "r"), ("C3", "on_recording", 590, 598, "seen early", "r")),
         ans(("C1", "on_recording", 700, 712, "deputy opens the door", "r"), ("C2", "not_found", -1, -1, "", "r"), ("C3", "on_recording", 588, 596, "seen", "r")),
         ans(("C1", "check", 1300, 1310, "a different door", "r"), ("C2", "on_recording", 1500, 1520, "the man says it", "r"), ("C3", "not_found", -1, -1, "", "r"))]
    merged, info = R.merge_answers(a, PIECES, ["C1", "C2", "C3", "C4"])
    by = {c["id"]: c for c in merged["claims"]}
    assert by["C1"]["status"] == "on_recording" and by["C1"]["start_s"] == 700 and info["claims"]["C1"]["piece"] == 2
    assert by["C2"]["status"] == "on_recording" and by["C2"]["start_s"] == 1500  # on_recording beats an earlier check
    assert by["C3"]["start_s"] == 588 and info["claims"]["C3"]["piece"] == 2        # same status: the earlier timecode
    assert "C4" not in by and "no piece answered" in info["claims"]["C4"]["notes"][0]
    n1 = info["claims"]["C1"]["notes"]
    assert any("piece 3 also answered check at 00:21:40-00:21:50: a different door" in n for n in n1)
    assert not any("piece 1" in n for n in n1)  # a not_found from another piece is not a finding


def test_merge_not_found_everywhere_is_noted():
    a = [ans(("C1", "not_found", -1, -1, "", "first")), ans(("C1", "not_found", -1, -1, "", "second")), ans()]
    merged, info = R.merge_answers(a, PIECES, ["C1"])
    assert merged["claims"][0]["status"] == "not_found" and merged["claims"][0]["reason"] == "first"
    assert info["claims"]["C1"]["notes"] == ["not found in any of the 3 pieces"]
    assert info["claims"]["C1"]["piece"] is None  # found nowhere: no piece label on the page


def test_merge_on_recording_outside_its_piece_becomes_check():
    a = [ans(("C1", "on_recording", 1400, 1410, "a man waves", "r")), ans(("C1", "not_found", -1, -1, "", "r")), ans()]
    merged, info = R.merge_answers(a, PIECES, ["C1"])
    assert merged["claims"][0]["status"] == "check"
    assert "outside the piece the model watched" in info["claims"]["C1"]["change"]
    # inside the tolerance (key-frame cut can start a piece 2 s early) it stays
    a = [ans(), ans(("C1", "on_recording", 581, 590, "a man waves", "r")), ans()]
    merged, info = R.merge_answers(a, PIECES, ["C1"])
    assert merged["claims"][0]["status"] == "on_recording" and info["claims"]["C1"]["change"] is None


def test_merge_camera_clock_and_bad_answers():
    a = [ans(("C1", "on_recording", 10, 20, "x" * 20, "r"), ("C9", "on_recording", 10, 20, "x", "r"), ("C1", "check", 1, 2, "", ""),
             clock=[{"video_s": 0, "clock": "2026-05-01 21:00:00", "camera": "A"}, {"video_s": 900, "clock": "bad", "camera": ""}]),
         ans(clock=[{"video_s": 590, "clock": "2026-05-01 21:09:50", "camera": "A"}]),
         ans(clock=[{"video_s": 1799, "clock": "2026-05-01 21:29:59", "camera": "A"}])]
    merged, info = R.merge_answers(a, PIECES, ["C1"])
    assert [r["video_s"] for r in merged["camera_clock"]] == [0, 590, 1799]
    assert any("unknown claim C9" in n for n in info["notes"]) and any("duplicate answer for C1" in n for n in info["notes"])
    assert any("video_s=900 is outside the piece" in n for n in info["notes"])


def test_apply_merge_writes_a_code_note_and_keeps_the_model_status():
    from discovery_clock import verify as V
    a = [ans(("C1", "on_recording", 1400, 1410, "a man waves to the deputy", "r")), ans(), ans()]
    merged, info = R.merge_answers(a, PIECES, ["C1"])
    checked = R.apply_merge(V.verify(merged, [{"id": "C1", "text": "A man waved."}], [], 1800.0), info)
    c = checked["claims"][0]
    assert c["status"] == "check" and c["model"]["status"] == "on_recording" and c["piece"] == 1
    assert c["verify_notes"][0]["rule"] == "piece" and c["verify_notes"][0]["changed_status"]
    assert checked["summary"]["changed_by_code"] == 1
    assert R.apply_merge({"claims": [], "summary": {}}, None) == {"claims": [], "summary": {}}


# ---------------------------------------------------------------- absolute times
def test_transcribe_pieces_have_absolute_times(dirs, tiny_video, monkeypatch):
    monkeypatch.setattr(M, "plan_chunks", lambda d, max_s=M.CHUNK_MAX_S: [(0.0, 1.5), (1.5, 3.0)])
    tr = make_client(dirs).transcribe(tiny_video, 3.0)
    assert tr["pieces"] == 2 and [t["startMs"] for t in tr["turns"]] == [100, 1600]  # second piece shifted by 1.5 s


@pytest.fixture(scope="module")
def long_video(tmp_path_factory):
    """20 s, 10 fps, a running frame counter is not needed: the stamp is what we check."""
    p = tmp_path_factory.mktemp("long") / "long.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90:d=20:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-g", "250", "-c:a", "aac", str(p)], check=True)
    return p


def _stamp_px(video, t):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                           "-vf", "crop=96:26:12:12", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                          capture_output=True, check=True).stdout


def _keyframes(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                          "frame=pts_time", "-of", "default=nw=1:nk=1", str(video)], capture_output=True, text=True, check=True).stdout
    return [float(x) for x in out.split()]


def test_cut_piece_is_stream_copy_and_keeps_the_absolute_stamp(long_video, tmp_path):
    stamped = stamp.make_stamped(long_video, tmp_path / "s.mp4", 20.0, max_height=90, keyint_s=2)
    kf = _keyframes(stamped)
    assert len(kf) >= 9 and max(b - a for a, b in zip(kf, kf[1:])) <= 2.2  # a key frame at least every ~2 s
    piece = stamp.cut_piece(stamped, tmp_path / "p.mp4", 8.0, 14.0)
    d = pipeline.probe_duration(piece)
    assert 5.9 <= d <= 8.3  # starts at the key frame at or before 8 s
    start = 14.0 - d
    # stream copy: the piece's frames are the stamped copy's frames, so the burned-in "t=N s" is the full-recording time
    assert _stamp_px(piece, 1.0) == _stamp_px(stamped, start + 1.0)
    assert _stamp_px(piece, 1.0) != _stamp_px(stamped, 1.0)  # control: a different second has a different stamp
    codec = lambda v: subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                                      "-of", "csv=p=0", str(v)], capture_output=True, text=True).stdout.strip()
    assert codec(piece) == codec(stamped)


# ---------------------------------------------------------------- end to end
class PieceClient(FakeClient):
    """Answers per piece, read from the PIECE line of the prompt; claim C1 is in piece 2, C2 in piece 3 (and partly
    in piece 1), C3 nowhere, C4 points outside its piece."""

    def _post_json(self, url, body, key, timeout=900):
        if "baseten" in url:
            return super()._post_json(url, body, key, timeout)
        text = body["input"][0]["content"][0]["text"]
        m = re.search(r"piece (\d+) of (\d+)", text)
        if not m:
            return super()._post_json(url, body, key, timeout)
        k = int(m.group(1))
        nf = lambda cid: {"id": cid, "status": "not_found", "start_s": -1, "end_s": -1, "evidence": "", "reason": "not in this piece"}
        claims = {1: [nf("C1"), {"id": "C2", "status": "check", "start_s": 2.0, "end_s": 3.0, "evidence": "a man stands near a car", "reason": "partly"},
                      nf("C3"), {"id": "C4", "status": "on_recording", "start_s": 15.0, "end_s": 16.0, "evidence": "the deputy waves at the camera", "reason": "r"}],
                  2: [{"id": "C1", "status": "on_recording", "start_s": 9.0, "end_s": 10.5, "evidence": "the deputy opens the car door", "reason": "seen"},
                      nf("C2"), nf("C3"), nf("C4")],
                  3: [nf("C1"), {"id": "C2", "status": "on_recording", "start_s": 16.0, "end_s": 17.0, "evidence": "the man gets out of the car", "reason": "seen"},
                      nf("C3"), nf("C4")]}[k]
        clock = [{"video_s": {1: 0, 2: 7, 3: 13}[k], "clock": f"2026-05-01 21:00:{ {1: 0, 2: 7, 3: 13}[k]:02d}", "camera": "AXON"}]
        self.http.append(url)
        return {"status": "completed", "usage": {"input_tokens": 10000, "output_tokens": 1000},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"claims": claims, "camera_clock": clock})}]}]}


REPORT = "The deputy opened the car door.\nThe man got out of the car.\nA dog barked.\nThe deputy waved."


def test_long_pipeline_end_to_end_then_replay(dirs, long_video):
    client = make_client(dirs, case="long1", cls=PieceClient)
    res = pipeline.run("long1", long_video, REPORT, claims_mode="lines", client=client, runs_dir=dirs["runs"],
                       long_over_s=12, piece_max_s=7)
    pieces = [tuple(p["window"]) for p in res["review_pieces"]]
    assert len(pieces) == 3 and pieces[0][0] == 0 and pieces[-1][1] == pytest.approx(20.0, abs=0.05)
    uploads = [u for u in client.http if u.endswith("/files")]
    reviews = [u for u in client.http if u.endswith("/responses")]
    assert len(uploads) == 3 and len(reviews) == 3 + 1  # three pieces + the kind labels (Spark 1.3 fallback)
    by = {c["id"]: c for c in res["claims"]}
    assert by["C1"]["status"] == "on_recording" and (by["C1"]["start_s"], by["C1"]["end_s"]) == (9.0, 10.5) and by["C1"]["piece"] == 2
    assert by["C2"]["status"] == "on_recording" and by["C2"]["start_s"] == 16.0 and by["C2"]["piece"] == 3
    assert any("piece 1 also answered check" in n for n in by["C2"]["merge_notes"])
    assert by["C3"]["status"] == "not_found" and by["C3"]["merge_notes"] == ["not found in any of the 3 pieces"]
    assert by["C4"]["status"] == "check" and by["C4"]["verify_notes"][0]["rule"] == "piece" and by["C4"]["model"]["status"] == "on_recording"
    assert res["summary"]["changed_by_code"] == 1 and res["camera_clock"]["start"] == "21:00:00"
    assert [r["video_s"] for r in res["camera_clock"]["readings"]] == [0, 7, 13]
    assert set(res["timings_s"]) >= {"claims", "transcribe", "upload", "review", "verify"}
    rec = dirs["recordings"] / "long1"
    assert len(list(rec.glob("upload-p*.json"))) == 3 and len(list(rec.glob("review-p*.json"))) == 3
    assert json.loads((rec / "review-raw.json").read_text())["merge"]["claims"]["C1"]["piece"] == 2
    assert not list((dirs["runs"] / "long1").glob("*.mp4"))  # stamped copy and pieces removed

    # replay: the same statuses and timecodes, no HTTP, no stamped copy rebuilt
    rp = make_client(dirs, case="long1", cls=PieceClient, replay=True)
    res2 = pipeline.run("long1", long_video, REPORT, claims_mode="lines", client=rp, runs_dir=dirs["runs"],
                        long_over_s=12, piece_max_s=7)
    assert rp.http == []
    assert [(c["status"], c["start_s"]) for c in res2["claims"]] == [(c["status"], c["start_s"]) for c in res["claims"]]
    calls = M.read_calls(dirs["runs"], "long1")
    assert not any(c["replay_changed"] for c in calls if c["mode"] == "replay")

    # reverify over the saved merged answer keeps the merge notes
    out = pipeline.reverify("long1", runs_dir=dirs["runs"], recordings_dir=dirs["recordings"])
    assert {c["id"]: c["status"] for c in out["claims"]} == {c["id"]: c["status"] for c in res["claims"]}
    assert out["summary"]["changed_by_code"] == 1 and out["claims"][3]["verify_notes"][0]["rule"] == "piece"


def test_long_pipeline_stops_at_the_cap_before_the_pieces(dirs, long_video):
    with pytest.raises(M.SpendCapReached):
        pipeline.run("long2", long_video, REPORT, claims_mode="lines", client=make_client(dirs, case="long2", cls=PieceClient, cap_usd=0.15),
                     runs_dir=dirs["runs"], long_over_s=12, piece_max_s=7)
    st = json.loads((dirs["runs"] / "long2" / "status.json").read_text())
    assert st["state"] == "failed" and st["steps"]["review"]["state"] == "failed"  # claims + speech fit, the 3 pieces do not
    assert not list((dirs["recordings"] / "long2").glob("review-p*.json"))


def test_case_page_and_calls_for_a_long_case(dirs, long_video):
    from fastapi.testclient import TestClient
    from discovery_clock import app as A
    pipeline.run("long3", long_video, REPORT, claims_mode="lines", client=make_client(dirs, case="long3", cls=PieceClient),
                 runs_dir=dirs["runs"], long_over_s=12, piece_max_s=7)
    p = dirs["runs"] / "long3" / "result.json"
    r = json.loads(p.read_text())  # pretend it is a 45-minute recording: the page must show hours
    r["video"]["duration_s"] = 2722.4
    r["claims"][0]["start_s"], r["claims"][0]["end_s"] = 2405.0, 2410.0
    p.write_text(json.dumps(r))
    A.app.state.runs_dir = dirs["runs"]
    try:
        web = TestClient(A.app)
        t = web.get("/cases/long3").text
        assert "On recording · 00:40:05" in t and "3 calls, one per piece of ≤ 10 min" in t
        assert "Recording 00:45:22" in t and "piece boundary (3 pieces)" in t and 'id="ticks"' in t
        assert '<span class="kind">piece 2</span>' in t and "Other pieces:</b> piece 1 also answered check" in t
        assert "Changed by code check</b> (the model said “on recording”)" in t and "outside the piece the model watched" in t
        data = json.loads(t.split('type="application/json">', 1)[1].split("</script>", 1)[0])
        assert len(data["pieces"]) == 3 and data["duration"] == 2722.4
        calls = web.get("/api/calls?case=long3").json()["calls"]
        assert {"review-p0", "review-p1", "review-p2", "upload-p0", "upload-p2"} <= {c["step"] for c in calls}
        rv = next(c for c in calls if c["step"] == "review-p1")
        assert "piece 2 of 3" in rv["request"]["task"] and "PIECE: this file is piece 2 of 3" in rv["request"]["prompt"]
        assert web.get("/calls?case=long3").status_code == 200
    finally:
        A.app.state.runs_dir = None


def test_transcribe_waits_longer_than_an_upload_for_a_long_piece(dirs, tiny_video, monkeypatch):
    # measured: Voice Transcribe answers after ~0.2 s per audio second and sends nothing meanwhile; a 9:30 piece
    # (~115 s quiet) must not hit curl's 90 s stall limit
    assert M.transcribe_stall_s(570) >= 0.2 * 570 * 2 and M.transcribe_stall_s(570) > 90
    c = make_client(dirs)
    c.transcribe(tiny_video, 3.0)
    c.upload(tiny_video)
    assert c.stalls == [M.transcribe_stall_s(3.0), 90]


def test_make_stamped_can_be_cancelled(long_video, tmp_path):
    import threading
    ev = threading.Event()
    ev.set()
    with pytest.raises(stamp.StampCancelled):
        stamp.make_stamped(long_video, tmp_path / "s.mp4", 20.0, max_height=90, keyint_s=2, cancel=ev)
    assert not (tmp_path / "s.mp4").exists()


def test_failed_long_run_stops_stamping_and_leaves_no_video(dirs, long_video, monkeypatch):
    import threading
    started = threading.Event()
    real = stamp.make_stamped

    def slow(*a, **kw):
        started.set()
        return real(*a, **kw)
    monkeypatch.setattr(stamp, "make_stamped", slow)
    with pytest.raises(M.SpendCapReached):  # fails at the very first call, while the stamped copy is being made
        pipeline.run("long4", long_video, REPORT, claims_mode="lines", client=make_client(dirs, case="long4", cls=PieceClient, cap_usd=0.0),
                     runs_dir=dirs["runs"], long_over_s=12, piece_max_s=7)
    assert started.is_set()
    assert not list((dirs["runs"] / "long4").glob("*.mp4"))
    assert not [t for t in threading.enumerate() if t.name.startswith("stamp-long4") and t.is_alive()]
