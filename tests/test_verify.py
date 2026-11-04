from discovery_clock import verify as V

TURNS = [
    {"startMs": 780, "endMs": 8140, "speaker": "A", "transcript": "alright, Ocean 20 Central coming on scene whatever it's on scene I've got a buoy with me"},
    {"startMs": 85500, "endMs": 89420, "speaker": "A", "transcript": "Ocean 20 Central I'm getting in the water and I'll be waiting for"},
    {"startMs": 168700, "endMs": 173600, "speaker": "C", "transcript": "Alright we have eyes on 20. He has the patient."},
]
CLAIMS = [{"id": "C1", "text": "Deputies responded to the marina.", "kind": "action"},
          {"id": "C2", "text": "The deputy jumped into the water with a buoy.", "kind": "action"}]
DUR = 222.35


def ans(*items, clock=None):
    return {"claims": list(items), "camera_clock": clock or []}


def a(cid, status="on_recording", s=1.0, e=8.0, ev="Deputy runs onto the dock at night, boats visible", reason="r"):
    return {"id": cid, "status": status, "start_s": s, "end_s": e, "evidence": ev, "reason": reason}


def one(res, cid):
    return next(c for c in res["claims"] if c["id"] == cid)


def test_clean_answer_passes_unchanged():
    res = V.verify(ans(a("C1"), a("C2", s=85.5, e=89.4, ev='A: "I\'m getting in the water" and a green buoy is carried')), CLAIMS, TURNS, DUR)
    assert [c["status"] for c in res["claims"]] == ["on_recording", "on_recording"]
    assert res["summary"]["changed_by_code"] == 0
    assert one(res, "C2")["start_s"] == 85.5


def test_timecode_after_end_goes_to_check():
    res = V.verify(ans(a("C1", s=230, e=240)), CLAIMS[:1], TURNS, DUR)
    c = one(res, "C1")
    assert c["status"] == "check" and c["start_s"] is None
    assert c["verify_notes"][0]["rule"] == "timecode" and "outside recording" in c["verify_notes"][0]["detail"]
    assert c["model"]["status"] == "on_recording"  # original kept


def test_timecode_negative_and_reversed():
    assert one(V.verify(ans(a("C1", s=-5, e=3)), CLAIMS[:1], TURNS, DUR), "C1")["status"] == "check"
    assert one(V.verify(ans(a("C1", s=50, e=40)), CLAIMS[:1], TURNS, DUR), "C1")["status"] == "check"


def test_timecode_boundary_tolerance():
    c = one(V.verify(ans(a("C1", s=0, e=DUR + 0.5)), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "on_recording" and c["end_s"] == DUR


def test_timecode_not_a_number():
    c = one(V.verify(ans(a("C1", s="1:20", e=None)), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "check"


def test_not_found_drops_timecode_without_status_change():
    c = one(V.verify(ans(a("C1", status="not_found", s=-1, e=-1, ev="nothing")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "not_found" and c["start_s"] is None and c["verify_notes"] == []
    c = one(V.verify(ans(a("C1", status="not_found", s=12, e=14, ev="nothing")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "not_found" and c["start_s"] is None and c["verify_notes"][0]["changed_status"] is False


def test_quote_found_near_timecode_fuzzy():
    # transcript says "I'm getting in the water and I'll be waiting for"; quote has small differences
    ev = 'Deputy says "Im getting into the water and Ill be waiting"'
    c = one(V.verify(ans(a("C2", s=86, e=88, ev=ev)), CLAIMS[1:], TURNS, DUR), "C2")
    assert c["status"] == "on_recording", c["verify_notes"]


def test_quote_invented_goes_to_check():
    ev = 'Deputy says "I see the boater pulled him out already"'
    c = one(V.verify(ans(a("C2", s=86, e=88, ev=ev)), CLAIMS[1:], TURNS, DUR), "C2")
    assert c["status"] == "check"
    n = c["verify_notes"][0]
    assert n["rule"] == "quote" and "not in the transcript" in n["detail"] and n["changed_status"]


def test_quote_real_but_far_from_timecode():
    ev = 'Radio: "Alright we have eyes on 20. He has the patient."'
    c = one(V.verify(ans(a("C2", s=10, e=20, ev=ev)), CLAIMS[1:], TURNS, DUR), "C2")
    assert c["status"] == "check"
    assert "found at 168.7 s" in c["verify_notes"][0]["detail"]


def test_quote_within_ten_second_pad_counts():
    ev = '"Alright we have eyes on 20"'
    c = one(V.verify(ans(a("C2", s=176, e=182, ev=ev)), CLAIMS[1:], TURNS, DUR), "C2")  # turn ends 173.6, pad 10 s
    assert c["status"] == "on_recording"
    c = one(V.verify(ans(a("C2", s=185, e=190, ev=ev)), CLAIMS[1:], TURNS, DUR), "C2")  # 11.4 s away
    assert c["status"] == "check"


def test_single_quoted_speech_with_apostrophes():
    ev = "A: 'alright, Ocean 20 Central coming on scene whatever it's on scene I've got a buoy with me' while running"
    assert V.find_quotes(ev) == ["alright, Ocean 20 Central coming on scene whatever it's on scene I've got a buoy with me"]
    c = one(V.verify(ans(a("C1", s=0.8, e=8.1, ev=ev)), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "on_recording"


def test_quote_with_ellipsis_checks_each_part():
    assert V.find_quotes('"coming on scene ... buoy with me"') == ["coming on scene", "buoy with me"]
    ok = one(V.verify(ans(a("C1", ev='He says "coming on scene ... buoy with me"')), CLAIMS[:1], TURNS, DUR), "C1")
    assert ok["status"] == "on_recording"
    bad = one(V.verify(ans(a("C1", ev='He says "coming on scene ... he was drinking all night"')), CLAIMS[:1], TURNS, DUR), "C1")
    assert bad["status"] == "check"


def test_quote_on_check_claim_adds_note_but_stays_check():
    c = one(V.verify(ans(a("C2", status="check", s=86, e=88, ev='"totally made up words here"')), CLAIMS[1:], TURNS, DUR), "C2")
    assert c["status"] == "check" and c["verify_notes"][0]["changed_status"] is False


def test_on_recording_without_evidence():
    c = one(V.verify(ans(a("C1", ev="  ")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "check" and c["verify_notes"][0]["rule"] == "evidence"
    c = one(V.verify(ans(a("C1", ev="seen")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "check"


def test_on_recording_with_negative_evidence():
    c = one(V.verify(ans(a("C1", ev="The fall is not visible anywhere in the video")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "check" and "not shown" in c["verify_notes"][0]["detail"]


def test_missing_and_unknown_and_duplicate_answers():
    res = V.verify(ans(a("C1"), a("C1", status="check"), a("C99")), CLAIMS, TURNS, DUR)
    assert one(res, "C1")["status"] == "on_recording"  # first answer wins
    c2 = one(res, "C2")
    assert c2["status"] == "check" and c2["verify_notes"][0]["rule"] == "missing" and c2["verify_notes"][0]["changed_status"]
    assert any("C99" in n for n in res["notes"]) and any("duplicate" in n for n in res["notes"])


def test_invalid_status():
    c = one(V.verify(ans(a("C1", status="maybe")), CLAIMS[:1], TURNS, DUR), "C1")
    assert c["status"] == "check" and c["verify_notes"][0]["rule"] == "status"


def test_code_never_upgrades():
    for st in ("check", "not_found"):
        c = one(V.verify(ans(a("C1", status=st)), CLAIMS[:1], TURNS, DUR), "C1")
        assert c["status"] == st


# ---------------- clock
CLOCK = [{"video_s": 0, "clock": "2026-09-27 21:00:29 -0400", "camera": "AXON BODY 4 D01A16937"},
         {"video_s": 150, "clock": "2026-09-27 21:04:40 -0400", "camera": "AXON BODY 4 D01AP085E"},
         {"video_s": 222, "clock": "2026-09-27 21:12:02 -0400", "camera": "AXON BODY 4 D01AP085E"}]


def test_report_times_parsing():
    assert V.report_times("At 21:05 deputies arrived")[0][1] == [21 * 60 + 5]
    assert V.report_times("Around 8:55 last night")[0][1] == [8 * 60 + 55, 20 * 60 + 55]
    assert V.report_times("at 9:05 p.m. the man")[0][1] == [21 * 60 + 5]
    assert V.report_times("at 9 PM")[0][1] == [21 * 60]
    assert V.report_times("at 2105 hours")[0][1] == [21 * 60 + 5]
    assert V.report_times("at 12:30 a.m.")[0][1] == [30]
    assert V.report_times("he had 2 dogs") == []
    assert len(V.report_times("between 9:05 PM and 9:20 PM")) == 2


def test_clock_parse_and_interpolate():
    r, notes = V.clean_readings(CLOCK + [{"video_s": 5, "clock": "garbled", "camera": ""}], DUR)
    assert len(r) == 3 and notes
    assert V.clock_at(r, 60).strftime("%H:%M:%S") == "21:01:29"
    assert V.clock_at(r, 160).strftime("%H:%M:%S") == "21:04:50"  # second camera segment


def test_time_claim_within_ten_minutes_ok():
    claims = [{"id": "C1", "text": "Around 8:55 deputies responded to the marina.", "kind": "time"}]
    res = V.verify(ans(a("C1", s=1, e=8), clock=CLOCK), claims, TURNS, DUR)
    c = one(res, "C1")
    assert c["status"] == "on_recording"
    assert "within 10 min" in c["verify_notes"][0]["detail"] and "-6 min" in c["verify_notes"][0]["detail"]
    assert res["camera_clock"]["start"] == "21:00:29" and res["camera_clock"]["date"] == "2026-09-27"


def test_time_claim_more_than_ten_minutes_goes_to_check_with_both_times():
    claims = [{"id": "C1", "text": "At 8:30 PM the deputy jumped into the water.", "kind": "time"}]
    c = one(V.verify(ans(a("C1", s=86, e=89, ev="deputy steps off the dock into the water"), clock=CLOCK), claims, TURNS, DUR), "C1")
    assert c["status"] == "check"
    d = c["verify_notes"][0]["detail"]
    assert "8:30 PM" in d and "21:01:55" in d and "-32 min" in d


def test_time_claim_exactly_ten_minutes_is_ok():
    claims = [{"id": "C1", "text": "At 21:11:55 deputies arrived.", "kind": "time"}]  # clock at 30 s = 21:00:59 -> but use 21:10
    claims = [{"id": "C1", "text": "At 21:10 deputies arrived.", "kind": "time"}]
    c = one(V.verify(ans(a("C1", s=0, e=3), clock=[{"video_s": 0, "clock": "2026-09-27 21:00:00", "camera": ""}]), claims, TURNS, DUR), "C1")
    assert c["status"] == "on_recording"


def test_time_claim_midnight_wrap():
    claims = [{"id": "C1", "text": "At 11:58 PM the call came in.", "kind": "time"}]
    c = one(V.verify(ans(a("C1", s=0, e=3), clock=[{"video_s": 0, "clock": "2026-09-28 00:03:00", "camera": ""}]), claims, TURNS, DUR), "C1")
    assert c["status"] == "on_recording" and "+" not in c["verify_notes"][0]["detail"].split("(")[-1][:1]


def test_time_claim_not_found_gets_note_only():
    claims = [{"id": "C1", "text": "At 7:00 PM the man was drinking.", "kind": "time"}]
    c = one(V.verify(ans(a("C1", status="not_found", s=-1, e=-1, ev="nothing"), clock=CLOCK), claims, TURNS, DUR), "C1")
    assert c["status"] == "not_found" and "recording runs 21:00:29" in c["verify_notes"][0]["detail"]


def test_time_claim_without_camera_clock():
    claims = [{"id": "C1", "text": "Shortly after, deputies arrived.", "kind": "time"}]
    c = one(V.verify(ans(a("C1")), claims, TURNS, DUR), "C1")
    assert c["status"] == "on_recording" and "no camera clock" in c["verify_notes"][0]["detail"]


def test_clock_reading_outside_recording_ignored():
    r, notes = V.clean_readings([{"video_s": 999, "clock": "21:00:00", "camera": ""}], DUR)
    assert r == [] and "outside" in notes[0]


def test_summary_counts():
    res = V.verify(ans(a("C1"), a("C2", status="not_found", s=-1, e=-1, ev="x")), CLAIMS, TURNS, DUR)
    assert res["summary"] == {"total": 2, "on_recording": 1, "not_found": 1, "check": 0, "changed_by_code": 0}
