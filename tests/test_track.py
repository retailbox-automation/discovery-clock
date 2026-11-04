import subprocess

import pytest

from discovery_clock import meta as M, track as T
from conftest import FakeClient, make_client

SAM_TEXT = ("<5f>1<|box;x1=10;y1=20;x2=50;y2=80;w=160;h=90|><|mask;x=0;y=0;data=3,4,!abc|>\n"
            "<3f>1<|box;x1=12;y1=22;x2=52;y2=82;w=160;h=90;c=0.91|><|mask;x=0;y=0;c=0.91;data=3,4,~x~y|>,2<|box;x1=100;y1=5;x2=120;y2=25;w=160;h=90|><|mask;x=0;y=0;data=1,1,!z|>\n")


class SamClient(FakeClient):
    def _post_json(self, url, body, key, timeout=900):
        if body.get("model") == M.SAM_MODEL:
            self.http.append(url)
            self.sam_bodies = getattr(self, "sam_bodies", []) + [body]
            return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": SAM_TEXT}]}]}
        return super()._post_json(url, body, key, timeout)


def test_parse_sam_any_order_and_confidence():
    recs = T.parse_sam(SAM_TEXT)
    assert [(r["frame"], r["id"]) for r in recs] == [(3, 1), (3, 2), (5, 1)]
    assert recs[0]["c"] == pytest.approx(0.91) and recs[2]["c"] is None
    assert recs[2]["box"] == [10, 20, 50, 80] and recs[2]["w"] == 160
    assert T.parse_sam("") == []


def test_to_video_scales_pixels_and_frames():
    out = T.to_video(T.parse_sam(SAM_TEXT), a=100.0, src_w=320, src_h=180)
    first = out[0]
    assert first["t"] == pytest.approx(100.3) and first["box"] == [24, 44, 104, 164]


def test_window_clamps():
    assert T.window(2.0, 222.0, 4, 8) == (0.0, 10.0)
    assert T.window(220.0, 222.0, 4, 8) == (216.0, 222.0)


def test_blur_plan_skips_followed_person_and_adds_margin():
    faces = [{"t": 10.0, "id": 1, "box": [100, 100, 140, 140]}, {"t": 10.0, "id": 2, "box": [400, 100, 440, 140]}]
    person = [{"t": 10.0, "id": 1, "box": [80, 80, 200, 400]}]
    rects = T.blur_plan(faces, person, 10.04, 1280, 720)
    assert rects == [(390, 90, 450, 150)]                     # only the bystander, grown by 25 %
    assert len(T.blur_plan(faces, None, 10.0, 1280, 720)) == 2  # nobody followed: blur every face
    assert T.blur_plan(faces, None, 10.5, 1280, 720) == []      # no SAM frame within 0.1 s


def test_track_records_and_replays(dirs, tiny_video):
    live = make_client(dirs, case="c1", cls=SamClient)
    out = T.track(live, tiny_video, "sha-x", 3.0, t=1.0, x=80, y=45, before=1, after=1)
    body = live.sam_bodies[0]
    prompt = body["input"][0]["content"][0]["text"]
    assert prompt == "<10f>1<|point;x=80;y=45;w=160;h=90|>"  # 1.0 s into a window starting at 0 s, 10 fps
    assert body["metadata"]["propagation_direction"] == "both" and body["store"] is False
    assert [b["id"] for b in out["boxes"]] == [1, 1] and out["window"] == [0.0, 2.0]
    cost = [c for c in M.read_calls(dirs["runs"], "c1") if c["model"] == M.SAM_MODEL][0]["usd"]
    assert cost == pytest.approx(M.sam_cost(20))

    rp = make_client(dirs, case="c1", cls=SamClient, replay=True)
    out2 = T.track(rp, tiny_video, "sha-x", 3.0, t=1.0, x=80, y=45, before=1, after=1)
    assert rp.http == [] and out2["boxes"] == out["boxes"]


def test_export_clip_blurs_only_face_boxes(tiny_video, tmp_path):
    # draw a sharp checkerboard so blur is measurable
    src = tmp_path / "check.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc=s=160x90:d=2:r=10,geq=lum='if(mod(floor(X/4)+floor(Y/4),2),235,16)':cb=128:cr=128",
                    "-f", "lavfi", "-i", "sine=duration=2", "-shortest", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    faces = [{"t": round(0.1 * i, 3), "id": 1, "box": [20, 20, 60, 60]} for i in range(25)]
    out = T.export_clip(src, 0.0, 2.0, faces, tmp_path / "out.mp4")

    def region_var(path, crop):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "1.0", "-i", str(path), "-frames:v", "1", "-vf", f"crop={crop},format=gray",
                              "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        m = sum(raw) / len(raw)
        return sum((b - m) ** 2 for b in raw) / len(raw)
    assert region_var(out, "20:20:30:30") < region_var(src, "20:20:30:30") * 0.3   # face area blurred
    assert region_var(out, "20:20:110:40") > region_var(src, "20:20:110:40") * 0.7  # elsewhere sharp
