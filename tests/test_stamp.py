import subprocess

from discovery_clock import pipeline, stamp


def test_render_stamps(tmp_path):
    stamp.render_stamps(3, tmp_path, height=360)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["00000.png", "00001.png", "00002.png", "00003.png"]
    assert stamp.stamp_text(85) == "t=85 s"


def test_make_stamped_keeps_duration_and_draws_top_left(tiny_video, tmp_path):
    out = stamp.make_stamped(tiny_video, tmp_path / "s.mp4", 3.0, max_height=90)
    assert abs(pipeline.probe_duration(out) - 3.0) < 0.2
    # the stamp is yellow text on black in the top-left corner; the source is plain grey
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "1.5", "-i", str(out), "-frames:v", "1", "-vf", "crop=96:26:12:12",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    assert any(r > 150 and g > 130 and b < 90 for r, g, b in px), "no yellow stamp pixels"
    assert any(max(p) < 40 for p in px), "no black stamp box"
