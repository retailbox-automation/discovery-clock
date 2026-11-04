import json
import subprocess

import pytest

from discovery_clock.meta import MetaClient


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "tiny.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90:d=3:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(p)], check=True)
    return p


REVIEW_ANSWER = {"claims": [
    {"id": "C1", "status": "on_recording", "start_s": 0.5, "end_s": 2.0, "evidence": 'Deputy says "getting in the water"', "reason": "said"},
    {"id": "C2", "status": "on_recording", "start_s": 0.5, "end_s": 2.5, "evidence": 'He says "the boater pulled him out"', "reason": "said"},
    {"id": "C3", "status": "not_found", "start_s": -1, "end_s": -1, "evidence": "no mention", "reason": "absent"}],
    "camera_clock": [{"video_s": 0, "clock": "2026-09-27 21:00:29 -0400", "camera": "AXON BODY 4 X"}]}


class FakeClient(MetaClient):
    """Real MetaClient logic (log, ledger, cap, record/replay); only the HTTP layer is canned."""

    def __init__(self, *a, review_answer=None, baseten_fail=False, **kw):
        super().__init__(*a, **kw)
        self.review_answer = review_answer or REVIEW_ANSWER
        self.baseten_fail = baseten_fail
        self.http = []

    def _meta_key(self):
        return "test-key-not-real"

    def _post_multipart(self, url, parts, key, timeout=900, stall_s=90):
        self.http.append(url)
        self.stalls = getattr(self, 'stalls', []) + [stall_s]
        if url.endswith("/files"):
            return {"id": "file-123"}
        return {"audioDurationMs": 3000, "turns": [
            {"startMs": 100, "endMs": 1900, "speaker": "A", "transcript": "I'm getting in the water now"}]}

    def _post_json(self, url, body, key, timeout=900):
        self.http.append(url)
        if "baseten" in url:
            if self.baseten_fail:
                from discovery_clock.meta import MetaAPIError
                raise MetaAPIError("HTTP 503 from baseten")
            msgs = body["messages"][0]["content"]
            ids = [l.split(".")[0] for l in msgs.split("CLAIMS:\n", 1)[1].splitlines() if l.strip()]
            return {"choices": [{"message": {"content": json.dumps({"claims": [{"id": i, "kind": "action"} for i in ids]})}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        inp = body.get("input")
        text = inp[0]["content"][0]["text"] if isinstance(inp, list) else str(inp)
        if "Label each claim" in text:
            ids = [l.split(".")[0] for l in text.split("CLAIMS:\n", 1)[1].splitlines() if l.strip()]
            out = {"claims": [{"id": i, "kind": "speech"} for i in ids]}
        else:
            out = self.review_answer
        return {"status": "completed", "usage": {"input_tokens": 60000, "output_tokens": 4000},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(out)}]}]}


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("DC_REPLAY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    return {"runs": tmp_path / "runs", "recordings": tmp_path / "recordings", "ledger": tmp_path / "runs" / "ledger.jsonl"}


def make_client(dirs, case="t1", cls=FakeClient, **kw):
    return cls(case, runs_dir=dirs["runs"], recordings_dir=dirs["recordings"], ledger_path=dirs["ledger"], **kw)
