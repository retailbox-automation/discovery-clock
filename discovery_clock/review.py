"""One Muse Spark 1.3 call: the video + diarized speech (from Muse Voice Transcribe, as text) + the claims ->
for each claim: status, start_s, end_s, evidence, reason; plus the camera clock burned into the frames.
The answer is checked afterwards by verify.py; this module only asks and parses."""
from __future__ import annotations

import json
import os

from .meta import SPARK_MODEL, parse_json_text

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims", "camera_clock"], "properties": {
    "claims": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "status", "start_s", "end_s", "evidence", "reason"],
        "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["on_recording", "not_found", "check"]},
                       "start_s": {"type": "number"}, "end_s": {"type": "number"},
                       "evidence": {"type": "string"}, "reason": {"type": "string"}}}},
    "camera_clock": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["video_s", "clock", "camera"],
        "properties": {"video_s": {"type": "number"}, "clock": {"type": "string"}, "camera": {"type": "string"}}}}}}

PROMPT = """You review a body-worn camera recording against a written account of the same incident.
For each numbered CLAIM, find where the recording shows or says it.
status: on_recording = the video or the speech clearly supports the claim at start_s..end_s;
not_found = the claim is not visible or audible anywhere in this recording (start_s=end_s=-1);
check = the recording shows something that does not match the claim, or support is only partial. Never guess: if unsure, use check.
evidence: what is visible or said at that time. When you quote speech, copy the words exactly from TURNS inside double quotes.
reason: one sentence why this status. Times are seconds from the start of the video.
camera_clock: read the date and time burned into the frame by the camera (for example "2026-09-27 21:00:29 -0400  AXON BODY 4 D01A16937")
on the first frame, on the last frame, and at each point where the camera id or the clock jumps (the video can join several cameras).
For each reading give video_s, clock exactly as printed (date and time) and camera (the id printed, or ""). If no clock is printed, return an empty list.
TURNS (speech with speaker labels and times, from a separate transcriber):
{turns}
CLAIMS (kind in brackets):
{claims}"""


def format_turns(turns: list[dict]) -> str:
    return "\n".join(f"[{t['startMs']/1000:.1f}-{t['endMs']/1000:.1f}s] {t.get('speaker', '?')}: {t.get('transcript', '')}"
                     for t in turns) or "(no speech detected)"


def build_body(file_id: str, turns: list[dict], claims: list[dict]) -> dict:
    cc = "\n".join(f"{c['id']}. [{c.get('kind', 'action')}] {c['text']}" for c in claims)
    return {"model": SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": PROMPT.format(turns=format_turns(turns), claims=cc)},
        {"type": "input_file", "file_id": file_id}]}],
        "text": {"format": {"type": "json_schema", "name": "claims_review", "schema": SCHEMA, "strict": True}},
        "max_output_tokens": int(os.environ.get("DC_MAX_OUT", "30000")),
        "reasoning": {"effort": os.environ.get("DC_EFFORT", "medium")}}


def review(client, file_id: str, turns: list[dict], claims: list[dict], duration_s: float) -> tuple[dict, dict]:
    """Returns (parsed model answer, raw response)."""
    body = build_body(file_id, turns, claims)
    # rough guard estimate: ~280 input tokens per video second + prompt, up to 8k output tokens
    est = (duration_s * 280 + 4000) / 1e6 * 1.25 + 8000 / 1e6 * 4.25
    summary = {"task": "place claims on the recording", "video": f"input_file {file_id} ({duration_s:.0f} s)",
               "turns": len(turns), "claims": len(claims), "effort": body["reasoning"]["effort"],
               "format": "json_schema claims_review (strict)",
               "prompt": body["input"][0]["content"][0]["text"][:700]}
    text, raw = client.responses(body, step="review", summary=summary, estimate=est)
    if not text.strip():
        raise RuntimeError(f"empty answer from {SPARK_MODEL}, status={raw.get('status')} usage={json.dumps(raw.get('usage'))}")
    return parse_json_text(text), raw
