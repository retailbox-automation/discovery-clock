"""Report text -> atomic claims [{id, text, kind, line}].

Two modes:
- "extract": the model splits a free-text report into atomic checkable claims (Muse Spark Open on Baseten,
  fallback Muse Spark 1.3).
- "lines": one claim per non-empty line, text kept verbatim (used for evaluation keys); the model only labels the kind.
Whatever the model returns is validated by code; if it is unusable we fall back and say so in the result.
"""
from __future__ import annotations

import json
import re

from .meta import SPARK_MODEL, SPARK_OPEN_MODEL, MetaAPIError, ReplayMissing, parse_json_text

KINDS = ["action", "speech", "time", "object"]

_NUM = re.compile(r"^\s*(?:[Cc]?\d{1,3}[.):]\s+|[-*•]\s+)")
_TIME = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b|\b(?:[01]\d|2[0-3])[0-5]\d\s*(?:hours|hrs)\b", re.I)
_SPEECH = re.compile(r"[\"“”]|\b(?:said|says|told|asked|stated|yelled|shouted|responded|replied|admitted|denied|thanked|called out|screamed)\b", re.I)


def split_lines(text: str) -> list[dict]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        t = _NUM.sub("", line).strip()
        if t:
            out.append({"id": f"C{len(out) + 1}", "text": t, "line": n})
    return out


def guess_kind(text: str) -> str:
    if _TIME.search(text):
        return "time"
    if _SPEECH.search(text):
        return "speech"
    return "action"


EXTRACT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "kind", "line"],
                               "properties": {"text": {"type": "string"}, "kind": {"type": "string", "enum": KINDS},
                                              "line": {"type": "integer"}}}}}}

KIND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "kind"],
                               "properties": {"id": {"type": "string"}, "kind": {"type": "string", "enum": KINDS}}}}}}

EXTRACT_PROMPT = """Split this written incident report into atomic, checkable claims, in report order.
One claim = one fact that a body-worn camera recording could show or contradict. Keep the report's wording; do not add facts.
Split sentences that state two separate facts. Skip opinions and legal conclusions that a recording cannot show.
kind: action (someone did something), speech (someone said something), time (a clock time or date is stated), object (a thing, place or condition was present).
line: the 1-based line number of the report where the claim comes from.
Answer with JSON only: {{"claims":[{{"text":"...","kind":"...","line":1}}]}}
REPORT (line numbers in brackets):
{report}"""

KIND_PROMPT = """Label each claim from an incident report with its kind.
kind: action (someone did something), speech (someone said something), time (a clock time or date is stated), object (a thing, place or condition was present).
Keep every id exactly as given. Answer with JSON only: {{"claims":[{{"id":"C1","kind":"action"}}]}}
CLAIMS:
{claims}"""


def _numbered(text: str) -> str:
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(text.splitlines(), 1))


def _validate_extract(data, n_lines: int) -> list[dict]:
    items = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("no claims in model output")
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        kind = it.get("kind") if it.get("kind") in KINDS else guess_kind(text)
        line = it.get("line") if isinstance(it.get("line"), int) and 1 <= it.get("line") <= max(n_lines, 1) else None
        out.append({"id": f"C{len(out) + 1}", "text": text, "kind": kind, "line": line})
    if not out:
        raise ValueError("model output had no usable claims")
    return out


def _apply_kinds(base: list[dict], data) -> list[dict]:
    items = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("no claims in model output")
    kinds = {str(it.get("id")): it.get("kind") for it in items if isinstance(it, dict)}
    if set(kinds) != {c["id"] for c in base}:
        raise ValueError(f"model changed the claim ids: got {sorted(kinds)}")
    return [{**c, "kind": kinds[c["id"]] if kinds[c["id"]] in KINDS else guess_kind(c["text"])} for c in base]


def _ask(client, prompt: str, schema: dict, step: str, summary: dict, use_open: bool):
    if use_open:
        text, _ = client.baseten_chat([{"role": "user", "content": prompt}], step=step, summary=summary, schema=schema)
        return parse_json_text(text), SPARK_OPEN_MODEL
    body = {"model": SPARK_MODEL, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "text": {"format": {"type": "json_schema", "name": "claims", "schema": schema, "strict": True}},
            "max_output_tokens": 16000, "reasoning": {"effort": "low"}}
    text, _ = client.responses(body, step=step, summary=summary, estimate=0.08)
    return parse_json_text(text), SPARK_MODEL


def extract(client, report_text: str, mode: str = "extract", *, use_open: bool = True) -> dict:
    """Returns {"claims": [...], "model": <model that produced them or "code">, "mode": mode, "notes": [...]}."""
    notes = []
    report_text = report_text.replace("\r\n", "\n").strip()
    if mode == "lines":
        base = [{**c, "kind": guess_kind(c["text"])} for c in split_lines(report_text)]
        if not base:
            raise ValueError("the report has no claims")
        prompt = KIND_PROMPT.format(claims="\n".join(f"{c['id']}. {c['text']}" for c in base))
        attempts = ([True] if use_open else []) + [False]
        for opened in attempts:
            step = "claims-kind-open" if opened else "claims-kind"
            try:
                data, model = _ask(client, prompt, KIND_SCHEMA, step,
                                   {"task": "label claim kinds", "claims": len(base), "prompt": prompt[:600]}, opened)
                return {"claims": _apply_kinds(base, data), "model": model, "mode": mode, "notes": notes}
            except (MetaAPIError, ReplayMissing, ValueError, json.JSONDecodeError, KeyError) as e:
                notes.append(f"{SPARK_OPEN_MODEL if opened else SPARK_MODEL}: {str(e)[:200]}")
        notes.append("kinds guessed by code")
        return {"claims": base, "model": "code", "mode": mode, "notes": notes}

    if mode != "extract":
        raise ValueError(f"unknown claims mode {mode!r}")
    n_lines = len(report_text.splitlines())
    prompt = EXTRACT_PROMPT.format(report=_numbered(report_text))
    attempts = ([True] if use_open else []) + [False]
    for opened in attempts:
        step = "claims-open" if opened else "claims"
        try:
            data, model = _ask(client, prompt, EXTRACT_SCHEMA, step,
                               {"task": "split report into claims", "report_chars": len(report_text), "prompt": prompt[:600]}, opened)
            return {"claims": _validate_extract(data, n_lines), "model": model, "mode": mode, "notes": notes}
        except (MetaAPIError, ReplayMissing, ValueError, json.JSONDecodeError, KeyError) as e:
            notes.append(f"{SPARK_OPEN_MODEL if opened else SPARK_MODEL}: {str(e)[:200]}")
    raise RuntimeError("claims could not be extracted: " + " | ".join(notes))
