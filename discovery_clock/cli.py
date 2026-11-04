"""python -m discovery_clock run|replay|serve"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _print_result(res: dict):
    s = res["summary"]
    print(f"{s['total']} claims checked · {s['on_recording']} on recording · {s['not_found']} not found · "
          f"{s['check']} to check · changed by code {s['changed_by_code']} · spend ${res['spend_usd']:.4f}")
    for c in res["claims"]:
        t = f"{c['start_s']:.1f}-{c['end_s']:.1f}s" if c["start_s"] is not None else "-"
        print(f"{c['id']:>4} {c['status']:12} {t:>14} | {c['text'][:70]}")
        for n in c["verify_notes"]:
            print(f"{'':>6}code: {n['rule']}: {n['detail'][:140]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="discovery_clock")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="check a report against a recording")
    r.add_argument("--case", required=True)
    r.add_argument("--video", required=True)
    r.add_argument("--report", required=True, help="text file with the report")
    r.add_argument("--claims", choices=["extract", "lines"], default="extract",
                   help="extract: the model splits the report; lines: one claim per line (evaluation)")
    r.add_argument("--no-open", action="store_true", help="skip Muse Spark Open on Baseten, use Muse Spark 1.3 for claims")
    p = sub.add_parser("replay", help="re-run a recorded case from recordings/ without network or key")
    p.add_argument("--case", required=True)
    p.add_argument("--video", required=True)
    s = sub.add_parser("serve", help="web interface on 127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)

    from . import pipeline
    if a.cmd == "run":
        res = pipeline.run(a.case, a.video, Path(a.report).read_text(), claims_mode=a.claims, use_open=not a.no_open)
        _print_result(res)
    elif a.cmd == "replay":
        os.environ["DC_REPLAY"] = "1"
        rc = pipeline.recorded_case(a.case)
        res = pipeline.run(a.case, a.video, rc["report"], claims_mode=rc.get("claims_mode", "extract"))
        _print_result(res)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("discovery_clock.app:app", host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
