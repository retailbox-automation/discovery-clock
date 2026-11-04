"""Discovery Clock: check each claim of a written incident report against the body-worn camera recording."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("DC_RUNS", ROOT / "runs"))
RECORDINGS = Path(os.environ.get("DC_RECORDINGS", ROOT / "recordings"))

__all__ = ["ROOT", "RUNS", "RECORDINGS"]
