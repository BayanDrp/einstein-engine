#!/usr/bin/env python3
"""Thin shim: works under any python; the heavy render runs inside uv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from visualization.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
