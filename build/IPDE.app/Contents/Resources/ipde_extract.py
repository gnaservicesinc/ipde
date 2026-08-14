#!/usr/bin/env python3
"""Source-tree and app-bundle entry point for IPDE."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from ipde.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
