"""Pytest path setup.

The project is laid out with a ``src`` package directory (see ``pyproject.toml``)
and is not necessarily installed into the active environment. Prepending ``src``
to ``sys.path`` lets tests import ``qwen_edit_project`` without an editable
install, while the repository root (added by pytest's rootdir handling) keeps
``scripts`` imports working.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
