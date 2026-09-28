"""Compatibility entry point; implementation lives in ``codes.chapters.ch06.aec_crossband_demo``."""

from __future__ import annotations

import importlib as _importlib
import runpy as _runpy
import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys

    _repository_root = str(_Path(__file__).resolve().parents[2])
    if _repository_root not in _sys.path:
        _sys.path.insert(0, _repository_root)
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    _runpy.run_module("codes.chapters.ch06.aec_crossband_demo", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("codes.chapters.ch06.aec_crossband_demo")
