"""Compatibility import and CLI for Chapter 1's canonical experiment module."""

import importlib
import runpy
import sys

_TARGET = "codes.chapters.ch01.chapter01_experiments"

if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys

    _repository_root = str(_Path(__file__).resolve().parents[2])
    if _repository_root not in _sys.path:
        _sys.path.insert(0, _repository_root)
    runpy.run_module(_TARGET, run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module(_TARGET)
