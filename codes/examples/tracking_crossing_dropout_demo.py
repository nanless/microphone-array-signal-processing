"""Compatibility alias for the Chapter 9 tracking counterexample."""

if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys

    _repository_root = str(_Path(__file__).resolve().parents[2])
    if _repository_root not in _sys.path:
        _sys.path.insert(0, _repository_root)
    import runpy

    runpy.run_module("codes.chapters.ch09.tracking_crossing_dropout_demo", run_name="__main__")
else:
    import sys
    from codes.chapters.ch09 import tracking_crossing_dropout_demo as _implementation

    sys.modules[__name__] = _implementation
