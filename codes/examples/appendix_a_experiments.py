"""Compatibility alias for the Appendix A experiment implementation."""

if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys

    _repository_root = str(_Path(__file__).resolve().parents[2])
    if _repository_root not in _sys.path:
        _sys.path.insert(0, _repository_root)
    import runpy

    runpy.run_module("codes.chapters.appendix_a.appendix_a_experiments", run_name="__main__")
else:
    import sys
    from codes.chapters.appendix_a import appendix_a_experiments as _implementation

    sys.modules[__name__] = _implementation
