"""Fixed meeting-scoring interfaces: original calls and controlled extraction.

No downloads, model inference or challenge recordings. Normal tests inspect the
saved report and independent oracles; --run needs the fixed clean checkouts and
an environment with NumPy/SciPy. Upstream source is never patched or compiled.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import types

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from codes.chapters.ch04.core import upstream_contracts as upstream
from codes.chapters.ch10.examples.run_industrial_interfaces import clean_environment, verify_failure_sources

CODES = Path(__file__).resolve().parents[3]
CURRENT = upstream.ROOT / "codes/chapters/ch11/reports/meeting_scoring_interfaces_current.json"
SOURCES = {
    "meeteval": {
        "revision": "6e3dc81284f2d6928f7ef9e620fd3b6906daa429",
        "files": {
            "LICENSE": "5515c2bdda551fa20d771e99ab31deebbb4f9626bef6f6c25f26125c964fd34c",
            "doc/algorithms.md": "be8a5721286973d737a09c8810028afd76a4ac5d5d287ec4f703e6ed994c3593",
            "meeteval/io/seglst.py": "10d643f3a2784a20f6351e6d549f40da39d4ab4a2481442797f5aa44a484d755",
            "meeteval/wer/wer/cp.py": "2ead616c81b2119e6381532fb337e550c1b12e4ae0ecdfcc96557df32404c1f2",
            "meeteval/wer/wer/di_cp.py": "7db500db70164de2006bced8550fef0ffaa03baf2d760493923de303574f6c53",
            "meeteval/wer/wer/orc.py": "7aac2cfdaeb8fbe975aa5087baaa64bff99a08a6093aaffac014cfdb8b2ce9b7",
            "meeteval/wer/wer/siso.py": "98ae734c9c2554a53dae3c8482a073a319d7c2826bb9efe3985178fc6a30b0c3",
            "meeteval/wer/wer/utils.py": "32cfd8434dfafdcb463220a96609e3ad55b800ee490f9b293f655f562c955963",
            "meeteval/wer/matching/levenshtein.h": "04e013ba7265a6df8f261f793624ba425b57b7beee9ec9b572f4dd4f1e79ce06",
            "setup.py": "4f6d5e2372b7786ec99b8b8f4c0c8079e4297d3f7d0628e07a551c82eff3d32e",
        },
    },
    "chime-utils": {
        "revision": "152882404f572d40769ef02bf91c5a9a9cfc9c78",
        "files": {
            "LICENSE": "e2f48f797fd76ec12f6792c39555e986b8b4329604194e7da1416837b68c8ee4",
            "chime_utils/scoring/meeteval.py": "5b2573e673cf373932bed76bf063432804d92c902177e7af8ca5cba6db3ac1ad",
        },
    },
}
CONFIG = {
    "slot_reuse": {"reference_speakers": ["a", "b", "c"],
                   "hypothesis_slots": ["ac", "b"], "reference_words": 3,
                   "word_encoding": "one character is one word in upstream documentation",
                   "order": "A:a and B:b overlap first; C:c follows and reuses slot 0"},
    "utterance_boundary": {"reference_utterances": ["a", "bc"], "hypothesis_slots": ["ab", "c"]},
    "chime": {"scenarios": ["chime6", "mixer6", "dipco", "notsofar1"],
              "dset_part": "dev", "words": "aaa", "start_time": 0, "end_time": 1,
              "normalizers": ["identity", "remove_first_character"]},
    "multifile": {"allowed_empty_examples_ratio": 0.1,
                  "cases": ["one_missing_of_ten", "one_missing_of_two", "extra_hypothesis", "partial_extra_hypothesis", "empty_reference"]},
    "scope": "synthetic text; no ASR, diarization, audio, challenge data or hardware",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_sha256():
    return hashlib.sha256(json.dumps({"sources": SOURCES, "config": CONFIG},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources(root):
    root = upstream.validate_parent_chain(root)
    identities = {}
    for name, spec in SOURCES.items():
        checkout = root / name
        relatives = list(spec["files"])
        if name == "meeteval":
            package_root = checkout / "meeteval"
            for path in package_root.rglob("*"):
                if path.suffix in {".pyc", ".pyo", ".so", ".pyd", ".dylib"}:
                    raise ValueError("unverified author bytecode/native extension is forbidden: " + str(path))
            # This present source closure is verified before any author import;
            # the worker reports only actual imported files separately.
            relatives.extend(str(p.relative_to(checkout)) for p in sorted(package_root.rglob("*.py")))
        identity = upstream.verify_project(name, checkout, tuple(dict.fromkeys(relatives)))
        for relative, expected in spec["files"].items():
            if identity["used_files"][relative]["sha256"] != expected:
                raise ValueError("source digest mismatch: " + name + "/" + relative)
        identities[name] = identity
    return identities


def call_result(fn):
    try:
        return {"status": "returned", "value": fn()}
    except Exception as exc:
        return {"status": "exception", "exception_type": type(exc).__name__, "exception_message": str(exc)}


@contextmanager
def capture_logging():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger()
    handlers, level = logger.handlers[:], logger.level
    logger.handlers, logger.level = [handler], logging.DEBUG
    try:
        yield stream
    finally:
        logger.handlers, logger.level = handlers, level


def documentation_calls(root):
    """Execute selected original doc functions, not the compiled WER kernels."""
    import numpy as np
    path = root / "meeteval/doc/algorithms.md"
    names = {"update_lev_row", "levenshtein_distance", "cp_lev", "orc_lev_brute_force",
             "orc_lev_dynamic_programming", "di_cp_lev"}
    namespace, lines = {"np": np}, {}
    for match in re.finditer(r"```python\n(.*?)```", path.read_text(), re.S):
        for node in ast.parse(match.group(1)).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                lines[node.name] = path.read_text()[:match.start(1)].count("\n") + node.lineno
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    if set(lines) != names:
        raise ValueError("documentation function selection changed")
    cfg = CONFIG["slot_reuse"]
    ref, hyp = cfg["reference_speakers"], cfg["hypothesis_slots"]
    cp = int(namespace["cp_lev"](ref, hyp))
    orc = int(namespace["orc_lev_brute_force"](ref, hyp))
    boundary = CONFIG["utterance_boundary"]
    return {"execution": "original documentation functions extracted by AST; not production compiled kernels",
            "function_lines": lines, "cp_errors": cp, "orc_errors": orc,
            "cp_rate": cp / cfg["reference_words"], "orc_rate": orc / cfg["reference_words"],
            "orc_indivisible_utterance_errors": int(namespace["orc_lev_brute_force"](
                boundary["reference_utterances"], boundary["hypothesis_slots"])),
            "identity_cp_errors": int(namespace["cp_lev"](["a", "b"], ["b", "a"])),
            "di_cp_documentation_call": call_result(lambda: namespace["di_cp_lev"](ref, hyp))}


def package_calls(root):
    sys.path.insert(0, str((root / "meeteval").resolve()))
    import meeteval
    from meeteval.io.seglst import SegLST, apply_multi_file
    if Path(meeteval.__file__).resolve().parent != (root / "meeteval/meeteval").resolve():
        raise ValueError("unexpected MeetEval package")
    ref = SegLST([{"session_id": "example", "speaker": speaker, "words": word,
                   "start_time": start, "end_time": start + 1}
                  for speaker, word, start in [("A", "a", 0), ("B", "b", 0), ("C", "c", 2)]])
    hyp = SegLST([{"session_id": "example", "speaker": slot, "words": word,
                   "start_time": start, "end_time": start + 1}
                  for slot, word, start in [("slot0", "a", 0), ("slot1", "b", 0), ("slot0", "c", 2)]])
    def serializable_cp():
        import dataclasses
        return dataclasses.asdict(meeteval.wer.cp_word_error_rate(ref, hyp))
    production = call_result(serializable_cp)

    def segments(ids):
        return SegLST([{"session_id": str(i), "speaker": "A", "words": "a"} for i in ids])
    # Original session dispatcher with a deliberately simple callback; no WER
    # kernel is being replaced or evaluated by this callback.
    def counts(r, h):
        return {"reference_segments": len(r), "hypothesis_segments": len(h)}
    cases = [("one_missing_of_ten", range(10), range(9), False),
             ("one_missing_of_two", range(2), range(1), False),
             ("extra_hypothesis", range(1), range(2), False),
             ("partial_extra_hypothesis", range(1), range(2), True),
             ("empty_reference", [], [], False)]
    outputs = []
    for name, rids, hids, partial in cases:
        with capture_logging() as logs:
            result = call_result(lambda: apply_multi_file(counts, segments(rids), segments(hids), partial=partial))
        outputs.append({"case": name, "reference_ids": list(rids), "hypothesis_ids": list(hids),
                        "partial": partial, **result, "logs": logs.getvalue().splitlines()})
    return {"execution": "original package imported from fixed checkout",
            "package_path": str(Path(meeteval.__file__).resolve()),
            "cp_word_error_rate": {"reference": list(ref), "hypothesis": list(hyp), **production},
            "multifile_execution": "original apply_multi_file with segment-count callback; no WER scoring",
            "multifile_cases": outputs}


def chime_controlled_calls(root):
    """Original outer function; fake I/O and fake normalizers isolate control flow."""
    path = root / "chime-utils/chime_utils/scoring/meeteval.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_load_and_prepare")
    scenarios = CONFIG["chime"]["scenarios"]
    records = []
    cases = [("complete", None, None, False, "identity"),
             ("first_reference_missing", "chime6", None, False, "identity"),
             ("first_hypothesis_missing", None, "chime6", False, "identity"),
             ("second_reference_missing", "mixer6", None, False, "identity"),
             ("second_hypothesis_missing", None, "mixer6", False, "identity"),
             ("skip_second_reference", "mixer6", None, True, "identity"),
             ("non_idempotent_normalizer", None, None, False, "remove_first_character")]
    class FakeSegLST(list):
        def map(self, fn):
            return FakeSegLST([fn(dict(segment)) for segment in self])

    for name, missing_ref, missing_hyp, ignore, normalizer in cases:
        class FakePath:
            def __init__(self, text):
                self.text = text
            def __truediv__(self, part):
                return FakePath(self.text + "/" + part)
            def __str__(self):
                return self.text
            def glob(self, pattern):
                return [] if missing_ref and "/" + missing_ref + "/" in self.text else [self]
            def exists(self):
                return not (missing_hyp and self.text.endswith("/" + missing_hyp + ".json"))
        def load(fake_path):
            text = str(fake_path)
            if "/uem/" in text:
                return "fake_uem"
            scenario = next(s for s in scenarios if "/" + s + "/" in text or text.endswith("/" + s + ".json"))
            return FakeSegLST([{"session_id": scenario + "_session", "speaker": "A",
                                "start_time": CONFIG["chime"]["start_time"],
                                "end_time": CONFIG["chime"]["end_time"], "words": CONFIG["chime"]["words"]}])
        normalizer_fn = (lambda text: text) if normalizer == "identity" else (lambda text: text[1:])
        namespace = {"logging": logging, "get_txt_norm": lambda _: normalizer_fn}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        fake_module = types.SimpleNamespace(io=types.SimpleNamespace(SegLST=FakeSegLST, load=load))
        previous = sys.modules.get("meeteval")
        sys.modules["meeteval"] = fake_module
        yielded = []
        try:
            with capture_logging() as logs:
                def consume():
                    for part, scenario, h, r, _ in namespace["_load_and_prepare"](
                            FakePath("hyp"), FakePath("root"), CONFIG["chime"]["dset_part"], normalizer, ignore):
                        yielded.append({"part": part, "scenario": scenario, "reference": list(r), "hypothesis": list(h)})
                    return None
                outcome = call_result(consume)
        finally:
            if previous is None:
                sys.modules.pop("meeteval", None)
            else:
                sys.modules["meeteval"] = previous
        records.append({"case": name, "missing_reference_scenario": missing_ref,
                        "missing_hypothesis_scenario": missing_hyp, "ignore_missing": ignore,
                        "normalizer": normalizer, **outcome, "yielded": yielded,
                        "logs": logs.getvalue().splitlines()})
    return {"execution": "original AST-extracted _load_and_prepare; fake paths, loader, SegLST and normalizer",
            "challenge_pipeline_executed": False, "official_normalizer_executed": False,
            "source_line": node.lineno, "cases": records}


def word_time_calls():
    from meeteval.io import SegLST
    from meeteval.wer.wer.time_constrained import get_pseudo_word_level_timings
    segments = SegLST([{"words": "abc b", "start_time": 0, "end_time": 4}])
    reference = list(get_pseudo_word_level_timings(segments, "character_based"))
    hypothesis = list(get_pseudo_word_level_timings(segments, "character_based_points"))
    # Independent character counts 3:1 split four seconds at t=3.
    expected_ref = [{"words": "abc", "start_time": 0., "end_time": 3.},
                    {"words": "b", "start_time": 3., "end_time": 4.}]
    expected_hyp = [{"words": "abc", "start_time": 1.5, "end_time": 1.5},
                    {"words": "b", "start_time": 3.5, "end_time": 3.5}]
    if reference != expected_ref or hypothesis != expected_hyp:
        raise ValueError("original pseudo-word timing differs from independent 3:1 oracle")
    return {"execution": "original package timing helpers only; no complete tcpWER call",
            "input_segment": list(segments), "reference_strategy": "character_based",
            "hypothesis_strategy": "character_based_points", "reference": reference,
            "hypothesis": hypothesis, "expected_reference": expected_ref,
            "expected_hypothesis": expected_hyp, "matched_independent_expected": True}


def _worker(root):
    if any(name == "meeteval" or name.startswith("meeteval.") for name in sys.modules):
        raise ValueError("preloaded MeetEval modules are forbidden")
    identities = verify_sources(root)
    try:
        docs = documentation_calls(root)
        package = package_calls(root)
        word_times = word_time_calls()
        chime = chime_controlled_calls(root)
        imported = {}
        checkout = Path(identities["meeteval"]["checkout"])
        for name, module in tuple(sys.modules.items()):
            if name != "meeteval" and not name.startswith("meeteval."):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                raise ValueError("author module has no source file: " + name)
            path = upstream.ordinary_file(filename)
            if not path.is_relative_to(checkout):
                raise ValueError("author import outside fixed checkout: " + name)
            relative = path.relative_to(checkout).as_posix()
            if relative not in identities["meeteval"]["used_files"]:
                raise ValueError("author dependency was not preflighted: " + relative)
            imported[name] = {"path": relative, **identities["meeteval"]["used_files"][relative]}
        import numpy as np
        import scipy
        return {"documentation": docs, "package": package, "chime": chime,
                "word_timing": word_times, "actual_imported_original_modules": imported,
                "environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__}}
    finally:
        error = sys.exception()
        if error is not None:
            verify_failure_sources(identities, error)
        else:
            for identity in identities.values():
                upstream.check_unchanged(identity)


def run(root):
    if any(name == "meeteval" or name.startswith("meeteval.") for name in sys.modules):
        raise ValueError("preloaded MeetEval modules are forbidden")
    root = upstream.validate_parent_chain(root)
    identities = verify_sources(root)
    command = [sys.executable, "-B", "-m", __spec__.name if __spec__ is not None else
               "codes.chapters.ch11.examples.audit_meeting_scoring_interfaces", "--worker", "--upstream-root", str(root)]
    try:
        completed = subprocess.run(command, cwd=upstream.ROOT, env=clean_environment(),
                                   capture_output=True, text=True, check=True, timeout=60)
        result = upstream.strict_json_loads(completed.stdout)
    finally:
        error = sys.exception()
        if error is not None:
            verify_failure_sources(identities, error)
        else:
            for identity in identities.values():
                upstream.check_unchanged(identity)
    return {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
            "harness_sha256": sha256(__file__), "source_config_sha256": binding_sha256(),
            "actual_dependencies": upstream.dependencies(__file__, [Path(clean_environment.__code__.co_filename)]),
            "source_identities": identities, "sources": SOURCES, "config": CONFIG,
            "worker": {"command": command, "stderr": completed.stderr,
                       "isolation": "fresh -B Python child; no upstream compilation or installation"},
            **result,
            "static_di_cp": {"execution": "static only; production greedy algorithm not executed",
                             "entrypoint": "meeteval/wer/wer/di_cp.py::greedy_di_cp_word_error_rate",
                             "reference_hypothesis_swapped_for_orc": True,
                             "insertions_deletions_swapped_back": True,
                             "normalization": "original reference word count"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="compatibility flag; the default also runs the limited audit")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--upstream-root", type=Path, default=CODES / "chapters/ch00/upstream/_downloads")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None:
        upstream.report_target(args.output, CURRENT, (upstream.CACHE, args.upstream_root))
    if args.worker:
        if args.output is not None:
            raise ValueError("worker output is stdout only")
        report = _worker(args.upstream_root)
    else:
        report = run(args.upstream_root)
    if args.output is not None:
        upstream.write_report(args.output, report, CURRENT, (upstream.CACHE, args.upstream_root))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
