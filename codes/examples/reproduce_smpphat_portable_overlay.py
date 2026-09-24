"""隔离修复锁定 SMP-PHAT 的负时延索引并与独立 DFT 基线对照。

只在临时目录复制并改写两处上游 C 源码；忽略目录中的锁定工作树保持原样。
此实验是 GPL-3.0 上游程序的两行适配版，不代表原版数值通过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

try:
    from codes.examples import reproduce_smpphat_reference as reference
except ModuleNotFoundError:  # 直接执行文件时，脚本目录在 sys.path[0]。
    import reproduce_smpphat_reference as reference


OLD_LOOKUP = "((unsigned int) roundf(tdoa * (float) interpolation_rate))"
NEW_LOOKUP = "((int) roundf(tdoa * (float) interpolation_rate))"


def patch_source(source: str) -> str:
    """仅把两个负浮点时延到无符号数的转换改为有符号转换。"""
    if source.count(OLD_LOOKUP) != 2:
        raise ValueError("固定版 system.c 的两处目标表达式不匹配；停止适配")
    return source.replace(OLD_LOOKUP, NEW_LOOKUP)


def run(*, source_dir: Path, fftw_prefix: Path) -> dict[str, object]:
    source_dir = source_dir.resolve()
    original_hashes = reference.verify_upstream(source_dir)
    fftw_prefix, fftw_info = reference.validate_fftw_prefix(fftw_prefix)
    with tempfile.TemporaryDirectory(prefix="smpphat-signed-overlay-") as temp:
        root = Path(temp)
        (root / "src").mkdir()
        shutil.copytree(source_dir / "include", root / "include")
        shutil.copy2(source_dir / "src" / "signal.c", root / "src" / "signal.c")
        source_text = (source_dir / "src" / "system.c").read_text(encoding="utf-8")
        patched = patch_source(source_text)
        (root / "src" / "system.c").write_text(patched, encoding="utf-8")
        executable, _command, compiler = reference.compile_harness(
            root, fftw_prefix, root / "build")
        directions = reference.fixed_directions()
        cases = [reference._run_case(executable, root / "inputs", name, geometry,
                                     directions, description)
                 for name, geometry, description in reference.fixed_geometries()]
        patched_sha256 = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    if reference.verify_upstream(source_dir) != original_hashes:
        raise ValueError("运行期间锁定上游源码发生变化")
    for case in cases:
        comparison = case["comparison"]
        if (comparison["intended_signed_lookup_failed"]
                or comparison["expected_equivalence_failed"]
                or comparison["c_srp_vs_c_smp_max_abs"] >= 2e-4):
            raise ArithmeticError(f"有符号索引适配未通过独立对照：{case['case']}")
    return {
        "experiment": "smpphat_signed_delay_overlay",
        "status": "passed_patched_teaching_case",
        "upstream_revision": reference.UPSTREAM_REVISION,
        "upstream_license": "GPL-3.0",
        "source_file_sha256": original_hashes,
        "overlay": {
            "file": "src/system.c", "replacement_count": 2,
            "old_expression": OLD_LOOKUP, "new_expression": NEW_LOOKUP,
            "patched_system_c_sha256": patched_sha256,
            "scope": "temporary copy; original source remains unchanged",
        },
        "dependency": fftw_info,
        "compiler": compiler,
        "cases": [{
            "name": case["case"],
            "pair_count": case["upstream_c"]["pairs_count"],
            "merged_group_count": case["upstream_c"]["groups_count"],
            "true_peak_index": reference.TRUE_INDEX,
            "srp_peak_index": case["upstream_c"]["srp_peak_index"],
            "smp_peak_index": case["upstream_c"]["smp_peak_index"],
            "signed_lookup_mismatch_count": case["lookup_diagnosis"]["srp_lookup_mismatch_count"],
            "srp_max_abs_error": case["comparison"]["c_srp_vs_intended_signed_lookup_max_abs"],
            "smp_max_abs_error": case["comparison"]["c_smp_vs_intended_signed_lookup_max_abs"],
            "srp_vs_smp_max_abs": case["comparison"]["c_srp_vs_c_smp_max_abs"],
        } for case in cases],
        "limitations": [
            "only two deterministic far-field four-microphone synthetic cases",
            "does not validate unmodified upstream, real recordings, or runtime speed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=reference.ROOT / "upstream/_downloads/smpphat")
    parser.add_argument("--fftw-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(source_dir=args.source_dir, fftw_prefix=args.fftw_prefix)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
