"""Read-only AST checks of a fixed TF-Locoformer source snapshot.

Never imports or executes upstream code. Shape replay is restricted to the
explicit assert/transpose/concatenate/Conv2d statements checked below.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "7a615460d347ff7334a13dbb831d16280da72cdc"
EXPECTED = {
    "standalone/tflocoformer_separator.py": "6c01fa3cd005c0da5705ebf5399c48c686aba7c0319820bece0aaefa86cc6aa2",
    "espnet2/enh/separator/tflocoformer_separator.py": "033af99d38e2df367ff64f8e54201f4dfd2abf1ca1c0e0b117f312d7d728876b",
    "egs2/wsj0_2mix/enh1/separate.py": "7e3fe24db3d4da6d0f0ff7f6a29e0129617f14d0a7f8f9a07f482eab95c175d9",
    "egs2/whamr/enh1/conf/tuning/train_enh_tflocoformer.yaml": "379ab49ac27c5776416fa02d08516137358cff4c0206496567f8af59ade9eada",
    "README.md": "6394db55bac96048a2899241e1e75800f2df2641dd51ee9a3c66c4bc7c9085db",
    "LICENSE.md": "0b3204d175388251410569d1e6ae8efc9e7e4ae2572ef11ace4015e0ca8d61b7",
    "LICENSES/Apache-2.0.md": "0b3204d175388251410569d1e6ae8efc9e7e4ae2572ef11ace4015e0ca8d61b7",
}


def _method(tree, class_name, method_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def inspect_separator(source):
    """Extract supported static contracts; refuse an unrecognized shape path."""
    tree = ast.parse(source)
    init = _method(tree, "TFLocoformerSeparator", "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    default = ast.literal_eval(defaults["norm_type"])
    block_init = _method(tree, "LocoformerBlock", "__init__")
    norm_assignment = next(n for n in block_init.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "Norm" for t in n.targets))
    allowed = [ast.literal_eval(key) for key in norm_assignment.value.keys]
    norm_assert = next(n for n in block_init.body if isinstance(n, ast.Assert)
                       and ast.unparse(n.test) == "norm_type in Norm")
    forward = _method(tree, "TFLocoformerSeparator", "forward")
    branch = next(n for n in ast.walk(forward) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "input.ndim == 4")
    assertion = next(n for n in branch.body if isinstance(n, ast.Assert))
    assignment = next(n for n in branch.body if isinstance(n, ast.Assign))
    conv = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "nn.Conv2d")
    cat = next(n for n in ast.walk(forward) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == "torch.cat")
    pattern = (ast.unparse(assertion.test), ast.unparse(assignment.value),
               ast.unparse(cat), ast.literal_eval(conv.args[0]))
    expected = ("input.shape[1] == 1", "input.transpose(1, 2)",
                "torch.cat((batch0.real, batch0.imag), dim=1)", 2)
    if pattern != expected:
        raise ValueError("4D path changed: no shape conclusion available for this source")
    # Replay checked shape operations on dimensions only, not on tensors.
    shape_cases = []
    for shape in ((2, 5, 1, 17), (2, 1, 5, 17), (2, 1, 1, 17)):
        case = {"input": list(shape), "passes_axis1_assertion": shape[1] == 1}
        if shape[1] == 1:
            transposed = list(shape)
            transposed[1], transposed[2] = transposed[2], transposed[1]
            transposed[1] *= 2
            case.update({"real_imag_conv_input": transposed,
                         "matches_conv_channels": transposed[1] == 2})
        shape_cases.append(case)
    return {
        "default_norm": {"value": default, "allowed": allowed,
                         "mismatch": default not in allowed,
                         "default_line": defaults["norm_type"].lineno,
                         "assert_line": norm_assert.lineno},
        "four_dimensional_path": {"assert_line": assertion.lineno,
                                  "transpose_line": assignment.lineno,
                                  "concat_line": cat.lineno,
                                  "conv_line": conv.lineno,
                                  "cases": shape_cases},
        "scope": "static source and dimension replay only; no upstream construction or inference",
    }


def audit(root):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("source revision differs from the audited revision")
    files = {}
    for relative, expected in EXPECTED.items():
        payload = (root / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"source digest differs: {relative}")
        files[relative] = {"sha256": digest}
        if relative.endswith("tflocoformer_separator.py"):
            files[relative]["inspection"] = inspect_separator(payload.decode("utf-8"))
    return {"source": "https://github.com/merlresearch/tf-locoformer",
            "revision": revision, "method": "AST and dimension-only replay",
            "executed_upstream": False, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "upstream/_downloads/tf-locoformer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.source), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
