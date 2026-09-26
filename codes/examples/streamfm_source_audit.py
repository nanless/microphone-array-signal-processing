"""Read-only AST evidence for two Stream.FM state-interface concerns.

Does not import/execute upstream code, download weights or test inference.
This intentionally narrow audit is not a general Python attribute checker.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

COMMIT = 'ab2700c1154acc5c2ce67a5344182028336413f5'
SOURCE_SHA256 = 'd88c7bb526a97ae11d3cdede0e03e81b7c6e38cd2c6ad92296a37f974668d80a'
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'upstream/_downloads/streamfm/sgmse/backbones/streaming_unet.py'


def inspect_source(text: str) -> dict:
    """Report direct tuple contracts and local self-attribute assignments.

    Dynamic/inherited attributes and call reachability are deliberately outside
    the evidence. Nested classes or functions are not treated as class methods.
    """
    tree = ast.parse(text)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    def methods(name):
        return {node.name: node for node in classes[name].body if isinstance(node, ast.FunctionDef)}
    block = methods('CausalResnetBlockBigGANpp')
    initial = [{'line': n.lineno, 'arity': len(n.value.elts)}
               for n in ast.walk(block['init_state'])
               if isinstance(n, ast.Return) and isinstance(n.value, (ast.Tuple, ast.List))]
    unpack = [{'line': n.lineno, 'arity': len(target.elts)}
              for n in ast.walk(block['forward_step']) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Name) and n.value.id == 'state'
              for target in n.targets if isinstance(target, (ast.Tuple, ast.List))]
    conv = methods('CausalConv2d')
    assignments = sorted({n.attr for method in conv.values() for n in ast.walk(method)
                          if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                          and isinstance(n.value, ast.Name) and n.value.id == 'self'})
    reads = [{'attribute': n.attr, 'line': n.lineno}
             for n in ast.walk(conv['forward_step'])
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
             and isinstance(n.value, ast.Name) and n.value.id == 'self'
             and n.attr in {'depthwise_separable', 'pointwise_conv'}]
    return {
        'resnet_init_tuple_returns': initial,
        'resnet_step_state_unpacks': unpack,
        'resnet_direct_contract_mismatch': bool(initial and unpack and
              any(left['arity'] != right['arity'] for left in initial for right in unpack)),
        'conv_step_selected_attribute_reads': reads,
        'conv_local_assigned_attributes': assignments,
        'conv_selected_reads_without_local_assignment': sorted({item['attribute'] for item in reads}-set(assignments)),
        'scope': 'static AST only; no inference; inherited/dynamic assignments and reachability not established',
    }


def audit_file(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('source differs from audited fixed-commit file; review before using this report')
    return {'upstream_commit': COMMIT, 'relative_source': 'sgmse/backbones/streaming_unet.py',
            'source_sha256': digest, 'observations': inspect_source(raw.decode('utf-8'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    print(json.dumps(audit_file(args.source), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
