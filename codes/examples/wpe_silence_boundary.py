"""Reproduce, without patching, the locked online-WPE silence overflow.

This is a failure probe, not a robust online processor. The adapted reference
keeps upstream behavior intentionally. Optional upstream execution verifies the
exact installed source hash before importing its algorithm; nothing is fetched.
Run with --upstream --report PATH to record a real comparison (strict JSON).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from codes.array_tutorial.conventions import finite_real_scalar
from codes.examples.compare_online_wpe_reference import (
    NumpyOnlineWPE011, NARA_WPE_VERSION, NARA_WPE_MODULE_SHA256,
    _load_locked_upstream,
)


def classify_scalar(value) -> dict:
    z = complex(np.asarray(value).item())
    finite = bool(np.isfinite(z))
    result = {'finite': finite, 'value_class': 'finite' if finite else
              ('nan' if np.isnan(z.real) or np.isnan(z.imag) else 'infinity')}
    if finite:
        result.update(real=z.real, imag=z.imag)
    return result


def silence_probe(alpha=.95, *, implementation='adapted', max_frames=16000) -> dict:
    """One-based processed-frame counts, F=C=K=delay=1, initial P=1/g=0."""
    alpha = finite_real_scalar(alpha, 'alpha')
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0,1]')
    if isinstance(max_frames, (bool, np.bool_)) or not isinstance(max_frames, (int, np.integer)) or max_frames < 1:
        raise ValueError('max_frames must be a positive integer')
    if implementation == 'adapted':
        state = NumpyOnlineWPE011(taps=1, delay=1, alpha=alpha, frequency_bins=1, channels=1)
        covariance_name = 'inverse_covariance'
    elif implementation == 'upstream':
        upstream, *_ = _load_locked_upstream()
        state = upstream(taps=1, delay=1, alpha=alpha, frequency_bins=1, channel=1)
        covariance_name = 'inv_cov'
    else:
        raise ValueError('implementation must be adapted or upstream')
    first_bad = None
    previous = None
    # Warnings are deliberately suppressed only in this named failure probe;
    # the report records non-finite state rather than serializing NaN/Infinity.
    with np.errstate(all='ignore'):
        for count in range(1, max_frames + 1):
            previous = classify_scalar(getattr(state, covariance_name))
            output = state.step_frame(np.zeros((1, 1), complex))
            covariance = getattr(state, covariance_name)
            if not np.all(np.isfinite(covariance)):
                first_bad = count
                break
        final_covariance = classify_scalar(covariance)
        final_output = classify_scalar(output)
        resume = [classify_scalar(state.step_frame(np.ones((1, 1), complex))) for _ in range(3)]
    return {'implementation': implementation, 'alpha': alpha,
            'dimensions': {'frequency_bins': 1, 'channels': 1, 'taps': 1, 'delay_argument': 1},
            'initial_inverse_covariance': 1., 'initial_filter': 0., 'initial_buffer': 'all zeros',
            'observations': 'zero until first non-finite inverse covariance or cap, then three unit frames',
            'frame_count_convention': 'one-based number of processed zero frames',
            'maximum_zero_frames': max_frames, 'processed_zero_frames': count,
            'first_nonfinite_state_frame': first_bad,
            'last_pre_step_covariance': previous,
            'final_zero_frame_output': final_output,
            'final_zero_frame_covariance': final_covariance,
            'resumed_unit_frame_outputs': resume,
            'first_nonfinite_output_resume_frame': next((i for i, r in enumerate(resume, 1) if not r['finite']), None),
            'illustrative_hop_seconds': .008,
            'first_nonfinite_state_seconds_at_8ms_hop': None if first_bad is None else first_bad * .008,
            'analytic_zero_history_rule': 'P_n = alpha**(-n), gain=0 until floating-point range is exceeded'}


def run_report(*, upstream=False) -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = ['codes/examples/wpe_silence_boundary.py', 'codes/examples/compare_online_wpe_reference.py']
    records = []
    for alpha in (.5, .95):
        records.append(silence_probe(alpha))
        if upstream:
            records.append(silence_probe(alpha, implementation='upstream'))
    return {'schema_version': 1, 'scope': 'deterministic float64 silence failure, not room or speech quality',
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'system': platform.system(), 'machine': platform.machine()},
            'source': {'project': 'nara-wpe', 'version': NARA_WPE_VERSION,
                       'revision': 'a166779cca2088817e330481bd20af1a2c598555',
                       'url': 'https://github.com/fgnt/nara_wpe/blob/a166779cca2088817e330481bd20af1a2c598555/nara_wpe/wpe.py',
                       'installed_module_sha256': NARA_WPE_MODULE_SHA256 if upstream else None,
                       'upstream_execution': 'executed_after_hash_verification' if upstream else 'not_run',
                       'adaptation_notice': 'codes/licenses/nara_wpe_MIT.txt; adapted comparison preserves upstream behavior'},
            'generator_inputs': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
            'records': records,
            'interpretation': 'No repair is applied. Silence gating, resetting, or bounded covariance are distinct algorithm changes requiring their own validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run_report(upstream=args.upstream), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    else:
        print(encoded, end='')
