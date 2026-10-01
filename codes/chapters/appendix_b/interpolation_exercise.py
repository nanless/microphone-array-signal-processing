"""E13-02: fractional-delay amplitudes in regenerated and published PCM.

No files, network, randomness or experiments on import. Run as a module.
The ideal output is evaluated from a known continuous signal, not reconstructed.
"""
from __future__ import annotations

# Allow the documented direct-file command as well as python -m.
if __name__ == "__main__" and not __package__:
    import sys as _chapter_entry_sys
    from pathlib import Path as _ChapterEntryPath
    _chapter_entry_sys.path.insert(0, str(_ChapterEntryPath(__file__).resolve().parents[3]))

import json
import hashlib
from pathlib import Path
import numpy as np

from codes.chapters.ch00.core.audio_samples import interpolation_case, prepare_exports, read_pcm16
from codes.chapters.appendix_b.examples.check_main_interpolation import check_main_interpolation_assets

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'codes/chapters/ch00/audio/MANIFEST.json'
PUBLISHED_AUDIO = ROOT / 'codes/chapters/ch02/audio'


def run_exercises() -> dict:
    case = interpolation_case()
    files, groups = prepare_exports({'interpolation': case})
    fs = case['parameters']['sample_rate_hz']
    start, stop = case['parameters']['scoring_interval_samples']
    n = np.arange(start, stop)
    frequencies = np.array(case['parameters']['frequencies_hz'])
    # At alpha=.5, H=e^(-j*w/2)*cos(w/2); the frequencies are below Nyquist.
    expected_one = np.cos(np.pi * frequencies / fs)
    def measured_amplitude(pcm):
        return [float(2 * abs(np.sum(
            pcm[0, start:stop] * np.exp(-2j * np.pi * f * n / fs))) / len(n))
            for f in frequencies]

    amplitudes = {}
    for name, (blob, _) in files.items():
        _, pcm = read_pcm16(blob)
        amplitudes[name[:-4]] = measured_amplitude(pcm)
    measured = {}
    for output, reference in [('linear_half', 'ideal_half'), ('linear_twice', 'ideal_one')]:
        measured[output] = (np.array(amplitudes['interpolation_' + output]) /
                            amplitudes['interpolation_' + reference]).tolist()

    manifest, actual_buffers = check_main_interpolation_assets(ROOT, manifest_path=MANIFEST, audio_directory=PUBLISHED_AUDIO)
    records = {item['file']: item for item in manifest['files']
               if item.get('group') == 'interpolation'}
    if set(records) != set(files):
        raise ValueError('published interpolation WAV set differs from the manifest or generator')
    published_amplitudes = {}
    published_hashes = {}
    identical_to_regenerated = {}
    for name, (regenerated_blob, _) in files.items():
        path = PUBLISHED_AUDIO / name
        blob = actual_buffers[name]
        digest = hashlib.sha256(blob).hexdigest()
        if digest != records[name]['sha256']:
            raise ValueError(f'published interpolation PCM digest differs: {name}')
        rate, pcm = read_pcm16(blob)
        if (rate != fs or pcm.shape != (records[name]['channels'],
                                       records[name]['samples'])):
            raise ValueError(f'published interpolation PCM format differs: {name}')
        published_amplitudes[name[:-4]] = measured_amplitude(pcm)
        published_hashes[name] = digest
        identical_to_regenerated[name] = blob == regenerated_blob
    for reference in ("interpolation_ideal_half", "interpolation_ideal_one"):
        if any(not np.isfinite(value) or value <= 0 for value in published_amplitudes[reference]):
            raise ValueError("published PCM reference tone amplitude must be positive and finite")
    published_ratios = {}
    for output, reference in [('linear_half', 'ideal_half'), ('linear_twice', 'ideal_one')]:
        published_ratios[output] = (
            np.array(published_amplitudes['interpolation_' + output]) /
            published_amplitudes['interpolation_' + reference]).tolist()
    return {'E13-02': {
        'frequencies_hz': frequencies.tolist(), 'sample_rate_hz': fs,
        'scoring_interval_samples': [start, stop],
        'common_export_gain': groups['interpolation']['common_export_gain'],
        'analytic_one_pass_amplitude': expected_one.tolist(),
        'analytic_two_pass_amplitude': (expected_one**2).tolist(),
        'analytic_one_pass_db': (20*np.log10(expected_one)).tolist(),
        'analytic_two_pass_db': (40*np.log10(expected_one)).tolist(),
        'pcm_amplitudes': amplitudes, 'pcm_amplitude_ratios': measured,
        'published_pcm': {
            'source': 'four checked-in PCM16 WAV in codes/chapters/ch02/audio',
            'file_sha256': published_hashes,
            'amplitudes': published_amplitudes,
            'amplitude_ratios': published_ratios,
            'byte_identical_to_in_memory_regeneration': identical_to_regenerated},
        'limits': case['limits'],
    }}


if __name__ == '__main__':
    print(json.dumps(run_exercises(), ensure_ascii=False, indent=2, allow_nan=False))
