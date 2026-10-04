"""Read actual Chapter09 PCM, localize, then apply one-slot lifecycle rules.

No generation, repaired metadata, enhanced waveform or automatic playback.
Default stdout; callers may save its finite JSON output outside the project.
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
from pathlib import Path
from codes.chapters.ch00.io_contracts import validate_parent_chain
from codes.chapters.ch09.core.lifecycle import teaching_lifecycle
from codes.chapters.ch09.core.tracking_audio import analyze_array, read_pcm16
from codes.chapters.ch09.examples.chapter09_tracking_audio import OUTPUT, check_assets, SOURCE_PATHS as AUDIO_SOURCES

ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATHS = (
    'codes/chapters/ch09/core/lifecycle.py',
    'codes/chapters/ch09/examples/tracking_lifecycle_demo.py',
    *AUDIO_SOURCES,
)


def run_demo(audio_directory=OUTPUT):
    """Only real directory PCM drives validity; clean truth never sets events."""
    directory = Path(audio_directory)
    manifest = check_assets(directory)
    array_blob = (directory/'array_noisy.wav').read_bytes()
    analysis = analyze_array(read_pcm16(array_blob), export_gain=manifest['common_export_gain'])
    lifecycle = teaching_lifecycle(analysis['frames'])
    comparison = teaching_lifecycle(analysis['frames'], age_clock='state')
    result = {'schema_version': 1, 'exercise_id': 'E09-25',
        'source_sha256': {path: hashlib.sha256(validate_parent_chain(ROOT/path).read_bytes()).hexdigest()
                          for path in SOURCE_PATHS},
        'input_sha256': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                         for name in ('source.wav', 'array_noisy.wav', 'MANIFEST.json')},
        'analysis_scores': analysis['scores'], 'analysis_frames': analysis['frames'],
        'lifecycle': lifecycle, 'state_age_comparison': comparison,
        'limits': 'Actual two-channel synthetic PCM is re-read after strict read-only asset verification. Only its GCC validity and declared sample timestamps drive lifecycle events. Confirmation is delayed until three consecutive valid frames; a local ID is not permanent speaker identity. The underlying original KF remains a separate continuous diagnostic and is not reset by this one-slot manager. Available-time age and state-time age are different contracts. No new WAV, Bernoulli/LMB filter, multi-target association, hardware result or formal listening study.'}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-directory', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.audio_directory), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
