"""Generate/check the independent eight-WAV two-scene FIR experiment.

--check validates ordinary paths, strict JSON, current sources and actual PCM,
then replays the complete model in memory. It never repairs a stale directory.
"""
from __future__ import annotations
if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
from codes.chapters.ch00.core.audio_samples import pcm16_bytes
from codes.chapters.ch11.core.selection_audio import (
    SAMPLE_RATE, SAMPLES, STEMS, GAIN, build_fixture, analytic_results, analyze_fixture, analyze_pcm,
    decode_pcm, measure_pcm,
)

from codes.chapters.ch00.io_contracts import (
    validate_asset_directory as _shared_asset_directory, strict_json_loads, same_metadata,
)

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'codes/chapters/ch11/scenario_audio'
SOURCE_PATHS = (
    'codes/chapters/ch11/core/selection_audio.py',
    'codes/chapters/ch11/examples/generate_selection_audio.py',
    'codes/chapters/ch00/core/audio_samples.py',
    'codes/chapters/ch00/io_contracts.py',
)
MEMBERS = tuple(stem+'.wav' for stem in STEMS)+('MANIFEST.json',)


def validate_asset_directory(directory, *, check: bool):
    """Shared path preflight with this experiment's exact member policy."""
    return _shared_asset_directory(directory, MEMBERS, check=check)


def _strict_json(data: bytes):
    return strict_json_loads(data)


def _same_metadata(actual, expected, path='manifest'):
    if not same_metadata(actual, expected):
        raise ValueError(f'{path} has invalid types or differs from current source/PCM replay')


def expected_assets():
    """Pure full replay, with source hashes from the actual four dependencies."""
    fixture = build_fixture()
    buffers = {}
    for name, signal in fixture['signals'].items():
        if np.max(np.abs(signal*GAIN)) > 32767/32768:
            raise ValueError('fixture would clip PCM at common gain .8')
        buffers[name+'.wav'] = pcm16_bytes(signal*GAIN, SAMPLE_RATE)
    metadata = {
        'schema_version': 1, 'sample_rate_hz': SAMPLE_RATE, 'common_export_gain': GAIN,
        'origin': 'mathematical synthetic single/dual target tone/FIR example; not speech or a recording',
        'parameters': fixture['parameters'], 'analytic': analytic_results(fixture), 'floating_point': analyze_fixture(fixture),
        'pcm_analysis': analyze_pcm(buffers),
        'quantization': 'signed little-endian PCM16; nearest-even; no dither; decoded amplitude=int16/32768',
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_PATHS},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
        'files': {name: {'channels': 1, 'samples_per_channel': SAMPLES,
                         'sha256': hashlib.sha256(data).hexdigest()} for name, data in buffers.items()},
    }
    buffers['MANIFEST.json'] = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    return buffers, metadata


def check_assets(directory=OUTPUT):
    """Strictly read actual bytes/score them, then compare full current replay."""
    directory = Path(directory)
    validate_asset_directory(directory, check=True)
    metadata = _strict_json((directory/'MANIFEST.json').read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError('manifest must be an object')
    actual = {name: (directory/name).read_bytes() for name in MEMBERS if name.endswith('.wav')}
    # This analysis runs on the directory bytes, not on regenerated stand-ins.
    measured = analyze_pcm(actual)
    expected, replay = expected_assets()
    _same_metadata(metadata, replay)
    _same_metadata(metadata['pcm_analysis'], measured, 'actual PCM analysis')
    for name, data in actual.items():
        if hashlib.sha256(data).hexdigest() != metadata['files'][name]['sha256']:
            raise ValueError(f'actual WAV digest differs: {name}')
        if data != expected[name]:
            raise ValueError(f'actual WAV differs from current complete replay: {name}')
    return metadata


def generate(directory=OUTPUT, *, check=False):
    validate_asset_directory(directory, check=check)
    if check:
        return check_assets(directory)
    buffers, metadata = expected_assets()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in buffers.items():
        (directory/name).write_bytes(data)
    return metadata


def check_main_selection_assets(repo_root=ROOT):
    """Read the four published main WAVs; never generate or repair any asset.

    Verifies strict JSON, all 20 genuine main-source digests, the four stored
    file hashes/formats and complete current byte replay of this group. The
    old return keys remain, but PCM analysis comes from actual file integers.
    """
    from codes.chapters.ch00.core.audio_samples import selection_tradeoff_case
    from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
    repo_root = Path(repo_root)
    if '..' in repo_root.parts:
        raise ValueError('main root must not contain lexical parent traversal')
    for target in (repo_root/'codes/chapters/ch00/audio', repo_root/'codes/chapters/ch11/audio'):
        for component in (*reversed(target.absolute().parents), target.absolute()):
            if component.is_symlink() and not (str(component) in ('/tmp','/var','/etc') and component.resolve()==Path('/private')/component.name):
                raise ValueError('main audio ancestors must be ordinary directories')
            if component.exists() and not component.is_dir():
                raise ValueError('main audio ancestors must be directories')
    manifest_path = repo_root/'codes/chapters/ch00/audio/MANIFEST.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('main manifest must be an ordinary file')
    manifest = _strict_json(manifest_path.read_bytes())
    if not isinstance(manifest,dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 2:
        raise ValueError('invalid main manifest schema')
    sources = manifest.get('generator_inputs')
    if not isinstance(sources,dict) or set(sources) != set(INPUTS):
        raise ValueError('main source set must match the twenty actual dependencies')
    for path, digest in sources.items():
        source = repo_root/path
        if source.is_symlink() or not source.is_file() or not isinstance(digest,str) or (
                hashlib.sha256(source.read_bytes()).hexdigest() != digest or
                hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest):
            raise ValueError(f'main source digest mismatch: {path}')
    records = manifest.get('files')
    if not isinstance(records,list) or any(not isinstance(v,dict) or not isinstance(v.get('file'),str) for v in records):
        raise ValueError('invalid main file records')
    names = [v['file'] for v in records]
    if len(names) != len(set(names)):
        raise ValueError('duplicate main file records')
    records = dict(zip(names,records))
    case = selection_tradeoff_case()
    if not isinstance(manifest.get('groups'),dict):
        raise ValueError('invalid main group records')
    group = manifest['groups'].get('selection_tradeoff')
    expected_group = {key: value for key,value in case.items() if key != 'signals'}
    expected_group['common_export_gain'] = .8
    _same_metadata(group,expected_group,'main selection group')
    actual, integers = {}, {}
    for stem, values in case['signals'].items():
        name = stem+'.wav'
        path = repo_root/'codes/chapters/ch11/audio'/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'main PCM must be an ordinary file: {name}')
        record = records.get(name)
        if not isinstance(record,dict) or record.get('chapter') != 'ch11' or record.get('group') != 'selection_tradeoff':
            raise ValueError(f'missing/invalid main selection record: {name}')
        for key, value in {'sample_rate_hz':16000,'channels':1,'samples':32000,'common_export_gain':.8}.items():
            if type(record.get(key)) is not type(value) or record[key] != value:
                raise ValueError(f'main selection record format mismatch: {name}/{key}')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.get('sha256'):
            raise ValueError(f'main PCM digest mismatch: {name}')
        integers[stem] = decode_pcm(data,samples=32000)
        if data != pcm16_bytes(values*.8,16000):
            raise ValueError(f'main PCM differs from full current replay: {name}')
        actual[name] = {'sha256':hashlib.sha256(data).hexdigest(),'channels':1,'samples_per_channel':32000}
    # The measurements below consume the directory bytes, not replay values.
    pcm_report = {'candidates':{}}
    def amplitudes(stem,lo=1600,hi=30400):
        time=np.arange(lo,hi)/16000
        columns=[np.ones(hi-lo)]
        for f in (500,1500,3500):columns.extend([np.cos(2*np.pi*f*time),np.sin(2*np.pi*f*time)])
        coefficients=np.linalg.lstsq(np.column_stack(columns),np.asarray(integers[stem][lo:hi])/32768,rcond=None)[0]
        return {str(f):float(np.hypot(coefficients[1+2*i],coefficients[2+2*i])) for i,f in enumerate((500,1500,3500))}
    pcm_report['clean_fitted_amplitudes']=amplitudes('selection_clean')
    pcm_report['mixture_fitted_amplitudes']=amplitudes('selection_mixture')
    integer_analysis={}
    for L in (3,9):
        stem=f'selection_fir{L}'
        measured=measure_pcm(integers['selection_clean'],integers['selection_mixture'],integers[stem],(L-1)//2)
        # Preserve historical PCM key names; add exact sums separately.
        values={key:measured[key] for key in case['pcm_analysis']['candidates'][stem]}
        fitted=amplitudes(stem,1600+(L-1)//2,30400+(L-1)//2)
        values['fitted_amplitudes']=fitted
        for frequency in ('500','1500'):
            values['target_'+frequency+'_retention']=fitted[frequency]/pcm_report['clean_fitted_amplitudes'][frequency]
        values['noise_3500_retention']=fitted['3500']/pcm_report['mixture_fitted_amplitudes']['3500']
        values['noise_attenuation_db']=-20*math.log10(values['noise_3500_retention'])
        pcm_report['candidates'][stem]=values
        integer_analysis[stem]={key:measured[key] for key in ('integer_error_squared_sum','integer_reference_squared_sum',
            'scored_samples','reference_score','output_score')}
    result={key:value for key,value in case.items() if key != 'signals'}
    result['pcm_analysis']=pcm_report
    result['integer_analysis']=integer_analysis
    result['published_audio']={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                              'source_sha256':sources,'files':actual}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    metadata = generate(args.output, check=args.check)
    print(json.dumps({'mode': 'verified' if args.check else 'generated',
                      'directory': str(args.output), 'pcm_analysis': metadata['pcm_analysis']}, indent=2))


if __name__ == '__main__':
    main()
