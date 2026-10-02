"""Strictly read the three published E12-08 main WAVs; never repair them."""
from __future__ import annotations
import hashlib
import io
import struct
import wave
from pathlib import Path
from codes.chapters.ch00.core.audio_samples import math_block_case, prepare_exports
from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
from codes.chapters.appendix_a.examples.generate_weighted_audio import _strict_json, _same_metadata

ROOT = Path(__file__).resolve().parents[4]


def _ordinary_path(target):
    if '..' in Path(target).parts:
        raise ValueError('main path must not contain lexical parent traversal')
    for component in (*reversed(Path(target).absolute().parents), Path(target).absolute()):
        if component.is_symlink() and not (str(component) in ('/tmp','/var','/etc') and component.resolve()==Path('/private')/component.name):
            raise ValueError('main paths must have ordinary ancestors/members')


def check_main_math_assets(repo_root=ROOT):
    """Current 20 sources, full math_block replay, formats and actual integers.

    Floating-point construction and published PCM are separate fields. The
    latter reads wave/struct bytes from disk and supplies exact integer sums.
    Missing, stale, malformed or zero-reference assets raise ValueError.
    """
    repo_root = Path(repo_root)
    manifest_path = repo_root/'codes/chapters/ch00/audio/MANIFEST.json'
    _ordinary_path(manifest_path)
    if not manifest_path.is_file():
        raise ValueError('main manifest must be an ordinary file')
    manifest = _strict_json(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 2:
        raise ValueError('invalid main manifest schema')
    sources = manifest.get('generator_inputs')
    if not isinstance(sources, dict) or set(sources) != set(INPUTS):
        raise ValueError('main sources must match all twenty true dependencies')
    for name, digest in sources.items():
        path = repo_root/name
        _ordinary_path(path)
        if not path.is_file() or type(digest) is not str or (
            hashlib.sha256(path.read_bytes()).hexdigest()!=digest or
            hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest):
            raise ValueError(f'main source digest mismatch: {name}')
    records = manifest.get('files')
    if not isinstance(records, list) or any(not isinstance(row, dict) or type(row.get('file')) is not str for row in records):
        raise ValueError('invalid main file records')
    names = [row['file'] for row in records]
    if len(names)!=len(set(names)):
        raise ValueError('duplicate main file records')
    records = dict(zip(names, records))
    case = math_block_case()
    expected, groups = prepare_exports({'math_block': case})
    if not isinstance(manifest.get('groups'), dict):
        raise ValueError('invalid main groups')
    _same_metadata(manifest['groups'].get('math_block'), groups['math_block'], 'main math_block group')
    integers, actual = {}, {}
    for filename, (blob, info) in expected.items():
        path = repo_root/'codes/chapters/appendix_a/audio'/filename
        _ordinary_path(path)
        if not path.is_file():
            raise ValueError(f'main PCM must be an ordinary file: {filename}')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected_record = {'file':filename, 'chapter':'appendix_a', 'sha256':hashlib.sha256(blob).hexdigest(), **info}
        _same_metadata(records.get(filename), expected_record, f'main record {filename}')
        if digest != records[filename]['sha256'] or data!=blob:
            raise ValueError(f'main PCM differs from complete replay: {filename}')
        try:
            with wave.open(io.BytesIO(data)) as reader:
                if (reader.getframerate(), reader.getnframes(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype())!=(16000,32000,1,2,'NONE'):
                    raise ValueError('unexpected main math PCM format')
                raw = reader.readframes(32000)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid main math PCM') from error
        if len(raw)!=64000:
            raise ValueError('truncated main math PCM')
        integers[filename[:-4]] = struct.unpack('<32000h', raw)
        actual[filename] = {'sha256':digest,'channels':1,'samples_per_channel':32000}
    dry, linear, wrong = (integers[name] for name in ('math_block_dry','math_block_linear','math_block_circular'))
    denominator = sum(value*value for value in linear)
    dry_power = sum(value*value for value in dry)
    if dry_power<=0 or denominator<=0:
        raise ValueError('main PCM reference must have positive power')
    numerator = sum((a-b)**2 for a,b in zip(wrong,linear))
    pcm = {
        'first_dry_sample_500':dry[500]/32768,
        'first_linear_sample_500':linear[500]/32768,
        'first_linear_echo_sample_620':linear[620]/32768,
        'first_wrong_wrap_sample_108':wrong[108]/32768,
        'first_wrong_sample_500':wrong[500]/32768,
        'first_wrong_sample_620':wrong[620]/32768,
        'wrong_minus_linear_nonzero_count':sum(a!=b for a,b in zip(wrong,linear)),
    }
    return {'parameters':case['parameters'],'float_analysis':case['float_analysis'],
        'pcm_analysis':pcm,'integer_analysis':{'scored_samples':32000,
            'integer_error_squared_sum':numerator,'integer_reference_squared_sum':denominator,
            'dry_integer_squared_sum':dry_power,'nmse':numerator/denominator,
            'mse':numerator/(32000*32768**2),'decoded_amplitude_denominator':32768},
        'published_files':actual, 'source_sha256':sources}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = check_main_math_assets(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
