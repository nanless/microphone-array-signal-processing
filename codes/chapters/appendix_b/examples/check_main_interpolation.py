"""Read-only current 20-source and full four-WAV interpolation audit."""
from __future__ import annotations
import hashlib
import io
import wave
from pathlib import Path
from codes.chapters.ch00.core.audio_samples import interpolation_case, prepare_exports
from codes.chapters.ch00.examples.generate_audio_samples import INPUTS
from codes.chapters.appendix_b.examples.room_srp_exercise import ordinary_path, strict_json

ROOT = Path(__file__).resolve().parents[4]


def same_tree(actual, expected, name='metadata'):
    if type(actual) is not type(expected):
        raise ValueError(f'{name} has an invalid type')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'{name} has missing/extra fields')
        for key in expected:
            same_tree(actual[key], expected[key], name+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{name} has the wrong length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same_tree(a, b, name+f'[{i}]')
    elif actual != expected:
        raise ValueError(f'{name} differs from current replay')


def check_main_interpolation_assets(repo_root=ROOT, *, manifest_path=None, audio_directory=None):
    """Return manifest and actual WAV buffers; no write, no repair, no network."""
    root = Path(repo_root)
    path = ordinary_path(manifest_path or root/'codes/chapters/ch00/audio/MANIFEST.json')
    directory = ordinary_path(audio_directory or root/'codes/chapters/ch02/audio', directory=True)
    if not path.is_file() or not directory.is_dir():
        raise ValueError('published interpolation manifest/audio is missing')
    manifest = strict_json(path.read_bytes())
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 2:
        raise ValueError('invalid main audio manifest schema')
    sources = manifest.get('generator_inputs')
    if not isinstance(sources, dict) or set(sources) != set(INPUTS):
        raise ValueError('main source set must match all twenty real dependencies')
    for name, digest in sources.items():
        source = ordinary_path(root/name)
        if (not source.is_file() or type(digest) is not str or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest):
            raise ValueError(f'main source digest differs: {name}')
    records = manifest.get('files')
    if not isinstance(records, list) or len(records) != 109 or any(not isinstance(r, dict) or type(r.get('file')) is not str for r in records):
        raise ValueError('main manifest needs 109 distinct file records')
    if len({r['file'] for r in records}) != 109:
        raise ValueError('duplicate main PCM records')
    selected = {r['file']: r for r in records if r.get('group') == 'interpolation'}
    case = interpolation_case()
    expected, groups = prepare_exports({'interpolation': case})
    if set(selected) != set(expected):
        raise ValueError('published interpolation WAV set differs from complete replay')
    if not isinstance(manifest.get('groups'), dict):
        raise ValueError('invalid main group records')
    same_tree(manifest['groups'].get('interpolation'), groups['interpolation'], 'interpolation group')
    actual = {}
    for name, (data, info) in expected.items():
        record = {'file': name, 'chapter': 'ch02', 'sha256': hashlib.sha256(data).hexdigest(), **info}
        same_tree(selected[name], record, name+' record')
        target = ordinary_path(directory/name)
        if not target.is_file():
            raise ValueError(f'published interpolation PCM is missing: {name}')
        blob = target.read_bytes()
        try:
            with wave.open(io.BytesIO(blob), 'rb') as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getnframes(), reader.getsampwidth(), reader.getcomptype()) != (16000, 1, 32000, 2, 'NONE'):
                    raise ValueError('published interpolation PCM format differs')
                raw = reader.readframes(32000)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid interpolation PCM WAV') from error
        if len(raw) != 64000 or hashlib.sha256(blob).hexdigest() != selected[name]['sha256'] or blob != data:
            raise ValueError(f'published interpolation PCM differs from complete replay: {name}')
        actual[name] = blob
    return manifest, actual
