"""Strict read-only room assets audit; optional replay really requires PRA.

The default validates the 21 ordinary members, strict metadata, current repo
source closure, actual PCM and published report bindings. It does not claim
to recompute RIRs without pyroomacoustics. --replay additionally runs the
original generator into a fresh temporary directory and compares all WAVs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import tempfile
import wave
from pathlib import Path
import numpy as np
from codes.chapters.appendix_b.examples import room_srp_exercise as room

ROOT = room.ROOT
ROOM = ROOT/'codes/chapters/appendix_b/room_audio'
CASE_NAMES = ('fixed_near_left', 'fixed_near_right', 'fixed_far_left',
              'fixed_far_right', 'seeded_1', 'seeded_2')
WAV_NAMES = tuple(name+'_'+role+'.wav' for name in CASE_NAMES for role in ('source', 'direct', 'full'))
MEMBERS = set(WAV_NAMES) | {'MANIFEST.json', 'RESULTS.json', 'ROOM_RESULTS.png'}
# Independently verified against the fixed local upstream tree and installed
# PRA 0.10.0. These identify only these three Python files, not compiled
# extensions, all package files, or a complete binary/runtime identity.
PRA_PYTHON_SOURCE_SHA256 = {
    'room.py': 'b4189f582459811f8dbbf3885882e115db60b7b48b4ca0a3672dbef39118e4dc',
    'acoustics.py': '3196254709071ab1061982133f05be94e7da554f2619db98d829203839d7c4d2',
    'parameters.py': 'dac6389cdaf949659cb2d8dbe01655593f423b35492bc857a43c4af9b24b4ed2',
}


def _same(actual, expected, name):
    if type(actual) is not type(expected):
        raise ValueError(f'{name} has an invalid type')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'{name} has missing or extra fields')
        for key in expected:
            _same(actual[key], expected[key], name+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'{name} has an invalid length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            _same(a, b, name+f'[{index}]')
    elif actual != expected:
        raise ValueError(f'{name} differs from the declared contract')


def _number(value, name, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite real number')


def check_assets(directory=ROOM, *, replay=False):
    if type(replay) is not bool:
        raise ValueError('replay must be bool')
    directory = room.ordinary_path(directory, directory=True)
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != MEMBERS:
        raise ValueError('room directory must contain exactly 21 expected members')
    for name in MEMBERS:
        if not room.ordinary_path(directory/name).is_file():
            raise ValueError(f'room member must be an ordinary file: {name}')
    manifest = room.strict_json((directory/'MANIFEST.json').read_bytes())
    result = room.strict_json((directory/'RESULTS.json').read_bytes())
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise ValueError('room metadata must be JSON objects')
    _same(result.get('schema_version'), 1, 'room result schema')
    _same(result.get('status'), 'pyroomacoustics_simulation_executed', 'room result status')
    current_sources = room.source_hashes()
    for document in (manifest, result):
        _same(document.get('source_sha256'), current_sources, 'room source closure')
        environment = document.get('environment')
        if not isinstance(environment, dict):
            raise ValueError('room runtime environment is missing')
        for key in ('python', 'numpy', 'scipy', 'pyroomacoustics', 'platform'):
            if type(environment.get(key)) is not str or not environment[key]:
                raise ValueError(f'room environment field is invalid: {key}')
        _same(environment['pyroomacoustics'], room.PRA_VERSION, 'PRA version')
        library_sources = environment.get('pyroomacoustics_source_sha256')
        if (not isinstance(library_sources, dict) or set(library_sources) != {'room.py', 'acoustics.py', 'parameters.py'}
                or any(type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                       for value in library_sources.values())):
            raise ValueError('original installed PRA source identities are missing or invalid')
        _same(library_sources, PRA_PYTHON_SOURCE_SHA256, 'fixed PRA Python source identities')
    _same(manifest['environment'], result['environment'], 'room environment binding')
    config = room.configuration()
    # Published experiment declaration, not inferred from a matching source SHA.
    # Default mode checks these settings; it does not rerun PRA or estimate DOA.
    fixed_doa = {'method': 'far-field SRP-PHAT', 'n_fft': 512, 'hop_length': 128,
                 'frequency_band_hz': [300.0, 2000.0], 'azimuth_grid_deg': [-80, 80, 1]}
    _same(config.get('doa'), fixed_doa, 'published room DOA source configuration')
    _same(result.get('doa'), fixed_doa, 'published room DOA declaration')
    _same(result.get('fractional_delay_filter_length_samples'), 81,
          'published room fractional-delay filter length')
    # PRA's centered odd-length 81-tap delay filter has 40 samples of library
    # padding. This is not a fitted propagation compensation or an RIR replay.
    library_global_delay = (81-1)//2
    for key in ('room_dimensions_m', 'microphones_m', 'sample_rate_hz', 'sound_speed_m_s', 'target_t60_s'):
        _same(manifest.get(key), config[key], 'room manifest '+key)
        _same(result.get(key), config[key], 'room results '+key)
    _same(manifest.get('max_order'), result.get('actual_max_order'), 'room image order')
    if type(manifest.get('max_order')) is not int or not 2 <= manifest['max_order'] <= 80:
        raise ValueError('room image order is outside the teaching resource budget 2..80')
    _number(manifest.get('common_gain'), 'room gain', positive=True)
    _number(manifest.get('maximum_pre_gain_peak'), 'room pre-gain peak', positive=True)
    _same(manifest.get('common_gain'), .8/manifest['maximum_pre_gain_peak'], 'room common gain')
    _same(manifest.get('maximum_post_gain_peak'), .8, 'room target peak')
    for key, expected in (('source_and_microphone_clocks_aligned', True),
                          ('propagation_delay_and_gain_compensated', False)):
        _same(manifest.get(key), expected, key)
    for key in ('rir_highpass_enabled', 'rir_highpass_cutoff_hz'):
        _same(manifest.get(key), result.get(key), 'room highpass '+key)
    if type(manifest.get('rir_highpass_enabled')) is not bool:
        raise ValueError('room highpass flag must be bool')
    if manifest['rir_highpass_enabled']:
        _number(manifest.get('rir_highpass_cutoff_hz'), 'room highpass cutoff', positive=True)
    _same(manifest.get('wall_energy_absorption_full'), config['sabine_energy_absorption'], 'full absorption')
    _same(manifest.get('wall_energy_absorption_direct'), 1.0, 'direct absorption')
    _same(manifest.get('pcm_scale'), 'writer int16=round_even(amplitude*32767); decode int16/32768', 'room codec')
    _same(result.get('generator'), {'path': room.SOURCE_PATHS[0], 'sha256': current_sources[room.SOURCE_PATHS[0]]}, 'room generator')
    _same(result.get('assets'), {
        'figure': {'file': 'ROOM_RESULTS.png', 'sha256': hashlib.sha256((directory/'ROOM_RESULTS.png').read_bytes()).hexdigest()},
        'audio_manifest': {'file': 'MANIFEST.json', 'sha256': hashlib.sha256((directory/'MANIFEST.json').read_bytes()).hexdigest()}}, 'room asset binding')
    rows = result.get('results')
    if (not isinstance(rows, list) or len(rows) != 6 or any(not isinstance(row, dict) for row in rows)
            or [row.get('name') for row in rows] != list(CASE_NAMES)):
        raise ValueError('room report must contain the six declared rows')
    for row, case in zip(rows, config['cases']):
        for key in ('name', 'source_m', 'distance_m', 'true_azimuth_deg'):
            _same(row.get(key), case[key], 'room case '+key)
        for key in ('drr_db_median', 't60_s_median', 'estimated_azimuth_deg', 'absolute_doa_error_deg'):
            _number(row.get(key), key, positive=key == 't60_s_median')
        for key in ('drr_db_per_mic', 't60_s_per_mic'):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError('room per-microphone metrics must contain four values')
            for value in values:
                _number(value, key, positive=key == 't60_s_per_mic')
        _same(row['drr_db_median'], float(np.median(row['drr_db_per_mic'])), 'median DRR')
        _same(row['t60_s_median'], float(np.median(row['t60_s_per_mic'])), 'median T60')
        _same(row['absolute_doa_error_deg'], abs(row['estimated_azimuth_deg']-row['true_azimuth_deg']), 'room DOA error')
    records = manifest.get('files')
    if (not isinstance(records, list) or len(records) != 18 or any(not isinstance(r, dict) or type(r.get('file')) is not str for r in records)
            or {r.get('file') for r in records} != set(WAV_NAMES)):
        raise ValueError('room manifest must describe exactly eighteen distinct WAVs')
    case_records = manifest.get('cases')
    if not isinstance(case_records, list) or len(case_records) != 6 or any(not isinstance(r, dict) for r in case_records):
        raise ValueError('room manifest needs six excitation records')
    sources = {}
    for record, case in zip(case_records, config['cases']):
        for key in ('name', 'source_m', 'excitation_seed_sequence'):
            _same(record.get(key), case[key], 'room excitation '+key)
        source = room._excitation(case)
        _same(record.get('source_float64_le_sha256'), hashlib.sha256(np.asarray(source, dtype='<f8').tobytes()).hexdigest(), 'room excitation digest')
        sources[case['name']] = source
    sizes = {}
    for record in records:
        name = record['file']
        case, role = record.get('case'), record.get('role')
        if case not in CASE_NAMES or role not in ('source', 'direct', 'full') or name != case+'_'+role+'.wav':
            raise ValueError('room WAV case/role disagrees with its filename')
        for key in ('peak_before_gain', 'peak_after_gain'):
            _number(record.get(key), 'room WAV '+key, positive=True)
        if record['peak_after_gain'] > .8+1e-12:
            raise ValueError('room declared PCM peak exceeds common headroom')
        data = (directory/name).read_bytes()
        _same(record.get('sha256'), hashlib.sha256(data).hexdigest(), 'room WAV digest '+name)
        channels = 1 if role == 'source' else 4
        frames = record.get('frames')
        if type(frames) is not int or frames <= 0 or type(record.get('channels')) is not int or record['channels'] != channels:
            raise ValueError('room PCM dimensions must be positive integer fields')
        try:
            with wave.open(str(directory/name), 'rb') as reader:
                if (reader.getframerate(), reader.getnframes(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (16000, frames, channels, 2, 'NONE'):
                    raise ValueError('room PCM format differs from the manifest')
                raw = reader.readframes(frames)
        except (wave.Error, EOFError) as error:
            raise ValueError('invalid room PCM') from error
        if len(raw) != 2*frames*channels:
            raise ValueError('truncated room PCM')
        if role == 'source':
            expected = np.rint(sources[case]*manifest['common_gain']*32767).astype('<i2').tobytes()
            if frames != 16000 or raw != expected:
                raise ValueError('room source PCM differs from complete excitation replay')
        else:
            sizes[(case, role)] = frames
    for case, row in zip(CASE_NAMES, rows):
        lengths = row.get('rir_length_samples_per_mic')
        if not isinstance(lengths, list) or len(lengths) != 4 or any(type(v) is not int or v <= 0 for v in lengths):
            raise ValueError('room RIR lengths must be four positive integers')
        _same(sizes[(case, 'full')], 16000+max(lengths)-1, 'complete convolution tail')
        _same(sizes[(case, 'direct')], sizes[(case, 'full')], 'same direct/full time axis')
    if replay:
        new_result = room.run_experiment(manifest['max_order'])
        with tempfile.TemporaryDirectory(prefix='masp-room-replay-') as temp:
            destination = Path(temp)/'audio'
            room.export_audio(new_result, destination)
            for name in WAV_NAMES:
                if (destination/name).read_bytes() != (directory/name).read_bytes():
                    raise ValueError(f'room WAV differs from full PRA replay: {name}; compiled/runtime differences may change PCM LSBs')
    return {'manifest': manifest, 'results': result, 'validation': {
        'actual_pcm_files': 18, 'ordinary_members': 21, 'source_closure_checked': True,
        'published_doa_configuration_checked': True, 'fractional_delay_filter_length_samples': 81,
        'library_global_delay_samples': library_global_delay,
        'excitation_replayed': True, 'rir_and_output_replayed': replay}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOM)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_assets(args.directory, replay=args.replay)['validation'], allow_nan=False))
