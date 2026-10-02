"""Standard-library IO checks shared by asset generators and source reports.

These are preflight contracts for local, non-concurrent workflows, not a
race-proof filesystem sandbox. Callers retain their member sets, metadata
schemas, numerical comparisons, and generation algorithms.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import tempfile


def validate_parent_chain(path) -> Path:
    """Inspect the lexical path before resolving anything; never create it.

    Existing ancestors must be ordinary directories. Only the conventional
    macOS /tmp, /var and /etc aliases to their /private counterparts are allowed.
    An existing leaf may be a directory or a singly linked regular file.
    """
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('path must not contain lexical parent traversal')
    path = path.absolute()
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            if (str(part) in ('/tmp', '/var', '/etc')
                    and part.resolve() == Path('/private') / part.name):
                continue
            raise ValueError(f'path must not follow a symbolic link: {part}')
        if part != path or stat.S_ISDIR(info.st_mode):
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f'path ancestor must be an ordinary directory: {part}')
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f'path leaf must be a singly linked ordinary file: {part}')
    return path


def validate_asset_directory(directory, members, *, check: bool) -> Path:
    """Require exact ordinary members; only generation permits a new/empty set."""
    if type(check) is not bool:
        raise ValueError('check must be bool')
    expected = set(members)
    if not expected or any(not isinstance(name, str) or not name
                           or Path(name).name != name or name in ('.', '..')
                           or '/' in name or '\\' in name for name in expected):
        raise ValueError('members must be explicit ordinary leaf names')
    directory = validate_parent_chain(directory)
    if not directory.exists():
        if check:
            raise ValueError('asset directory is missing')
        return directory
    if not directory.is_dir():
        raise ValueError('asset path must be an ordinary directory')
    actual = list(directory.iterdir())
    for member in actual:
        info = member.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f'asset members must be singly linked ordinary files: {member.name}')
    names = {member.name for member in actual}
    if names != expected and (check or names):
        raise ValueError(f'asset member set differs: missing={sorted(expected-names)}, extra={sorted(names-expected)}')
    return directory


def strict_json_loads(data: str | bytes):
    """Decode UTF-8 JSON, rejecting duplicate keys and nonfinite numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('JSON number exceeds finite float range')
        return value

    def constant(text):
        raise ValueError(f'nonfinite JSON constant: {text}')

    if not isinstance(data, (str, bytes)):
        raise ValueError('JSON input must be text or UTF-8 bytes')
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data, object_pairs_hook=pairs, parse_float=number,
                          parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError('input must be strict UTF-8 JSON') from error


def same_metadata(actual, expected, *, allow_int_for_float=False) -> bool:
    """Exact recursive values/types; an optional caller policy allows int/float."""
    if type(allow_int_for_float) is not bool:
        raise ValueError('allow_int_for_float must be bool')
    if type(actual) is not type(expected):
        if not (allow_int_for_float and type(expected) is float
                and type(actual) is int):
            return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(same_metadata(actual[k], v, allow_int_for_float=allow_int_for_float)
                        for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (len(actual) == len(expected)
                and all(same_metadata(a, b, allow_int_for_float=allow_int_for_float)
                        for a, b in zip(actual, expected)))
    if type(expected) is float and not math.isfinite(expected):
        return False
    return actual == expected


def validate_report_destination(path, *, forbidden_roots=()) -> Path:
    """Preflight a report before downloads, cache changes, or replacement.

    Each forbidden path excludes itself and descendants (a lock file can be
    passed as a singleton forbidden root). Existing ordinary reports may be
    explicitly replaced; links, directories, and special files are rejected.
    """
    path = validate_parent_chain(path)
    if path.exists() and not path.is_file():
        raise ValueError('report destination must be an ordinary file')
    for root in forbidden_roots:
        root = Path(root).absolute()
        # Resolving *after* leaf/parent validation also handles allowed system aliases.
        normalized = root.resolve()
        target = path.resolve()
        if (target == normalized or normalized in target.parents
                or target in normalized.parents):
            raise ValueError(f'report destination overlaps a protected path: {root}')
    return path


def write_json_report(path, data, *, forbidden_roots=()) -> None:
    """Serialize first, then publish via same-directory replace; no crash proof claim."""
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        strict_json_loads(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('report must be finite JSON data') from error
    path = validate_report_destination(path, forbidden_roots=forbidden_roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.report-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
        # Recheck before replacement, without claiming protection from concurrent races.
        validate_report_destination(path, forbidden_roots=forbidden_roots)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
