"""Read-only bindings for historical source reports and complete saved JSON.

A report's SHA identifies the exact bytes used at that time. A preserved
snapshot may differ from today's complete index, but every explicitly used
project must retain its entire record, including acquisition and licence
policy. This verifies provenance only: it does not execute source code, repair
reports, upgrade failed selections, or guarantee concurrent filesystem safety.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from codes.chapters.ch00.io_contracts import same_metadata, strict_json_loads, validate_parent_chain

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / 'codes/chapters/ch00/SOURCES.lock.json'
STATUS = ROOT / 'codes/chapters/ch00/SOURCE_STATUS.json'
SNAPSHOT_ROOT = ROOT / 'codes/chapters/ch00/source_snapshots'
# A correctly named file alone does not establish that a historical snapshot
# was reviewed and registered. Extend these sets only with verified raw bytes.
_HISTORICAL_SNAPSHOTS = {
    'SOURCES': frozenset({
        '55ab323ba665633141c4864763095046f9c6161ce2d88ca2aa9332dde7ec23f0',
        'e3478006c7dbc6cec442bf6bccc4df9eca946d7d353b661e947596dd8b87608a',
    }),
    'SOURCE_STATUS': frozenset({
        'e3b3176d835837441224e4906b7c2befadcdc4fe2ce6163245b7d9a9ad0d9229',
        'b113b63c97767d19b76ceb44677303961ff310ce8d96f4e9e777944b44916d3c',
    }),
}


def _sha(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('source binding requires a complete lowercase SHA-256')
    return value


def _project_id(value):
    if type(value) is not str or re.fullmatch(r'[a-z0-9][a-z0-9_-]*', value) is None:
        raise ValueError('project IDs must be ordinary nonempty identifiers')
    return value


def _ids(project_ids):
    if isinstance(project_ids, (str, bytes, dict, set, frozenset)):
        raise ValueError('project IDs must be an explicit ordered iterable')
    try:
        result = tuple(_project_id(v) for v in project_ids)
    except TypeError as error:
        raise ValueError('project IDs must be an explicit ordered iterable') from error
    if not result or len(set(result)) != len(result):
        raise ValueError('project IDs must be nonempty and unique')
    return result


def _read(path):
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError('source binding requires an existing ordinary JSON file: ' + str(path))
    raw = path.read_bytes()
    return strict_json_loads(raw), hashlib.sha256(raw).hexdigest(), path


def _records(document, *, status=False):
    if (type(document) is not dict or type(document.get('schema_version')) is not int
            or document['schema_version'] != 1 or type(document.get('projects')) is not list
            or not document['projects']):
        raise ValueError('source binding requires schema 1 with a nonempty projects list')
    records = {}
    for row in document['projects']:
        if type(row) is not dict:
            raise ValueError('project records must be JSON objects')
        name = _project_id(row.get('id'))
        if name in records:
            raise ValueError('duplicate source project ID: ' + name)
        revision = row.get('revision')
        if type(revision) is not str or re.fullmatch(r'[0-9a-f]{40}', revision) is None:
            raise ValueError('source project requires a complete fixed Git revision: ' + name)
        fields = ('status',) if status else ('url', 'license')
        if any(type(row.get(field)) is not str or not row[field] for field in fields):
            raise ValueError('source project identity fields are missing: ' + name)
        if status and 'source_selection_verified' in row and type(row['source_selection_verified']) is not bool:
            raise ValueError('source selection verification must retain its boolean type: ' + name)
        records[name] = row
    return records


def _bound_document(report_sha, current_path, snapshots, prefix):
    report_sha = _sha(report_sha)
    directory = validate_parent_chain(snapshots)
    if directory.exists() and not directory.is_dir():
        raise ValueError('source snapshots must be an ordinary directory')
    current, current_sha, _ = _read(current_path)
    if report_sha == current_sha:
        return current, current, {'report_sha256': report_sha, 'current_sha256': current_sha,
                                  'historical': False, 'snapshot_path': None}
    if report_sha not in _HISTORICAL_SNAPSHOTS[prefix]:
        raise ValueError('source report SHA-256 has no registered historical snapshot')
    saved, saved_sha, path = _read(directory / (prefix + '.' + report_sha + '.json'))
    if saved_sha != report_sha:
        raise ValueError('source snapshot bytes do not match the report SHA-256')
    return saved, current, {'report_sha256': report_sha, 'current_sha256': current_sha,
                            'historical': True, 'snapshot_path': str(path)}


def _compare_records(saved, current, ids, *, status=False):
    old, new = _records(saved, status=status), _records(current, status=status)
    for name in ids:
        if name not in old or name not in new:
            raise ValueError('requested project is absent from historical/current source records: ' + name)
        if not same_metadata(new[name], old[name]):
            raise ValueError('historical/current project record differs in content or type: ' + name)
    return {name: old[name] for name in ids}


def verify_lock_binding(report_sha, project_ids, *, current_lock=LOCK, snapshots=SNAPSHOT_ROOT):
    """Bind a report SHA and compare its explicit projects' complete lock entries.

    Returns byte-SHA evidence and ``records`` for the requested IDs. Unrelated
    new projects may be present today; changed used entries, unknown hashes,
    duplicated IDs, linked files, and fabricated snapshot bytes are rejected.
    The function neither writes nor changes the caller's report or any source.
    """
    ids = _ids(project_ids)
    saved, current, evidence = _bound_document(report_sha, current_lock, snapshots, 'SOURCES')
    evidence.update(project_ids=list(ids), records=_compare_records(saved, current, ids))
    return evidence


def verify_status_binding(status_sha, report_lock_sha, project_ids, *, current_status=STATUS,
                          current_lock=LOCK, snapshots=SNAPSHOT_ROOT):
    """Bind a complete historical acquisition report and preserve its exact state.

    Both status files must bind their actual historical/current lock respectively.
    Complete requested state records must match, including failure and ``False``
    selection flags. This does not supply missing older snapshots or turn source
    selection mismatch into successful acquisition or algorithm execution.
    """
    ids = _ids(project_ids)
    report_lock_sha = _sha(report_lock_sha)
    lock_binding = verify_lock_binding(report_lock_sha, ids, current_lock=current_lock, snapshots=snapshots)
    saved, current, evidence = _bound_document(status_sha, current_status, snapshots, 'SOURCE_STATUS')
    records = _compare_records(saved, current, ids, status=True)
    if (_sha(saved.get('lock_sha256')) != report_lock_sha
            or _sha(current.get('lock_sha256')) != lock_binding['current_sha256']):
        raise ValueError('historical/current acquisition status does not bind its actual lock')
    evidence.update(project_ids=list(ids), records=records,
                    report_lock_sha256=report_lock_sha, current_lock_sha256=lock_binding['current_sha256'],
                    lock_binding=lock_binding)
    return evidence
