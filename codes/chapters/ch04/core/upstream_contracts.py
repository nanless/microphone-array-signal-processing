"""Shared fixed-source and finite write-preflight contracts, first used in chapter 4.

Used original blobs, complete acquisition selection, and actual method execution
are separate records. Checks do not eliminate concurrent filesystem races.
No fetching, installation, upstream edits, or implicit report writes occur here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from codes.chapters.ch00.io_contracts import (
    strict_json_loads, same_metadata, validate_parent_chain, validate_report_destination,
    write_json_report,
)
from codes.chapters.ch00.upstream.fetch_upstreams import run_git, sparse_patterns, validate_project

ROOT = Path(__file__).resolve().parents[4]
LOCK = ROOT / "codes/chapters/ch00/SOURCES.lock.json"
STATUS = LOCK.with_name("SOURCE_STATUS.json")
CACHE = LOCK.parent / "upstream/_downloads"
HISTORICAL_REVISION = "e4d54a158d892b224e37cc49b93066e667babf08"
PROJECTS = {
    "pyaec": ("https://github.com/ewan-xu/pyaec.git",
        "5b9c02c57075d790b7df8652884618189d49bbc4", "Apache-2.0", "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "echocatzh-pfdkf": ("https://github.com/echocatzh/PFDKF.git",
        "7c8c86b5691966c330015d8e0960db0733b4844f", "MIT", "LICENSE",
        "b6c53ea4c5cf363eeef563f592f0b7308ad0943627e7803e9b0df05bf592b5d5"),
    "dtln_aec": ("https://github.com/breizhn/DTLN-aec.git",
        "9d24e128b4f409db18227b8babb343016625921f", "MIT", "LICENSE",
        "aa95acd8c8a7341bfcdb2823694dcbb27a2a4143e860bb52db1ff0f29c85e5a1"),
    "pb_bss": ("https://github.com/fgnt/pb_bss.git",
        "10acc347fc9ea21e3d312806a0bd751d0d0af183", "MIT", "LICENSE",
        "48241e1eae6ab4c15c5718992ca60d3d15961212a4e324e09e5dbfbc79b78214"),
    "sof": ("https://github.com/thesofproject/sof.git",
        "b6c6a05d52536313fe8e8752b1c4e069b1cc4002",
        "BSD-3-Clause and per-file licenses", "LICENCE",
        "cb6a9aa5baac3ea573feebcdc298526f701e2b7cd005a22208799778e26f0068"),
    "pyroomacoustics": ("https://github.com/LCAV/pyroomacoustics.git",
        "0dd39f2614b7fc44b2cc63dbe7d60f4641068890", "MIT", "LICENSE",
        "0922c9a0c1f5bb35a1e0df7b56954864e27cfc625fc3b041acde0707ce797e46"),
    "doatools": ("https://github.com/morriswmz/doatools.py.git",
        "9469db201e0418aef6b97583ef54b6fec2769502", "MIT", "LICENSE.md",
        "39ddd5dcfeb6eb77f18c99e8955341dc7b4d4823a3a121e0124e20de336b507a"),
    "sbl": ("https://github.com/gerstoft/SBL.git",
        "d4bba35e9b60907d3024473ba5a41046450baae0", "GPL-3.0", "LICENSE",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
    "smpphat": ("https://github.com/FrancoisGrondin/smpphat.git",
        "6fd33e6eb3251078a4cd9793dde909e2500265cc", "GPL-3.0", "LICENSE",
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
    "said-spatial-imaging": ("https://github.com/IN03X/SAID.git",
        "cf52ede4f38361cdbb03aa93c8254109583dfe2b",
        "MIT AND Apache-2.0 (source components); separate non-commercial model-weight terms", "LICENSE",
        "d3757b5359fd0d6d1ed7dbe9b37da90b11c4a6cabaa11081b26b5f6aa95794af"),
    'btk20': ('https://github.com/kkumatani/distant_speech_recognition.git', 'feff19ec8bcb770f6530fe280dc3ccafc2f5984a', 'MIT; retain per-file notices', 'LICENSE', '1dfe95044a48d8c90dde9fe0d32fa75beba51633b866cc91f10b803489edeec0'),
    'espnet': ('https://github.com/espnet/espnet.git', 'be79590bb2ff26ffb01bc825c5f68cb9418b7f0d', 'Apache-2.0', 'LICENSE', '4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5'),
    'gss': ('https://github.com/desh2608/gss.git', '10fad18cae85e2e4342c77421abc70c9c5da23ed', 'MIT', 'LICENSE', '5658d3e38fcd75f27608d9ba7ee83a2c1a04cc7a76334509907b41c87b92fc8b'),
    'metaaf': ('https://github.com/adobe-research/MetaAF.git', '56c4665bdc51c2e0595a7c0cd9b1266408adceff', 'LicenseRef-NCSA-and-Adobe-Research', 'metaaf/LICENSE', '4281265bd0d7692b781b56f3395f9673e9a67e74b10348e470a6c43a4259d3fb'),
    'nara_wpe': ('https://github.com/fgnt/nara_wpe.git', 'a166779cca2088817e330481bd20af1a2c598555', 'MIT', 'LICENSE', 'f9fad5befd31a5c5540fb671fa56efd59d959959ffe1b6a8156a66549d6410f0'),
    'nemo_wpe': ('https://github.com/NVIDIA/NeMo.git', '2381f42f6979449b5b99538f8f80135831009b51', 'Apache-2.0', 'LICENSE', '43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1'),
    'speexdsp': ('https://gitlab.xiph.org/xiph/speexdsp.git', '8e29a256ef0235ebbe7fcb8417b5ac7731eb8307', 'BSD-3-Clause', 'COPYING', '2654a4264b2bfe298dedc508748d140111840c315cc8eb646a3a68c13fa75b01'),
    'tso_vace_wpe': ('https://github.com/dreadbird06/tso_vace_wpe.git', '10ee77dd020d58af508feb77251f9353208cb33a', 'MIT; retain LICENSE and per-file notices', 'LICENSE', '664fb260188108b8c9b3e21ab0c8c8f57c94b421f8134dc1544ede4627d2713b'),
    'ssspy': ('https://github.com/tky823/ssspy.git', '38b9389e8b1914422561f1936d9b28d042d62d2c', 'Apache-2.0', 'LICENSE', '809ba860a2750092fd00a8ca4cb4dc459ae4f9f6e538f574ea52b8b1daae91b6'),
    'arraydps': ('https://github.com/ArrayDPS/ArrayDPS.git', '750ac2b7c75458f4ca5bad203dafda528f575e55', 'MIT', 'LICENSE', '5d40b048a5fb04cc63482634406ddc2d9df2b14eddfe600c4f40a66872a4de0e'),
    'asteroid': ('https://github.com/asteroid-team/asteroid.git', 'fce87469132760fbab41c20616ea0f0e079aad38', 'MIT', 'LICENSE', 'c12aebc7a4eeeec2e482414004fd5d68275d7608552031cd48f4088b403f902d'),
    'mamba_tasnet': ('https://github.com/xi-j/Mamba-TasNet.git', 'a35c692f27213781a11b1606c375cda1e1f0fb62', 'GPL-3.0', 'LICENSE', '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'),
    'notsofar1': ('https://github.com/microsoft/NOTSOFAR1-Challenge.git', '6f58e08b008f7530ba4141f0aeb02447c70b6fd7', 'MIT code; DATA_LICENSE and dataset-version restrictions separate', 'LICENSE', 'c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383'),
    'streamfm': ('https://github.com/sp-uhh/streamfm.git', 'ab2700c1154acc5c2ce67a5344182028336413f5', 'AGPL-3.0; weights and datasets separately', 'LICENSE', '8486a10c4393cee1c25392769ddd3b2d6c242d6ec7928e1414efff7dfb2f07ef'),
    'tf-locoformer': ('https://github.com/merlresearch/tf-locoformer.git', '7a615460d347ff7334a13dbb831d16280da72cdc', 'Apache-2.0 and per-file SPDX notices; weights and data separately', 'LICENSE.md', '0b3204d175388251410569d1e6ae8efc9e7e4ae2572ef11ace4015e0ca8d61b7'),
    'overiva-author': ('https://github.com/onolab-tmu/overiva.git', '1cb3189112889ebfe2ccbbde0f55b1db6020fc48', 'MIT', 'LICENSE', 'a872a2a232076ca5034f53a33709efecfe61c7481dbda84b595549f7015f03bc'),
    'odas': ('https://github.com/introlab/odas.git', 'bcb845434495e293df3d48f1203b7a86e1852449', 'MIT', 'LICENSE', 'd80055b1eac9fdb001f429f262c79ce6a7619b85064a759f17b0b33bbb9c8b2a'),
    'spatial-audio-framework': ('https://github.com/leomccormack/Spatial_Audio_Framework.git', '18fd5aba46e20787b51f28f7197a68506c965c07', 'ISC core; GPL-2.0-or-later optional tracker module (individual headers control)', 'LICENSE.md', '0148c9bfe5f2093cd06cecd3359bc16bc29a2200863a59af81a90b7922c67af3'),
    'filterpy': ('https://github.com/rlabbe/filterpy.git', '3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33', 'MIT', 'LICENSE', '8ffce1097f1b1c0fba42e4449ef49aa05cb7b45566dd0989407839dba07af99c'),
    'stonesoup': ('https://github.com/dstl/Stone-Soup.git', '8d1edeb07ef8505ed065cbef435cfb5e517d9bdc', 'MIT', 'LICENSE', '2462f3d8a857f601e266f048a4ff051366c1e05f098cf3a1524eaf922f879815'),

    'pystoi': ('https://github.com/mpariente/pystoi.git', '74872b000753a7a42ff51aa0868af8c82c7f9053', 'MIT for Python core; MATLAB test notices separate', 'LICENSE', '35c25f6087c4e1857ba6d4bc0b7957b7bc523c9ff190a9e2cccb4b5ecba018ab'),
    'deepfilternet': ('https://github.com/Rikorose/DeepFilterNet.git', 'd375b2d8309e0935d165700c91da9de862a99c31', 'Apache-2.0 OR MIT', 'LICENSE-MIT', '24e6bb09c928af8d8e56268082f87413247ce36b39dd5d33add2f9893968065e'),
    'cmsis_dsp': ('https://github.com/ARM-software/CMSIS-DSP.git', '83a2d7bc98c81b4bbe4a6f48b1f2ecf179868a0b', 'Apache-2.0', 'LICENSE', 'b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1'),
    'webrtc': ('https://webrtc.googlesource.com/src', '0467d2b91cc20b9b001c2bbb73d43ea6b2491f3e', 'BSD-3-Clause', 'LICENSE', 'ab00a482b6a3902e40211b43c5d0441962ea99b6cc7c25c0f243fa270b78d482'),
    'rnnoise': ('https://gitlab.xiph.org/xiph/rnnoise.git', '70f1d256acd4b34a572f999a05c87bf00b67730d', 'BSD-3-Clause', 'COPYING', '45d37ca1cdb278c088e1aa85e0e65ca3a534ed86a28dcc96ca16810248a61d35'),
    'fastenhancer': ('https://github.com/aask1357/fastenhancer.git', 'f85223bd546b27f39dc0744e0310dcd246f750a4', 'MIT; released weights and datasets separately', 'LICENSE', '38164c22720cdccceb5f0dbb8f7218dcf17bac97ac536cad2e77c33dfbc45a18'),
    'libsoxr': ('https://git.code.sf.net/p/soxr/code', '945b592b70470e29f917f4de89b4281fbbd540c0', 'LGPL-2.1-or-later; embedded PFFFT terms separate', 'LICENCE', 'dc98676341fdcd29d9f279c9679d6a75288785b174ded8d1b2e316c366166135'),
    'libebur128': ('https://github.com/jiixyj/libebur128.git', '67b33abe1558160ed76ada1322329b0e9e058b02', 'MIT', 'COPYING', 'd6b4754bb67bdd08b97d5d11b2d7434997a371585a78fe77007149df3af8d09c'),
    'libsndfile': ('https://github.com/libsndfile/libsndfile.git', 'b9103bd48b6c8fb517ae737fe3baee0c718b804c', 'LGPL-2.1-or-later; dependencies separately', 'COPYING', 'ad01ea5cd2755f6048383c8d54c88459cd6fcb17757c5c8892f8c5ea060f6140'),
    'libsamplerate': ('https://github.com/libsndfile/libsamplerate.git', '0844c208f683527c08ea8a80acc13b398aa9c8bf', 'BSD-2-Clause', 'COPYING', '2c1f76ce2effdddb425018405d5690c0b1ab4e6976e35296b0a6db65c5e1a55d'),
    'stk': ('https://github.com/thestk/stk.git', '6aacd357d76250bb7da2b1ddf675651828784bbc', 'MIT-STK; see per-file notices', 'LICENSE', '342901da98ec7c044426f3d1c2e1b8dd2ac79001e0916b49c2cf71b158b13aa5'),
    'meeteval': ('https://github.com/fgnt/meeteval.git', '6e3dc81284f2d6928f7ef9e620fd3b6906daa429', 'MIT', 'LICENSE', '5515c2bdda551fa20d771e99ab31deebbb4f9626bef6f6c25f26125c964fd34c'),
    'chime-utils': ('https://github.com/chimechallenge/chime-utils.git', '152882404f572d40769ef02bf91c5a9a9cfc9c78', 'MIT', 'LICENSE', 'e2f48f797fd76ec12f6792c39555e986b8b4329604194e7da1416837b68c8ee4'),
}


def ordinary_file(path):
    path = validate_parent_chain(path)
    if not path.is_file():
        raise ValueError(f"Existing ordinary source file required: {path}")
    return path


def sha(path):
    return hashlib.sha256(ordinary_file(path).read_bytes()).hexdigest()


def git(checkout, *arguments):
    return run_git(list(arguments), cwd=checkout)


def source_documents(lock_path=None, status_path=None):
    lock_path = LOCK if lock_path is None else lock_path
    status_path = STATUS if status_path is None else status_path
    lock_bytes = ordinary_file(lock_path).read_bytes()
    status_bytes = ordinary_file(status_path).read_bytes()
    lock, status = strict_json_loads(lock_bytes), strict_json_loads(status_bytes)
    for document in (lock, status):
        if (type(document) is not dict or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1 or type(document.get("projects")) is not list
                or any(type(row) is not dict for row in document["projects"])):
            raise ValueError("Source documents require schema 1 and object project records")
        ids = [row.get("id") for row in document["projects"]]
        if any(type(i) is not str for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Source documents require unique string project IDs")
    digest = hashlib.sha256(lock_bytes).hexdigest()
    if status.get("lock_sha256") != digest:
        raise ValueError("Current source status must bind current lock bytes")
    return lock, status, digest, hashlib.sha256(status_bytes).hexdigest()


def record_files(identity, relatives):
    checkout = Path(identity["checkout"])
    for relative in relatives:
        if (type(relative) is not str or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise ValueError("Original source path must be relative to its checkout")
        path = ordinary_file(checkout / relative)
        payload = path.read_bytes()
        expected = git(checkout, "rev-parse", f"{identity['head']}:{relative}")
        actual = git(checkout, "hash-object", "--no-filters", "--", str(path))
        if actual != expected:
            raise ValueError(f"Original source differs from fixed Git blob: {relative}")
        record = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                  "git_blob": expected, "actual_blob": actual}
        previous = identity["used_files"].get(relative)
        if previous is not None and previous != record:
            raise ValueError(f"Original source changed during execution: {relative}")
        identity["used_files"][relative] = record


def verify_project(project, checkout=None, relatives=(), *, all_python=False,
                   lock_path=None, status_path=None):
    """Check fixed identity before use; never upgrade selection mismatch."""
    lock, status, lock_digest, status_digest = source_documents(lock_path, status_path)
    url, revision, license_name, license_path, license_digest = PROJECTS[project]
    entries = [p for p in lock["projects"] if p["id"] == project]
    records = [p for p in status["projects"] if p["id"] == project]
    if len(entries) != 1 or len(records) != 1:
        raise ValueError("Expected exactly one lock and status project record")
    entry, recorded = entries[0], records[0]
    validate_project(entry)
    if (entry["url"], entry["revision"], entry["license"]) != (url, revision, license_name):
        raise ValueError(f"Review changed fixed source lock: {project}")
    checkout = validate_parent_chain(CACHE / project if checkout is None else checkout)
    validate_parent_chain(checkout / ".git")
    if not checkout.is_dir() or not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"Existing independent checkout required: {checkout}")
    if git(checkout, "rev-parse", "--show-toplevel") != str(checkout.resolve()):
        raise ValueError("Source directory must be its own Git top level")
    origin, head = git(checkout, "remote", "get-url", "origin"), git(checkout, "rev-parse", "HEAD")
    if (origin, head) != (url, revision):
        raise ValueError("Original source origin or HEAD differs from fixed identity")
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Original source worktree must be entirely clean")
    ignored = git(checkout, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
    sparse_file = validate_parent_chain(checkout / ".git/info/sparse-checkout")
    sparse = sparse_file.exists()
    patterns = ordinary_file(sparse_file).read_text().splitlines() if sparse else []
    expected = sparse_patterns(entry) if sparse else []
    matches = patterns == expected if sparse else not entry.get("source_paths")
    missing = [p for p in entry.get("entrypoints", []) if not (checkout / p).exists()]
    live = {"id": project, "revision": revision,
            "status": "source_selection_mismatch" if not matches else
                      "entrypoints_missing" if missing else "source_verified",
            "source_selection_verified": bool(matches), "missing_entrypoints": missing,
            "observed_sparse_patterns": patterns, "expected_sparse_patterns": expected,
            "requested_source_paths": entry.get("source_paths", ["repository"]),
            "asset_policy": "sparse_source_checkout" if sparse else "existing_checkout_preserved"}
    if checkout.resolve() == (CACHE / project).resolve():
        if any(not same_metadata(recorded.get(k), v) for k, v in live.items()):
            raise ValueError("Recorded complete selection differs from actual checkout")
    identity = {"project": project, "checkout": str(checkout.resolve()), "origin": origin,
                "head": head, "lock_entry": entry, "lock_sha256": lock_digest,
                "source_document_paths": {"lock": str(LOCK if lock_path is None else lock_path),
                                          "status": str(STATUS if status_path is None else status_path)},
                "status_sha256": status_digest, "recorded_complete_selection": recorded,
                "live_complete_selection": live, "used_source_identity": "verified",
                "file_identity_scope": "declared original files plus tracked Python preflight when requested; does not claim every verified file executed",
                "git_clean_scope": "tracked and nonignored untracked files; ignored members listed separately",
                "ignored_members_before": ignored,
                "used_files": {}, "clean_before": True, "clean_after": False}
    names = list(relatives)
    if all_python:
        names.extend(git(checkout, "ls-files", "--", "*.py").splitlines())
    record_files(identity, [license_path, *dict.fromkeys(names)])
    if identity["used_files"][license_path]["sha256"] != license_digest:
        raise ValueError("Original license differs from fixed verified SHA-256")
    return identity


def check_unchanged(identity):
    checkout = Path(identity["checkout"])
    if (git(checkout, "rev-parse", "HEAD") != identity["head"]
            or git(checkout, "remote", "get-url", "origin") != identity["origin"]
            or git(checkout, "status", "--porcelain", "--untracked-files=all")):
        raise ValueError("Original source changed during execution")
    if (sha(identity['source_document_paths']['lock']) != identity['lock_sha256']
            or sha(identity['source_document_paths']['status']) != identity['status_sha256']):
        raise ValueError("Source lock or recorded selection changed during execution")
    ignored_after = git(checkout, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
    if ignored_after != identity['ignored_members_before']:
        raise ValueError("Ignored source membership changed during execution")
    identity['ignored_members_after'] = ignored_after
    record_files(identity, tuple(identity["used_files"]))
    identity["clean_after"] = True
    return identity


def dependencies(script, extras=()):
    paths = (script, Path(__file__), ROOT / "codes/chapters/ch00/io_contracts.py",
             ROOT / "codes/chapters/ch00/upstream/fetch_upstreams.py", *extras)
    return {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in paths}


def report_target(path, current=None, protected=()):
    target = validate_report_destination(path, forbidden_roots=protected)
    if target.resolve().is_relative_to(ROOT.resolve()):
        if current is None or target.resolve() != Path(current).resolve():
            raise ValueError("Only the designated new current report may be written inside the repository")
    return target


def write_report(path, data, current=None, protected=()):
    # Repeat the finite preflight immediately before shared atomic replacement.
    target = report_target(path, current, protected)
    write_json_report(target, data, forbidden_roots=protected)


def work_target(path, protected=()):
    target = validate_parent_chain(path)
    if target.resolve().is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(target.resolve()):
        raise ValueError("Temporary work directory must be outside the repository")
    if target.exists() and not target.is_dir():
        raise ValueError("Temporary work destination must be an ordinary directory")
    for root in protected:
        root = Path(root).resolve()
        if target.resolve().is_relative_to(root) or root.is_relative_to(target.resolve()):
            raise ValueError("Temporary work directory overlaps original source")
    return target


def historical_bytes(path):
    """Read the real pre-edit Git object; strip handling preserves exact bytes."""
    relative = Path(path).resolve().relative_to(ROOT)
    revision_path = HISTORICAL_REVISION + ":" + str(relative)
    payload = (git(ROOT, "show", revision_path) + "\n").encode()
    if len(payload) != int(git(ROOT, "cat-file", "-s", revision_path)):
        raise ValueError("Historical Git object cannot be represented byte-exactly")
    return payload
