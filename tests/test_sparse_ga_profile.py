"""Profile audit catches corrupt decisions/counts without timing thresholds."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from benchmarks import check_sparse_ga_kernel as gate
from benchmarks.archive_provenance import MANIFEST, history_kind, safe_path
from benchmarks.metadata_io import binding, read_json


def _archive_mode(root=gate.ROOT):
    """Choose explicit archive verification only for genuinely Git-free exports."""
    try:
        history_kind(root)
    except ValueError:
        # This still rejects shallow checkouts and broken/parent Git metadata.
        history_kind(root, archive=True)
        return True
    return False


def _check(record, root=gate.ROOT):
    return gate.check_record(record, root, archive=_archive_mode(root))


@pytest.fixture
def record():
    return read_json(gate.ROOT / "benchmarks/reference_results/sparse_ga_kernel.json")


@pytest.fixture
def sparse_archive():
    # Use system scratch (or TMPDIR), outside every enclosing checkout. Broken
    # parent Git metadata must remain a failure rather than be hidden here.
    with tempfile.TemporaryDirectory(prefix="sparse-ga-archive-") as directory:
        root = Path(directory)
        manifest = read_json(gate.ROOT / MANIFEST)
        entry = manifest["records"][gate.RECORD_PATH]
        paths = {gate.RECORD_PATH, entry["config"]["path"]}
        paths.update(item["snapshot"] for item in entry["sources"].values())
        for relative in paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(gate.ROOT, relative), target)
        target = root / MANIFEST
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema": manifest["schema"],
                                      "records": {gate.RECORD_PATH: entry}}))
        assert history_kind(root, archive=True) == "archive"
        yield root


def test_committed_profile_and_large_sparse_bounds_are_valid(record):
    _check(record)
    large = next(row for row in record["fixtures"] if row["fixture"] == "n12_grade12_shared")
    assert large["candidate_count"] == 34 <= large["candidate_bound"] == 36


@pytest.mark.parametrize("failure", ["decision", "count", "definition", "cache", "nonfinite", "order", "source", "scope"])
def test_profile_refuses_corrupt_values_and_bindings(record, failure):
    broken = deepcopy(record)
    row = broken["fixtures"][0]
    measured = row["measurements"]["active"]
    if failure == "decision":
        measured["is_blade"] = False
    elif failure == "count":
        row["candidate_count"] += 1
    elif failure == "definition":
        row["masks"][0] = 0
    elif failure == "cache":
        measured["whole_grade_cache_entries"] = 1
    elif failure == "nonfinite":
        measured["cold_ns"] = float("inf")
    elif failure == "order":
        measured["warm_ns"]["min"] = measured["warm_ns"]["max"] + 1
    elif failure == "source":
        broken["source_files"][0]["sha256"] = "0" * 64
    else:
        broken["scope"] = "quantum speedup established"
    with pytest.raises(ValueError):
        _check(broken)


def test_timings_are_observations_without_fixed_performance_thresholds(record, sparse_archive):
    for row in record["fixtures"]:
        for measured in row["measurements"].values():
            measured["cold_ns"] = 10**15
            measured["warm_ns"] = {"min": 10**12, "median": 10**13, "max": 10**14}
            measured["cold_tracemalloc_peak_bytes"] = 10**12
    # Bind these altered observations coherently in an isolated test archive;
    # they must pass numerical validation without weakening content checks.
    path = sparse_archive / gate.RECORD_PATH
    path.write_text(json.dumps(record))
    manifest_path = sparse_archive / MANIFEST
    manifest = read_json(manifest_path)
    manifest["records"][gate.RECORD_PATH]["record_sha256"] = binding(path)["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    gate.check_record(record, sparse_archive, archive=True)


@pytest.mark.parametrize("failure", ["dirty", "missing_commit", "wrong_commit"])
def test_execution_identity_must_be_clean_and_contain_the_sources(record, failure):
    if failure == "dirty":
        record["provenance"]["git_dirty"] = True
    else:
        record["provenance"]["git_sha"] = ("0" * 40 if failure == "missing_commit"
                                              else "5793459cfc3ab59f900af56b4ce30699345b6f1e")
    with pytest.raises(ValueError):
        _check(record)


def test_later_worktree_changes_do_not_rebind_historical_measurements(record, tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-b", "main")
    git("config", "user.name", "Profile fixture")
    git("config", "user.email", "profile@example.invalid")
    git("config", "commit.gpgsign", "false")
    execution = record["provenance"]["git_sha"]
    archive = _archive_mode()
    if archive:
        gate.check_record(record, archive=True)
        entry = read_json(gate.ROOT / MANIFEST)["records"][gate.RECORD_PATH]
    for source in record["source_files"]:
        if archive:
            raw = safe_path(gate.ROOT, entry["sources"][source["path"]]["snapshot"]).read_bytes()
        else:
            raw = subprocess.check_output(["git", "show", f"{execution}:{source['path']}"], cwd=gate.ROOT)
        target = tmp_path / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git("add", ".")
    git("commit", "-m", "measured source")
    record["provenance"]["git_sha"] = git("rev-parse", "HEAD")
    (tmp_path / "clifford_qc/multivector.py").write_text("# a later unrelated source edit\n")
    gate.check_record(record, tmp_path)


def test_live_candidate_enumeration_regression_is_observed(record, monkeypatch):
    from clifford_qc import multivector as kernel
    original = kernel._blade_contraction_buckets
    def extra(*args):
        buckets = original(*args)
        buckets[0] = []  # Extra work leaves the blade decision unchanged.
        return buckets
    monkeypatch.setattr(kernel, "_blade_contraction_buckets", extra)
    with pytest.raises(ValueError, match="live kernel candidate enumeration differs"):
        _check(record)


@pytest.mark.parametrize("warm", [[1, 2, 3], None, "timing"])
def test_malformed_warm_timing_is_a_clean_cli_failure(record, tmp_path, capsys, warm):
    record["fixtures"][0]["measurements"]["active"]["warm_ns"] = warm
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(record))
    mode = ["--archive"] if _archive_mode() else []
    assert gate.main(["--record", str(path), *mode]) == 1
    errors = capsys.readouterr().err
    assert "warm timing shape differs" in errors and "Traceback" not in errors


def test_producer_refuses_dirty_sources_before_measurement(monkeypatch):
    from benchmarks import run_sparse_ga_kernel as producer
    original = producer.subprocess.check_output
    def dirty(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        if command[:3] == ["git", "status", "--porcelain"]:
            return " M clifford_qc/multivector.py\n"
        return original(command, **kwargs)
    monkeypatch.setattr(producer.subprocess, "check_output", dirty)
    with pytest.raises(ValueError, match="clean committed source tree"):
        producer.clean_execution()


def test_archive_requires_explicit_mode_and_binds_the_inspected_record(record, sparse_archive):
    with pytest.raises(ValueError, match="full Git history required"):
        gate.check_record(record, sparse_archive)
    gate.check_record(record, sparse_archive, archive=True)
    changed = deepcopy(record)
    changed["fixtures"][0]["measurements"]["active"]["cold_ns"] += 1
    with pytest.raises(ValueError, match="inspected record contents"):
        gate.check_record(changed, sparse_archive, archive=True)


@pytest.mark.parametrize("target", ["record", "configuration", "source", "missing", "manifest"])
def test_archive_detects_tampered_or_missing_content(record, sparse_archive, target):
    manifest_path = sparse_archive / MANIFEST
    manifest = read_json(manifest_path)
    entry = manifest["records"][gate.RECORD_PATH]
    if target == "manifest":
        entry["sources"][record["source_files"][0]["path"]]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        relative = (gate.RECORD_PATH if target == "record" else gate.CONFIG_PATH
                    if target == "configuration" else next(iter(entry["sources"].values()))["snapshot"])
        path = sparse_archive / relative
        if target == "missing":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        gate.check_record(record, sparse_archive, archive=True)


def test_archive_cannot_downgrade_broken_or_shallow_git(record, sparse_archive, tmp_path):
    (sparse_archive / ".git").write_text("gitdir: missing\n")
    for archive in (False, True):
        with pytest.raises(ValueError, match="unavailable/incomplete"):
            gate.check_record(record, sparse_archive, archive=archive)
    (sparse_archive / ".git").unlink()
    for arguments in (("init", "-b", "main"), ("config", "user.name", "Sparse archive fixture"),
                      ("config", "user.email", "archive@example.invalid"),
                      ("config", "commit.gpgsign", "false"), ("add", "."),
                      ("commit", "-m", "fixture")):
        subprocess.run(["git", *arguments], cwd=sparse_archive, capture_output=True, check=True)
    with pytest.raises(ValueError, match="cannot bypass"):
        gate.check_record(record, sparse_archive, archive=True)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "--depth=1", sparse_archive.as_uri(), str(shallow)],
                   capture_output=True, check=True)
    for archive in (False, True):
        with pytest.raises(ValueError, match="full git history"):
            gate.check_record(record, shallow, archive=archive)


def test_real_export_cli_reports_unverified_chronology_and_refuses_tampering(sparse_archive):
    for source in (gate.ROOT / "clifford_qc").rglob("*.py"):
        target = sparse_archive / source.relative_to(gate.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("check_sparse_ga_kernel", "archive_provenance", "metadata_io", "sparse_ga_audit"):
        shutil.copyfile(gate.ROOT / "benchmarks" / f"{name}.py",
                        sparse_archive / "benchmarks" / f"{name}.py")
    script = sparse_archive / "benchmarks/check_sparse_ga_kernel.py"
    for arguments, expected in (([], 1), (["--require-git"], 1), (["--archive"], 0)):
        result = subprocess.run([sys.executable, str(script), *arguments], cwd=sparse_archive,
                                capture_output=True, text=True)
        assert result.returncode == expected, result.stdout + result.stderr
        if expected == 0:
            assert "commit chronology UNVERIFIED" in result.stdout
        else:
            assert "full Git history required" in result.stderr
    original = sparse_archive / gate.RECORD_PATH
    modified = read_json(original)
    modified["fixtures"][0]["measurements"]["active"]["cold_ns"] += 1
    alternate = sparse_archive / "alternate.json"
    alternate.write_text(json.dumps(modified))
    result = subprocess.run([sys.executable, str(script), "--archive", "--record", str(alternate)],
                            cwd=sparse_archive, capture_output=True, text=True)
    assert result.returncode == 1
    assert "content bindings verified" not in result.stdout
    original.write_text(json.dumps(modified))
    result = subprocess.run([sys.executable, str(script), "--archive"], cwd=sparse_archive,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "archive record digest differs" in result.stderr
