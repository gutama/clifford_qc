"""Immutable, outcome-free Phase 19 declaration and temporal admission gate."""

from __future__ import annotations

import copy
import json
import subprocess

import numpy as np
import pytest

from benchmarks import check_phase19_preregistration as gate
from benchmarks import phase19_structure as structure


@pytest.fixture(scope="module")
def config():
    return gate.load_config()


def write(tmp_path, payload):
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(payload))
    return path


def test_committed_declaration_passes_without_sampling_or_state_outcomes(config, monkeypatch):
    import clifford_qc.dense_reference as dense
    import clifford_qc.states as states
    import clifford_qc.measurement.cost as cost

    def forbidden(*args, **kwargs):
        raise AssertionError("an outcome-bearing computation was attempted")

    for module, names in ((np.random, ("default_rng", "multinomial", "binomial")),
                          (np.linalg, ("eigh", "eigvalsh")),
                          (dense, ("to_matrix",)), (states, ("expectation",)),
                          (cost, ("cost_schedule", "setting_fidelity"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    assert gate.static_problems(config, historical_sources=True) == []


@pytest.mark.parametrize("bank_id,terms,qwc,cliques", [
    ("h4", 184, 68, 45), ("lih", 192, 42, 54), ("beh2", 60, 9, 36),
])
def test_structural_manifests_pin_the_declared_cover_not_the_exploratory_counts(
        config, bank_id, terms, qwc, cliques):
    row = next(r for r in config["bank_manifests"] if r["id"] == bank_id)
    bank = gate.read_json(structure.ROOT / row["path"])
    assert bank["n_qubits"] == 8 and bank["nonidentity_terms"] == terms
    assert bank["reference_occupied_spin_orbitals"] == [0, 1, 2, 3]
    assert bank["n_electrons"] == 4 and bank["ms2"] == 0
    coefficients = {code: float.fromhex(value) for code, value in bank["coefficients_hex"]}
    nonidentity = sorted(code for code in coefficients if code)
    counts = {"qwc": qwc, "clique": cliques}
    for arm in bank["arms"]:
        assert len(arm["settings"]) == counts[arm["name"]]
        assert sorted(code for s in arm["settings"] for code in s["word_codes"]) == nonidentity
        for setting in arm["settings"]:
            assert setting["word_codes"] == sorted(setting["word_codes"])
            assert setting["max_abs_value"] > 0
            assert len(setting["gate_sequence_sha256"]) == 64
            if arm["name"] == "qwc":
                assert setting["resources"]["n_2q"] == setting["resources"]["d_2q"] == 0
                assert setting["readout"]["kind"] == "weighted_sum_of_parities"
                assert setting["rotations"] == []
            else:
                assert setting["readout"]["kind"] == "weighted_parity"
                assert "word_qubits" not in setting["readout"]
                assert len(setting["rotations"]) <= len(setting["word_codes"]) - 1


@pytest.mark.parametrize("path,value", [
    (("protocol", "accuracy_hartree"), 0.01),
    (("protocol", "confidence", "delta"), 0.5),
    (("protocol", "confidence", "family"), 2),
    (("protocol", "confidence", "rounds"), 16),
    (("protocol", "allocator", "outcome_independent"), False),
    (("protocol", "allocator", "minimum_shots_per_setting"), 3),
    (("protocol", "effective_shot_endpoints"), [65536]),
    (("protocol", "seeds", "headline"), 132813000),
    (("compilation", "clique_rule"), "another-cover"),
    (("compilation", "qwc_rule"), "reverse-input-order"),
    (("acceptance", "material_shot_reduction_factor"), 1),
    (("authorization", "sampled_execution_count"), 2),
    (("authorization", "require_clean_execution_tree"), 1),
    (("protocol", "numerical_bias_allowance_hartree"), 0.0),
])
def test_protocol_keys_types_and_values_cannot_be_refitted_after_the_draw(config, tmp_path, path, value):
    broken = copy.deepcopy(config)
    node = broken
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="configuration drifted"):
        gate.load_config(write(tmp_path, broken))


@pytest.mark.parametrize("payload", [
    {"metadata": {"results": [0.2]}},
    {"lineage": [{"provenance": {"observed_radius": 0.1}}]},
    {"bank": {"exact_energy": -1.0}},
])
def test_outcome_fields_are_refused_even_inside_nested_metadata(tmp_path, payload):
    with pytest.raises(ValueError, match="result field"):
        gate.read_json(write(tmp_path, payload))


def test_unknown_fields_cannot_hide_outcomes_under_an_unrecognized_name(config, tmp_path):
    broken = copy.deepcopy(config)
    broken["scope"]["arbitrary_metadata"] = {"chosen_value": 1.2}
    with pytest.raises(ValueError, match="configuration drifted"):
        gate.load_config(write(tmp_path, broken))


@pytest.mark.parametrize("raw", [
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}',
    '{"nested":{"x":1,"x":2}}',
])
def test_json_cannot_overwrite_keys_or_hide_nonfinite_values(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        gate.read_json(path)


def test_manifest_and_input_hashes_are_binding(config, tmp_path):
    row = config["bank_manifests"][0]
    path = tmp_path / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes((structure.ROOT / row["path"]).read_bytes() + b"\n")
    binding = {key: row[key] for key in ("path", "sha256", "git_blob_sha1")}
    assert "digest drifted" in gate._binding_problems(binding, tmp_path)[0]
    path.unlink()
    assert "missing frozen file" in gate._binding_problems(binding, tmp_path)[0]
    assert len(config["lineage"]["files"]) == len(structure.INPUTS + structure.IMPLEMENTATIONS)


def test_regeneration_comparison_is_type_strict_and_has_closed_key_sets():
    assert gate._compare({"n": 1}, {"n": True})
    assert gate._compare({"n": 1}, {"n": 1.0})
    assert gate._compare({"angle": 0.2}, {"angle": 0.2, "metadata": {}})
    assert gate._compare([1, 2], [1])
    assert gate._compare(0.2, 0.20001)
    assert gate._compare(0.2, float("nan"))
    assert gate._compare(0.2, 0.2+1e-15) == []


def test_range_allocation_floor_exact_remainders_and_ties():
    assert structure.coefficient_range_schedule([1., 1., 1.], 8) == (3, 3, 2)
    assert structure.coefficient_range_schedule([1., 2.], 10) == (4, 6)
    assert structure.coefficient_range_schedule([1., 2.], 4) == (2, 2)
    assert structure.coefficient_range_schedule([1e-300, 1e300], 6) == (2, 4)
    assert structure.coefficient_range_schedule([1e-300, 2e-300], 10) == (4, 6)


@pytest.mark.parametrize("scores,budget", [
    ([], 4), ([1.], 1), ([0.], 4), ([-1.], 4), ([float("nan")], 4),
    ([float("inf")], 4), ([True], 4), ([1.], True), ([1.], 4.0),
])
def test_invalid_allocations_are_refused(scores, budget):
    with pytest.raises((ValueError, TypeError)):
        structure.coefficient_range_schedule(scores, budget)


def test_accuracy_budget_and_simultaneous_family_are_explicit(config):
    p = config["protocol"]
    assert p["confidence"]["family"] == len(config["bank_manifests"]) * len(config["compilation"]["arms"])
    assert p["confidence"]["rounds"] == len(p["effective_shot_endpoints"]) == 17
    assert p["numerical_bias_allowance_hartree"] + p["stochastic_radius_target_hartree"] == p["accuracy_hartree"]
    assert p["covariance_audit"]["replicas"] == 1000
    assert "not a Ritz derivative" in config["scope"]["estimand"]
    assert "no molecular ground-state accuracy" in config["reporting"]["evidence_boundary"]
    assert "generic 1q gate" in p["cost_boundary"]


def test_older_seed_roots_are_disjoint_and_not_recycled(config):
    old = gate.historical_config_seeds()
    assert 132813000 in old
    roots = {config["protocol"]["seeds"][name] for name in ("headline", "covariance_audit")}
    assert len(roots) == 2 and not roots & old


@pytest.fixture
def history(tmp_path, config):
    """Real merge/commit DAG, kept outside the user's worktree."""
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Phase 19 fixture")
    git("config", "user.email", "phase19@example.invalid")

    def commit(message):
        git("add", ".")
        git("commit", "-m", message)
        return git("rev-parse", "HEAD")

    (root / "baseline").write_text("base")
    commit("baseline")
    git("switch", "-c", "declaration")
    declaration_path = root / structure.CONFIG_PATH
    declaration_path.parent.mkdir(parents=True)
    declaration_path.write_text(json.dumps(config))
    declaration = commit("freeze declaration")
    git("switch", "main")
    (root / "other").write_text("main advanced")
    commit("advance main")
    git("merge", "--no-ff", "declaration", "-m", "merge declaration")
    merge = git("rev-parse", "HEAD")
    (root / "implementation").write_text("reviewed toy implementation")
    from benchmarks.archive_provenance import lineage_source_path
    for row in config["lineage"]["files"]:
        if row["path"] in structure.IMPLEMENTATIONS:
            target = root / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(lineage_source_path(structure.ROOT, structure.RECORD_PATH, row).read_bytes())
    execution = commit("execution source")
    record = {"declaration_sha256": gate.EXPECTED_CONFIG_SHA256,
              "provenance": {"git_commit": execution, "preregistration_merge_commit": merge}}
    record_path = root / structure.RECORD_PATH
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record))
    first_record = commit("record alone")
    return root, record, git, commit, declaration, merge, execution, first_record


def test_future_record_accepts_only_the_declared_merge_execution_record_order(config, history):
    root, *_ = history
    assert gate.temporal_problems(config, root) == []


@pytest.mark.parametrize("broken", ["same_execution", "nonmerge", "record_execution", "abbreviated", "digest"])
def test_future_record_rejects_lost_order_and_wrong_execution_identity(config, history, broken):
    root, record, _, _, _, merge, execution, first_record = history
    if broken == "same_execution":
        record["provenance"]["git_commit"] = merge
    elif broken == "nonmerge":
        record["provenance"]["preregistration_merge_commit"] = execution
    elif broken == "record_execution":
        record["provenance"]["git_commit"] = first_record
    elif broken == "abbreviated":
        record["provenance"]["git_commit"] = execution[:7]
    else:
        record["declaration_sha256"] = "0" * 64
    (root / structure.RECORD_PATH).write_text(json.dumps(record))
    assert gate.temporal_problems(config, root)


def test_future_record_requires_full_history(config, history):
    root, _, _, _, _, _, _, first_record = history
    (root / ".git/shallow").write_text(first_record + "\n")
    assert gate.temporal_problems(config, root) == ["record chronology requires full git history"]


def test_editing_declaration_after_record_is_refused(config, history):
    root, _, _, commit, *_ = history
    path = root / structure.CONFIG_PATH
    path.write_text(path.read_text() + "\n")
    commit("late declaration edit")
    assert gate.temporal_problems(config, root)


def test_before_a_record_exists_no_history_is_required(config, tmp_path):
    assert gate.temporal_problems(config, tmp_path) == []


def test_main_reports_static_drift_alongside_missing_history(config, history, monkeypatch, capsys):
    root, _, _, _, _, _, _, first_record = history
    (root / ".git/shallow").write_text(first_record + "\n")
    broken = copy.deepcopy(config)
    broken["protocol"]["accuracy_hartree"] = 0.01
    path = root / structure.CONFIG_PATH
    path.write_text(json.dumps(broken))
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "CONFIG", path)
    assert gate.main([]) == 1
    errors = capsys.readouterr().err
    assert "configuration drifted" in errors
    assert "requires full git history" in errors


def test_main_declaration_only_does_not_inspect_history(config, monkeypatch, tmp_path):
    path = write(tmp_path, config)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "CONFIG", path)
    monkeypatch.setattr(gate, "static_problems", lambda *args, **kwargs: [])
    monkeypatch.setattr(gate, "history_kind", lambda *args, **kwargs: pytest.fail("no record needs no history"))
    assert gate.main([]) == 0


def test_historical_sources_preserve_declaration_without_licensing_live_drift(config, frozen_phase19_checkout):
    root = frozen_phase19_checkout
    assert gate.static_problems(config, root) == []
    assert gate.static_problems(config, root, historical_sources=True) == []
    path = root / "clifford_qc/multivector.py"
    path.write_bytes(path.read_bytes() + b"# injected live drift\n")
    assert any("multivector.py" in error for error in gate.static_problems(config, root))
    assert gate.static_problems(config, root, historical_sources=True) == []


def test_live_rebuild_engine_stays_pinned_with_historical_sources(config, frozen_phase19_checkout):
    root = frozen_phase19_checkout
    path = root / "benchmarks/phase19_structure.py"
    path.write_bytes(path.read_bytes() + b"# injected engine drift\n")
    assert any("phase19_structure.py" in error
               for error in gate.static_problems(config, root, historical_sources=True))


@pytest.mark.parametrize("manifest", [[], None, {"schema": "clifford_qc.archive_bindings.v1", "records": []}])
def test_malformed_manifest_is_reported_without_traceback(config, frozen_phase19_checkout, manifest, monkeypatch, capsys):
    from benchmarks.archive_provenance import MANIFEST
    root = frozen_phase19_checkout
    (root / MANIFEST).write_text(json.dumps(manifest))
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "CONFIG", root / structure.CONFIG_PATH)
    assert gate.main(["--archive"]) == 1
    errors = capsys.readouterr().err
    assert "frozen source binding failed" in errors
    assert "Traceback" not in errors


def test_archive_gate_reads_the_manifest_once(frozen_phase19_checkout, monkeypatch):
    from benchmarks import archive_provenance
    root = frozen_phase19_checkout
    original = archive_provenance.read_json
    calls = []
    def read(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(archive_provenance, "read_json", read)
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "CONFIG", root / structure.CONFIG_PATH)
    assert gate.main(["--archive"]) == 0
    assert len(calls) == 1


def test_packaged_lineage_tampering_is_rejected(config, tmp_path):
    from benchmarks.archive_provenance import lineage_source_path
    from benchmarks.archive_provenance import MANIFEST
    import shutil
    manifest = tmp_path / MANIFEST
    manifest.parent.mkdir(parents=True)
    shutil.copyfile(structure.ROOT / MANIFEST, manifest)
    row = next(row for row in config["lineage"]["files"] if row["path"] == "clifford_qc/multivector.py")
    original = lineage_source_path(structure.ROOT, structure.RECORD_PATH, row)
    relative = original.relative_to(structure.ROOT)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(original.read_bytes() + b"# tampered\n")
    with pytest.raises(ValueError, match="frozen execution source digest differs"):
        lineage_source_path(tmp_path, structure.RECORD_PATH, row)
