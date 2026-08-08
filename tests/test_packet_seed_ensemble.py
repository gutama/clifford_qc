import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import run_packet_seed_ensemble as ensemble
from benchmarks import summarize_packet_ensemble as summary


def _cell(seed, ordering, log_ratio):
    return {
        "system": "toy",
        "ordering": ordering,
        "shots": 64,
        "seed": seed,
        "packet_eligible": True,
        "matched_M": True,
        "packet_error": 10.0 ** log_ratio,
        "dressed_error": 1.0,
        "packet_selection_work": 5,
        "dressed_selection_work": 4,
        "packet_directions": 1,
        "log_ratio": log_ratio,
    }


def test_seed_cluster_inference_does_not_pseudoreplicate_orderings():
    cells = []
    for seed in range(8):
        cells.extend([
            _cell(seed, "physics", -0.1),
            _cell(seed, "graph", -0.1),
            _cell(seed, "probability", -0.1),
        ])

    row = summary._summarize(cells, "policy")

    assert row["compared"] == 24
    assert row["seed_clusters"] == 8
    assert row["seed_wins"] == 8
    assert row["sign_test_p"] == pytest.approx(0.0078125)
    assert row["median_log_ratio"] == pytest.approx(-0.1)
    assert row["ci_low"] == pytest.approx(-0.1)
    assert row["ci_high"] == pytest.approx(-0.1)


def test_summary_reports_full_factorization_and_unadjusted_secondary_pvalues(tmp_path):
    path = tmp_path / "ensemble.jsonl"
    rows = [{
        "record": "header",
        "schema": "clifford_qc.packet_seed_ensemble.v2",
        "evidence": "oracle_sampled; selector behaviour only",
        "claim_boundary": "test boundary",
        "provenance": {"git_sha": "abc", "git_dirty": False},
    }]
    for seed in range(8):
        rows.extend([
            _cell(seed, "physics", -0.1),
            _cell(seed, "random", 0.0),
        ])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    text, groups = summary.summarize([path])

    assert "complete 1 systems x 2 orderings x 1 shot settings x 8 seed clusters" in text
    assert "p-values are unadjusted across secondary groups" in text
    assert groups[0]["seed_clusters"] == 8


def test_duplicate_treatment_cells_across_input_files_are_rejected(tmp_path):
    row = _cell(0, "physics", -0.1)
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        header = {
            "record": "header",
            "schema": "clifford_qc.packet_seed_ensemble.v2",
            "systems": ["toy"],
            "orderings": ["physics"],
            "shots": [64],
            "seeds": 1,
        }
        path.write_text(json.dumps(header) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate treatment cell"):
        summary.summarize(paths)


def test_publication_ensemble_includes_requested_molecular_systems():
    assert "h2o_qsci" in ensemble.DEFAULT_SYSTEMS
    assert "beh2_stretched" in ensemble.DEFAULT_SYSTEMS


def test_probability_ordering_uses_the_state_that_was_sampled(monkeypatch):
    captured = {}
    backend = SimpleNamespace(
        basis=np.array([1, 2], dtype=np.int64),
        state_from_program=lambda _program: np.array([1.0, 0.0]),
    )
    model = SimpleNamespace(reference=object(), n=2)
    operator = SimpleNamespace(restrict=lambda _indices: np.eye(2))
    state = SimpleNamespace(amplitudes=np.array([0.6, 0.8], dtype=complex))

    def ordering(words, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(words=np.asarray(words))

    monkeypatch.setattr(ensemble, "configuration_ordering", ordering)
    monkeypatch.setattr(
        ensemble, "configuration_generators_from_words",
        lambda *_args, **_kwargs: [SimpleNamespace()])
    monkeypatch.setattr(
        ensemble, "dressed_family",
        lambda *_args, **_kwargs: SimpleNamespace(generators=[]))

    ensemble._ordered_inputs(
        model, backend, operator, state, np.array([1, 2]), np.array([0, 1]),
        method="probability", max_support=4, max_generators=4, seed=0)

    assert captured["probabilities"] == pytest.approx([0.36, 0.64])


def test_incomplete_grid_is_rejected(tmp_path):
    path = tmp_path / "truncated.jsonl"
    header = {
        "record": "header",
        "schema": "clifford_qc.packet_seed_ensemble.v2",
        "systems": ["toy"],
        "orderings": ["physics"],
        "shots": [64],
        "seeds": 2,
    }
    path.write_text(json.dumps(header) + "\n" + json.dumps(_cell(0, "physics", -0.1)) + "\n")

    with pytest.raises(ValueError, match="incomplete treatment grid"):
        summary.summarize([path])


def test_missing_matched_budget_flag_is_rejected():
    cell = _cell(0, "physics", -0.1)
    del cell["matched_M"]
    with pytest.raises(ValueError, match="matched_M"):
        summary._summarize([cell], "broken")


def test_shot_groups_use_balanced_system_subset(tmp_path):
    path = tmp_path / "unbalanced.jsonl"
    header = {
        "record": "header",
        "schema": "clifford_qc.packet_seed_ensemble.v2",
        "systems": ["old", "new"],
        "orderings": ["physics"],
        "shots": [32, 128],
        "seeds": 8,
    }
    rows = []
    for seed in range(8):
        for shots in (32, 128):
            cell = _cell(seed, "physics", -0.02)
            cell.update(system="old", shots=shots)
            rows.append(cell)
        cell = _cell(seed, "physics", 0.2)
        cell.update(system="new", shots=128)
        rows.append(cell)
    # This is intentionally unbalanced, so omit the rectangular header fields
    # that would assert a complete Cartesian product and record the true count.
    header.pop("systems")
    header.pop("orderings")
    header.pop("shots")
    header.pop("seeds")
    header["total_cells"] = len(rows)
    path.write_text(
        json.dumps(header) + "\n" + "\n".join(json.dumps(row) for row in rows) + "\n")

    _, groups = summary.summarize([path])
    shot_rows = [row for row in groups if row["group"].startswith("shots=")]
    assert len(shot_rows) == 2
    assert all(row["cells"] == 8 for row in shot_rows)
    assert all(row["median_log_ratio"] == pytest.approx(-0.02) for row in shot_rows)


def test_output_guard_requires_force_and_matching_design(tmp_path):
    path = tmp_path / "existing.jsonl"
    header = {
        "record": "header",
        "systems": ["toy"],
        "orderings": ["physics"],
        "shots": [64],
        "seeds": 2,
        "max_size": 10,
    }
    path.write_text(json.dumps(header) + "\n")
    kwargs = dict(
        systems=["toy"], orderings=["physics"], shots=[64],
        seeds=2, max_size=10)
    with pytest.raises(SystemExit, match="--force"):
        ensemble._validate_output(path, force=False, **kwargs)
    ensemble._validate_output(path, force=True, **kwargs)
    with pytest.raises(SystemExit, match="header differs"):
        ensemble._validate_output(
            path, systems=["different"], orderings=["physics"], shots=[64],
            seeds=2, max_size=10, force=True)


def test_molecular_benchmark_definitions_when_chemistry_extra_is_available(monkeypatch):
    pytest.importorskip("openfermion")
    chemistry = pytest.importorskip("clifford_qc.models.chemistry")
    calls = []

    def fake_model(geometry, **kwargs):
        calls.append((geometry, kwargs))
        return object()

    monkeypatch.setattr(chemistry, "molecule_model", fake_model)
    chemistry.h2o_qsci()
    chemistry.beh2_frozen_core(3.0)

    _, h2o = calls[0]
    assert h2o["occupied_indices"] == [0, 1]
    assert h2o["active_indices"] == [2, 3, 4, 5, 6]
    _, beh2 = calls[1]
    assert beh2["occupied_indices"] == [0]
    assert beh2["active_indices"] == [1, 2, 3, 4, 5, 6]
