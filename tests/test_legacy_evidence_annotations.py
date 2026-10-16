"""Legacy annotations disclose evidence scope without rewriting execution records."""

import pytest

from benchmarks import evidence_metadata as metadata
from clifford_qc.evidence import SCHEMA, validate_evidence


QUANTITY_TIERS = {
    "clifford_hierarchy_beh2_v2.json": {
        "bank_energy_error_and_conditioning": "exact_small_sector_simulation",
        "grouping_and_logical_clifford_counts": "exact_algebra",
        "uniform_shot_totals_and_break_even_ratios": "illustrative_model",
    },
    "clifford_hierarchy_h4_v2.json": {
        "bank_energy_error_and_conditioning": "exact_small_sector_simulation",
        "grouping_and_logical_clifford_counts": "exact_algebra",
        "uniform_shot_totals_and_break_even_ratios": "illustrative_model",
    },
    "h4_exact_trajectory.json": {
        "classical_execution_counts_and_timings": "classical_process_measurement",
        "energies_gradients_trajectory_and_sector_diagnostics": "exact_small_sector_simulation",
        "exact_arm_shot_and_circuit_accounting": "exact_algebra",
    },
    "krylov_width.json": {
        "krylov_power_union_identity": "exact_algebra",
        "word_counts_and_pencil_numerical_certificates": "exact_small_sector_simulation",
    },
    "orbital_basis.json": {
        "conditional_sector_solutions_and_determinant_truncations": "exact_small_sector_simulation",
        "disorder_sweep_basis_comparisons": "finite_sample_empirical",
        "pauli_word_and_logical_rotation_counts": "exact_algebra",
    },
}


def _record(name):
    return metadata.evidence_view(metadata.ROOT / "benchmarks/reference_results" / name)


@pytest.mark.parametrize("name", QUANTITY_TIERS)
def test_five_annotations_bind_frozen_bytes_and_disclose_quantity_tiers(name):
    path = metadata.ROOT / "benchmarks/reference_results" / name
    manifest = metadata.ROOT / metadata.MANIFEST
    before = path.read_bytes()
    assert metadata.sha256(manifest) == metadata.LEGACY_MANIFEST_SHA256
    frozen = metadata.read_json(manifest)["records"][name]
    assert metadata.sha256(path) == frozen
    annotation = metadata.read_json(
        metadata.ROOT / metadata.ANNOTATIONS / f"{name}.evidence.json")
    assert annotation["record_sha256"] == frozen
    raw = metadata.read_json(path)
    assert "evidence_schema" not in raw
    assert "evidence" not in raw
    labelled = _record(name)
    validate_evidence(labelled)
    assert labelled["evidence_schema"] == SCHEMA
    assert {key: value["tier"] for key, value in labelled["evidence"].items()} == QUANTITY_TIERS[name]
    assert {key: value for key, value in labelled.items()
            if key not in {"evidence_schema", "evidence"}} == raw
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", QUANTITY_TIERS)
def test_each_new_annotation_rejects_record_drift(tmp_path, name):
    original = metadata.ROOT / "benchmarks/reference_results" / name
    copied = tmp_path / "benchmarks/reference_results" / name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())
    sidecar = tmp_path / metadata.ANNOTATIONS / f"{name}.evidence.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(
        (metadata.ROOT / metadata.ANNOTATIONS / sidecar.name).read_bytes())
    assert metadata.evidence_view(copied, root=tmp_path)["evidence_schema"] == SCHEMA
    # Even a whitespace-only edit breaks the binding to execution bytes.
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity/digest differs"):
        metadata.evidence_view(copied, root=tmp_path)


@pytest.mark.parametrize("name", ["clifford_hierarchy_beh2_v2.json", "clifford_hierarchy_h4_v2.json"])
def test_hierarchy_uniform_shot_columns_are_model_projections(name):
    record = _record(name)
    shots = record["shots_per_setting"]
    assert shots == 8000
    for row in record["rows"]:
        assert row["state_preparations_at_uniform_shots"] == shots * row["settings"]
        assert row["logical_cx_applications_at_uniform_shots"] == shots * row["logical_cx_per_sweep"]
    assert record["qwc_to_full"]["settings_saved"] == (
        record["rows"][0]["settings"] - record["rows"][-1]["settings"])
    scope = record["evidence"]["uniform_shot_totals_and_break_even_ratios"]
    assert scope["tier"] == "illustrative_model"
    assert "no shot outcomes" in scope["scope"]
    assert "finite_sample_empirical" not in {value["tier"] for value in record["evidence"].values()}


def test_exact_h4_simulation_and_local_classical_measurements_are_separate():
    record = _record("h4_exact_trajectory.json")
    assert record["arm"] == "exact"
    assert record["selection_metadata"]["noisy_selection"] is False
    assert record["total_shots"] == record["total_circuits"] == 0
    assert all(row["shots_added"] == row["circuits_executed"] == 0 for row in record["trajectory"])
    assert record["optimizer_evaluations"] > 0
    assert record["wall_seconds"] > 0
    assert record["evidence"]["classical_execution_counts_and_timings"]["tier"] == "classical_process_measurement"
    assert record["evidence"]["energies_gradients_trajectory_and_sector_diagnostics"]["tier"] == "exact_small_sector_simulation"


def test_krylov_numerical_support_is_scoped_to_its_stored_cutoff_certificate():
    record = _record("krylov_width.json")
    assert record["certificate"]["cuts_are_nonrecursive"] is True
    reported = repr(record["reported_threshold"])
    for row in record["rows"]:
        diagnostic = row["pencil_diagnostics_by_threshold"][reported]
        assert row["word_universe"] == diagnostic["word_universe"]
        assert diagnostic["certificate_passed"] is True
        assert diagnostic["ground_energy_absolute_error"] <= record["certificate"]["max_energy_error"]
    evidence = record["evidence"]["word_counts_and_pencil_numerical_certificates"]
    assert evidence["tier"] == "exact_small_sector_simulation"
    assert "baseline coefficient pruning" in evidence["scope"]
    assert "not exact symbolic support" in evidence["scope"]


def test_orbital_disorder_comparisons_remain_finite_seed_evidence():
    record = _record("orbital_basis.json")
    parameters = record["parameters"]
    assert parameters["disorder_seeds"] == 3
    assert len(parameters["disorder_strengths"]) == 4
    for strength in parameters["disorder_strengths"]:
        for seed in range(parameters["disorder_seeds"]):
            rows = [row for row in record["disorder"]
                    if row["disorder"] == strength and row["seed"] == seed]
            assert {row["basis"] for row in rows} == {"site", "momentum", "db2", "natural"}
    assert record["evidence"]["disorder_sweep_basis_comparisons"]["tier"] == "finite_sample_empirical"
    assert record["evidence"]["conditional_sector_solutions_and_determinant_truncations"]["tier"] == "exact_small_sector_simulation"
    assert "asymptotic_bound" not in {value["tier"] for value in record["evidence"].values()}


def test_all_records_have_evidence_and_historical_bindings_still_pass():
    from paper_architecture.make_tables import record_census

    assert metadata.check_records() == []
    assert record_census()["forms"]["none"] == 0
