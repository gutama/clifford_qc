import pytest

from benchmarks import run_phase12_paper_b as phase12


def test_pareto_frontiers_never_mix_evidence_categories():
    rows = [
        {
            "method": "exact_small",
            "evidence_category": "exact_simulation",
            "absolute_error": 0.2,
            "M": 2,
        },
        {
            "method": "exact_accurate",
            "evidence_category": "exact_simulation",
            "absolute_error": 0.1,
            "M": 4,
        },
        {
            "method": "oracle_tiny",
            "evidence_category": "oracle_sampled",
            "absolute_error": 0.0,
            "M": 1,
        },
    ]

    fronts = phase12.pareto_frontiers(rows)

    assert fronts["exact_simulation"]["M"] == [
        "exact_small", "exact_accurate"]
    assert fronts["oracle_sampled"]["M"] == ["oracle_tiny"]
    assert "oracle_tiny" not in fronts["exact_simulation"]["M"]


def test_resolve_source_sha_rejects_an_unresolved_override(monkeypatch):
    monkeypatch.setenv("PHASE12_BASE_SHA", "definitely-not-a-revision")
    with pytest.raises(SystemExit, match="does not resolve to a commit"):
        phase12._resolve_source_sha({"git_sha": "unknown"})


def test_phase12_hubbard_smoke_has_complete_honest_ladder():
    record = phase12.run_system(
        "hubbard_2x2",
        shots=16,
        seed=3,
        max_size=2,
        max_generators=12,
        max_support=16,
        max_packet_support=8,
    )
    rows = {row["method"]: row for row in record["arms"]}

    assert set(phase12.REQUIRED_ARMS) <= set(rows)
    for row in rows.values():
        assert set(phase12.REQUIRED_FIELDS) <= set(row)
        assert row["energy"] >= record["exact_energy"] - 1e-9

    assert rows["reference_state"]["evidence_category"] == "exact_simulation"
    assert rows["exact_sector"]["evidence_category"] == "reference"
    sampled = (
        "qsci", "excitation_closure", "selected_ci", "budget_selected_ci",
        "qsci_dressed_acase", "qsci_haar_dressed_acase")
    for method in sampled:
        assert rows[method]["evidence_category"] == "oracle_sampled"
        assert rows[method]["state_preparation_category"] == "oracle"
        assert rows[method]["sampling_shots"] == 16

    for method in ("qsci", "excitation_closure", "selected_ci",
                   "budget_selected_ci"):
        assert rows[method]["W"] == 0

    haar = rows["qsci_haar_dressed_acase"]
    assert haar["metadata"]["configuration_order"] == "physics"
    assert haar["metadata"]["haar_stage"] is True
    assert haar["metadata"]["haar_object"] == (
        "support-pruned orthonormal Haar packet hierarchy")
    assert haar["metadata"]["packet_directions"] > 0
    assert haar["metadata"]["frontiers_scored"] >= 2

    assert set(record["pareto_frontiers"]) == {
        "exact_simulation", "oracle_sampled", "reference"}
