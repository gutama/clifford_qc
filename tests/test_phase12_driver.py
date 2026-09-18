import pytest
from types import SimpleNamespace

from benchmarks import run_phase12_paper_b as phase12


def test_phase12_does_not_promote_unvalidated_molecular_ladders():
    assert "h2o_qsci" not in phase12.PRIMARY_SYSTEMS
    assert "beh2_stretched" not in phase12.PRIMARY_SYSTEMS
    with pytest.raises(ValueError, match="unknown Phase 12 primary system"):
        phase12.run_system("beh2_stretched")


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


def test_observed_and_qsci_rows_use_incremental_peak_memory(monkeypatch):
    class FakeRSS:
        def __enter__(self):
            self.baseline = 100
            self.peak = 160
            return self

        def __exit__(self, *exc):
            pass

        @property
        def delta(self):
            return self.peak - self.baseline

    monkeypatch.setattr(phase12, "_PeakRSS", FakeRSS)
    value, _, peak = phase12._observed(lambda: "done")
    assert value == "done"
    assert peak == 60

    sampling = SimpleNamespace(
        accepted_shots=1, unique_configurations=1, raw_shots=1,
        duplicate_fraction=0.0, discarded_fraction=0.0,
        repaired_fraction=0.0, state_preparations=None,
        state_preparation_executions=None, retained_probability=1.0)
    state = SimpleNamespace(category="oracle", label="oracle", metadata={})
    result = SimpleNamespace(
        sampling=sampling, subspace_dimension=1, energy=0.0,
        matrix_nonzeros=1, matrix_bytes=16, build_seconds=0.1,
        solve_seconds=0.2, peak_rss_bytes=999, peak_rss_delta_bytes=7,
        hermiticity_residual=0.0)
    row = phase12._qsci_row(result, state, exact_energy=0.0, seed=0)
    assert row["peak_memory_bytes"] == 7


def test_finite_adapt_reports_size_and_real_execution_accounting(monkeypatch):
    def fake_run_method(*args, **kwargs):
        return {
            "energy": -1.0,
            "evidence": "finite_sample",
            "operators": 2,
            "total_shots": 120,
            "total_circuits": 6,
            "support_peak": 4,
        }

    monkeypatch.setattr(phase12.ladder, "run_method", fake_run_method)
    row = phase12._ladder_row(
        "adapt_vqe_finite", {"kind": "adapt_shot"}, None, {},
        exact_energy=-2.0, seed=4)

    assert row["M"] == 2
    assert row["retained_rank"] is None
    assert row["kappa_S"] is None
    assert "not defined" in row["metadata"]["kappa_S_semantics"]
    assert row["sampling_shots"] == 120
    assert row["state_preparations"] == 3
    assert row["state_preparation_executions"] == 120
    assert row["grouping_contexts"] == 6
    assert row["certified_shot_cost"] == 120


def test_finish_record_enforces_populated_core_resources():
    row = phase12._blank_record("broken", "exact_simulation", seed=0)
    row.update(phase12._no_sampling_fields())
    row.update({"energy": 0.0, "wall_seconds": 0.0})
    with pytest.raises(AssertionError, match="required resource fields unpopulated"):
        phase12._finish_record(row, exact_energy=0.0)


def test_main_resolves_revision_before_running_systems(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        phase12, "execution_provenance", lambda: {"git_sha": "fake"})

    def resolve(_):
        events.append("resolve")
        return "resolved-sha"

    def run_system(name, **kwargs):
        events.append("run")
        return {
            "system_key": name,
            "exact_energy": 0.0,
            "sampling": {"unique_configurations": 1},
            "arms": [],
        }

    monkeypatch.setattr(phase12, "_resolve_source_sha", resolve)
    monkeypatch.setattr(phase12, "run_system", run_system)
    phase12.main([
        "--systems", "hubbard_2x2", "--out", str(tmp_path / "phase12.json")])
    assert events == ["resolve", "run"]


def test_one_configuration_takes_costed_nonpareto_haar_fallback():
    record = phase12.run_system(
        "hubbard_2x2", shots=1, seed=0, max_size=2, max_generators=8,
        max_support=16, max_packet_support=8)
    rows = {row["method"]: row for row in record["arms"]}
    dressed = rows["qsci_dressed_acase"]
    haar = rows["qsci_haar_dressed_acase"]

    assert haar["metadata"]["haar_stage"] is False
    assert haar["metadata"]["haar_unavailable_reason"] == (
        "fewer than two non-reference sampled configurations")
    assert haar["metadata"]["pareto_eligible"] is False
    assert haar["energy"] == pytest.approx(dressed["energy"], abs=1e-12)
    assert haar["wall_seconds"] == dressed["wall_seconds"]
    assert haar["peak_memory_bytes"] == dressed["peak_memory_bytes"]
    assert all(
        "qsci_haar_dressed_acase" not in methods
        for methods in record["pareto_frontiers"]["oracle_sampled"].values())


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
    assert rows["reference_state"]["wall_seconds"] > 0.0
    assert rows["exact_sector"]["wall_seconds"] > 0.0
    assert rows["exact_sector"]["solve_seconds"] == rows["exact_sector"]["wall_seconds"]
    assert rows["adapt_vqe"]["M"] == rows["adapt_vqe"]["metadata"]["operators"]
    assert rows["adapt_vqe"]["retained_rank"] is None
    assert "not defined" in rows["adapt_vqe"]["metadata"]["kappa_S_semantics"]
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
    assert haar["metadata"]["selection_work"] >= haar["metadata"]["frontiers_scored"]

    assert set(record["pareto_frontiers"]) == {
        "exact_simulation", "oracle_sampled", "reference"}


def test_krylov_depth_cap_is_declared_and_bounded():
    """A narrowed Krylov arm must be visibly narrowed, not silently smaller."""
    import pytest

    from benchmarks.run_phase12_paper_b import _krylov_depth_metadata, run_system

    uncapped = _krylov_depth_metadata(6, 6)
    assert uncapped == {"krylov_depth": 6, "krylov_depth_capped": False}

    capped = _krylov_depth_metadata(6, 3)
    assert capped["krylov_depth"] == 3
    assert capped["krylov_depth_capped"] is True
    assert capped["krylov_depth_budget"] == 6
    assert "H^3" in capped["krylov_depth_semantics"]

    # A cap above the ladder budget would let fixed_krylov span more than every
    # other arm, which is the one direction that breaks the comparison.
    for bad in (0, 7):
        with pytest.raises(ValueError, match="krylov_size must lie in"):
            run_system("hubbard_2x2", max_size=6, krylov_size=bad)


def test_krylov_cap_narrows_the_recorded_arm():
    from benchmarks.run_phase12_paper_b import run_system

    capped = run_system("hubbard_2x2", shots=16, seed=0, max_size=4,
                        krylov_size=2)
    row = {arm["method"]: arm for arm in capped["arms"]}["fixed_krylov"]
    assert row["metadata"]["krylov_depth"] == 2
    assert row["metadata"]["krylov_depth_capped"] is True
    assert row["metadata"]["krylov_depth_budget"] == 4
    # The cap must bind the subspace, not merely be annotated onto it.
    assert row["M"] <= 3
    # Capped or not, the arm is still a variational Rayleigh-Ritz subspace.
    assert row["energy"] >= capped["exact_energy"] - 1e-9
