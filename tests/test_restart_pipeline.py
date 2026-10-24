"""Declared small molecular compatibility and density-free workflow acceptance."""
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.pipeline import solve_prepared, main
from clifford_qc.prepared import PreparedProblem
from clifford_qc.reference_states import UnsupportedReferenceOperation
from clifford_qc.subspace.adaptive import ACASEConfig
from clifford_qc.subspace.convergence import ConvergenceConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ACASEConfig(max_size=2)


@pytest.fixture(scope="module", params=["h4_sto3g_r0.9.FCIDUMP", "lih_sto3g_r1.5949_cas4e4o.FCIDUMP"])
def molecular(request):
    model = fcidump_model(ROOT / "benchmarks/data" / request.param)
    prepared = PreparedProblem.from_model(model, preparation={"source": "restart acceptance test"})
    legacy, record = solve_prepared(prepared, config=CONFIG, max_candidates=14, reference_mode="legacy_mv")
    return prepared, legacy


@pytest.mark.parametrize("policy,storage", [("retain_all", "object"), ("retain_all", "packed"),
                                           ("stream_recompute", "object"), ("stream_recompute", "packed")])
def test_default_determinant_workflow_preserves_legacy_decisions_and_matrix_bytes(
        molecular, policy, storage, monkeypatch):
    prepared, legacy = molecular
    def forbidden(*args, **kwargs):
        raise AssertionError("density/reference-oracle construction on the determinant workflow")
    import clifford_qc.states as states
    import clifford_qc.dense_reference as dense
    import clifford_qc.backends as backends
    monkeypatch.setattr(ExactMVBackend, "__init__", forbidden)
    monkeypatch.setattr(ExactMVBackend, "state", forbidden)
    monkeypatch.setattr(states, "ket_density", forbidden)
    monkeypatch.setattr(dense, "to_matrix", forbidden)
    monkeypatch.setattr(backends, "SectorStatevectorBackend", forbidden)
    actual, record = solve_prepared(prepared, config=CONFIG, max_candidates=14,
                                     policy=policy, storage=storage, frontier_pairs=1)
    assert actual.labels == legacy.labels
    assert actual.stopped_reason == legacy.stopped_reason
    assert np.asarray(actual.energy_history).tobytes() == np.asarray(legacy.energy_history).tobytes()
    assert [r.selected_label for r in actual.records] == [r.selected_label for r in legacy.records]
    assert [(r.predicted_lowering, r.orthogonal_fraction, r.rejected_conditioning)
            for r in actual.records] == [(r.predicted_lowering, r.orthogonal_fraction, r.rejected_conditioning)
                                         for r in legacy.records]
    for new, old in zip(actual.bank.matrices(actual.indices), legacy.bank.matrices(legacy.indices)):
        assert new.tobytes() == old.tobytes()
    assert record["schema"] == "clifford_qc.prepared_run.v2"
    assert record["reference"]["mode"] == "determinant"
    assert record["reference"]["arithmetic"] == "legacy_trace"
    costs = record["resources"]["reference_evaluation"]
    assert costs["storage"]["density_coefficients"] == 0
    assert costs["storage"]["occupation_payload_bytes"] == prepared.model().n
    assert costs["calls"] > 0 and costs["terms_scanned"] > 0
    assert record["resources"]["hamiltonian_identity"]
    assert not record["stopping_uses_oracle"]


def test_analytic_workflow_is_explicit_and_numerically_compatible(molecular):
    prepared, legacy = molecular
    actual, record = solve_prepared(prepared, config=CONFIG, max_candidates=14,
                                     reference_arithmetic="analytic")
    assert actual.labels == legacy.labels
    assert actual.stopped_reason == legacy.stopped_reason
    np.testing.assert_allclose(actual.energy_history, legacy.energy_history, rtol=1e-12, atol=1e-12)
    for new, old in zip(actual.bank.matrices(actual.indices), legacy.bank.matrices(legacy.indices)):
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)
    assert record["reference"]["arithmetic"] == "analytic"


@pytest.mark.parametrize("options", [
    {"reference_mode": "unknown"}, {"reference_arithmetic": "unknown"},
    {"reference_mode": "legacy_mv", "reference_arithmetic": "analytic"},
])
def test_invalid_reference_choices_fail_before_model_preparation(options):
    with pytest.raises(ValueError): solve_prepared(None, **options)


@pytest.mark.parametrize("config", [
    ACASEConfig(convergence=ConvergenceConfig(1e-6)),
    ACASEConfig(leakage_tol=1e-9, leakage_mode="reference", sector_target=(4, 0.0)),
])
def test_unmigrated_reference_operations_fail_before_work(config):
    with pytest.raises(UnsupportedReferenceOperation, match="legacy_mv"):
        solve_prepared(None, config=config)


def test_cli_records_explicit_reference_modes(molecular, tmp_path):
    import json
    prepared, _ = molecular
    path = tmp_path / "prepared.json"
    prepared.save(path)
    for mode in ("determinant", "legacy_mv"):
        output = tmp_path / f"{mode}.json"
        assert main(["solve", str(path), "--output", str(output), "--max-additions", "1",
                     "--max-candidates", "6", "--reference-mode", mode]) is None
        record = json.loads(output.read_text())
        assert record["reference"]["mode"] == mode
        assert record["reference"]["identity"]


@pytest.mark.parametrize("metadata", [{"spin_convention": "blocked"}, {"fermion_encoding": "bk"}])
@pytest.mark.parametrize("mode", ["determinant", "legacy_mv"])
def test_prepared_excitation_conventions_are_checked_before_bank_construction(metadata, mode, monkeypatch):
    from types import SimpleNamespace
    import clifford_qc.pipeline as pipeline
    class Prepared:
        def model(self):
            return SimpleNamespace(name="unsupported convention", metadata={"kind": "molecular", **metadata,
                                  **({"spin_convention": "interleaved"} if "spin_convention" not in metadata else {})})
    def forbidden(*args, **kwargs):
        raise AssertionError("bank/backend work before convention validation")
    monkeypatch.setattr(ExactMVBackend, "__init__", forbidden)
    with pytest.raises(UnsupportedReferenceOperation):
        pipeline.solve_prepared(Prepared(), reference_mode=mode)
