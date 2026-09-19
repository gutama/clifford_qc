"""Regression checks for molecular physics, caching and recoverable records."""
import json
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.pipeline import solve_prepared
from clifford_qc.prepared import _atomic_json, prepare_fcidump
from clifford_qc.sparse import to_sparse
from clifford_qc.subspace import occupied_spin_orbitals
from clifford_qc.subspace.adaptive import ACASEConfig
from molecular.chemistry import prepare_chemistry
from molecular.run import can_resume, load_records, main, output_lock, seal_record
from molecular.simulation import determinant_ci_energy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"


def test_determinant_restriction_matches_full_register_reference():
    model = fcidump_model(SOURCE)
    model.metadata.update(kind="fermionic_lattice", sites=model.n // 2, n_orbitals=1)
    occupied = occupied_spin_orbitals(model)
    backend = SectorStatevectorBackend(model.n, 4, 0)
    reference = sum(1 << (model.n - 1 - q) for q in occupied)
    kept = [int(word) for word in backend.basis
            if (int(word) ^ reference).bit_count() // 2 <= 2]
    expected = np.linalg.eigvalsh(to_sparse(model.hamiltonian).tocsr()[np.ix_(kept, kept)].toarray())[0]
    with patch("clifford_qc.sparse.to_sparse", side_effect=AssertionError("full-register allocation")):
        energy, count, spin = determinant_ci_energy(model.hamiltonian, model, model.n, occupied, 4, 0)
    assert energy == pytest.approx(expected, abs=1e-10)
    assert count == len(kept)
    assert spin == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("target_s2", [0.0, 2.0])
def test_determinant_ci_resolves_spin_degenerate_ground_space(target_s2):
    # At zero hopping the singly occupied determinants have energy zero.
    # Each has <S^2>=1, although their singlet/triplet combinations have
    # S^2=0/2. Filtering eigenvectors of H by spin expectation misses both.
    model = hubbard(2, t=0, U=1, mu=0)
    energy, count, spin = determinant_ci_energy(
        model.hamiltonian, model, model.n, occupied_spin_orbitals(model),
        2, 0, target_s2=target_s2)
    assert energy == pytest.approx(0, abs=1e-12)
    assert count == 4
    assert spin == pytest.approx(target_s2, abs=1e-12)


def test_determinant_ci_rejects_missing_spin_and_spin_mixing():
    from clifford_qc.pauli import Z
    model = hubbard(2, t=0, U=1, mu=0)
    args = (model, model.n, occupied_spin_orbitals(model), 2, 0)
    with pytest.raises(RuntimeError, match="no root"):
        determinant_ci_energy(model.hamiltonian, *args, target_s2=6)
    with pytest.raises(ValueError, match="not invariant"):
        determinant_ci_energy(Z(model.n, 0), *args)


def test_candidate_budget_and_storage_preserve_scientific_result(tmp_path):
    prepared, _, _ = prepare_fcidump(SOURCE, tmp_path)
    options = dict(config=ACASEConfig(max_size=1), max_candidates=14)
    reference, record = solve_prepared(prepared, **options)
    actual, streamed = solve_prepared(prepared, **options, storage="packed",
                                     policy="stream_recompute", frontier_pairs=1)
    assert record["candidate_count"] == 14 < record["available_candidates"]
    assert actual.labels == reference.labels
    assert len(actual.labels) == 2
    np.testing.assert_allclose(actual.energy_history, reference.energy_history, atol=1e-12)
    assert streamed["resources"]["storage_policy"] == "stream_recompute"


@pytest.mark.parametrize("options", [{"max_candidates": -1}, {"max_candidates": True},
    {"storage": "invalid"}, {"policy": "invalid"}, {"frontier_pairs": 0}, {"max_rank": 0}])
def test_invalid_solver_options_fail_before_model_preparation(options):
    with pytest.raises(ValueError):
        solve_prepared(None, **options)


def test_resume_checks_configuration_integrals_and_record_contents(tmp_path):
    source = tmp_path / "lih.fcidump"
    source.write_text("integral snapshot")
    request = {"budget": 1}
    record = seal_record({"key": "lih", "energy": -1}, request, source)
    assert can_resume(record, request, source)
    assert not can_resume(record, {"budget": 2}, source)
    assert not can_resume({"key": "lih"}, request, source)
    assert not can_resume({**record, "energy": -2}, request, source)
    source.write_text("changed snapshot")
    assert not can_resume(record, request, source)


def test_recover_summary_after_interruption_and_reject_corrupt_records(tmp_path):
    _atomic_json(tmp_path / "lih_results.json", {"key": "lih", "energy": -1})
    # A previous process can die after committing the molecule but before the
    # summary. The next driver recovers from the authoritative per-molecule file.
    _atomic_json(tmp_path / "results_summary.json", {})
    assert load_records(tmp_path)["lih"]["energy"] == -1
    (tmp_path / "hf_results.json").write_text('{"key":')
    with pytest.raises(json.JSONDecodeError):
        load_records(tmp_path)


def test_atomic_json_preserves_old_record_on_serialization_failure(tmp_path):
    path = tmp_path / "record.json"
    _atomic_json(path, {"energy": -1})
    with pytest.raises(ValueError):
        _atomic_json(path, {"energy": float("nan")})
    assert json.loads(path.read_text()) == {"energy": -1}
    assert list(tmp_path.iterdir()) == [path]


def test_output_lock_rejects_concurrent_writer_and_releases(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another molecular run"):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


def test_driver_isolates_workers_and_recovers_each_completed_record(tmp_path):
    seen = []
    def worker(command, **kwargs):
        key = command[command.index("--worker") + 1]
        seen.append(key)
        assert kwargs["pass_fds"] and kwargs["cwd"] == ROOT
        _atomic_json(tmp_path / f"{key}_results.json", {"key": key})
    with patch("molecular.run.subprocess.run", side_effect=worker):
        main(["--molecules", "lih", "hf", "--output-directory", str(tmp_path), "--max-additions", "0"])
    assert seen == ["lih", "hf"]
    assert set(json.loads((tmp_path / "results_summary.json").read_text())) == set(seen)


def test_chemistry_cache_reuses_integrals_and_detects_corruption(tmp_path):
    pytest.importorskip("pyscf")
    spec = {"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", "charge": 0, "spin": 0}
    expected, exported, hit = prepare_chemistry("h2", spec, tmp_path)
    assert not hit and expected["rhf_converged"]
    integral = exported.read_bytes()
    exported.unlink()
    with patch("pyscf.scf.RHF", side_effect=AssertionError("unnecessary chemistry recomputation")):
        actual, exported, hit = prepare_chemistry("h2", spec, tmp_path)
    assert hit and actual == expected and exported.read_bytes() == integral
    # Exercise both numerical arms and verify their large banks cannot overlap.
    from molecular import simulation
    original_solve = simulation.solve_prepared
    original_sd = simulation.solve_subspace
    bank_refs = []
    def adaptive(*args, **kwargs):
        result, record = original_solve(*args, **kwargs)
        bank_refs.append(weakref.ref(result.bank))
        return result, record
    def complete_sd(*args, **kwargs):
        assert bank_refs[0]() is None
        return original_sd(*args, **kwargs)
    with patch.object(simulation, "solve_prepared", side_effect=adaptive), patch.object(
            simulation, "solve_subspace", side_effect=complete_sd):
        result = simulation.simulate("h2", {**spec, "name": "H2"}, tmp_path,
                                     max_additions=0, complete_sd="on")
    assert result["e_sd_complete"] == pytest.approx(expected["e_cisd"], abs=1e-10)
    assert result["chemistry_cache_hit"]
    cache = next((tmp_path / "cache/chemistry").glob("*.fcidump"))
    cache.write_text("corrupted")
    with pytest.raises(ValueError, match="--refresh-chemistry"):
        prepare_chemistry("h2", spec, tmp_path)


def test_unconverged_rhf_never_publishes_chemistry(tmp_path):
    pytest.importorskip("pyscf")
    spec = {"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", "charge": 0, "spin": 0}
    with patch("pyscf.scf.RHF") as rhf:
        rhf.return_value.converged = False
        with pytest.raises(RuntimeError, match="RHF did not converge"):
            prepare_chemistry("h2", spec, tmp_path)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.fcidump"))


def test_report_counts_accuracy_from_records_and_preserves_spin_caveat():
    from benchmarks.summarize_molecular import render_tex, render_markdown
    summary = json.loads((ROOT / "molecular/results/results_summary.json").read_text())
    accurate = sum(abs(record["error_adaptive_mha"]) <= 1.5936 for record in summary.values())
    tex = render_tex(summary)
    assert f"on {accurate} of {len(summary)} geometries" in tex
    assert "Conserving $S_z$ does not enforce total spin" in tex
    assert "does not enforce a singlet" in render_markdown(summary)
