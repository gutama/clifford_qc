"""Regression checks for the numerical claims made by the Phase 16B pilot."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import probe_realtime_krylov as probe


def test_toeplitz_matches_independent_complex_statevector_pencil():
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (matrix + matrix.conj().T) / 2
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    c, d = probe.toeplitz_scalars(H, psi, 4, 0.3)
    actual = probe.assemble(c, d, 4)
    expected = probe.direct_pencil(H, psi, 4, 0.3)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=0)
    # Reversing lag signs is Hermitian and Toeplitz but is the wrong pencil.
    reversed_c = {k: value.conjugate() for k, value in c.items()}
    assert np.max(np.abs(probe.assemble(reversed_c, d, 4)[0] - expected[0])) > 0.1


def test_noise_preserves_component_variance_and_known_normalization():
    rng = np.random.default_rng(21)
    base = {0: 1.0 + 0j, 1: 0.2 + 0.3j, -1: 0.2 - 0.3j}
    draws = [probe.sample_scalars(base, 1, 0.1, rng, exact_zero=True)
             for _ in range(12000)]
    assert all(row[0] == 1 and row[-1] == row[1].conjugate() for row in draws)
    noise = np.array([row[1] - base[1] for row in draws])
    assert np.var(noise.real) == pytest.approx(0.01, rel=0.04)
    assert np.var(noise.imag) == pytest.approx(0.01, rel=0.04)
    # Hermitian assembly must not halve variance by averaging independent +/- lags.
    assembled = np.array([probe.assemble(row, row, 2)[0][0, 1] for row in draws])
    np.testing.assert_array_equal(assembled, noise + base[1])


@pytest.mark.parametrize("shift", [0.0, -40.0, 40.0])
def test_unitary_energy_branch_restores_large_identity_shifts(shift):
    H = np.diag([-1.0, 0.5, 2.0]) + shift * np.eye(3)
    psi = np.sqrt([0.2, 0.3, 0.5]).astype(complex)
    dt = np.pi / 3
    c, _ = probe.toeplitz_scalars(H, psi, 4, dt)
    energy, rank = probe.solve_unitary(c, 3, dt, 1e-13, energy_shift=shift + 0.5)
    assert rank == 3
    assert energy == pytest.approx(shift - 1, abs=2e-12)
    if shift:
        aliased, _ = probe.solve_unitary(c, 3, dt, 1e-13)
        assert abs(aliased - energy) > 10


def test_degenerate_spectral_support_is_basis_invariant():
    values = np.array([0.0, 0.0, 2.0])
    psi = np.sqrt([0.5, 0.5, 0.0])
    vectors = np.eye(3)
    rotated = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    assert probe.spectral_support(values, vectors, psi) == 1
    assert probe.spectral_support(values, rotated, psi) == 1


def test_failures_remain_in_median_and_tail():
    summary = probe.error_summary([0.0] * 8 + [np.nan, np.inf], 0.0)
    assert summary == {"median": 0.0, "p90": np.inf, "failures": 2}
    summary = probe.error_summary([0.0, np.nan, np.nan], 0.0)
    assert np.isinf(summary["median"])
    assert np.isinf(probe.error_summary([np.nan] * 3, 0.0)["p90"])
    with pytest.raises(ValueError, match="replica"):
        probe.error_summary([], 0.0)


def test_rank_zero_and_vanished_unitary_root_fail_explicitly():
    energy, rank = probe.solve(-np.eye(2), np.eye(2), 0.01)
    assert np.isnan(energy) and rank == 0
    c = {0: 1 + 0j, 1: 0j, -1: 0j}
    energy, rank = probe.solve_unitary(c, 1, 1.0, 0)
    assert np.isnan(energy) and rank == 1


@pytest.mark.parametrize("args", [["--replicas", "0"], ["--seed", "-1"],
                                 ["--qubits", "0"]])
def test_invalid_cli_parameters_are_rejected(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["probe_realtime_krylov.py", *args])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2


def test_cli_smoke_has_independent_residuals_and_both_arms(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["probe_realtime_krylov.py", "--replicas", "2",
                                     "--qubits", "2", "--seed", "1"])
    monkeypatch.setattr(probe, "BASIS_SIZES", (2, 4))
    monkeypatch.setattr(probe, "NOISE_LEVELS", (0.0, 1e-3))
    assert probe.main() == 0
    output = capsys.readouterr().out
    assert "independent statevector construction" in output
    assert "real components H/U=13/8" in output
    assert "hermitian" in output and "unitary" in output
    assert "failed" in output and "TFIM J=1 energy units" in output
