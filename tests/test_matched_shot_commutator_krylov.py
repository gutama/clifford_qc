"""The matched-shot commutator/Krylov study: chemistry layer, shot model, selection.

The study works in exact sector vectors built with PySCF, so these tests check
that layer against the package's own operator path where both apply, and
check the declared shot model's statistics directly.
"""

import math

import numpy as np
import pytest

pytest.importorskip("pyscf")

from benchmarks import run_matched_shot_commutator_krylov as study  # noqa: E402

SMALL = next(s for s in study.SYSTEMS if s["key"] == "h2o_2x_cas4e3o")
CHAIN = next(s for s in study.SYSTEMS if s["key"] == "h4_r2.0")


@pytest.fixture(scope="module")
def chain():
    return study.build_sector(CHAIN)


@pytest.fixture(scope="module")
def water():
    return study.build_sector(SMALL)


def test_the_sector_reproduces_pyscf_casci_and_rhf(chain):
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom=CHAIN["atom"], basis=CHAIN["basis"], verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    casci = mcscf.CASCI(mf, mol.nao, mol.nelectron)
    casci.verbose = 0
    assert chain.eigenvalues[0] == pytest.approx(casci.kernel()[0], abs=1e-9)
    reference = chain.reference
    assert reference @ chain.hamiltonian @ reference == pytest.approx(mf.e_tot, abs=1e-9)
    assert np.allclose(chain.hamiltonian, chain.hamiltonian.T)
    target, index = study.target_state(chain)
    assert index == 0 and abs(chain.spins[0]) < 1e-8 and target == chain.eigenvalues[0]


def test_the_qubit_hamiltonian_is_the_importer_s(chain, tmp_path):
    from pyscf.tools import fcidump

    from clifford_qc.models.fcidump import fcidump_model

    path = tmp_path / "chain.FCIDUMP"
    fcidump.from_integrals(str(path), chain.h1, chain.eri, chain.norb, sum(chain.nelec),
                           chain.ecore, ms=0, tol=1e-15)
    imported = fcidump_model(path).hamiltonian.to_mv()
    assembled = study.qubit_hamiltonian(chain)
    assert set(imported.terms) == set(assembled.terms)
    assert max(abs(imported.terms[c] - assembled.terms[c]) for c in imported.terms) < 1e-10
    # The qubit and sector Hamiltonians share their spectrum's bottom.
    from clifford_qc.dense_reference import exact_ground

    assert exact_ground(assembled)[0] == pytest.approx(chain.eigenvalues[0], abs=1e-8)


def test_excitation_generators_are_anti_hermitian_and_conserve_the_sector(chain):
    rng = np.random.default_rng(0)
    u = rng.standard_normal(chain.dimension)
    v = rng.standard_normal(chain.dimension)
    for excitation in study.excitations(chain)[::5]:
        tu = study.apply_excitation(chain, excitation, u)
        tv = study.apply_excitation(chain, excitation, v)
        assert tu.shape == u.shape
        assert np.vdot(u, tv) == pytest.approx(-np.vdot(tu, v), abs=1e-12)


def test_commutator_vectors_match_the_operator_identity(water):
    H = water.hamiltonian
    psi = water.reference
    pool = study.candidate_pool(water, depth=2)
    by_label = {label: (vector, h) for label, vector, h in pool}
    for excitation in study.excitations(water):
        label = excitation[0]
        T = np.column_stack([study.apply_excitation(water, excitation, e)
                             for e in np.eye(water.dimension)])
        for name, expected in ((f"[H,{label}]", (H @ T - T @ H) @ psi),
                               (f"[H,[H,{label}]]",
                                (H @ (H @ T - T @ H) - (H @ T - T @ H) @ H) @ psi)):
            if name not in by_label:
                assert np.linalg.norm(expected) < 1e-10
                continue
            vector, h = by_label[name]
            unit = expected / np.linalg.norm(expected)
            assert abs(abs(np.vdot(unit, vector)) - 1.0) < 1e-10
            assert h == pytest.approx(float(unit @ H @ unit), abs=1e-10)


def test_greedy_selection_matches_run_acase_energies(chain):
    """The sector-vector greedy reproduces the library's A-CASE path."""
    from clifford_qc import ket_density
    from clifford_qc.subspace import (
        Generator, determinant_excitations, identity_generator, run_acase,
    )

    qubit = study.qubit_hamiltonian(chain)
    occupied = [2 * p + s for p in range(chain.nelec[0]) for s in (0, 1)]
    bits = "".join("1" if q in occupied else "0" for q in range(qubit.n))
    pool = study.candidate_pool(chain, depth=0)
    _, basis = study.greedy_acase(chain, pool, 6)
    ours = []
    for size in range(2, basis.shape[1] + 1):
        S, Hm = study.pencil(basis[:, :size], chain.hamiltonian)
        ours.append(float(study.solve_exact(S, Hm).energies[0]))
    found = run_acase(ket_density(qubit.n, bits), qubit,
                      determinant_excitations(qubit.n, occupied),
                      initial=[identity_generator(qubit.n)], max_size=5)
    assert isinstance(found.records[0].selected_label, str)
    library = [record.energy for record in found.records]
    assert np.allclose(ours[:len(library)], library, atol=1e-8)
    assert Generator  # the import documents the generator type compared


def test_real_time_krylov_is_toeplitz_and_power_krylov_spans_the_krylov_space(chain):
    identity = float(study.qubit_hamiltonian(chain).terms[0].real)
    bases = study.krylov_bases(chain, 5, identity)
    real_time, details = bases["rt_krylov"]
    S, Hm = study.pencil(real_time, chain.hamiltonian)
    for d in range(1, 5):
        diagonal = np.diagonal(S, d)
        assert np.allclose(diagonal, diagonal[0], atol=1e-12)
        diagonal = np.diagonal(Hm, d)
        assert np.allclose(diagonal, diagonal[0], atol=1e-10)
    assert details["time_step"] == pytest.approx(math.pi / details["spectral_range"])
    power, _ = bases["power_krylov"]
    H = chain.hamiltonian
    raw = np.column_stack([np.linalg.matrix_power(H, k) @ chain.reference for k in range(5)])
    # Same span: projecting the raw powers onto the normalized basis loses nothing.
    Q, _ = np.linalg.qr(power)
    assert np.linalg.norm(raw - Q @ (Q.conj().T @ raw)) < 1e-8 * np.linalg.norm(raw)


@pytest.mark.parametrize("arm", ["rt_krylov", "commutator_acase"])
def test_the_shot_model_is_unbiased_with_the_declared_variance(arm):
    rng = np.random.default_rng(3)
    size = 3
    vectors = rng.standard_normal((6, size))
    vectors /= np.linalg.norm(vectors, axis=0)
    operator = np.diag(rng.uniform(-1, 1, 6))
    if arm == "rt_krylov":
        # A complex Toeplitz pencil from unitary powers of a diagonal operator.
        w = np.diagonal(operator)
        vectors = np.column_stack([np.exp(-1j * w * k) * vectors[:, 0] for k in range(size)])
    S, Hp = study.pencil(vectors, operator)
    scale, shots, draws = 2.0, 400, 20000
    S_hat, H_hat = study.sample_pencil(arm, S, Hp, shots, scale, rng, draws)
    assert np.allclose(np.diagonal(S_hat, axis1=1, axis2=2), 1.0)
    assert np.allclose(S_hat, np.conj(np.transpose(S_hat, (0, 2, 1))))
    assert np.allclose(H_hat, np.conj(np.transpose(H_hat, (0, 2, 1))))
    assert np.allclose(S_hat.mean(axis=0), S, atol=0.01)
    assert np.allclose(H_hat.mean(axis=0), Hp, atol=0.02)
    observed = H_hat[:, 0, 1].real.var()
    assert observed == pytest.approx((scale ** 2 - Hp[0, 1].real ** 2) / shots, rel=0.05)
    observed = S_hat[:, 0, 1].real.var()
    assert observed == pytest.approx((1.0 - S[0, 1].real ** 2) / shots, rel=0.05)
    if arm == "rt_krylov":
        observed = S_hat[:, 0, 1].imag.var()
        assert observed == pytest.approx((1.0 - S[0, 1].imag ** 2) / shots, rel=0.05)
    else:
        # A real pencil's imaginary parts vanish and are not measured.
        assert not np.iscomplexobj(S_hat) and not np.iscomplexobj(H_hat)


def test_a_real_arm_refuses_a_complex_pencil():
    S = np.array([[1.0, 0.5j], [-0.5j, 1.0]])
    with pytest.raises(ValueError, match="complex"):
        study.sample_pencil("power_krylov", S, S, 10, 1.0, np.random.default_rng(0), 2)


def test_the_measured_quantity_counts_and_floor_follow_the_declaration():
    assert [study.quantities("rt_krylov", m) for m in (4, 8)] == [13, 29]
    # A real symmetric pencil: M(M-1)/2 overlaps and M(M+1)/2 Hamiltonian elements.
    assert [study.quantities("power_krylov", m) for m in (4, 8)] == [16, 64]
    assert study.quantities("commutator_acase", 12) == 144
    assert study.overlap_floor(8, 100) == pytest.approx(2 * math.sqrt(16 / 100))
    assert study.overlap_floor(8, 100, complex_entries=False) == pytest.approx(
        2 * math.sqrt(8 / 100))
    assert study.overlap_floor(8, 10 ** 6) < study.overlap_floor(8, 10 ** 4)
    assert study.overlap_floor(8, 100, 20.0) == pytest.approx(10 * study.overlap_floor(8, 100))


def test_summaries_count_failures_and_separate_shot_noise_from_subspace_error():
    errors = [0.001, -0.002, 0.050, float("nan")]
    found = study.summarize(errors, failures=1, exact_error=0.0)
    assert found["replicas"] == 4 and found["failures"] == 1
    assert found["within_chemical"] == pytest.approx(1 / 4)
    assert found["below_target_beyond_chemical"] == pytest.approx(1 / 4)
    assert found["median_abs_error"] == pytest.approx(0.002)
    shifted = study.summarize([0.10, 0.11, 0.12], failures=0, exact_error=0.11)
    assert shifted["median_abs_shot_deviation"] == pytest.approx(0.01)
    assert study.summarize([float("nan")], failures=1, exact_error=0.0)["median_abs_error"] is None


def test_a_small_run_is_complete_and_reproducible(tmp_path):
    """The script, run as users run it, gives the same record twice.

    It pins BLAS and OpenMP to one thread before NumPy loads, which only a
    fresh process can do, so it runs in subprocesses here.
    """
    import json
    import subprocess
    import sys

    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        subprocess.run([sys.executable, str(study.ROOT / "benchmarks" /
                                            "run_matched_shot_commutator_krylov.py"),
                        "--systems", SMALL["key"], "--replicas", "5", "--out", str(out)],
                       check=True, capture_output=True, cwd=study.ROOT)
    a, b = (json.loads(path.read_text()) for path in (first, second))
    for record in (a, b):
        record["metadata"].pop("utc")
        for system in record["systems"]:
            system.pop("wall_seconds")
    assert a == b
    [system] = a["systems"]
    assert set(system["arms"]) == set(study.ARMS)
    row = system["arms"]["rt_krylov"]["sizes"]["4"]
    assert set(row["budgets"]) == {str(b) for b in study.BUDGETS}
    assert row["budgets"]["10000"]["shots_per_quantity"] == 10000 // 13
    text = study.report(a)
    assert "`h2o_2x_cas4e3o`" in text and "Real-time Krylov" in text
