"""SecondMomentBank: exact second moments and true Ritz residual norms.

The block is held to a dense ``B' H^2 B`` built with no multivector product,
and every residual to ``reference.dense_residual_norm``, the dense
reconstruction that PLAN.md section 4.4 names as the validation route. The
rows must be the ones the Phase 15 preflight priced: on the declared 8-qubit
banks, the word universe, the coefficient count and the word-set digest must
equal the committed record's.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from benchmarks import check_phase15_preregistration as gate
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.backends import ExactMVBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import (
    SECOND_MOMENT_LABEL,
    MatrixElementBank,
    SecondMomentBank,
    dense_residual_norm,
)
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.reference import dense_basis
from clifford_qc.subspace.second_moment import RESOLUTION

PREFLIGHT = gate.RECORD


def _dimer(size=None):
    model = hubbard(2)
    raw = _raw_pool(model)
    generators = [identity_generator(model.n), *raw[:size]]
    rho = ExactMVBackend().state(model.reference, ())
    return model, generators, MatrixElementBank(rho, model.hamiltonian, generators)


def _dense_square(model, generators, rho):
    basis = dense_basis(rho, generators)
    h = to_matrix(as_multivector(model.hamiltonian))
    return basis.conj().T @ h @ h @ basis


@pytest.mark.parametrize("size", [1, 2, None])
def test_the_block_is_the_dense_second_moment(size):
    model, generators, bank = _dimer(size)
    moments = SecondMomentBank(bank)
    block = moments.matrix()
    dense = _dense_square(model, generators, bank.reference)
    assert np.allclose(block, dense, rtol=0, atol=1e-12)
    assert np.allclose(block, block.conj().T)
    assert np.linalg.eigvalsh(block)[0] >= -1e-12 * np.max(np.abs(block))


def test_residuals_equal_the_dense_reconstruction_on_the_dimer():
    model, generators, bank = _dimer(1)
    result = bank.solve()
    moments = SecondMomentBank(bank)
    for residual in moments.residuals(result):
        dense = dense_residual_norm(bank.reference, model.hamiltonian, generators,
                                    result, residual.root)
        assert abs(residual.variance - dense ** 2) <= RESOLUTION * residual.cancellation_scale
        assert residual.resolved
        assert residual.residual_norm == pytest.approx(dense, rel=1e-10)
        assert residual.rayleigh_energy == pytest.approx(residual.energy, abs=1e-12)


def test_a_complete_basis_has_no_resolved_residual():
    """The full sector: every Ritz state is an eigenstate, to rounding."""
    _, _, bank = _dimer()
    residuals = SecondMomentBank(bank).residuals(bank.solve())
    assert residuals and not any(r.resolved for r in residuals)
    assert all(abs(r.variance) <= RESOLUTION * r.cancellation_scale for r in residuals)


def test_the_variance_at_the_rayleigh_quotient_is_the_residual():
    _, _, bank = _dimer(2)
    result = bank.solve()
    moments = SecondMomentBank(bank)
    for residual in moments.residuals(result):
        variance = moments.energy_variance(result.coefficients[:, residual.root],
                                           result.indices)
        assert variance == pytest.approx(residual.variance, abs=1e-12)


def test_rows_are_the_banks_own_observable_rows():
    """H^2 registers once, and the bank reports its words under that label."""
    _, _, bank = _dimer()
    moments = SecondMomentBank(bank)
    moments.matrix()
    words = bank.resources()["projected_observables"][SECOND_MOMENT_LABEL]["words"]
    assert words == len(moments.word_set())
    assert moments.row(0, 1) is moments.row(0, 1)


def test_a_result_from_another_bank_is_refused():
    _, _, first = _dimer(1)
    _, _, second = _dimer(1)
    with pytest.raises(ValueError, match="not solved on this bank"):
        SecondMomentBank(first).residual(second.solve())
    with pytest.raises(ValueError, match="coefficients for a basis"):
        SecondMomentBank(first).moments(np.ones(5))
    with pytest.raises(TypeError):
        SecondMomentBank(object())


# ------------------------------------------------------------------ declared banks

@pytest.fixture(scope="module")
def preflight():
    return json.loads(PREFLIGHT.read_text(encoding="utf-8"))


def _digest(words):
    codes = np.fromiter(sorted(words), dtype="<u8", count=len(words))
    return hashlib.sha256(codes.tobytes()).hexdigest()


@pytest.mark.parametrize("name", ["h4", "beh2", "hubbard_2x2"])
def test_declared_banks_hold_the_rows_the_preflight_priced(preflight, name):
    """Counts and word-set digest equal the committed record, and the residuals
    equal the dense reconstruction on every root."""
    model, selected = gate.bank_inputs(name)
    bank = gate.first_moment_bank(model, selected)
    moments = SecondMomentBank(bank)
    resources = moments.resources()
    counts = preflight["banks"][name]["counts"]
    assert resources["second_moment_word_universe"] == counts["k_word_universe"]
    assert (resources["second_moment_coefficient_occurrences"]
            == counts["k_coefficient_occurrences"])
    assert resources["combined_word_universe"] == counts["combined_word_universe"]
    assert resources["sh_coefficient_occurrences"] == counts["sh_coefficient_occurrences"]
    assert resources["largest_row_terms"] == counts["largest_row_terms"]
    assert resources["hamiltonian_square_terms"] == counts["h2_terms"]
    assert _digest(moments.word_set()) == preflight["banks"][name]["digests"]["k_words_sha256"]

    result = bank.solve()
    for residual in moments.residuals(result):
        dense = dense_residual_norm(bank.reference, model.hamiltonian, selected,
                                    result, residual.root)
        assert abs(residual.variance - dense ** 2) <= RESOLUTION * residual.cancellation_scale
