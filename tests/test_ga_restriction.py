"""Contracts for the Phase G1 pre-encoding restriction chain (PLAN.md section 3.5).

The filters are cheap to implement in a way that looks right and measures the
wrong thing, so the tests here pin the distinctions the plan spends its prose
on rather than the counts:

* filter B is the *reference-conditioned* condition, not the global commutant --
  a candidate that fails ``[A,Q]=0`` while acting in sector must survive, which
  is what lets G1 agree with the shipped reference-aware filter at all;
* filter E is a quotient of the physical action, not Pauli-word deduplication --
  two candidates with different Pauli support must collapse when they drive the
  reference in the same direction;
* a marginal is a property of the pool and the position in the chain, so D's
  in-order and standalone containment removals are asserted to differ;
* the target character is a live parameter, because hard-coding it forecloses
  section 7.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from clifford_qc.dense_reference import to_matrix
from clifford_qc.fermion import total_number_op, total_sz_op
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace.generator_core import Generator, scalar_free_key
from clifford_qc.subspace.ga_restriction import (
    FILTER_ORDER,
    SectorCharacter,
    basis_state_character,
    determinant_index,
    even_subalgebra_witness,
    hermitian_majorana_monomial,
    majorana_monomial_pool,
    parity_operator,
    reference_action,
    stabilizer_complexification_witness,
    structural_preconditioner,
    word_on_basis_state,
)
from clifford_qc.subspace.restriction import Restriction
from clifford_qc.subspace.symmetry import reference_sector_leakage

N = 4
OCCUPIED = (0, 1)


def _reference_density(n: int, occupied) -> MV:
    bits = "".join("1" if q in set(occupied) else "0" for q in range(n))
    return ket_density(n, bits)


def _chain(pool=None, **kwargs):
    pool = majorana_monomial_pool(N, max_degree=4) if pool is None else pool
    return structural_preconditioner(pool, reference_occupied=OCCUPIED, **kwargs)


# --------------------------------------------------------------------------
# The action layer is exact, and it is the substrate every filter reads
# --------------------------------------------------------------------------

def test_word_action_matches_dense_matrices():
    """Every filter reads this, so it is checked against arithmetic it shares no code with."""
    rng = np.random.default_rng(20260907)
    for _ in range(60):
        code = int(rng.integers(0, 4 ** N))
        index = int(rng.integers(0, 2 ** N))
        vector = np.zeros(2 ** N, dtype=complex)
        vector[index] = 1.0
        expected = to_matrix(MV(N, {code: 1.0})) @ vector
        phase, image = word_on_basis_state(N, code, index)
        actual = np.zeros(2 ** N, dtype=complex)
        actual[image] = phase
        assert np.allclose(expected, actual)


def test_determinant_index_and_character_agree_with_the_occupation_convention():
    index = determinant_index(N, OCCUPIED)
    assert format(index, f"0{N}b") == "1100"
    # Interleaved: even spin orbitals are up, odd are down.
    assert basis_state_character(N, index) == (2, 0.0)


# --------------------------------------------------------------------------
# Section 3.5's algebra claims, computed rather than restated
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", range(N))
def test_occupation_stabilizer_forces_the_complex_algebra(p):
    """``(gamma_2p gamma_2p+1)^2 = -1``, so ``n_p`` carries the ``i`` explicitly.

    This is the computed form of section 3.5's argument that the real
    ``Cl(2n,0)`` branch is vacuous for the stabilizers this plan uses.
    """
    assert stabilizer_complexification_witness(N, p) == pytest.approx(-1.0)


def test_hermitian_monomials_carry_the_canonical_phase():
    for degree in range(5):
        operator = hermitian_majorana_monomial(N, tuple(range(degree)))
        assert operator.is_hermitian()
        # A Majorana monomial is one Pauli word up to phase, which is what makes
        # filter E's "not deduplication" check meaningful.
        assert operator.nnz() == 1


def test_monomial_indices_must_be_distinct_and_ordered():
    with pytest.raises(ValueError, match="distinct"):
        hermitian_majorana_monomial(N, (0, 0))
    with pytest.raises(ValueError, match="increasing"):
        hermitian_majorana_monomial(N, (1, 0))
    with pytest.raises(ValueError, match="must lie in"):
        hermitian_majorana_monomial(N, (0, 2 * N))


def test_parity_operator_is_the_fermion_parity():
    """``prod_j Z_j = (-1)^N`` because ``Z_j = I - 2 n_j``."""
    parity = parity_operator(N)
    for index in range(2 ** N):
        number, _ = basis_state_character(N, index)
        phase, image = word_on_basis_state(N, next(iter(parity.terms)), index)
        assert image == index
        assert complex(phase * next(iter(parity.terms.values()))).real == pytest.approx(
            (-1.0) ** number
        )


# --------------------------------------------------------------------------
# Filter A
# --------------------------------------------------------------------------

def test_filter_a_removes_exactly_the_odd_monomials():
    pool = majorana_monomial_pool(N, max_degree=4)
    report = _chain(pool)
    stage = next(s for s in report.stages if s.key == "A")
    removed = set(stage.removed_labels)
    for candidate in pool:
        degree = 0 if candidate.label == "G()" else candidate.label.count(",") + 1
        assert (candidate.label in removed) == (degree % 2 == 1)
        assert (even_subalgebra_witness(candidate.mv) > 1e-12) == (degree % 2 == 1)


# --------------------------------------------------------------------------
# Filter B -- the distinction section 3.5B insists on
# --------------------------------------------------------------------------

def test_filter_b_is_reference_conditioned_not_the_global_commutant():
    """A candidate failing ``[A,Q]=0`` while acting in sector must survive B.

    This is the contract that makes G1's agreement gate reachable at all: the
    shipped ``reference_sector_leakage`` accepts such candidates, so a filter B
    written as the global commutant would disagree with it by construction --
    and would close section 7.4's excited-state track on the way past.
    """
    report = _chain()
    survivors = set(report.surviving_labels) | {
        label
        for stage in report.stages
        if stage.key in ("D", "C", "E")
        for label in stage.removed_labels
    }
    global_survivors = set(report.global_centralizer_survivors)
    assert global_survivors < survivors, (
        "the global commutant should be a strict subset of what B admits; if it "
        "is not, this instance cannot exercise section 3.5B's distinction"
    )

    number, sz = total_number_op(N), total_sz_op(N)
    witness = sorted(survivors - global_survivors)[0]
    operator = next(c.mv for c in majorana_monomial_pool(N, max_degree=4)
                    if c.label == witness)
    commutes = all(
        (operator * symmetry - symmetry * operator).norm_hs() <= 1e-12
        for symmetry in (number, sz)
    )
    assert not commutes, "the witness was supposed to fail global commutation"
    index = determinant_index(N, OCCUPIED)
    assert {basis_state_character(N, i) for i in reference_action(operator, index)} == {
        (2, 0.0)
    }, "the witness was supposed to act inside the reference sector"


def test_target_character_is_a_live_parameter():
    """Section 3.5B requires this from the first commit, not as a later option."""
    ground = _chain()
    excited = _chain(target_character=SectorCharacter(2, 1.0))
    b_ground = next(s for s in ground.stages if s.key == "B").survived
    b_excited = next(s for s in excited.stages if s.key == "B").survived
    assert b_ground != b_excited
    assert b_excited > 0, "the probe must show B responding, not rejecting everything"


# --------------------------------------------------------------------------
# Filter D, and the overlap the declared order creates
# --------------------------------------------------------------------------

def test_filter_d_containment_is_zero_in_order_and_nonzero_standalone():
    """The overlap with B is measured, not asserted.

    B precedes D and tests the same character on the same action, so D's
    containment clause has nothing left to remove *at its position in the
    chain*. Its standalone removal is B's, and reporting only the in-order zero
    would make D look vacuous when what happened is that B ran first.
    """
    report = _chain()
    stage = next(s for s in report.stages if s.key == "D")
    assert stage.detail["leaves_target_sector"] == 0
    assert report.standalone_removals["D_leaves_target_sector"] > 0
    assert (report.standalone_removals["D_leaves_target_sector"]
            == report.standalone_removals["B"])


def test_filter_d_removes_a_candidate_that_annihilates_the_reference():
    """A pure Majorana monomial never annihilates a determinant, so build one."""
    index = determinant_index(N, OCCUPIED)
    identity = hermitian_majorana_monomial(N, ())
    stabilizer = hermitian_majorana_monomial(N, (0, 1))
    action = reference_action(stabilizer, index)
    scale = next(iter(action.values()))
    annihilating = stabilizer - scale * identity
    assert not annihilating.is_zero()
    assert reference_action(annihilating, index) == {}

    pool = [Generator("A0", identity), Generator("annihilating", annihilating)]
    report = structural_preconditioner(pool, reference_occupied=OCCUPIED)
    stage = next(s for s in report.stages if s.key == "D")
    assert "annihilating" in stage.removed_labels
    assert stage.detail["annihilates_reference"] == 1


# --------------------------------------------------------------------------
# Filter C -- through the shipped primitive, which is G1's third gate
# --------------------------------------------------------------------------

def test_filter_c_runs_through_the_restriction_primitive():
    """An identity restriction fixes nothing, so C must remove nothing."""
    reference = _reference_density(N, OCCUPIED)
    hamiltonian = total_number_op(N)
    report = _chain(
        restriction=Restriction.identity(N),
        hamiltonian=hamiltonian,
        reference_state=reference,
    )
    stage = next(s for s in report.stages if s.key == "C")
    assert stage.removed == 0
    assert stage.detail["qubits_removed"] == 0


@pytest.mark.parametrize("supplied, missing", [
    ({}, "hamiltonian and reference_state"),
    ({"hamiltonian": True}, "reference_state"),
    ({"reference_state": True}, "hamiltonian"),
])
def test_a_restriction_without_its_congruent_inputs_is_refused(supplied, missing):
    """The incongruent-projection call has to fail by name, not deep inside.

    ``Restriction.transport`` moves the Hamiltonian, the reference and the
    candidates together; handing it a restriction and no Hamiltonian used to
    surface as ``TypeError: expected a multivector, got NoneType`` from inside
    the primitive, naming neither the argument nor the filter. Transporting the
    candidates alone is precisely the incongruent projection section 3.5's
    congruence rule exists to prevent, so it is refused at the boundary.
    """
    reference = _reference_density(N, OCCUPIED)
    kwargs = {"hamiltonian": total_number_op(N), "reference_state": reference}
    passed = {name: kwargs[name] for name in supplied}
    with pytest.raises(ValueError, match=f"not {missing}\\."):
        _chain(restriction=Restriction.identity(N), **passed)


def test_filter_c_records_that_it_was_not_evaluated_without_a_restriction():
    """A dropped filter must stay in the ledger, or the chain looks complete."""
    report = _chain()
    stage = next(s for s in report.stages if s.key == "C")
    assert stage.detail == {"not_evaluated": "no restriction supplied"}
    assert [s.key for s in report.stages] == list(FILTER_ORDER)


# --------------------------------------------------------------------------
# Filter E -- the one filter with no post-encoding analogue
# --------------------------------------------------------------------------

def test_filter_e_is_not_pauli_word_deduplication():
    """Distinct Pauli words with proportional action must collapse.

    ``i gamma_0 gamma_1 = 2 n_0 - 1`` is a stabilizer of any determinant, so
    ``G(0,1)`` drives the reference in the identity's direction while being a
    different Pauli word. The package's own ``scalar_free_key`` -- the word
    identity up to an overall factor -- separates them, and E does not.
    """
    identity = hermitian_majorana_monomial(N, ())
    stabilizer = hermitian_majorana_monomial(N, (0, 1))
    assert scalar_free_key(identity) != scalar_free_key(stabilizer)

    pool = [Generator("G()", identity), Generator("G(0,1)", stabilizer)]
    report = structural_preconditioner(pool, reference_occupied=OCCUPIED)
    stage = next(s for s in report.stages if s.key == "E")
    assert stage.removed == 1
    assert report.surviving_labels == ("G()",)
    assert report.distinct_pauli_words == 2


def test_filter_e_keeps_candidates_whose_actions_differ():
    """The converse: shared structure is not enough to collapse two candidates."""
    pool = majorana_monomial_pool(N, max_degree=4)
    report = _chain(pool)
    index = determinant_index(N, OCCUPIED)
    operators = {c.label: c.mv for c in pool}
    reached = [
        frozenset(reference_action(operators[label], index))
        for label in report.surviving_labels
    ]
    assert len(reached) == len(set(reached)), "two survivors drive the same direction"


def test_equivalence_classes_partition_what_entered_e():
    report = _chain()
    stage = next(s for s in report.stages if s.key == "E")
    members = [label for members in report.equivalence_classes for label in members]
    assert len(members) == stage.entered
    assert len(set(members)) == len(members)
    assert len(report.equivalence_classes) == stage.survived


# --------------------------------------------------------------------------
# G1's first gate, on a small instance
# --------------------------------------------------------------------------

def test_filters_a_to_d_reproduce_the_existing_pauli_side_filter():
    """G1's gate 1, at a size the suite can afford to run every time.

    The comparison is against the shipped ``reference_sector_leakage`` rather
    than a reimplementation, so a shared error cannot make the two agree.
    """
    pool = majorana_monomial_pool(N, max_degree=4)
    reference = _reference_density(N, OCCUPIED)
    report = _chain(pool)
    removed = {
        label
        for stage in report.stages
        if stage.key in ("A", "B", "D", "C")
        for label in stage.removed_labels
    }
    ga = sorted(c.label for c in pool if c.label not in removed)

    accepted = []
    for candidate in pool:
        try:
            leakage = reference_sector_leakage(
                candidate.mv, reference, sector_target=(2, 0.0))
        except ValueError as exc:
            if "annihilates the reference" in str(exc):
                continue
            raise
        if leakage["target_sector"] == 0.0:
            accepted.append(candidate.label)
    assert ga == sorted(accepted)


# --------------------------------------------------------------------------
# Input contracts
# --------------------------------------------------------------------------

def test_pool_contracts():
    identity = hermitian_majorana_monomial(N, ())
    with pytest.raises(ValueError, match="empty"):
        structural_preconditioner([], reference_occupied=OCCUPIED)
    with pytest.raises(ValueError, match="duplicate labels"):
        structural_preconditioner(
            [Generator("A", identity), Generator("A", identity)],
            reference_occupied=OCCUPIED)
    with pytest.raises(ValueError, match="different algebras"):
        structural_preconditioner(
            [Generator("A", identity),
             Generator("B", hermitian_majorana_monomial(N + 1, ()))],
            reference_occupied=OCCUPIED)
    with pytest.raises(ValueError, match="out of range"):
        determinant_index(N, (N,))


def test_stage_arithmetic_closes():
    report = _chain()
    entering = report.pool_size
    for stage in report.stages:
        assert stage.entered == entering
        assert stage.entered == stage.survived + stage.removed
        assert stage.removed == len(stage.removed_labels)
        entering = stage.survived
    assert entering == len(report.surviving_labels)
