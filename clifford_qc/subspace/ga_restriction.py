"""Phase G1 -- the pre-encoding structural preconditioner of PLAN.md section 3.5.

Five filters run over an abstract candidate pool *before* a fermion-to-qubit
encoding is chosen, and the deliverable is not the survivor count but the
**per-filter marginal in the order applied**:

    A_raw
      -> A, B   parity / conserved-quantity character   -> A_sym
      -> D      reference ideal                          -> A_ref
      -> C      sector projection                        -> A_phys
      -> E      action-equivalence quotient              -> A_unique

An aggregate "GA removed 82%" is not interpretable here, because filters A-D
have post-encoding analogues the package already applies -- ``sector_leakage``
and ``reference_sector_leakage`` in :mod:`clifford_qc.subspace.symmetry` -- and
reporting their removals as new content double-counts machinery that has been
in the tree since Phase 4.  Filter E is the one filter with no such analogue,
so its marginal is reported separately and is what decides whether Track G has
independent content at all (QG1).

**What "before encoding" does and does not mean here.**  The candidates are
Majorana monomials, and a monomial is fixed by its index tuple
``(mu_1, ..., mu_k)`` -- representation-independent data.  Every filter below is
a function of that data, the declared conserved quantities, and the reference
determinant.  But this package's only concrete arithmetic substrate is the
Jordan-Wigner Pauli image (``CONVENTIONS.md``), so that is what the operators
are *computed* in.  The record must say so: these results are pre-encoding in
the sense that no filter consults the encoding, not in the sense that the
arithmetic ran in some encoding-free representation.  A claim of the second
kind would not survive review.

**The algebra is complex, and not by preference.**  The ``2n`` Majorana
generators generate the real ``Cl(2n,0)``, but the sector stabilizers this
phase actually uses are the occupations
``n_p = (1 + i gamma_2p gamma_2p+1)/2``: a product of two distinct Majoranas
squares to ``-1``, so the ``i`` is carried explicitly and cannot be dropped.
:func:`stabilizer_complexification_witness` computes that square rather than
asserting it, so a future implementation that reaches for the real algebra
fails a check instead of a docstring.

**Filter B is reference-conditioned, and that is load-bearing.**  Global
commutation ``[A, Q] = 0`` is sufficient for sector preservation and *not
necessary*: the condition A-CASE actually needs is
``(Q - q_target) A psi_ref = 0``.  A G1 that hard-codes the global test would
reject candidates the existing reference-aware Pauli filter accepts -- failing
this phase's own agreement gate -- and would foreclose the excited-state track
of section 7.4, which needs a *different* declared character rather than the
reference's.  So the character is a parameter here from the first commit, and
the global centralizer is recorded as a descriptor beside it rather than used
as the filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..clifford import gamma
from ..multivector import MV
from ..pauli import I, Z
from .generator_core import Generator

__all__ = [
    "SectorCharacter",
    "FilterStage",
    "PreconditionerReport",
    "basis_state_character",
    "determinant_index",
    "even_subalgebra_witness",
    "hermitian_majorana_monomial",
    "majorana_monomial_pool",
    "parity_operator",
    "reference_action",
    "stabilizer_complexification_witness",
    "structural_preconditioner",
    "word_on_basis_state",
]

# Amplitudes below this are treated as absent from an action.  Every action
# computed here is a Pauli word applied to a computational basis state -- unit
# modulus phases times the operator's own coefficients -- so nothing lands near
# this threshold by accident; it exists to absorb cancellation between the
# terms of a multi-word candidate.
ACTION_TOLERANCE = 1e-12

FILTER_ORDER = ("A", "B", "D", "C", "E")

FILTER_STATEMENTS = {
    "A": "parity / even subalgebra: keep A with [A, (-1)^N] = 0",
    "B": "conserved-quantity character: keep A with (Q - q_target) A psi_ref = 0 "
         "for every declared Q, which is the reference-conditioned form of the "
         "centralizer and not the global commutant",
    "D": "reference ideal: discard A psi_ref = 0, and require "
         "P_phys A psi_ref = A psi_ref",
    "C": "sector idempotents: keep A with P_s A P_s != 0, evaluated through the "
         "shipped Restriction primitive so the congruence rule holds by "
         "construction",
    "E": "action-equivalence quotient: carry one representative of each class "
         "of P_phys A psi_ref up to a nonzero scalar",
}


@dataclass(frozen=True)
class SectorCharacter:
    """The ``(N, S_z)`` eigenvalues filter B requires of ``A psi_ref``.

    Defaulting this to the reference's own character is the ground-state case.
    Section 3.5B keeps it a parameter because the excited-state track of
    section 7.4 targets a *different* irrep, and an implementation that fixed
    it to the reference's values would close that track at the first commit.
    """

    particle_number: int
    sz: float

    def as_dict(self) -> dict:
        return {"particle_number": int(self.particle_number), "sz": float(self.sz)}


@dataclass(frozen=True)
class FilterStage:
    """One filter's marginal, at the position in the chain where it ran."""

    key: str
    statement: str
    entered: int
    survived: int
    removed_labels: tuple[str, ...]
    detail: dict = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return len(self.removed_labels)

    def as_dict(self) -> dict:
        return {
            "filter": self.key,
            "statement": self.statement,
            "entered": int(self.entered),
            "survived": int(self.survived),
            "removed": int(self.removed),
            "removed_labels": list(self.removed_labels),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class PreconditionerReport:
    """Everything G1 is contracted to deliver for one pool on one instance."""

    n: int
    pool_size: int
    reference_occupied: tuple[int, ...]
    target_character: SectorCharacter
    stages: tuple[FilterStage, ...]
    surviving_labels: tuple[str, ...]
    standalone_removals: dict
    equivalence_classes: tuple[tuple[str, ...], ...]
    distinct_pauli_words: int
    global_centralizer_survivors: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "n_qubits": int(self.n),
            "pool_size": int(self.pool_size),
            "reference_occupied": list(self.reference_occupied),
            "target_character": self.target_character.as_dict(),
            "filter_order": list(FILTER_ORDER),
            "stages": [stage.as_dict() for stage in self.stages],
            "surviving_labels": list(self.surviving_labels),
            "surviving": len(self.surviving_labels),
            "standalone_removals": dict(self.standalone_removals),
            "equivalence_classes": [list(members) for members in self.equivalence_classes],
            "distinct_pauli_words_before_E": int(self.distinct_pauli_words),
            "global_centralizer_survivors": list(self.global_centralizer_survivors),
        }


# --------------------------------------------------------------------------
# The candidate pool
# --------------------------------------------------------------------------

def hermitian_majorana_monomial(n: int, indices: Sequence[int]) -> MV:
    """``i^{k(k-1)/2} gamma_{mu_1} ... gamma_{mu_k}``, which is Hermitian.

    The phase is the canonical one that makes a Majorana blade Hermitian: the
    reverse of a degree-``k`` blade carries ``(-1)^{k(k-1)/2}``, so that factor
    of ``i`` to the same power cancels it.  Fixing it here means two candidates
    differing only by a phase convention cannot enter the pool as distinct
    objects and inflate every marginal below.
    """
    order = tuple(indices)
    if len(set(order)) != len(order):
        raise ValueError("a Majorana monomial needs distinct generator indices")
    if any(not isinstance(mu, int) or not 0 <= mu < 2 * n for mu in order):
        raise ValueError(f"Majorana indices must lie in [0, {2 * n})")
    if list(order) != sorted(order):
        raise ValueError("Majorana indices must be given in increasing order")
    out = I(n)
    for mu in order:
        out = out * gamma(n, mu)
    k = len(order)
    return (1j ** (k * (k - 1) // 2)) * out


def majorana_monomial_pool(n: int, *, max_degree: int) -> list[Generator]:
    """Hermitian Majorana monomials of degree ``0..max_degree``, in index order.

    This is section 3.5's "Majorana-generated candidate pool" taken literally,
    and it is deliberately wider than the excitation family the mapping records
    use.  A pool that has already had parity, particle number and ``S_z``
    imposed at construction cannot measure what filters A and B remove -- it
    would report zero marginals and call the filters vacuous when what happened
    is that the pool builder applied them first.
    """
    from itertools import combinations

    if not isinstance(max_degree, int) or not 0 <= max_degree <= 2 * n:
        raise ValueError(f"max_degree must lie in [0, {2 * n}]")
    out: list[Generator] = []
    for degree in range(max_degree + 1):
        for indices in combinations(range(2 * n), degree):
            label = "G(" + ",".join(str(mu) for mu in indices) + ")"
            out.append(Generator(label, hermitian_majorana_monomial(n, indices)))
    return out


# --------------------------------------------------------------------------
# Exact action on a determinant reference
# --------------------------------------------------------------------------

def determinant_index(n: int, occupied: Iterable[int]) -> int:
    """Basis index of the determinant filling ``occupied``.

    Qubit ``q`` occupies bit ``1 << (n - 1 - q)``, which is the convention
    ``SectorStatevectorBackend.state_from_program`` reads a reference Program
    in, and occupation is the ``Z_q = -1`` eigenvalue because
    ``n_q = (I - Z_q)/2``.
    """
    index = 0
    for q in occupied:
        if not isinstance(q, int) or not 0 <= q < n:
            raise ValueError(f"occupied spin orbital {q!r} out of range for n={n}")
        index |= 1 << (n - 1 - q)
    return index


def word_on_basis_state(n: int, code: int, index: int) -> tuple[complex, int]:
    """``(phase, image)`` for one packed Pauli word on one basis state.

    A Pauli word maps a computational basis state to exactly one other basis
    state times a unit-modulus phase, so the whole action layer below is exact
    up to the candidate's own coefficients -- no eigensolve, no projector, and
    no ``2^n`` intermediate.
    """
    phase = 1.0 + 0.0j
    image = index
    for q in range(n):
        letter = (code >> (2 * q)) & 3
        if letter == 0:
            continue
        bit = (index >> (n - 1 - q)) & 1
        if letter == 3:                      # Z: diagonal, sign on occupation
            if bit:
                phase = -phase
            continue
        image ^= 1 << (n - 1 - q)            # X and Y both flip
        if letter == 2:                      # Y = [[0,-i],[i,0]]
            phase *= -1j if bit else 1j
    return phase, image


def reference_action(operator: MV, index: int) -> dict[int, complex]:
    """``A |index>`` as a sparse map from basis index to amplitude."""
    out: dict[int, complex] = {}
    for code, value in operator.terms.items():
        phase, image = word_on_basis_state(operator.n, code, index)
        out[image] = out.get(image, 0.0) + phase * value
    return {i: v for i, v in out.items() if abs(v) > ACTION_TOLERANCE}


def basis_state_character(n: int, index: int, *, spin_ordering="interleaved"
                          ) -> tuple[int, float]:
    """``(N, S_z)`` of one computational determinant, read off its bits."""
    from ..backends.sector_statevector import _spin_sites

    up_sites, _ = _spin_sites(n, spin_ordering)
    up = set(up_sites)
    number = 0
    sz = 0.0
    for q in range(n):
        if (index >> (n - 1 - q)) & 1:
            number += 1
            sz += 0.5 if q in up else -0.5
    return number, sz


# --------------------------------------------------------------------------
# Structural witnesses -- computed, never asserted
# --------------------------------------------------------------------------

def parity_operator(n: int) -> MV:
    """``(-1)^N = prod_j Z_j``, since ``Z_j = I - 2 n_j``."""
    out = I(n)
    for q in range(n):
        out = out * Z(n, q)
    return out


def even_subalgebra_witness(operator: MV) -> float:
    """Relative Hilbert-Schmidt norm of ``[A, (-1)^N]``; zero iff A is even."""
    norm = operator.norm_hs()
    if norm <= 0.0:
        return 0.0
    parity = parity_operator(operator.n)
    return (operator * parity - parity * operator).norm_hs() / norm


def stabilizer_complexification_witness(n: int, p: int) -> complex:
    """``(gamma_2p gamma_2p+1)^2``, which is ``-1`` and forces the complex algebra.

    Section 3.5 argues that the real ``Cl(2n,0)`` branch is vacuous for the
    stabilizers this plan uses, because occupation is built from a ``k = 2``
    Majorana product.  This computes the square rather than restating the
    argument, so the record carries a number a checker can re-derive.
    """
    bilinear = gamma(n, 2 * p) * gamma(n, 2 * p + 1)
    return complex((bilinear * bilinear).scalar_part())


# --------------------------------------------------------------------------
# The filters
# --------------------------------------------------------------------------

def _sharp_character(action: dict[int, complex], n: int, *, spin_ordering
                     ) -> tuple[int, float] | None:
    """The one ``(N, S_z)`` every occupied component carries, or ``None``."""
    characters = {
        basis_state_character(n, index, spin_ordering=spin_ordering)
        for index in action
    }
    if len(characters) != 1:
        return None
    return next(iter(characters))


def _filter_a(labels, actions, operators, **_):
    return [label for label in labels
            if even_subalgebra_witness(operators[label]) > 1e-12]


def _filter_b(labels, actions, operators, *, n, target, spin_ordering, **_):
    """Reference-conditioned character, not the global commutant.

    An empty action carries no character; that candidate is filter D's to
    remove, and removing it here would move D's marginal into B's.
    """
    removed = []
    for label in labels:
        action = actions[label]
        if not action:
            continue
        character = _sharp_character(action, n, spin_ordering=spin_ordering)
        if character != (target.particle_number, target.sz):
            removed.append(label)
    return removed


def _filter_d(labels, actions, operators, *, n, target, spin_ordering, **_):
    """The reference ideal: annihilation, then target-sector containment."""
    removed = []
    for label in labels:
        action = actions[label]
        if not action:
            removed.append(label)
            continue
        if any(basis_state_character(n, index, spin_ordering=spin_ordering)
               != (target.particle_number, target.sz) for index in action):
            removed.append(label)
    return removed


def _filter_d_split(labels, actions, *, n, target, spin_ordering):
    """D's two clauses counted separately, so the record shows which fired."""
    annihilated, leaking = [], []
    for label in labels:
        action = actions[label]
        if not action:
            annihilated.append(label)
        elif any(basis_state_character(n, index, spin_ordering=spin_ordering)
                 != (target.particle_number, target.sz) for index in action):
            leaking.append(label)
    return annihilated, leaking


def _filter_c(labels, actions, operators, *, restriction, hamiltonian,
              reference_state, **_):
    """Section 3.5C through the shipped ``Restriction``, not a second copy.

    ``transport`` moves the Hamiltonian, the reference and the candidates
    together, which is the congruence rule holding by construction rather than
    by review.  A candidate whose restricted image is zero is exactly section
    3.5C's ``P_s A P_s = 0``, and ``RestrictedProblem`` already reports it.
    """
    if restriction is None:
        return []
    ordered = list(labels)
    problem = restriction.transport(
        hamiltonian=hamiltonian,
        reference=reference_state,
        generators=[operators[label] for label in ordered],
    )
    return [ordered[i] for i in problem.annihilated_indices()]


def _action_class_key(action: dict[int, complex], *, n, target, spin_ordering):
    """Canonical form of ``P_phys A psi_ref`` up to a nonzero complex scalar.

    This is the quotient of section 3.5E and *not* Pauli-word deduplication:
    the key is built from the physical action, so two candidates with entirely
    different Pauli support collapse together when they drive the reference in
    the same direction, and two candidates sharing Pauli support stay apart when
    they do not.

    The projection is applied here rather than assumed.  At E's position in the
    declared chain it is a no-op -- D has already required every survivor's
    action to lie in the target sector -- but ``standalone_removals`` runs each
    filter over the unfiltered pool, and there the distinction is real: without
    the projection, E's standalone number would quotient the *unprojected*
    action and would not be the quantity section 3.5E defines.
    """
    projected = {
        index: value for index, value in action.items()
        if basis_state_character(n, index, spin_ordering=spin_ordering)
        == (target.particle_number, target.sz)
    }
    if not projected:
        return None
    pivot = projected[min(projected)]
    return tuple(
        (index, complex(round((projected[index] / pivot).real, 12),
                        round((projected[index] / pivot).imag, 12)))
        for index in sorted(projected)
    )


def _filter_e(labels, actions, operators, *, n, target, spin_ordering, **_):
    """Keep the first candidate of each action-equivalence class."""
    seen = set()
    removed = []
    for label in labels:
        key = _action_class_key(actions[label], n=n, target=target,
                                spin_ordering=spin_ordering)
        if key is None:                       # no physical direction to share
            continue
        if key in seen:
            removed.append(label)
        else:
            seen.add(key)
    return removed


_FILTERS = {"A": _filter_a, "B": _filter_b, "D": _filter_d,
            "C": _filter_c, "E": _filter_e}


def structural_preconditioner(
    pool: Sequence[Generator],
    *,
    reference_occupied: Sequence[int],
    target_character: SectorCharacter | None = None,
    spin_ordering: str = "interleaved",
    restriction=None,
    hamiltonian: MV | None = None,
    reference_state: MV | None = None,
) -> PreconditionerReport:
    """Run filters A, B, D, C, E in that order and report each marginal.

    ``restriction`` is the object filter C evaluates through.  Passing ``None``
    runs the chain with C recorded as a zero-marginal stage whose detail says it
    was not supplied, rather than silently dropping a filter from the ledger.
    Supplying one *without* ``hamiltonian`` and ``reference_state`` is refused:
    filter C evaluates through ``Restriction.transport``, which moves all three
    objects together, and that congruence is the whole reason C runs through the
    shipped primitive rather than a private projection.
    """
    if not pool:
        raise ValueError("the candidate pool is empty")
    n = pool[0].n
    if any(candidate.n != n for candidate in pool):
        raise ValueError("pool candidates live in different algebras")
    labels = [candidate.label for candidate in pool]
    if len(set(labels)) != len(labels):
        raise ValueError("pool contains duplicate labels")
    if restriction is not None:
        missing = [
            name for name, value in (("hamiltonian", hamiltonian),
                                     ("reference_state", reference_state))
            if value is None
        ]
        if missing:
            raise ValueError(
                f"filter C was given a restriction but not {' and '.join(missing)}. "
                "Section 3.5C is evaluated through Restriction.transport, which "
                "carries the Hamiltonian, the reference and the candidates together; "
                "transporting the candidates alone is exactly the incongruent "
                "projection that primitive exists to prevent. Pass both, or pass "
                "restriction=None to record C as not evaluated."
            )
    operators = {candidate.label: candidate.mv for candidate in pool}

    index = determinant_index(n, reference_occupied)
    if target_character is None:
        number, sz = basis_state_character(n, index, spin_ordering=spin_ordering)
        target_character = SectorCharacter(number, sz)
    actions = {label: reference_action(operators[label], index) for label in labels}

    context = {
        "n": n,
        "target": target_character,
        "spin_ordering": spin_ordering,
        "restriction": restriction,
        "hamiltonian": hamiltonian,
        "reference_state": reference_state,
    }

    stages: list[FilterStage] = []
    live = list(labels)
    entering_e: list[str] = []
    for key in FILTER_ORDER:
        if key == "E":
            entering_e = list(live)
        removed = _FILTERS[key](live, actions, operators, **context)
        removed_set = set(removed)
        detail: dict = {}
        if key == "D":
            annihilated, leaking = _filter_d_split(
                live, actions, n=n, target=target_character,
                spin_ordering=spin_ordering)
            detail = {
                "annihilates_reference": len(annihilated),
                "leaves_target_sector": len(leaking),
                "clause_overlap_with_B": (
                    "B precedes D and tests the same character on the same "
                    "action, so leaves_target_sector is expected to be 0 here. "
                    "It is reported rather than dropped because its value is "
                    "what shows the two filters overlap, and standalone_removals "
                    "records what D's containment clause removes on its own."
                ),
            }
        if key == "C" and restriction is None:
            detail = {"not_evaluated": "no restriction supplied"}
        elif key == "C":
            detail = {
                "restriction_label": getattr(restriction, "label", None),
                "qubits_removed": int(getattr(restriction, "qubits_removed", 0)),
            }
        live = [label for label in live if label not in removed_set]
        stages.append(FilterStage(
            key=key,
            statement=FILTER_STATEMENTS[key],
            entered=len(live) + len(removed_set),
            survived=len(live),
            removed_labels=tuple(removed),
            detail=detail,
        ))

    standalone = {}
    for key in FILTER_ORDER:
        standalone[key] = len(_FILTERS[key](labels, actions, operators, **context))
    annihilated, leaking = _filter_d_split(
        labels, actions, n=n, target=target_character, spin_ordering=spin_ordering)
    standalone["D_annihilates_reference"] = len(annihilated)
    standalone["D_leaves_target_sector"] = len(leaking)

    # Every candidate that reaches E, grouped by the class E quotients over.
    classes: dict = {}
    for label in entering_e:
        key = _action_class_key(actions[label], n=n, target=target_character,
                                spin_ordering=spin_ordering)
        classes.setdefault(key, []).append(label)

    # The count that makes E's marginal a finding rather than deduplication.
    # ``scalar_free_key`` is the package's existing Pauli-word identity up to an
    # overall factor, so a pool whose entrants are all distinct under it cannot
    # have had a single removal explained by duplicate words.
    from .generator_core import scalar_free_key
    words = {scalar_free_key(operators[label]) for label in entering_e}

    number_op, sz_op = _global_symmetry_operators(n, spin_ordering)
    global_survivors = tuple(
        label for label in labels
        if _commutes(operators[label], number_op) and _commutes(operators[label], sz_op)
    )

    return PreconditionerReport(
        n=n,
        pool_size=len(pool),
        reference_occupied=tuple(sorted(reference_occupied)),
        target_character=target_character,
        stages=tuple(stages),
        surviving_labels=tuple(live),
        standalone_removals=standalone,
        equivalence_classes=tuple(
            tuple(members) for _, members in sorted(
                classes.items(), key=lambda item: item[1][0])
        ),
        distinct_pauli_words=len(words),
        global_centralizer_survivors=global_survivors,
    )


def _global_symmetry_operators(n: int, spin_ordering):
    from ..fermion import total_number_op, total_sz_op
    return total_number_op(n), total_sz_op(n, spin_ordering=spin_ordering)


def _commutes(operator: MV, symmetry: MV, tol: float = 1e-12) -> bool:
    norm = operator.norm_hs()
    if norm <= 0.0:
        return True
    return (operator * symmetry - symmetry * operator).norm_hs() / norm <= tol
