"""Preconditioned residual expansion: the corrected accuracy hierarchy.

This driver replaces the "residual oracle" that an earlier plan proposed as a
research phase.  Unpreconditioned residual expansion needs no phase, because it
is a theorem rather than a question.  With ``|Psi_m> = p_{m-1}(H)|psi>`` the
Ritz residual is

    |r_m> = (H - E_m) p_{m-1}(H) |psi>  in  K_{m+1},

and Galerkin orthogonality gives ``r_m ⟂ K_m``.  Except at happy breakdown,
normalising and appending ``r_m`` therefore *is* Lanczos written in an
orthogonal basis: it spans the Krylov space at ``S = I`` instead of at the
``kappa(S) ~ 1e8..1e16`` that the committed ``fixed_krylov`` arm carries.  So
``orthogonal_residual`` appears here only as a **regression arm**.

What the arms are for
---------------------

``orthogonal_residual``
    Lanczos regression test.  Must reproduce the in-script ``power_krylov``
    energy at matched ``M`` while holding ``S = I``.

``davidson``
    The principal method.  ``(D - E_m + mu)^-1`` applied to the residual before
    orthogonalisation.  This is the largest per-``M`` accuracy lever measured in
    this repository, and it beats the Krylov arm the accuracy program has been
    calibrating against -- Krylov is an *unpreconditioned baseline*, not a span
    ceiling.

``packet_davidson``
    The implementability question: does a top-``K`` determinant packet retain
    the Davidson direction, and what does it cost in Pauli words?  Gated behind
    a word-cost preflight, never run blind.

``matched_selected_ci``
    Mandatory classical control.  Its repeated ties with A-CASE in the Phase 12
    record mean the operator construction has not yet demonstrated greater
    energy compactness than classical determinant selection, so every accuracy
    claim here is reported against it.

``random``
    The floor of the same family: the same number of determinants drawn
    uniformly from the sector, scoring nothing.  ``matched_selected_ci`` asks
    whether an arm beats *smart* classical selection at its budget; this asks
    whether it beats *chance* at that budget, which has to be settled first --
    otherwise "this subspace is good" and "a subspace of this size is good" are
    the same number and the arm takes credit for the second.  Its seed is
    derived from the record's own seed, so the row moves with the run and is
    reproducible within it; ``draw_sha256`` pins the draw itself.

Claim boundaries
----------------

1. **The preconditioner is classical.**  ``(D - E_m + mu)^-1`` is cheap in the
   sector/statevector validation backend.  It is *not* a cheap quantum
   operation, and nothing here compiles it into a bounded-support measurable
   packet.  Every Davidson row is stamped ``exact_simulation`` with
   ``preconditioner_category = "classical_preconditioner"`` and
   ``implementable = false``.  The ``packet_davidson`` arm is the first step
   toward removing that caveat; it does not remove it yet.

2. **No exact ground energy enters any selection.**  The shift ``mu`` is chosen
   on the projected Ritz energy, which is variational and therefore a legal
   classical criterion; references are chosen by declared fixed policies or by
   the sample-independent classical selected-CI ranking.  Arms that do consult
   the exact ground state are named ``oracle_*`` and carry
   ``evidence_category = "oracle_diagnostic"``.  They are diagnostics, never
   headline accuracy.

3. **Reference sensitivity is not reference-optimisation advantage.**  The
   committed H4 warm-start rows show sensitivity; they do not establish a
   general advantage, and the sector-projected row's QND projection circuit is
   unpriced in its own record.  Both facts are restated in the output.

Run from the repository root, for example::

    python -m benchmarks.run_preconditioned_expansion --systems hubbard_2x2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# The documented command invokes this file directly, so the repository root has
# to be importable before the sibling benchmark modules are read.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import run_phase10_hybrid as phase10
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import identity_generator, run_control
from clifford_qc.subspace.configuration import (configuration_generator,
                                                determinant_program)
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.symmetry import (sector_leakage,
                                          subspace_sector_certificate)

ROOT = Path(__file__).resolve().parent
PRIMARY_SYSTEMS = phase10.PRIMARY_SYSTEMS

REQUIRED_ARMS = (
    "power_krylov",
    "orthonormalized_power_krylov",
    "orthogonal_residual",
    "davidson",
    "matched_selected_ci",
    "random",
)

# Every row carries every field.  A field may be ``None`` where the quantity is
# genuinely undefined for an arm, but it may never disappear from the row.
REQUIRED_FIELDS = (
    "method",
    "evidence_category",
    "implementable",
    "preconditioner_category",
    "seed",
    "M",
    "declared_total_directions",
    "realized_total_directions",
    "stopped_reason",
    "nested_energies",
    "reference_policy",
    "reference_identity",
    "reference_block_size",
    "shift_rule",
    "mu",
    "packet_K",
    "retained_rank",
    "kappa_S",
    "orthonormality_defect",
    "energy",
    "energy_error",
    "absolute_error",
    "variance",
    "true_residual",
    "matvecs",
    "selection_work",
    "W_total",
    "W_incremental",
    "W_scope",
    "grouping_contexts",
    "build_seconds",
    "solve_seconds",
    "wall_seconds",
)

# Declared once, swept in full, and retained in full.  The selected value is
# whichever minimises the projected Ritz energy -- variational, so legal.
DEFAULT_SHIFT_GRID = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

DEFAULT_PACKET_K = (2, 4, 8, 16, 32)

# The preflight refuses to price a packet whose incremental word universe is
# already larger than this multiple of the retained bank.  A K that fails here
# is reported and skipped, never silently run.
DEFAULT_WORD_BUDGET_RATIO = 4.0

# The Lanczos regression is gated against the SVD-orthonormalized Krylov basis,
# so the tolerance can be absolute rather than sized by kappa(S).  A tolerance
# proportional to the raw monomial arm's conditioning is not a test: on
# hubbard_2x3 that arm reaches kappa(S) ~ 7e15, which would admit tens of
# hartree.
REGRESSION_ENERGY_TOLERANCE = 1e-9
REGRESSION_SPAN_TOLERANCE = 1e-7


# --------------------------------------------------------------- operator glue


class _CountingOperator:
    """Sector operator wrapper that counts matvecs as a reported resource."""

    def __init__(self, operator):
        self._operator = operator
        self.matvecs = 0

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self.matvecs += 1
        return self._operator.matvec(vector)


def _sector_diagonal(apply, dimension: int) -> np.ndarray:
    """``D_II = H_II`` on the sector basis, by one matvec per basis state.

    Dense-free but linear in the sector dimension, which is why this driver
    stays on the small primary systems.  The matvec count is reported rather
    than hidden: Davidson's advantage is not free even classically.
    """
    diagonal = np.empty(dimension, dtype=float)
    probe = np.zeros(dimension, dtype=complex)
    for index in range(dimension):
        probe[index] = 1.0
        diagonal[index] = float(np.real(apply(probe)[index]))
        probe[index] = 0.0
    return diagonal


def _ritz(basis: np.ndarray, apply):
    """Lowest Ritz pair on the span of ``basis``, in an orthonormal basis.

    Returns ``(energy, vector, Q, defect)``.  ``defect`` is
    ``||Q'Q - I||_max``: the ``S = I`` invariant is measured, not assumed.
    """
    Q, _ = np.linalg.qr(basis)
    projected = np.column_stack([apply(Q[:, j]) for j in range(Q.shape[1])])
    matrix = Q.conj().T @ projected
    matrix = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(matrix)
    gram = Q.conj().T @ Q
    defect = float(np.abs(gram - np.eye(Q.shape[1])).max())
    return float(values[0]), Q @ vectors[:, 0], Q, defect


def _residual(apply, state: np.ndarray, energy: float) -> np.ndarray:
    return apply(state) - energy * state


def _top_k_packet(vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-``k`` determinant components of ``vector`` and their positions."""
    kept = np.argsort(-np.abs(vector))[:k]
    packet = np.zeros_like(vector)
    packet[kept] = vector[kept]
    return packet, np.sort(kept)


def expand(apply, reference: np.ndarray, target_M: int, *,
           diagonal: np.ndarray | None = None, mu: float | None = None,
           packet_K: int | None = None) -> dict:
    """Residual-driven subspace expansion to ``target_M`` directions.

    ``mu is None`` gives the unpreconditioned (Lanczos) arm.  A float ``mu``
    applies the Davidson preconditioner ``(D - E_m + mu)^-1`` to the residual
    before orthogonalisation.  ``packet_K`` truncates the correction to its
    largest ``K`` determinant components, re-orthogonalising afterwards so that
    the retained span stays exactly nested.
    """
    basis = reference.reshape(-1, 1).astype(complex)
    stopped = "budget_exhausted"
    energies: list[float] = []
    packet_supports: list[np.ndarray] = []
    packet_coefficients: list[np.ndarray] = []
    packet_norm_captures: list[float] = []
    energy, state, Q, defect = _ritz(basis, apply)
    energies.append(energy)

    while basis.shape[1] < target_M:
        correction = _residual(apply, state, energy)
        if mu is not None:
            if diagonal is None:
                raise ValueError("the Davidson arm needs the sector diagonal")
            denominator = diagonal - energy + mu
            # A zero denominator is a genuine breakdown of this shift, not a
            # number to clip silently: the shift sweep exists to avoid it, and
            # a swept mu that hits it is reported as a failed grid point.
            if np.min(np.abs(denominator)) < 1e-12:
                return {"breakdown": "singular_denominator", "mu": mu,
                        "M": int(basis.shape[1]), "energies": energies}
            correction = correction / denominator
        # Two Gram-Schmidt passes, not one.  A single pass loses orthogonality
        # precisely where the residual norm gets small -- which is where this
        # arm is supposed to agree with power Krylov to the last digit it can.
        correction = correction - Q @ (Q.conj().T @ correction)
        correction = correction - Q @ (Q.conj().T @ correction)
        if packet_K is not None:
            full_correction_norm = float(np.linalg.norm(correction))
            correction, support = _top_k_packet(correction, packet_K)
            retained_correction_norm = float(np.linalg.norm(correction))
            packet_norm_captures.append(
                0.0 if full_correction_norm <= 0.0 else
                (retained_correction_norm / full_correction_norm) ** 2)
            # The packet that is actually retained at this step, coefficients
            # and all.  Pricing needs every one of them: the trajectory appends
            # a *different* packet at every iteration, so a cost read off the
            # first one describes a benchmark this driver does not run.
            packet_supports.append(support)
            packet_coefficients.append(correction[support].copy())
            correction = correction - Q @ (Q.conj().T @ correction)
            correction = correction - Q @ (Q.conj().T @ correction)
        norm = float(np.linalg.norm(correction))
        if norm < 1e-12:                       # happy breakdown: span is closed
            stopped = "happy_breakdown"
            break
        basis = np.hstack([basis, (correction / norm).reshape(-1, 1)])
        energy, state, Q, defect = _ritz(basis, apply)
        energies.append(energy)

    residual = _residual(apply, state, energy)
    acted = apply(state)
    return {
        "energy": energy,
        "state": state,
        "basis": Q,
        "M": int(basis.shape[1]),
        "stopped_reason": stopped,
        "energies": energies,
        "orthonormality_defect": defect,
        "kappa_S": float(np.linalg.cond(Q.conj().T @ Q)),
        "retained_rank": int(np.linalg.matrix_rank(basis, tol=1e-10)),
        "true_residual": float(np.linalg.norm(residual)),
        "variance": float(np.real(np.vdot(acted, acted)) - energy ** 2),
        "packet_supports": packet_supports,
        "packet_coefficients": packet_coefficients,
        "packet_norm_captures": packet_norm_captures,
    }


def orthonormalized_power_krylov(apply, reference: np.ndarray, M: int, *,
                                 rank_tol: float = 1e-10) -> dict:
    """The Krylov space solved in an SVD-orthonormalized basis.

    This is the arm the regression gate compares against.  Solving the raw
    monomial basis through a Gram-whitened pencil loses digits in proportion to
    ``kappa(S)``, and a tolerance sized by that conditioning is not a test: at
    ``kappa(S) = 7.4e15`` it would admit an energy error of tens of hartree.
    Orthonormalizing the same basis by SVD removes the conditioning from the
    comparison entirely, so ``orthogonal_residual`` can be held to a tight
    absolute tolerance against a basis for the *same span*.

    The retained rank is reported because it is the honest way the two arms can
    legitimately differ: a rank-deficient monomial basis spans less, and that is
    a difference of span, not of arithmetic.
    """
    columns = [reference.astype(complex)]
    for _ in range(M - 1):
        columns.append(apply(columns[-1]))
    basis = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(basis, full_matrices=False)
    keep = singular > rank_tol * float(singular.max())
    Q = left[:, keep]
    projected = np.column_stack([apply(Q[:, j]) for j in range(Q.shape[1])])
    matrix = Q.conj().T @ projected
    matrix = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(matrix)
    state = Q @ vectors[:, 0]
    acted = apply(state)
    energy = float(values[0])
    return {
        "energy": energy,
        "state": state,
        "basis": Q,
        "M": M,
        "kappa_S": 1.0,
        "retained_rank": int(keep.sum()),
        "rank_tol": float(rank_tol),
        "orthonormality_defect": float(
            np.abs(Q.conj().T @ Q - np.eye(Q.shape[1])).max()),
        "true_residual": float(np.linalg.norm(acted - energy * state)),
        "variance": float(np.real(np.vdot(acted, acted)) - energy ** 2),
    }


def subspace_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Largest principal-angle sine between two orthonormal bases.

    ``max sin(theta)`` of ``(I - Q1 Q1') Q2`` -- zero exactly when the second
    span sits inside the first.  This is what decides whether two arms really
    solved the same subspace, independently of what their eigensolves returned.
    """
    residual = second - first @ (first.conj().T @ second)
    return float(np.clip(np.linalg.svd(residual, compute_uv=False), 0.0, 1.0).max())


def power_krylov(apply, reference: np.ndarray, M: int) -> dict:
    """Raw ``H^k`` monomials, solved through the repository's thresholded GEP.

    The ill-conditioned comparator, kept so the regression invariant is checked
    against an arm built in this script rather than against a remembered number.
    """
    columns = [reference.astype(complex)]
    for _ in range(M - 1):
        columns.append(apply(columns[-1]))
    basis = np.column_stack(columns)
    overlap = basis.conj().T @ basis
    projected = basis.conj().T @ np.column_stack(
        [apply(basis[:, j]) for j in range(basis.shape[1])])
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = np.linalg.eigh(overlap)
    keep = values > 1e-14 * float(values.max())
    whitened = vectors[:, keep] / np.sqrt(values[keep])
    reduced = whitened.conj().T @ projected @ whitened
    reduced = 0.5 * (reduced + reduced.conj().T)
    spectrum, reduced_vectors = np.linalg.eigh(reduced)
    state = basis @ (whitened @ reduced_vectors[:, 0])
    state = state / np.linalg.norm(state)
    acted = apply(state)
    energy = float(spectrum[0])
    return {
        "energy": energy,
        "M": M,
        "kappa_S": float(np.linalg.cond(overlap)),
        "retained_rank": int(keep.sum()),
        "orthonormality_defect": None,
        "true_residual": float(np.linalg.norm(acted - energy * state)),
        "variance": float(np.real(np.vdot(acted, acted)) - energy ** 2),
    }


# ------------------------------------------------------------- shift selection


def select_shift(apply, reference: np.ndarray, target_M: int,
                 diagonal: np.ndarray, grid) -> dict:
    """Sweep the declared ``mu`` grid and keep the whole curve.

    Selection uses the *projected* Ritz energy.  That is variational, so
    minimising it is a legal classical criterion and never consults the exact
    ground energy.  The selection work -- grid points and matvecs spent -- is
    part of the method's cost and is reported as such.
    """
    curve = []
    best = None
    for mu in grid:
        started = time.perf_counter()
        run = expand(apply, reference, target_M, diagonal=diagonal, mu=float(mu))
        point = {
            "mu": float(mu),
            "breakdown": run.get("breakdown"),
            "projected_energy": None if "breakdown" in run else run["energy"],
            "true_residual": None if "breakdown" in run else run["true_residual"],
            "variance": None if "breakdown" in run else run["variance"],
            "M": int(run["M"]),
            "seconds": time.perf_counter() - started,
        }
        curve.append(point)
        if point["projected_energy"] is None:
            continue
        if best is None or point["projected_energy"] < best["projected_energy"]:
            best = point
    if best is None:
        raise RuntimeError("every declared shift broke down; widen the grid")
    return {
        "curve": curve,
        "selected_mu": best["mu"],
        "selection_criterion": "min projected Ritz energy (variational)",
        "oracle_free": True,
        "grid": [float(mu) for mu in grid],
        "selection_work": len(curve),
    }


# ----------------------------------------------------------- reference policies


def _occupied_from_mask(mask: int, n: int) -> tuple[int, ...]:
    """Occupied spin orbitals of a sector bitmask (qubit ``j`` is bit ``n-1-j``)."""
    return tuple(j for j in range(n) if (mask >> (n - 1 - j)) & 1)


def _neel_mask(backend: SectorStatevectorBackend) -> int | None:
    """A fixed physics-informed lattice reference: alternating site occupation.

    Interleaved spin ordering puts site ``s`` on qubits ``2s`` (up) and
    ``2s+1`` (down), so the antiferromagnetic determinant takes up on even
    sites and down on odd ones.  Returned only if it lands in the sector.
    """
    n = backend.n
    occupied = [2 * site if site % 2 == 0 else 2 * site + 1
                for site in range(n // 2)]
    mask = 0
    for orbital in occupied:
        mask |= 1 << (n - 1 - orbital)
    return mask if mask in set(backend.basis.tolist()) else None


def reference_policies(model, backend, diagonal: np.ndarray,
                       seed: int) -> dict[str, dict]:
    """Declared single-determinant reference policies, all oracle-free."""
    basis = backend.basis
    policies: dict[str, dict] = {}

    model_mask = int(basis[int(np.argmax(np.abs(
        backend.state_from_program(model.reference))))])
    policies["model_reference"] = {
        "mask": model_mask,
        "note": "model.reference, the determinant every committed arm uses"}

    policies["lowest_diagonal"] = {
        "mask": int(basis[int(np.argmin(diagonal))]),
        "note": "argmin of the sector diagonal; classical, no eigenvector seen"}

    neel = _neel_mask(backend)
    if neel is not None:
        policies["physics_informed_neel"] = {
            "mask": neel,
            "note": "fixed antiferromagnetic determinant, declared in advance"}

    rng = np.random.default_rng(seed)
    policies["fixed_seed_random"] = {
        "mask": int(basis[int(rng.integers(basis.size))]),
        "note": f"seeded control determinant (seed={seed}); a null policy"}

    for name, entry in policies.items():
        position = int(np.searchsorted(basis, entry["mask"]))
        entry["position"] = position
        entry["occupied"] = list(_occupied_from_mask(entry["mask"], backend.n))
        entry["identity"] = f"{name}:mask={entry['mask']}"
    return policies


def classical_reference_block(operator, anchor_position: int, L: int,
                              exact_energy: float | None = None) -> dict:
    """Top-``L`` determinant block from the sample-independent classical ranking.

    This is ``matched_selected_ci``'s selection rule reused as a *reference
    builder*: the anchor determinant plus the highest-scoring determinants under
    one classical criterion.  It never sees a sampled set and never sees the
    exact ground energy, so an ``L``-block reference stays oracle-free.
    """
    control = run_control(operator, np.array([anchor_position], dtype=np.int64),
                          name=f"reference_block_L{L}", kind="matched_selected_ci",
                          max_determinants=L, exact_energy=exact_energy)
    return {"indices": np.asarray(control.determinants, dtype=np.int64),
            "record": control.to_record()}


def block_vector(operator, indices: np.ndarray, dimension: int) -> np.ndarray:
    """Lowest eigenvector of the Hamiltonian restricted to ``indices``.

    The reference block is compressed to *one* direction before expansion, so
    an ``L``-determinant block costs ``L`` determinants of state complexity but
    contributes a single Ritz direction.  That is what makes the budget match
    below meaningful.
    """
    restricted = operator.restrict(indices)
    values, vectors = np.linalg.eigh(np.asarray(restricted, dtype=complex))
    vector = np.zeros(dimension, dtype=complex)
    vector[indices] = vectors[:, 0]
    return vector / np.linalg.norm(vector)


# ----------------------------------------------------------- word-cost preflight


def _packet_generators(reference_occupied, masks, n: int, prefix: str,
                       coefficients=None):
    """Aligned ``(generator, coefficient)`` pairs for one packet.

    The reference determinant enters as the identity, which the growth already
    seeds, so it is dropped -- and it must be dropped from the *coefficients
    too*.  Filtering the generators while truncating the coefficient array from
    the end silently shifts every coefficient after the reference onto the wrong
    generator whenever the reference is not last, which is the common case: in
    the committed Hubbard 2x2 ``K=32`` packet the reference sits at index 22 of
    32.  Carrying the pair through one filter makes that misalignment
    unrepresentable.
    """
    reference_program = determinant_program(n, reference_occupied)
    weights = (None if coefficients is None
               else np.asarray(coefficients, dtype=complex).reshape(-1))
    if weights is not None and weights.size != len(masks):
        raise ValueError(
            f"packet has {len(masks)} determinants but {weights.size} "
            "coefficients; they must be given in the same order")
    reference_key = tuple(sorted(reference_occupied))
    pairs = []
    for position, mask in enumerate(masks):
        occupied = _occupied_from_mask(int(mask), n)
        if tuple(sorted(occupied)) == reference_key:
            continue                      # the identity direction, already seeded
        generator = configuration_generator(
            reference_program, determinant_program(n, occupied),
            label=f"{prefix}[{position}]")
        pairs.append((generator,
                      None if weights is None else complex(weights[position])))
    return pairs


def _compile_packet(pairs, label: str):
    """The single generator ``A_new = sum_k c_k A_k`` from aligned pairs."""
    from clifford_qc.subspace.generator_core import Generator

    combined = None
    for generator, weight in pairs:
        term = generator.mv * (1.0 if weight is None else weight)
        combined = term if combined is None else combined + term
    return Generator(label, combined)


def price_packet_basis(model, backend, reference_entry, retained_masks,
                       packets, *, spin_ordering="interleaved",
                       numeric_certificate_max_qubits: int = 8,
                       abort_above: int | None = None) -> dict:
    """Price the packet basis a run actually retains.

    ``packets`` is one ``(masks, coefficients)`` pair per *realized* packet
    direction.  Pricing the first packet only and reusing its number for the
    whole trajectory describes a one-step benchmark, not this one: the
    expansion appends a different top-``K`` correction at every iteration, and
    the packet-packet cross elements among those directions are exactly the
    cost that decides whether the compression is affordable.

    Three word counts, in declared scopes, because they are not
    interchangeable:

    ``determinant_baseline_W``
        the matched determinant bank at the same ``M`` -- what a determinant
        A-CASE arm would measure.

    ``packet_bank_W_total``
        the full universe of the retained packet bank.  This is the number that
        is like-for-like with an A-CASE row's ``W``, and it is what the gate
        compares.

    ``W_incremental``
        what the packet directions add *on top of* the matched determinant
        bank.  Never comparable to an A-CASE total, and never reported as one.

    ``abort_above`` stops the pricing as soon as the running universe passes a
    budget, so a ``K`` that is going to be rejected is not paid for in full.
    """
    n = model.n
    sector_masks = set(int(mask) for mask in backend.basis.tolist())
    reference_occupied = reference_entry["occupied"]
    from clifford_qc.backends import ExactMVBackend
    rho = ExactMVBackend().state(
        determinant_program(n, reference_occupied), ())

    retained = ([identity_generator(n)]
                + [generator for generator, _ in _packet_generators(
                    reference_occupied, retained_masks, n, "ret")])

    compiled, parts, all_masks, products = [], [], [], 0
    for step, (masks, coefficients) in enumerate(packets):
        pairs = _packet_generators(reference_occupied, masks, n,
                                   f"pkt{step}", coefficients)
        if not pairs:
            continue
        compiled.append(_compile_packet(pairs, f"packet[step={step}]"))
        parts.extend(generator for generator, _ in pairs)
        all_masks.extend(int(mask) for mask in masks)
    if not compiled:
        return {"skipped": "every packet was the reference determinant"}

    # (0) matched determinant bank.
    baseline = MatrixElementBank(rho, model.hamiltonian, retained)
    baseline.matrices()
    baseline_words = baseline.word_set()

    # (1) the retained packet bank, priced pair by pair so a hopeless K can be
    #     abandoned rather than completed.
    packet_bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(n)] + compiled)
    order = list(packet_bank.resolve(None))
    universe: set[int] = set()
    aborted = None
    for i_index, i in enumerate(order):
        for j in order[i_index:]:
            universe.update(packet_bank.overlap_operator(i, j).terms)
            universe.update(packet_bank.element_operator(i, j).terms)
        if abort_above is not None and len(universe) > abort_above:
            aborted = {"aborted_after_generators": i_index + 1,
                       "of_generators": len(order),
                       "words_so_far": int(len(universe)),
                       "abort_above": int(abort_above)}
            break

    result = {
        "packet_directions": len(compiled),
        "packet_parts_total": len(parts),
        "determinant_baseline_W": int(len(baseline_words)),
        "packet_bank_W_total": int(len(universe)),
        "W_incremental": int(len(universe - baseline_words)),
        "scope_note": ("packet_bank_W_total is the full universe of the "
                       "retained packet bank and is the only count comparable "
                       "with an A-CASE row's W; W_incremental is measured "
                       "against the matched determinant bank"),
    }
    if aborted is not None:
        result["pricing_aborted"] = aborted
        return result

    # (2) no-cancellation bound over the products among parts and retained.
    bound_bank = MatrixElementBank(rho, model.hamiltonian, retained)
    bound_bank.matrices()
    part_ids = [bound_bank.add(generator) for generator in parts]
    retained_ids = list(bound_bank.resolve(None))
    bound_words: set[int] = set()
    for i in part_ids:
        for j in part_ids + retained_ids:
            bound_words.update(bound_bank.overlap_operator(i, j).terms)
            bound_words.update(bound_bank.element_operator(i, j).terms)
            products += 1
    result["pair_support_bound"] = int(len(bound_words - baseline_words))
    result["products"] = int(products)
    result["cancellation_factor"] = float(
        result["pair_support_bound"] / max(1, result["W_incremental"]))

    # (3) sector certificates.  Structural first, because it is a proof: every
    #     packet part reaches one sector basis determinant, so no combination
    #     of them can leave the sector.  The numeric certificate builds the
    #     explicit sector projector -- a small-n object -- so it runs only as a
    #     cross-check where it is affordable.
    stray = sorted({mask for mask in all_masks if mask not in sector_masks})
    result["structural_sector_certificate"] = {
        "packet_determinants": len(all_masks),
        "outside_sector": stray,
        "in_sector": not stray,
        "argument": "every packet part reaches one sector basis determinant, "
                    "so the packet cannot leave the sector"}
    result["operator_global_leakage"] = {
        key: float(value)
        for key, value in sector_leakage(compiled[0],
                                         spin_ordering=spin_ordering).items()}
    result["operator_global_leakage_note"] = (
        "nonzero by construction: a configuration generator is an X-string and "
        "does not commute with N or Sz. Not a defect, and not the certificate "
        "the invariant checks")
    if n <= numeric_certificate_max_qubits:
        result["reference_conditioned_certificate"] = {
            key: (value if isinstance(value, (str, bool)) else float(value))
            for key, value in subspace_sector_certificate(
                rho, compiled, spin_ordering=spin_ordering).items()}
    else:
        result["reference_conditioned_certificate"] = {
            "skipped": f"n={n} exceeds numeric_certificate_max_qubits="
                       f"{numeric_certificate_max_qubits}; the explicit sector "
                       "projector is a small-n object and the structural "
                       "certificate already decides this case"}
    return result


DEFAULT_GROUPING_WORD_LIMIT = 20000


def grouping_contexts(model, backend, reference_entry, packets, *,
                      word_limit: int = DEFAULT_GROUPING_WORD_LIMIT):
    """QWC groups for the *whole* retained packet bank, not its first step.

    The scope travels in the return value, because a group count over one
    packet direction reported beside a word count over seven is not a cost
    model -- it is two different experiments in adjacent fields.
    """
    n = model.n
    from clifford_qc.backends import ExactMVBackend

    reference_occupied = reference_entry["occupied"]
    rho = ExactMVBackend().state(determinant_program(n, reference_occupied), ())
    compiled = []
    for step, (masks, coefficients) in enumerate(packets):
        pairs = _packet_generators(reference_occupied, masks, n,
                                   f"pkt{step}", coefficients)
        if pairs:
            compiled.append(_compile_packet(pairs, f"packet[step={step}]"))
    if not compiled:
        return None
    bank = MatrixElementBank(rho, model.hamiltonian,
                             [identity_generator(n)] + compiled)
    bank.matrices()
    universe = len(bank.word_set())
    scope = "packet bank (identity + realized packet directions)"
    # The greedy QWC partition is quadratic in the universe size, which is why
    # the bank keeps it off the default resource report.  Above the limit this
    # returns the reason instead of the count.
    if universe > word_limit:
        return {"skipped": f"word universe {universe} exceeds the grouping "
                           f"limit {word_limit}; the QWC partition is "
                           "quadratic in the universe size",
                "scope": scope, "word_universe": int(universe)}
    return {"groups": int(bank.qwc_group_count()), "scope": scope,
            "word_universe": int(universe)}


# ------------------------------------------------------------------ row builder


def _blank_row(method: str, category: str, seed: int) -> dict:
    row = {field: None for field in REQUIRED_FIELDS}
    row.update({"method": method, "evidence_category": category, "seed": seed,
                "implementable": False, "preconditioner_category": "none",
                "shift_rule": "none", "selection_work": 0})
    return row


def _finish_row(row: dict, run: dict, exact_energy: float, *,
                matvecs: int, wall: float) -> dict:
    """Fill one row, folding in any cost already charged to it.

    Shift selection is charged onto the row by ``_charged_shift_sweep`` before
    the expansion runs.  Folding it in here -- and consuming it, so a row
    cannot be billed twice -- is what keeps the audit and reference-block arms
    comparable with the primary Davidson arm, which used to be the only one
    whose sweep was counted.
    """
    energy = float(run["energy"])
    matvecs = int(matvecs) + int(row.pop("_charged_matvecs", 0))
    wall = float(wall) + float(row.pop("_charged_seconds", 0.0))
    row.update({
        "M": int(run["M"]),
        "stopped_reason": run.get("stopped_reason"),
        "nested_energies": [float(value) for value in run.get("energies", ())],
        "retained_rank": run.get("retained_rank"),
        "kappa_S": run.get("kappa_S"),
        "orthonormality_defect": run.get("orthonormality_defect"),
        "energy": energy,
        "energy_error": energy - exact_energy,
        "absolute_error": abs(energy - exact_energy),
        "variance": run.get("variance"),
        "true_residual": run.get("true_residual"),
        "matvecs": int(matvecs),
        "wall_seconds": float(wall),
    })
    if row.get("realized_total_directions") is None:
        row["realized_total_directions"] = int(
            (row.get("reference_block_size") or 1) + int(run["M"]) - 1)
    return row


# ------------------------------------------------------------------ the ladder


def _charged_shift_sweep(counting, reference, target_M, diagonal, grid, row):
    """Sweep the shift and bill the sweep to the row that consumed it.

    The primary Davidson arm folded its sweep into its own matvec and wall
    counts; the audit and reference-block arms called ``select_shift`` outside
    their timing window and so reported a tenth of their true cost.  Every arm
    that selects a shift pays for the selection here.
    """
    before, started = counting.matvecs, time.perf_counter()
    sweep = select_shift(counting, reference, target_M, diagonal, grid)
    row["mu"] = sweep["selected_mu"]
    row["selection_work"] = sweep["selection_work"]
    row["_charged_matvecs"] = counting.matvecs - before
    row["_charged_seconds"] = time.perf_counter() - started
    return sweep


def run_system(name: str, *, seed: int = 0, total_directions: int = 7,
               shift_grid=DEFAULT_SHIFT_GRID, packet_K=DEFAULT_PACKET_K,
               word_budget_ratio: float = DEFAULT_WORD_BUDGET_RATIO,
               reference_blocks=(1, 2, 4),
               regression_tolerance: float = REGRESSION_ENERGY_TOLERANCE,
               span_tolerance: float = REGRESSION_SPAN_TOLERANCE) -> dict:
    """The full preconditioned-expansion ladder for one primary system."""
    if name not in PRIMARY_SYSTEMS:
        raise ValueError(f"unknown primary system {name!r}")
    if total_directions < 2:
        raise ValueError("total_directions must be at least 2")

    model, construction = phase10.build_system(name)
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]), float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    dimension = int(backend.dimension)
    counting = _CountingOperator(operator)

    values, _ = backend.ground_state(model.hamiltonian, k=1)
    exact_energy = float(values[0])

    diagonal_start = counting.matvecs
    diagonal = _sector_diagonal(counting, dimension)
    diagonal_matvecs = counting.matvecs - diagonal_start

    policies = reference_policies(model, backend, diagonal, seed)
    primary = policies["model_reference"]

    def unit(position: int) -> np.ndarray:
        vector = np.zeros(dimension, dtype=complex)
        vector[position] = 1.0
        return vector

    rows: list[dict] = []

    def timed(row, thunk):
        before, started = counting.matvecs, time.perf_counter()
        run = thunk()
        return _finish_row(row, run, exact_energy,
                           matvecs=counting.matvecs - before,
                           wall=time.perf_counter() - started), run

    # --- power Krylov: the ill-conditioned comparator, built in this script.
    row = _blank_row("power_krylov", "exact_simulation", seed)
    row.update({"reference_policy": "model_reference",
                "reference_identity": primary["identity"],
                "reference_block_size": 1, "declared_total_directions": total_directions})
    krylov_row, krylov_run = timed(
        row, lambda: power_krylov(counting, unit(primary["position"]),
                                  total_directions))
    rows.append(krylov_row)

    # --- the same Krylov span, orthonormalized: the regression comparator.
    row = _blank_row("orthonormalized_power_krylov", "exact_simulation", seed)
    row.update({"reference_policy": "model_reference",
                "reference_identity": primary["identity"],
                "reference_block_size": 1,
                "declared_total_directions": total_directions,
                "shift_rule": "none (unpreconditioned)"})
    orthonormal_row, orthonormal_run = timed(
        row, lambda: orthonormalized_power_krylov(
            counting, unit(primary["position"]), total_directions))
    rows.append(orthonormal_row)

    # --- orthogonal residual: the Lanczos regression arm.
    row = _blank_row("orthogonal_residual", "exact_simulation", seed)
    row.update({"reference_policy": "model_reference",
                "reference_identity": primary["identity"],
                "reference_block_size": 1, "declared_total_directions": total_directions,
                "shift_rule": "none (unpreconditioned)"})
    orthogonal_row, orthogonal_run = timed(
        row, lambda: expand(counting, unit(primary["position"]), total_directions))
    rows.append(orthogonal_row)

    # --- Davidson: the principal method, over the declared shift sweep.
    row = _blank_row("davidson", "exact_simulation", seed)
    sweep = _charged_shift_sweep(counting, unit(primary["position"]),
                                 total_directions, diagonal, shift_grid, row)
    row.update({
        "reference_policy": "model_reference",
        "reference_identity": primary["identity"],
        "reference_block_size": 1, "declared_total_directions": total_directions,
        "preconditioner_category": "classical_preconditioner",
        "shift_rule": "(D - E_m + mu)^-1, mu from the declared grid by "
                      "min projected Ritz energy"})
    davidson_row, davidson_run = timed(
        row, lambda: expand(counting, unit(primary["position"]), total_directions,
                            diagonal=diagonal, mu=sweep["selected_mu"]))
    rows.append(davidson_row)

    # --- mandatory classical control at the matched direction budget.
    started = time.perf_counter()
    control = run_control(operator, np.array([primary["position"]], dtype=np.int64),
                          name="matched_selected_ci", kind="matched_selected_ci",
                          max_determinants=total_directions,
                          exact_energy=exact_energy)
    row = _blank_row("matched_selected_ci", "classical_control", seed)
    row.update({
        "reference_policy": "model_reference",
        "reference_identity": primary["identity"],
        "reference_block_size": 1,
        "declared_total_directions": total_directions,
        "implementable": True,
        "M": int(control.determinant_count),
        "energy": float(control.energy),
        "energy_error": float(control.energy) - exact_energy,
        "absolute_error": abs(float(control.energy) - exact_energy),
        "variance": float(control.variance),
        "kappa_S": 1.0,
        "retained_rank": int(control.determinant_count),
        "realized_total_directions": int(control.determinant_count),
        "stopped_reason": "determinant_budget",
        "selection_work": int(control.selection_work),
        "matvecs": 0,
        "build_seconds": float(control.build_seconds),
        "solve_seconds": float(control.solve_seconds),
        "wall_seconds": time.perf_counter() - started})
    rows.append(row)

    # --- the floor of that family: the same budget, drawn without scoring.
    # Reported beside `matched_selected_ci` rather than instead of it: the
    # matched control is the ceiling an arm has to beat to claim a selection
    # advantage, and this is the floor it has to clear to claim anything at all.
    started = time.perf_counter()
    floor = run_control(operator, np.array([primary["position"]], dtype=np.int64),
                        name="random", kind="random",
                        max_determinants=total_directions,
                        seed=seed, exact_energy=exact_energy)
    row = _blank_row("random", "classical_control", seed)
    row.update({
        "reference_policy": "model_reference",
        "reference_identity": primary["identity"],
        "reference_block_size": 1,
        "declared_total_directions": total_directions,
        "implementable": True,
        "M": int(floor.determinant_count),
        "energy": float(floor.energy),
        "energy_error": float(floor.energy) - exact_energy,
        "absolute_error": abs(float(floor.energy) - exact_energy),
        "variance": float(floor.variance),
        "kappa_S": 1.0,
        "retained_rank": int(floor.determinant_count),
        "realized_total_directions": int(floor.determinant_count),
        "stopped_reason": "determinant_budget",
        "selection_work": int(floor.selection_work),
        "matvecs": 0,
        "build_seconds": float(floor.build_seconds),
        "solve_seconds": float(floor.solve_seconds),
        "wall_seconds": time.perf_counter() - started})
    rows.append(row)

    # --- reference-policy audit at a fixed direction budget.
    audit = []
    for policy_name, entry in sorted(policies.items()):
        base = _blank_row(f"davidson[{policy_name}]", "exact_simulation", seed)
        base.update({
            "reference_policy": policy_name,
            "reference_identity": entry["identity"],
            "reference_block_size": 1,
            "declared_total_directions": total_directions,
            "preconditioner_category": "classical_preconditioner",
            "shift_rule": "declared-grid mu, reselected per reference",
            "mu": None})
        local = _charged_shift_sweep(counting, unit(entry["position"]),
                                     total_directions, diagonal, shift_grid, base)
        filled, _ = timed(base, lambda e=entry, m=local["selected_mu"]: expand(
            counting, unit(e["position"]), total_directions,
            diagonal=diagonal, mu=m))
        audit.append(filled)

        plain = _blank_row(f"orthogonal_residual[{policy_name}]",
                           "exact_simulation", seed)
        plain.update({"reference_policy": policy_name,
                      "reference_identity": entry["identity"],
                      "reference_block_size": 1,
                      "declared_total_directions": total_directions,
                      "shift_rule": "none (unpreconditioned)"})
        filled, _ = timed(plain, lambda e=entry: expand(
            counting, unit(e["position"]), total_directions))
        audit.append(filled)

    # --- budget-matched reference blocks: L determinants + (total - L) growth.
    blocks = []
    for L in reference_blocks:
        if L >= total_directions:
            continue
        block = classical_reference_block(operator, primary["position"], L,
                                          exact_energy)
        indices = block["indices"]
        vector = block_vector(operator, indices, dimension)
        target_M = total_directions - L + 1
        for label, mu in (("orthogonal_residual", None), ("davidson", None)):
            base = _blank_row(f"{label}[classical_block_L{L}]",
                              "exact_simulation", seed)
            chosen = None
            if label == "davidson":
                local = _charged_shift_sweep(counting, vector, target_M,
                                             diagonal, shift_grid, base)
                chosen = local["selected_mu"]
                base["preconditioner_category"] = "classical_preconditioner"
                base["shift_rule"] = "declared-grid mu, reselected per block"
            base.update({
                "reference_policy": f"classical_block_L{L}",
                "reference_identity": f"classical_block_L{L}:"
                                      f"dets={indices.tolist()}",
                "reference_block_size": int(indices.size),
                "declared_total_directions": total_directions,
                "mu": chosen})
            filled, _ = timed(base, lambda v=vector, t=target_M, m=chosen: expand(
                counting, v, t, diagonal=diagonal, mu=m))
            # The budget identity the invariant checks: block determinants
            # plus growth directions.  A block that closes its span early
            # realizes *less* than the declared budget; that is a happy
            # breakdown to report, not a budget violation to hide.
            filled["realized_total_directions"] = int(
                filled["reference_block_size"] + filled["M"] - 1)
            blocks.append(filled)

    # --- oracle diagnostics, quarantined by name and category.
    ground_vector = backend.ground_state(model.hamiltonian, k=1)[1][:, 0]
    oracle = []
    for L in reference_blocks:
        if L >= total_directions:
            continue
        kept = np.argsort(-np.abs(ground_vector))[:L]
        vector = np.zeros(dimension, dtype=complex)
        vector[kept] = ground_vector[kept]
        vector = vector / np.linalg.norm(vector)
        target_M = total_directions - L + 1
        base = _blank_row(f"oracle_distilled_L{L}", "oracle_diagnostic", seed)
        base.update({
            "reference_policy": f"oracle_distilled_L{L}",
            "reference_identity": f"oracle_distilled_L{L}:"
                                  f"dets={sorted(int(i) for i in kept)}",
            "reference_block_size": int(L),
            "declared_total_directions": total_directions,
            "shift_rule": "none (unpreconditioned); oracle reference"})
        filled, _ = timed(base, lambda v=vector, t=target_M: expand(
            counting, v, t))
        filled["realized_total_directions"] = int(L + filled["M"] - 1)
        oracle.append(filled)

    # --- the regression evidence: same span, and same answer on it.
    krylov_agreement = {
        "orthonormalized_energy": float(orthonormal_run["energy"]),
        "orthogonal_residual_energy": float(orthogonal_run["energy"]),
        "energy_gap": float(abs(orthonormal_run["energy"]
                                - orthogonal_run["energy"])),
        "energy_tolerance": float(regression_tolerance),
        "orthonormalized_rank": int(orthonormal_run["retained_rank"]),
        "orthogonal_residual_rank": int(orthogonal_run["M"]),
        "ranks_match": bool(orthonormal_run["retained_rank"]
                            == orthogonal_run["M"]),
        "subspace_gap": float(subspace_gap(orthonormal_run["basis"],
                                           orthogonal_run["basis"])),
        "subspace_tolerance": float(span_tolerance),
        "raw_power_energy": float(krylov_run["energy"]),
        "raw_power_rank": int(krylov_run["retained_rank"]),
        "raw_power_kappa_S": float(krylov_run["kappa_S"]),
        "raw_power_gap": float(abs(krylov_run["energy"]
                                   - orthogonal_run["energy"])),
        "note": ("the gate compares orthogonal_residual against the SVD-"
                 "orthonormalized Krylov basis at a fixed absolute tolerance "
                 "and against its span by principal angle. The raw monomial "
                 "solve is reported for contrast only: its Gram-whitened "
                 "pencil loses digits with kappa(S) and can retain fewer "
                 "directions, so it is not fit to gate anything")}

    # --- word-cost preflight, then grouping only for the survivors.
    packet_report = _packet_program(
        model, backend, counting, diagonal, primary, exact_energy,
        total_directions, sweep["selected_mu"], packet_K, word_budget_ratio,
        seed, timed, name)

    record = {
        "schema": "clifford_qc.preconditioned_expansion.v1",
        "system": name,
        "construction": construction,
        "n_qubits": int(model.n),
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "sector_dimension": dimension,
        "sector_masks": [int(mask) for mask in backend.basis.tolist()],
        "spin_ordering": str(backend.spin_ordering),
        "exact_energy": exact_energy,
        "total_directions": total_directions,
        "seed": seed,
        "diagonal_matvecs": int(diagonal_matvecs),
        "arms": rows,
        "reference_audit": audit,
        "reference_blocks": blocks,
        "oracle_diagnostics": oracle,
        "shift_sweep": sweep,
        "krylov_agreement": krylov_agreement,
        "packet_program": packet_report,
        "reference_policies": {
            key: {"identity": value["identity"], "note": value["note"],
                  "occupied": value["occupied"]}
            for key, value in sorted(policies.items())},
        "claim_boundary": (
            "orthogonal_residual is a Lanczos regression arm, not a result: "
            "normalised residual expansion spans the Krylov space by "
            "construction. davidson is exact_simulation with a classical "
            "preconditioner and is not an implementable hardware primitive "
            "until the correction is compiled into a bounded-support "
            "measurable packet. oracle_* rows consult the exact ground state "
            "and are diagnostics only. The committed H4 warm-start rows show "
            "reference *sensitivity*; they do not establish a general "
            "reference-optimisation advantage, the 2/4/6-operator ordering "
            "there is a single system and remains unexplained, and that "
            "record's sector-projected row leaves its QND projection circuit "
            "unpriced."),
        "evidence": {
            "energies": "exact sector statevector simulation",
            "resource_counts": "exact algorithmic counts",
            "physical_shot_budget": "not estimated",
            "quantum_advantage_claim": False},
    }
    check_invariants(record)
    return record


COMMITTED_LADDER = ROOT / "results" / "phase12_paper_b_five_system.json"


def committed_acase_word_cost(system: str) -> dict:
    """Words per retained direction of the committed A-CASE arm, if available.

    The packet gate needs an anchor with a meaning: "does one residual packet
    direction cost more Pauli words than one A-CASE direction on this same
    system, and by how much".  A self-referential ratio against a bank the
    driver just built answers nothing.  When the committed record is absent the
    gate falls back to the identity-baseline ratio and says so.
    """
    if not COMMITTED_LADDER.exists():
        return {"available": False,
                "note": "committed Phase 12 record not present; the gate falls "
                        "back to the identity-baseline ratio"}
    document = json.loads(COMMITTED_LADDER.read_text(encoding="utf-8"))
    for record in document.get("systems", ()):
        if record.get("system_key") != system:
            continue
        for arm in record.get("arms", ()):
            if arm.get("method") != "acase" or not arm.get("W"):
                continue
            directions = max(1, int(arm.get("M") or 1))
            return {"available": True,
                    "source": str(COMMITTED_LADDER.name),
                    "acase_W": int(arm["W"]),
                    "acase_M": directions,
                    "words_per_direction": float(arm["W"]) / directions}
    return {"available": False,
            "note": f"no A-CASE arm with a word count for {system!r}"}


def _packet_program(model, backend, counting, diagonal, primary, exact_energy,
                    total_directions, mu, packet_K, word_budget_ratio, seed,
                    timed, system: str) -> dict:
    """Run each ``K``, price the basis it actually retained, then gate grouping.

    The expansion itself is only matvecs, so it is cheap; the word bank is what
    costs.  Running first and pricing the realized trajectory is therefore both
    cheaper and honest, where pricing one probe packet and reusing the number
    for a seven-step run was neither.  Pricing carries an abort budget, so a
    ``K`` headed for rejection is abandoned rather than completed, and grouping
    is paid only for the survivors.
    """
    dimension = int(backend.dimension)
    vector = np.zeros(dimension, dtype=complex)
    vector[primary["position"]] = 1.0

    # A determinant bank matched to this M: the reference plus the determinants
    # carrying the most weight in the uncompressed Ritz state.  This is what a
    # determinant A-CASE arm at the same budget would have to measure.
    uncompressed = expand(counting, vector, total_directions,
                          diagonal=diagonal, mu=mu)
    ranked = np.argsort(-np.abs(uncompressed["state"]))[:total_directions]
    retained_masks = [int(backend.basis[primary["position"]])]
    retained_masks += [int(backend.basis[i]) for i in ranked
                       if int(backend.basis[i]) not in retained_masks]
    retained_masks = retained_masks[:total_directions]

    anchor = committed_acase_word_cost(system)
    budget = (int(word_budget_ratio * anchor["acase_W"])
              if anchor.get("available") else None)

    pricing, accepted, rejected, runs = [], [], [], []
    packet_masks: dict[str, list[int]] = {}
    for K in packet_K:
        if K > dimension:
            continue
        row = _blank_row(f"packet_davidson[K={K}]", "exact_simulation", seed)
        row.update({
            "reference_policy": "model_reference",
            "reference_identity": primary["identity"],
            "reference_block_size": 1,
            "declared_total_directions": total_directions,
            "preconditioner_category": "classical_preconditioner",
            "shift_rule": "(D - E_m + mu)^-1 at the selected mu",
            "mu": float(mu), "packet_K": int(K)})
        filled, run = timed(row, lambda k=K: expand(
            counting, vector, total_directions, diagonal=diagonal, mu=mu,
            packet_K=k))

        packets = [([int(backend.basis[i]) for i in support], coefficients)
                   for support, coefficients in zip(run["packet_supports"],
                                                    run["packet_coefficients"])]
        packet_masks[str(int(K))] = sorted(
            {mask for masks, _ in packets for mask in masks})
        try:
            entry = price_packet_basis(model, backend, primary, retained_masks,
                                       packets, abort_above=budget)
        except (ValueError, MemoryError) as error:      # pragma: no cover
            entry = {"failed": str(error)}
        entry["K"] = int(K)
        entry["packet_steps"] = len(packets)

        total = entry.get("packet_bank_W_total")
        if total is None:
            entry["accepted"] = False
            entry["gate_basis"] = "pricing failed"
        elif budget is not None:
            entry["acase_W"] = anchor["acase_W"]
            entry["cost_vs_acase_total"] = float(total / anchor["acase_W"])
            entry["accepted"] = bool("pricing_aborted" not in entry
                                     and total <= budget)
            entry["gate_basis"] = ("committed A-CASE total word universe on "
                                   "this system, like for like")
        else:
            ratio = total / max(1, entry["determinant_baseline_W"])
            entry["incremental_ratio"] = float(ratio)
            entry["accepted"] = bool(ratio <= word_budget_ratio)
            entry["gate_basis"] = "matched determinant bank (fallback)"
        pricing.append(entry)

        # Word counts go into explicitly scoped fields.  ``W`` stays None on
        # these rows: an unlabeled total-cost column that silently held an
        # increment is what made these rows read as comparable with A-CASE
        # totals when they were not.
        filled["W_total"] = total
        filled["W_incremental"] = entry.get("W_incremental")
        filled["W_scope"] = entry.get("scope_note")
        if entry["accepted"]:
            accepted.append(int(K))
            filled["grouping_contexts"] = grouping_contexts(
                model, backend, primary, packets)
        else:
            rejected.append(int(K))
            filled["grouping_contexts"] = {
                "skipped": "K rejected by the word-cost gate",
                "scope": "packet bank (identity + realized packet directions)"}
        runs.append(filled)

    # The full-K control: keeping every component must reproduce the
    # uncompressed Davidson direction exactly.
    full = expand(counting, vector, total_directions, diagonal=diagonal, mu=mu,
                  packet_K=dimension)
    return {
        "pricing": pricing,
        "packet_masks": packet_masks,
        "accepted_K": accepted,
        "rejected_K": rejected,
        "word_budget_ratio": float(word_budget_ratio),
        "word_budget": budget,
        "anchor": anchor,
        "gate_rule": ("each K is run (matvecs only), the basis it actually "
                      "retained is priced with an abort budget, and a K is "
                      "accepted only if its packet-bank total word universe is "
                      "within word_budget_ratio times the committed A-CASE "
                      "total on this system. QWC grouping is paid for "
                      "survivors only, over the whole packet bank"),
        "runs": runs,
        "full_K_control": {
            "full_K_energy": full["energy"],
            "uncompressed_energy": uncompressed["energy"],
            "difference": abs(full["energy"] - uncompressed["energy"])},
    }


# -------------------------------------------------------------------invariants


def check_invariants(record: dict, *, tolerance: float = 1e-9) -> None:
    """Fail at the producer boundary, before an unusable record is written."""
    exact = float(record["exact_energy"])
    arms = {row["method"]: row for row in record["arms"]}
    problems: list[str] = []

    for name in REQUIRED_ARMS:
        if name not in arms:
            problems.append(f"required arm {name!r} is missing")
    every = (record["arms"] + record["reference_audit"]
             + record["reference_blocks"] + record["oracle_diagnostics"]
             + record["packet_program"]["runs"])
    for row in every:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            problems.append(f"{row.get('method')}: missing fields {missing}")
        energy = row.get("energy")
        if energy is not None and energy < exact - 1e-9:
            problems.append(
                f"{row['method']} is below the variational bound: "
                f"{energy} < {exact}")

    # 1. Orthogonal residual reproduces the Krylov space.  Equality is checked
    #    as "no worse, and not better than the conditioning gap allows": the
    #    orthogonal arm is a *sharper* Ritz value of the same span, so a strict
    #    equality assertion would fail exactly where the arm is doing its job.
    orthogonal = arms.get("orthogonal_residual")
    orthonormal = arms.get("orthonormalized_power_krylov")
    if orthogonal and orthonormal and orthogonal["M"] != orthonormal["M"]:
        problems.append("orthogonal_residual and orthonormalized_power_krylov "
                        "are not at matched M")
    agreement = record.get("krylov_agreement")
    if agreement is None:
        problems.append("no Krylov agreement evidence was recorded")
    else:
        # Same span, orthonormal bases on both sides, so the comparison is a
        # real one: a fixed absolute tolerance, plus the principal angle that
        # decides whether the spans are actually equal.
        if agreement["ranks_match"]:
            if agreement["energy_gap"] > agreement["energy_tolerance"]:
                problems.append(
                    f"orthogonal_residual and the orthonormalized Krylov basis "
                    f"disagree by {agreement['energy_gap']:.3g}, beyond the "
                    f"tolerance {agreement['energy_tolerance']:.3g}")
            if agreement["subspace_gap"] > agreement["subspace_tolerance"]:
                problems.append(
                    f"orthogonal_residual does not span the Krylov space: "
                    f"principal-angle gap {agreement['subspace_gap']:.3g} "
                    f"exceeds {agreement['subspace_tolerance']:.3g}")
        elif agreement["orthogonal_residual_rank"] < \
                agreement["orthonormalized_rank"]:
            # The residual arm spanning *less* is a defect; spanning more just
            # means the monomial basis went numerically rank-deficient first.
            problems.append(
                "orthogonal_residual retained fewer directions "
                f"({agreement['orthogonal_residual_rank']}) than the "
                f"orthonormalized Krylov basis "
                f"({agreement['orthonormalized_rank']})")

    # 2. S = I for every orthonormal expansion arm.
    for row in every:
        defect = row.get("orthonormality_defect")
        if defect is not None and defect > 1e-10:
            problems.append(f"{row['method']}: orthonormality defect {defect:.3e}")

    # 3. Nested Ritz energies are non-increasing.
    for row in every:
        curve = row.get("nested_energies") or ()
        for earlier, later in zip(curve, curve[1:]):
            if later > earlier + tolerance:
                problems.append(
                    f"{row['method']}: Ritz energy increased along nesting "
                    f"({earlier} -> {later})")

    # 4. Full-K packet reproduces the uncompressed Davidson direction.
    control = record["packet_program"]["full_K_control"]
    if control["difference"] > 1e-10:
        problems.append(
            "the full-K packet does not reproduce uncompressed Davidson "
            f"(difference {control['difference']:.3e})")

    # 5. Reference and total-direction budgets match exactly.
    declared = int(record["total_directions"])
    matched = record["reference_blocks"] + record["oracle_diagnostics"]
    for row in matched:
        if int(row["declared_total_directions"]) != declared:
            problems.append(
                f"{row['method']}: declared budget "
                f"{row['declared_total_directions']} != {declared}")
        realized = int(row["realized_total_directions"])
        if realized > declared:
            problems.append(
                f"{row['method']}: realized budget {realized} exceeds the "
                f"declared {declared}")
        if realized < declared and row["stopped_reason"] != "happy_breakdown":
            problems.append(
                f"{row['method']}: realized budget {realized} < declared "
                f"{declared} without a happy breakdown "
                f"(stopped_reason={row['stopped_reason']!r})")

    # 6. No exact ground energy enters reference or shift selection.
    sweep = record["shift_sweep"]
    if not sweep.get("oracle_free"):
        problems.append("the shift sweep is not marked oracle-free")
    if "projected" not in sweep["selection_criterion"]:
        problems.append("the shift criterion is not the projected Ritz energy")
    recomputed = min((point for point in sweep["curve"]
                      if point["projected_energy"] is not None),
                     key=lambda point: point["projected_energy"])
    if abs(recomputed["mu"] - sweep["selected_mu"]) > 0:
        problems.append("the selected shift is not the curve's own minimiser")
    for row in record["arms"] + record["reference_audit"]:
        if row["evidence_category"] == "oracle_diagnostic":
            problems.append(f"{row['method']} is an oracle row outside the "
                            "quarantined diagnostics list")
    for row in record["oracle_diagnostics"]:
        if not row["method"].startswith("oracle_"):
            problems.append(f"{row['method']} is in the oracle list unnamed")

    # 7. Spin ordering and sector certificates propagate.
    if not record.get("spin_ordering"):
        problems.append("the spin ordering convention is not recorded")
    for entry in record["packet_program"]["pricing"]:
        structural = entry.get("structural_sector_certificate")
        if structural is not None and not structural["in_sector"]:
            problems.append(
                f"packet K={entry['K']} reaches determinants outside the "
                f"sector: {structural['outside_sector']}")
        if entry.get("accepted") and entry.get("packet_steps", 0) < 1:
            problems.append(
                f"packet K={entry['K']} was accepted without pricing any "
                "realized packet direction")
        certificate = entry.get("reference_conditioned_certificate") or {}
        if "max_sector_leakage" not in certificate:
            continue                       # skipped above the qubit threshold
        worst = float(certificate["max_sector_leakage"])
        if worst > 1e-9:
            problems.append(
                f"packet K={entry['K']} fails the numeric reference-conditioned "
                f"certificate: {worst:.3g}")
        if structural is not None and structural["in_sector"] and worst > 1e-9:
            problems.append(
                f"packet K={entry['K']}: the structural and numeric sector "
                "certificates disagree")
    for masks in record["packet_program"].get("packet_masks", {}).values():
        stray = [mask for mask in masks if mask not in record["sector_masks"]]
        if stray:
            problems.append(f"packet determinants outside the sector: {stray}")

    if problems:
        raise AssertionError("preconditioned-expansion invariants failed:\n  "
                             + "\n  ".join(problems))


# ------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", default=",".join(PRIMARY_SYSTEMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-directions", type=int, default=7,
                        help="M for the matched comparison (Phase 12 uses 7)")
    parser.add_argument("--shift-grid", default=",".join(
        str(mu) for mu in DEFAULT_SHIFT_GRID))
    parser.add_argument("--packet-k", default=",".join(
        str(k) for k in DEFAULT_PACKET_K))
    parser.add_argument("--word-budget-ratio", type=float,
                        default=DEFAULT_WORD_BUDGET_RATIO)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "preconditioned_expansion.json")
    args = parser.parse_args(argv)

    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    unknown = sorted(set(systems) - set(PRIMARY_SYSTEMS))
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    shift_grid = tuple(float(mu) for mu in args.shift_grid.split(","))
    packet_K = tuple(int(k) for k in args.packet_k.split(","))

    records = [run_system(name, seed=args.seed,
                          total_directions=args.total_directions,
                          shift_grid=shift_grid, packet_K=packet_K,
                          word_budget_ratio=args.word_budget_ratio)
               for name in systems]
    document = stamp_record({
        "schema": "clifford_qc.preconditioned_expansion_record.v1",
        "systems": records,
        "required_arms": list(REQUIRED_ARMS),
        "required_fields": list(REQUIRED_FIELDS),
        "claim_boundary": records[0]["claim_boundary"] if records else "",
    }, execution_provenance())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    for record in records:
        arms = {row["method"]: row for row in record["arms"]}
        print(f"{record['system']}: exact={record['exact_energy']:.9f} "
              f"dim={record['sector_dimension']}")
        for name in REQUIRED_ARMS:
            row = arms[name]
            print(f"  {name:<22s} M={row['M']:<3d} "
                  f"err={abs(row['absolute_error'])*1e3:12.5g} mHa "
                  f"kappa={row['kappa_S']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
