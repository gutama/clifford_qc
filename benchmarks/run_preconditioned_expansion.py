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
    "orthogonal_residual",
    "davidson",
    "matched_selected_ci",
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
    "W",
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
            correction, support = _top_k_packet(correction, packet_K)
            correction = correction - Q @ (Q.conj().T @ correction)
            correction = correction - Q @ (Q.conj().T @ correction)
            packet_supports.append(support)
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
        "M": int(basis.shape[1]),
        "stopped_reason": stopped,
        "energies": energies,
        "orthonormality_defect": defect,
        "kappa_S": float(np.linalg.cond(Q.conj().T @ Q)),
        "retained_rank": int(np.linalg.matrix_rank(basis, tol=1e-10)),
        "true_residual": float(np.linalg.norm(residual)),
        "variance": float(np.real(np.vdot(acted, acted)) - energy ** 2),
        "packet_supports": packet_supports,
    }


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


def _packet_generators(reference_occupied, masks, n: int, prefix: str):
    """One configuration generator per packet determinant, from this reference."""
    reference_program = determinant_program(n, reference_occupied)
    generators = []
    for position, mask in enumerate(masks):
        occupied = _occupied_from_mask(int(mask), n)
        if tuple(sorted(occupied)) == tuple(sorted(reference_occupied)):
            continue                      # the identity direction, already seeded
        generators.append(configuration_generator(
            reference_program, determinant_program(n, occupied),
            label=f"{prefix}[{position}]"))
    return generators


def word_cost_preflight(model, backend, reference_entry, retained_masks,
                        packet_masks, coefficients, *,
                        spin_ordering="interleaved",
                        numeric_certificate_max_qubits: int = 8) -> dict:
    """``Delta W(K)``: what one packet direction adds to the word universe.

    Two numbers, because they answer different questions:

    ``pair_support_bound``
        The no-cancellation union over the ``K^2 + K M`` products
        ``A_k' A_l`` and ``A_k' H A_l``.  This is what a cost model that treats
        the packet as a sum of independent directions would predict.

    ``coefficient_aware``
        The true incremental universe of the single compiled generator
        ``A_new = sum_k c_k A_k``, after the coefficients have been applied and
        cancellations have happened.  This is what a measurement plan pays.

    The gap between them is the whole implementability question, which is why
    the preflight runs *before* any accuracy sweep.
    """
    n = model.n
    sector_basis = backend.basis
    reference_occupied = reference_entry["occupied"]
    rho = backend.density_from_program(determinant_program(n, reference_occupied)) \
        if hasattr(backend, "density_from_program") else None
    if rho is None:
        from clifford_qc.backends import ExactMVBackend
        rho = ExactMVBackend().state(determinant_program(n, reference_occupied), ())

    retained = ([identity_generator(n)]
                + _packet_generators(reference_occupied, retained_masks, n, "ret"))
    packet_parts = _packet_generators(reference_occupied, packet_masks, n, "pkt")
    if not packet_parts:
        return {"K": int(len(packet_masks)), "skipped": "packet is the reference"}

    # ``word_set`` reports the universe of pairs that have actually been
    # built, and the bank is lazy.  Forcing the full (S, H) assembly is what
    # makes ``W(B)`` the measurement plan for the retained bank rather than an
    # empty set -- and therefore what makes Delta W a genuine increment.
    baseline = MatrixElementBank(rho, model.hamiltonian, retained)
    baseline.matrices()
    baseline_words = baseline.word_set()

    # (a) no-cancellation bound over the K^2 + K M products.
    with_parts = MatrixElementBank(rho, model.hamiltonian, retained)
    with_parts.matrices()
    part_ids = [with_parts.add(generator) for generator in packet_parts]
    retained_ids = [with_parts.add(generator) for generator in retained]
    products = 0
    bound_words: set[int] = set()
    for i in part_ids:
        for j in part_ids + retained_ids:
            bound_words.update(with_parts.overlap_operator(i, j).terms)
            bound_words.update(with_parts.element_operator(i, j).terms)
            products += 1

    # (b) coefficient-aware universe of the single compiled generator.
    combined = None
    weights = np.asarray(coefficients, dtype=complex).reshape(-1)
    if weights.size != len(packet_parts):
        weights = weights[:len(packet_parts)]
    for weight, generator in zip(weights, packet_parts):
        term = generator.mv * complex(weight)
        combined = term if combined is None else combined + term
    from clifford_qc.subspace.generator_core import Generator

    packet = Generator(f"packet[K={len(packet_parts)}]", combined)
    exact_bank = MatrixElementBank(rho, model.hamiltonian, retained)
    exact_bank.add(packet)
    exact_bank.matrices()
    exact_words = exact_bank.word_set()

    # A configuration generator ``A = V R'`` is an X-string, so it does *not*
    # commute with N and Sz: operator-global leakage is nonzero by construction
    # and is reported, not asserted on.  The certificate that actually binds is
    # reference-conditioned -- what ``A|psi>`` does, not what ``A`` does.
    #
    # For a configuration packet that certificate is *structural* and exact:
    # every part carries the reference onto one sector basis determinant, so
    # ``A_new|psi>`` is a combination of sector basis states and cannot leave
    # the sector.  Checking the packet's determinants against the sector basis
    # is therefore a proof, not an estimate, and it costs ``O(K)``.
    #
    # ``subspace_sector_certificate`` states the same thing numerically, but it
    # builds the explicit sector projector -- one term per pattern times ``2^n``
    # word products -- which is a small-``n`` object by construction and is
    # hopeless by 12 qubits.  So it runs as a cross-check of the structural
    # argument where it is affordable, and is reported as skipped where it is
    # not.  The invariant reads the structural result either way.
    leakage = sector_leakage(packet, spin_ordering=spin_ordering)
    sector_masks = set(int(mask) for mask in sector_basis)
    stray = sorted(int(mask) for mask in packet_masks
                   if int(mask) not in sector_masks)
    structural = {"packet_determinants": int(len(packet_masks)),
                  "outside_sector": stray,
                  "in_sector": not stray,
                  "argument": "every packet part reaches one sector basis "
                              "determinant, so the packet cannot leave the "
                              "sector"}
    if n <= numeric_certificate_max_qubits:
        certificate = {
            key: (value if isinstance(value, (str, bool)) else float(value))
            for key, value in subspace_sector_certificate(
                rho, [packet], spin_ordering=spin_ordering).items()}
    else:
        certificate = {
            "skipped": f"n={n} exceeds numeric_certificate_max_qubits="
                       f"{numeric_certificate_max_qubits}; the explicit sector "
                       "projector is a small-n object and the structural "
                       "certificate already decides this case"}
    return {
        "K": int(len(packet_parts)),
        "products": int(products),
        "products_formula": int(len(packet_parts) ** 2
                                + len(packet_parts) * len(retained)),
        "baseline_W": int(len(baseline_words)),
        "pair_support_bound": int(len(set(bound_words) - baseline_words)),
        "coefficient_aware": int(len(exact_words - baseline_words)),
        "packet_generator_support": int(packet.support()),
        "operator_global_leakage": {key: float(value)
                                    for key, value in leakage.items()},
        "operator_global_leakage_note": (
            "nonzero by construction: a configuration generator is an X-string "
            "and does not commute with N or Sz. Not a defect, and not the "
            "certificate the invariant checks"),
        "structural_sector_certificate": structural,
        "reference_conditioned_certificate": certificate,
    }


DEFAULT_GROUPING_WORD_LIMIT = 20000


def grouping_contexts(model, backend, reference_entry, retained_masks,
                      packet_masks, coefficients, *,
                      word_limit: int = DEFAULT_GROUPING_WORD_LIMIT):
    """QWC group count for a packet that survived the preflight.

    Deliberately separate from the preflight: grouping is the expensive count,
    and paying it for a ``K`` whose word universe already failed would be work
    spent on a configuration that is not going to be run.
    """
    n = model.n
    reference_occupied = reference_entry["occupied"]
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace.generator_core import Generator

    rho = ExactMVBackend().state(determinant_program(n, reference_occupied), ())
    retained = ([identity_generator(n)]
                + _packet_generators(reference_occupied, retained_masks, n, "ret"))
    parts = _packet_generators(reference_occupied, packet_masks, n, "pkt")
    if not parts:
        return None
    combined = None
    weights = np.asarray(coefficients, dtype=complex).reshape(-1)[:len(parts)]
    for weight, generator in zip(weights, parts):
        term = generator.mv * complex(weight)
        combined = term if combined is None else combined + term
    bank = MatrixElementBank(rho, model.hamiltonian,
                             retained + [Generator("packet", combined)])
    bank.matrices()
    universe = len(bank.word_set())
    # The greedy QWC partition is quadratic in the universe size, which is why
    # the bank keeps it off the default resource report.  Above the limit this
    # returns the reason instead of the count: a number that took an hour to
    # produce for one packet configuration is not worth more than saying so.
    if universe > word_limit:
        return {"skipped": f"word universe {universe} exceeds the grouping "
                           f"limit {word_limit}; the QWC partition is "
                           "quadratic in the universe size",
                "word_universe": int(universe)}
    return int(bank.qwc_group_count())


# ------------------------------------------------------------------ row builder


def _blank_row(method: str, category: str, seed: int) -> dict:
    row = {field: None for field in REQUIRED_FIELDS}
    row.update({"method": method, "evidence_category": category, "seed": seed,
                "implementable": False, "preconditioner_category": "none",
                "shift_rule": "none", "selection_work": 0})
    return row


def _finish_row(row: dict, run: dict, exact_energy: float, *,
                matvecs: int, wall: float) -> dict:
    energy = float(run["energy"])
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


def run_system(name: str, *, seed: int = 0, total_directions: int = 7,
               shift_grid=DEFAULT_SHIFT_GRID, packet_K=DEFAULT_PACKET_K,
               word_budget_ratio: float = DEFAULT_WORD_BUDGET_RATIO,
               reference_blocks=(1, 2, 4)) -> dict:
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
    sweep_start = counting.matvecs
    sweep_wall = time.perf_counter()
    sweep = select_shift(counting, unit(primary["position"]), total_directions,
                         diagonal, shift_grid)
    sweep_matvecs = counting.matvecs - sweep_start
    sweep_seconds = time.perf_counter() - sweep_wall

    row = _blank_row("davidson", "exact_simulation", seed)
    row.update({
        "reference_policy": "model_reference",
        "reference_identity": primary["identity"],
        "reference_block_size": 1, "declared_total_directions": total_directions,
        "preconditioner_category": "classical_preconditioner",
        "shift_rule": "(D - E_m + mu)^-1, mu from the declared grid by "
                      "min projected Ritz energy",
        "mu": sweep["selected_mu"],
        "selection_work": sweep["selection_work"]})
    davidson_row, davidson_run = timed(
        row, lambda: expand(counting, unit(primary["position"]), total_directions,
                            diagonal=diagonal, mu=sweep["selected_mu"]))
    davidson_row["matvecs"] = int(davidson_row["matvecs"] + sweep_matvecs)
    davidson_row["wall_seconds"] = float(davidson_row["wall_seconds"]
                                         + sweep_seconds)
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
        local = select_shift(counting, unit(entry["position"]), total_directions,
                             diagonal, shift_grid)
        base["mu"] = local["selected_mu"]
        base["selection_work"] = local["selection_work"]
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
                local = select_shift(counting, vector, target_M, diagonal,
                                     shift_grid)
                chosen = local["selected_mu"]
                base["selection_work"] = local["selection_work"]
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
    """Preflight every ``K``, then run and group only the survivors."""
    dimension = int(backend.dimension)
    vector = np.zeros(dimension, dtype=complex)
    vector[primary["position"]] = 1.0

    # One uncompressed Davidson step supplies the correction whose top-K
    # truncation the preflight prices.  Pricing a packet needs the coefficients
    # the method would actually produce, not a placeholder.
    seed_run = expand(counting, vector, 2, diagonal=diagonal, mu=mu)
    probe = _residual(counting, seed_run["state"], seed_run["energy"])
    probe = probe / (diagonal - seed_run["energy"] + mu)

    # A bank matched to the expansion: the reference plus the determinants
    # carrying the most weight in the current Ritz state.  This is the object a
    # determinant A-CASE arm at this M would actually have to measure, so its
    # word universe is the right thing to price the packet against.
    ranked = np.argsort(-np.abs(seed_run["state"]))[:total_directions]
    retained_masks = [int(backend.basis[primary["position"]])]
    retained_masks += [int(backend.basis[i]) for i in ranked
                       if int(backend.basis[i]) not in retained_masks]
    retained_masks = retained_masks[:total_directions]

    anchor = committed_acase_word_cost(system)
    preflight, accepted, rejected = [], [], []
    packet_masks: dict[str, list[int]] = {}
    for K in packet_K:
        if K > dimension:
            continue
        packet, support = _top_k_packet(probe, K)
        masks = [int(backend.basis[i]) for i in support]
        packet_masks[str(int(K))] = masks
        try:
            entry = word_cost_preflight(model, backend, primary, retained_masks,
                                        masks, packet[support])
        except (ValueError, MemoryError) as error:      # pragma: no cover
            entry = {"K": int(K), "failed": str(error)}
        entry["K"] = int(K)
        if "coefficient_aware" in entry:
            delta = entry["coefficient_aware"]
            entry["cancellation_factor"] = float(
                entry["pair_support_bound"] / max(1, delta))
            entry["incremental_ratio"] = float(delta
                                               / max(1, entry["baseline_W"]))
            if anchor.get("available"):
                per_direction = anchor["words_per_direction"]
                entry["acase_words_per_direction"] = per_direction
                entry["cost_vs_acase_direction"] = float(delta / per_direction)
                entry["accepted"] = bool(
                    delta <= word_budget_ratio * per_direction)
                entry["gate_basis"] = "committed A-CASE words per direction"
            else:
                entry["accepted"] = bool(
                    entry["incremental_ratio"] <= word_budget_ratio)
                entry["gate_basis"] = "identity-baseline ratio (fallback)"
            (accepted if entry["accepted"] else rejected).append(int(K))
        preflight.append(entry)

    runs = []
    for K in accepted:
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
        entry = next(item for item in preflight if item["K"] == K)
        filled["W"] = entry.get("coefficient_aware")
        if run["packet_supports"]:
            masks = [int(backend.basis[i]) for i in run["packet_supports"][0]]
            coefficients = probe[run["packet_supports"][0]]
            filled["grouping_contexts"] = grouping_contexts(
                model, backend, primary, retained_masks, masks, coefficients)
        runs.append(filled)

    # The full-K control: keeping every component must reproduce the
    # uncompressed Davidson direction exactly.
    full = expand(counting, vector, total_directions, diagonal=diagonal, mu=mu,
                  packet_K=dimension)
    uncompressed = expand(counting, vector, total_directions, diagonal=diagonal,
                          mu=mu)
    return {
        "preflight": preflight,
        "packet_masks": packet_masks,
        "accepted_K": accepted,
        "rejected_K": rejected,
        "word_budget_ratio": float(word_budget_ratio),
        "anchor": anchor,
        "gate_rule": ("a K is run only if its coefficient-aware incremental "
                      "word universe is within word_budget_ratio times the "
                      "committed A-CASE cost of one retained direction on this "
                      "system; QWC grouping is computed for survivors only"),
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
    orthogonal, krylov = arms.get("orthogonal_residual"), arms.get("power_krylov")
    if orthogonal and krylov:
        if orthogonal["M"] != krylov["M"]:
            problems.append("orthogonal_residual and power_krylov are not "
                            "at matched M")
        # Same span, so the two Ritz values may differ only by what the raw
        # monomial basis loses to its own conditioning.  Asserting exact
        # equality would fail precisely where the orthogonal arm is doing its
        # job; asserting nothing would make the regression arm vacuous.
        signed = orthogonal["energy"] - krylov["energy"]
        bound = max(1e-8, float(np.finfo(float).eps)
                    * float(krylov["kappa_S"]) * max(1.0, abs(krylov["energy"])))
        record["krylov_agreement"] = {
            "signed_gap": float(signed),
            "gap": float(abs(signed)),
            "conditioning_bound": float(bound),
            "kappa_S_krylov": float(krylov["kappa_S"]),
            "note": ("the orthogonal arm spans the same Krylov space at S = I. "
                     "In exact arithmetic it is the sharper Ritz value; in "
                     "floating point the monomial arm's own conditioning can "
                     "put it a hair lower, so the tolerance both directions "
                     "are held to is the conditioning bound, not a constant")}
        if abs(signed) > bound:
            problems.append(
                f"orthogonal_residual and power_krylov disagree by "
                f"{abs(signed):.3g}, beyond the conditioning bound "
                f"{bound:.3g}; they are supposed to span the same Krylov space")

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
    for entry in record["packet_program"]["preflight"]:
        structural = entry.get("structural_sector_certificate")
        if structural is not None and not structural["in_sector"]:
            problems.append(
                f"packet K={entry['K']} reaches determinants outside the "
                f"sector: {structural['outside_sector']}")
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
