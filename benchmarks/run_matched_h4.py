"""Matched-contract cost comparison on the frozen H4 FCIDUMP.

The manuscript's ladder compares arms on energy and on a resource ledger, but
the ledger is not matched: arms differ in what they hold fixed, and the
warm-start result is explicitly a fixed-A-CASE-budget statement rather than a
matched total-cost claim.  This benchmark fixes the contract and prices every
arm in the same currencies.

Held fixed across every arm:

* the frozen ``h4_sto3g_r0.9.FCIDUMP`` (digest checked against provenance);
* the ``(N=4, S_z=0)`` sector and its dense reference energy;
* the Hartree-Fock determinant as the reference state;
* the operator pool -- the 26 symmetry-preserving determinant excitations, and
  the 160 odd-Y Pauli words those excitations decompose into;
* the budget: eight additions, i.e. ``M=9`` for subspace arms and eight rotors
  for ADAPT-VQE. ADAPT-GCIM adds two states per iteration, so it is reported at
  four iterations (``M=8``, nearest size match) and eight iterations
  (``M=16``, iteration match);
* the convergence rule: run the budget out, no early stop, exact arithmetic.

The one thing that cannot be held fixed is *granularity*.  ADAPT-VQE consumes
single Pauli words; A-CASE's default candidates are whole determinant
excitations.  Those are the same operator content at two resolutions, so both
are run, and the difference between them turns out to be the most interesting
column in the table.

Costs are reported in four currencies rather than collapsed into a score,
because they are not interchangeable on hardware:

* ``state_preparations``   -- distinct states the arm must prepare.
  Fixed-reference operator subspaces need one; ADAPT-GCIM needs its ``M``
  generating-function states; ADAPT-VQE needs a fresh state per selection step
  and optimizer evaluation.
* ``ansatz_rotors``        -- rotors in the deepest prepared state, a depth
  proxy. ADAPT-GCIM also records the sum over all distinct basis circuits.
* ``selection_evaluations`` -- candidate scorings summed over steps.
* ``selection_words`` / ``final_words`` -- distinct Pauli words, with their QWC
  group counts, for scoring candidates and for the final object.

ADAPT-GCIM has a genuinely different final measurement primitive: off-diagonal
transition matrix elements between separately prepared generating-function
states.  Its ``final_words`` is therefore not forced into A-CASE's
single-reference word universe.  The record reports the unique upper-triangle
Hamiltonian pairs and off-diagonal overlap pairs separately.

``sector_weight`` is reported for every subspace arm because one arm earns it:
the word-granularity pool has generators whose *operator* leakage out of
``(N, S_z)`` is maximal, and which the default rejection rule therefore
discards, yet whose Ritz vector lands exactly in the sector. The rejection test
asks whether a generator could take *any* reference out of the sector; on a
determinant reference the relevant question is narrower.

    python benchmarks/run_matched_h4.py
    python benchmarks/run_matched_h4.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from clifford_qc.algorithms import run_adapt
from clifford_qc.algorithms.pools import PoolOperator, is_odd_y
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.bank import CommutatorBank
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.measurement.grouping import qwc_groups
from clifford_qc.models import fcidump_model
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    adapt_warm_start,
    dense_basis,
    determinant_excitations,
    identity_generator,
    krylov_response,
    occupied_spin_orbitals,
    pauli_orbit,
    run_adapt_gcim,
    run_acase,
    solve_subspace,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "reference_results" / "matched_h4.json"
FCIDUMP = ROOT / "data" / "h4_sto3g_r0.9.FCIDUMP"
PROVENANCE = ROOT / "data" / "h4_sto3g_r0.9.provenance.json"

BUDGET = 8
CHEMICAL_ACCURACY_HARTREE = 1.6e-3
WARM_OPERATORS = 2


def word_pool(model, candidates) -> list[PoolOperator]:
    """The odd-Y words the determinant excitations decompose into."""
    pool: dict[int, PoolOperator] = {}
    for generator in candidates:
        for code in sorted(generator.mv.terms):
            word = PauliWord(model.n, code)
            if code and is_odd_y(word) and code not in pool:
                pool[code] = PoolOperator(word.label, word)
    return list(pool.values())


# qwc_groups is a greedy colouring over the QWC-incompatibility graph, so it
# builds O(W^2) pairs. That is affordable for the word sets this comparison
# turns on and ruinous for the 15k-word selection caches, where the group count
# is a derived convenience rather than a quantity any claim rests on.
MAX_GROUPED_WORDS = 5_000


def _groups(n: int, codes) -> int | None:
    codes = list(codes)
    if len(codes) > MAX_GROUPED_WORDS:
        return None
    return len(qwc_groups([PauliWord(n, code) for code in codes]))


def selection_width(model, pool) -> tuple[int, int]:
    """Words and QWC groups ADAPT must measure to score its pool.

    The selection observable is ``G_j = -i/2 [H, P_j]``; the bank builds each
    row from the anticommuting terms of H, so the union of the rows is exactly
    what a selection round has to read.
    """
    bank = CommutatorBank(model.hamiltonian, [op.word for op in pool],
                          [op.label for op in pool])
    words: set[int] = set()
    for row in bank.coeffs:
        words |= set(row)
    return len(words), _groups(model.n, words)


def _state_sector_weight(model, psi) -> float:
    """Weight of a normalized dense state inside the declared sector."""
    norm = np.linalg.norm(psi)
    if norm <= 0.0:
        return float("nan")
    probability = np.abs(psi / norm) ** 2
    index = np.arange(2 ** model.n)
    electrons = np.array([bin(i).count("1") for i in index])
    spin = np.array([
        sum(0.5 if (i >> q) & 1 else 0.0 for q in range(0, model.n, 2))
        - sum(0.5 if (i >> q) & 1 else 0.0 for q in range(1, model.n, 2))
        for i in index])
    keep = (electrons == model.metadata["n_electrons"]) & (np.abs(spin) < 1e-12)
    return float(probability[keep].sum())


def sector_weight(model, rho, result, generators) -> float:
    """Weight of the Ritz vector inside ``(N=4, S_z=0)``.

    Dense, and only affordable because this rung is eight qubits. It is the
    check that decides whether a symmetry-breaking *generator* produced a
    symmetry-breaking *state*, which is the whole question the operator-level
    leakage test cannot answer.
    """
    # The adaptive path carries bank indices; the fixed path does not, so fall
    # back to resolving the retained set by label.
    if result.indices:
        retained = [generators[i] for i in result.indices]
    else:
        by_label = {g.label: g for g in generators}
        retained = [by_label[label] for label in result.basis_labels]
    basis = dense_basis(rho, retained)
    psi = basis @ np.asarray(result.coefficients)[:, 0]
    return _state_sector_weight(model, psi)


def _subspace_row(name, pool_kind, result, exact_energy, *,
                  model, rho, generators, selection_evaluations,
                  selection_words, selection_groups, final_groups=None,
                  state_preparations=1, ansatz_rotors=0,
                  optimizer_evaluations=0, notes=""):
    energy = result.ground_energy if hasattr(result, "ground_energy") else result.energies[0]
    error = energy - exact_energy
    final_words = result.resources.get("word_universe")
    return {
        "arm": name,
        "pool": pool_kind,
        "basis_size": len(result.basis_labels),
        "labels": list(result.basis_labels),
        "ground_energy": energy,
        "error_hartree": error,
        "error_millihartree": error * 1e3,
        "chemical_accuracy": bool(abs(error) < CHEMICAL_ACCURACY_HARTREE),
        "condition_number": result.condition_number,
        "effective_rank": result.effective_rank,
        "sector_weight": sector_weight(model, rho, result, generators),
        "state_preparations": state_preparations,
        "ansatz_rotors": ansatz_rotors,
        "optimizer_evaluations": optimizer_evaluations,
        "selection_evaluations": selection_evaluations,
        "selection_words": selection_words,
        "selection_qwc_groups": selection_groups,
        "final_words": final_words,
        "final_qwc_groups": final_groups,
        "notes": notes,
    }


def build_record() -> dict:
    import run_acase_ladder as ladder

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    model = fcidump_model(FCIDUMP, name=provenance["name"])
    if model.metadata["source_sha256"] != provenance["fcidump_sha256"]:
        raise ValueError("FCIDUMP digest disagrees with its provenance record")

    sector = SectorStatevectorBackend(
        model.n, model.metadata["n_electrons"], model.metadata["sz"])
    exact_energy = float(
        sector.ground_state(model.hamiltonian, k=4, method="dense")[0][0])
    if abs(exact_energy - provenance["reference_energies"]["fci"]) > 1e-10:
        raise ValueError("sector-exact energy disagrees with external FCI provenance")

    rho = ExactMVBackend().state(model.reference, ())
    occupied = occupied_spin_orbitals(model)
    determinants = determinant_excitations(model.n, occupied, max_rank=2)
    pool = word_pool(model, determinants)
    words = pauli_orbit([op.word for op in pool])
    sel_words, sel_groups = selection_width(model, pool)
    # The determinant-resolution commutator union is the same physical
    # gradient observable as the decomposed word pool, but keeping this direct
    # construction makes the ADAPT-GCIM resource row independent of that fact.
    H_mv = model.hamiltonian.to_mv()
    gcim_selection_codes: set[int] = set()
    for generator in determinants:
        commutator = H_mv * generator.mv - generator.mv * H_mv
        gcim_selection_codes.update(commutator.terms)
    gcim_sel_words = len(gcim_selection_codes)
    gcim_sel_groups = _groups(model.n, gcim_selection_codes)
    hamiltonian_groups = _groups(model.n, model.hamiltonian.terms)

    identity = identity_generator(model.n)
    rows: list[dict] = []

    def acase(name, candidates, pool_kind, *, leakage_tol, notes="",
              reference=None, prelude=None):
        started = time.perf_counter()
        kwargs = {"max_size": BUDGET, "exact_ground_energy": exact_energy}
        if leakage_tol is not None:
            kwargs["leakage_tol"] = leakage_tol
        result = run_acase(reference if reference is not None else rho,
                           model.hamiltonian, candidates, **kwargs)
        scored = sum(record.candidates_scored for record in result.records)
        # The cache universe, not the retained subspace's: scoring a candidate
        # that is then rejected still costs its words. The manuscript's W is
        # the retained figure, which is the smaller of the two.
        bank = result.bank
        selection_codes = bank.word_set(range(len(bank._generators)))
        final_codes = bank.word_set(result.result.indices)
        row = _subspace_row(
            name, pool_kind, result.result, exact_energy, model=model,
            rho=reference if reference is not None else rho,
            generators=bank._generators,
            selection_evaluations=scored,
            selection_words=len(selection_codes),
            selection_groups=_groups(model.n, selection_codes),
            final_groups=_groups(model.n, final_codes),
            state_preparations=1 if prelude is None else prelude["preparations"] + 1,
            ansatz_rotors=0 if prelude is None else prelude["rotors"],
            optimizer_evaluations=0 if prelude is None else prelude["optimizer_evaluations"],
            notes=notes)
        if prelude is not None:
            row["selection_evaluations"] += prelude["selection_evaluations"]
            row["prelude"] = prelude
        print(f"  {name}: {time.perf_counter() - started:.1f}s", flush=True)
        rows.append(row)
        return result

    # --- fixed subspace arms, same budget, no adaptivity ------------------
    for name, family, kind in (
            ("QSE", [identity, *pauli_orbit(
                [op.word for op in ladder.fixed_slice(pool, BUDGET, "stride")])],
             "odd-Y words"),
            ("Krylov", [identity, *krylov_response(model.hamiltonian, BUDGET)],
             "Hamiltonian powers"),
            ("generator coordinate",
             [identity, *ladder.fixed_slice(determinants, BUDGET, "stride")],
             "determinant excitations")):
        started = time.perf_counter()
        tracked = name != "Krylov"
        final_groups = None
        if tracked:
            # Through the bank so the width and its QWC grouping are the same
            # quantities the adaptive arms report, rather than lookalikes.
            bank = MatrixElementBank(rho, model.hamiltonian, family)
            result = bank.solve()
            final_groups = _groups(
                model.n, bank.word_set(range(len(family))))
        else:
            result = solve_subspace(rho, model.hamiltonian, family,
                                    track_support=False)
        row = _subspace_row(
            name, kind, result, exact_energy, model=model, rho=rho,
            generators=family, selection_evaluations=0,
            selection_words=0, selection_groups=0, final_groups=final_groups,
            notes="fixed basis: committed before any measurement")
        if not tracked:
            row["final_words"] = None
            row["notes"] += ("; deep Krylov elements are not tractable through "
                             "the tracked route -- see run_krylov_width.py")
        print(f"  {name}: {time.perf_counter() - started:.1f}s", flush=True)
        rows.append(row)

    # --- exact published ADAPT-GCIM rule on the matched local pool --------
    def adapt_gcim(name, iterations, notes):
        started = time.perf_counter()
        gcim = run_adapt_gcim(
            rho, model.hamiltonian, determinants,
            max_iterations=iterations)
        result = gcim.result
        energy = result.ground_energy
        error = energy - exact_energy
        ritz_state = gcim.basis @ np.asarray(result.coefficients)[:, 0]
        rows.append({
            "arm": name,
            "pool": "determinant excitations",
            "basis_size": len(result.basis_labels),
            "labels": list(result.basis_labels),
            "selected_labels": list(gcim.selected_labels),
            "trajectory": [{
                "iteration": record.iteration,
                "selected_index": record.selected_index,
                "selected_label": record.selected_label,
                "gradient": record.gradient,
                "active_candidates": record.active_candidates,
                "basis_size": record.basis_size,
                "basis_rotor_depths": list(record.basis_rotor_depths),
                "ground_energy": record.ground_energy,
                "effective_rank": record.effective_rank,
                "condition_number": record.condition_number,
            } for record in gcim.records],
            "iterations": iterations,
            "theta": gcim.theta,
            "overlap_threshold": result.resources["overlap_threshold"],
            "condition_cap": None,
            "ground_energy": energy,
            "error_hartree": error,
            "error_millihartree": error * 1e3,
            "chemical_accuracy": bool(
                abs(error) < CHEMICAL_ACCURACY_HARTREE),
            "condition_number": result.condition_number,
            "effective_rank": result.effective_rank,
            "sector_weight": _state_sector_weight(model, ritz_state),
            # Distinct generating-function state circuits. The cumulative
            # surrogate states used for selection are already members of this
            # final set, so they are not double-counted.
            "state_preparations": len(result.basis_labels),
            # The deepest basis circuit; total rotor applications across all
            # distinct basis circuits is reported in its own field below.
            "ansatz_rotors": max(gcim.basis_rotor_depths),
            "basis_state_rotor_applications": sum(
                gcim.basis_rotor_depths),
            "optimizer_evaluations": 0,
            "selection_evaluations": sum(
                record.active_candidates for record in gcim.records),
            "selection_words": gcim_sel_words,
            "selection_qwc_groups": gcim_sel_groups,
            # No single-reference W exists for transition measurements between
            # different prepared states.
            "final_words": None,
            "final_qwc_groups": None,
            "hamiltonian_matrix_pairs": (
                len(result.basis_labels) *
                (len(result.basis_labels) + 1) // 2),
            "overlap_offdiagonal_pairs": (
                len(result.basis_labels) *
                (len(result.basis_labels) - 1) // 2),
            "measurement_model": (
                "off-diagonal H and S transition matrix elements between "
                "separately prepared generating-function states"),
            "notes": notes,
        })
        print(f"  {name}: {time.perf_counter() - started:.1f}s", flush=True)

    adapt_gcim(
        "ADAPT-GCIM (4 iter., M=8)", 4,
        "published fixed theta=pi/4 rule; nearest basis-size match to M=9; "
        "same local determinant-excitation pool")
    adapt_gcim(
        "ADAPT-GCIM (8 iter., M=16)", 8,
        "published fixed theta=pi/4 rule; iteration-matched to the "
        "eight-addition contract; same local determinant-excitation pool")

    # --- ADAPT-VQE on the word pool ---------------------------------------
    started = time.perf_counter()
    adapt = run_adapt(model, pool, max_operators=BUDGET,
                      compute_exact_reference=False)
    adapt_error = adapt.energy - exact_energy
    adapt_scored = sum(record.active_candidates for record in adapt.records)
    adapt_preparations = len(adapt.records) + adapt.optimizer_evaluations
    rows.append({
        "arm": "ADAPT-VQE",
        "pool": "odd-Y words",
        "basis_size": None,
        "labels": list(adapt.labels),
        "ground_energy": adapt.energy,
        "error_hartree": adapt_error,
        "error_millihartree": adapt_error * 1e3,
        "chemical_accuracy": bool(abs(adapt_error) < CHEMICAL_ACCURACY_HARTREE),
        "condition_number": None,
        "effective_rank": None,
        "sector_weight": None,
        "state_preparations": adapt_preparations,
        "ansatz_rotors": len(adapt.labels),
        "optimizer_evaluations": adapt.optimizer_evaluations,
        "selection_evaluations": adapt_scored,
        "selection_words": sel_words,
        "selection_qwc_groups": sel_groups,
        "final_words": len(model.hamiltonian.terms),
        "final_qwc_groups": hamiltonian_groups,
        "notes": ("no overlap matrix: a product ansatz has no kappa_S, and the "
                  "prepared state is a sector eigenstate by construction"),
    })

    # --- A-CASE at both granularities -------------------------------------
    acase("A-CASE (determinant)", determinants, "determinant excitations",
          leakage_tol=1e-10,
          notes="the manuscript's default arm")
    acase("A-CASE (word, leakage rejected)", words, "odd-Y words",
          leakage_tol=1e-10,
          notes=("every candidate rejected: operator leakage out of (N, S_z) "
                 "is sqrt(2) for every odd-Y word"))
    acase("A-CASE (word)", words, "odd-Y words", leakage_tol=None,
          notes=("same pool ADAPT consumes, leakage rejection disabled; the "
                 "Ritz vector is still exactly in sector"))

    # --- warm-started A-CASE, with the prelude priced ---------------------
    warm_rho, warm_adapt = adapt_warm_start(
        model, pool, max_operators=WARM_OPERATORS, compute_exact_reference=False)
    prelude = {
        "stage": f"ADAPT-VQE, {len(warm_adapt.labels)} operators",
        "energy": warm_adapt.energy,
        "error_millihartree": (warm_adapt.energy - exact_energy) * 1e3,
        "rotors": len(warm_adapt.labels),
        "preparations": len(warm_adapt.records) + warm_adapt.optimizer_evaluations,
        "optimizer_evaluations": warm_adapt.optimizer_evaluations,
        "selection_evaluations": sum(r.active_candidates for r in warm_adapt.records),
        "selection_words": sel_words,
        "selection_qwc_groups": sel_groups,
    }
    acase("A-CASE (determinant, ADAPT warm start)", determinants,
          "determinant excitations", leakage_tol=1e-10, reference=warm_rho,
          prelude=prelude,
          notes=("total cost includes the ADAPT prelude that prepared rho: "
                 "its rotors, optimizer evaluations, and pool scorings"))

    return {
        "schema": "clifford_qc.matched_h4.v2",
        "evidence": {
            "energies": "exact",
            "resource_counts": "exact",
            "quantum_advantage_claim": False,
        },
        "matched": {
            "input": str(FCIDUMP.relative_to(ROOT.parent)),
            "sha256": model.metadata["source_sha256"],
            "sector": {"n_electrons": model.metadata["n_electrons"],
                       "sz": model.metadata["sz"], "dimension": 36},
            "reference_state": "Hartree-Fock determinant",
            "budget": BUDGET,
            "adapt_gcim_basis_rule": (
                "M=2k: k=4 gives the nearest size match M=8; k=8 gives "
                "the iteration match M=16"),
            "convergence": "fixed budget, no early stop, exact arithmetic",
            "pool_determinant_excitations": len(determinants),
            "pool_odd_y_words": len(pool),
            "unmatched": (
                "generator granularity: ADAPT-VQE consumes single Pauli "
                "words, while A-CASE's default and ADAPT-GCIM candidates are "
                "whole determinant excitations. ADAPT-GCIM also measures "
                "off-diagonal transition elements rather than one fixed-"
                "reference word bank; both costs are reported without an "
                "assumed exchange rate."),
        },
        "reference_energy": exact_energy,
        "hamiltonian_words": len(model.hamiltonian.terms),
        "hamiltonian_qwc_groups": hamiltonian_groups,
        "adapt_selection_words": sel_words,
        "adapt_selection_qwc_groups": sel_groups,
        "column_notes": {
            "selection_words": (
                "distinct Pauli words the arm must read to score its "
                "candidates: for A-CASE the whole element cache, "
                "including rows for candidates that were then rejected, "
                "which is strictly larger than the retained final_words "
                "the manuscript ledger reports"),
            "state_preparations": (
                "distinct states prepared. Circuit executions are this "
                "times the QWC group count times shots per group; the "
                "shot factor is outside this exact-arithmetic record. For "
                "ADAPT-GCIM this is the number of distinct generating-"
                "function basis circuits; its cumulative selector states are "
                "already members of that set."),
            "adapt_gcim_matrix_pairs": (
                "unique upper-triangle Hamiltonian pairs and off-diagonal "
                "overlap pairs. These transition measurements are not a "
                "single-reference Pauli-word universe and are therefore not "
                "reported as final_words."),
        },
        "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    record = stamp_record(build_record())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for row in record["rows"]:
        error = row["error_millihartree"]
        print(f"{row['arm']:38s} err={error:10.4f} mHa  "
              f"prep={row['state_preparations']:>6}  "
              f"sel={row['selection_evaluations']:>6}  "
              f"W={row['final_words']}", flush=True)
    print(args.out)


if __name__ == "__main__":
    main()
