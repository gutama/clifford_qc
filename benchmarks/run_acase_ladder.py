"""A-CASE validation ladder: systems x methods -> JSONL (ACASE_RESEARCH_PLAN.md §7).

One output line per (system, method) run, carrying the energy *and* the §6
resource metrics -- basis size, word universe, element supports, retained rank,
conditioning -- plus shots, circuits, and abstentions where a method spends
them. The comparison set is the one §7 names: the reference determinant, exact
diagonalization in the reference's symmetry sector, plain QSE, fixed Krylov, a
generator-coordinate-style fixed subspace, ADAPT-VQE (exact and finite-shot),
and A-CASE (exact and finite-shot certified).

Every row also carries an ``evidence`` label, because the methods do not all
produce the same kind of number: ``exact`` for noiseless arithmetic,
``asymptotic`` for a delta-method interval, ``finite_sample`` for a certified
growth decision, ``reference`` for the exact diagonalization a row is scored
against. Comparing a certified run's energy with an exact one on equal footing
is precisely the confusion the labels exist to prevent.

Run:
    python benchmarks/run_acase_ladder.py --config benchmarks/configs/acase_ladder.json \\
        --out benchmarks/reference_results/acase_ladder.jsonl
    python benchmarks/summarize_ladder.py benchmarks/reference_results/acase_ladder.jsonl \\
        --csv benchmarks/reference_results/acase_ladder_summary.csv \\
        > benchmarks/reference_results/acase_ladder_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

CHEMICAL_ACCURACY = 1.6e-3  # Hartree


# --------------------------------------------------------------- system builders


def _water_geometry(scale: float):
    bond, angle = 0.9584 * scale, math.radians(104.45)
    return [("O", (0.0, 0.0, 0.0)),
            ("H", (bond * math.sin(angle / 2), 0.0, bond * math.cos(angle / 2))),
            ("H", (-bond * math.sin(angle / 2), 0.0, bond * math.cos(angle / 2)))]


def build_system(spec: dict):
    """``(model, kind)`` for one ladder rung; chemistry imports stay lazy."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind in ("h2", "h4", "lih", "water"):
        from clifford_qc.models import chemistry

        if kind == "h2":
            return chemistry.h2(**spec), "molecular"
        if kind == "h4":
            return chemistry.h4_chain(**spec), "molecular"
        if kind == "lih":
            return chemistry.lih(**spec), "molecular"
        scale = spec.pop("scale", 1.0)
        active = spec.pop("active", "4e4o")
        frozen = [0, 1, 2] if active == "4e4o" else [0]
        orbitals = [3, 4, 5, 6] if active == "4e4o" else [1, 2, 3, 4, 5, 6]
        return chemistry.molecule_model(
            _water_geometry(scale), name=f"h2o_{active}(scale={scale})",
            occupied_indices=frozen, active_indices=orbitals, **spec), "molecular"
    if kind == "hubbard":
        from clifford_qc.models.lattice import hubbard

        shape = spec.pop("shape")
        return hubbard(tuple(shape) if isinstance(shape, list) else shape,
                       **spec), "fermionic_lattice"
    if kind == "kitaev":
        from clifford_qc.models.lattice import kitaev_honeycomb

        return kitaev_honeycomb(**spec), "spin_lattice"
    if kind == "tfim":
        from clifford_qc.models.spin import tfim

        return tfim(**spec), "spin_lattice"
    raise ValueError(f"unknown system type {kind!r}")


def occupied_orbitals(model, kind: str):
    from clifford_qc.subspace import occupied_spin_orbitals

    return occupied_spin_orbitals(model) if kind != "spin_lattice" else ()


def build_candidates(model, kind: str, krylov_order: int = 6):
    """A-CASE candidate generators: symmetry-preserving where a symmetry exists."""
    from clifford_qc.subspace import (commutator_response, determinant_excitations,
                                      krylov_response, pauli_orbit)

    if kind == "spin_lattice":
        words = [op.word for op in word_pool(model, kind)]
        return (pauli_orbit(words) + commutator_response(model.hamiltonian, words)
                + krylov_response(model.hamiltonian, krylov_order))
    return determinant_excitations(model.n, occupied_orbitals(model, kind))


def word_pool(model, kind: str):
    """Word-level qubit-ADAPT pool: what ADAPT-VQE and plain QSE consume.

    For a molecule this is the chemistry module's odd-Y split of the JW
    excitations. For a fermionic lattice the same construction is done here from
    the native generators -- individual words of a conserving generator, which do
    *not* conserve the symmetry (that asymmetry with the A-CASE candidates is one
    of the things the ladder is measuring).
    """
    from clifford_qc.algorithms.pools import PoolOperator, is_odd_y, local_pool, odd_y_filter

    if kind == "spin_lattice":
        return odd_y_filter(local_pool(model.n))
    if kind == "molecular":
        from clifford_qc.models.chemistry import excitation_pool

        return excitation_pool(model.n, model.metadata["n_electrons"])
    from clifford_qc.ir import PauliWord
    from clifford_qc.subspace import determinant_excitations

    pool: dict[int, PoolOperator] = {}
    for generator in determinant_excitations(model.n, occupied_orbitals(model, kind)):
        for code in sorted(generator.mv.terms):
            word = PauliWord(model.n, code)
            if code and is_odd_y(word) and code not in pool:
                pool[code] = PoolOperator(word.label, word)
    return list(pool.values())


def reference_energy(model, kind: str):
    """Exact energy in the reference's own symmetry sector.

    The sector, not the whole space: A-CASE stays where its reference lives, and
    a grand-canonical cluster's global minimum can sit at a different filling
    (see ``sparse.sector_indices``). For a spin model there is no fermionic
    sector and the whole space is the answer.
    """
    from clifford_qc.sparse import sparse_ground

    if kind == "spin_lattice":
        return float(sparse_ground(model.hamiltonian, k=1)[0][0]), "whole space"
    from clifford_qc.backends import SectorStatevectorBackend

    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"],
                                       model.metadata["sz"])
    value = backend.ground_state(model.hamiltonian, k=1)[0][0]
    return float(value), f"N={model.metadata['n_electrons']},Sz={model.metadata['sz']}"


def observable_set(model, kind: str) -> dict:
    """The material observables §7 asks the lattice rungs to report."""
    if kind == "spin_lattice" and model.metadata.get("lattice") == "honeycomb":
        from clifford_qc.models.observables import link_correlations

        return {f"link_{name}": operator
                for name, operator in sorted(link_correlations(model).items())}
    if kind == "fermionic_lattice":
        from clifford_qc.models.observables import (double_occupancy,
                                                    spin_correlation,
                                                    structure_factor,
                                                    total_spin_squared)

        return {"double_occupancy": double_occupancy(model),
                "spin_correlation_01": spin_correlation(model, 0, 1),
                "structure_factor": structure_factor(model),
                "total_spin_squared": total_spin_squared(model)}
    return {}


# ------------------------------------------------------------------- the methods

METHOD_KINDS = frozenset({
    "exact", "reference_state", "qse", "krylov", "generator_coordinate",
    "acase_exact", "acase_certified", "adapt_exact", "adapt_shot",
})


def _subspace_row(result, observables) -> dict:
    resources = result.resources
    row = {
        "basis_size": len(result.basis_labels),
        "retained_rank": result.effective_rank,
        "condition_number": result.condition_number,
        "word_universe": resources.get("word_universe"),
        "max_generator_support": resources.get("max_generator_support"),
        "max_element_support": resources.get("max_hamiltonian_element_support"),
        "labels": list(result.basis_labels),
    }
    if observables and result.bank is not None:
        row["observables"] = {name: result.expectation(operator)
                              for name, operator in observables.items()}
    return row


def run_method(name: str, spec: dict, model, kind: str, context: dict) -> dict:
    """One (system, method) run, as a record row without the shared fields."""
    from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
    from clifford_qc.subspace import (MatrixElementBank, identity_generator,
                                      krylov_response, pauli_orbit, run_acase,
                                      run_certified_acase)

    spec = dict(spec)
    method = spec.pop("kind")
    if method not in METHOD_KINDS:
        raise ValueError(f"unknown method kind {method!r}; known: "
                         f"{', '.join(sorted(METHOD_KINDS))}")
    rho = context["rho"]
    observables = context["observables"]
    seed = int(spec.pop("seed", 0))

    if method == "exact":
        return {"energy": context["reference"], "evidence": "reference",
                "sector": context["sector"],
                "observables": {name: context["exact_observables"][name]
                                for name in observables}}

    if method == "reference_state":
        energy = ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
        return {"energy": float(energy), "evidence": "exact", "basis_size": 1}

    if method in ("qse", "krylov", "generator_coordinate"):
        from clifford_qc.subspace import solve_subspace

        size = int(spec.pop("size", 8))
        widest = int(spec.pop("max_tracked_support", 512))
        selection = None
        if method == "qse":
            words = [op.word for op in context["pool"][:size]]
            generators = [identity_generator(model.n)] + pauli_orbit(words)
        elif method == "krylov":
            generators = ([identity_generator(model.n)]
                          + krylov_response(model.hamiltonian, size))
        else:
            # A generator-coordinate subspace is chosen *before* any measurement,
            # so which fixed slice of the candidate family it takes is part of the
            # method. The natural prefix is the wrong one to report: the
            # excitation family lists singles first, and on a Hartree-Fock
            # reference every single is Brillouin-dead, so a prefix of eight
            # reproduces the reference energy to machine precision and says
            # nothing about non-adaptive subspaces. An even stride is just as
            # blind and spans singles and doubles alike.
            candidates = context["candidates"]
            selection = str(spec.pop("selection", "stride"))
            if selection == "stride":
                candidates = candidates[::max(1, len(candidates) // max(size, 1))]
            elif selection != "prefix":
                raise ValueError(f"unknown generator-coordinate selection "
                                 f"{selection!r}; known: prefix, stride")
            generators = [identity_generator(model.n)] + candidates[:size]
        # The element-operator route is what makes W and S_H observable, and it
        # costs O(|A_i| |H| |A_j|) per pair. Deep Krylov generators on eight
        # qubits carry thousands of words each (H^4 on H4 already reaches 4224),
        # which puts the tracked route out of reach -- that is itself the §6
        # finding, so the row falls back to the cyclic contraction and reports
        # the support columns as unavailable rather than guessing them.
        if max(generator.support() for generator in generators) > widest:
            result = solve_subspace(rho, model.hamiltonian, generators,
                                    track_support=False)
            row = {"energy": result.ground_energy, "evidence": "exact",
                   "support_tracked": False, "selection": selection}
            row.update(_subspace_row(result, {}))
            return row
        bank = MatrixElementBank(rho, model.hamiltonian, generators)
        result = bank.solve()
        row = {"energy": result.ground_energy, "evidence": "exact",
               "support_tracked": True, "selection": selection}
        row.update(_subspace_row(result, observables))
        return row

    if method == "acase_exact":
        result = run_acase(rho, model.hamiltonian, context["candidates"],
                           exact_ground_energy=context["reference"],
                           max_size=int(spec.pop("max_size", 8)),
                           roots=int(spec.pop("roots", 1)),
                           leakage_tol=spec.pop("leakage_tol", None))
        row = {"energy": result.energy, "evidence": "exact",
               "stopped_reason": result.stopped_reason,
               "energy_history": list(result.energy_history),
               "root_energies": list(result.root_energies)}
        row.update(_subspace_row(result.result, observables))
        return row

    if method == "acase_certified":
        result = run_certified_acase(
            rho, model.hamiltonian, context["candidates"],
            FiniteShotBackend(seed=seed),
            max_size=int(spec.pop("max_size", 4)),
            construction_shots=int(spec.pop("construction_shots", 4000)),
            certification_shots=int(spec.pop("certification_shots", 4000)),
            delta=float(spec.pop("delta", 0.05)),
            threshold=float(spec.pop("threshold", 0.05)),
            leakage_tol=spec.pop("leakage_tol", None),
            exact_ground_energy=context["reference"])
        row = {"energy": result.energy, "evidence": "finite_sample",
               "stopped_reason": result.stopped_reason,
               "total_shots": result.total_shots,
               "total_circuits": result.total_circuits,
               "abstentions": result.abstentions,
               "certified_steps": sum(1 for r in result.records if r.certified),
               "energy_history": list(result.energy_history)}
        row.update(_subspace_row(result.result, observables))
        return row

    if method in ("adapt_exact", "adapt_shot"):
        from clifford_qc.algorithms import ConfidenceSelector, run_adapt
        from clifford_qc.measurement import UniformDoubling

        kwargs = {"max_operators": int(spec.pop("max_operators", 8)),
                  "compute_exact_reference": False}
        if method == "adapt_shot":
            kwargs.update(
                selector=ConfidenceSelector(delta=float(spec.pop("delta", 0.05)),
                                            near_tol=float(spec.pop("near_tol", 0.05)),
                                            bound=spec.pop("bound", "eb")),
                allocator=UniformDoubling(base=int(spec.pop("base", 256)),
                                          max_factor=int(spec.pop("max_factor", 64))),
                backend=FiniteShotBackend(seed=seed),
                grouping=True, accept_ambiguous=False)
        result = run_adapt(model, context["pool"], **kwargs)
        return {"energy": result.energy,
                "evidence": "exact" if method == "adapt_exact" else "finite_sample",
                "operators": len(result.labels), "labels": list(result.labels),
                "total_shots": result.total_shots,
                "total_circuits": result.total_circuits,
                "abstentions": result.abstentions,
                "stopped_reason": result.stopped_reason,
                "support_peak": result.support_peak}

    raise AssertionError(f"method kind {method!r} is declared but not implemented")


def run_rung(rung: dict, methods: dict) -> list[dict]:
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.sparse import to_sparse

    model, kind = build_system(rung["system"])
    reference, sector = reference_energy(model, kind)
    rho = ExactMVBackend().state(model.reference, ())
    observables = observable_set(model, kind)

    exact_observables = {}
    if observables:
        from clifford_qc.sparse import sparse_ground, sparse_ground_in_sector

        if kind == "spin_lattice":
            psi = sparse_ground(model.hamiltonian, k=1)[1][:, 0]
        else:
            psi = sparse_ground_in_sector(model.hamiltonian,
                                          model.metadata["n_electrons"],
                                          model.metadata["sz"])[1][:, 0]
        exact_observables = {
            name: float((psi.conj() @ (to_sparse(operator) @ psi)).real)
            for name, operator in observables.items()}

    context = {"rho": rho, "reference": reference, "sector": sector,
               "observables": observables, "exact_observables": exact_observables,
               "pool": word_pool(model, kind),
               "candidates": build_candidates(model, kind,
                                              rung.get("krylov_order", 6))}

    rows = []
    for name in rung["methods"]:
        spec = methods[name]
        started = time.perf_counter()
        row = run_method(name, spec, model, kind, context)
        row.update({
            "system": model.name, "rung": rung.get("rung", 0), "n": model.n,
            "method": name, "family": spec.get("family", name.split("_")[0]),
            "hamiltonian_terms": len(model.hamiltonian.terms),
            "candidate_pool": len(context["candidates"]),
            "word_pool": len(context["pool"]),
            "reference_energy": reference, "sector": row.get("sector", sector),
            "wall_seconds": time.perf_counter() - started,
        })
        row["error"] = row["energy"] - reference
        row["relative_error"] = abs(row["error"]) / max(abs(reference), 1e-12)
        row["chemical_accuracy"] = bool(abs(row["error"]) < CHEMICAL_ACCURACY)
        rows.append(row)
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rungs", default=None,
                        help="comma-separated rung names to run (default: all)")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    methods = config["methods"]
    wanted = set(args.rungs.split(",")) if args.rungs else None
    ladder = [rung for rung in config["ladder"]
              if wanted is None or rung["name"] in wanted]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as handle:
        for rung in ladder:
            for row in run_rung(rung, methods):
                row["rung_name"] = rung["name"]
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                written += 1
                print(f"[{written}] {row['rung_name']:22s} {row['method']:20s} "
                      f"E={row['energy']:.8f} err={row['error']:+.2e} "
                      f"shots={row.get('total_shots', 0):,}", flush=True)
    print(f"wrote {written} runs to {out}")


if __name__ == "__main__":
    main()
