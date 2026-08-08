"""Seed-level replication of the Phase 12 matched-budget M arms.

The Phase 12 ladder runs one seed. Review of PR #48 correctly objected that a
single draw cannot support a claim about the packet arm, and that the M=7
classical comparator in the committed record was a truncation of the sample
rather than a selection over it.  This driver replicates exactly the
sampling-dependent arms at the ladder's budget, across seeds and all four
Phase 11C orderings, so each becomes a distribution rather than a point.

Arms replicated, all at the same retained-direction budget and all fed the
identical draw within a cell:

``budget_selected_ci``
    classical selection that *has* seen the sample.  Below the seed size it
    ranks seed and outside determinants together by estimated squared
    amplitude, so the two compete under one criterion instead of the seed
    preceding every candidate by construction.
``matched_selected_ci``
    classical selection that has *not* seen the sample; sample-independent, so
    it is constant across seeds within a system and acts as a fixed baseline.
``acase``
    bare operator-generated subspace over the conserving excitation family.
``qsci_dressed_acase``
    sampled configurations plus the dressed family.
``qsci_haar_dressed_acase``
    the same pool entered through the coarse-to-fine Haar packet hierarchy.

``fixed_krylov`` is deliberately absent: it is deterministic, does not consult
the sample, and costs ~43 minutes per draw on the 12-qubit rung.  Replicating
it would buy nothing and dominate the runtime.

Evidence boundary: the sampling input is the exact sector ground state, Phase
8's validation oracle.  These rows test subspace construction, not an
implementable state preparation.

Run from the repository root::

    PYTHONPATH=. python benchmarks/run_m7_seed_replication.py \
        --systems hubbard_2x2,hubbard_2x3 --seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import run_acase_ladder as ladder
from benchmarks import run_phase10_hybrid as phase10
from benchmarks import run_phase12_paper_b as phase12
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import (
    ACASEConfig,
    run_acase,
    run_coarse_to_fine_acase,
    run_control,
)
from clifford_qc.subspace.qsci import (
    exact_ground_state_oracle,
    sample_state_input,
)
from clifford_qc.subspace.multiresolution import configuration_ordering

ORDERINGS = ("probability", "physics", "graph", "random")
DEFAULT_SYSTEMS = ("hubbard_2x2", "hubbard_2x3")

# The arms whose value depends on the draw.  Ordered so the two classical
# comparators sit next to each other in every emitted row.
ARMS = (
    "budget_selected_ci",
    "matched_selected_ci",
    "acase",
    "qsci_dressed_acase",
    "qsci_haar_dressed_acase",
)


def _ordered_hybrid_inputs(model, backend, operator, state, words, indices, *,
                           method: str, max_support: int, max_generators: int,
                           seed: int):
    """Phase 12's hybrid inputs with the §11C ordering parameterized."""
    reference_word = phase12._reference_word(backend, model)
    kwargs = {"reference_word": reference_word, "n": model.n, "seed": seed}
    if method == "probability":
        # Reuse the vector that generated the draw; an independent eigensolve
        # can return a different vector inside a degenerate eigenspace.
        amplitudes = np.asarray(state.amplitudes, dtype=complex)
        kwargs["probabilities"] = np.abs(amplitudes[indices]) ** 2
    if method == "graph":
        kwargs["graph_matrix"] = operator.restrict(indices)
    ordering = configuration_ordering(words, method=method, **kwargs)

    from clifford_qc.subspace import (
        configuration_generators_from_words,
        dressed_family,
        identity_generator,
    )

    ordered = ordering.words
    reference_sampled = bool(np.any(ordered == reference_word))
    try:
        configurations = configuration_generators_from_words(
            model, ordered, label_prefix=f"m7{method[:3]}")
    except ValueError as exc:
        if "every sampled word was the reference determinant" not in str(exc):
            raise
        configurations = []
    sampled_basis = list(configurations)
    if reference_sampled:
        sampled_basis.append(identity_generator(model.n))
    if not sampled_basis:
        raise RuntimeError("sampled input produced no usable direction")
    family = dressed_family(
        sampled_basis, model, kind="excitation", max_support=max_support,
        max_generators=max_generators)
    seed_basis = None if reference_sampled else [configurations[0]]
    return configurations, family, seed_basis


def run_cell(name: str, model, backend, operator, rho, exact_energy: float, *,
             state, ordering: str, shots: int, seed: int, max_size: int,
             max_generators: int, max_support: int,
             max_packet_support: int) -> dict:
    """Every replicated arm on one draw, so the comparison is paired."""
    indices, sampling = sample_state_input(state, shots=shots, seed=seed)
    words = backend.basis[indices]
    budget = max_size + 1

    row = {
        "system": name,
        "ordering": ordering,
        "shots": int(shots),
        "seed": int(seed),
        "max_size": int(max_size),
        "determinant_budget": int(budget),
        "exact_energy": float(exact_energy),
        "unique_configurations": int(sampling.unique_configurations),
        "evidence_category": "oracle_sampled",
        "errors": {},
        "M": {},
        "metadata": {},
    }

    def record(arm: str, energy: float, m: int, meta: dict | None = None) -> None:
        row["errors"][arm] = abs(float(energy) - exact_energy)
        row["M"][arm] = int(m)
        if meta:
            row["metadata"][arm] = meta

    for arm, kind in (("budget_selected_ci", "budget_matched"),
                      ("matched_selected_ci", "matched_selected_ci")):
        control = run_control(
            operator, indices, name=arm, kind=kind, n=model.n,
            exact_energy=exact_energy, max_determinants=budget)
        record(arm, control.energy, control.determinant_count, {
            "selected_added": control.metadata.get("selected_added"),
            "seed_truncated": control.metadata.get("seed_truncated"),
            "unified_amplitude_ranking": control.metadata.get(
                "unified_amplitude_ranking"),
            "sample_independent": control.metadata.get("sample_independent"),
        })

    kind = "fermionic_lattice" if name.startswith("hubbard_") else "molecular"
    candidates = ladder.build_candidates(model, kind)
    bare = run_acase(rho, model.hamiltonian, candidates,
                     max_size=max_size, exact_ground_energy=exact_energy)
    record("acase", bare.energy, bare.basis_size)

    configurations, family, seed_basis = _ordered_hybrid_inputs(
        model, backend, operator, state, words, indices, method=ordering,
        max_support=max_support, max_generators=max_generators, seed=seed)
    final_pool = list(configurations) + list(family.generators)

    dressed = run_acase(rho, model.hamiltonian, final_pool, initial=seed_basis,
                        max_size=max_size, exact_ground_energy=exact_energy)
    record("qsci_dressed_acase", dressed.energy, dressed.basis_size)

    if len(configurations) >= 2:
        try:
            hierarchy = run_coarse_to_fine_acase(
                rho, model.hamiltonian, configurations, final_pool,
                initial=seed_basis,
                config=ACASEConfig(max_size=max_size,
                                   exact_ground_energy=exact_energy),
                packet_steps=max(1, max_size // 3),
                max_packet_support=max_packet_support,
                label_prefix=f"m7H{ordering[:3]}")
        except (ValueError, RuntimeError) as exc:
            expected = {"the packet support cap admits no coarse direction",
                        "packet hierarchy has no admissible frontier"}
            if str(exc) not in expected:
                raise
            row["packet_ineligible_reason"] = str(exc)
        else:
            packet = hierarchy.result
            record("qsci_haar_dressed_acase", packet.energy, packet.basis_size,
                   {"packet_directions": len(hierarchy.packet_labels)})
    else:
        row["packet_ineligible_reason"] = (
            "fewer than two non-reference sampled configurations")
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--orderings", default=",".join(ORDERINGS))
    parser.add_argument("--max-size", type=int, default=6)
    parser.add_argument("--max-generators", type=int, default=128)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--max-packet-support", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmarks/results/m7_seed_replication.jsonl"))
    args = parser.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]
    unknown = sorted(set(orderings) - set(ORDERINGS))
    if unknown:
        raise SystemExit(f"unknown orderings: {unknown}")
    if args.out.exists() and not args.force:
        raise SystemExit(f"output {args.out} exists; pass --force to replace it")

    provenance = execution_provenance()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(systems) * len(orderings) * args.seeds
    done = 0
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(stamp_record({
            "schema": "clifford_qc.m7_seed_replication.v1",
            "record": "header",
            "arms": list(ARMS),
            "systems": systems,
            "orderings": orderings,
            "shots": [int(args.shots)],
            "seeds": int(args.seeds),
            "max_size": int(args.max_size),
            "inference_unit": "seed_cluster",
            "evidence": "oracle_sampled; subspace construction only",
            "claim_boundary": (
                "paired within-draw comparison of the ladder's matched-budget "
                "arms; fixed_krylov excluded as deterministic"),
        }, provenance)) + "\n")

        for name in systems:
            model, _ = phase10.build_system(name)
            backend = SectorStatevectorBackend(
                model.n, int(model.metadata["n_electrons"]),
                float(model.metadata["sz"]))
            operator = backend.operator(model.hamiltonian)
            exact_energy = float(
                backend.ground_state(model.hamiltonian, k=1)[0][0])
            rho = ExactMVBackend().state(model.reference, ())
            state = exact_ground_state_oracle(backend, model.hamiltonian)

            for ordering in orderings:
                for seed in range(args.seeds):
                    started = time.perf_counter()
                    row = run_cell(
                        name, model, backend, operator, rho, exact_energy,
                        state=state, ordering=ordering, shots=args.shots,
                        seed=seed, max_size=args.max_size,
                        max_generators=args.max_generators,
                        max_support=args.max_support,
                        max_packet_support=args.max_packet_support)
                    row["cell_seconds"] = time.perf_counter() - started
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    done += 1
                print(f"{name:14s} {ordering:12s} [{done}/{total}]", flush=True)

    print(f"wrote {done} cells to {args.out}", flush=True)


if __name__ == "__main__":
    main()
