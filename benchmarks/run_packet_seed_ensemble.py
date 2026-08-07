"""Seed-ensemble sweep for the Phase 11B coarse-to-fine packet claim.

The Phase 10 primary record establishes the packet arm's advantage at a single
draw per system (``seed=0``, ``shots=128``, one ordering, one retained-direction
budget).  Five rows at one draw each is a screen, not evidence: the §11C
go/no-go explicitly requires that a packet result "is not an accident of one
ordering", and nothing in the record yet bounds the seed-to-seed spread.

This driver runs the *paired* comparison that closes that gap.  For each
``(system, ordering, shots, seed)`` cell it draws one sample and hands the
identical sampled configurations to both hybrid arms:

``dressed``
    sampled configurations plus the dressed excitation family -- Phase 10C
    arm 2, the baseline the packet stage has to beat.
``packets``
    the same pool entered through the support-pruned coarse-to-fine Haar
    packet hierarchy -- Phase 10C arm 3.

Pairing is what makes ~20 seeds informative: both arms see the same draw, the
same ordering, and the same retained-direction budget, so the per-cell
difference isolates the packet stage rather than the sampling noise.  The
summarizer therefore reports paired statistics (win rate, sign test, bootstrap
CI on the median log ratio) rather than comparing two independent means.

Evidence boundary: the default sampling input is the exact sector ground state.
That is Phase 8's validation oracle.  These rows test *selector* behaviour and
carry no implementable state-preparation claim, exactly as in Phases 10-12.

Run from the repository root::

    PYTHONPATH=. python benchmarks/run_packet_seed_ensemble.py \
        --systems hubbard_2x2,hubbard_2x3 --seeds 20

then summarize with ``benchmarks/summarize_packet_ensemble.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import run_phase10_hybrid as phase10
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import (
    ACASEConfig,
    configuration_generators_from_words,
    configuration_ordering,
    dressed_family,
    exact_ground_state_oracle,
    identity_generator,
    run_acase,
    run_coarse_to_fine_acase,
    sample_state_input,
)

ORDERINGS = ("probability", "physics", "graph", "random")

# The §11C ablations are the point of the sweep, so every ordering is run on
# every cell.  A packet advantage that survives only under `physics` ordering is
# a reportable negative result, not a win.
DEFAULT_SYSTEMS = ("hubbard_2x2", "hubbard_2x3", "h4_equilibrium", "h4_stretched")


def _ordered_inputs(model, backend, operator, words, indices, *, method: str,
                    max_support: int, max_generators: int, seed: int):
    """Phase 12's hybrid input construction, with the ordering parameterized."""
    reference_word = _reference_word(backend, model)
    kwargs = {"reference_word": reference_word, "n": model.n, "seed": seed}
    if method in ("probability", "graph"):
        amplitudes = np.asarray(operator_state_amplitudes(backend, model), dtype=complex)
        kwargs["probabilities"] = np.abs(amplitudes[indices]) ** 2
    if method == "graph":
        kwargs["graph_matrix"] = operator.restrict(indices)

    ordering = configuration_ordering(words, method=method, **kwargs)
    ordered_words = ordering.words
    reference_sampled = bool(np.any(ordered_words == reference_word))
    try:
        configurations = configuration_generators_from_words(
            model, ordered_words, label_prefix=f"ens{method[:3]}")
    except ValueError as exc:
        if "every sampled word was the reference determinant" not in str(exc):
            raise
        configurations = []
    sampled_basis = list(configurations)
    if reference_sampled:
        sampled_basis.append(identity_generator(model.n))
    if not sampled_basis:
        raise RuntimeError("sampled input produced no usable variational direction")
    family = dressed_family(
        sampled_basis, model, kind="excitation", max_support=max_support,
        max_generators=max_generators)
    seed_basis = None if reference_sampled else [configurations[0]]
    return ordering, configurations, family, seed_basis, reference_sampled


def _reference_word(backend: SectorStatevectorBackend, model) -> int:
    """Occupation word of the reference program, as Phase 12 resolves it."""
    vector = backend.state_from_program(model.reference)
    return int(backend.basis[int(np.argmax(np.abs(vector)))])


def operator_state_amplitudes(backend, model):
    """Exact sector ground-state amplitudes, cached per backend instance."""
    cached = getattr(backend, "_ensemble_amplitudes", None)
    if cached is None:
        _, vectors = backend.ground_state(model.hamiltonian, k=1)
        cached = np.asarray(vectors)[:, 0]
        backend._ensemble_amplitudes = cached
    return cached


def run_cell(name: str, model, backend, operator, rho, exact_energy: float, *,
             ordering_method: str, shots: int, seed: int, max_size: int,
             max_generators: int, max_support: int,
             max_packet_support: int) -> dict:
    """One paired dressed-vs-packets comparison at a single draw."""
    state = exact_ground_state_oracle(backend, model.hamiltonian)
    indices, sampling = sample_state_input(state, shots=shots, seed=seed)
    words = backend.basis[indices]

    ordering, configurations, family, seed_basis, reference_sampled = _ordered_inputs(
        model, backend, operator, words, indices, method=ordering_method,
        max_support=max_support, max_generators=max_generators, seed=seed)
    final_pool = list(configurations) + list(family.generators)

    started = time.perf_counter()
    dressed = run_acase(
        rho, model.hamiltonian, final_pool, initial=seed_basis,
        max_size=max_size, exact_ground_energy=exact_energy)
    dressed_seconds = time.perf_counter() - started

    row = {
        "system": name,
        "ordering": ordering_method,
        "shots": int(shots),
        "seed": int(seed),
        "max_size": int(max_size),
        "exact_energy": float(exact_energy),
        "unique_configurations": int(sampling.unique_configurations),
        "reference_sampled": reference_sampled,
        "configuration_count": len(configurations),
        "retained_probability": float(sampling.retained_probability),
        "evidence_category": "oracle_sampled",
        "dressed_energy": float(dressed.energy),
        "dressed_error": abs(float(dressed.energy) - exact_energy),
        "dressed_M": int(dressed.basis_size),
        "dressed_seconds": dressed_seconds,
        "dressed_selection_work": int(
            sum(r.candidates_scored for r in dressed.records)),
    }

    # Fewer than two non-reference configurations leaves no interval to
    # transform; record the cell as packet-ineligible rather than dropping it,
    # so the denominator of the win rate stays honest.
    if len(configurations) < 2:
        row.update({
            "packet_eligible": False,
            "packet_energy": None,
            "packet_error": None,
            "packet_M": None,
            "packet_seconds": None,
            "packet_selection_work": None,
            "packet_directions": 0,
            "log_ratio": None,
        })
        return row

    packet_steps = max(1, max_size // 3)
    started = time.perf_counter()
    hierarchy = run_coarse_to_fine_acase(
        rho, model.hamiltonian, configurations, final_pool,
        initial=seed_basis,
        config=ACASEConfig(max_size=max_size, exact_ground_energy=exact_energy),
        packet_steps=packet_steps,
        max_packet_support=max_packet_support,
        label_prefix=f"ensH{ordering_method[:3]}")
    packet_seconds = time.perf_counter() - started
    packet = hierarchy.result

    packet_error = abs(float(packet.energy) - exact_energy)
    dressed_error = row["dressed_error"]
    # Log ratio is the paired statistic: negative means the packet arm is more
    # accurate at the same budget.  Guard the degenerate exactly-zero cases.
    floor = 1e-15
    log_ratio = float(np.log10(max(packet_error, floor)
                               / max(dressed_error, floor)))
    row.update({
        "packet_eligible": True,
        "packet_energy": float(packet.energy),
        "packet_error": packet_error,
        "packet_M": int(packet.basis_size),
        "packet_seconds": packet_seconds,
        "packet_selection_work": int(
            sum(r.candidates_scored for r in packet.records)
            + hierarchy.frontiers_scored),
        "packet_directions": len(hierarchy.packet_labels),
        "packet_labels": list(hierarchy.packet_labels),
        "refined_intervals": [list(i) for i in hierarchy.refined_intervals],
        "log_ratio": log_ratio,
        "matched_M": int(packet.basis_size) == int(dressed.basis_size),
    })
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--seeds", type=int, default=20,
                        help="seeds 0..N-1 per (system, ordering, shots) cell")
    parser.add_argument("--shots", default="32,64,128,256")
    parser.add_argument("--orderings", default=",".join(ORDERINGS))
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--max-generators", type=int, default=128)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--max-packet-support", type=int, default=16)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/results/packet_seed_ensemble.jsonl"))
    args = parser.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    shot_grid = [int(s) for s in args.shots.split(",") if s.strip()]
    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]
    unknown = sorted(set(orderings) - set(ORDERINGS))
    if unknown:
        raise SystemExit(f"unknown orderings: {unknown}")

    # Provenance is a precondition: fail before a long sweep, not after it.
    provenance = execution_provenance()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(systems) * len(orderings) * len(shot_grid) * args.seeds
    done = 0
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(stamp_record({
            "schema": "clifford_qc.packet_seed_ensemble.v1",
            "record": "header",
            "evidence": "oracle_sampled; selector behaviour only",
            "claim_boundary": (
                "paired dressed-vs-packet comparison at matched budget; no "
                "implementable state-preparation claim"),
            "systems": systems, "orderings": orderings,
            "shots": shot_grid, "seeds": args.seeds,
            "max_size": args.max_size,
        }, provenance)) + "\n")

        for name in systems:
            model, _ = phase10.build_system(name)
            backend = SectorStatevectorBackend(
                model.n, int(model.metadata["n_electrons"]),
                float(model.metadata["sz"]))
            operator = backend.operator(model.hamiltonian)
            values, _ = backend.ground_state(model.hamiltonian, k=1)
            exact_energy = float(values[0])
            rho = ExactMVBackend().state(model.reference, ())

            for ordering_method in orderings:
                for shots in shot_grid:
                    for seed in range(args.seeds):
                        row = run_cell(
                            name, model, backend, operator, rho, exact_energy,
                            ordering_method=ordering_method, shots=shots,
                            seed=seed, max_size=args.max_size,
                            max_generators=args.max_generators,
                            max_support=args.max_support,
                            max_packet_support=args.max_packet_support)
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                        done += 1
                    print(f"{name:18s} {ordering_method:12s} shots={shots:<5d} "
                          f"[{done}/{total}]", flush=True)

    print(f"wrote {done} cells to {args.out}", flush=True)


if __name__ == "__main__":
    main()
