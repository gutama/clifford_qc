"""Seed-ensemble sweep for the Phase 11B coarse-to-fine packet claim.

Each ``(system, ordering, shots, seed)`` cell uses one oracle sample and hands
the identical sampled configurations to the plain dressed and packet-dressed
arms.  Orderings and shot budgets are repeated conditions within a seed; the
companion summarizer therefore performs inference by resampling whole seeds,
not individual treatment cells.

Evidence boundary: the default sampling input is the exact sector ground
state.  These rows test selector/subspace behaviour and carry no implementable
state-preparation claim.
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
    ORACLE,
    StateInput,
    configuration_generators_from_words,
    configuration_ordering,
    dressed_family,
    identity_generator,
    run_acase,
    run_coarse_to_fine_acase,
    sample_state_input,
)

ORDERINGS = ("probability", "physics", "graph", "random")

DEFAULT_SYSTEMS = (
    "hubbard_2x2",
    "hubbard_2x3",
    "h4_equilibrium",
    "h4_stretched",
    "h2o_qsci",
    "beh2_stretched",
)


def _reference_word(backend: SectorStatevectorBackend, model) -> int:
    vector = backend.state_from_program(model.reference)
    return int(backend.basis[int(np.argmax(np.abs(vector)))])


def _ordered_inputs(model, backend, operator, state, words, indices, *, method: str,
                    max_support: int, max_generators: int, seed: int):
    reference_word = _reference_word(backend, model)
    kwargs = {"reference_word": reference_word, "n": model.n, "seed": seed}
    if method == "probability":
        # These indices were sampled from this exact StateInput.  Reusing the
        # same vector matters for a degenerate/near-degenerate eigenspace: an
        # independent ground-state solve could return a different vector and
        # rank configurations by probabilities that did not generate the draw.
        amplitudes = np.asarray(state.amplitudes, dtype=complex)
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


def run_cell(name: str, model, backend, operator, rho, exact_energy: float, *,
             state: StateInput,
             ordering_method: str, shots: int, seed: int, max_size: int,
             max_generators: int, max_support: int,
             max_packet_support: int) -> dict:
    """One paired dressed-vs-packets comparison at a single draw."""
    indices, sampling = sample_state_input(state, shots=shots, seed=seed)
    words = backend.basis[indices]

    ordering, configurations, family, seed_basis, reference_sampled = _ordered_inputs(
        model, backend, operator, state, words, indices, method=ordering_method,
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
            sum(record.candidates_scored for record in dressed.records)),
    }

    if len(configurations) < 2:
        row.update({
            "packet_eligible": False,
            "packet_ineligible_reason": (
                "fewer than two non-reference sampled configurations"),
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
    try:
        hierarchy = run_coarse_to_fine_acase(
            rho, model.hamiltonian, configurations, final_pool,
            initial=seed_basis,
            config=ACASEConfig(max_size=max_size, exact_ground_energy=exact_energy),
            packet_steps=packet_steps,
            max_packet_support=max_packet_support,
            label_prefix=f"ensH{ordering_method[:3]}")
    except (ValueError, RuntimeError) as exc:
        expected = {
            "the packet support cap admits no coarse direction",
            "packet hierarchy has no admissible frontier",
        }
        if str(exc) not in expected:
            raise
        row.update({
            "packet_eligible": False,
            "packet_ineligible_reason": str(exc),
            "packet_energy": None,
            "packet_error": None,
            "packet_M": None,
            "packet_seconds": time.perf_counter() - started,
            "packet_selection_work": None,
            "packet_directions": 0,
            "log_ratio": None,
        })
        return row
    packet_seconds = time.perf_counter() - started
    packet = hierarchy.result

    packet_error = abs(float(packet.energy) - exact_energy)
    dressed_error = row["dressed_error"]
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
            sum(record.candidates_scored for record in packet.records)
            + hierarchy.frontiers_scored),
        "packet_directions": len(hierarchy.packet_labels),
        "packet_labels": list(hierarchy.packet_labels),
        "refined_intervals": [list(interval) for interval in hierarchy.refined_intervals],
        "log_ratio": log_ratio,
        "matched_M": int(packet.basis_size) == int(dressed.basis_size),
    })
    return row


def _validate_output(path: Path, *, systems: list[str], orderings: list[str],
                     shots: list[int], seeds: int, max_size: int,
                     force: bool) -> None:
    """Refuse accidental replacement, especially across experiment designs."""
    if not path.exists():
        return
    try:
        first = next(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip())
        header = json.loads(first)
    except (StopIteration, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"refusing to overwrite unrecognized existing output {path}") from exc

    expected = {
        "systems": systems,
        "orderings": orderings,
        "shots": shots,
        "seeds": seeds,
        "max_size": max_size,
    }
    mismatches = [
        key for key, value in expected.items()
        if key not in header or header[key] != value
    ]
    if header.get("record") != "header" or mismatches:
        detail = ", ".join(mismatches) if mismatches else "record type"
        raise SystemExit(
            f"refusing to overwrite {path}: experiment header differs in {detail}")
    if not force:
        raise SystemExit(f"output {path} already exists; pass --force to replace it")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
                        default=Path(
                            "benchmarks/results/packet_seed_ensemble_rerun.jsonl"))
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing output only when its experiment header matches")
    args = parser.parse_args(argv)

    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    shot_grid = [int(item) for item in args.shots.split(",") if item.strip()]
    orderings = [item.strip() for item in args.orderings.split(",") if item.strip()]
    unknown = sorted(set(orderings) - set(ORDERINGS))
    if unknown:
        raise SystemExit(f"unknown orderings: {unknown}")
    if args.seeds < 1 or not shot_grid or min(shot_grid) < 1:
        raise SystemExit("seeds and shot counts must be positive")

    _validate_output(
        args.out, systems=systems, orderings=orderings, shots=shot_grid,
        seeds=args.seeds, max_size=args.max_size, force=args.force)

    provenance = execution_provenance()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(systems) * len(orderings) * len(shot_grid) * args.seeds
    done = 0
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(stamp_record({
            "schema": "clifford_qc.packet_seed_ensemble.v2",
            "record": "header",
            "evidence": "oracle_sampled; selector behaviour only",
            "claim_boundary": (
                "paired dressed-vs-packet comparison at matched budget; no "
                "implementable state-preparation claim"),
            "inference_unit": "seed_cluster",
            "systems": systems,
            "orderings": orderings,
            "shots": shot_grid,
            "seeds": args.seeds,
            "max_size": args.max_size,
        }, provenance)) + "\n")

        for name in systems:
            model, construction = phase10.build_system(name)
            backend = SectorStatevectorBackend(
                model.n, int(model.metadata["n_electrons"]),
                float(model.metadata["sz"]))
            operator = backend.operator(model.hamiltonian)
            values, vectors = backend.ground_state(model.hamiltonian, k=1)
            exact_energy = float(values[0])
            state = StateInput(
                label="exact_ground_oracle",
                category=ORACLE,
                amplitudes=np.asarray(vectors, dtype=complex)[:, 0],
                basis=backend.basis,
                metadata={"state_kind": "exact_eigenvector"},
            )
            rho = ExactMVBackend().state(model.reference, ())

            handle.write(json.dumps({
                "record": "system",
                "system": name,
                "model": model.name,
                "n_qubits": int(model.n),
                "n_electrons": int(model.metadata["n_electrons"]),
                "sector_dimension": int(backend.dimension),
                "construction": construction,
            }, sort_keys=True) + "\n")

            for ordering_method in orderings:
                for shots in shot_grid:
                    for seed in range(args.seeds):
                        row = run_cell(
                            name, model, backend, operator, rho, exact_energy,
                            state=state,
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
