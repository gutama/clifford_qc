"""Experimental configuration-space Haar tier for A-CASE.

This benchmark tests the narrow wavelet proposal that survived the orbital-
basis falsification: change basis among *virtual determinant configurations*,
use only support-pruned Haar details during early growth, then hand the retained
coarse directions to the ordinary convergence-complete level-4 pool.

The transform is classical.  It does not add a quantum wavelet circuit and it
does not rotate the Hubbard Hamiltonian.  Results are reported at matched basis
size together with the projected Pauli-word universe ``W``, maximum generator
support ``S_A``, and maximum projected-element support ``S_H``.

Run from the repository root::

    python benchmarks/run_configuration_packets.py \
        --out benchmarks/reference_results/configuration_packets.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np


def physics_order_key(occupied, reference, sites: int):
    """Deterministic Hubbard ordering used as configuration-space locality.

    Nearby leaves first share excitation rank from the reference, then doublon
    count, charge pattern, and spin pattern.  This is a declared heuristic; the
    Haar construction itself does not infer a metric on determinants.
    """
    occupied, reference = frozenset(occupied), frozenset(reference)
    up = tuple(int(2 * site in occupied) for site in range(sites))
    down = tuple(int(2 * site + 1 in occupied) for site in range(sites))
    excitation_rank = len(reference - occupied)
    doublons = sum(a * b for a, b in zip(up, down))
    charge = tuple(a + b for a, b in zip(up, down))
    spin = tuple(a - b for a, b in zip(up, down))
    return excitation_rank, doublons, charge, spin, tuple(sorted(occupied))


def ordered_sector_configurations(model, *, max_configurations: int = 4096):
    """All non-reference determinants in the model's fixed ``(N, S_z)`` block."""
    from clifford_qc.subspace import occupied_spin_orbitals

    metadata = model.metadata
    if metadata.get("kind") != "fermionic_lattice":
        raise ValueError("configuration packets require a fermionic lattice")
    sites = int(metadata["sites"])
    electrons = int(metadata["n_electrons"])
    sz = float(metadata["sz"])
    n_up = int(round(electrons / 2 + sz))
    n_down = electrons - n_up
    if abs(0.5 * (n_up - n_down) - sz) > 1e-12:
        raise ValueError("particle number and S_z do not define integer spin counts")
    total = comb(sites, n_up) * comb(sites, n_down)
    if total > max_configurations:
        raise ValueError(
            f"sector has {total} configurations, above max_configurations="
            f"{max_configurations}; configuration packets are a small-sector tier")

    reference = tuple(occupied_spin_orbitals(model))
    determinants = []
    for up_sites in combinations(range(sites), n_up):
        for down_sites in combinations(range(sites), n_down):
            occupied = tuple(sorted(
                [2 * site for site in up_sites]
                + [2 * site + 1 for site in down_sites]))
            if occupied != reference:
                determinants.append(occupied)
    determinants.sort(key=lambda occupied:
                      physics_order_key(occupied, reference, sites))
    return determinants


def _level4_pool(model):
    from clifford_qc.models.lattice import competing_orders
    from clifford_qc.subspace import (compound_response, configuration_generators,
                                      determinant_excitations,
                                      occupied_spin_orbitals)

    base = determinant_excitations(model.n, occupied_spin_orbitals(model))
    configurations = configuration_generators(model, competing_orders(model))
    return base + configurations + compound_response(
        configurations, base, max_support=64)


def _rounded(value, digits=12):
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _row(name, result, exact):
    selected = [result.bank.generator(index) for index in result.indices]
    resources = result.result.resources
    return {
        "name": name,
        "basis_size": len(result.indices),
        "energy": _rounded(result.energy),
        "error": _rounded(result.energy - exact),
        "word_universe": resources["word_universe"],
        "max_generator_support": max(generator.support() for generator in selected),
        "max_element_support": resources["max_hamiltonian_element_support"],
        "condition_number": _rounded(result.result.condition_number),
        "packet_directions": sum(generator.label.startswith("cfgH")
                                 for generator in selected),
        "stopped_reason": result.stopped_reason,
    }


def run_experiment(*, shape=(2, 2), t: float = 1.0, u: float = 4.0,
                   coarse_size: int = 20, final_size: int = 30,
                   max_packet_support: int = 16,
                   max_configurations: int = 4096, gamma: float = 0.5):
    """Run baseline, coarse packet, and packet-to-level-4 handoff comparisons."""
    from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
    from clifford_qc.models.lattice import hubbard
    from clifford_qc.subspace import (configuration_generator,
                                      configuration_haar_packets, run_acase)

    if coarse_size < 2:
        raise ValueError("coarse_size must be at least 2")
    if final_size < coarse_size:
        raise ValueError("final_size must be at least coarse_size")

    model = hubbard(shape, t=t, U=u)
    rho = ExactMVBackend().state(model.reference, ())
    exact = _rounded(SectorStatevectorBackend(
        model.n, model.metadata["n_electrons"], model.metadata["sz"]
    ).ground_state(model.hamiltonian, k=1)[0][0])

    determinants = ordered_sector_configurations(
        model, max_configurations=max_configurations)
    leaves = [configuration_generator(
        model.reference, occupied, label=f"cfg[{index}]")
        for index, occupied in enumerate(determinants)]
    packets = configuration_haar_packets(
        leaves, max_support=max_packet_support, label_prefix="cfg")
    complete = configuration_haar_packets(
        leaves, max_support=None, include_scaling=True, label_prefix="check")

    # Named invariant: for distinct determinant configurations the complete
    # unbalanced-Haar family must have S=I.  This also establishes Parseval for
    # the finite coefficient transform under the declared ordering.
    overlap = np.array([
        [(left.mv.dagger() * right.mv * rho).trace()
         for right in complete] for left in complete
    ])
    orthogonality_residual = float(
        np.max(np.abs(overlap - np.eye(len(complete)))))

    level4 = _level4_pool(model)
    common = dict(exact_ground_energy=exact, gamma=gamma)
    baseline_coarse = run_acase(
        rho, model.hamiltonian, level4,
        max_size=coarse_size - 1, **common)
    packet_coarse = run_acase(
        rho, model.hamiltonian, level4 + packets,
        max_size=coarse_size - 1, **common)
    baseline_final = run_acase(
        rho, model.hamiltonian, level4,
        max_size=final_size - 1, **common)

    retained = [packet_coarse.bank.generator(index)
                for index in packet_coarse.indices]
    staged = run_acase(
        rho, model.hamiltonian, level4, initial=retained,
        bank=packet_coarse.bank,
        max_size=final_size - len(retained), **common)

    return {
        "evidence": "exact numerical benchmark",
        "system": model.name,
        "shape": list(shape) if isinstance(shape, tuple) else shape,
        "t": t,
        "u": u,
        "exact_ground_energy": exact,
        "ordering": ["excitation_rank", "doublon_count",
                     "charge_pattern", "spin_pattern"],
        "configuration_leaves": len(leaves),
        "complete_packet_directions": len(complete),
        "retained_packet_candidates": len(packets),
        "max_packet_support": max_packet_support,
        "orthogonality_residual": orthogonality_residual,
        "coarse_size": coarse_size,
        "final_size": final_size,
        "gamma": gamma,
        "results": [
            _row("level4_coarse", baseline_coarse, exact),
            _row("level4_plus_packets_coarse", packet_coarse, exact),
            _row("level4_final", baseline_final, exact),
            _row("packet_then_level4_final", staged, exact),
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="+", default=[2, 2])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--coarse-size", type=int, default=20)
    parser.add_argument("--final-size", type=int, default=30)
    parser.add_argument("--max-packet-support", type=int, default=16)
    parser.add_argument("--max-configurations", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    shape = args.shape[0] if len(args.shape) == 1 else tuple(args.shape)
    record = run_experiment(
        shape=shape, t=args.t, u=args.u, coarse_size=args.coarse_size,
        final_size=args.final_size, max_packet_support=args.max_packet_support,
        max_configurations=args.max_configurations, gamma=args.gamma)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return record


if __name__ == "__main__":
    main()
