"""Separate preparation, A-CASE solving and optional sector validation.

Run ``python -m clifford_qc.pipeline --help``. Each CLI invocation is an
independent process, so preparation and validation need not coexist with a bank.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

from .prepared import PreparedProblem, prepare_fcidump, _atomic_json, implementation_fingerprint


def solve_prepared(prepared, *, config=None, storage="object", policy="retain_all",
                   frontier_pairs=32, max_rank=2):
    from .backends import ExactMVBackend
    from .subspace.adaptive import ACASEConfig, run_acase
    from .subspace.fermionic_generators import determinant_excitations, occupied_spin_orbitals
    from .subspace.projection import MatrixElementBank
    from .subspace.streaming import StreamingMatrixElementBank

    if policy not in ("retain_all", "stream_recompute"):
        raise ValueError("policy must be retain_all or stream_recompute")
    started = time.perf_counter()
    model = prepared.model()
    rho = ExactMVBackend().state(model.reference, ())
    candidates = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=max_rank)
    cfg = config if config is not None else ACASEConfig()
    bank = (MatrixElementBank(rho, model.hamiltonian, storage=storage)
            if policy == "retain_all" else StreamingMatrixElementBank(
                rho, model.hamiltonian, storage=storage, frontier_pairs=frontier_pairs))
    result = run_acase(rho, model.hamiltonian, candidates, bank=bank, config=cfg)
    record = {
        "schema": "clifford_qc.prepared_run.v1", "problem_fingerprint": prepared.fingerprint,
        "implementation": implementation_fingerprint(), "method": "acase", "evidence_tier": "exact",
        "stopping_uses_oracle": cfg.exact_ground_energy is not None and cfg.target_error is not None,
        "config": asdict(cfg), "max_rank": max_rank, "candidate_count": len(candidates),
        "labels": result.labels, "energy": result.energy, "energy_history": result.energy_history,
        "stopped_reason": result.stopped_reason, "resources": result.resources,
        "wall_seconds": time.perf_counter() - started,
        "accuracy_statement": "Exact projected arithmetic; convergence to the full ground state is not certified.",
    }
    return result, record


def validate_prepared(prepared, *, method="auto", roots=1):
    from .backends import SectorStatevectorBackend
    import numpy as np
    model = prepared.model()
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], model.metadata["sz"])
    if isinstance(roots, bool) or int(roots) != roots or not 1 <= roots <= backend.dimension:
        raise ValueError("roots must be an integer between 1 and the sector dimension")
    roots = int(roots)
    values, vectors = backend.ground_state(model.hamiltonian, k=roots, method=method)
    operator = backend.operator(model.hamiltonian)
    residuals = [float(np.linalg.norm(operator.matvec(vectors[:, i]) - value * vectors[:, i]))
                 for i, value in enumerate(values)]
    return {"schema": "clifford_qc.prepared_validation.v1",
            "problem_fingerprint": prepared.fingerprint, "method": method,
            "implementation": implementation_fingerprint(), "energies": values.tolist(),
            "residual_norms": residuals, "sector_dimension": backend.dimension,
            "scope": "optional classical reference; not charged to the solver record"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", help="Cache an FCIDUMP model; no exact reference solve")
    prepare.add_argument("source")
    prepare.add_argument("--cache-directory", required=True)
    prepare.add_argument("--integral-tolerance", type=float, default=1e-12)
    for stage in ("solve", "validate"):
        command = sub.add_parser(stage)
        command.add_argument("prepared")
        command.add_argument("--output", required=True)
        if stage == "solve":
            command.add_argument("--storage", choices=("object", "packed"), default="object")
            command.add_argument("--policy", choices=("retain_all", "stream_recompute"), default="retain_all")
            command.add_argument("--frontier-pairs", type=int, default=32)
            command.add_argument("--max-additions", type=int, default=10)
            command.add_argument("--max-rank", type=int, default=2)
        else:
            command.add_argument("--method", choices=("auto", "dense", "eigsh", "lanczos"), default="auto")
            command.add_argument("--roots", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepared, path, hit = prepare_fcidump(args.source, args.cache_directory,
                                             integral_tolerance=args.integral_tolerance)
        print(json.dumps({"prepared": str(path), "fingerprint": prepared.fingerprint, "cache_hit": hit}))
    else:
        prepared = PreparedProblem.load(args.prepared)
        if args.stage == "solve":
            from .subspace.adaptive import ACASEConfig
            _, record = solve_prepared(prepared, config=ACASEConfig(max_size=args.max_additions),
                                       storage=args.storage, policy=args.policy,
                                       frontier_pairs=args.frontier_pairs, max_rank=args.max_rank)
        else:
            record = validate_prepared(prepared, method=args.method, roots=args.roots)
        _atomic_json(args.output, record)
        print(args.output)


if __name__ == "__main__":
    main()
