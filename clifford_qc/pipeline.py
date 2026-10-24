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
                   frontier_pairs=32, max_rank=2, max_candidates=0,
                   reference_mode="determinant", reference_arithmetic="legacy_trace"):
    from .models.metadata import spin_convention
    from .backends import ExactMVBackend
    from .subspace.adaptive import ACASEConfig, run_acase
    from .subspace.fermionic_generators import determinant_excitations, occupied_spin_orbitals
    from .subspace.projection import MatrixElementBank
    from .subspace.streaming import StreamingMatrixElementBank
    from .reference_states import (DeterminantReference, UnsupportedReferenceOperation,
                                   operator_identity)
    from .subspace.providers import ExactProjectedProvider

    if policy not in ("retain_all", "stream_recompute"):
        raise ValueError("policy must be retain_all or stream_recompute")
    if storage not in ("object", "packed"):
        raise ValueError("storage must be object or packed")
    if reference_mode not in ("determinant", "legacy_mv"):
        raise ValueError("reference_mode must be determinant or legacy_mv")
    if reference_arithmetic not in ("legacy_trace", "analytic"):
        raise ValueError("reference_arithmetic must be legacy_trace or analytic")
    if reference_mode == "legacy_mv" and reference_arithmetic != "legacy_trace":
        raise ValueError("reference_arithmetic applies to the determinant path")
    if isinstance(max_rank, bool) or max_rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")
    for name, value, minimum in (("max_candidates", max_candidates, 0),
                                  ("frontier_pairs", frontier_pairs, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    cfg = config if config is not None else ACASEConfig()
    if isinstance(cfg.max_size, bool) or not isinstance(cfg.max_size, int) or cfg.max_size < 0:
        raise ValueError("max_size must be a nonnegative integer")
    if reference_mode == "determinant" and (cfg.convergence is not None or (
            cfg.leakage_tol is not None and cfg.leakage_mode != "operator")):
        raise UnsupportedReferenceOperation(
            "reference-conditioned certification/H-squared reporting requires reference_mode='legacy_mv'; "
            "the action reporting consumer is a subsequent restart step")
    started = time.perf_counter()
    model = prepared.model()
    ordering = spin_convention(model)
    if ordering != "interleaved":
        raise UnsupportedReferenceOperation("prepared determinant excitations require interleaved spin ordering")
    if model.metadata.get("fermion_encoding", "jw") != "jw":
        raise UnsupportedReferenceOperation("prepared determinant excitations require Jordan-Wigner occupation bits")
    provider = None
    if reference_mode == "determinant":
        reference = DeterminantReference.from_program(model.reference, arithmetic=reference_arithmetic)
        occupied = reference.occupied
    else:
        occupied = occupied_spin_orbitals(model)
    candidates = determinant_excitations(model.n, occupied, max_rank=max_rank)
    available_candidates = len(candidates)
    if max_candidates:
        candidates = candidates[:max_candidates]
    if reference_mode == "determinant":
        provider = ExactProjectedProvider(reference, model.hamiltonian, storage=storage,
                                           policy=policy, frontier_pairs=frontier_pairs)
        bank = provider.bank
        reference_identity = provider.reference_identity
    else:
        reference = ExactMVBackend().state(model.reference, ())
        bank = (MatrixElementBank(reference, model.hamiltonian, storage=storage)
                if policy == "retain_all" else StreamingMatrixElementBank(
                    reference, model.hamiltonian, storage=storage, frontier_pairs=frontier_pairs))
        reference_identity = operator_identity(reference)
    result = run_acase(reference, model.hamiltonian, candidates, bank=bank, config=cfg)
    if provider is not None:
        result.resources.update(provider.execution_metadata())
    record = {
        "schema": "clifford_qc.prepared_run.v2", "problem_fingerprint": prepared.fingerprint,
        "implementation": implementation_fingerprint(), "method": "acase", "evidence_tier": "exact",
        "reference": {"mode": reference_mode, "identity": reference_identity,
                       "arithmetic": reference_arithmetic if provider is not None else "density_mv",
                       "bit_order": "q0-leftmost", "spin_ordering": ordering},
        "stopping_uses_oracle": cfg.exact_ground_energy is not None and cfg.target_error is not None,
        "config": asdict(cfg), "max_rank": max_rank, "candidate_count": len(candidates),
        "max_candidates": max_candidates, "available_candidates": available_candidates,
        "labels": result.labels, "energy": result.energy, "energy_history": result.energy_history,
        "stopped_reason": result.stopped_reason, "resources": result.resources,
        "wall_seconds": time.perf_counter() - started,
        "accuracy_statement": "Exact projected arithmetic; convergence to the full ground state is not certified.",
    }
    return result, record


def solve_prepared_sector(prepared, *, config=None, target_spin=0., precompute="auto"):
    """Independent bounded classical action workflow, with separately priced setup."""
    from .backends import SectorStatevectorBackend
    from .backends.sector_reference import SectorSpec, SectorStateReference, SectorActionProvider, sector_input_plan
    from .models.metadata import spin_convention
    from .reference_states import DeterminantReference, UnsupportedReferenceOperation
    from .subspace.bounded_actions import ActionMemoryLimit, RSSMonitor
    from .subspace.sector_solver import SectorSolveConfig, sector_memory_plan, solve_sector

    cfg = SectorSolveConfig() if config is None else config
    if not isinstance(cfg, SectorSolveConfig):
        raise TypeError("config must be a SectorSolveConfig")
    started = time.perf_counter()
    model = prepared.model()
    if spin_convention(model) != "interleaved" or model.metadata.get("fermion_encoding", "jw") != "jw":
        raise UnsupportedReferenceOperation("sector-action workflow requires interleaved Jordan-Wigner inputs")
    spec = SectorSpec(model.n, model.metadata["n_electrons"], model.metadata["sz"])
    determinant = DeterminantReference.from_program(model.reference)
    if (len(determinant.occupied) != spec.n_electrons
            or sum(0.5 if q % 2 == 0 else -0.5 for q in determinant.occupied) != spec.sz):
        raise ValueError("prepared determinant disagrees with the declared particle/spin sector")
    plan = setup_monitor = None
    try:
        if precompute != "auto" and not isinstance(precompute,bool):
            raise ValueError("precompute must be auto, True or False")
        if cfg.roots > spec.validate_spin(target_spin):
            raise ValueError("requested roots exceed the target total-spin dimension")
        for use_precompute in ((True,False) if precompute == "auto" else (precompute,)):
            try:
                payload,_ = sector_input_plan(spec,model.hamiltonian,precompute=use_precompute,
                                             max_owned_bytes=cfg.max_owned_bytes)
                plan = sector_memory_plan(spec,cfg,target_spin=target_spin,input_bytes=payload['admission_bytes'])
                break
            except ActionMemoryLimit:
                if precompute != "auto" or not use_precompute:
                    raise
        setup_monitor = RSSMonitor(plan)
        setup_monitor.check(admission=True)
        backend = SectorStatevectorBackend(spec.n, spec.n_electrons, spec.sz)
        amplitudes = backend.occupation_state(determinant.bits)
        reference = SectorStateReference(spec, amplitudes, max_owned_bytes=cfg.max_owned_bytes)
        del backend, amplitudes
        provider = SectorActionProvider(reference, model.hamiltonian, target_spin=target_spin,
                                        max_owned_bytes=cfg.max_owned_bytes,precompute=use_precompute)
        setup_monitor.check()
        preparation_seconds = time.perf_counter() - started
        result = solve_sector(provider, config=cfg)
        record = result.as_dict()
        record.update(problem_fingerprint=prepared.fingerprint, implementation=implementation_fingerprint(),
                      preparation_seconds=preparation_seconds, preparation_rss=setup_monitor.as_dict(),
                      preparation_memory_plan=plan.as_dict(),precompute_requested=precompute,
                      preparation_scope="model reconstruction, sector reference, H/S² compilation and validation; solver wall_seconds reported separately")
        return result, record
    except (ActionMemoryLimit,ValueError) as exc:
        return None, {"schema": "clifford_qc.sector_solve.v1",
                      "status": "resource_limited" if isinstance(exc,ActionMemoryLimit) else "invalid_configuration",
                      "problem_fingerprint": prepared.fingerprint, "energies": [],
                      "ground_state_status": "not_established", "stopping_uses_spectral_oracle": False,
                      "reason": str(exc), "elapsed_seconds": time.perf_counter() - started,
                      "config": asdict(cfg), "sector_dimension": spec.dimension,
                      "preparation_rss": setup_monitor.as_dict() if setup_monitor is not None else None,
                      "admission_plan": plan.as_dict() if plan is not None else None}


def validate_prepared(prepared, *, method="auto", roots=1):
    from .backends import SectorStatevectorBackend
    import numpy as np
    started = time.perf_counter()
    model = prepared.model()
    backend = SectorStatevectorBackend(model.n, model.metadata["n_electrons"], model.metadata["sz"])
    if isinstance(roots, bool) or int(roots) != roots or not 1 <= roots <= backend.dimension:
        raise ValueError("roots must be an integer between 1 and the sector dimension")
    roots = int(roots)
    values, vectors, operator = backend.ground_state(
        model.hamiltonian, k=roots, method=method, return_operator=True)
    residuals = [float(np.linalg.norm(operator.matvec(vectors[:, i]) - value * vectors[:, i]))
                 for i, value in enumerate(values)]
    return {"schema": "clifford_qc.prepared_validation.v1",
            "problem_fingerprint": prepared.fingerprint, "method": method,
            "implementation": implementation_fingerprint(), "energies": values.tolist(),
            "residual_norms": residuals, "sector_dimension": backend.dimension,
            "wall_seconds": time.perf_counter() - started,
            "scope": "optional classical reference; not charged to the solver record"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", help="Cache an FCIDUMP model; no exact reference solve")
    prepare.add_argument("source")
    prepare.add_argument("--cache-directory", required=True)
    prepare.add_argument("--integral-tolerance", type=float, default=1e-12)
    for stage in ("solve", "solve-sector", "validate"):
        command = sub.add_parser(stage)
        command.add_argument("prepared")
        command.add_argument("--output", required=True)
        if stage == "solve":
            command.add_argument("--reference-mode", choices=("determinant", "legacy_mv"), default="determinant")
            command.add_argument("--reference-arithmetic", choices=("legacy_trace", "analytic"),
                                 default="legacy_trace",
                                 help="determinant row arithmetic; legacy_trace requires fewer than 40 qubits")
            command.add_argument("--storage", choices=("object", "packed"), default="object")
            command.add_argument("--policy", choices=("retain_all", "stream_recompute"), default="retain_all")
            command.add_argument("--frontier-pairs", type=int, default=32)
            command.add_argument("--max-additions", type=int, default=10)
            command.add_argument("--max-rank", type=int, default=2)
            command.add_argument("--max-candidates", type=int, default=0,
                                 help="0 uses the full candidate pool")
        elif stage == "solve-sector":
            command.add_argument("--roots", type=int, default=1)
            command.add_argument("--target-spin", type=float, default=0.)
            command.add_argument("--ncv", type=int, default=32)
            command.add_argument("--max-iterations", type=int, default=300)
            command.add_argument("--recovery-attempts", type=int, default=2)
            command.add_argument("--residual-tolerance", type=float, default=1e-8)
            command.add_argument("--seed", type=int, default=20261024)
            command.add_argument("--max-owned-bytes", type=int, default=256 * 2**20)
            command.add_argument("--max-rss-bytes", type=int)
            command.add_argument("--rss-headroom-bytes", type=int, default=0)
            command.add_argument("--precompute", choices=("auto","on","off"), default="auto",
                                 help="retain gather tables when the full owned-memory plan admits them")
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
                                       frontier_pairs=args.frontier_pairs, max_rank=args.max_rank,
                                       max_candidates=args.max_candidates, reference_mode=args.reference_mode,
                                       reference_arithmetic=args.reference_arithmetic)
        elif args.stage == "solve-sector":
            from .subspace.sector_solver import SectorSolveConfig
            try:
                cfg = SectorSolveConfig(roots=args.roots, ncv=args.ncv, max_iterations=args.max_iterations,
                                    recovery_attempts=args.recovery_attempts,
                                    residual_tolerance=args.residual_tolerance, seed=args.seed,
                                    max_owned_bytes=args.max_owned_bytes, max_rss_bytes=args.max_rss_bytes,
                                    rss_headroom_bytes=args.rss_headroom_bytes)
                _, record = solve_prepared_sector(prepared,config=cfg,target_spin=args.target_spin,
                                                 precompute={"auto":"auto","on":True,"off":False}[args.precompute])
            except ValueError as exc:
                record = {"schema":"clifford_qc.sector_solve.v1","status":"invalid_configuration",
                          "energies":[],"reason":str(exc),"ground_state_status":"not_established",
                          "stopping_uses_spectral_oracle":False,"problem_fingerprint":prepared.fingerprint}
        else:
            record = validate_prepared(prepared, method=args.method, roots=args.roots)
        _atomic_json(args.output, record)
        print(args.output)


if __name__ == "__main__":
    main()
