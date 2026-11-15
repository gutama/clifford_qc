"""Exploratory, matrix-only replay of frozen PRD finite-shot seed records.

Compare a fixed-threshold control with nearest-correlation repair using the
same measured S/H matrices and the same numerical cutoffs. An optional fixed
rotation adds two paired arms. No measurements, exact-reference evaluations,
angle fitting, or confidence intervals are made.
The archived shot-calibrated estimators are not replaced by this comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import run_prd_case_finite_shot as finite
from benchmarks import run_prd_case_paper_suite as suite
from clifford_qc import selection
from clifford_qc.subspace import linalg, pencil
from clifford_qc.subspace.linalg import solve_projected
from clifford_qc.subspace.pencil import repair_overlap, rotation_parameters, solve_rotated


SCHEMA = "clifford_qc.pencil_estimator_replay.v2"


def implementation_sources() -> dict[str, Path]:
    """Hash the modules actually imported, including campaign acceptance policy."""
    sources = {"benchmarks/replay_pencil_estimators.py": Path(__file__).resolve()}
    for module in (finite, suite, linalg, pencil, selection):
        if module.__file__ is None:
            raise ValueError(f"implementation source is unavailable: {module.__name__}")
        sources[module.__name__.replace(".", "/") + ".py"] = Path(module.__file__).resolve()
    return sources


def evaluate_record(record: dict, solver_kwargs: dict, *, violation_tolerance: float,
                    chemical_accuracy_mha: float, rotation: dict | None = None) -> dict:
    """Evaluate paired arms; reference energies never enter a solver or policy."""
    if rotation is not None:
        rotation = rotation_parameters(**rotation)
        if set(solver_kwargs) - {"tau_s", "rel_tau", "max_condition", "norm_floor"}:
            raise ValueError("rotated replay supports fixed numerical thresholds only")
    if record.get("schema") != finite.SCHEMA:
        raise ValueError("expected a PRD finite-shot seed record")
    S = finite.decode_complex_matrix(record["S_hat"], name="S_hat")
    H = finite.decode_complex_matrix(record["H_hat"], name="H_hat")
    if S.shape != H.shape or S.shape[0] != record["M"]:
        raise ValueError("matrix dimensions disagree with the recorded basis size")
    truth = float(record["exact_energy"])
    packet = float(record["exact_packet_energy"])
    if not np.isfinite(truth) or not np.isfinite(packet):
        raise ValueError("recorded reference energies must be finite")
    if record["physical_shots"] != record["task"]["shot_budget"]:
        raise ValueError("recorded physical shots disagree with the task")
    arms = []
    policies = [("fixed_threshold", False, False), ("nearest_correlation", True, False)]
    if rotation is not None:
        policies += [("rotated_fixed_threshold", False, True),
                     ("rotated_nearest_correlation", True, True)]
    for estimator, repaired, rotated in policies:
        try:
            overlap, diagnostics = (repair_overlap(S, norm_floor=solver_kwargs["norm_floor"])
                                    if repaired else (S, None))
            result = (solve_rotated(overlap, H, **rotation, **solver_kwargs) if rotated
                      else solve_projected(overlap, H, **solver_kwargs))
            energy = result.ground_energy
            if not np.isfinite(energy):
                raise np.linalg.LinAlgError("nonfinite Ritz value")
        except (ValueError, np.linalg.LinAlgError) as error:
            arms.append({"estimator": estimator, "status": "failed_solve",
                         "error_type": type(error).__name__, "error": str(error)})
            continue
        arms.append({
            "estimator": estimator, "status": "ok", "energy": energy,
            "energy_error_mHa": (energy - truth) * 1000.0,
            "absolute_error_mHa": abs(energy - truth) * 1000.0,
            "sampling_error_mHa": (energy - packet) * 1000.0,
            "variational_violation": energy < truth - violation_tolerance,
            "chemical_accuracy": abs(energy - truth) * 1000.0 <= chemical_accuracy_mha,
            "retained_rank": result.effective_rank,
            "condition_number": result.condition_number,
            "smallest_retained_overlap_eigenvalue":
                result.resources["smallest_retained_overlap_eigenvalue"],
            "repair": diagnostics,
            "rotation": result.resources.get("pencil_rotation"),
        })
    return {"task": record["task"], "physical_shots": record["physical_shots"],
            "arms": arms}


def summarize(cells: list[dict]) -> list[dict]:
    """Keep failed solves in the census; label conditional error statistics."""
    groups = defaultdict(list)
    for cell in cells:
        task = cell["task"]
        for arm in cell["arms"]:
            groups[(task["system"], task["shot_budget"], arm["estimator"])].append(arm)
    rows = []
    for (system, budget, estimator), arms in sorted(groups.items()):
        ok = [arm for arm in arms if arm["status"] == "ok"]
        errors = [arm["absolute_error_mHa"] for arm in ok]
        rows.append({
            "system": system, "shot_budget": budget, "estimator": estimator,
            "cells": len(arms), "successful_solves": len(ok),
            "failed_solves": len(arms) - len(ok),
            "failure_rate": (len(arms) - len(ok)) / len(arms),
            "chemical_accuracy_rate_all_cells": sum(a["chemical_accuracy"] for a in ok)
                / len(arms),
            "variational_violation_rate_successful_solves": (
                sum(a["variational_violation"] for a in ok) / len(ok) if ok else None),
            "median_absolute_error_mHa_successful_solves": (
                float(np.median(errors)) if ok else None),
            "p90_absolute_error_mHa_successful_solves": (
                float(np.quantile(errors, 0.9)) if ok else None),
        })
    return rows


def replay_records(paths, manifest: dict, frozen_bases: dict, *,
                   rotation: dict | None = None) -> dict:
    """Read without modifying sources; reject duplicate or foreign campaign cells."""
    paths = sorted(Path(path) for path in paths)
    if not paths:
        raise ValueError("no PRD seed records found")
    if rotation is not None:
        rotation = rotation_parameters(**rotation)
    config_hash = suite.config_sha256(manifest)
    declared = {task["task_id"]: task for task in finite.build_tasks(manifest)}
    solver = manifest["finite_shot_tier"]["solver"]
    kwargs = {"tau_s": float(solver["tau_s"]),
              "rel_tau": float(solver["relative_tau_s"]),
              "max_condition": float(solver["max_condition"]),
              "norm_floor": float(solver["norm_floor"])}
    summary = manifest["finite_shot_tier"]["summary"]
    cells, sources, seen = [], [], set()
    basis_hashes = {}
    for path in paths:
        payload = path.read_bytes()
        record = json.loads(payload)
        if record.get("config_sha256") != config_hash:
            raise ValueError(f"{path}: campaign config hash disagrees")
        task = record["task"]
        key = task["task_id"]
        if declared.get(key) != task:
            raise ValueError(f"{path}: undeclared campaign task")
        if key in seen:
            raise ValueError(f"duplicate campaign task: {key}")
        seen.add(key)
        basis = record["basis_sha256"]
        expected_basis = frozen_bases["systems"][task["system"]]["basis_sha256"]
        if basis != expected_basis:
            raise ValueError(f"{path}: basis digest disagrees with the frozen declaration")
        basis_hashes[task["system"]] = basis
        if record["M"] != manifest["finite_shot_tier"]["M"]:
            raise ValueError(f"{path}: undeclared basis size")
        cells.append(evaluate_record(
            record, kwargs,
            violation_tolerance=float(summary["variational_violation_tolerance_hartree"]),
            chemical_accuracy_mha=float(summary["chemical_accuracy_mHa"]), rotation=rotation))
        sources.append({"task_id": key, "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    implementations = implementation_sources()
    return {
        "schema": SCHEMA, "evidence": "exploratory_matrix_replay",
        "config_sha256": config_hash, "basis_sha256_by_system": basis_hashes,
        "solver_kwargs": kwargs,
        "rotation_policy": ({**rotation, "angle_policy": "fixed",
                             "selection": "caller_supplied_no_data_selection"}
                            if rotation is not None else None),
        "expected_cells": len(declared), "replayed_cells": len(cells),
        "complete_campaign": len(cells) == len(declared),
        "missing_task_ids": sorted(set(declared) - seen),
        "additional_physical_shots": 0,
        "interval_coverage": None,
        "claim_boundary": "Fixed-threshold point-estimate comparison only. No coverage, "
            "variationality, quantum advantage, or preregistered improvement claim. "
            "Archived shot-calibrated policies remain separate baselines.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "implementation_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in implementations.items()},
        "implementation_paths": {name: str(path) for name, path in implementations.items()},
        "sources": sources, "cells": cells, "summary": summarize(cells),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=finite.DEFAULT_CONFIG)
    parser.add_argument("--pencil-angle", type=float, help="fixed angle in radians; adds paired arms")
    parser.add_argument("--energy-shift", type=float, default=0.0)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--pole-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new exploratory artifact")
    rotation = None
    if args.pencil_angle is not None:
        try:
            rotation = rotation_parameters(angle=args.pencil_angle, energy_shift=args.energy_shift,
                                           energy_scale=args.energy_scale,
                                           pole_tolerance=args.pole_tolerance)
        except ValueError as error:
            parser.error(str(error))
    elif (args.energy_shift, args.energy_scale, args.pole_tolerance) != (0.0, 1.0, 1e-10):
        parser.error("rotation controls require --pencil-angle")
    manifest = suite.load_manifest(args.config)
    frozen_bases = finite.load_frozen_bases(args.config, manifest)
    paths = list(args.input_dir.glob("*__shots*__seed*.json"))
    report = replay_records(paths, manifest, frozen_bases, rotation=rotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwriting a source or a racing writer.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "replayed_cells": report["replayed_cells"],
                      "complete_campaign": report["complete_campaign"]}))


if __name__ == "__main__":
    main()
