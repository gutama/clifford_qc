"""Run the preregistered Phase 16A comparison (Q15), once.

Pooled Trotter-circuit time-evolved QSCI against sample-independent iterated
selected CI of the same size, exactly as ``benchmarks/configs/phase16a_te_qsci.json``
froze it. See ``benchmarks/PHASE16A_PREREGISTRATION.md``.

The producer refuses before a shot is drawn unless all of the following hold:

* the preregistration gate passes, both its static clauses and its admission
  recomputation from the committed inputs;
* the run is the declared one (no reduced replicas or instance subset
  written to the committed record path);
* no record already exists there (the experiment runs once);
* the working tree is clean, so the record's provenance names the code that
  produced it;
* the environment is one the committed records declare (the stamp's guard,
  tripped before sampling rather than after it).

It then derives the time grid and Trotter step counts from the declared rules
and checks them against ``measured_before_freezing``; a mismatch makes the
instance INVALID. It draws every (instance, arm, budget, replica) cell from its
declared ``SeedSequence`` stream, solves QSCI with ``run_qsci``, and evaluates
both controls at each candidate replica's configuration count. Both controls
are sample-blind, so each is computed once per count. Finally it applies the
frozen decision rule over the required instances and stamps provenance.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_phase16a_te_qsci.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase16a_preregistration as gate
except ImportError:  # pragma: no cover - script execution
    import check_phase16a_preregistration as gate

SCHEMA = "clifford_qc.phase16a_te_qsci.v1"
RECORD = gate.RECORD
CANDIDATE = "te_trotter_pooled"
SAMPLED_ARMS = ("te_trotter_pooled", "te_exact_pooled", "exact_ground_oracle")
CLAIM_BOUNDARY = (
    "This record reports the one comparison benchmarks/configs/phase16a_te_qsci.json "
    "froze: pooled Trotter-circuit time-evolved QSCI against sample-independent iterated "
    "selected CI at each replica's configuration count, under the frozen time grid, Trotter "
    "rule, shot grid, replicas, seeds and decision rule. The verdict reads the required "
    "instances only; the diagnostic instance and arms are reported and cannot promote. "
    "Draws come from the declared states' exact probabilities, so no device noise, "
    "circuit-depth cost, hardware, scaling or quantum-advantage claim follows.")
STATISTICS = {
    "median": "numpy.median over replicas",
    "p90": "numpy.quantile(q=0.9, method='higher') over replicas",
}


def seed_for(config: dict, instance: str, arm: str, budget_index: int,
             replica: int) -> int:
    """The declared stream of one cell: ``SeedSequence(root, spawn_key=...)``."""
    seeds = config["seeds"]
    key = (seeds["instance_order"].index(instance), seeds["arm_indices"][arm],
           int(budget_index), int(replica))
    sequence = np.random.SeedSequence(int(seeds["root"]), spawn_key=key)
    return int(sequence.generate_state(1)[0])


def p90(values) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), 0.9, method="higher"))


@dataclass
class Prepared:
    """One instance ready to sample: operator, exact reference, and the states."""

    name: str
    role: str
    model: object
    backend: object
    operator: object
    exact: float
    ground_gap: float | None
    diagonal: np.ndarray
    states: dict
    structural: dict
    checks: dict
    trotter_resources: list
    controls: dict = field(default_factory=lambda: {"iterated": {}, "matched": {}})


def prepare_instance(config: dict, name: str, *, model=None,
                     structural: dict | None = None) -> Prepared:
    """Build the declared states for one instance and check them against the freeze."""
    from clifford_qc.backends import SectorStatevectorBackend
    from clifford_qc.subspace import ORACLE, StateInput, time_evolved_state

    spec = config["instances"][name]
    model = gate.build_instance(spec) if model is None else model
    if structural is None:
        structural = gate.structural_quantities(config, spec, model=model)
    frozen = config["measured_before_freezing"][name]
    metadata = model.metadata
    backend = SectorStatevectorBackend(model.n, metadata["n_electrons"], metadata["sz"],
                                       spin_ordering=metadata["spin_convention"])
    operator = backend.operator(model.hamiltonian)
    dense = np.column_stack([operator.matvec(column)
                             for column in np.eye(backend.dimension)])
    values, vectors = np.linalg.eigh(0.5 * (dense + dense.conj().T))
    gap = float(values[1] - values[0]) if values.size > 1 else None

    times = [float(t) for t in frozen["times"]]
    steps = [int(s) for s in frozen["trotter_steps"]]
    order = int(config["trotter"]["order"])
    trotter = [time_evolved_state(backend, model, t, method="trotter",
                                  trotter_steps=s, trotter_order=order)
               for t, s in zip(times, steps)]
    exact_states = [time_evolved_state(backend, model, t, method="expm_multiply")
                    for t in times]
    ground = StateInput(
        label="exact_ground_oracle", category=ORACLE, amplitudes=vectors[:, 0],
        basis=backend.basis,
        metadata={"state_kind": "exact_eigenvector",
                  "solver": "numpy.linalg.eigh on the dense sector matrix"})
    checks = {
        "times_match_frozen": len(times) == len(structural["times"]) and all(
            gate._close(a, b) for a, b in zip(times, structural["times"])),
        "trotter_steps_match_frozen": steps == structural["trotter_steps"],
        "expected_support_matches_frozen": (
            frozen["expected_support_at_top_budget"]
            == structural["expected_support_at_top_budget"]),
        "ground_state_nondegenerate": gap is None or gap > 1e-8,
    }
    resources = [{
        "time": t, "trotter_steps": s,
        "rotor_count": state.metadata["rotor_count"],
        "fidelity_to_exact": state.metadata["fidelity_to_exact"],
        "sector_leakage": state.metadata["sector_leakage"],
        "post_selected": state.post_selection is not None,
    } for t, s, state in zip(times, steps, trotter)]
    return Prepared(
        name=name, role=spec["role"], model=model, backend=backend,
        operator=operator, exact=float(values[0]), ground_gap=gap,
        diagonal=np.real(np.diag(dense)).copy(),
        states={"te_trotter_pooled": trotter, "te_exact_pooled": exact_states,
                "exact_ground_oracle": [ground]},
        structural=structural, checks=checks, trotter_resources=resources)


def sample_cell(prepared: Prepared, arm: str, total: int, seed: int):
    """Draw one cell's pooled configurations and solve QSCI on them."""
    from clifford_qc.subspace import run_qsci, sample_state_inputs

    states = prepared.states[arm]
    indices, sampling = sample_state_inputs(
        states, shots=total // len(states), seed=seed, stability_bootstrap=None)
    result = run_qsci(prepared.operator, indices, sampling=sampling,
                      exact_energy=prepared.exact)
    return indices, sampling, result


def control_at(prepared: Prepared, kind: str, size: int, sampled) -> dict:
    """A sample-blind control at one configuration count, memoized by count."""
    from clifford_qc.subspace import run_control

    memo = prepared.controls["iterated" if kind == "iterated_selected_ci" else "matched"]
    if size not in memo:
        control = run_control(prepared.operator, sampled, name=kind, kind=kind,
                              max_determinants=size, score="epstein_nesbet",
                              diagonal=prepared.diagonal, exact_energy=prepared.exact)
        memo[size] = {"size": control.determinant_count,
                      "error": float(control.energy - prepared.exact),
                      "closed": bool(control.metadata.get("closed_before_budget", False))}
    return memo[size]


def evaluate(config: dict, candidate_cells: dict, checks: dict) -> dict:
    """The frozen instance rule, read off the candidate's cells."""
    target = float(config["target"]["value"])
    tie = float(config["decision_rule"]["tie_tolerance"])
    if not all(checks.values()):
        failed = sorted(key for key, ok in checks.items() if not ok)
        return {"status": "INVALID", "shots_to_target": None,
                "median_paired_difference": None, "failed_checks": failed}
    budgets = sorted(int(b) for b in candidate_cells)
    reached = next((b for b in budgets
                    if float(np.median(candidate_cells[str(b)]["errors"])) <= target), None)
    if reached is None:
        return {"status": "UNDETERMINED", "shots_to_target": None,
                "median_paired_difference": None}
    cell = candidate_cells[str(reached)]
    paired = (np.asarray(cell["errors"], dtype=float)
              - np.asarray(cell["iterated_control"]["errors"], dtype=float))
    median = float(np.median(paired))
    return {"status": "PASS" if median < -tie else "FAIL",
            "shots_to_target": reached, "median_paired_difference": median}


def run_instance(config: dict, prepared: Prepared, *, replicas: int,
                 progress=None) -> dict:
    budgets = [int(b) for b in config["shot_grid"]["total_shots"]]
    arms = {}
    for arm in SAMPLED_ARMS:
        cells = {}
        for budget_index, total in enumerate(budgets):
            cell = {"seeds": [], "unique_configurations": [], "errors": [],
                    "discarded_shots": [], "sampled_outside_sector": 0}
            if arm == CANDIDATE:
                cell["iterated_control"] = {"sizes": [], "errors": [], "closed": []}
                cell["matched_control"] = {"sizes": [], "errors": []}
            for replica in range(replicas):
                seed = seed_for(config, prepared.name, arm, budget_index, replica)
                indices, sampling, result = sample_cell(prepared, arm, total, seed)
                cell["seeds"].append(seed)
                cell["unique_configurations"].append(int(indices.size))
                cell["errors"].append(float(result.energy - prepared.exact))
                cell["discarded_shots"].append(int(sampling.discarded_shots))
                cell["sampled_outside_sector"] += int(
                    np.count_nonzero(indices >= prepared.backend.dimension))
                if arm == CANDIDATE:
                    iterated = control_at(prepared, "iterated_selected_ci",
                                          int(indices.size), indices)
                    matched = control_at(prepared, "matched_selected_ci",
                                         int(indices.size), indices)
                    cell["iterated_control"]["sizes"].append(iterated["size"])
                    cell["iterated_control"]["errors"].append(iterated["error"])
                    cell["iterated_control"]["closed"].append(iterated["closed"])
                    cell["matched_control"]["sizes"].append(matched["size"])
                    cell["matched_control"]["errors"].append(matched["error"])
            cell["median_error"] = float(np.median(cell["errors"]))
            cell["p90_error"] = p90(cell["errors"])
            cell["median_unique_configurations"] = float(
                np.median(cell["unique_configurations"]))
            cells[str(total)] = cell
            if progress is not None:
                progress(f"{prepared.name} {arm} {total}: median "
                         f"{cell['median_error']:.3e}")
        arms[arm] = {"budgets": cells}

    checks = dict(prepared.checks)
    checks["replicas_complete"] = all(
        len(cell["errors"]) == replicas for arm in arms.values()
        for cell in arm["budgets"].values())
    checks["sampled_inside_sector"] = all(
        cell["sampled_outside_sector"] == 0 for arm in arms.values()
        for cell in arm["budgets"].values())
    decision = evaluate(config, arms[CANDIDATE]["budgets"], checks)
    decision["promotes"] = prepared.role == "required_decision"
    measured = {key: prepared.structural[key] for key in (
        gate._INT_FIELDS + gate._LIST_INT_FIELDS + gate._FLOAT_FIELDS + ("times",))}
    return {
        "role": prepared.role,
        "n_qubits": int(prepared.model.n),
        "sector_dimension": int(prepared.backend.dimension),
        "exact_ground_energy": prepared.exact,
        "ground_gap": prepared.ground_gap,
        "reference_error": prepared.structural["reference_error"],
        "measured_at_execution": measured,
        "deterministic_checks": checks,
        "trotter_resources": prepared.trotter_resources,
        "arms": arms,
        "decision": decision,
    }


def run_experiment(config: dict, *, config_bytes: bytes, replicas: int | None = None,
                   instances=None, build=None, structural: dict | None = None,
                   progress=None) -> dict:
    """The declared experiment as a record (unstamped). ``build`` is for tests."""
    replicas = int(config["replicas"]) if replicas is None else int(replicas)
    required = list(config["required_decision_instances"])
    diagnostic = list(config["diagnostic_instances"])
    names = list(instances) if instances is not None else required + diagnostic
    started = time.perf_counter()
    record = {
        "schema": SCHEMA,
        "config_path": str(gate.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistration": {
            "gate": "benchmarks/check_phase16a_preregistration.py",
            "revision": config["revisions"][-1]["revision"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "quantum_advantage_claim": False,
        "evidence": config["evidence"],
        "target": config["target"],
        "tie_tolerance": config["decision_rule"]["tie_tolerance"],
        "shot_grid": [int(b) for b in config["shot_grid"]["total_shots"]],
        "replicas": replicas,
        "seed_root": int(config["seeds"]["root"]),
        "statistics": STATISTICS,
        "instances": {},
    }
    for name in names:
        model = None if build is None else build(config["instances"][name])
        prepared = prepare_instance(
            config, name, model=model,
            structural=None if structural is None else structural.get(name))
        record["instances"][name] = run_instance(config, prepared, replicas=replicas,
                                                 progress=progress)
    statuses = {name: record["instances"][name]["decision"]["status"]
                for name in required if name in record["instances"]}
    record["decision"] = {
        "required_instances": required,
        "instance_statuses": statuses,
        "diagnostic_statuses": {
            name: record["instances"][name]["decision"]["status"]
            for name in diagnostic if name in record["instances"]},
        "verdict": gate.verdict_of(statuses[name] for name in required)
        if all(name in statuses for name in required) else "INCOMPLETE",
        "rule": "frozen in the config; see decision_rule",
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return record


def refusals(args, *, reduced: bool) -> list[str]:
    """Reasons to stop before any computation; each keeps the one run honest."""
    problems = []
    if args.out.resolve() == RECORD.resolve():
        if reduced:
            problems.append("a reduced run (--replicas or --instances) is not the "
                            "declared experiment and may not write the committed record")
        if RECORD.exists() and not args.overwrite:
            problems.append(f"{RECORD.name} exists: the declared "
                            "experiment runs once (pass --overwrite only to "
                            "regenerate it deliberately)")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RECORD,
                        help="record path (default: the committed record)")
    parser.add_argument("--replicas", type=int, default=None,
                        help="reduced replica count; refused for the committed path")
    parser.add_argument("--instances", nargs="*", default=None,
                        help="instance subset; refused for the committed path")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate an existing committed record deliberately")
    args = parser.parse_args(argv)
    reduced = args.replicas is not None or args.instances is not None
    problems = refusals(args, reduced=reduced)
    if problems:
        for problem in problems:
            print(f"REFUSE {problem}")
        return 1

    from clifford_qc.reproducibility import execution_provenance, stamp_record

    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    problems = gate.static_problems(config)
    notes: list[str] = []
    computed: dict = {}
    if not problems:
        problems = gate.admission_problems(config, notes, computed)
    if problems:
        for problem in problems:
            print(f"REFUSE preregistration gate: {problem}")
        return 1
    for note in notes:
        print(note)
    provenance = execution_provenance()
    if args.out.resolve() == RECORD.resolve() and provenance["git_dirty"]:
        print("REFUSE the working tree is dirty; commit first so the record's "
              "provenance names the code that produced it")
        return 1
    # The stamp enforces the record-environment contract. Trip it before the
    # first shot, so an undeclared environment costs nothing, not the run.
    stamp_record({}, provenance)

    record = run_experiment(config, config_bytes=config_bytes, replicas=args.replicas,
                            instances=args.instances, structural=computed,
                            progress=lambda line: print(line, flush=True))
    stamped = stamp_record(record, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamped, indent=1) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\nverdict: {decision['verdict']}  statuses: {decision['instance_statuses']}  "
          f"diagnostic: {decision['diagnostic_statuses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
