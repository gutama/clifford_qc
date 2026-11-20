#!/usr/bin/env python3
"""Run the declared molecular acquisition study and write its record.

The systems, schedule, arms, stopping targets, bootstrap plan, seeds, labels
and criteria are fixed by ``docs/restart/molecular_acquisition_declaration.json``
before any evaluation seed is drawn; ``docs/restart/MOLECULAR_ACQUISITION.md``
states what is and is not claimed. This producer only executes it:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python benchmarks/run_molecular_acquisition.py --jobs 4

Each replica of each system is one seeded unit of work. Three runs per replica
read the same pilot: uniform allocation with the correction computed at every
look (which gives both the ``uniform`` and ``uniform_cv`` arms, since uniform
allocation does not depend on what is reported), Ritz-targeted allocation on
the raw functional, and Ritz-targeted allocation on the corrected functional.
Every run spends the whole schedule; an arm's stop is the first look at which
its reported standard error meets a target, which is exact because a stopping
rule only truncates the run it would otherwise have continued. The loop is
``acquire_ritz``'s, so that the cache at each stop is available for its
intervals and bootstrap; the checker replays rows against ``acquire_ritz``
itself.

Rows are appended to ``--work`` as they complete, and a restarted run skips
the replicas already there, so an interruption changes no row. The record is
assembled when every declared replica is present. An existing record is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_interval_coverage import derived_seed, sha256, write_record  # noqa: E402
from benchmarks.run_target_acquisition import _proportion, _ratio_interval  # noqa: E402
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.dense_reference import density_from_statevector  # noqa: E402
from clifford_qc.matrix import to_matrix  # noqa: E402
from clifford_qc.measurement.confidence import simultaneous_z_radius  # noqa: E402
from clifford_qc.measurement.functionals import ritz_functional  # noqa: E402
from clifford_qc.measurement.session import SharedMeasurement  # noqa: E402
from clifford_qc.models import fcidump_model  # noqa: E402
from clifford_qc.reference_states import DeterminantReference  # noqa: E402
from clifford_qc.reproducibility import stamp_record  # noqa: E402
from clifford_qc.subspace import MatrixElementBank, identity_generator  # noqa: E402
from clifford_qc.subspace.acquisition import (  # noqa: E402
    ALLOCATIONS, neyman_group_plan, uniform_group_plan,
)
from clifford_qc.subspace.coverage import delta_method_applies  # noqa: E402
from clifford_qc.subspace.symmetry_pools import spin_adapted_excitations  # noqa: E402
from clifford_qc.subspace.symmetry_shifts import (  # noqa: E402
    certify_bank_family, corrected_ritz_functional, sector_shift_family, shift_pair_table,
)
from clifford_qc.subspace.uncertainty import (  # noqa: E402
    bootstrap_ritz_samples, bootstrap_spread, bootstrap_spread_interval, ritz_uncertainty,
)

DECLARATION = ROOT / "docs" / "restart" / "molecular_acquisition_declaration.json"
RESULTS = ROOT / "docs" / "restart" / "molecular_acquisition_results.json.gz"
WORK = ROOT / "work" / "molecular-acquisition" / "rows.jsonl"
DECLARATION_SCHEMA = "clifford_qc.molecular_acquisition_declaration.v1"
RESULTS_SCHEMA = "clifford_qc.molecular_acquisition_results.v1"
# The sources whose bytes decide a row.
IMPLEMENTATION = (
    "benchmarks/run_molecular_acquisition.py",
    "benchmarks/run_target_acquisition.py",
    "benchmarks/run_interval_coverage.py",
    "clifford_qc/subspace/acquisition.py",
    "clifford_qc/subspace/symmetry_shifts.py",
    "clifford_qc/subspace/symmetry_pools.py",
    "clifford_qc/subspace/uncertainty.py",
    "clifford_qc/subspace/coverage.py",
    "clifford_qc/subspace/linalg.py",
    "clifford_qc/subspace/elements.py",
    "clifford_qc/subspace/projection.py",
    "clifford_qc/measurement/session.py",
    "clifford_qc/measurement/functionals.py",
    "clifford_qc/measurement/cache.py",
    "clifford_qc/measurement/confidence.py",
    "clifford_qc/measurement/grouping.py",
    "clifford_qc/backends/finite_shot.py",
    "clifford_qc/models/fcidump.py",
    "clifford_qc/multivector.py",
)
STOP_REASONS = ("target_met", "schedule_exhausted", "deterministic")
VERDICTS = ("fewer_shots_at_matched_error", "no_resolved_difference",
            "more_shots_at_matched_error")
SOLVE_FAILURES = (ValueError, np.linalg.LinAlgError)


def implementation_digests(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256(root / relative) for relative in IMPLEMENTATION}


# ---------------------------------------------------------------- declaration

def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def load_declaration(path: Path = DECLARATION) -> dict:
    """Read and validate a declaration; every later step trusts its shape."""
    declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    if declaration.get("schema") != DECLARATION_SCHEMA:
        raise ValueError("unsupported molecular-acquisition declaration schema")
    if declaration.get("target") != "exact_subspace_ground_energy":
        raise ValueError("the only supported target is exact_subspace_ground_energy")
    systems = [system["key"] for system in declaration["systems"]]
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("system keys must be nonempty and distinct")
    for system in declaration["systems"]:
        if system.get("role") not in ("development", "held_out"):
            raise ValueError(f"{system['key']} must be development or held_out")
        if not _positive_int(system.get("replicas")) or system["replicas"] < 2:
            raise ValueError(f"{system['key']} needs at least two replicas")
        source = system["fcidump"]
        if set(source) not in ({"inline"}, {"path", "sha256"}):
            raise ValueError(f"{system['key']} names its FCIDUMP inline or by path and digest")
        if system["reference"]["kind"] not in ("random_singlet", "cisd_in_pool"):
            raise ValueError(f"{system['key']} has an unknown reference")
        if system["electrons"] % 2 or not 0 < system["electrons"] < 2 * system["n_spatial"]:
            raise ValueError(f"{system['key']} must be a closed shell in its active space")
    runs = {run["key"]: run for run in declaration["runs"]}
    if len(runs) != len(declaration["runs"]) or any(
            run["allocation"] not in ALLOCATIONS for run in runs.values()):
        raise ValueError("runs must have distinct keys and declared allocations")
    arms = [arm["key"] for arm in declaration["arms"]]
    if len(set(arms)) != len(arms):
        raise ValueError("arm keys must be distinct")
    for arm in declaration["arms"]:
        if arm["run"] not in runs or (arm["corrected"] and not runs[arm["run"]]["control_variates"]):
            raise ValueError(f"arm {arm['key']} reads a run that cannot report it")
    cumulative = declaration["cumulative_shots_per_group"]
    if len(cumulative) < 2 or not all(_positive_int(shots) for shots in cumulative) \
            or any(b <= a for a, b in zip(cumulative, cumulative[1:])):
        raise ValueError("cumulative_shots_per_group must be increasing positive integers")
    targets = declaration["target_std_errors"]
    if not targets or len(set(targets)) != len(targets) or any(
            not isinstance(t, float) or not math.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("target_std_errors must be distinct positive finite floats")
    if not delta_method_applies(declaration["solver_kwargs"]):
        raise ValueError("the declared estimator must be one the delta method accepts")
    for name in ("delta", "summary_confidence"):
        if not isinstance(declaration[name], float) or not 0.0 < declaration[name] < 1.0:
            raise ValueError(f"{name} must be a float in (0, 1)")
    replicas = {system["key"]: system["replicas"] for system in declaration["systems"]}
    for plan in declaration["bootstrap"]["plans"]:
        if not set(plan["systems"]) <= set(replicas) or not set(plan["arms"]) <= set(arms):
            raise ValueError("a bootstrap plan names an undeclared system or arm")
        if not _positive_int(plan["draws"]) or plan["draws"] < 2 or not _positive_int(
                plan["replicas"]) or any(plan["replicas"] > replicas[s] for s in plan["systems"]):
            raise ValueError("a bootstrap plan needs at least two draws on declared replicas")
    for reference, arm in declaration["comparison"]["pairs"]:
        if reference not in arms or arm not in arms:
            raise ValueError("a comparison names an undeclared arm")
    criteria = declaration["criteria"]
    if not set(criteria["systems"]) <= set(replicas) or criteria["arm"] not in arms \
            or [criteria["reference"], criteria["arm"]] not in declaration["comparison"]["pairs"]:
        raise ValueError("the criteria name an undeclared system, arm or comparison")
    if not _positive_int(declaration["spot_replay_replicas"]):
        raise ValueError("spot_replay_replicas must be positive")
    return declaration


def target_key(target: float) -> str:
    return repr(float(target))


def schedule(declaration: dict, groups: int) -> tuple[int, list[int]]:
    """``(pilot shots per group, physical-shot budget of each round)`` for a bank."""
    cumulative = declaration["cumulative_shots_per_group"]
    return cumulative[0], [groups * (b - a) for a, b in zip(cumulative, cumulative[1:])]


def looks(declaration: dict) -> int:
    return len(declaration["cumulative_shots_per_group"])


def measurement_seed(declaration: dict, system: int, replica: int) -> int:
    return derived_seed(declaration["seed_root"], system, replica)


def comparison_seed(declaration: dict, system: int, slot: int) -> int:
    return derived_seed(declaration["seed_root"], 1_000_000 + system, slot)


def interval_seed(declaration: dict, system: int, replica: int, arm: int, target: int) -> int:
    return derived_seed(declaration["seed_root"], 2_000_000 + system, replica, 10 * arm + target)


def bootstrap_plan(declaration: dict, system_key: str, arm_key: str,
                   replica: int) -> int | None:
    """The draws an arm's stops get on this replica, or ``None`` for no bootstrap."""
    for plan in declaration["bootstrap"]["plans"]:
        if system_key in plan["systems"] and arm_key in plan["arms"] \
                and replica < plan["replicas"]:
            return plan["draws"]
    return None


# --------------------------------------------------------------------- systems

def _reference_state(spec: dict, model, generators) -> np.ndarray:
    reference = spec["reference"]
    n_spatial, n = spec["n_spatial"], 2 * spec["n_spatial"]
    if reference["kind"] == "random_singlet":
        # A correlated singlet-sector state on interleaved spin orbitals: the
        # bank is not built on a determinant.
        rng = np.random.default_rng(reference["seed"])
        psi = np.zeros(1 << n, complex)
        for a in range(n_spatial):
            for b in range(n_spatial):
                bits = ["0"] * n
                bits[2 * a] = bits[2 * b + 1] = "1"
                psi[int("".join(bits), 2)] += rng.normal() * (
                    1.0 if a == b else reference["off_diagonal"])
        return psi / np.linalg.norm(psi)
    # The CISD state in this pool, from the model's own reference determinant.
    determinant = DeterminantReference.from_program(model.reference)
    cisd = MatrixElementBank(determinant, model.hamiltonian, generators).solve()
    occupied = "".join("1" if q in determinant.occupied else "0" for q in range(n))
    start = np.zeros(1 << n, complex)
    start[int(occupied, 2)] = 1.0
    psi = sum(complex(c) * (to_matrix(g.mv) @ start)
              for c, g in zip(cisd.ritz_vector(0), generators))
    return psi / np.linalg.norm(psi)


def build_system(spec: dict) -> dict:
    """The bank, its certified correction family and its exact facts. No shots."""
    source = spec["fcidump"]
    if "inline" in source:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{spec['key']}.FCIDUMP"
            path.write_text(source["inline"])
            model = fcidump_model(path)
    else:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"{source['path']} does not have its declared digest")
        model = fcidump_model(path)
    n = 2 * spec["n_spatial"]
    occupied = tuple(range(spec["electrons"] // 2))
    generators = [identity_generator(n)] + spin_adapted_excitations(spec["n_spatial"], occupied)
    rho = density_from_statevector(_reference_state(spec, model, generators))
    shared = SharedMeasurement(MatrixElementBank(rho, model.hamiltonian.to_mv(), generators))
    exact = shared.bank.solve()
    family = sector_shift_family(spec["n_spatial"], spec["electrons"], sz=0.0)
    certify_bank_family(shared.bank, family)
    overlap = np.linalg.eigvalsh(shared.exact_matrices()[0])
    design = {"qubits": n, "generators": len(generators),
              "exact_rank": int(exact.effective_rank),
              "target": float(exact.ground_energy),
              "kappa_S": float(overlap.max() / overlap.min()),
              "words": len({word.code for group in shared.groups for word in group}),
              "settings": len(shared.groups), "family_terms": len(family)}
    return {"shared": shared, "family": family, "rho": rho, "design": design,
            "table": shift_pair_table(shared.bank, shared.indices, family)}


def prepare(system: dict) -> None:
    """Compile every setting's outcome distribution once; replicas only reseed."""
    backend = FiniteShotBackend(seed=0)
    backend.prepare_grouped_state(system["shared"].bank.reference, system["shared"].groups)
    system["backend"] = backend


# ------------------------------------------------------------------------ rows

def _bounds(lower: float, upper: float, target: float) -> dict:
    lower, upper = float(lower), float(upper)
    return {"lower": lower, "upper": upper, "covered": bool(lower <= target <= upper)}


def run_schedule(declaration: dict, system: dict, *, system_index: int, replica: int,
                 run: dict, arms: list[dict], backend) -> dict:
    """One seeded run over the whole schedule, with every arm it reports.

    Mirrors ``acquire_ritz``: the same solve, functional, correction (with the
    same pair table), allocation and intervals at each look. At the first look
    where an arm's standard error meets a target, that arm's stop is recorded
    with its intervals and, where the plan asks, its bootstrap of that look's
    data. A look that cannot be solved ends the run; arms still running fail
    there with the shots already spent.
    """
    shared, family, table = system["shared"], system["family"], system["table"]
    spec = declaration["systems"][system_index]
    target = system["design"]["target"]
    pilot, budgets = schedule(declaration, len(shared.groups))
    family_looks = looks(declaration)
    delta = declaration["delta"]
    goals = declaration["target_std_errors"]
    corrected_run = run["control_variates"]
    seed = measurement_seed(declaration, system_index, replica)
    backend.reseed(seed)
    cache = shared.measure(backend, pilot)
    ledger: list[dict] = []
    stops: dict = {arm["key"]: {} for arm in arms}
    failure = None
    stopped_reason = "schedule_exhausted"

    def interval(arm: dict, point, correction, look_count: int) -> tuple[float, float]:
        if arm["corrected"]:
            radius = simultaneous_z_radius(correction.std_error ** 2, delta, look_count)
            return correction.energy - radius, correction.energy + radius
        found = ritz_uncertainty(shared, cache, point, delta=delta, family=look_count)
        return found.lower, found.upper

    def record(arm: dict, goal_index: int, index: int, point, correction, reason: str) -> dict:
        energy = correction.energy if arm["corrected"] else float(point.energies[0])
        sigma = correction.std_error if arm["corrected"] else ledger[index]["raw_std_error"]
        out = {"outcome": "ok", "stopped": reason, "look": index,
               "shots": int(cache.total_shots), "estimate": float(energy),
               "std_error": float(sigma),
               "interval": _bounds(*interval(arm, point, correction, family_looks), target),
               "naive_interval": _bounds(*interval(arm, point, correction, 1), target),
               "bootstrap": None}
        draws = bootstrap_plan(declaration, spec["key"], arm["key"], replica)
        if draws is not None:
            arm_index = [a["key"] for a in declaration["arms"]].index(arm["key"])
            statistic = None
            if arm["corrected"]:
                def statistic(replica_cache, solved):
                    return corrected_ritz_functional(shared, replica_cache, solved, family,
                                                     table=table)[0].energy
            entry: dict = {"draws": draws, "seed": interval_seed(
                declaration, system_index, replica, arm_index, goal_index)}
            try:
                values = bootstrap_ritz_samples(shared, cache, replicates=draws,
                                                seed=entry["seed"], result=point,
                                                statistic=statistic)
            except SOLVE_FAILURES as error:
                entry["failure"] = f"{type(error).__name__}: {error}"
            else:
                found = bootstrap_spread_interval(energy, values, delta=delta,
                                                  family=family_looks)
                entry.update(spread=bootstrap_spread(values),
                             interval=_bounds(found.lower, found.upper, target))
            out["bootstrap"] = entry
        return out

    point = correction = None
    for index in range(len(budgets) + 1):
        try:
            point = shared.solve(cache, **declaration["solver_kwargs"])
            if len(point.energies) < 1:
                raise ValueError("the solve retained no root")
        except SOLVE_FAILURES as error:
            failure = {"look": index, "shots": int(cache.total_shots),
                       "error": f"{type(error).__name__}: {error}"}
            break
        raw = ritz_functional(shared.bank, point.indices, point.ritz_vector(0),
                              point.energies[0])
        retained = point.resources.get("smallest_retained_overlap_eigenvalue")
        entry = {"shots": int(cache.total_shots), "raw_energy": float(point.energies[0]),
                 "raw_std_error": math.sqrt(max(0.0, raw.variance(cache))),
                 "retained_rank": int(point.effective_rank),
                 "smallest_retained_overlap": None if retained is None else float(retained)}
        functional = raw
        if corrected_run:
            correction, corrected = corrected_ritz_functional(shared, cache, point, family,
                                                              table=table)
            entry.update(energy=correction.energy, std_error=correction.std_error,
                         terms_used=len(correction.used))
            if run["allocation"] == "ritz_targeted":
                functional = corrected
        ledger.append(entry)
        for arm in arms:
            sigma = entry["std_error"] if arm["corrected"] else entry["raw_std_error"]
            for goal_index, goal in enumerate(goals):
                key = target_key(goal)
                if key not in stops[arm["key"]] and sigma <= goal:
                    stops[arm["key"]][key] = record(arm, goal_index, index, point, correction,
                                                    "target_met")
        if index == len(budgets):
            break
        if run["allocation"] == "uniform":
            plan = uniform_group_plan(shared.groups, budgets[index])
        else:
            plan = neyman_group_plan(shared.groups, cache, functional, budgets[index])
        if not plan:
            stopped_reason = "deterministic"
            break
        shared.measure_plan(backend, plan, cache)

    fixed = {}
    for arm in arms:
        for goal_index, goal in enumerate(goals):
            key = target_key(goal)
            if key in stops[arm["key"]]:
                continue
            if failure is not None:
                stops[arm["key"]][key] = {"outcome": "solve_failed", **failure}
            else:
                stops[arm["key"]][key] = record(arm, goal_index, len(ledger) - 1, point,
                                                correction, stopped_reason)
        if failure is None:
            low, high = interval(arm, point, correction, 1)
            energy = correction.energy if arm["corrected"] else float(point.energies[0])
            fixed[arm["key"]] = {"outcome": "ok", "look": len(ledger) - 1,
                                 "estimate": float(energy),
                                 "interval": _bounds(low, high, target)}
        else:
            fixed[arm["key"]] = {"outcome": "solve_failed", **failure}
    return {"ledger": ledger, "failure": failure, "stops": stops, "fixed": fixed}


def replica_row(declaration: dict, system: dict, system_index: int, replica: int,
                backend=None) -> dict:
    """Every run and arm of one replica of one system."""
    spec = declaration["systems"][system_index]
    backend = backend if backend is not None else system["backend"]
    row: dict = {"system": spec["key"], "replica": replica,
                 "measurement_seed": measurement_seed(declaration, system_index, replica),
                 "runs": {}, "arms": {}}
    for run in declaration["runs"]:
        arms = [arm for arm in declaration["arms"] if arm["run"] == run["key"]]
        found = run_schedule(declaration, system, system_index=system_index, replica=replica,
                             run=run, arms=arms, backend=backend)
        row["runs"][run["key"]] = {"ledger": found["ledger"], "failure": found["failure"]}
        for arm in arms:
            row["arms"][arm["key"]] = {"fixed": found["fixed"][arm["key"]],
                                       "stopping": found["stops"][arm["key"]]}
    return row


# ------------------------------------------------------------------- summaries

def _errors(values: list[float]) -> dict | None:
    if not values:
        return None
    errors = np.asarray(values, dtype=float)
    return {"n": len(values), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
            "median_abs_error": float(np.median(np.abs(errors))),
            "max_abs_error": float(np.max(np.abs(errors)))}


def summarize_arm(members: list[dict], arm: str, declaration: dict, design: dict) -> dict:
    """Cost, error and coverage of one arm over every replica, failures included."""
    confidence, nominal = declaration["summary_confidence"], 1.0 - declaration["delta"]
    target = design["target"]
    fixed = [row["arms"][arm]["fixed"] for row in members]
    out: dict = {"replicas": len(members), "fixed": {
        "errors": _errors([run["estimate"] - target for run in fixed if run["outcome"] == "ok"]),
        "coverage": _proportion(sum(run["outcome"] == "ok" and run["interval"]["covered"]
                                    for run in fixed), len(fixed), confidence, nominal)},
        "stopping": {}}
    for goal in declaration["target_std_errors"]:
        runs = [row["arms"][arm]["stopping"][target_key(goal)] for row in members]
        ok = [run for run in runs if run["outcome"] == "ok"]
        shots = [run["shots"] for run in runs]
        cell = {
            "solve_failures": len(runs) - len(ok),
            "stopped": {reason: sum(run["stopped"] == reason for run in ok)
                        for reason in STOP_REASONS},
            "mean_shots": float(np.mean(shots)),
            "mean_shots_per_setting": float(np.mean(shots)) / design["settings"],
            "errors": _errors([run["estimate"] - target for run in ok]),
            "rms_std_error": (float(np.sqrt(np.mean([run["std_error"] ** 2 for run in ok])))
                              if ok else None),
            "coverage": {
                "schedule_valid": _proportion(sum(run["interval"]["covered"] for run in ok),
                                              len(runs), confidence, nominal),
                "naive": _proportion(sum(run["naive_interval"]["covered"] for run in ok),
                                     len(runs), confidence, nominal)},
            "bootstrap": None}
        # Only the replicas the plan bootstraps; a failure of either kind is a miss.
        planned = [run for row, run in zip(members, runs)
                   if bootstrap_plan(declaration, row["system"], arm, row["replica"])]
        if planned:
            covered = sum(run["outcome"] == "ok" and "interval" in run["bootstrap"]
                          and run["bootstrap"]["interval"]["covered"] for run in planned)
            spreads = [run["bootstrap"]["spread"] / run["std_error"] for run in planned
                       if run["outcome"] == "ok" and "spread" in run["bootstrap"]
                       and run["std_error"] > 0]
            cell["bootstrap"] = {
                "coverage": _proportion(covered, len(planned), confidence, nominal),
                "failures": sum(run["outcome"] == "ok" and "failure" in run["bootstrap"]
                                for run in planned),
                "median_spread_over_std_error": (float(np.median(spreads))
                                                 if spreads else None)}
        out["stopping"][target_key(goal)] = cell
    return out


def _direction(found: dict) -> str:
    """At equal shots: whether the reference-over-arm MSE ratio is resolved from one."""
    if found["interval"] is None:
        return "indistinguishable"
    lower, upper = found["interval"]
    return "arm_better" if lower > 1.0 else "reference_better" if upper < 1.0 \
        else "indistinguishable"


def _verdict(found: dict) -> str:
    if found["interval"] is None:
        return "no_resolved_difference"
    lower, upper = found["interval"]
    return ("fewer_shots_at_matched_error" if lower > 1.0 else
            "more_shots_at_matched_error" if upper < 1.0 else "no_resolved_difference")


def compare(members: list[dict], declaration: dict, design: dict, system: int) -> dict:
    """Each declared pair, reference over arm, paired by replica.

    At equal shots: the ratio of mean-squared errors at the last look. At
    matched error under a stopping target: the ratio of work-normalized
    mean-squared errors, ``mean shots x MSE at stop``, as in the
    target-acquisition study. Replicas on which either arm failed are left out
    of the error ratios and counted.
    """
    resample = {"replicates": declaration["comparison"]["bootstrap_replicates"],
                "confidence": declaration["summary_confidence"]}
    target = design["target"]
    out = {}
    for pair_index, (reference, arm) in enumerate(declaration["comparison"]["pairs"]):
        slot = 10 * pair_index

        def paired(select):
            both = [row for row in members
                    if all(select(row, name)["outcome"] == "ok" for name in (reference, arm))]
            return both, {name: np.array([(select(row, name)["estimate"] - target) ** 2
                                          for row in both]) for name in (reference, arm)}

        both, squared = paired(lambda row, name: row["arms"][name]["fixed"])
        budget = _ratio_interval([squared[reference]], [squared[arm]],
                                 seed=comparison_seed(declaration, system, slot), **resample)
        budget["label"] = _direction(budget)
        stopping = {}
        for goal_index, goal in enumerate(declaration["target_std_errors"], start=1):
            key = target_key(goal)

            def select(row, name, key=key):
                return row["arms"][name]["stopping"][key]
            both, squared = paired(select)
            spent = {name: np.array([float(select(row, name)["shots"]) for row in both])
                     for name in (reference, arm)}
            matched = _ratio_interval([spent[reference], squared[reference]],
                                      [spent[arm], squared[arm]],
                                      seed=comparison_seed(declaration, system,
                                                           slot + goal_index), **resample)
            every = {name: float(np.mean([select(row, name)["shots"] for row in members]))
                     for name in (reference, arm)}
            stopping[key] = {"matched_error": matched, "verdict": _verdict(matched),
                             "shot_ratio": every[reference] / every[arm],
                             "excluded_replicas": len(members) - len(both)}
        out[f"{reference}/{arm}"] = {"reference": reference, "arm": arm,
                                     "equal_shots": budget, "stopping": stopping}
    return out


def criteria(declaration: dict, systems: list[dict]) -> dict:
    """C1 to C3 on the held-out systems, as declared."""
    declared = declaration["criteria"]
    pair = f"{declared['reference']}/{declared['arm']}"
    by_key = {entry["system"]: entry for entry in systems}
    cells = []
    for system in declared["systems"]:
        entry = by_key[system]
        for goal in declaration["target_std_errors"]:
            key = target_key(goal)
            arm = entry["arms"][declared["arm"]]["stopping"][key]
            cells.append({"system": system, "target": key,
                          "verdict": entry["comparisons"][pair]["stopping"][key]["verdict"],
                          "delta_label": arm["coverage"]["schedule_valid"]["label"],
                          "bootstrap_label": (arm["bootstrap"]["coverage"]["label"]
                                              if arm["bootstrap"] else None)})
    return {"cells": cells,
            "C1": all(cell["verdict"] == "fewer_shots_at_matched_error" for cell in cells),
            "C2": all(cell["delta_label"] != "under" for cell in cells),
            "C3": all(cell["bootstrap_label"] not in ("under", None) for cell in cells)}


def summarize(declaration: dict, rows: list[dict], designs: dict) -> dict:
    systems = []
    for index, spec in enumerate(declaration["systems"]):
        members = sorted((row for row in rows if row["system"] == spec["key"]),
                         key=lambda row: row["replica"])
        design = designs[spec["key"]]
        systems.append({
            "system": spec["key"], "role": spec["role"],
            "arms": {arm["key"]: summarize_arm(members, arm["key"], declaration, design)
                     for arm in declaration["arms"]},
            "comparisons": compare(members, declaration, design, index)})
    return {"summaries": systems, "criteria": criteria(declaration, systems)}


# ------------------------------------------------------------------- execution

_SYSTEMS: dict = {}


def _work(task: tuple) -> dict:
    declaration, index, replica = task
    return replica_row(declaration, _SYSTEMS[declaration["systems"][index]["key"]],
                       index, replica)


def expected_tasks(declaration: dict) -> list[tuple[int, int]]:
    return [(index, replica) for index, spec in enumerate(declaration["systems"])
            for replica in range(spec["replicas"])]


def read_rows(path: Path) -> dict:
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[(row["system"], row["replica"])] = row
    return rows


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_record(declaration_path: Path, rows: list[dict], designs: dict, *,
                 declaration_commit: str) -> dict:
    declaration = load_declaration(declaration_path)
    return {"schema": RESULTS_SCHEMA, "declaration": declaration,
            "declaration_sha256": sha256(declaration_path),
            "declaration_commit": declaration_commit,
            "implementation": implementation_digests(), "systems": designs,
            "rows": rows, **summarize(declaration, rows, designs)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, default=DECLARATION)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.output.exists():
        print(f"FAIL {args.output} exists; a declared record is never overwritten")
        return 1
    declaration = load_declaration(args.declaration)
    relative = args.declaration.resolve().relative_to(ROOT).as_posix()
    if _git("status", "--porcelain", "--", relative, *IMPLEMENTATION):
        print("FAIL the declaration and implementation must be committed before the run")
        return 1
    commit = _git("log", "-1", "--format=%H", "--", relative)
    args.work.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows(args.work)
    todo = [(index, replica) for index, replica in expected_tasks(declaration)
            if (declaration["systems"][index]["key"], replica) not in done]
    started = time.time()
    # Build and compile every system once, before the workers fork, so each
    # worker inherits the compiled outcome distributions instead of repaying them.
    for index in sorted({index for index, _ in todo} | set(range(len(declaration["systems"])))):
        spec = declaration["systems"][index]
        _SYSTEMS[spec["key"]] = build_system(spec)
        if any(task[0] == index for task in todo):
            prepare(_SYSTEMS[spec["key"]])
        print(f"built {spec['key']} ({time.time() - started:.0f}s)", flush=True)
    # Longest first: the eight-qubit banks with bootstraps dominate.
    todo.sort(key=lambda task: (declaration["systems"][task[0]]["key"] == "tri",
                                -(bootstrap_plan(declaration,
                                                 declaration["systems"][task[0]]["key"],
                                                 "targeted_cv", task[1]) or 0), task))
    tasks = [(declaration, index, replica) for index, replica in todo]
    context = multiprocessing.get_context("fork")
    with open(args.work, "a", encoding="utf-8") as handle, \
            ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        pending = [pool.submit(_work, task) for task in tasks]
        for count, future in enumerate(as_completed(pending), start=1):
            # Written as each replica finishes, so an interruption loses only
            # the replicas still running.
            handle.write(json.dumps(future.result(), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            if count % 10 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)} rows ({time.time() - started:.0f}s)", flush=True)
    rows = read_rows(args.work)
    missing = [task for task in expected_tasks(declaration)
               if (declaration["systems"][task[0]]["key"], task[1]) not in rows]
    if missing:
        print(f"FAIL {len(missing)} declared replicas are missing from {args.work}")
        return 1
    ordered = [rows[(declaration["systems"][index]["key"], replica)]
               for index, replica in expected_tasks(declaration)]
    designs = {key: system["design"] for key, system in _SYSTEMS.items()}
    record = stamp_record(build_record(args.declaration, ordered, designs,
                                       declaration_commit=commit))
    write_record(record, args.output)
    print(f"wrote {args.output} ({len(ordered)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
