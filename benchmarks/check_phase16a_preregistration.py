"""Gate the result-free Phase 16A preregistration: time-evolved QSCI vs selection.

The declaration asks one question per required instance. Does QSCI on
configurations pooled from Trotter circuits at a frozen time grid reach
chemical accuracy within a frozen shot grid? And at the first budget where it
does, is its determinant set better than the one classical selected CI picks at
the same size without seeing the sample? This gate checks that the declaration
can answer that question as frozen, and that it was frozen before any answer
existed:

* completeness, and the absence of any result-shaped value;
* the SHA-256 of every input the run will read: FCIDUMPs, their provenance, and
  the 16A implementation;
* an exhaustive, mutually exclusive status ladder and a total combination rule
  in which no censored required instance can yield GO;
* the declaration's clauses tested against each other, the lesson of Phase
  16B v3, whose frozen continuity rule its own setting model made
  unreachable. The shot grid must split evenly over the time grid, the
  admission budget must be the grid's top, the tie tolerance must sit far
  below the target, the arms must be partitioned by category and role, and
  every required instance must carry the target's units;
* a recomputation, from the committed inputs alone, of every number in
  ``measured_before_freezing``. That covers the time grid, the Trotter step
  counts the declared rule selects, and the admission criterion: the
  reference misses the target, the candidate's expected pooled support at the
  top budget reaches it in exact arithmetic, and that support leaves part of
  the sector unselected. Each threshold must be cleared by a margin a platform
  cannot flip;
* commit order, from git history, once a record exists.

It draws no sample, solves no sampled subspace, runs no control, and computes
no status or verdict.

    python benchmarks/check_phase16a_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

CONFIG = HERE / "configs" / "phase16a_te_qsci.json"
RECORD = HERE / "reference_results" / "phase16a_te_qsci.json"
SCHEMA = "clifford_qc.phase16a_te_qsci_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "purpose", "contains_results", "claim_boundary",
    "question", "target", "instances", "required_decision_instances",
    "screened_and_not_admitted", "exploratory_disclosure", "time_grid", "trotter",
    "shot_grid", "replicas", "seeds", "arms", "primary_candidate_arm",
    "primary_control_arm", "admission_criterion", "robustness",
    "measured_before_freezing", "decision_rule", "evidence", "excluded_from_cost",
    "prespecified_followup", "implementation_lineage", "record_requirements",
)
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "status_observed",
    "instance_status", "instance_statuses", "median_error", "p90_error",
    "shots_to_target_observed", "paired_difference_observed", "conclusion",
    "elapsed_seconds", "sampled_configurations", "qsci_error",
})
STATUSES = ("PASS", "FAIL", "UNDETERMINED")
CATEGORIES = {"implementable", "oracle", "classical"}
UNTENSED = ("has been", "was run", "were run", "no sampling has been performed")


# ------------------------------------------------------------------ loading

def _walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def load_config(path: Path = CONFIG) -> dict:
    """Read the declaration, refusing one that carries a result."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for where, key, value in _walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            raise ValueError(f"config carries a result field {where}.{key}")
    if config.get("contains_results") is not False:
        raise ValueError("contains_results must be literally false")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ static

def status_of(reaches: bool, beats: bool) -> str:
    """The frozen ladder, as a function of its two predicates."""
    if not reaches:
        return "UNDETERMINED"
    return "PASS" if beats else "FAIL"


def verdict_of(statuses) -> str:
    statuses = tuple(statuses)
    if "INVALID" in statuses:
        return "INVALID"
    if all(value == "PASS" for value in statuses):
        return "GO"
    if all(value == "FAIL" for value in statuses):
        return "NO_GO"
    return "CONDITIONAL"


def ladder_problems(config: dict) -> list[str]:
    problems = []
    rule = config["decision_rule"]
    order = [row["status"] for row in rule["instance_status_order"]]
    if order != list(STATUSES):
        problems.append(f"status ladder order moved: {order}")
    reached = {status_of(r, b) for r, b in itertools.product((False, True), repeat=2)}
    if reached != set(STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable {set(STATUSES) - reached}")
    count = len(config["required_decision_instances"])
    if count < 1:
        return problems + ["there must be at least one required decision instance"]
    tuples = list(itertools.product(STATUSES, repeat=count))
    if verdict_of(("PASS",) * count) != "GO" or verdict_of(("FAIL",) * count) != "NO_GO":
        problems.append("all-PASS must give GO and all-FAIL NO_GO")
    for tuple_ in tuples:
        if "UNDETERMINED" in tuple_ and verdict_of(tuple_) == "GO":
            problems.append(f"{tuple_} gives GO despite a censored required instance")
    if rule.get("combination_rule") != (
            "all required PASS gives GO; all FAIL gives NO_GO; anything else is "
            "CONDITIONAL"):
        problems.append("combination_rule text moved from the rule this gate enforces")
    return problems


def reference_problems(config: dict) -> list[str]:
    problems = []
    instances, arms = config["instances"], config["arms"]
    required = config["required_decision_instances"]
    for name in required:
        if name not in instances:
            problems.append(f"required instance {name} is not declared")
        elif instances[name].get("role") != "required_decision":
            problems.append(f"required instance {name} is not declared with that role")
    overlap = set(required) & set(config["screened_and_not_admitted"])
    if overlap:
        problems.append(f"an instance is both required and screened out: {sorted(overlap)}")
    for key in ("primary_candidate_arm", "primary_control_arm"):
        if config[key] not in arms:
            problems.append(f"{key} names an undeclared arm")
    candidate = arms.get(config["primary_candidate_arm"], {})
    control = arms.get(config["primary_control_arm"], {})
    if candidate.get("category") != "implementable":
        problems.append("the primary candidate must be an implementable input")
    if control.get("category") != "classical":
        problems.append("the primary control must be classical: it consumes no "
                        "sampling input, which is what lets it sit beside an "
                        "implementable one without breaking the §8D split")
    for name, arm in arms.items():
        if arm.get("category") not in CATEGORIES:
            problems.append(f"arm {name} has category {arm.get('category')!r}")
    blocked = set(config["decision_rule"]["diagnostics_cannot_promote"])
    promoters = {config["primary_candidate_arm"], config["primary_control_arm"]}
    if promoters & blocked:
        problems.append(f"a primary arm is listed as non-promoting: {sorted(promoters & blocked)}")
    oracle_arms = {name for name, arm in arms.items() if arm.get("category") == "oracle"}
    if not oracle_arms <= blocked:
        problems.append(f"oracle arms must never promote: {sorted(oracle_arms - blocked)}")
    unaccounted = set(arms) - promoters - blocked
    if unaccounted:
        problems.append(f"arms with no declared role in the decision: {sorted(unaccounted)}")
    return problems


def consistency_problems(config: dict) -> list[str]:
    """The declaration's clauses against each other (the Phase 16B v3 lesson)."""
    problems = []
    multipliers = config["time_grid"]["multipliers"]
    count = len(multipliers)
    budgets = config["shot_grid"]["total_shots"]
    if list(budgets) != sorted(set(budgets)):
        problems.append("the shot grid must be strictly increasing")
    if any(b % count for b in budgets):
        problems.append(f"every total budget must split evenly over the {count} times")
    if any(b & (b - 1) for b in budgets):
        problems.append("the shot grid must be powers of two")
    admission = config["admission_criterion"]
    if admission.get("top_budget") != max(budgets):
        problems.append("admission must be read at the top of the shot grid, "
                        f"{max(budgets)}, not {admission.get('top_budget')}")
    target = float(config["target"]["value"])
    tie = float(config["decision_rule"]["tie_tolerance"])
    if not 0.0 < tie <= target * 1e-4:
        problems.append("the tie tolerance must be positive and at most 1e-4 of the target")
    trotter = config["trotter"]
    steps = trotter["step_grid"]
    if steps != [2 ** k for k in range(len(steps))]:
        problems.append("the Trotter step grid must be 1, 2, 4, ... in order")
    if not 0.0 < float(trotter["min_fidelity"]) < 1.0:
        problems.append("min_fidelity must lie strictly inside (0, 1)")
    if trotter.get("order") not in (1, 2):
        problems.append("trotter order must be 1 or 2")
    if multipliers != sorted(set(multipliers)) or min(multipliers) <= 0:
        problems.append("time multipliers must be positive and strictly increasing")
    if not isinstance(config["replicas"], int) or config["replicas"] < 2:
        problems.append("replicas must be an integer of at least two")
    seeds = config["seeds"]
    sampled = {name for name, arm in config["arms"].items() if arm.get("sampled")}
    indices = seeds["arm_indices"]
    if set(indices) != sampled:
        problems.append(f"seed streams must cover exactly the sampled arms {sorted(sampled)}")
    if len(set(indices.values())) != len(indices):
        problems.append("two sampled arms share a seed stream")
    if seeds["instance_order"] != config["required_decision_instances"]:
        problems.append("the seed instance order must be the required-instance order")
    units = config["target"]["units"]
    for name in config["required_decision_instances"]:
        if config["instances"][name].get("energy_unit") != units:
            problems.append(f"{name} is not declared in the target's units ({units})")
    return problems


def claim_problems(config: dict) -> list[str]:
    problems = []
    boundary = config["claim_boundary"]
    for phrase in UNTENSED:
        if phrase in boundary:
            problems.append(f"claim boundary is tensed ({phrase!r}); a record that "
                            "inherits it would contradict its own sampling")
    record = config["record_requirements"]
    if record.get("must_carry", {}).get("quantum_advantage_claim") is not False:
        problems.append("records must carry quantum_advantage_claim: false")
    if config["prespecified_followup"].get("permitted") != "none":
        problems.append("this declaration permits no follow-up; say so literally")
    return problems


def lineage_problems(config: dict) -> list[str]:
    problems = []
    for row in config["implementation_lineage"]["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            problems.append(f"lineage file missing: {row['path']}")
        elif _sha256(path) != row["sha256"]:
            problems.append(f"declared sha256 drifted for {row['path']}")
    for name, spec in config["instances"].items():
        for key in ("source", "provenance"):
            if key in spec and not (ROOT / spec[key]).is_file():
                problems.append(f"{name}: {key} {spec[key]} is missing")
    declared = {row["path"] for row in config["implementation_lineage"]["files"]}
    for name in config["required_decision_instances"]:
        spec = config["instances"][name]
        for key in ("source", "provenance"):
            if spec.get(key) not in declared:
                problems.append(f"{name}: {key} is not bound in implementation_lineage")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    missing = [key for key in REQUIRED_TOP if key not in config]
    if missing:
        return problems + [f"missing required keys {missing}"]
    for check in (ladder_problems, reference_problems, consistency_problems,
                  claim_problems, lineage_problems):
        problems.extend(check(config))
    return problems


# ------------------------------------------------------------------ structural

def build_instance(spec: dict):
    """The model a required instance names, from committed inputs only."""
    from clifford_qc.models.fcidump import fcidump_model

    if spec.get("builder") != "clifford_qc.models.fcidump.fcidump_model":
        raise ValueError(f"unsupported builder {spec.get('builder')!r}")
    return fcidump_model(ROOT / spec["source"])


def structural_quantities(config: dict, spec: dict) -> dict:
    """Every frozen number of one instance, recomputed from its inputs.

    The candidate's pooled distribution is the equal-shot mixture of its
    Trotter states, so the expected number of draws of configuration ``c``
    over the pooled top budget is ``N_max * mean_k p_k(c)``, counting raw
    shots before post-selection. That is the support the admission criterion
    reads.
    """
    import numpy as np

    from clifford_qc.backends import SectorStatevectorBackend
    from clifford_qc.pauli_action import PauliLinearOperator
    from clifford_qc.subspace.time_evolution import (
        apply_program, propagate, trotter_program,
    )
    from clifford_qc.subspace.qsci import _determinant_amplitudes

    model = build_instance(spec)
    metadata = model.metadata
    backend = SectorStatevectorBackend(model.n, metadata["n_electrons"],
                                       metadata["sz"],
                                       spin_ordering=metadata["spin_convention"])
    operator = backend.operator(model.hamiltonian)
    dense = np.column_stack([operator.matvec(column)
                             for column in np.eye(backend.dimension)])
    exact = float(np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))[0])
    reference = backend.state_from_program(model.reference)
    mean = float(operator.expectation(reference).real)
    applied = operator.matvec(reference)
    sigma = math.sqrt(max(float(np.vdot(applied, applied).real) - mean ** 2, 0.0))
    tau = 1.0 / sigma

    trotter = config["trotter"]
    threshold = float(trotter["min_fidelity"])
    full = PauliLinearOperator(model.hamiltonian)
    reference_full = _determinant_amplitudes(model.reference, model.n)
    mixture = np.zeros(2 ** model.n)
    times, steps, rotors, fidelities, margins = [], [], [], [], []
    for multiplier in config["time_grid"]["multipliers"]:
        time = multiplier * tau
        exact_state = propagate(full, reference_full, [time])[0]
        chosen = None
        for count in trotter["step_grid"]:
            program = trotter_program(model.hamiltonian, time, steps=count,
                                      order=trotter["order"])
            state = apply_program(program, reference_full)
            fidelity = float(abs(np.vdot(exact_state, state)) ** 2
                             / (np.vdot(state, state).real
                                * np.vdot(exact_state, exact_state).real))
            if fidelity >= threshold:
                chosen = (count, len(program.ops), fidelity, state)
                break
        if chosen is None:
            raise ValueError(f"no step count in {trotter['step_grid']} reaches "
                             f"fidelity {threshold} at t = {time}")
        count, ops, fidelity, state = chosen
        weights = np.abs(state) ** 2
        mixture += weights / weights.sum()
        times.append(time)
        steps.append(count)
        rotors.append(ops)
        fidelities.append(fidelity)
        margins.append(fidelity - threshold)
    mixture /= len(times)

    top = float(config["admission_criterion"]["top_budget"])
    expected = top * mixture[backend.basis]
    support = np.flatnonzero(expected >= 1.0)
    block = dense[np.ix_(support, support)]
    support_error = float(np.linalg.eigvalsh(0.5 * (block + block.conj().T))[0]) - exact
    live = expected[expected > 0.0]
    return {
        "n_qubits": int(model.n),
        "sector_dimension": int(backend.dimension),
        "reference_error": mean - exact,
        "sigma_ref": sigma,
        "tau": tau,
        "times": times,
        "trotter_steps": steps,
        "rotor_counts": rotors,
        "trotter_fidelities": fidelities,
        "minimum_fidelity_margin": float(min(margins)),
        "expected_support_at_top_budget": int(support.size),
        "support_zero_noise_error": support_error,
        "minimum_support_margin": float(np.min(np.abs(live - 1.0))) if live.size else 1.0,
        "energy_unit": metadata.get("energy_unit"),
    }


_INT_FIELDS = ("n_qubits", "sector_dimension", "expected_support_at_top_budget")
_LIST_INT_FIELDS = ("trotter_steps", "rotor_counts")
_FLOAT_FIELDS = ("reference_error", "sigma_ref", "tau", "support_zero_noise_error")


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)


def admission_problems(config: dict, notes: list[str] | None = None) -> list[str]:
    """Recompute and compare every frozen number, then apply the criterion."""
    notes = [] if notes is None else notes
    problems = []
    target = float(config["target"]["value"])
    robust = config["robustness"]
    declared_all = config["measured_before_freezing"]
    for name in config["required_decision_instances"]:
        spec = config["instances"][name]
        declared = declared_all.get(name)
        if declared is None:
            problems.append(f"{name}: no measured_before_freezing entry")
            continue
        got = structural_quantities(config, spec)
        for key in _INT_FIELDS + _LIST_INT_FIELDS:
            if declared.get(key) != got[key]:
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        for key in _FLOAT_FIELDS:
            if not _close(float(declared.get(key, float("nan"))), got[key]):
                problems.append(f"{name}: {key} declared {declared.get(key)!r}, "
                                f"recomputed {got[key]!r}")
        if len(declared.get("times", ())) != len(got["times"]) or not all(
                _close(a, b) for a, b in zip(declared.get("times", ()), got["times"])):
            problems.append(f"{name}: times declared {declared.get('times')!r}, "
                            f"recomputed {got['times']!r}")
        if got["energy_unit"] != config["target"]["units"].lower():
            problems.append(f"{name}: model energy unit {got['energy_unit']!r} is not "
                            "the target's")
        if got["reference_error"] <= target:
            problems.append(f"{name}: ADMISSION FAILS -- the reference determinant is "
                            f"already within the target ({got['reference_error']:.3e})")
        if got["support_zero_noise_error"] > target:
            problems.append(
                f"{name}: ADMISSION FAILS -- the candidate's expected pooled support at "
                f"{config['admission_criterion']['top_budget']} shots reaches only "
                f"{got['support_zero_noise_error']:.3e} in exact arithmetic, so the "
                "comparison would be censored by construction")
        if got["expected_support_at_top_budget"] >= got["sector_dimension"]:
            problems.append(f"{name}: ADMISSION FAILS -- the expected support covers the "
                            "whole sector, so matched selection selects nothing")
        if got["minimum_fidelity_margin"] < float(robust["min_fidelity_margin"]):
            problems.append(f"{name}: a Trotter step choice sits within "
                            f"{got['minimum_fidelity_margin']:.2e} of its threshold")
        if got["minimum_support_margin"] < float(robust["min_support_margin"]):
            problems.append(f"{name}: an expected count sits within "
                            f"{got['minimum_support_margin']:.2e} of the support cutoff")
        notes.append(
            f"  {name}: admits -- sector {got['sector_dimension']}, reference "
            f"{got['reference_error']:.3e}, support {got['expected_support_at_top_budget']} "
            f"at {got['support_zero_noise_error']:.1e}, steps {got['trotter_steps']}")
    return problems


# ------------------------------------------------------------------ git

def _first_commit(path: Path):
    try:
        out = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", str(path)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commits = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return commits[-1] if commits else None


def commit_order_problems(notes: list[str]) -> list[str]:
    if not RECORD.exists():
        notes.append("  commit order: no record yet, nothing to order against")
        return []
    config_commit, record_commit = _first_commit(CONFIG), _first_commit(RECORD)
    if config_commit is None or record_commit is None:
        notes.append("  commit order: SKIP (not committed yet, or no git history)")
        return []
    if config_commit == record_commit:
        return ["the config and the record entered the repository in the same commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config_commit, record_commit],
        cwd=ROOT, capture_output=True, timeout=30)
    if ancestor.returncode != 0:
        return [f"config commit {config_commit[:12]} is not an ancestor of record "
                f"commit {record_commit[:12]}"]
    notes.append(f"  commit order: config {config_commit[:12]} precedes record "
                 f"{record_commit[:12]}")
    return []


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    notes: list[str] = []
    problems = static_problems(config)
    if not problems:
        problems.extend(admission_problems(config, notes))
        problems.extend(commit_order_problems(notes))
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print("OK Phase 16A preregistration is complete, result-free, jointly "
          "satisfiable, and admissible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
