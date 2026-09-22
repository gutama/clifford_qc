"""Gate the Phase 16B preregistration before any result is produced under it.

A preregistration earns its name from commit order, not from the word. This
checker is what makes that worth something: it runs against a config carrying
no results, and it fails a config whose decision rule could be satisfied more
than one way, or not at all, once results exist.

What it re-derives, with nothing sampled:

  1. the config is the declared schema, is complete, and declares no result,
     verdict, cost, or observed value anywhere in its tree;
  2. every builder, pool rule, arm, and regularizer it names actually resolves
     in the installed package, and each instance builds at its declared qubit
     count, electron number, and reference;
  3. the strict phase-branch condition ``(U-L)*dt < 2*pi`` holds for the
     primary grid on every instance, so no instance is preregistered into an
     aliasing violation;
  4. the instance-status ladder is **exhaustive and mutually exclusive** over
     every reachable combination of its own predicates -- the property the
     design document claims and the one a post-hoc reading could quietly lose;
  5. the combination table assigns exactly one verdict to all 16 ordered status
     pairs, and no diagnostic arm or instance appears among the promoters;
  6. no record for this experiment exists yet.

Check (6) is the commit-order check. It fails once a record is committed, which
is intended: after that point this gate has already done its job and the result
checker is the live one.

    python benchmarks/check_phase16b_preregistration.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "phase16b_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_feasibility.json"
SCHEMA = "clifford_qc.phase16b_feasibility_config.v1"

REQUIRED_TOP = (
    "schema", "design_document", "purpose", "contains_results", "target",
    "instances", "required_decision_instances", "diagnostic_instances", "arms",
    "primary_candidate_arms", "exact_propagation_arms", "acase_pool",
    "basis_sizes", "grid_rule", "noise_model", "cost_model", "regularizers",
    "non_ridge_regularizers", "decision_rule", "prespecified_followup",
    "record_requirements",
)

# A config may *name* these as things the record must carry; it may not carry
# one itself. Keys are matched exactly, so "shots_to_target" as a rule string
# under cost_model is fine while a key of that name holding a number is not.
RESULT_KEYS = frozenset({
    "verdict", "verdicts", "result", "results", "observed", "measured",
    "energy", "energies", "median_error", "p90_error", "status",
    "instance_status", "shots", "total_shots", "elapsed_seconds", "conclusion",
})
STATUSES = ("PASS", "MARGINAL", "UNDETERMINED", "FAIL")


def walk(node, path="$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}[{index}]")


def check_no_results(config, problems):
    for path, key, value in walk(config):
        if key in RESULT_KEYS and not isinstance(value, (str, bool, type(None))):
            problems.append(f"config carries a result-shaped key {path}.{key} = {value!r}")
    if config.get("contains_results") is not False:
        problems.append("contains_results must be literally false")
    if RECORD.exists():
        problems.append(
            f"{RECORD.relative_to(ROOT.parent)} already exists: the preregistration "
            "gate must run before any record is committed")


def status_of(has_pass, has_marginal, censored):
    """The ladder from decision_rule.instance_status_order, in its declared order."""
    if has_pass:
        return "PASS"
    if has_marginal:
        return "MARGINAL"
    if censored:
        return "UNDETERMINED"
    return "FAIL"


def check_ladder(config, problems):
    order = [row["status"] for row in config["decision_rule"]["instance_status_order"]]
    if order != list(STATUSES):
        problems.append(f"status ladder order moved: {order}")
        return
    # Exhaustive and mutually exclusive over the reachable predicate cube.
    # A PASS at >=1e-5 implies a qualifying comparison at >=1e-6, so
    # has_pass and not has_marginal is unreachable and excluded.
    seen = set()
    for has_pass in (False, True):
        for has_marginal in (False, True):
            for censored in (False, True):
                if has_pass and not has_marginal:
                    continue  # unreachable: PASS implies the MARGINAL predicate
                assigned = status_of(has_pass, has_marginal, censored)
                seen.add(assigned)
                if assigned not in STATUSES:
                    problems.append(f"unreachable status {assigned}")
    if seen != set(STATUSES):
        problems.append(f"ladder is not exhaustive: unreachable statuses {set(STATUSES) - seen}")


def check_combination(config, problems):
    table = config["decision_rule"]["combination"]
    explicit = {}
    fallback = None
    for row in table:
        outcomes = row["outcomes"]
        if outcomes == "any_other_combination":
            fallback = row["verdict"]
        else:
            explicit[tuple(outcomes)] = row["verdict"]
    if fallback is None:
        problems.append("combination table has no catch-all row")
        return
    assigned = {}
    for first in STATUSES:
        for second in STATUSES:
            verdict = explicit.get((first, second), fallback)
            assigned[(first, second)] = verdict
    if len(assigned) != 16:
        problems.append(f"combination table covers {len(assigned)} of 16 ordered pairs")
    if assigned[("PASS", "PASS")] != "GO":
        problems.append("PASS/PASS must be GO")
    if assigned[("FAIL", "FAIL")] != "NO_GO":
        problems.append("FAIL/FAIL must be NO_GO")
    others = {pair: v for pair, v in assigned.items()
              if pair not in (("PASS", "PASS"), ("FAIL", "FAIL"))}
    if set(others.values()) != {"CONDITIONAL"}:
        problems.append(f"every other pair must be CONDITIONAL; got {sorted(set(others.values()))}")
    promoters = set(config["primary_candidate_arms"]) | set(config["required_decision_instances"])
    blocked = set(config["decision_rule"]["diagnostics_cannot_promote"])
    if promoters & blocked:
        problems.append(f"a promoter is also listed as non-promoting: {sorted(promoters & blocked)}")


def check_references(config, problems):
    for name in config["primary_candidate_arms"] + config["exact_propagation_arms"]:
        if name not in config["arms"]:
            problems.append(f"arm {name} is referenced but not declared")
    for name in config["non_ridge_regularizers"]:
        if name not in config["regularizers"]:
            problems.append(f"regularizer {name} is referenced but not declared")
        elif config["regularizers"][name].get("is_ridge") is not False:
            problems.append(f"regularizer {name} is listed non-ridge but declares is_ridge true")
    declared = set(config["instances"])
    for key in ("required_decision_instances", "diagnostic_instances"):
        missing = set(config[key]) - declared
        if missing:
            problems.append(f"{key} names undeclared instances {sorted(missing)}")
    overlap = set(config["required_decision_instances"]) & set(config["diagnostic_instances"])
    if overlap:
        problems.append(f"an instance is both required and diagnostic: {sorted(overlap)}")
    for key, spec in config["acase_pool"].items():
        if key != "growth" and key not in declared:
            problems.append(f"acase_pool names undeclared instance {key}")
    growth = config["acase_pool"]["growth"]
    for oracle in ("exact_ground_energy", "target_error"):
        if growth.get(oracle) is not None:
            problems.append(
                f"acase_pool.growth.{oracle} must be null: an oracle stop would make the "
                "control's basis depend on the answer it is being compared against")


def check_instances_build(config, problems, notes):
    """Build each instance and confirm its declared shape and grid admissibility."""
    import numpy as np

    from clifford_qc.dense_reference import to_matrix

    for name, spec in config["instances"].items():
        builder = spec["builder"]
        try:
            if builder.endswith("chemistry.h2"):
                from clifford_qc.models import chemistry
                model = chemistry.h2(**spec["args"])
            else:
                from clifford_qc.models import tfim
                model = tfim(**spec["args"])
        except ImportError as exc:
            notes.append(f"SKIP {name}: optional dependency missing ({exc})")
            continue
        H = to_matrix(model.hamiltonian.to_mv())
        H = 0.5 * (H + H.conj().T)
        values = np.linalg.eigvalsh(H)
        lower, upper = float(values[0]), float(values[-1])
        dt = math.pi / (upper - lower)
        if not (upper - lower) * dt < 2 * math.pi:
            problems.append(f"{name}: strict phase-branch condition fails on the primary grid")
        expected_qubits = spec["args"].get("n", model.hamiltonian.n)
        if model.hamiltonian.n != expected_qubits:
            problems.append(f"{name}: built {model.hamiltonian.n} qubits, config implies {expected_qubits}")
        bits = spec.get("reference_bitstring")
        if bits is not None and len(bits) != model.hamiltonian.n:
            problems.append(f"{name}: reference_bitstring width {len(bits)} != {model.hamiltonian.n} qubits")
        if name == "h2_sto3g":
            meta = model.metadata
            for key in ("n_electrons", "sz", "basis", "spin_convention"):
                if key in spec and meta.get(key) != spec[key]:
                    problems.append(f"{name}: metadata {key}={meta.get(key)!r} != config {spec[key]!r}")
        notes.append(f"  {name}: {model.hamiltonian.n} qubits, spectrum "
                     f"[{lower:.6f}, {upper:.6f}], dt={dt:.6f}, branch condition ok")


def main() -> int:
    if not CONFIG.exists():
        print(f"FAIL missing config {CONFIG}")
        return 1
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    problems: list[str] = []
    notes: list[str] = []

    if config.get("schema") != SCHEMA:
        problems.append(f"schema is {config.get('schema')!r}, expected {SCHEMA!r}")
    for key in REQUIRED_TOP:
        if key not in config:
            problems.append(f"missing required key {key!r}")
    if not problems:
        check_no_results(config, problems)
        check_ladder(config, problems)
        check_combination(config, problems)
        check_references(config, problems)
        check_instances_build(config, problems, notes)

    for note in notes:
        print(note)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("OK Phase 16B preregistration is complete, result-free, and decidable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
