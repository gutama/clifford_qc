"""Gate Q18-S1's separate post-hoc declaration, without second-moment rows.

The original uniform statistic, banks, grouping and tenfold bar are retained.
The derivative validator changes after the original ratios and sweep were seen,
so commit order protects execution identity, not a preregistered inference.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_measured_residual_preregistration as original
from benchmarks import check_phase15_preregistration as lineage

CONFIG = HERE / "configs" / "phase15_residual_successor.json"
RECORD = HERE / "reference_results" / "phase15_residual_successor.json"
SCHEMA = "clifford_qc.phase15_residual_successor_config.v1"
BOUND_IMPLEMENTATIONS = (
    *original.BOUND_IMPLEMENTATIONS,
    "benchmarks/run_phase15_measured_residual_preflight.py",
    "benchmarks/check_phase15_measured_residual_preflight.py",
    "benchmarks/phase15_residual_successor.py",
    "benchmarks/check_phase15_residual_successor_declaration.py",
    "benchmarks/run_phase15_residual_successor.py",
    "benchmarks/check_phase15_residual_successor.py",
)


def load_config(path=CONFIG):
    return original.load_config(path)


def static_problems(config, *, record_exists=None):
    projected = copy.deepcopy(config)
    projected["schema"] = original.SCHEMA
    problems = original.static_problems(projected)
    if config.get("schema") != SCHEMA:
        problems.append("wrong successor config schema")
    predecessor = original.load_config()
    # This is a numerical-validator successor, not a change of scientific question.
    for key in ("banks", "grouping", "statistic", "decision_rule", "measured_before_freezing"):
        if config.get(key) != predecessor[key]:
            problems.append(f"successor changes the original {key}")
    if config.get("linearization") != predecessor["linearization"]:
        problems.append("successor changes the original estimator or direction definitions")
    successor = config.get("successor", {})
    if successor.get("design_status") != "post_hoc":
        problems.append("successor must disclose post_hoc design")
    if successor.get("predecessor_record") != str(original.RECORD.relative_to(ROOT)):
        problems.append("wrong predecessor record")
    if successor.get("predecessor_sha256") != original._sha256(original.RECORD):
        problems.append("original INVALID record changed")
    if json.loads(original.RECORD.read_text())["decision"]["verdict"] != "INVALID":
        problems.append("predecessor must remain INVALID")
    if not successor.get("disclosure") or successor.get("estimator_licensed") is not False:
        problems.append("successor needs disclosure and no allocation estimator licence")
    if config.get("question_id") != "Q18-S1":
        problems.append("wrong successor question identifier")
    if config.get("evidence", {}).get("design_status") != "post_hoc":
        problems.append("evidence must explicitly label the post-hoc design")
    rule = config.get("derivative_validation", {})
    steps = rule.get("steps", [])
    valid_steps = (len(steps) == 4 and all(isinstance(h, (int, float))
                   and not isinstance(h, bool) and math.isfinite(h) and 0 < h < 1 for h in steps)
                   and all(math.isclose(a, 2 * b, rel_tol=1e-15)
                           for a, b in zip(steps, steps[1:])))
    if not valid_steps:
        problems.append("derivative steps must be four positive successive halvings")
    if rule.get("relative_tolerance") != predecessor["linearization"]["finite_difference"][
            "relative_tolerance"]:
        problems.append("derivative relative tolerance must retain the original value")
    if rule.get("required_pairs") != "last_two" or rule.get("require_endpoint_domain") is not True:
        problems.append("require both final Richardson pairs, stability and endpoint domain")
    allocation = config.get("allocation_diagnostic", {})
    fraction = allocation.get("standard_error_fraction_of_sigma", 0)
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 0.25:
        problems.append("allocation standard error must be inside the delta-method regime")
    for key in ("pilot_shots_per_setting", "minimum_shots_per_setting"):
        value = allocation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            problems.append(f"allocation {key} must be a positive integer")
    if allocation.get("decision_role") != "diagnostic_only" or allocation.get(
            "variance_source") != "exact_oracle":
        problems.append("allocation must remain an exact-oracle diagnostic, outside the decision")
    exists = RECORD.exists() if record_exists is None else record_exists
    for group in ("inputs", "implementation"):
        if group == "implementation" and exists:
            continue
        for binding in config["implementation_lineage"][group]:
            path = ROOT / binding["path"]
            if not path.is_file() or original._sha256(path) != binding["sha256"]:
                problems.append(f"successor binding drift: {binding['path']}")
    bound = {row["path"] for row in config["implementation_lineage"]["implementation"]}
    for path in BOUND_IMPLEMENTATIONS:
        if path not in bound:
            problems.append(f"successor implementation must bind {path}")
    inputs = {row["path"] for row in config["implementation_lineage"]["inputs"]}
    for path in (str(original.CONFIG.relative_to(ROOT)), str(original.RECORD.relative_to(ROOT))):
        if path not in inputs:
            problems.append(f"successor must bind predecessor {path}")
    return problems


def commit_order_problems(notes, *, require_config=False):
    config_commit = lineage._last_commit(CONFIG)
    if config_commit is None:
        return ["commit the successor declaration before executing it"] if require_config else []
    if not RECORD.exists():
        notes.append("  successor record absent; declaration carries no new result")
        return []
    record_commit = lineage._first_commit(RECORD)
    if record_commit is None:
        notes.append("  successor record not committed yet")
        return []
    if {config_commit, record_commit} & lineage._shallow_boundary():
        notes.append("  commit order: SKIP (shallow history)")
        return []
    if config_commit == record_commit or subprocess.run(
            ["git", "merge-base", "--is-ancestor", config_commit, record_commit],
            cwd=ROOT, capture_output=True, timeout=30).returncode:
        return ["successor declaration must strictly precede its record"]
    notes.append(f"  declaration {config_commit[:12]} precedes record {record_commit[:12]}; "
                 "design remains post_hoc")
    return []


def main():
    try:
        config = load_config()
        problems = static_problems(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}")
        return 1
    notes = []
    if not problems:
        problems += original.structural_problems(config, notes)
        problems += commit_order_problems(notes)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("OK Q18-S1 declaration: bound inputs, unchanged uniform question, post_hoc design")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
