"""Gate Q18-S1's separate post-hoc declaration, without second-moment rows.

The original uniform statistic, banks, grouping and tenfold bar are retained.
The derivative validator changes after the original ratios and sweep were seen,
so commit order protects execution identity, not a preregistered inference.
"""

from __future__ import annotations

import copy
import hashlib
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
ARTIFACT_PATHS = {
    "schema": "clifford_qc.phase15_residual_successor.v1",
    "producer": "benchmarks/run_phase15_residual_successor.py",
    "checker": "benchmarks/check_phase15_residual_successor.py",
    "record": str(RECORD.relative_to(ROOT)),
}
# A bounded, post-execution metadata repair. These hashes anchor the executed
# declaration and record; reconstructing them forbids any scientific revision.
METADATA_CORRECTION = {
    "revision": 1,
    "changes": "Correct record_requirements artifact paths after execution; "
               "measurements and execution provenance are unchanged.",
    "scope": "record_artifact_paths_only",
    "execution_config_digest": "6402dc6cce0037f757fc0b79fa9c4f1ecf12937f3fcdc51b5e914ef6a261c4b0",
    "execution_record_sha256": "8061bc790f03322d2eaf8605343de2da6b56864ff78edb42cb11445c596c575d",
    "original_artifact_paths": {
        "producer": "benchmarks/run_phase15_measured_residual_preflight.py",
        "checker": "benchmarks/check_phase15_measured_residual_preflight.py",
        "record": str(original.RECORD.relative_to(ROOT)),
    },
}
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


def metadata_correction_problems(config):
    revisions = config.get("revisions", [])
    if len(revisions) != 2 or revisions[-1] != METADATA_CORRECTION:
        return ["successor must disclose its bounded artifact-path metadata correction"]
    executed = copy.deepcopy(config)
    executed["revisions"].pop()
    executed["record_requirements"].update(METADATA_CORRECTION["original_artifact_paths"])
    digest = hashlib.sha256((json.dumps(executed, indent=1) + "\n").encode()).hexdigest()
    if digest != METADATA_CORRECTION["execution_config_digest"]:
        return ["metadata correction changes the executed declaration beyond artifact paths"]
    return []


def static_problems(config, *, record_exists=None):
    projected = copy.deepcopy(config)
    projected["schema"] = original.SCHEMA
    # The predecessor forbids post-execution revisions. This successor's
    # metadata repair is checked separately against the exact execution hashes.
    projected["revisions"] = projected["revisions"][:1]
    problems = original.static_problems(projected)
    if config.get("schema") != SCHEMA:
        problems.append("wrong successor config schema")
    for key, value in ARTIFACT_PATHS.items():
        if config.get("record_requirements", {}).get(key) != value:
            problems.append(f"successor record_requirements {key} must name its own artifact")
    problems += metadata_correction_problems(config)
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
    record_commit = lineage._last_commit(RECORD)
    if record_commit is None:
        notes.append("  successor record not committed yet")
        return []
    first_config, first_record = lineage._first_commit(CONFIG), lineage._first_commit(RECORD)
    if first_config is None or first_record is None:
        notes.append("  commit order: SKIP (execution history unavailable)")
        return []
    if {config_commit, record_commit, first_config, first_record} & lineage._shallow_boundary():
        notes.append("  commit order: SKIP (shallow history)")
        return []
    for before, after, label in (
            (first_config, first_record, "execution declaration"),
            (config_commit, record_commit, "metadata correction declaration")):
        if before == after or subprocess.run(
                ["git", "merge-base", "--is-ancestor", before, after],
                cwd=ROOT, capture_output=True, timeout=30).returncode:
            return [f"successor {label} must strictly precede its record"]
        notes.append(f"  {label} {before[:12]} precedes record {after[:12]}")
    notes.append("  design remains post_hoc; execution hashes bind the metadata-only repair")
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
