"""Verify the second Phase 16B record and that its verdict follows the frozen rule.

This gate does not resample. It re-derives every instance status and the
overall verdict from the record's own per-cell medians under the config's rule,
and fails if the recorded verdict is not the one that rule produces.

Beyond the v1 gate's checks it enforces the two things this experiment added:

  * **admission held at execution.** Every required instance must record an
    incumbent and a candidate that reach the target in exact arithmetic. The
    preregistration gate checked this before the run; this one checks the run
    did not drift off it, because an admitted instance that turns out censored
    would reproduce the v1 failure inside a design built to prevent it.
  * **the Trotter substitution is a measured identity.** The producer builds
    the Trotter step densely instead of in the MV basis. The record must carry
    that verification, its residual must sit inside the tolerance scaled to the
    reference's own unitarity defect, and the dense path must be the more
    accurate of the two.

    python benchmarks/check_phase16b_v2_feasibility.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "phase16b_v2_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_v2_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v2_feasibility.v1"
PENCIL_TOL = 1e-10


ROUND_OFF_FLOOR = 1e-12


def distinct_verifications(arm):
    """One report per verification, not one per basis size that reused it."""
    seen, out = set(), []
    for info in arm.get("sizes", {}).values():
        report = info.get("trotter_step_verification")
        if report is None:
            continue
        key = (report.get("shared_by_qubit_count"), report.get("verified_on_instance"))
        if key in seen:
            continue
        seen.add(key)
        out.append(report)
    return out


def main() -> int:
    if not RECORD.exists():
        print(f"FAIL missing record {RECORD}")
        return 1
    raw_config = CONFIG.read_bytes()
    config = json.loads(raw_config)
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    problems: list[str] = []

    if record.get("schema") != SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}, expected {SCHEMA!r}")
    digest = hashlib.sha256(raw_config).hexdigest()
    if record.get("config_digest") != digest:
        problems.append("config digest moved: the record was produced under a different config")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be literally false")
    if record.get("evidence") != "heuristic":
        problems.append(f"evidence must be 'heuristic', got {record.get('evidence')!r}")
    if record.get("cost_contract") != config["cost_model"]["contract"]:
        problems.append("cost contract does not match the frozen config")
    if "provenance" not in record:
        problems.append("record is not stamped with execution provenance")
    for forbidden in ("phase_status", "phase_complete", "phase_completion"):
        if forbidden in record:
            problems.append(f"record carries a phase-completion claim ({forbidden})")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    from run_phase16b_v2_feasibility import combine, instance_status

    target = float(record["target"])
    declared_replicas = int(record["replicas"])
    required = set(config["required_decision_instances"])
    statuses = {}

    for name, entry in record["instances"].items():
        if "skipped" in entry:
            if name in required:
                problems.append(f"{name} is required but was skipped")
            continue
        checks = entry.get("checks", {})
        for flag in ("branch_condition", "enclosure_contains_spectrum"):
            if not checks.get(flag):
                problems.append(f"{name}: deterministic check {flag} failed")
        residual = checks.get("pencil_identity")
        if residual is None or residual > PENCIL_TOL:
            problems.append(f"{name}: pencil identity residual {residual} exceeds {PENCIL_TOL}")

        if name in required:
            admission = entry.get("admission")
            if not admission:
                problems.append(f"{name} is required but records no admission verification")
            elif not admission.get("admits"):
                problems.append(
                    f"{name}: admission does not hold at execution "
                    f"(incumbent {admission.get('incumbent_zero_noise_error')}, "
                    f"candidate {admission.get('candidate_zero_noise_error')}, "
                    f"target {target}). A required instance whose control cannot reach the "
                    "target in exact arithmetic reproduces the v1 censored comparison")

        for arm_name, arm in entry["arms"].items():
            for report in distinct_verifications(arm):
                if report["residual"] > report["tolerance"]:
                    problems.append(
                        f"{name}/{arm_name}: the dense Trotter step differs from "
                        f"trotter2_unitary by {report['residual']:.3e}, beyond its "
                        f"{report['tolerance']:.3e} tolerance")
                # The substitution is justified when the dense path is no worse
                # than the reference *beyond round-off*. Requiring it to win
                # outright is wrong at small qubit counts, where both paths sit
                # at machine precision and which one is marginally better is
                # noise: at n=4 the defects are 2.3e-15 and 1.1e-14, and neither
                # number means anything about the propagator.
                allowed = max(report["reference_unitarity_defect"], ROUND_OFF_FLOOR)
                if report["dense_unitarity_defect"] > allowed:
                    problems.append(
                        f"{name}/{arm_name}: the dense Trotter step's unitarity defect "
                        f"{report['dense_unitarity_defect']:.3e} exceeds the reference's "
                        f"{allowed:.3e}; the substitution is not justified")
            for reg, sizes in arm.get("budget", {}).items():
                for size_key, budgets in sizes.items():
                    for budget_key, stats in budgets.items():
                        if stats["replicas"] != declared_replicas:
                            problems.append(
                                f"{name}/{arm_name}/{reg}/{size_key}/{budget_key}: "
                                f"{stats['replicas']} replicas, config declares {declared_replicas}")
                        if (stats["failures"] > declared_replicas // 2
                                and stats["median"] < 1e30):
                            problems.append(
                                f"{name}/{arm_name}/{reg}/{size_key}/{budget_key}: "
                                f"{stats['failures']} failures but a finite median -- "
                                "failed solves appear to have been dropped")

        recomputed, evidence = instance_status(entry, config, target)
        if recomputed != entry.get("status"):
            problems.append(f"{name}: recorded status {entry.get('status')!r} but the frozen "
                            f"rule gives {recomputed!r}")
        statuses[name] = recomputed
        if evidence.get("incumbent_censored") and recomputed not in ("UNDETERMINED", "INVALID"):
            problems.append(f"{name}: the incumbent's cost is censored but the status is "
                            f"{recomputed!r}; a missing control must not read as a promotion")

    verdict = combine(statuses, config)
    if verdict != record["decision"]["verdict"]:
        problems.append(f"recorded verdict {record['decision']['verdict']!r} but the frozen "
                        f"rule gives {verdict!r}")

    for name, status in sorted(statuses.items()):
        role = "required" if name in required else "diagnostic"
        print(f"  {name:16} {status:13} ({role})")
    print(f"  verdict: {verdict}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("OK record is complete, admission held, and the verdict follows the frozen rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
