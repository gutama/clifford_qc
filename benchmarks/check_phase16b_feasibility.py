"""Verify the Phase 16B record and that its verdict follows the frozen rule.

The expensive part of this experiment is the sampling; the part that can go
wrong silently is the arithmetic between the samples and the verdict. So this
gate does not resample. It re-derives every status and the overall verdict from
the record's own per-cell medians using the config's rule, and fails if the
recorded verdict is not the one that rule produces.

It also checks the properties a later edit could quietly break:

  * the record is the declared schema and its config digest still matches the
    committed config, so the rule being applied is the rule that was frozen;
  * every deterministic check the design document calls INVALID-on-failure
    actually passed, and the pencil identity is at round-off;
  * no cell dropped a replica -- failures are counted, never removed, which is
    the `nanmedian` defect the pilot already had once;
  * a censored cost comparison produced UNDETERMINED rather than being allowed
    to stand in for a pass, which is the one substitution that would turn a
    missing control into a promotion;
  * the record claims no advantage, no phase completion, and no status-ledger
    update.

    python benchmarks/check_phase16b_feasibility.py

Exits nonzero on any problem.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "phase16b_feasibility.json"
RECORD = ROOT / "reference_results" / "phase16b_feasibility.json"
SCHEMA = "clifford_qc.phase16b_feasibility.v1"
PENCIL_TOL = 1e-10


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
        problems.append("config digest moved: the record was produced under a different config "
                        f"({record.get('config_digest')} != {digest})")
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

    from run_phase16b_feasibility import combine, instance_status

    target = float(record["target"])
    declared_replicas = int(record["replicas"])
    statuses = {}

    for name, entry in record["instances"].items():
        if "skipped" in entry:
            continue
        checks = entry.get("checks", {})
        if not checks.get("branch_condition"):
            problems.append(f"{name}: strict phase-branch condition failed")
        residual = checks.get("pencil_identity")
        if residual is None or residual > PENCIL_TOL:
            problems.append(f"{name}: pencil identity residual {residual} exceeds {PENCIL_TOL}")

        # Every replica must still be present in every statistic.
        for arm_name, arm in entry["arms"].items():
            for reg, sizes in arm.get("budget", {}).items():
                for m_key, budgets in sizes.items():
                    for budget_key, stats in budgets.items():
                        if stats["replicas"] != declared_replicas:
                            problems.append(
                                f"{name}/{arm_name}/{reg}/m={m_key}/{budget_key}: "
                                f"{stats['replicas']} replicas, config declares {declared_replicas}")
                        if stats["failures"] and stats["median"] != float("inf"):
                            # A failure must raise the error, never be dropped: with
                            # more than half the replicas failing the median is infinite.
                            if stats["failures"] > declared_replicas // 2 and stats["median"] < 1e30:
                                problems.append(
                                    f"{name}/{arm_name}/{reg}/m={m_key}/{budget_key}: "
                                    f"{stats['failures']} failures but a finite median -- "
                                    "failed solves appear to have been dropped")

        recomputed, evidence = instance_status(entry, config, target)
        if recomputed != entry.get("status"):
            problems.append(f"{name}: recorded status {entry.get('status')!r} but the frozen "
                            f"rule gives {recomputed!r}")
        statuses[name] = recomputed
        if evidence.get("acase_censored") and recomputed not in ("UNDETERMINED", "INVALID"):
            problems.append(f"{name}: the incumbent's cost is censored but the status is "
                            f"{recomputed!r}; a missing control must not read as a promotion")

    verdict = combine(statuses, config)
    if verdict != record["decision"]["verdict"]:
        problems.append(f"recorded verdict {record['decision']['verdict']!r} but the frozen "
                        f"rule gives {verdict!r}")

    for name, status in sorted(statuses.items()):
        role = "required" if name in config["required_decision_instances"] else "diagnostic"
        print(f"  {name:14} {status:13} ({role})")
    print(f"  verdict: {verdict}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    print("OK record is complete and its verdict follows the frozen rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
