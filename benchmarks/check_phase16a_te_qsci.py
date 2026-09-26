"""Verify the Phase 16A record against its preregistration.

``run_phase16a_te_qsci.py`` writes ``benchmarks/reference_results/phase16a_te_qsci.json``.
This gate trusts none of the record's summary fields.

1. **Declaration.** The record names the config by digest, and the config is
   unchanged since the run. It carries its own tenseless claim boundary, one
   that does not deny its sampling, quotes the config's boundary rather than
   inheriting it, and makes no advantage claim.
2. **Completeness.** Every instance, sampled arm, budget and replica the
   config declares is present, and every seed is the declared
   ``SeedSequence`` stream, recomputed here.
3. **The freeze.** Each instance's ``measured_at_execution`` equals the
   config's ``measured_before_freezing``.
4. **Statistics and decision.** Every median, p90, shots-to-target, paired
   difference, instance status and the verdict are re-derived from the
   per-replica arrays under the frozen rule. This is an implementation
   separate from the producer's, and the verdict reads the required
   instances only.
5. **Controls.** Both sample-blind controls are recomputed from the committed
   inputs at every configuration count the record holds.
6. **Sampling.** Replica 0 of every (instance, arm, budget) cell is replayed
   from its seed and must match exactly (``--replay-all`` replays every
   replica, which costs as much as the run).
7. **Order.** Through the preregistration gate, the config's last change must
   strictly precede the record's commit.

    python benchmarks/check_phase16a_te_qsci.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase16a_preregistration as gate
    from benchmarks import run_phase16a_te_qsci as producer
except ImportError:  # pragma: no cover - script execution
    import check_phase16a_preregistration as gate
    import run_phase16a_te_qsci as producer

STATUSES = {"PASS", "FAIL", "UNDETERMINED", "INVALID"}
DENIALS = ("no sampling", "no sampled draw", "carries no sampled")


def _close(a, b, tol=1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def declaration_problems(config: dict, record: dict, config_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != producer.SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("config_digest") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("the config changed after the run: its digest no longer matches")
    boundary = str(record.get("claim_boundary", ""))
    if boundary == config["claim_boundary"]:
        problems.append("the record inherits the config's claim boundary instead of "
                        "stating its own")
    if any(phrase in boundary for phrase in DENIALS):
        problems.append("the record's claim boundary denies the sampling it reports")
    quoted = record.get("preregistration", {}).get("config_claim_boundary_at_landing")
    if quoted != config["claim_boundary"]:
        problems.append("the record does not quote the config's boundary at landing")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be false")
    return problems


def completeness_problems(config: dict, record: dict) -> list[str]:
    problems = []
    replicas = int(config["replicas"])
    if record.get("replicas") != replicas:
        problems.append(f"record has {record.get('replicas')} replicas, config {replicas}")
    budgets = [int(b) for b in config["shot_grid"]["total_shots"]]
    names = config["required_decision_instances"] + config["diagnostic_instances"]
    if sorted(record.get("instances", {})) != sorted(names):
        return problems + [f"instances {sorted(record.get('instances', {}))} are not "
                           f"the declared {sorted(names)}"]
    for name in names:
        arms = record["instances"][name]["arms"]
        if sorted(arms) != sorted(producer.SAMPLED_ARMS):
            problems.append(f"{name}: arms {sorted(arms)} are not the sampled arms")
            continue
        for arm in producer.SAMPLED_ARMS:
            cells = arms[arm]["budgets"]
            if sorted(int(b) for b in cells) != budgets:
                problems.append(f"{name} {arm}: budgets are not the declared grid")
                continue
            for index, total in enumerate(budgets):
                cell = cells[str(total)]
                arrays = ["seeds", "unique_configurations", "errors", "discarded_shots"]
                if arm == producer.CANDIDATE:
                    for control in ("iterated_control", "matched_control"):
                        arrays += [f"{control}.{key}" for key in cell[control]]
                for key in arrays:
                    head, _, tail = key.partition(".")
                    values = cell[head][tail] if tail else cell[head]
                    if len(values) != replicas:
                        problems.append(f"{name} {arm} {total}: {key} has "
                                        f"{len(values)} entries")
                expected = [producer.seed_for(config, name, arm, index, r)
                            for r in range(len(cell["seeds"]))]
                if cell["seeds"] != expected:
                    problems.append(f"{name} {arm} {total}: seeds are not the declared "
                                    "streams")
                if min(cell["unique_configurations"], default=1) < 1:
                    problems.append(f"{name} {arm} {total}: an empty sampled subspace")
                if min(cell["errors"], default=0.0) < -1e-9:
                    problems.append(f"{name} {arm} {total}: a QSCI energy below the "
                                    "exact ground energy")
                if arm == producer.CANDIDATE and any(
                        size > m for size, m in zip(cell["iterated_control"]["sizes"],
                                                    cell["unique_configurations"])):
                    problems.append(f"{name} {total}: a control used more determinants "
                                    "than the candidate")
    return problems


def freeze_problems(config: dict, record: dict) -> list[str]:
    problems = []
    for name, entry in record["instances"].items():
        frozen = config["measured_before_freezing"][name]
        measured = entry["measured_at_execution"]
        for key in gate._INT_FIELDS + gate._LIST_INT_FIELDS:
            if measured.get(key) != frozen[key]:
                problems.append(f"{name}: {key} at execution {measured.get(key)!r} is "
                                f"not the frozen {frozen[key]!r}")
        for key in gate._FLOAT_FIELDS:
            if not gate._close(measured.get(key, float("nan")), frozen[key]):
                problems.append(f"{name}: {key} at execution differs from the freeze")
        if not all(entry["deterministic_checks"].values()) and (
                entry["decision"]["status"] != "INVALID"):
            problems.append(f"{name}: a failed deterministic check did not make the "
                            "instance INVALID")
    return problems


def derive_status(config: dict, cells: dict, checks: dict) -> tuple[str, int | None,
                                                                     float | None]:
    """The frozen instance rule, restated independently of the producer."""
    if not all(checks.values()):
        return "INVALID", None, None
    target = float(config["target"]["value"])
    tie = float(config["decision_rule"]["tie_tolerance"])
    for total in sorted(int(b) for b in cells):
        errors = np.asarray(cells[str(total)]["errors"], dtype=float)
        if float(np.median(errors)) <= target:
            controls = np.asarray(cells[str(total)]["iterated_control"]["errors"],
                                  dtype=float)
            median = float(np.median(errors - controls))
            return ("PASS" if median < -tie else "FAIL"), total, median
    return "UNDETERMINED", None, None


def derive_verdict(statuses: list[str]) -> str:
    if "INVALID" in statuses:
        return "INVALID"
    if statuses and all(s == "PASS" for s in statuses):
        return "GO"
    if statuses and all(s == "FAIL" for s in statuses):
        return "NO_GO"
    return "CONDITIONAL"


def statistic_problems(config: dict, record: dict) -> list[str]:
    problems = []
    for name, entry in record["instances"].items():
        for arm, body in entry["arms"].items():
            for total, cell in body["budgets"].items():
                if not _close(cell["median_error"], np.median(cell["errors"])):
                    problems.append(f"{name} {arm} {total}: median_error drifted")
                if not _close(cell["p90_error"], producer.p90(cell["errors"])):
                    problems.append(f"{name} {arm} {total}: p90_error drifted")
        status, reached, median = derive_status(
            config, entry["arms"][producer.CANDIDATE]["budgets"],
            entry["deterministic_checks"])
        decision = entry["decision"]
        if decision.get("status") != status:
            problems.append(f"{name}: recorded status {decision.get('status')} is not "
                            f"the rule's {status}")
        if decision.get("shots_to_target") != reached:
            problems.append(f"{name}: shots_to_target {decision.get('shots_to_target')} "
                            f"is not {reached}")
        if not _close(decision.get("median_paired_difference"), median):
            problems.append(f"{name}: median paired difference drifted")
        expected_promotes = name in config["required_decision_instances"]
        if decision.get("promotes") is not expected_promotes:
            problems.append(f"{name}: promotes flag contradicts its role")
    required = config["required_decision_instances"]
    statuses = [record["instances"][name]["decision"]["status"] for name in required]
    verdict = derive_verdict(statuses)
    summary = record.get("decision", {})
    if summary.get("verdict") != verdict:
        problems.append(f"recorded verdict {summary.get('verdict')} is not the rule's "
                        f"{verdict}")
    if summary.get("instance_statuses") != dict(zip(required, statuses)):
        problems.append("the summary's instance statuses are not the required "
                        "instances' own")
    if set(summary.get("instance_statuses", {})) & set(config["diagnostic_instances"]):
        problems.append("a diagnostic instance entered the verdict")
    return problems


def control_problems(config: dict, record: dict, *, build=None) -> list[str]:
    """Recompute both sample-blind controls at every count the record holds."""
    from clifford_qc.backends import SectorStatevectorBackend
    from clifford_qc.subspace import run_control

    problems = []
    for name, entry in record["instances"].items():
        spec = config["instances"][name]
        model = gate.build_instance(spec) if build is None else build(spec)
        metadata = model.metadata
        backend = SectorStatevectorBackend(model.n, metadata["n_electrons"],
                                           metadata["sz"],
                                           spin_ordering=metadata["spin_convention"])
        operator = backend.operator(model.hamiltonian)
        exact = float(entry["exact_ground_energy"])
        seen: dict[tuple[str, int], tuple[int, float]] = {}
        for cell in entry["arms"][producer.CANDIDATE]["budgets"].values():
            for label, kind in (("iterated_control", "iterated_selected_ci"),
                                ("matched_control", "matched_selected_ci")):
                for m, size, error in zip(cell["unique_configurations"],
                                          cell[label]["sizes"], cell[label]["errors"]):
                    key = (kind, int(m))
                    if key not in seen:
                        control = run_control(operator, [0], name=kind, kind=kind,
                                              max_determinants=int(m),
                                              score="epstein_nesbet")
                        seen[key] = (control.determinant_count,
                                     float(control.energy - exact))
                    want_size, want_error = seen[key]
                    if size != want_size or not _close(error, want_error, 1e-9):
                        problems.append(f"{name}: {kind} at {m} determinants recomputes "
                                        f"to ({want_size}, {want_error:.3e}), recorded "
                                        f"({size}, {error:.3e})")
    return problems


def replay_problems(config: dict, record: dict, *, replicas=(0,), build=None) -> list[str]:
    """Regenerate sampled cells from their seeds; they must match exactly."""
    problems = []
    budgets = [int(b) for b in config["shot_grid"]["total_shots"]]
    for name, entry in record["instances"].items():
        model = None if build is None else build(config["instances"][name])
        prepared = producer.prepare_instance(config, name, model=model)
        for arm in producer.SAMPLED_ARMS:
            cells = entry["arms"][arm]["budgets"]
            for index, total in enumerate(budgets):
                cell = cells[str(total)]
                chosen = range(len(cell["seeds"])) if replicas is None else replicas
                for replica in chosen:
                    seed = producer.seed_for(config, name, arm, index, replica)
                    indices, sampling, result = producer.sample_cell(
                        prepared, arm, total, seed)
                    got = (int(indices.size), float(result.energy - prepared.exact),
                           int(sampling.discarded_shots))
                    want = (cell["unique_configurations"][replica],
                            cell["errors"][replica], cell["discarded_shots"][replica])
                    if got[0] != want[0] or got[2] != want[2] or not _close(
                            got[1], want[1], 1e-10):
                        problems.append(f"{name} {arm} {total} replica {replica}: "
                                        f"replays to {got}, recorded {want}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--replay-all", action="store_true",
                        help="replay every replica instead of replica 0 of each cell")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the sampling replay")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    problems = declaration_problems(config, record, config_bytes)
    problems += completeness_problems(config, record)
    if not problems:
        problems += freeze_problems(config, record)
        problems += statistic_problems(config, record)
        problems += control_problems(config, record)
        if not args.no_replay:
            problems += replay_problems(config, record,
                                        replicas=None if args.replay_all else (0,))
    notes: list[str] = []
    if args.record.resolve() == producer.RECORD.resolve():
        problems += gate.commit_order_problems(notes)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    decision = record["decision"]
    print(f"OK Phase 16A record re-derives under the frozen rule: verdict "
          f"{decision['verdict']}, statuses {decision['instance_statuses']}, "
          f"diagnostic {decision['diagnostic_statuses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
