"""Validate the result-free Phase 14b QWC-versus-FC preregistration.

This gate is structural by design: it reads hashes and already-reviewed records,
but draws no samples and produces no scientific result.  The one authorized
sampled execution belongs to a later commit.

    python benchmarks/check_phase14b_preregistration.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "phase14b_qwc_vs_fc.json"
HIERARCHY = HERE / "reference_results" / "clifford_hierarchy_beh2.json"
PROTOCOL_AXIS = HERE / "reference_results" / "protocol_axis.json"
DEVICE_CARDS = HERE / "configs" / "device_cards"

CONFIG_SCHEMA = "clifford_qc.phase14b_qwc_vs_fc_config.v1"
EXPECTED_STATUS = "preregistration_only_no_run_yet"
EXPECTED_PHASE14A_MERGE = "6eee8a39d25bb4a43aa809cb0b0e024164f4b4a0"
EXPECTED_CLAIM_BOUNDARY = (
    "This configuration contains no Phase 14b sample, radius, cost, crossing, "
    "decision, or covariance ratio. A later record may report only the frozen "
    "BeH2/JW QWC-versus-fully-commuting comparison under this allocator, confidence "
    "family, endpoint grid, covariance audit, reconstruction gate, and illustrative "
    "device cards. It may not change the word universe, functional, grouping, "
    "synthesis, accuracy target, delta, schedules, seeds, acceptance rules, or "
    "inherited Phase 14a and R3 evidence."
)

EXPECTED_LINEAGE = {
    "benchmarks/configs/protocol_axis.json": (
        "6e0f99791e4cb418adae490572002805850fb027",
        "908dba85bd8755526682d5c58bf8b41df55583aded2764415fb1a38963bbc7b9",
    ),
    "benchmarks/reference_results/protocol_axis.json": (
        "55d9ec92fac9a35406b31d0f787f4d0adce524f7",
        "d262a4abc3d7cf5dea65e60c5d7bc35142b504444d8594f2a1fc4f5eecd59e92",
    ),
    "benchmarks/reference_results/clifford_hierarchy_beh2.json": (
        "417c559ec3ccbf41dccc18ecee6256a284452660",
        "ad5babccb230a9f66e3ac44633062254368c5b08a2103a080f24ba3e686525b5",
    ),
    "benchmarks/data/beh2_sto3g_r1.3264.FCIDUMP": (
        "7f2b73022cd7b40108e4765aeab5d6ee9230d0a4",
        "01ef4422b2e9e7f711895ae897a94ea11c1b0fae5a5fd91c3396fbd4d69fba76",
    ),
    "benchmarks/data/beh2_sto3g_r1.3264.provenance.json": (
        "4a514949dbff443bcfd00ba0fd0d6f98e5415989",
        "fd4eb5de5bf94e2555f8d3d512e7918146224f49b84f8b890527ad5b8a25c73c",
    ),
    "benchmarks/configs/device_cards/ion-like.json": (
        "45ab9fd9b59638d2b5d0a94dafe3a5bcb311cb26",
        "1ac47a1a80b621bfeb3d5cc36dd9ab859bf94dc782da4c6fee8aebfb3d9c53c2",
    ),
    "benchmarks/configs/device_cards/logical-alltoall.json": (
        "21fab44375442ce632d12c2d6f3ba9bdad3815f7",
        "5eb4668daa124b4f8f375d37839ab4eeb31f10f8cf27612869e2aebcb9f30fe3",
    ),
    "benchmarks/configs/device_cards/superconducting-like.json": (
        "befb3fb9f09087174f7d07f59d2e1f4e8e4f7316",
        "99fac3de98a34324f31dd734918bc6448ce48dd315752c12dbcc503b1d82e248",
    ),
}

EXPECTED_LABELS = [
    "I",
    "E(4,5<-0,1)",
    "E(6,7<-0,1)",
    "E(4,5<-2,3)",
    "E(6,7<-2,3)",
]
EXPECTED_ENDPOINTS = {
    "qwc": {
        "block_size": 1,
        "settings": 353,
        "ritz_touched_assigned_settings": 179,
        "resource_sums": {"N_1q": 3648, "D_1q": 697, "N_2q": 0, "D_2q": 0},
    },
    "fully_commuting": {
        "block_size": 8,
        "settings": 14,
        "ritz_touched_assigned_settings": 7,
        "resource_sums": {"N_1q": 297, "D_1q": 145, "N_2q": 296, "D_2q": 283},
    },
}
EXPECTED_SHOT_ENDPOINTS = [2**power for power in range(16, 33)]
EXPECTED_CARD_SHA256 = {
    "ion-like": "9cc7e53e935e4ec3c265efbd9b6218f17a68c51d727d658c5a5bcaf98d34f453",
    "logical-alltoall": "07e9cc637e590668bc89b7d407535f34f509155e08c95d2ecd0f8f931738265b",
    "superconducting-like": "cf78e3240ad48839a8113a1570a43e614c230a8428fe8ceda7783e7ff75e1a13",
}
FORBIDDEN_RESULT_KEYS = {
    "result",
    "results",
    "verdict",
    "decision",
    "winner",
    "certified_endpoint",
    "certified_total_shots",
    "shot_crossing",
    "crossing",
    "observed_radius",
    "empirical_variance",
    "predicted_variance",
    "covariance_ratio",
    "cost_us",
    "C_time_epsilon_us",
    "provenance",
}
TEMPORAL_BOUNDARY_PHRASES = (
    "has not run",
    "has not been sampled",
    "no sampling has been performed",
    "not run yet",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _result_key_paths(value, prefix=()):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = prefix + (str(key),)
            if key in FORBIDDEN_RESULT_KEYS:
                paths.append(".".join(here))
            paths.extend(_result_key_paths(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_result_key_paths(child, prefix + (str(index),)))
    return paths


def load_config(path: Path = CONFIG) -> dict:
    config = _read(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported Phase 14b preregistration schema")
    if config.get("status") != EXPECTED_STATUS:
        raise ValueError("the sampled Phase 14b record must land in a later commit")
    leaked = _result_key_paths(config)
    if leaked:
        raise ValueError(f"result field in Phase 14b preregistration: {leaked[0]}")
    return config


def lineage_problems(config: dict) -> list[str]:
    problems = []
    lineage = config.get("parent_lineage", {})
    if lineage.get("phase14a_merge_commit") != EXPECTED_PHASE14A_MERGE:
        problems.append("Phase 14b does not descend from the reviewed Phase 14a merge")
    declared = {row.get("path"): row for row in lineage.get("files", [])
                if isinstance(row, dict)}
    if set(declared) != set(EXPECTED_LINEAGE):
        problems.append("Phase 14b lineage file set differs from the frozen inputs")
    for relative, (blob, digest) in EXPECTED_LINEAGE.items():
        row = declared.get(relative, {})
        path = ROOT / relative
        if row.get("git_blob_sha1") != blob:
            problems.append(f"{relative} declared git_blob_sha1 drifted")
        if row.get("sha256") != digest:
            problems.append(f"{relative} declared sha256 drifted")
        if not path.exists():
            problems.append(f"{relative} is missing")
            continue
        if _git_blob_sha1(path) != blob or _sha256(path) != digest:
            problems.append(f"{relative} content differs from the preregistered digest")
    return problems


def system_problems(config: dict) -> list[str]:
    problems = []
    system = config.get("system", {})
    hierarchy = _read(HIERARCHY)
    expected = {
        "system": "beh2",
        "mapping": "jw",
        "n_qubits": hierarchy.get("n_qubits"),
        "basis_size": hierarchy.get("basis_size"),
        "retained_rank": hierarchy.get("basis_size"),
        "basis_labels": hierarchy.get("basis_labels"),
        "total_pauli_words_including_identity": hierarchy.get("word_universe"),
        "measured_word_universe": hierarchy.get("word_universe", 0) - 1,
        "inherited_exact_subspace_bias_millihartree": hierarchy.get("error_millihartree"),
    }
    for field, value in expected.items():
        if system.get(field) != value:
            problems.append(f"Phase 14b system field {field} drifted from the BeH2 record")
    if system.get("basis_labels") != EXPECTED_LABELS:
        problems.append("Phase 14b basis is not the frozen five-vector DA-CASE bank")
    if system.get("ritz_functional_measured_support") != 966:
        problems.append("the frozen Ritz functional must retain its 966 measured words")
    if "analytic exact value one" not in system.get("identity_treatment", ""):
        problems.append("identity must stay analytic and outside the measured universe")

    axis = _read(PROTOCOL_AXIS)
    axis_system = next((row for row in axis.get("systems", [])
                        if row.get("system") == "beh2"), None)
    axis_arm = next((row for row in (axis_system or {}).get("arms", [])
                     if row.get("mapping") == "jw"), None)
    if axis_arm is None:
        problems.append("the BeH2/JW parent arm is absent from protocol_axis")
    else:
        for field in ("basis_size", "retained_rank"):
            if axis_arm.get(field) != system.get(field):
                problems.append(f"protocol_axis and Phase 14b disagree on {field}")
    return problems


def comparison_problems(config: dict) -> list[str]:
    problems = []
    comparison = config.get("comparison", {})
    rows = comparison.get("endpoints", [])
    endpoints = {row.get("name"): row for row in rows if isinstance(row, dict)}
    if set(endpoints) != set(EXPECTED_ENDPOINTS) or len(rows) != 2:
        problems.append("Phase 14b must compare exactly QWC and fully commuting endpoints")
        return problems
    hierarchy = _read(HIERARCHY)
    hierarchy_rows = {row["block_size"]: row for row in hierarchy.get("rows", [])}
    for name, expected in EXPECTED_ENDPOINTS.items():
        row = endpoints[name]
        for field, value in expected.items():
            if row.get(field) != value:
                problems.append(f"{name} {field} differs from the frozen endpoint")
        parent = hierarchy_rows.get(expected["block_size"], {})
        if parent.get("settings") != expected["settings"]:
            problems.append(f"{name} setting count differs from the parent hierarchy")
        # The current parent stores min/max/mean/sum rows. Normalize a future
        # sums-only representation too, but never let its shape skip the check.
        sums = {
            key: value.get("sum") if isinstance(value, dict) else value
            for key, value in parent.get("resource_metrics", {}).items()
        }
        if sums != expected["resource_sums"]:
            problems.append(f"{name} synthesis resources differ from the parent hierarchy")
    if "largest-conflict-degree" not in comparison.get("fixed_grouping", ""):
        problems.append("the deterministic grouping rule is not frozen")
    if "sampled and priced" not in comparison.get("fixed_synthesis", ""):
        problems.append("the sampled and priced Clifford circuits are not bound together")
    return problems


def _historical_seed_roots() -> set[int]:
    roots = set()
    for path in (HERE / "configs").glob("*.json"):
        if path == CONFIG:
            continue
        try:
            payload = _read(path)
        except (OSError, json.JSONDecodeError):
            continue

        def visit(value, seed_namespace=False):
            if isinstance(value, dict):
                for key, child in value.items():
                    child_namespace = seed_namespace or "seed" in key.lower()
                    if (child_namespace and isinstance(child, int)
                            and not isinstance(child, bool)):
                        roots.add(child)
                    visit(child, child_namespace)
            elif isinstance(value, list):
                for child in value:
                    if (seed_namespace and isinstance(child, int)
                            and not isinstance(child, bool)):
                        roots.add(child)
                    visit(child, seed_namespace)

        visit(payload)
    return roots


def protocol_problems(config: dict) -> list[str]:
    problems = []
    protocol = config.get("protocol", {})
    allocator = protocol.get("allocator", {})
    if allocator != {
        "name": "coefficient_range_neyman",
        "outcome_independent": True,
        "group_score": "L_g = sum of absolute Ritz coefficients assigned to setting g",
        "allocation_rule": (
            "two shots per compiled setting, then largest-remainder apportionment of "
            "the remaining total physical-shot budget proportional to L_g; zero-score "
            "settings receive only the two-shot floor"
        ),
        "same_rule_for_both_endpoints": True,
        "min_shots_per_setting": 2,
    }:
        problems.append("Phase 14b allocator differs from the fixed outcome-independent rule")
    if protocol.get("primary_estimator") != (
            "single_assignment covariance-aware grouped reconstruction"):
        problems.append("Phase 14b primary estimator drifted")
    if "cannot determine" not in protocol.get("pooled_estimator_role", ""):
        problems.append("the pooled covariance audit may not select the headline winner")
    accuracy = protocol.get("accuracy", {})
    if accuracy.get("total_target_millihartree") != 1.6:
        problems.append("Phase 14b accuracy target must remain 1.6 mHa")
    expected_radius = math.sqrt(
        (1.6e-3) ** 2
        - (config.get("system", {}).get(
            "inherited_exact_subspace_bias_millihartree", math.inf
        ) * 1e-3) ** 2
    )
    if not math.isclose(
        accuracy.get("stochastic_radius_hartree", math.inf),
        expected_radius,
        rel_tol=0.0,
        abs_tol=1e-18,
    ):
        problems.append("Phase 14b stochastic radius no longer matches target and bias")
    confidence = protocol.get("confidence", {})
    expected_confidence = {"delta": 0.05, "family": 2, "rounds": 17}
    for field, value in expected_confidence.items():
        if confidence.get(field) != value:
            problems.append(f"Phase 14b confidence {field} drifted")
    if "empirical Bernstein" not in confidence.get("bound", ""):
        problems.append("Phase 14b must retain the finite-sample empirical-Bernstein bound")
    if protocol.get("total_physical_shot_endpoints") != EXPECTED_SHOT_ENDPOINTS:
        problems.append("Phase 14b total-shot endpoint grid drifted")
    if "independent fixed schedules" not in protocol.get("endpoint_sampling", ""):
        problems.append("Phase 14b endpoints must use independent precomputed schedules")
    roots = protocol.get("seed_roots", {})
    if set(roots) != {"headline", "covariance_audit"}:
        problems.append("Phase 14b must declare headline and covariance-audit roots")
    elif (not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                  for value in roots.values()) or len(set(roots.values())) != 2):
        problems.append("Phase 14b seed roots must be distinct non-negative integers")
    elif set(roots.values()) & _historical_seed_roots():
        problems.append("a Phase 14b seed root aliases an existing stream")
    if "spawn_key=(protocol_index, endpoint_or_replica_index)" not in protocol.get(
            "seed_derivation", ""):
        problems.append("Phase 14b seed namespace derivation drifted")

    reconstruction = protocol.get("matrix_reconstruction", {})
    if reconstruction.get("max_absolute_entry_error_tolerance") != 5e-13:
        problems.append("Phase 14b exact matrix reconstruction tolerance drifted")
    if reconstruction.get("matrices") != ["S", "H"]:
        problems.append("Phase 14b must reconstruct both S and H")
    audit = protocol.get("covariance_audit", {})
    if audit.get("replicas") != 1000 or audit.get("total_physical_shots") != 2**22:
        problems.append("Phase 14b covariance audit size or budget drifted")
    if audit.get("estimators") != ["single_assignment", "pooled"]:
        problems.append("Phase 14b covariance audit must cover assigned and pooled estimates")
    if audit.get("acceptable_empirical_to_predicted_interval") != [0.8, 1.2]:
        problems.append("Phase 14b covariance agreement interval drifted")
    return problems


def card_problems(config: dict) -> list[str]:
    problems = []
    declared = {row.get("name"): row for row in config.get("protocol", {}).get(
        "device_cards", []) if isinstance(row, dict)}
    if set(declared) != set(EXPECTED_CARD_SHA256):
        problems.append("Phase 14b device-card set drifted")
        return problems
    for name, digest in EXPECTED_CARD_SHA256.items():
        source = _read(DEVICE_CARDS / f"{name}.json")
        row = declared[name]
        if row.get("sha256") != digest:
            problems.append(f"{name} semantic card hash drifted")
        if row.get("connectivity") != source.get("connectivity"):
            problems.append(f"{name} connectivity differs from its card")
        if row.get("routing") != source.get("routing"):
            problems.append(f"{name} routing assumption differs from its card")
        if row.get("status") != source.get("calibration_status"):
            problems.append(f"{name} calibration status differs from its card")
    if "report logical runtime separately per card" not in config.get(
            "acceptance", {}).get("device_reporting_rule", ""):
        problems.append("Phase 14b may not collapse the device cards into one winner")
    return problems


def acceptance_problems(config: dict) -> list[str]:
    problems = []
    acceptance = config.get("acceptance", {})
    gates = acceptance.get("blocking_gates", [])
    if len(gates) != 4:
        problems.append("Phase 14b must retain all four blocking gates")
    joined = " ".join(gates)
    for phrase in ("signed readouts", "5e-13", "total budget", "all four covariance-audit"):
        if phrase not in joined:
            problems.append(f"Phase 14b blocking gates omit {phrase!r}")
    if acceptance.get("material_shot_reduction_fraction") != 0.5:
        problems.append("Phase 14b materiality threshold drifted")
    if "0.5 times" not in acceptance.get("shot_efficiency_go_rule", ""):
        problems.append("Phase 14b shot-efficiency go rule drifted")
    if "suppresses the efficiency verdict" not in acceptance.get("failure_rule", ""):
        problems.append("a protocol failure must suppress the efficiency verdict")
    return problems


def claim_problems(config: dict) -> list[str]:
    problems = []
    leaked = _result_key_paths(config)
    if leaked:
        problems.append(f"result field in Phase 14b preregistration: {leaked[0]}")
    boundary = config.get("claim_boundary")
    if boundary != EXPECTED_CLAIM_BOUNDARY:
        problems.append("Phase 14b claim boundary differs from the frozen wording")
    lower = str(boundary).lower()
    for phrase in TEMPORAL_BOUNDARY_PHRASES:
        if phrase in lower:
            problems.append(f"Phase 14b claim boundary is temporal: {phrase!r}")
    authorization = config.get("authorization", {})
    if authorization.get("exactly_one_later_sampled_execution") is not True:
        problems.append("Phase 14b must authorize exactly one later sampled execution")
    if authorization.get("record") != (
            "benchmarks/reference_results/phase14b_qwc_vs_fc.json"):
        problems.append("the future Phase 14b record path drifted")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    for check in (
        lineage_problems,
        system_problems,
        comparison_problems,
        protocol_problems,
        card_problems,
        acceptance_problems,
        claim_problems,
    ):
        try:
            problems.extend(check(config))
        except (KeyError, TypeError, ValueError, StopIteration, OSError, json.JSONDecodeError) as exc:
            problems.append(f"{check.__name__} could not validate the config: {exc}")
    return problems


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Phase 14b preregistration: FAIL")
        print(f"  - {exc}")
        return 1
    problems = static_problems(config)
    print("Phase 14b preregistration:", "FAIL" if problems else "PASS")
    for problem in problems:
        print(f"  - {problem}")
    if not problems:
        print("  result-free contract; no Phase 14b sampling executed")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
