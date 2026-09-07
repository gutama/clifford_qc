"""Validate the result-free R4a contextual-interaction preregistration.

This checker binds the four-arm design, contextual constructor, structural
admission gate, and conditionally frozen sampled protocol. It does not compile
the BeH2 contextual ladder, solve a new projected bank, draw samples, or
produce an R4a scientific result.

    python benchmarks/check_r4a_preregistration.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "r4a_contextual_interaction.json"
PRICEABILITY = HERE / "reference_results" / "priceability_screen.json"
DEVICE_CARDS = HERE / "configs" / "device_cards"

CONFIG_SCHEMA = "clifford_qc.r4a_contextual_interaction_config.v1"
EXPECTED_STATUS = "preregistration_only_no_structural_or_sampled_run_yet"
EXPECTED_BASE = "9abe407852c83bd96b3b70d789b831bff37a7b31"
EXPECTED_CLAIM_BOUNDARY = (
    "This configuration contains no R4a contextual stabilizer selection, "
    "structural-screen result, selected contextual rung, bias floor, word universe, "
    "setting count, sample, cost bracket, ratio, Delta, or QR5 classification. "
    "A later structural record may report only the frozen BeH2/JW four-arm admission "
    "screen. Sampling remains unauthorized unless that record admits every arm under "
    "the declared bias and word-universe gates; any sampled execution, if authorized "
    "by a later commit, must preserve this bank, constructor, rung rule, measurement "
    "grid, estimators, endpoints, seeds, pass rule, and device cards."
)

EXPECTED_LINEAGE = {
    "benchmarks/configs/priceability_screen.json": (
        "5a803b8812d50ab3af70cb1916da16de466b3e45",
        "f43fac4ad8386ecdfa279cfbf5e61661c7a24f3459c7f2f229ad10b0b7f9ec3e",
    ),
    "benchmarks/reference_results/priceability_screen.json": (
        "bf648f05ce60a0e5213d00f3bffae256090b6b7d",
        "61ee716540b9aad7be49193255dac12a097a1c46000f6b0f8534139a646a175d",
    ),
    "benchmarks/reference_results/protocol_axis.json": (
        "55d9ec92fac9a35406b31d0f787f4d0adce524f7",
        "d262a4abc3d7cf5dea65e60c5d7bc35142b504444d8594f2a1fc4f5eecd59e92",
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
EXPECTED_CARD_SHA256 = {
    "ion-like": "9cc7e53e935e4ec3c265efbd9b6218f17a68c51d727d658c5a5bcaf98d34f453",
    "logical-alltoall": "07e9cc637e590668bc89b7d407535f34f509155e08c95d2ecd0f8f931738265b",
    "superconducting-like": "cf78e3240ad48839a8113a1570a43e614c230a8428fe8ceda7783e7ff75e1a13",
}
EXPECTED_ARMS = [
    ("full_qse", "C0", False, False),
    ("cs_qse", "C_CS", True, False),
    ("acase", "C_ACASE", False, True),
    ("cs_acase", "C_CS+ACASE", True, True),
]
EXPECTED_BLOCK_SIZES = [1, 2, 4, 8]
EXPECTED_SHOT_ENDPOINTS = [64, 256, 1024, 4096, 16384, 65536]
EXPECTED_SEED_ROOTS = {
    "exploratory": 150814000,
    "confirmatory": 150914000,
    "bootstrap": 151014000,
}
EXPECTED_STRUCTURAL_FIELDS = {
    "basis_size_before_deduplication",
    "basis_size_after_deduplication",
    "annihilated_candidate_indices",
    "duplicate_projected_candidate_indices",
    "active_qubits",
    "bias_millihartree",
    "hamiltonian_removed_hs_fraction",
    "word_universe",
    "settings_by_block_size",
}
FORBIDDEN_RESULT_KEYS = {
    "result",
    "results",
    "verdict",
    "decision",
    "outcome",
    "selected_contextual_rung",
    "selected_stabilizers",
    "observed_bias_millihartree",
    "observed_word_universe",
    "observed_settings",
    "cost_bracket",
    "cost_us",
    "r_CS",
    "r_A",
    "r_joint",
    "Delta",
    "qr5_classification",
    "provenance",
}


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
        raise ValueError("unsupported R4a preregistration schema")
    if config.get("status") != EXPECTED_STATUS:
        raise ValueError("R4a structural and sampled records must land later")
    leaked = _result_key_paths(config)
    if leaked:
        raise ValueError(f"result field in R4a preregistration: {leaked[0]}")
    return config


def lineage_problems(config: dict) -> list[str]:
    problems = []
    lineage = config.get("parent_lineage", {})
    if lineage.get("base_commit") != EXPECTED_BASE:
        problems.append("R4a does not descend from the reviewed Phase G1 head")
    declared = {
        row.get("path"): row
        for row in lineage.get("files", [])
        if isinstance(row, dict)
    }
    if set(declared) != set(EXPECTED_LINEAGE):
        problems.append("R4a lineage file set differs from the frozen inputs")
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
    expected_scalars = {
        "system": "beh2",
        "mapping": "jw",
        "n_qubits": 8,
    }
    for field, expected in expected_scalars.items():
        if system.get(field) != expected:
            problems.append(f"R4a system field {field} drifted")
    family = system.get("candidate_family", {})
    if family.get("name") != "complete symmetry-preserving determinant singles and doubles":
        problems.append("the full-QSE candidate family drifted")
    if family.get("full_qse_uses_every_candidate") is not True:
        problems.append("full QSE must retain the complete declared candidate family")
    if "max_rank=2, conserve_sz=True" not in family.get("constructor", ""):
        problems.append("the determinant candidate constructor drifted")

    priceability = _read(PRICEABILITY)
    beh2 = next(
        (row for row in priceability.get("candidates", [])
         if row.get("candidate") == "beh2"),
        None,
    )
    stop = (beh2 or {}).get("margin_stop")
    if stop is None:
        problems.append("the frozen BeH2 margin-stop bank is absent")
        return problems
    acase = system.get("acase_bank", {})
    if acase.get("selection_source") != (
        "benchmarks/reference_results/priceability_screen.json candidate beh2.margin_stop"
    ):
        problems.append("the A-CASE selection source drifted")
    if acase.get("basis_size") != stop.get("basis_size"):
        problems.append("the A-CASE basis size differs from the frozen R3S bank")
    if acase.get("basis_labels") != stop.get("labels"):
        problems.append("the A-CASE basis labels differ from the frozen R3S bank")
    if not math.isclose(
        acase.get("inherited_bias_millihartree", math.inf),
        stop.get("bias_millihartree", -math.inf),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        problems.append("the inherited A-CASE bias differs from R3S")
    if acase.get("inherited_word_universe") != stop.get("binding_word_universe"):
        problems.append("the inherited A-CASE word universe differs from R3S")
    if acase.get("selection_may_not_be_reopened") is not True:
        problems.append("R4a may not reopen the frozen A-CASE selection")
    return problems


def comparison_problems(config: dict) -> list[str]:
    problems = []
    rows = config.get("comparison", {}).get("arms", [])
    observed = [
        (
            row.get("name"),
            row.get("cost_symbol"),
            row.get("contextual_restriction"),
            row.get("adaptive_compression"),
        )
        for row in rows
        if isinstance(row, dict)
    ]
    if observed != EXPECTED_ARMS:
        problems.append("R4a must retain exactly the four ordered comparator arms")
    matched = " ".join(config.get("comparison", {}).get("matched_rules", []))
    for phrase in (
        "same FCIDUMP",
        "two QSE arms",
        "two contextual arms",
        "may not be silently replaced",
    ):
        if phrase not in matched:
            problems.append(f"the matched-arm rules omit {phrase!r}")
    classifications = set(
        config.get("comparison", {}).get("future_classification", {})
    )
    if classifications != {
        "complementary", "multiplicative", "redundant",
        "antagonistic", "undetermined",
    }:
        problems.append("the preregistered QR5 classification set drifted")
    return problems


def contextual_problems(config: dict) -> list[str]:
    problems = []
    contextual = config.get("contextual_restriction", {})
    if contextual.get("implementation") != (
        "clifford_qc.subspace.contextual.compile_contextual_restriction"
    ):
        problems.append("the contextual constructor path drifted")
    if contextual.get("term_order") != (
        "descending absolute real coefficient, then ascending packed Pauli-word code"
    ):
        problems.append("the contextual term tie-break drifted")
    if "coefficient-minimising sign" not in contextual.get("sign_rule", ""):
        problems.append("the contextual sign rule drifted")
    if "Hartree-Fock expectation" not in contextual.get("reference_rule", ""):
        problems.append("the reference compatibility gate drifted")
    if "GF(2) symplectic rank" not in contextual.get("independence_rule", ""):
        problems.append("the contextual independence gate drifted")
    if contextual.get("fixed_qubit_ladder") != list(range(1, 8)):
        problems.append("the eight-qubit contextual ladder must remain 1 through 7")
    if contextual.get("tolerance") != 1e-9:
        problems.append("the contextual comparison tolerance drifted")
    if contextual.get("tolerance_scope") != (
        "reference expectation and compiled-image checks only; every represented "
        "non-identity Hamiltonian word remains a selection candidate regardless of "
        "coefficient magnitude"
    ):
        problems.append("the contextual tolerance scope drifted")
    if "largest fixed-qubit count" not in contextual.get("rung_rule", ""):
        problems.append("the contextual rung-selection rule drifted")
    if "Restriction.transport remains strict" not in contextual.get(
        "exact_symmetry_boundary", ""
    ):
        problems.append("R4a may not weaken exact restriction transport")
    projection_rule = contextual.get("projection_rule", "")
    if "non-identity Hamiltonian Hilbert-Schmidt norm" not in projection_rule:
        problems.append("the contextual Hamiltonian leakage denominator drifted")
    if "complete Hamiltonian, including its identity term" not in projection_rule:
        problems.append("the contextual projection may not discard the identity shift")

    from clifford_qc.subspace.contextual import (
        compile_contextual_restriction,
        project_contextual_problem,
        select_contextual_stabilizers,
    )
    if not all(callable(item) for item in (
        select_contextual_stabilizers,
        compile_contextual_restriction,
        project_contextual_problem,
    )):
        problems.append("the frozen contextual API is unavailable")
    return problems


def structural_screen_problems(config: dict) -> list[str]:
    problems = []
    screen = config.get("structural_screen", {})
    if screen.get("evidence_tier") != "structural":
        problems.append("the R4a admission screen must remain structural")
    if screen.get("accuracy_target_millihartree") != 1.6:
        problems.append("the R4a accuracy target must remain 1.6 mHa")
    if screen.get("margin_factor") != 3.0:
        problems.append("the inherited R3S margin factor drifted")
    if not math.isclose(
        screen.get("admissible_bias_millihartree", math.inf),
        screen.get("accuracy_target_millihartree", 0.0)
        / screen.get("margin_factor", math.nan),
        rel_tol=0.0,
        abs_tol=1e-16,
    ):
        problems.append("the admissible bias is not target divided by margin")
    if screen.get("word_universe_ceiling") != 2048:
        problems.append("the inherited word-universe ceiling drifted")
    if screen.get("word_universe_convention") != "non_identity_words_only":
        problems.append("the priceability word convention drifted")
    if screen.get("hamiltonian_removed_hs_fraction_denominator") != (
        "Hilbert-Schmidt norm of the non-identity Hamiltonian; "
        "identity energy shifts are excluded"
    ):
        problems.append("the structural Hamiltonian leakage denominator drifted")
    if screen.get("block_sizes") != EXPECTED_BLOCK_SIZES:
        problems.append("the structural grouping grid drifted")
    if set(screen.get("required_per_arm_fields", [])) != EXPECTED_STRUCTURAL_FIELDS:
        problems.append("the structural record field set drifted")
    if screen.get("all_arms_must_clear_bias_margin") is not True:
        problems.append("every arm must clear the structural bias margin")
    if screen.get("all_arms_must_clear_word_universe_ceiling") is not True:
        problems.append("every arm must clear the word-universe ceiling")
    if "stops R4a without sampling" not in screen.get("failure_rule", ""):
        problems.append("a failed structural gate must suppress sampling")
    if "only when an arm's exact bias floor exceeds epsilon" not in screen.get(
        "infinite_cost_rule", ""
    ):
        problems.append("screen rejection may not be relabelled as infinite cost")
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
                    if (
                        child_namespace
                        and isinstance(child, int)
                        and not isinstance(child, bool)
                    ):
                        roots.add(child)
                    visit(child, child_namespace)
            elif isinstance(value, list):
                for child in value:
                    if (
                        seed_namespace
                        and isinstance(child, int)
                        and not isinstance(child, bool)
                    ):
                        roots.add(child)
                    visit(child, seed_namespace)

        visit(payload)
    return roots


def sampled_protocol_problems(config: dict) -> list[str]:
    problems = []
    protocol = config.get("conditional_sampled_protocol", {})
    if protocol.get("authorized_by_this_config") is not False:
        problems.append("this preregistration may not authorize R4a sampling")
    if protocol.get("evidence_tier") != "exact":
        problems.append("the future R4a cost tier must remain exact")
    if protocol.get("block_sizes") != EXPECTED_BLOCK_SIZES:
        problems.append("the future sampled block-size grid drifted")
    if protocol.get("estimators") != ["single_assignment", "pooled"]:
        problems.append("the future sampled estimator set drifted")
    if protocol.get(
        "search_endpoints_effective_shots_per_setting"
    ) != EXPECTED_SHOT_ENDPOINTS:
        problems.append("the future sampled endpoint grid drifted")
    if protocol.get("exploratory_replicas") != 30:
        problems.append("the future exploratory replica count drifted")
    if protocol.get("confirmatory_replicas") != 100:
        problems.append("the future confirmatory replica count drifted")
    if protocol.get("bootstrap_replicates") != 10000:
        problems.append("the future bootstrap size drifted")
    if protocol.get("one_sided_delta") != 0.05:
        problems.append("the future one-sided error rate drifted")
    if protocol.get("nested_endpoints") is not True:
        problems.append("the future shot endpoints must remain nested")
    roots = protocol.get("seed_roots", {})
    if roots != EXPECTED_SEED_ROOTS:
        problems.append("the future R4a seed roots drifted")
    if set(roots.values()) & _historical_seed_roots():
        problems.append("a future R4a seed root aliases an existing stream")
    if "arm_index, contextual_rung_index" not in protocol.get(
        "seed_derivation", ""
    ):
        problems.append("the future R4a seed namespace drifted")
    if "all 100 confirmatory solves succeed" not in protocol.get("pass_rule", ""):
        problems.append("the future exact-tier pass rule drifted")
    if "right-censored" not in protocol.get("censoring_rule", ""):
        problems.append("the future censoring rule drifted")
    if "no midpoint-only QR5 classification" not in protocol.get(
        "interaction_uncertainty", ""
    ):
        problems.append("QR5 must propagate cost-bracket uncertainty")

    cards = {
        row.get("name"): row.get("sha256")
        for row in protocol.get("device_cards", [])
        if isinstance(row, dict)
    }
    if cards != EXPECTED_CARD_SHA256:
        problems.append("the future R4a device-card set drifted")
    environment = protocol.get("execution_environment", {})
    if environment != {
        "python_minor": "3.12",
        "numpy": "2.5.2",
        "scipy": "1.18.0",
        "stim": "1.16.0",
    }:
        problems.append("the future R4a execution environment drifted")
    return problems


def claim_problems(config: dict) -> list[str]:
    problems = []
    leaked = _result_key_paths(config)
    if leaked:
        problems.append(f"result field in R4a preregistration: {leaked[0]}")
    if config.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        problems.append("the R4a claim boundary differs from the frozen wording")
    authorization = config.get("authorization", {})
    if authorization.get("exactly_one_later_structural_execution") is not True:
        problems.append("R4a must authorize exactly one later structural execution")
    if authorization.get("sampled_execution_authorized_by_this_config") is not False:
        problems.append("R4a sampling must remain unauthorized at this stage")
    if authorization.get("structural_record") != (
        "benchmarks/reference_results/r4a_contextual_screen.json"
    ):
        problems.append("the future R4a structural-record path drifted")
    reporting = config.get("reporting_contract", {})
    for field in (
        "structural_record_is_not_a_cost_record",
        "existing_records_are_immutable",
        "structural_result_commit_must_be_separate",
        "sampled_result_commit_must_be_separate",
        "qr5_is_answered_only_by_the_sampled_four_arm_cost_interaction",
    ):
        if reporting.get(field) is not True:
            problems.append(f"R4a reporting contract {field} drifted")
    return problems


def static_problems(config: dict) -> list[str]:
    problems = []
    for check in (
        lineage_problems,
        system_problems,
        comparison_problems,
        contextual_problems,
        structural_screen_problems,
        sampled_protocol_problems,
        claim_problems,
    ):
        try:
            problems.extend(check(config))
        except (
            KeyError,
            TypeError,
            ValueError,
            StopIteration,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            problems.append(f"{check.__name__} could not validate the config: {exc}")
    return problems


def main() -> int:
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("R4a preregistration: FAIL")
        print(f"  - {exc}")
        return 1
    problems = static_problems(config)
    print("R4a preregistration:", "FAIL" if problems else "PASS")
    for problem in problems:
        print(f"  - {problem}")
    if not problems:
        print("  result-free contract; no R4a structural or sampled execution")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
