"""Rederive and rebuild the Q18-S3 record: certifying Krylov prefixes, then allocation.

Two layers, as for Q18-S1 and Q18-S2. The rederivation trusts no outcome flag
in the record. From the recorded inputs it restates each prefix's rank margin,
resolution, certifying predicate and derived fields; for every priced prefix
the independent-route agreements, both linearization checks, Q18-S1's
Richardson rule on the recorded endpoints, every ratio and integer-allocation
total from the recorded setting variances, every deterministic check and the
status; the three lineage blocks; the practical stage's summary and status
from its recorded replicates, or its absence; and the oracle and final
verdicts. Key sets are closed, provenance included, and types stay exact.

The rebuild then reruns the declared pipeline from committed inputs and
compares. A Krylov overlap is ill-conditioned by construction, so a float the
pencil determines is compared to a relative tolerance that grows with the
prefix's retained condition number, ``max(1e-9, 1000 eps kappa)``; the
eighty-digit values, which no BLAS kernel touches, are compared to ``1e-12``.
Counts whose last words are rounding-level weights are pinned to a
platform-robust band, as in Q18-S2's checker.

    python benchmarks/check_phase15_certifying_krylov.py [--no-recompute]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_certifying_krylov_declaration as declaration
from benchmarks import check_phase15_certifying_screen as q18s2_checker
from benchmarks import check_phase15_measured_residual_preflight as q18
from benchmarks import check_phase15_residual_successor as q18s1
from benchmarks import run_phase15_certifying_krylov as producer

ROUNDING_ULPS = 100.0
GAP_FLOOR = 1e-6
TAU_S, MAX_CONDITION = 1e-10, 1e12
EPSILON = 2.0 ** -52
ROUNDING_GROWTH = 1000.0
TOP_KEYS = frozenset({
    "schema", "config_path", "config_digest", "claim_boundary", "quantum_advantage_claim",
    "question", "design", "statistic", "certification", "rank_margin", "grouping",
    "derivative_validation", "fixed_step_validation", "allocation_diagnostic",
    "practical_allocation", "spectral_reference", "premises", "basis", "lineage",
    "practical", "decision", "declaration", "elapsed_seconds", "evidence_schema",
    "evidence", "provenance"})
BASE_KEYS = frozenset({
    "basis_size", "ground_energy", "error", "variance", "residual_norm",
    "variance_resolved", "cancellation_scale", "variance_resolution", "weinstein_upper",
    "ground_weight_lower_bound", "temple_lower_bound", "subspace_gap", "rank", "dense",
    "certifying", "status", "role", "in_domain", "seconds"})
PRICED_EXTRA = frozenset({"precise", "universes", "estimator", "precise_derivatives", "fixed_step",
                          "protocols", "deterministic_checks"})
RANK_KEYS = frozenset({"effective_rank", "smallest_overlap_eigenvalue",
                       "largest_overlap_eigenvalue", "overlap_condition_number",
                       "retained_condition_number", "full_rank", "rank_stable"})
CHECK_KEYS = (
    "full_rank", "ground_root_nondegenerate", "energy_consistent_with_spectrum",
    "precise_energy_agrees", "precise_variance_agrees", "dense_energy_agrees",
    "dense_variance_agrees", "residual_functional_reproduces_variance",
    "residual_functional_mean_zero", "linearization_validated", "fixed_step_validated",
    "partitions_valid", "group_variances_nonnegative")
SUMMARY_KEYS = frozenset({
    "valid_replicates", "energy_coverage", "sigma_coverage", "energy_z_spread",
    "sigma_z_spread", "energy_spread_over_target", "sigma_spread_over_target",
    "energy_bias_over_target", "sigma_bias_over_target", "expected_effective_cost_ratio",
    "median_effective_cost_ratio", "upper_decile_effective_cost_ratio",
    "median_total_cost_ratio", "certification_rate"})
PLATFORM_COUNTS = frozenset({"energy_functional_words", "residual_functional_words"})


def _rounding(energy, ground):
    return ROUNDING_ULPS * math.ulp(1.0) * max(1.0, abs(energy), abs(ground))


def _close(a, b, rel, abs_=1e-15) -> bool:
    return (isinstance(a, float) and isinstance(b, float)
            and math.isclose(a, b, rel_tol=rel, abs_tol=abs_))


def _derived(entry, key, value, problems, label):
    have = entry.get(key, "missing")
    if value is None or isinstance(value, (bool, int)) and not isinstance(value, float):
        ok = have == value and type(have) is type(value)
    else:
        ok = isinstance(have, float) and math.isclose(have, value, rel_tol=1e-12,
                                                      abs_tol=1e-15)
    if not ok:
        problems.append(f"{label}: {key} does not derive")


# ------------------------------------------------------------------ one prefix

def rank_problems(rank, size, label) -> tuple[list[str], bool]:
    problems = []
    if not isinstance(rank, dict) or set(rank) != RANK_KEYS:
        return [f"{label}: rank fields differ from the schema"], False
    smallest, largest = rank["smallest_overlap_eigenvalue"], rank["largest_overlap_eigenvalue"]
    condition = largest / smallest if smallest > 0.0 else None
    full = rank["effective_rank"] == size
    stable = (full and smallest >= 10.0 * TAU_S and condition <= MAX_CONDITION / 10.0)
    recorded = rank["overlap_condition_number"]
    derives = recorded is None if condition is None else _close(recorded, condition, 1e-12)
    if not derives:
        problems.append(f"{label}: overlap condition number does not derive")
    if rank["full_rank"] is not full or rank["rank_stable"] is not stable:
        problems.append(f"{label}: rank margin does not derive")
    return problems, stable


def priced_problems(config, size, entry, spectrum, label) -> tuple[list[str], str]:
    """Restate everything a priced prefix decides from its recorded inputs."""
    problems = []
    energy, variance = entry["ground_energy"], entry["variance"]
    scale = entry["cancellation_scale"]
    estimator, universes = entry["estimator"], entry["universes"]
    if set(universes) != q18s2_checker.UNIVERSE_KEYS or set(
            estimator) != q18s2_checker.ESTIMATOR_KEYS:
        return [f"{label}: estimator or universe fields differ from the schema"], "INVALID"
    if (universes["measured_sh_words"] != universes["raw_sh_word_universe"] - 1
            or universes["measured_combined_words"]
            != universes["raw_combined_word_universe"] - 1):
        problems.append(f"{label}: measured universes are not the raw ones minus the identity")
    sigma = math.sqrt(max(variance, 0.0))
    if not _close(estimator["matched_precision_half"], sigma / 4.0, 1e-12):
        problems.append(f"{label}: matched precision does not derive")
    precise, dense = entry["precise"], entry["dense"]
    agreement = config["certification"]
    bound = agreement["variance_agreement"] * abs(precise["variance"])
    unit = agreement["energy_agreement"] * max(1.0, abs(precise["energy"]))
    rule = config["derivative_validation"]
    linear_ok = sorted(entry["precise_derivatives"]) == sorted(rule["directions"])
    norms = {"energy": estimator["energy_functional_norm"],
             "variance": estimator["residual_functional_norm"]}
    for direction, row in entry["precise_derivatives"].items():
        if sorted(row) != ["energy", "variance"]:
            problems.append(f"{label}/{direction}: linearization sides differ")
            linear_ok = False
            continue
        for side, check in row.items():
            if set(check) != {"analytic", "numeric", "tolerance", "passes"}:
                problems.append(f"{label}/{direction}/{side}: linearization fields differ")
                linear_ok = False
                continue
            tolerance = rule["relative_tolerance"] * norms[side]
            passes = (math.isfinite(check["numeric"])
                      and abs(check["numeric"] - check["analytic"]) <= tolerance)
            if not _close(check["tolerance"], tolerance, 1e-12) or check["passes"] is not passes:
                problems.append(f"{label}/{direction}/{side}: linearization does not derive")
            linear_ok &= passes
    fixed_ok = sorted(entry["fixed_step"]) == sorted(rule["directions"])
    for direction, row in entry["fixed_step"].items():
        issues, passes = q18s1.derivative_problems(row, config["fixed_step_validation"],
                                                   estimator["residual_functional_norm"], scale)
        problems += [f"{label}/{direction}: fixed step {issue}" for issue in issues]
        fixed_ok &= passes
    grouping = config["grouping"]
    declared, alternative = grouping["declared_protocol"], grouping["alternative_protocol"]
    if sorted(entry["protocols"]) != sorted(p for p in (declared, alternative) if p):
        return problems + [f"{label}: protocols differ from the declaration"], "INVALID"
    ratios, partitions_ok, variances_ok = {}, True, True
    roles = {declared: "declared", alternative: "alternative"}
    functional_variance = estimator["functional_variance"]
    for protocol, price in entry["protocols"].items():
        if set(price) != q18s2_checker.PRICE_KEYS or price["role"] != roles[protocol]:
            return problems + [f"{label}/{protocol}: price fields or role differ"], "INVALID"
        partitions_ok &= (not price["partition_problems"]
                          and price["energy_partitioned_words"] == universes["measured_sh_words"]
                          and price["residual_partitioned_words"]
                          == universes["measured_combined_words"])
        if not price["variances_nonnegative"]:
            variances_ok = False
            ratios[protocol] = None
            continue
        for side in ("energy", "residual"):
            values = price[f"{side}_setting_variances"]
            if len(values) != price[f"{side}_settings"] or any(
                    not math.isfinite(v) or v < -1e-10 for v in values):
                problems.append(f"{label}/{protocol}: {side} setting variances are malformed")
            if not q18._close(math.fsum(max(v, 0.0) for v in values),
                              price[f"{side}_variance_one_shot"], rel=1e-12, abs_=1e-12):
                problems.append(f"{label}/{protocol}: {side} variance sum does not derive")
            if not q18._close(math.fsum(math.sqrt(max(v, 0.0)) for v in values),
                              price[f"{side}_neyman_sum"], rel=1e-12, abs_=1e-12):
                problems.append(f"{label}/{protocol}: {side} Neyman sum does not derive")
        if not q18._close(q18.derive_ratio(price, functional_variance), price["cost_ratio"],
                          rel=1e-12, abs_=0):
            problems.append(f"{label}/{protocol}: uniform ratio does not derive")
        neyman = q18.derive_neyman(price, functional_variance)
        if not q18._close(neyman, price["neyman_ratio"], rel=1e-12, abs_=0):
            problems.append(f"{label}/{protocol}: Neyman ratio does not derive")
        ratios[protocol] = neyman
        got = q18s2_checker.allocation_totals(price, functional_variance,
                                              config["allocation_diagnostic"])
        problems += [f"{label}/{protocol}: allocation {path} does not derive"
                     for path in q18s2_checker._typed_disagreements(
                         got, price["allocation_diagnostic"])]
    gap = entry["subspace_gap"]
    checks = {
        "full_rank": entry["rank"]["effective_rank"] == size,
        "ground_root_nondegenerate": gap is None or gap > GAP_FLOOR * max(1.0, abs(energy)),
        "energy_consistent_with_spectrum": energy >= spectrum["ground_energy"]
        - _rounding(energy, spectrum["ground_energy"]),
        "precise_energy_agrees": abs(energy - precise["energy"]) <= unit,
        "precise_variance_agrees": abs(variance - precise["variance"]) <= bound,
        "dense_energy_agrees": abs(dense["energy"] - precise["energy"]) <= unit,
        "dense_variance_agrees": abs(dense["variance"] - precise["variance"]) <= bound,
        "residual_functional_reproduces_variance": abs(functional_variance - variance)
        <= 1e-9 * (1.0 + scale),
        "residual_functional_mean_zero": abs(estimator["residual_reference_mean"])
        <= 1e-9 * (1.0 + scale),
        "linearization_validated": bool(linear_ok),
        "fixed_step_validated": bool(fixed_ok),
        "partitions_valid": bool(partitions_ok),
        "group_variances_nonnegative": variances_ok,
    }
    if (tuple(entry["deterministic_checks"]) != CHECK_KEYS
            or not q18s2_checker._strict_equal(entry["deterministic_checks"], checks)):
        problems.append(f"{label}: deterministic checks do not derive")
    threshold = config["statistic"]["max_ratio"]
    if not all(checks.values()) or any(value is None for value in ratios.values()):
        status = "INVALID"
    else:
        passes = [ratios[p] <= threshold for p in (declared, alternative) if p]
        status = ("AFFORDABLE" if all(passes) else
                  "PROHIBITIVE" if not any(passes) else "GROUPING_SENSITIVE")
    return problems, status


def prefix_problems(config, size, entry, spectrum) -> tuple[list[str], str, bool]:
    """One prefix from its recorded inputs: problems, status, domain."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    label = f"M={size}"
    if not isinstance(entry, dict):
        return [f"{label}: malformed prefix"], "INVALID", False
    problems = []
    if not q18s2_checker._duration(entry.get("seconds")):
        problems.append(f"{label}: seconds must be a non-negative number")
    rank_issues, stable = rank_problems(entry.get("rank"), size, label)
    problems += rank_issues
    energy, variance = entry["ground_energy"], entry["variance"]
    scale = entry["cancellation_scale"]
    ground, first = spectrum["ground_energy"], spectrum["first_excited_energy"]
    resolved = variance > RESOLUTION * scale
    priced = stable and resolved
    keys = BASE_KEYS | (PRICED_EXTRA if priced else {"precise"} if stable else set())
    if set(entry) != keys:
        return problems + [f"{label}: prefix fields differ from the record schema"], "INVALID", False
    if variance < -RESOLUTION * scale:
        problems.append(f"{label}: variance below minus its resolution floor (checker sanity "
                        "rule)")
    sigma = math.sqrt(max(variance, 0.0))
    certifying = resolved and energy + sigma < first and energy >= ground - _rounding(
        energy, ground)
    for key, value in {
        "basis_size": size, "error": energy - ground, "residual_norm": sigma,
        "variance_resolved": resolved, "variance_resolution": RESOLUTION * scale,
        "weinstein_upper": energy + sigma,
        "ground_weight_lower_bound": max(0.0, min(1.0, 1.0 - max(energy - ground, 0.0)
                                                  / (first - ground))),
        "temple_lower_bound": None if energy >= first else energy - variance / (first - energy),
        "certifying": certifying,
    }.items():
        _derived(entry, key, value, problems, label)
    if size == 1 and entry["subspace_gap"] is not None:
        problems.append(f"{label}: a one-root pencil has no gap")
    if not priced:
        status = "RANK_UNSTABLE" if not stable else "UNRESOLVED"
        if (entry["status"], entry["role"], entry["in_domain"]) != (status, "unpriced", False):
            problems.append(f"{label}: an unpriced prefix must stay unpriced as {status}")
        return problems, status, False
    issues, status = priced_problems(config, size, entry, spectrum, label)
    problems += issues
    if entry["status"] != status:
        problems.append(f"{label}: status does not derive")
    role = "decision" if certifying else "diagnostic"
    if entry["role"] != role or entry["in_domain"] is not certifying:
        problems.append(f"{label}: role or domain does not derive")
    return problems, status, certifying


# ------------------------------------------------------------------ the rest

def lineage_problems(config, record, predecessor, screen_record, ladder) -> tuple[list, bool]:
    lineage = record["lineage"]
    if set(lineage) != {"frozen_bank_entry", "spectral_reference", "frozen_bank",
                        "ladder_row", "passes"}:
        return ["lineage fields differ from the schema"], False
    spectrum, frozen = record["spectral_reference"], screen_record["spectral_reference"]
    final = record["basis"]["prefixes"][str(config["basis"]["prefixes"][-1])]
    expected = {
        "spectral_reference": {
            key: (abs(spectrum[key] - frozen[key]) <= producer.SPECTRUM_TOLERANCE
                  if isinstance(frozen[key], float) else spectrum[key] == frozen[key])
            for key in sorted(frozen)},
        "frozen_bank": q18s2_checker.restate_lineage(
            lineage["frozen_bank_entry"], predecessor, config["lineage_check"]),
        "ladder_row": {
            "energy": abs(final["ground_energy"] - ladder["energy"])
            <= declaration.FLOAT_TOLERANCE,
            "retained_rank": final["rank"]["effective_rank"] == ladder["retained_rank"]},
    }
    expected["passes"] = all(all(block.values()) for block in expected.values())
    problems = []
    for key, value in expected.items():
        if not q18s2_checker._strict_equal(lineage[key], value):
            problems.append(f"lineage {key} does not derive")
    return problems, expected["passes"]


def restate_summary(replicates, rule, target, exact_energy, exact_sigma, first_excited):
    """The practical summary and status, restated from the recorded replicates."""
    import numpy as np

    validation = rule["validation"]
    failures = sum(1 for row in replicates
                   if row["energy"]["failure"] or row["residual"]["failure"])
    good = [row for row in replicates
            if not row["energy"]["failure"] and not row["residual"]["failure"]]
    if len(good) < 2:
        return None, None, "VALIDATION_FAILED", failures

    def effective(side):
        return np.array([row[side]["pilot_shots"] + row[side]["production_shots"]
                         * (row[side]["standard_error"] / target) ** 2 for row in good])

    E = np.array([row["energy"]["estimate"] for row in good])
    S = np.array([row["residual"]["estimate"] for row in good])
    zE = (E - exact_energy) / np.array([row["energy"]["standard_error"] for row in good])
    zS = (S - exact_sigma) / np.array([row["residual"]["standard_error"] for row in good])
    seE = np.array([row["energy"]["standard_error"] for row in good])
    seS = np.array([row["residual"]["standard_error"] for row in good])
    residual, energy = effective("residual"), effective("energy")
    totals = np.array([row["residual"]["total_shots"] / row["energy"]["total_shots"]
                       for row in good])
    summary = {
        "valid_replicates": len(good),
        "energy_coverage": float(np.mean(np.abs(zE) <= 1.959963984540054)),
        "sigma_coverage": float(np.mean(np.abs(zS) <= 1.959963984540054)),
        "energy_z_spread": float(np.std(zE, ddof=1)),
        "sigma_z_spread": float(np.std(zS, ddof=1)),
        "energy_spread_over_target": float(np.std(E, ddof=1) / target),
        "sigma_spread_over_target": float(np.std(S, ddof=1) / target),
        "energy_bias_over_target": float((np.mean(E) - exact_energy) / target),
        "sigma_bias_over_target": float((np.mean(S) - exact_sigma) / target),
        "expected_effective_cost_ratio": float(np.mean(residual) / np.mean(energy)),
        "median_effective_cost_ratio": float(np.median(residual / energy)),
        "upper_decile_effective_cost_ratio": float(np.quantile(residual / energy, 0.9)),
        "median_total_cost_ratio": float(np.median(totals)),
        "certification_rate": float(np.mean(
            E + S + 1.6448536269514722 * np.sqrt(seE ** 2 + seS ** 2) < first_excited)),
    }
    checks = {
        "failures_within_limit": failures <= validation["max_failures"],
        "energy_coverage": summary["energy_coverage"] >= validation["min_coverage"],
        "sigma_coverage": summary["sigma_coverage"] >= validation["min_coverage"],
        "energy_calibrated": summary["energy_z_spread"] <= validation["max_z_spread"],
        "sigma_calibrated": summary["sigma_z_spread"] <= validation["max_z_spread"],
    }
    if not all(checks.values()):
        status = "VALIDATION_FAILED"
    elif summary["expected_effective_cost_ratio"] <= rule["max_ratio"]:
        status = "PRACTICAL_AFFORDABLE"
    else:
        status = "PRACTICAL_PROHIBITIVE"
    return summary, checks, status, failures


def restate_budget(price, variance, rule) -> dict:
    """Each side's target and pilot, restated from the recorded Neyman sums."""
    target = math.sqrt(variance) * rule["standard_error_fraction_of_sigma"]
    out = {}
    for side, settings, scale in (("energy", "energy_settings", 1.0),
                                  ("residual", "residual_settings", 1.0 / (4.0 * variance))):
        oracle = price[f"{side}_neyman_sum"] ** 2 * scale / target ** 2
        out[side] = {"target_standard_error": target, "oracle_production_shots": oracle,
                     "pilot_shots_per_setting": max(
                         rule["minimum_pilot_shots_per_setting"],
                         math.ceil(rule["pilot_fraction"] * oracle / price[settings]))}
    return out


def practical_problems(config, record, statuses_expected) -> tuple[list[str], dict]:
    """Restate each practical summary and status from its recorded replicates."""
    practical = record["practical"]
    problems = []
    if sorted(practical, key=int) != sorted(statuses_expected, key=int):
        return ["practical stage ran on prefixes other than the AFFORDABLE certifying ones"], {}
    statuses = {}
    rule = config["practical_allocation"]
    spectrum = record["spectral_reference"]
    for size, block in practical.items():
        entry = record["basis"]["prefixes"][size]
        replicates = block.get("replicates")
        if not isinstance(replicates, list) or len(replicates) != rule["replicate_count"]:
            problems.append(f"practical M={size}: replicate count differs from the declaration")
            statuses[size] = "VALIDATION_FAILED"
            continue
        price = entry["protocols"][config["grouping"]["declared_protocol"]]
        budget = restate_budget(price, entry["variance"], rule)
        if q18s2_checker._typed_disagreements(budget, block.get("budget")):
            problems.append(f"practical M={size}: budget does not derive")
        summary, checks, status, failures = restate_summary(
            replicates, rule, budget["energy"]["target_standard_error"],
            entry["ground_energy"], entry["residual_norm"], spectrum["first_excited_energy"])
        expected = {"budget": block.get("budget"), "replicates": replicates,
                    "failures": failures, "summary": summary, "status": status}
        if checks is not None:
            expected["validation_checks"] = checks
        if q18s2_checker._typed_disagreements(expected, block):
            problems.append(f"practical M={size}: summary or status does not derive")
        statuses[size] = status
    return problems, statuses


def rederivation_problems(config, record, config_bytes, predecessor, screen_record,
                          ladder) -> list[str]:
    problems = []
    if set(record) != TOP_KEYS:
        problems.append(f"record fields differ from the schema: {sorted(set(record) ^ TOP_KEYS)}")
        return problems
    problems += q18s2_checker.provenance_problems(record.get("provenance"))
    if not q18s2_checker._duration(record.get("elapsed_seconds")):
        problems.append("elapsed_seconds must be a non-negative number")
    for key, value in {
        "schema": producer.SCHEMA, "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": producer.CLAIM_BOUNDARY, "quantum_advantage_claim": False,
        "question": config["question"], "design": config["evidence"],
        "statistic": config["statistic"], "certification": config["certification"],
        "rank_margin": config["rank_margin"], "grouping": config["grouping"],
        "derivative_validation": config["derivative_validation"],
        "fixed_step_validation": config["fixed_step_validation"],
        "allocation_diagnostic": config["allocation_diagnostic"],
        "practical_allocation": config["practical_allocation"],
        "evidence_schema": "clifford_qc.evidence.v1", "evidence": producer.EVIDENCE,
        "declaration": {"gate": "benchmarks/check_phase15_certifying_krylov_declaration.py",
                        "design_status": config["evidence"]["design_status"],
                        "basis_selection": config["evidence"]["basis_selection"],
                        "config_claim_boundary_at_landing": config["claim_boundary"]},
    }.items():
        if not q18s2_checker._strict_equal(record.get(key), value):
            problems.append(f"record {key} differs from the declaration")
    frozen = config["measured_before_freezing"]
    spectrum = record["spectral_reference"]
    if declaration._compare(frozen["spectrum"], spectrum):
        problems.append("spectral reference differs from the frozen one")
    premises = {k: v for k, v in frozen["premises"].items() if k != "exact_power_limit"}
    if declaration._compare(premises, record["premises"]):
        problems.append("premises differ from the frozen ones")
    basis = record["basis"]
    if set(basis) != {"labels", "prefixes"} or basis["labels"] != config["basis"]["labels"]:
        return problems + ["basis identity differs from the declaration"]
    if sorted(basis["prefixes"], key=int) != [str(m) for m in config["basis"]["prefixes"]]:
        return problems + ["prefixes differ from the declaration"]
    statuses, domain = {}, []
    for size in config["basis"]["prefixes"]:
        entry = basis["prefixes"][str(size)]
        first = frozen["basis"][str(size)]
        if entry.get("rank", {}).get("effective_rank") != first["effective_rank"]:
            problems.append(f"M={size}: effective rank differs from the frozen first moments")
        issues, status, certifying = prefix_problems(config, size, entry, spectrum)
        problems += issues
        statuses[str(size)] = status
        if certifying and status not in ("RANK_UNSTABLE", "UNRESOLVED"):
            domain.append(str(size))
    issues, lineage_ok = lineage_problems(config, record, predecessor, screen_record, ladder)
    problems += issues
    in_domain = [statuses[size] for size in domain]
    if not lineage_ok or "INVALID" in in_domain:
        oracle = "INVALID"
    elif not in_domain:
        oracle = "UNREACHED"
    elif "AFFORDABLE" in in_domain:
        oracle = "OPEN"
    elif all(status == "PROHIBITIVE" for status in in_domain):
        oracle = "CLOSED"
    else:
        oracle = "GROUPING_SENSITIVE"
    targets = [size for size in domain if statuses[size] == "AFFORDABLE"] if oracle == "OPEN" else []
    issues, practical = practical_problems(config, record, targets)
    problems += issues
    if oracle != "OPEN":
        final = oracle
    elif "PRACTICAL_AFFORDABLE" in practical.values():
        final = "LICENSED"
    elif "VALIDATION_FAILED" in practical.values():
        final = "VALIDATION_FAILED"
    else:
        final = "PRACTICAL_PROHIBITIVE"
    expected = {"domain": domain, "statuses": statuses, "oracle_verdict": oracle,
                "practical_prefixes": targets, "practical_statuses": practical,
                "final_verdict": final, "consequence": config["consequences"][final],
                "rule": "frozen in the config; see decision_rule"}
    if not q18s2_checker._strict_equal(record["decision"], expected):
        problems.append("the decision does not derive")
    return problems


# ------------------------------------------------------------------ rebuild

def condition_tolerance(entry) -> float:
    """Relative rebuild tolerance for floats the double-precision pencil sets."""
    return max(1e-9, ROUNDING_GROWTH * EPSILON * entry["rank"]["retained_condition_number"])


def _tree_problems(mine, theirs, rel, path) -> list[str]:
    """Typed comparison; floats to ``rel`` (Neyman fields to ``max(rel, 1e-3)``)."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        if sorted(mine) != sorted(theirs):
            return [f"{path} keys"]
        out = []
        for key in mine:
            if key in ("seconds", "elapsed_seconds"):
                continue
            out += _tree_problems(mine[key], theirs[key], rel, f"{path}.{key}")
        return out
    if isinstance(mine, list) and isinstance(theirs, list):
        if len(mine) != len(theirs):
            return [f"{path} length"]
        if mine and all(isinstance(v, float) for v in mine + theirs):
            # Setting variances: rounding-level entries are pinned to the largest.
            floor = 1e-9 * max(abs(v) for v in theirs)
            return [] if all(abs(a - b) <= rel * abs(b) + floor
                             for a, b in zip(mine, theirs)) else [path]
        out = []
        for index, (a, b) in enumerate(zip(mine, theirs)):
            out += _tree_problems(a, b, rel, f"{path}[{index}]")
        return out
    if type(mine) is not type(theirs):
        return [f"{path} type"]
    if isinstance(mine, float):
        tolerance = max(rel, q18.NEYMAN_RELATIVE) if path.rsplit(".", 1)[-1] in (
            q18.NEYMAN_FIELDS | {"cost_ratio"}) else rel
        return [] if math.isclose(mine, theirs, rel_tol=tolerance, abs_tol=1e-12) else [path]
    return [] if mine == theirs else [path]


def recompute_problems(config, record, *, model=None, predecessor=None, screen_record=None,
                       ladder=None, frozen=None) -> list[str]:
    """Rerun the declared pipeline and compare it with the record."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    record = copy.deepcopy(record)
    mine = json.loads(json.dumps(producer.run(
        config, config_bytes=b"", model=model, predecessor=predecessor,
        screen_record=screen_record, ladder=ladder, frozen=frozen), allow_nan=False))
    problems = []
    for key in ("spectral_reference", "premises"):
        problems += [f"{key}{p} does not rebuild"
                     for p in _tree_problems(mine[key], record[key], 1e-9, "")]
    lineage_mine, lineage_theirs = mine["lineage"], record["lineage"]
    problems += [f"lineage{p} does not rebuild" for p in _tree_problems(
        {k: v for k, v in lineage_mine.items() if k != "frozen_bank_entry"},
        {k: v for k, v in lineage_theirs.items() if k != "frozen_bank_entry"}, 1e-9, "")]
    for size, rebuilt in mine["basis"]["prefixes"].items():
        stored = record["basis"]["prefixes"][size]
        rel = condition_tolerance(rebuilt)
        if "estimator" in rebuilt and "estimator" in stored:
            for key in PLATFORM_COUNTS:
                count = stored["estimator"].pop(key)
                rebuilt["estimator"].pop(key)
                ceiling = stored["universes"][
                    "measured_sh_words" if key.startswith("energy") else
                    "measured_combined_words"]
                if not (isinstance(count, int) and not isinstance(count, bool)
                        and 0 < count <= ceiling):
                    problems.append(f"M={size}.estimator.{key} lies outside its band")
            # <psi|G|psi> is zero; what is recorded is rounding, whose size and
            # sign are the BLAS kernel's. Compare it to the bound its declared
            # check reads, 1e-9 (1 + cancellation_scale), not to its own last bits.
            bound = 1e-9 * (1.0 + rebuilt["cancellation_scale"])
            mean = stored["estimator"].pop("residual_reference_mean")
            if not (isinstance(mean, float) and abs(
                    rebuilt["estimator"].pop("residual_reference_mean") - mean) <= bound):
                problems.append(f"M={size}.estimator.residual_reference_mean does not rebuild "
                                "within its declared bound")
            for prices in (rebuilt["protocols"], stored["protocols"]):
                for price in prices.values():
                    price.pop("allocation_diagnostic", None)
        if not rebuilt["variance_resolved"] and not stored["variance_resolved"]:
            floor = RESOLUTION * rebuilt["cancellation_scale"]
            for key, tolerance in (("variance", floor), ("residual_norm", math.sqrt(floor)),
                                   ("weinstein_upper", math.sqrt(floor))):
                if not abs(rebuilt.pop(key) - stored.pop(key)) <= tolerance:
                    problems.append(f"M={size}.{key} does not rebuild within the floor")
            for key in ("temple_lower_bound",):
                a, b = rebuilt.pop(key), stored.pop(key)
                if (a is None) != (b is None):
                    problems.append(f"M={size}.{key} does not rebuild")
        if "precise" in rebuilt and "precise" in stored:
            problems += [f"M={size}.precise{p} does not rebuild" for p in _tree_problems(
                rebuilt.pop("precise"), stored.pop("precise"), 1e-12, "")]
        if "fixed_step" in rebuilt and "fixed_step" in stored:
            scale = rebuilt["cancellation_scale"]
            mine_rows, their_rows = rebuilt.pop("fixed_step"), stored.pop("fixed_step")
            if sorted(mine_rows) != sorted(their_rows):
                problems.append(f"M={size}: fixed-step directions do not rebuild")
            for direction, row in mine_rows.items():
                theirs = their_rows.get(direction, {"evaluations": [], "passes": None})
                for a, b in zip(row["evaluations"], theirs["evaluations"]):
                    for key in ("plus", "minus"):
                        if abs(a[key] - b[key]) > max(RESOLUTION * scale, rel * abs(b[key])):
                            problems.append(f"M={size}/{direction}: fixed-step {key} "
                                            "does not rebuild")
                    if a["domain_valid"] is not b["domain_valid"]:
                        problems.append(f"M={size}/{direction}: fixed-step domain "
                                        "does not rebuild")
                if row["passes"] is not theirs["passes"]:
                    problems.append(f"M={size}/{direction}: fixed-step verdict does not "
                                    "rebuild")
        problems += [f"M={size}{p} does not rebuild" for p in _tree_problems(
            rebuilt, stored, rel, "")]
    for key in ("decision", "practical"):
        problems += [f"{key}{p} does not rebuild"
                     for p in _tree_problems(mine[key], record[key], 1e-9, "")]
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-recompute", action="store_true",
                        help="rederive from the record only; skip the rebuild")
    args = parser.parse_args(argv)
    if not declaration.RECORD.exists():
        print("FAIL the Q18-S3 record is absent")
        return 1
    notes: list[str] = []
    try:
        config = declaration.load_config()
        record = q18s2_checker.load_record(declaration.RECORD)
        predecessor = json.loads(producer.PREDECESSOR.read_text(encoding="utf-8"))["banks"][
            declaration.SYSTEM]
        screen_record = json.loads(producer.SCREEN_RECORD.read_text(encoding="utf-8"))
        ladder = declaration.ladder_row()
        problems = declaration.static_problems(config)
        problems += rederivation_problems(config, record, declaration.CONFIG.read_bytes(),
                                          predecessor, screen_record, ladder)
        if not problems and not args.no_recompute:
            problems += recompute_problems(config, record, predecessor=predecessor,
                                           screen_record=screen_record, ladder=ladder)
        problems += declaration.commit_order_problems(notes)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            ZeroDivisionError) as exc:
        print(f"FAIL malformed Q18-S3 record: {exc}")
        return 1
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        decision = record["decision"]
        print(f"OK Q18-S3: oracle {decision['oracle_verdict']}, final "
              f"{decision['final_verdict']}; certifying rank-stable prefixes "
              f"{decision['domain']}")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
