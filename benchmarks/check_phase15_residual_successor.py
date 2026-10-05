"""Independently rederive and rebuild the separate post-hoc Q18-S1 record."""

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
sys.path.insert(0, str(ROOT))

from benchmarks import check_phase15_measured_residual_preflight as original
from benchmarks import check_phase15_residual_successor_declaration as declaration
from benchmarks import run_phase15_residual_successor as producer


def derivative_problems(row, rule, norm, scale):
    """Restate the Richardson and stability rule, trusting no outcome flag."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    problems = []
    evaluations = row["evaluations"]
    if [entry["step"] for entry in evaluations] != rule["steps"]:
        return ["derivative evaluation steps differ from the declaration"], False
    estimates = []
    relative = rule["relative_tolerance"] * norm
    for entry in evaluations:
        value = (entry["plus"] - entry["minus"]) / (2 * entry["step"])
        if not math.isfinite(value) or not original._close(value, entry["numeric"],
                                                           rel=1e-12, abs_=1e-14):
            problems.append("central difference is not its endpoints' quotient")
    for coarse, fine in zip(evaluations, evaluations[1:]):
        estimate = (4 * fine["numeric"] - coarse["numeric"]) / 3
        floor = RESOLUTION * scale * (4 / fine["step"] + 1 / coarse["step"]) / 3
        estimates.append({"coarse_step": coarse["step"], "fine_step": fine["step"],
                          "numeric": estimate, "rounding_floor": floor,
                          "tolerance": relative + floor,
                          "passes": abs(estimate - row["analytic"]) <= relative + floor})
    required = estimates[-2:]
    stability_tolerance = relative + required[0]["rounding_floor"] + required[1]["rounding_floor"]
    stable = abs(required[1]["numeric"] - required[0]["numeric"]) <= stability_tolerance
    domain = all(entry["domain_valid"] is True for entry in evaluations)
    passes = domain and stable and all(entry["passes"] for entry in required)
    expected = {"richardson": estimates, "numeric": estimates[-1]["numeric"],
                "tolerance": estimates[-1]["tolerance"], "stable": stable,
                "stability_tolerance": stability_tolerance, "domain_valid": domain,
                "passes": passes}
    for key, value in expected.items():
        if original._disagreements(value, row.get(key), key):
            problems.append(f"derivative {key} contradicts its evaluations")
    if set(row) != set(expected) | {"evaluations", "analytic"}:
        problems.append("derivative fields differ from the schema")
    return problems, passes and not problems


def allocation_problems(price, variance, rule):
    """Restate ceil-Neyman costs from recorded setting variances, no helper import."""
    diagnostic = price["allocation_diagnostic"]
    if not price["variances_nonnegative"]:
        return [] if diagnostic is None else ["invalid variances carry an allocation"]
    standard_error = math.sqrt(variance) * rule["standard_error_fraction_of_sigma"]
    expected = {"evidence": "oracle_integer_cost_with_charged_unperformed_pilot",
                "standard_error": standard_error, "estimator_licensed": False}
    for side in ("energy", "residual"):
        values = [max(v, 0.0) for v in price[f"{side}_setting_variances"]]
        if side == "residual":
            values = [v / (4 * variance) for v in values]
        roots = [math.sqrt(v) for v in values]
        total_root = math.fsum(roots)
        target = standard_error ** 2
        counts = [max(rule["minimum_shots_per_setting"], math.ceil(v * total_root / target))
                  for v in roots]
        achieved = math.fsum(v / n for v, n in zip(values, counts))
        pilot = len(values) * rule["pilot_shots_per_setting"]
        expected[side] = {"shots_per_setting": counts, "production_shots": sum(counts),
                          "pilot_shots": pilot, "total_shots": sum(counts) + pilot,
                          "achieved_variance": achieved, "target_variance": target,
                          "target_met": achieved <= target * (1 + 1e-12)}
    expected["total_cost_ratio"] = expected["residual"]["total_shots"] / expected["energy"][
        "total_shots"]
    return [f"allocation diagnostic {path} does not derive"
            for path in original._disagreements(expected, diagnostic)]


def rederivation_problems(config, record, config_bytes, validation):
    problems = []
    if len(config["revisions"]) > 1:
        correction = declaration.METADATA_CORRECTION
        if record.get("metadata_correction") != correction:
            problems.append("record must disclose the artifact-path metadata correction")
        executed = copy.deepcopy(record)
        executed.pop("metadata_correction", None)
        executed["config_digest"] = correction["execution_config_digest"]
        digest = hashlib.sha256((json.dumps(executed, indent=1) + "\n").encode()).hexdigest()
        if digest != correction["execution_record_sha256"]:
            problems.append("metadata correction changes the executed record beyond its config digest")
    elif "metadata_correction" in record:
        problems.append("record carries an undeclared metadata correction")
    for key, value in {
        "schema": producer.SCHEMA, "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": producer.CLAIM_BOUNDARY, "successor": config["successor"],
        "derivative_validation": config["derivative_validation"],
        "allocation_diagnostic": config["allocation_diagnostic"], "evidence": config["evidence"],
        "statistic": config["statistic"], "grouping": config["grouping"],
        "finite_difference": config["linearization"]["finite_difference"],
        "quantum_advantage_claim": False,
        "declaration": {"gate": "benchmarks/check_phase15_residual_successor_declaration.py",
                        "design_status": "post_hoc",
                        "config_claim_boundary_at_landing": config["claim_boundary"]},
    }.items():
        if record.get(key) != value:
            problems.append(f"record {key} differs from the declaration")
    problems += original.completeness_problems(config, record)
    if problems:
        return problems
    problems += original.freeze_problems(config, record)
    statuses = {}
    for name, entry in record["banks"].items():
        estimator = entry["estimator"]
        variance, scale = estimator["variance"], estimator["cancellation_scale"]
        root = validation["banks"][name]["roots"][0]
        if (not math.isfinite(variance) or variance <= 0 or
                not original._close(scale, root["cancellation_scale"], rel=0, abs_=0)):
            problems.append(f"{name}: invalid variance or cancellation scale")
            continue
        sigma = math.sqrt(variance)
        if not original._close(estimator["residual_norm"], sigma, rel=1e-12, abs_=0) or not (
                original._close(estimator["matched_precision_half"], sigma / 4, rel=1e-12, abs_=0)):
            problems.append(f"{name}: residual norm or precision does not derive")
        finite_ok = True
        for direction, row in entry["finite_differences"].items():
            issues, passes = derivative_problems(row, config["derivative_validation"],
                                                estimator["residual_functional_norm"], scale)
            problems += [f"{name}/{direction}: {issue}" for issue in issues]
            finite_ok &= passes
        ratios = {}
        for protocol, price in entry["protocols"].items():
            for side in ("energy", "residual"):
                values = price[f"{side}_setting_variances"]
                if len(values) != price[f"{side}_settings"]:
                    problems.append(f"{name}/{protocol}: setting variance count differs")
                if any(not math.isfinite(v) or v < -1e-10 for v in values):
                    problems.append(f"{name}/{protocol}: invalid setting variance")
                if not original._close(sum(max(v, 0.0) for v in values),
                                       price[f"{side}_variance_one_shot"], rel=1e-12, abs_=1e-12):
                    problems.append(f"{name}/{protocol}: variance sum does not derive")
                if not original._close(sum(math.sqrt(max(v, 0.0)) for v in values),
                                       price[f"{side}_neyman_sum"], rel=1e-12, abs_=1e-12):
                    problems.append(f"{name}/{protocol}: Neyman sum does not derive")
            ratio = original.derive_ratio(price, variance)
            ratios[protocol] = ratio
            if not original._close(ratio, price["cost_ratio"], rel=1e-12, abs_=0):
                problems.append(f"{name}/{protocol}: uniform ratio does not derive")
            if not original._close(original.derive_neyman(price, variance), price["neyman_ratio"],
                                   rel=1e-12, abs_=0):
                problems.append(f"{name}/{protocol}: Neyman ratio does not derive")
            problems += [f"{name}/{protocol}: {p}" for p in allocation_problems(
                price, variance, config["allocation_diagnostic"])]
        checks = original.derived_checks(config, name, entry, root, finite_ok=finite_ok)
        if checks != entry["deterministic_checks"]:
            problems.append(f"{name}: deterministic checks do not derive")
        if entry["mapping_axis_energy_variance"] != original.committed_energy_variance(name):
            problems.append(f"{name}: energy variance reference changed")
        grouping = config["grouping"]
        statuses[name] = original.derive_status(
            ratios, grouping["declared_protocol_by_system"][name],
            grouping["alternative_protocol_by_system"][name], config["statistic"]["max_ratio"], checks)
        if statuses[name] != entry["status"]:
            problems.append(f"{name}: status does not derive")
    verdict = original.derive_verdict(list(statuses.values()))
    expected = {"statuses": statuses, "verdict": verdict,
                "consequence": config["consequences"].get(verdict),
                "rule": "frozen in the config; see decision_rule"}
    if record["decision"] != expected:
        problems.append("uniform decision does not derive")
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("record must carry clean execution provenance")
    return problems


def recompute_problems(config, record, validation, *, inputs=None, banks=None):
    from clifford_qc.subspace.second_moment import RESOLUTION

    inputs = declaration.original.preflight_gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        model, selected = inputs(name)
        mine = producer.evaluate_bank(config, name, model, selected,
                                     config["measured_before_freezing"][name],
                                     validation["banks"][name]["roots"][0])
        theirs = copy.deepcopy(record["banks"][name])
        theirs.pop("measured_at_execution", None)
        scale = mine["estimator"]["cancellation_scale"]
        for direction, rebuilt in mine.pop("finite_differences").items():
            stored = theirs["finite_differences"][direction]
            # Rebuild endpoint values and analytic weights, not merely the outcome.
            for a, b in zip(rebuilt["evaluations"], stored["evaluations"]):
                for key in ("plus", "minus"):
                    if abs(a[key] - b[key]) > RESOLUTION * scale:
                        problems.append(f"{name}/{direction}: {key} endpoint does not rebuild")
                if a["domain_valid"] != b["domain_valid"]:
                    problems.append(f"{name}/{direction}: endpoint domain does not rebuild")
            if not original._close(rebuilt["analytic"], stored["analytic"], rel=1e-9, abs_=1e-12):
                problems.append(f"{name}/{direction}: analytic derivative does not rebuild")
        theirs.pop("finite_differences")
        for prices in (mine["protocols"], theirs["protocols"]):
            for price in prices.values():
                # Ceil counts can move by one across architectures. The recorded
                # counts are rederived exactly from its independently rebuilt
                # variance vector, rather than comparing ceil results across CPUs.
                price.pop("allocation_diagnostic")
        problems += [f"{name}: {path} does not rebuild"
                     for path in original._disagreements(mine, theirs)]
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-recompute", action="store_true")
    parser.add_argument("--banks", nargs="*", default=None)
    args = parser.parse_args(argv)
    if not declaration.RECORD.exists():
        print("FAIL successor record is absent")
        return 1
    try:
        config = declaration.load_config()
        record = json.loads(declaration.RECORD.read_text())
        validation = json.loads(declaration.original.VALIDATION.read_text())
        problems = declaration.static_problems(config)
        problems += rederivation_problems(config, record, declaration.CONFIG.read_bytes(), validation)
        if not problems and not args.no_recompute:
            problems += recompute_problems(config, record, validation, banks=args.banks)
        notes = []
        problems += declaration.commit_order_problems(notes)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"FAIL malformed successor record: {exc}")
        return 1
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK Q18-S1 post-hoc uniform decision: {record['decision']['verdict']}; "
              "allocation remains diagnostic")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
