"""Rederive and rebuild the Q18-S2 record: the screen on certifying Hubbard bases.

Two layers, as for Q18-S1. The rederivation trusts no outcome flag in the
record. It restates the certifying predicate from the recorded energy,
variance and spectral reference, the Richardson rule from the recorded
endpoints, every ratio and integer-allocation total from the recorded
per-setting variances, every deterministic check from its inputs, the
frozen-prefix lineage from the config's own tolerances, each prefix status,
and the verdict. Every key set in the record is closed, so a field the
producer never writes is a failure rather than a passenger. A prefix outside
the declared regime -- an energy below the exact ground energy, or a variance
below minus its resolution floor -- fails the record, because the frozen
verdict rule has no status for it. The rebuild then recomputes every prefix
from committed inputs and compares it with the record.

    python benchmarks/check_phase15_certifying_screen.py [--no-recompute]
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

from benchmarks import check_phase15_certifying_screen_declaration as declaration
from benchmarks import check_phase15_measured_residual_preflight as q18
from benchmarks import check_phase15_residual_successor as q18s1
from benchmarks import run_phase15_certifying_screen as producer

ROUNDING_ULPS = 100.0
TOP_KEYS = frozenset({
    "schema", "config_path", "config_digest", "claim_boundary", "quantum_advantage_claim",
    "question", "evidence", "statistic", "domain", "grouping", "derivative_validation",
    "allocation_diagnostic", "spectral_reference", "premises", "trajectories", "lineage",
    "decision", "declaration", "elapsed_seconds", "provenance"})
BLOCK_KEYS = frozenset({"candidate_family", "pool_size", "labels", "prefixes"})
PREFIX_KEYS = frozenset({
    "basis_size", "ground_energy", "ladder_energy", "error", "variance", "residual_norm",
    "variance_resolved", "cancellation_scale", "variance_resolution", "weinstein_upper",
    "ground_weight_lower_bound", "temple_lower_bound", "certifying", "effective_rank",
    "subspace_gap", "ground_dominated", "in_domain", "role", "status", "protocols",
    "deterministic_checks", "seconds"})
PRICED_KEYS = PREFIX_KEYS | {"universes", "estimator", "finite_differences"}
UNIVERSE_KEYS = frozenset({"raw_sh_word_universe", "raw_combined_word_universe",
                           "measured_sh_words", "measured_combined_words"})
ESTIMATOR_KEYS = frozenset({
    "functional_variance", "matched_precision_half", "residual_functional_words",
    "residual_functional_norm", "residual_identity_weight", "residual_reference_mean",
    "energy_functional_words", "energy_functional_norm"})
PRICE_KEYS = frozenset({
    "role", "energy_settings", "energy_partitioned_words", "energy_variance_one_shot",
    "energy_neyman_sum", "residual_settings", "residual_partitioned_words",
    "residual_variance_one_shot", "residual_neyman_sum", "cost_ratio", "neyman_ratio",
    "partition_problems", "variances_nonnegative", "energy_setting_variances",
    "residual_setting_variances", "allocation_diagnostic"})
EVALUATION_KEYS = frozenset({"step", "plus", "minus", "numeric", "domain_valid"})
LINEAGE_FIELDS = (
    "energy_settings", "residual_settings", "energy_partitioned_words",
    "residual_partitioned_words", "energy_variance_one_shot", "residual_variance_one_shot",
    "energy_neyman_sum", "residual_neyman_sum", "cost_ratio", "neyman_ratio")
NEYMAN_LINEAGE_FIELDS = frozenset({"energy_neyman_sum", "residual_neyman_sum", "neyman_ratio"})
CHECK_KEYS = ("full_rank", "ground_root_nondegenerate", "energy_reproduces_ladder",
              "energy_consistent_with_spectrum", "residual_functional_reproduces_variance",
              "residual_functional_mean_zero", "linearization_validated", "partitions_valid",
              "group_variances_nonnegative")


def _rounding(energy, ground):
    return ROUNDING_ULPS * math.ulp(1.0) * max(1.0, abs(energy), abs(ground))


def allocation_totals(price, variance, rule) -> dict:
    """Restate the integer allocation's totals from the recorded variances."""
    standard_error = math.sqrt(variance) * rule["standard_error_fraction_of_sigma"]
    target = standard_error ** 2
    out = {"evidence": "oracle_integer_cost_with_charged_unperformed_pilot",
           "standard_error": standard_error, "estimator_licensed": False}
    for side, scale in (("energy", 1.0), ("residual", 1.0 / (4.0 * variance))):
        values = [max(v, 0.0) * scale for v in price[f"{side}_setting_variances"]]
        roots = [math.sqrt(v) for v in values]
        total_root = math.fsum(roots)
        counts = [max(rule["minimum_shots_per_setting"], math.ceil(r * total_root / target))
                  for r in roots]
        achieved = math.fsum(v / n for v, n in zip(values, counts))
        pilot = len(values) * rule["pilot_shots_per_setting"]
        out[side] = {"production_shots": sum(counts), "pilot_shots": pilot,
                     "total_shots": sum(counts) + pilot, "achieved_variance": achieved,
                     "target_variance": target,
                     "target_met": achieved <= target * (1.0 + 1e-12)}
    out["production_ratio"] = (out["residual"]["production_shots"]
                               / out["energy"]["production_shots"])
    out["total_cost_ratio"] = out["residual"]["total_shots"] / out["energy"]["total_shots"]
    return out


def _lineage_close(a, b, rel) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b and type(a) is type(b)
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12)


def restate_lineage(entry, predecessor, rule) -> dict:
    """The frozen-prefix lineage, restated from the config's own tolerances."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    rel, neyman_rel = rule["relative_tolerance"], rule["neyman_relative_tolerance"]
    out = {
        "ground_energy": _lineage_close(entry["ground_energy"],
                                        predecessor["estimator"]["ground_energy"], rel),
        "variance": _lineage_close(entry["variance"], predecessor["estimator"]["variance"], rel),
        "universes": all(entry.get("universes", {}).get(key) == predecessor["universes"][key]
                         for key in sorted(UNIVERSE_KEYS)),
    }
    for protocol, theirs in predecessor["protocols"].items():
        mine = entry.get("protocols", {}).get(protocol)
        ok = mine is not None
        if ok:
            for key in LINEAGE_FIELDS:
                ok &= _lineage_close(mine[key], theirs[key],
                                     neyman_rel if key in NEYMAN_LINEAGE_FIELDS else rel)
            for side in ("energy", "residual"):
                a, b = mine[f"{side}_setting_variances"], theirs[f"{side}_setting_variances"]
                ok &= len(a) == len(b) and all(abs(x - y) <= 1e-12 + rel * abs(y)
                                               for x, y in zip(a, b))
        out[f"{protocol}_prices"] = bool(ok)
    floor = RESOLUTION * float(entry["cancellation_scale"])
    for direction, theirs in predecessor["finite_differences"].items():
        mine = entry.get("finite_differences", {}).get(direction)
        ok = (mine is not None and len(mine["evaluations"]) == len(theirs["evaluations"])
              and _lineage_close(mine["analytic"], theirs["analytic"], rel))
        if ok:
            for a, b in zip(mine["evaluations"], theirs["evaluations"]):
                ok &= a["step"] == b["step"] and all(abs(a[key] - b[key]) <= floor
                                                     for key in ("plus", "minus"))
        out[f"{direction}_endpoints"] = bool(ok)
    return out


def prefix_problems(config, method, size, entry, spectrum, history
                    ) -> tuple[list[str], str, bool]:
    """Restate one prefix from its recorded inputs: problems, status, domain."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    label = f"{method}/{size}"
    problems = []
    energy, variance = entry["ground_energy"], entry["variance"]
    scale = entry["cancellation_scale"]
    ground, first = spectrum["ground_energy"], spectrum["first_excited_energy"]
    sigma = math.sqrt(max(variance, 0.0))
    resolved = variance > RESOLUTION * scale
    # The frozen rule has no status for these: either one means the spectral
    # reference or the second-moment pencil is wrong, not that the prefix
    # quietly leaves the domain.
    if energy < ground - _rounding(energy, ground):
        problems.append(f"{label}: energy below the exact ground energy, outside the "
                        "declared regime")
    if variance < -RESOLUTION * scale:
        problems.append(f"{label}: variance below minus its resolution floor, outside the "
                        "declared regime")
    if set(entry) != (PRICED_KEYS if resolved else PREFIX_KEYS):
        problems.append(f"{label}: prefix fields differ from the record schema")
        return problems, "INVALID", False
    certifying = resolved and energy + sigma < first and energy >= ground - _rounding(
        energy, ground)
    weight = max(0.0, min(1.0, 1.0 - max(energy - ground, 0.0) / (first - ground)))
    temple = None if energy >= first else energy - variance / (first - energy)
    expected = {
        "basis_size": size, "ladder_energy": history[size - 1], "error": energy - ground,
        "residual_norm": sigma, "variance_resolved": resolved,
        "variance_resolution": RESOLUTION * scale, "weinstein_upper": energy + sigma,
        "ground_weight_lower_bound": weight, "temple_lower_bound": temple,
        "certifying": certifying, "ground_dominated": weight > 0.5, "in_domain": certifying,
    }
    for key, value in expected.items():
        have = entry.get(key, "missing")
        if (value is None or isinstance(value, (bool, int))) and not isinstance(value, float):
            ok = have == value and type(have) is type(value)
        else:
            ok = isinstance(have, float) and math.isclose(have, value, rel_tol=1e-12,
                                                          abs_tol=1e-15)
        if not ok:
            problems.append(f"{label}: {key} does not derive")
    if not resolved:
        if (entry.get("status"), entry.get("role")) != ("UNRESOLVED", "unpriced") or entry.get(
                "protocols") or entry.get("deterministic_checks"):
            problems.append(f"{label}: an unresolved prefix must stay unpriced")
        return problems, "UNRESOLVED", False

    estimator = entry["estimator"]
    universes = entry["universes"]
    if set(estimator) != ESTIMATOR_KEYS or set(universes) != UNIVERSE_KEYS:
        problems.append(f"{label}: estimator or universe fields differ from the record schema")
        return problems, "INVALID", certifying
    if (universes["measured_sh_words"] != universes["raw_sh_word_universe"] - 1
            or universes["measured_combined_words"] != universes["raw_combined_word_universe"] - 1):
        problems.append(f"{label}: measured universes are not the raw ones minus the identity")
    functional_variance = estimator["functional_variance"]
    if not math.isclose(estimator["matched_precision_half"], sigma / 4.0, rel_tol=1e-12):
        problems.append(f"{label}: matched precision does not derive")
    finite_ok = True
    for direction, row in entry["finite_differences"].items():
        if any(set(evaluation) != EVALUATION_KEYS for evaluation in row.get("evaluations", [])):
            problems.append(f"{label}/{direction}: evaluation fields differ from the schema")
        issues, passes = q18s1.derivative_problems(
            row, config["derivative_validation"], estimator["residual_functional_norm"], scale)
        problems += [f"{label}/{direction}: {issue}" for issue in issues]
        finite_ok &= passes
    if sorted(entry["finite_differences"]) != sorted(config["derivative_validation"]["directions"]):
        problems.append(f"{label}: derivative directions differ from the declaration")
    grouping = config["grouping"]
    declared, alternative = grouping["declared_protocol"], grouping["alternative_protocol"]
    if sorted(entry["protocols"]) != sorted(p for p in (declared, alternative) if p):
        problems.append(f"{label}: protocols differ from the declaration")
        return problems, "INVALID", certifying
    ratios, partitions_ok, variances_ok = {}, True, True
    roles = {declared: "declared", alternative: "alternative"}
    for protocol, price in entry["protocols"].items():
        if set(price) != PRICE_KEYS or price["role"] != roles[protocol]:
            problems.append(f"{label}/{protocol}: price fields or role differ from the schema")
            return problems, "INVALID", certifying
        partitions_ok &= (not price["partition_problems"]
                          and price["energy_partitioned_words"] == universes["measured_sh_words"]
                          and price["residual_partitioned_words"]
                          == universes["measured_combined_words"])
        populated = all(len(price[f"{side}_setting_variances"]) == price[f"{side}_settings"]
                        for side in ("energy", "residual"))
        if price["variances_nonnegative"] is not populated:
            problems.append(f"{label}/{protocol}: variances_nonnegative contradicts its data")
        if not populated:
            # A negative group variance: price_protocol keeps no setting
            # variances and no ratio, and the prefix is INVALID by its check.
            variances_ok = False
            if (price["cost_ratio"] is not None or price["neyman_ratio"] is not None
                    or price["allocation_diagnostic"] is not None
                    or price["energy_setting_variances"] or price["residual_setting_variances"]):
                problems.append(f"{label}/{protocol}: unpriced variances carry a price")
            ratios[protocol] = None
            continue
        for side in ("energy", "residual"):
            values = price[f"{side}_setting_variances"]
            if len(values) != price[f"{side}_settings"]:
                problems.append(f"{label}/{protocol}: setting variance count differs")
            if any(not math.isfinite(v) or v < -1e-10 for v in values):
                problems.append(f"{label}/{protocol}: invalid setting variance")
            if not q18._close(math.fsum(max(v, 0.0) for v in values),
                              price[f"{side}_variance_one_shot"], rel=1e-12, abs_=1e-12):
                problems.append(f"{label}/{protocol}: {side} variance sum does not derive")
            if not q18._close(math.fsum(math.sqrt(max(v, 0.0)) for v in values),
                              price[f"{side}_neyman_sum"], rel=1e-12, abs_=1e-12):
                problems.append(f"{label}/{protocol}: {side} Neyman sum does not derive")
        uniform = q18.derive_ratio(price, functional_variance)
        neyman = q18.derive_neyman(price, functional_variance)
        if not q18._close(uniform, price["cost_ratio"], rel=1e-12, abs_=0):
            problems.append(f"{label}/{protocol}: uniform ratio does not derive")
        if not q18._close(neyman, price["neyman_ratio"], rel=1e-12, abs_=0):
            problems.append(f"{label}/{protocol}: Neyman ratio does not derive")
        ratios[protocol] = neyman
        got = allocation_totals(price, functional_variance, config["allocation_diagnostic"])
        problems += [f"{label}/{protocol}: allocation {path} does not derive"
                     for path in q18._disagreements(got, price["allocation_diagnostic"])]
    gap_floor = declaration.q18.GAP_FLOOR
    checks = {
        "full_rank": entry["effective_rank"] == size,
        "ground_root_nondegenerate": entry["subspace_gap"] > gap_floor * max(1.0, abs(energy)),
        "energy_reproduces_ladder": abs(energy - history[size - 1])
        <= config["deterministic_checks"]["history_tolerance"],
        "energy_consistent_with_spectrum": energy >= ground - _rounding(energy, ground),
        "residual_functional_reproduces_variance": abs(functional_variance - variance)
        <= 1e-9 * (1.0 + scale),
        "residual_functional_mean_zero": abs(estimator["residual_reference_mean"])
        <= 1e-9 * (1.0 + scale),
        "linearization_validated": finite_ok,
        "partitions_valid": partitions_ok,
        "group_variances_nonnegative": variances_ok,
    }
    if entry["deterministic_checks"] != checks or tuple(checks) != CHECK_KEYS:
        problems.append(f"{label}: deterministic checks do not derive")
    threshold = config["statistic"]["max_ratio"]
    if not all(checks.values()) or any(value is None for value in ratios.values()):
        status = "INVALID"
    else:
        passes = [ratios[declared] <= threshold]
        if alternative is not None:
            passes.append(ratios[alternative] <= threshold)
        status = ("AFFORDABLE" if all(passes) else
                  "PROHIBITIVE" if not any(passes) else "GROUPING_SENSITIVE")
    if entry["status"] != status:
        problems.append(f"{label}: status does not derive")
    if entry["role"] != ("decision" if certifying else "diagnostic"):
        problems.append(f"{label}: role does not derive")
    return problems, status, certifying


def rederivation_problems(config, record, config_bytes, predecessor) -> list[str]:
    problems = []
    if set(record) != TOP_KEYS:
        problems.append("record fields differ from the record schema: "
                        f"{sorted(set(record) ^ TOP_KEYS)}")
    for key, value in {
        "schema": producer.SCHEMA, "config_path": str(declaration.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": producer.CLAIM_BOUNDARY, "quantum_advantage_claim": False,
        "question": config["question"], "evidence": config["evidence"],
        "statistic": config["statistic"], "domain": config["domain"],
        "grouping": config["grouping"], "derivative_validation": config["derivative_validation"],
        "allocation_diagnostic": config["allocation_diagnostic"],
        "declaration": {"gate": "benchmarks/check_phase15_certifying_screen_declaration.py",
                        "design_status": config["evidence"]["design_status"],
                        "system_selection": config["evidence"]["system_selection"],
                        "config_claim_boundary_at_landing": config["claim_boundary"]},
    }.items():
        if record.get(key) != value:
            problems.append(f"record {key} differs from the declaration")
    frozen = config["measured_before_freezing"]
    spectrum = record["spectral_reference"]
    if declaration._compare(frozen["spectrum"], spectrum):
        problems.append("spectral reference differs from the frozen one")
    if declaration._compare(frozen["premises"], record["premises"]):
        problems.append("premises differ from the frozen ones")
    if sorted(record["trajectories"]) != sorted(config["trajectories"]["rows"]):
        return problems + ["trajectories differ from the declaration"]
    statuses, domain = {}, []
    for method, block in record["trajectories"].items():
        spec = config["trajectories"]["rows"][method]
        if set(block) != BLOCK_KEYS:
            problems.append(f"{method}: trajectory fields differ from the record schema")
            continue
        if block["labels"] != spec["labels"] or block["candidate_family"] != spec[
                "candidate_family"] or block["pool_size"] != frozen["trajectories"][method][
                "pool_size"]:
            problems.append(f"{method}: trajectory identity differs")
        if sorted(block["prefixes"], key=int) != [str(m) for m in spec["prefixes"]]:
            problems.append(f"{method}: prefixes differ from the declaration")
            continue
        for size in spec["prefixes"]:
            entry = block["prefixes"][str(size)]
            issues, status, certifying = prefix_problems(config, method, size, entry,
                                                         spectrum, spec["energy_history"])
            problems += issues
            statuses[f"{method}/{size}"] = status
            if certifying:
                domain.append(f"{method}/{size}")
    lineage = {method: restate_lineage(
        record["trajectories"][method]["prefixes"][str(declaration.FROZEN_BANK_SIZE)],
        predecessor, config["lineage_check"]) for method in record["trajectories"]}
    lineage["passes"] = all(all(checks.values()) for checks in lineage.values())
    if record["lineage"] != lineage:
        problems.append("the frozen-prefix lineage does not derive")
    in_domain = [statuses[key] for key in domain]
    if not lineage["passes"] or "INVALID" in in_domain:
        verdict = "INVALID"
    elif not in_domain:
        verdict = "UNREACHED"
    elif "AFFORDABLE" in in_domain:
        verdict = "OPEN"
    elif all(status == "PROHIBITIVE" for status in in_domain):
        verdict = "CLOSED"
    else:
        verdict = "GROUPING_SENSITIVE"
    expected = {"domain": domain, "statuses": statuses, "verdict": verdict,
                "consequence": config["consequences"][verdict],
                "rule": "frozen in the config; see decision_rule"}
    if record["decision"] != expected:
        problems.append("the decision does not derive")
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("the record must carry clean execution provenance")
    return problems


def recompute_problems(config, record, *, model=None, inputs=None, predecessor=None) -> list[str]:
    """Rebuild every prefix from committed inputs and compare."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    mine = producer.run_screen(config, config_bytes=b"", model=model, inputs=inputs,
                               predecessor=predecessor)
    problems = []
    for key in ("spectral_reference", "premises", "lineage"):
        problems += [f"{key}{path} does not rebuild"
                     for path in q18._disagreements(mine[key], record[key])]
    for method, block in mine["trajectories"].items():
        for size, rebuilt in block["prefixes"].items():
            stored = copy.deepcopy(record["trajectories"][method]["prefixes"][size])
            rebuilt = copy.deepcopy(rebuilt)
            scale = rebuilt["cancellation_scale"]
            rebuilt_fd = rebuilt.pop("finite_differences", {})
            stored_fd = stored.pop("finite_differences", {})
            if sorted(rebuilt_fd) != sorted(stored_fd):
                problems.append(f"{method}/{size}: derivative directions do not rebuild")
            if not rebuilt["variance_resolved"] and not stored.get("variance_resolved", True):
                # sigma^2 here is rounding, c'Kc - E^2 at the exact ground state:
                # its sign is the BLAS kernel's. Compare it, and what it feeds, to
                # the resolution floor rather than to the last bit.
                floor = RESOLUTION * scale
                width = math.sqrt(floor)
                for key, tolerance in (("variance", floor), ("residual_norm", width),
                                       ("weinstein_upper", width)):
                    a, b = rebuilt.pop(key), stored.pop(key, math.nan)
                    if not abs(a - b) <= tolerance:
                        problems.append(f"{method}/{size}.{key} does not rebuild within "
                                        "the resolution floor")
                a, b = rebuilt.pop("temple_lower_bound"), stored.pop("temple_lower_bound", "x")
                if (a is None) != (b is None) or (a is not None and not abs(a - b) <= floor / (
                        record["spectral_reference"]["first_excited_energy"]
                        - rebuilt["ground_energy"])):
                    problems.append(f"{method}/{size}.temple_lower_bound does not rebuild "
                                    "within the resolution floor")
            for direction, row in rebuilt_fd.items():
                theirs = stored_fd.get(direction, {"evaluations": [], "analytic": math.nan})
                for a, b in zip(row["evaluations"], theirs["evaluations"]):
                    for key in ("plus", "minus"):
                        if abs(a[key] - b[key]) > RESOLUTION * scale:
                            problems.append(f"{method}/{size}/{direction}: {key} endpoint "
                                            "does not rebuild")
                    if a["domain_valid"] != b["domain_valid"]:
                        problems.append(f"{method}/{size}/{direction}: endpoint domain "
                                        "does not rebuild")
                if not q18._close(row["analytic"], theirs["analytic"], rel=1e-9, abs_=1e-12):
                    problems.append(f"{method}/{size}/{direction}: analytic derivative "
                                    "does not rebuild")
            for prices in (rebuilt.get("protocols", {}), stored.get("protocols", {})):
                for price in prices.values():
                    # Ceilings can move by one shot across architectures; the totals
                    # are rederived exactly from the rebuilt variances instead.
                    price.pop("allocation_diagnostic", None)
            problems += [f"{method}/{size}{path} does not rebuild"
                         for path in q18._disagreements(rebuilt, stored)]
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-recompute", action="store_true",
                        help="rederive from the record only; skip the rebuild")
    args = parser.parse_args(argv)
    if not declaration.RECORD.exists():
        print("FAIL the Q18-S2 record is absent")
        return 1
    notes: list[str] = []
    try:
        config = declaration.load_config()
        record = json.loads(declaration.RECORD.read_text(encoding="utf-8"))
        predecessor = json.loads(producer.PREDECESSOR.read_text(encoding="utf-8"))["banks"][
            declaration.SYSTEM]
        problems = declaration.static_problems(config)
        problems += rederivation_problems(config, record, declaration.CONFIG.read_bytes(),
                                          predecessor)
        if not problems and not args.no_recompute:
            problems += recompute_problems(config, record)
        problems += declaration.commit_order_problems(notes)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(f"FAIL malformed Q18-S2 record: {exc}")
        return 1
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        decision = record["decision"]
        print(f"OK Q18-S2 oracle screen: {decision['verdict']}; certifying prefixes "
              f"{decision['domain']}; no estimator licensed")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
