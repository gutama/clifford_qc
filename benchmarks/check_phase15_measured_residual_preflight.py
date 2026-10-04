"""Verify the Phase 15 measured-residual cost record against its preregistration.

``run_phase15_measured_residual_preflight.py`` writes
``benchmarks/reference_results/phase15_measured_residual_preflight.json``. This
gate trusts none of the record's summary fields.

1. **Declaration.** The record names the config by digest, and the config is
   unchanged since the run. It carries the producer's claim boundary word for
   word, quotes the config's, makes no advantage claim, and copies the config's statistic,
   grouping and finite-difference rule unchanged.
2. **Completeness.** Every declared bank is present, priced under exactly its
   declared protocol and its alternative where one is declared, with both
   finite-difference directions.
3. **The freeze.** Each bank's ``measured_at_execution`` equals the config's
   ``measured_before_freezing``.
4. **Re-derivation.** Under an independent statement of the rule written here:
   every ratio from its recorded settings and variances, every Neyman ratio
   from its sums, the Cauchy-Schwarz bound each Neyman sum must respect, each
   finite-difference tolerance and outcome, every deterministic check from the
   values it rests on, each status, the verdict and its quoted consequence.
   The variance, cancellation scale and energy-variance reference are read
   from the committed validation and mapping-axis records, not the record.
5. **Recomputation.** Every bank is rebuilt and every field must reproduce,
   each finite-difference value included: ``numeric`` within its own rounding
   floor, never within the acceptance tolerance (``--no-recompute`` skips
   this, ``--banks`` restricts it).
6. **Order.** Through the preregistration gate, the config's last change must
   strictly precede the record's commit.

    python benchmarks/check_phase15_measured_residual_preflight.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_measured_residual_preregistration as gate
    from benchmarks import run_phase15_measured_residual_preflight as producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_measured_residual_preregistration as gate
    import run_phase15_measured_residual_preflight as producer

DENIALS = ("holds no ratio", "no ratio has", "computes no ratio")


def _close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ------------------------------------------------------------------ the rule, restated

def derive_ratio(price: dict, variance: float) -> float | None:
    """Shots for sigma over shots for E at one matched standard error.

    Total shots at standard error s are G V / s^2 for E, and G V_sigma2 /
    (4 sigma^2 s^2) for sigma, since d sigma = d sigma^2 / (2 sigma).
    """
    numerator = price["residual_settings"] * price["residual_variance_one_shot"]
    denominator = (4.0 * variance * price["energy_settings"]
                   * price["energy_variance_one_shot"])
    return numerator / denominator if denominator > 0.0 else None


def derive_neyman(price: dict, variance: float) -> float | None:
    """Both sides at variance-optimal allocation: (sum sqrt v)^2 replaces G V."""
    denominator = 4.0 * variance * price["energy_neyman_sum"] ** 2
    return price["residual_neyman_sum"] ** 2 / denominator if denominator > 0.0 else None


def derive_status(ratios: dict, declared: str, alternative: str | None,
                  threshold: float, checks: dict) -> str:
    if not all(checks.values()) or any(value is None for value in ratios.values()):
        return "INVALID"
    passes = [ratios[declared] <= threshold]
    if alternative is not None:
        passes.append(ratios[alternative] <= threshold)
    if all(passes):
        return "AFFORDABLE"
    if not any(passes):
        return "PROHIBITIVE"
    return "GROUPING_SENSITIVE"


def derive_verdict(statuses: list[str]) -> str:
    if not statuses:
        return "INCOMPLETE"
    if "INVALID" in statuses:
        return "INVALID"
    affordable = statuses.count("AFFORDABLE")
    if affordable == len(statuses):
        return "FULL"
    return "NONE" if affordable == 0 else "RESTRICTED"


def committed_energy_variance(name: str) -> float | None:
    """The mapping-axis record's own V_E for the JW arm, where it committed one."""
    record = json.loads(gate.MAPPING_RECORD.read_text(encoding="utf-8"))
    for system in record["systems"]:
        if system["system"] == name:
            jw = next(arm for arm in system["arms"] if arm["mapping"] == "jw")
            return jw["accuracy_matched_cost"].get(
                "linearized_variance_ha2_at_one_shot_per_setting")
    return None


# ------------------------------------------------------------------ checks

def declaration_problems(config: dict, record: dict, config_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != producer.SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("schema") != config["record_requirements"]["schema"]:
        problems.append("the record's schema is not the one the config requires")
    if record.get("config_digest") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("the config changed after the run: its digest no longer matches")
    boundary = str(record.get("claim_boundary", ""))
    if boundary != producer.CLAIM_BOUNDARY:
        problems.append("the record's claim boundary is not the producer's, word for word")
    if boundary == config["claim_boundary"]:
        problems.append("the record inherits the config's claim boundary instead of "
                        "stating its own")
    if any(phrase in boundary for phrase in DENIALS):
        problems.append("the record's claim boundary denies the ratios it reports")
    if record.get("preregistration", {}).get(
            "config_claim_boundary_at_landing") != config["claim_boundary"]:
        problems.append("the record does not quote the config's boundary at landing")
    if record.get("preregistration", {}).get("revision") != config["revisions"][-1]["revision"]:
        problems.append("the record does not name the config's last revision")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be false")
    for key, source in (("statistic", config["statistic"]), ("grouping", config["grouping"]),
                        ("finite_difference", config["linearization"]["finite_difference"]),
                        ("evidence", config["evidence"])):
        if record.get(key) != source:
            problems.append(f"the record's {key} is not the config's")
    return problems


def completeness_problems(config: dict, record: dict) -> list[str]:
    declared = config["banks"]["systems"]
    if sorted(record.get("banks", {})) != sorted(declared):
        return [f"banks {sorted(record.get('banks', {}))} are not the declared {sorted(declared)}"]
    problems = []
    grouping = config["grouping"]
    directions = config["linearization"]["finite_difference"]["directions"]
    for name in declared:
        entry = record["banks"][name]
        main = grouping["declared_protocol_by_system"][name]
        other = grouping["alternative_protocol_by_system"][name]
        expected = {main: "declared", **({other: "alternative"} if other else {})}
        got = {protocol: price.get("role") for protocol, price in entry["protocols"].items()}
        if got != expected:
            problems.append(f"{name}: protocols {got} are not the declared {expected}")
        if sorted(entry["finite_differences"]) != sorted(directions):
            problems.append(f"{name}: finite differences are not the frozen directions")
    return problems


def freeze_problems(config: dict, record: dict) -> list[str]:
    problems = []
    for name, entry in record["banks"].items():
        frozen = config["measured_before_freezing"][name]
        measured = entry.get("measured_at_execution", {})
        problems += [f"{name}: at execution, {problem}"
                     for problem in gate.compare(frozen, {k: measured.get(k)
                                                          for k in gate.MEASURED_FIELDS})]
    return problems


def derived_checks(config: dict, name: str, entry: dict, validation_root: dict) -> dict:
    """Every deterministic check, from the values it rests on."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    frozen = config["measured_before_freezing"][name]
    universes, estimator = entry["universes"], entry["estimator"]
    scale = float(validation_root["cancellation_scale"])
    declared = config["grouping"]["declared_protocol_by_system"][name]
    reference = committed_energy_variance(name)
    rule = config["linearization"]["finite_difference"]
    finite_ok = True
    for row in entry["finite_differences"].values():
        tolerance = (float(rule["relative_tolerance"]) * estimator["residual_functional_norm"]
                     + RESOLUTION * scale / float(rule["step"]))
        finite_ok &= abs(row["numeric"] - row["analytic"]) <= tolerance
    partitions_ok = True
    for price in entry["protocols"].values():
        partitions_ok &= (not price["partition_problems"]
                          and price["energy_partitioned_words"]
                          == universes["measured_sh_words"]
                          and price["residual_partitioned_words"]
                          == universes["measured_combined_words"])
    return {
        "combined_universe_matches_preflight": (
            universes["raw_combined_word_universe"] == frozen["raw_combined_word_universe"]
            and universes["raw_combined_words_sha256"] == frozen["raw_combined_words_sha256"]
            and universes["measured_combined_words"]
            == universes["raw_combined_word_universe"] - 1),
        "energy_settings_reproduce_mapping_axis": (
            entry["protocols"][declared]["energy_settings"] == frozen["energy_settings"]),
        "energy_variance_reproduces_mapping_axis": (
            reference is None or math.isclose(
                entry["protocols"][declared]["energy_variance_one_shot"], reference,
                rel_tol=producer.ENERGY_VARIANCE_RELATIVE, abs_tol=0.0)),
        "partitions_valid": bool(partitions_ok),
        "residual_functional_reproduces_variance": (
            abs(estimator["variance"] - float(validation_root["variance"]))
            <= producer.VARIANCE_TOLERANCE * (1.0 + scale)),
        "residual_functional_mean_zero": (
            abs(estimator["residual_reference_mean"])
            <= producer.MEAN_TOLERANCE * (1.0 + scale)),
        "linearization_matches_finite_differences": bool(finite_ok),
        "group_variances_nonnegative": all(
            price["variances_nonnegative"] for price in entry["protocols"].values()),
    }


def rederivation_problems(config: dict, record: dict, validation: dict) -> list[str]:
    from clifford_qc.subspace.second_moment import RESOLUTION

    problems = []
    statuses = {}
    threshold = float(config["statistic"]["max_ratio"])
    rule = config["linearization"]["finite_difference"]
    grouping = config["grouping"]
    for name, entry in record["banks"].items():
        root = validation["banks"][name]["roots"][0]
        estimator = entry["estimator"]
        variance = estimator["variance"]
        scale = float(root["cancellation_scale"])
        if not _close(estimator["cancellation_scale"], scale, rel=0.0, abs_=0.0):
            problems.append(f"{name}: the cancellation scale is not the validation record's")
        if not _close(estimator["residual_norm"], math.sqrt(max(variance, 0.0)),
                      rel=1e-12, abs_=0.0):
            problems.append(f"{name}: residual_norm is not the root of its variance")
        if not _close(estimator["matched_precision_half"], estimator["residual_norm"] / 4.0,
                      rel=1e-12, abs_=0.0):
            problems.append(f"{name}: matched_precision_half is not sigma / 4")
        universes = entry["universes"]
        if universes["measured_sh_words"] != universes["raw_sh_word_universe"] - 1:
            problems.append(f"{name}: the measured (S, H) universe is not the raw one "
                            "without the identity")
        for key, row in entry["finite_differences"].items():
            tolerance = (float(rule["relative_tolerance"])
                         * estimator["residual_functional_norm"]
                         + RESOLUTION * scale / float(rule["step"]))
            if not _close(row["tolerance"], tolerance, rel=1e-12, abs_=0.0):
                problems.append(f"{name}: the {key} tolerance is not the frozen rule's")
            if row["passes"] != (abs(row["numeric"] - row["analytic"]) <= tolerance):
                problems.append(f"{name}: the {key} outcome contradicts its values")
        ratios = {}
        for protocol, price in entry["protocols"].items():
            ratio = derive_ratio(price, variance) if price["variances_nonnegative"] else None
            ratios[protocol] = ratio
            if not _close(price["cost_ratio"], ratio, rel=1e-12, abs_=0.0):
                problems.append(f"{name}/{protocol}: R {price['cost_ratio']!r} is not the "
                                f"derived {ratio!r}")
            neyman = derive_neyman(price, variance) if price["variances_nonnegative"] else None
            if not _close(price["neyman_ratio"], neyman, rel=1e-12, abs_=0.0):
                problems.append(f"{name}/{protocol}: the Neyman ratio is not its sums'")
            for side, settings in (("energy", "energy_settings"),
                                   ("residual", "residual_settings")):
                bound = price[settings] * price[f"{side}_variance_one_shot"]
                if price[f"{side}_neyman_sum"] ** 2 > bound * (1.0 + 1e-9) + 1e-300:
                    problems.append(f"{name}/{protocol}: the {side} Neyman sum exceeds "
                                    "its Cauchy-Schwarz bound")
        checks = derived_checks(config, name, entry, root)
        if entry["deterministic_checks"] != checks:
            problems.append(f"{name}: recorded deterministic checks are not what their "
                            f"values give: {checks}")
        if entry["mapping_axis_energy_variance"] != committed_energy_variance(name):
            problems.append(f"{name}: the energy-variance reference is not the "
                            "mapping-axis record's")
        status = derive_status(ratios, grouping["declared_protocol_by_system"][name],
                               grouping["alternative_protocol_by_system"][name],
                               threshold, checks)
        statuses[name] = status
        if entry["status"] != status:
            problems.append(f"{name}: recorded status {entry['status']} is not the rule's "
                            f"{status}")
    declared = config["banks"]["systems"]
    verdict = derive_verdict([statuses[n] for n in declared if n in statuses])
    decision = record.get("decision", {})
    if decision.get("statuses") != {n: statuses[n] for n in declared if n in statuses}:
        problems.append("the summary's statuses are not the banks' own")
    if decision.get("verdict") != verdict:
        problems.append(f"recorded verdict {decision.get('verdict')} is not the rule's {verdict}")
    if decision.get("consequence") != config["consequences"].get(verdict):
        problems.append("the recorded consequence does not quote the config's for the verdict")
    return problems


def _disagreements(mine, theirs, path="") -> list[str]:
    """Paths where a rebuilt entry differs: exact for everything but floats,
    which must agree to 1e-9 relative (variances are sums of many terms)."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        if sorted(mine) != sorted(theirs):
            return [f"{path or '.'} keys"]
        out = []
        for key in mine:
            if key == "seconds":
                continue
            out += _disagreements(mine[key], theirs[key], f"{path}.{key}")
        return out
    if isinstance(mine, list) and isinstance(theirs, list):
        if len(mine) != len(theirs):
            return [f"{path} length"]
        out = []
        for index, (a, b) in enumerate(zip(mine, theirs)):
            out += _disagreements(a, b, f"{path}[{index}]")
        return out
    if isinstance(mine, float) or isinstance(theirs, float):
        if isinstance(mine, bool) or isinstance(theirs, bool):
            return [] if mine == theirs else [path]
        return [] if _close(mine, theirs, rel=1e-9, abs_=1e-12) else [path]
    return [] if mine == theirs else [path]


def finite_difference_disagreements(name: str, mine: dict, theirs: dict, *, scale: float,
                                    step: float) -> list[str]:
    """Every recorded finite-difference field against its rebuilt value.

    ``analytic`` and ``tolerance`` are smooth sums and must agree to 1e-9
    relative. ``numeric`` is a difference quotient, so a rebuild on another
    platform can move it by its own rounding, ``RESOLUTION * scale / step``,
    the floor the frozen tolerance already carries; it gets that much and
    1e-9 relative, and nothing of the acceptance tolerance itself.
    """
    from clifford_qc.subspace.second_moment import RESOLUTION

    problems = []
    if sorted(mine) != sorted(theirs):
        return [f"{name}: the finite-difference directions do not recompute"]
    floor = RESOLUTION * scale / step
    for key in mine:
        a, b = mine[key], theirs[key]
        agree = (
            a["passes"] == b.get("passes")
            and _close(a["analytic"], b.get("analytic"), rel=1e-9, abs_=1e-12)
            and _close(a["tolerance"], b.get("tolerance"), rel=1e-9, abs_=1e-15)
            and b.get("numeric") is not None
            and abs(a["numeric"] - b["numeric"]) <= 1e-9 * max(1.0, abs(a["numeric"])) + floor)
        if not agree:
            problems.append(f"{name}: the {key} finite difference recomputes to {a}")
    return problems


def recompute_problems(config: dict, record: dict, validation: dict, *, banks=None,
                       inputs=None) -> list[str]:
    """Every bank rebuilt; every recorded field must reproduce."""
    inputs = producer.preflight_gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        entry = record["banks"][name]
        model, selected = inputs(name)
        got = producer.evaluate_bank(config, name, model, selected,
                                     config["measured_before_freezing"][name],
                                     validation["banks"][name]["roots"][0])
        theirs = {key: value for key, value in entry.items()
                  if key != "measured_at_execution"}
        mine_fd, their_fd = got.pop("finite_differences"), theirs.pop("finite_differences")
        problems += finite_difference_disagreements(
            name, mine_fd, their_fd, scale=float(entry["estimator"]["cancellation_scale"]),
            step=float(config["linearization"]["finite_difference"]["step"]))
        for path in _disagreements(got, theirs):
            problems.append(f"{name}: {path.lstrip('.')} does not recompute")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--no-recompute", action="store_true",
                        help="re-derive everything from the record without rebuilding a bank")
    parser.add_argument("--banks", nargs="*", default=None, help="recompute only these banks")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    validation = json.loads(gate.VALIDATION.read_text(encoding="utf-8"))
    problems = declaration_problems(config, record, config_bytes)
    problems += completeness_problems(config, record)
    if not problems:
        problems += freeze_problems(config, record)
        problems += rederivation_problems(config, record, validation)
        if not args.no_recompute:
            problems += recompute_problems(config, record, validation, banks=args.banks)
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("the record's provenance does not show a clean tree")
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
    print(f"OK Phase 15 measured-residual record re-derives under the frozen rule: "
          f"verdict {decision['verdict']}, statuses {decision['statuses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
