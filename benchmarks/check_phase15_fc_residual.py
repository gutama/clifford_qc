"""Independently re-derive and rebuild the post-hoc Q18-S4 record.

Re-derivation trusts no outcome flag. From the recorded values alone it
restates, per bank and protocol, every setting count, one-shot variance,
Neyman sum, ratio and agreement flag; Q18-S1's Richardson rule on both
derivative directions; the agreement of both functionals with Q18-S1's
record; every deterministic check; each status off the ladder; and the
verdict off the combination rule. The rebuild re-runs every bank, compares
each partition by digest, and recomputes every setting variance a third way,
from its own Pauli action on the reference statevector.

The committed record carries one post-execution change: the v1 evidence
declaration the evidence gate requires, which the executed producer did not
write. Undoing it must restore the executed record's bytes, whose SHA-256 is
pinned here, and everything above is restated from that executed record.

    python benchmarks/check_phase15_fc_residual.py                 # restate and rebuild
    python benchmarks/check_phase15_fc_residual.py --no-recompute  # restatement only
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

from benchmarks import check_phase15_certifying_screen as provenance_rules
from benchmarks import check_phase15_fc_residual_declaration as declaration
from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as lineage
from benchmarks import check_phase15_residual_successor as q18s1_checker
from benchmarks import check_phase15_residual_successor_declaration as q18s1
from benchmarks import phase15_fc_residual as fc
from benchmarks import run_phase15_fc_residual as producer

RESOLUTION = 1e-12
THRESHOLD = 10.0
# The record as the declared execution wrote it. Its one post-execution change,
# the v1 evidence declaration, must undo to exactly these bytes.
EXECUTED_RECORD_SHA256 = "67d46ee4826ea3e25ca85814da7b4d9ba3367a979e12bdea9bbf7708979a0159"
PRICE_KEYS = frozenset({
    "role", "energy_settings", "energy_partitioned_words", "energy_largest_setting",
    "energy_partition_sha256", "energy_variance_one_shot", "energy_neyman_sum",
    "residual_settings", "residual_partitioned_words", "residual_largest_setting",
    "residual_partition_sha256", "residual_variance_one_shot", "residual_neyman_sum",
    "cost_ratio", "neyman_ratio", "partition_problems", "variances_nonnegative",
    "statevector_agreement", "statevector_worst_relative_deviation",
    "energy_setting_variances", "residual_setting_variances",
})
CHECK_KEYS = (
    "combined_universe_matches_preflight", "energy_settings_reproduce_freeze",
    "functionals_reproduce_q18s1", "partitions_valid",
    "residual_functional_reproduces_variance", "residual_functional_mean_zero",
    "linearization_matches_finite_differences", "group_variances_nonnegative",
    "group_variances_match_statevector",
)


def _close(a, b, rel=1e-12, abs_=1e-15) -> bool:
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)
            and math.isclose(a, b, rel_tol=rel, abs_tol=abs_))


def ladder(declared, alternative) -> str:
    """Q18's ladder in this checker's own words."""
    low = (declared <= THRESHOLD, alternative <= THRESHOLD)
    if all(low):
        return "AFFORDABLE"
    if not any(low):
        return "PROHIBITIVE"
    return "GROUPING_SENSITIVE"


def combine(statuses) -> str:
    statuses = list(statuses)
    if "INVALID" in statuses:
        return "INVALID"
    affordable = statuses.count("AFFORDABLE")
    if affordable == len(statuses):
        return "FULL"
    return "NONE" if affordable == 0 else "RESTRICTED"


# ------------------------------------------------------------------ correction

def correction_problems(record: dict, *, executed_sha256: str = EXECUTED_RECORD_SHA256
                        ) -> list[str]:
    """A corrected record must be the executed one plus the v1 evidence, exactly."""
    if "metadata_correction" not in record:
        return []
    executed = fc.executed_record(record)
    if hashlib.sha256(fc.record_bytes(executed)).hexdigest() != executed_sha256:
        return ["undoing the correction does not restore the executed record"]
    try:
        expected = fc.correct_evidence(executed, executed_sha256)
    except (ValueError, TypeError) as exc:
        return [f"the correction cannot be re-applied: {exc}"]
    if expected != record or list(expected) != list(record):
        return ["the record's correction is not the evidence-only one"]
    return []


# ------------------------------------------------------------------ declaration

def declaration_problems(config: dict, record: dict, config_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != declaration.ARTIFACT_PATHS["schema"]:
        problems.append("wrong record schema")
    if record.get("config_digest") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("the record was not produced from this declaration")
    if record.get("claim_boundary") != producer.CLAIM_BOUNDARY:
        problems.append("the claim boundary is not the producer's")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("the record must carry quantum_advantage_claim: false")
    for key, source in (("evidence", config["evidence"]), ("statistic", config["statistic"]),
                        ("grouping", config["grouping"]),
                        ("derivative_validation", config["derivative_validation"]),
                        ("excluded", config["banks"]["excluded"]),
                        ("successor", config["successor"])):
        if record.get(key) != source:
            problems.append(f"the record's {key} is not the declaration's")
    if record.get("declaration", {}).get("design_status") != "post_hoc":
        problems.append("the record must carry its post-hoc design status")
    if list(record.get("banks", {})) != list(config["banks"]["systems"]):
        problems.append("the record's banks are not the declared ones")
    problems += provenance_rules.provenance_problems(record.get("provenance"))
    return problems


# ------------------------------------------------------------------ restatement

def price_problems(price: dict, variance: float, label: str, n: int) -> list[str]:
    """One protocol's numbers, restated from its recorded setting variances."""
    if set(price) != PRICE_KEYS:
        return [f"{label}: price fields differ from the schema"]
    problems = []
    for side in ("energy", "residual"):
        values = price[f"{side}_setting_variances"]
        if len(values) != price[f"{side}_settings"]:
            problems.append(f"{label}: {side} setting count is not its variance list's")
        if not _close(math.fsum(values), price[f"{side}_variance_one_shot"],
                      abs_=1e-15 * max(1.0, len(values))):
            problems.append(f"{label}: {side} one-shot variance is not its settings' sum")
        if not _close(math.fsum(math.sqrt(max(v, 0.0)) for v in values),
                      price[f"{side}_neyman_sum"]):
            problems.append(f"{label}: {side} Neyman sum does not derive")
        if not 1 <= price[f"{side}_largest_setting"] <= (1 << n) - 1:
            problems.append(f"{label}: a {side} setting holds more words than a fully "
                            "commuting set can")
    nonnegative = all(v >= 0.0 for side in ("energy", "residual")
                      for v in price[f"{side}_setting_variances"])
    if price["variances_nonnegative"] is not nonnegative:
        problems.append(f"{label}: variances_nonnegative does not derive")
    agreement = bool(nonnegative and price["statevector_worst_relative_deviation"]
                     <= fc.STATEVECTOR_TOLERANCE)
    if price["statevector_agreement"] is not agreement:
        problems.append(f"{label}: statevector_agreement does not derive")
    energy, residual = price["energy_variance_one_shot"], price["residual_variance_one_shot"]
    if nonnegative and energy > 0.0 and variance > 0.0:
        ratio = (price["residual_settings"] * residual) / (
            4.0 * variance * price["energy_settings"] * energy)
        neyman = price["residual_neyman_sum"] ** 2 / (
            4.0 * variance * price["energy_neyman_sum"] ** 2)
        if not _close(price["cost_ratio"], ratio):
            problems.append(f"{label}: R does not derive")
        if not _close(price["neyman_ratio"], neyman):
            problems.append(f"{label}: the Neyman ratio does not derive")
    elif price["cost_ratio"] is not None:
        problems.append(f"{label}: an unpriced protocol carries a ratio")
    return problems


def bank_problems(config: dict, name: str, entry: dict, predecessor: dict,
                  validation_root: dict) -> tuple[list[str], str]:
    """Every check and the status of one bank, from its recorded values."""
    problems = []
    frozen = config["measured_before_freezing"][name]
    estimator, universes = entry["estimator"], entry["universes"]
    scale = estimator["cancellation_scale"]
    variance = estimator["variance"]
    grouping = config["grouping"]
    declared = grouping["declared_protocol_by_system"][name]
    alternative = grouping["alternative_protocol_by_system"][name]
    if set(entry["protocols"]) != {declared, alternative}:
        return [f"{name}: the protocols are not the declared pair"], "INVALID"
    for protocol, price in entry["protocols"].items():
        problems += price_problems(price, variance, f"{name} {protocol}", entry["n_qubits"])
        role = "declared" if protocol == declared else "alternative"
        if price.get("role") != role:
            problems.append(f"{name} {protocol}: wrong role")
    if not _close(estimator["residual_norm"], math.sqrt(max(variance, 0.0))) or not _close(
            estimator["matched_precision_half"], estimator["residual_norm"] / 4.0):
        problems.append(f"{name}: the residual norm does not derive from sigma^2")
    if scale != validation_root["cancellation_scale"]:
        problems.append(f"{name}: the cancellation scale is not the validation record's")

    derivatives = {}
    for key in config["linearization"]["finite_difference"]["directions"]:
        found, passes = q18s1_checker.derivative_problems(
            entry["finite_differences"][key], config["derivative_validation"],
            estimator["residual_functional_norm"], scale)
        problems += [f"{name} {key}: {p}" for p in found]
        derivatives[key] = passes

    theirs = predecessor["estimator"]
    anchor = entry["predecessor_anchor"]
    agrees = {}
    for key, row in anchor.items():
        if row["executed"] != estimator[key] or row["q18s1"] != theirs[key]:
            problems.append(f"{name}: the Q18-S1 anchor misquotes {key}")
        if key.endswith("_words"):
            agrees[key] = estimator[key] == theirs[key]
        else:
            agrees[key] = math.isclose(estimator[key], theirs[key],
                                       rel_tol=fc.PREDECESSOR_RELATIVE, abs_tol=1e-12)
        if row["agrees"] is not agrees[key]:
            problems.append(f"{name}: the Q18-S1 anchor's {key} flag does not derive")

    prices = entry["protocols"].values()
    expected = {
        "combined_universe_matches_preflight": (
            universes["raw_combined_word_universe"] == frozen["raw_combined_word_universe"]
            and universes["raw_combined_words_sha256"] == frozen["raw_combined_words_sha256"]
            and universes["measured_combined_words"]
            == universes["raw_combined_word_universe"] - 1),
        "energy_settings_reproduce_freeze": all(
            entry["protocols"][p]["energy_settings"] == frozen["fc_energy_settings"][p]
            for p in entry["protocols"]),
        "functionals_reproduce_q18s1": bool(agrees) and all(agrees.values()),
        "partitions_valid": all(
            not p["partition_problems"]
            and p["energy_partitioned_words"] == universes["measured_sh_words"]
            and p["residual_partitioned_words"] == universes["measured_combined_words"]
            for p in prices),
        "residual_functional_reproduces_variance": (
            abs(variance - validation_root["variance"]) <= fc.VARIANCE_TOLERANCE * (1 + scale)),
        "residual_functional_mean_zero": (
            abs(estimator["residual_reference_mean"]) <= fc.MEAN_TOLERANCE * (1 + scale)),
        "linearization_matches_finite_differences": bool(derivatives) and all(
            derivatives.values()),
        "group_variances_nonnegative": all(p["variances_nonnegative"] for p in prices),
        "group_variances_match_statevector": all(p["statevector_agreement"] for p in prices),
    }
    checks = entry["deterministic_checks"]
    if tuple(checks) != CHECK_KEYS:
        problems.append(f"{name}: deterministic check names differ from the schema")
    for key, value in expected.items():
        if checks.get(key) is not value:
            problems.append(f"{name}: {key} does not derive")
    ratios = [entry["protocols"][p]["cost_ratio"] for p in (declared, alternative)]
    if all(expected.values()) and None not in ratios:
        status = ladder(*ratios)
    else:
        status = "INVALID"
    if entry["status"] != status:
        problems.append(f"{name}: status {entry['status']} does not derive ({status})")
    return problems, status


def rederivation_problems(config: dict, record: dict, config_bytes: bytes, *,
                          predecessor: dict | None = None,
                          validation: dict | None = None) -> list[str]:
    if predecessor is None:
        predecessor = json.loads(q18s1.RECORD.read_text(encoding="utf-8"))
    if validation is None:
        validation = json.loads(q18.VALIDATION.read_text(encoding="utf-8"))
    problems = correction_problems(record)
    record = fc.executed_record(record)
    problems += declaration_problems(config, record, config_bytes)
    statuses = {}
    for name in config["banks"]["systems"]:
        entry = record["banks"].get(name)
        if entry is None:
            problems.append(f"{name}: missing from the record")
            continue
        found, statuses[name] = bank_problems(config, name, entry, predecessor["banks"][name],
                                              validation["banks"][name]["roots"][0])
        problems += found
    decision = record["decision"]
    verdict = combine(statuses.values())
    if decision["statuses"] != statuses:
        problems.append("the decision's statuses do not derive")
    if decision["verdict"] != verdict:
        problems.append(f"verdict {decision['verdict']} does not derive ({verdict})")
    if decision["consequence"] != config["consequences"].get(verdict):
        problems.append("the consequence is not the declared one for the verdict")
    return problems


# ------------------------------------------------------------------ rebuild

def pauli_image(n: int, code: int, psi):
    """``P|psi>`` from this checker's own reading of the packed letters.

    Qubit ``j`` is bit ``n - 1 - j`` of a basis index. ``X`` flips the bit,
    ``Z`` signs it, and ``Y = iXZ``.
    """
    import numpy as np

    index = np.arange(psi.size)
    flip = sign = 0
    phase = 1.0 + 0.0j
    for qubit in range(n):
        letter = (code >> (2 * qubit)) & 3
        bit = 1 << (n - 1 - qubit)
        if letter in (1, 2):
            flip |= bit
        if letter in (2, 3):
            sign |= bit
        if letter == 2:
            phase *= 1j
    parity = np.zeros(psi.size, dtype=np.int64)
    masked = index & sign
    while masked.any():
        parity ^= masked & 1
        masked >>= 1
    out = np.zeros_like(psi)
    out[index ^ flip] = phase * np.where(parity, -1.0, 1.0) * psi
    return out


def own_variances(psi, n: int, groups, weights) -> list[float]:
    import numpy as np

    out = []
    for group in groups:
        image = np.zeros_like(psi)
        for code in group:
            if weights.get(code, 0.0) != 0.0:
                image = image + weights[code] * pauli_image(n, code, psi)
        out.append(float(np.vdot(image, image).real) - float(np.vdot(psi, image).real) ** 2)
    return out


def _rebuild_tolerance(variance: float, scale: float) -> float:
    """Relative drift ``sigma^2`` and everything divided by it may show across
    platforms: a rounding floor on the cancellation scale, over ``sigma^2``."""
    return 1e-6 + 4.0 * RESOLUTION * (1.0 + scale) / variance


def recompute_problems(config: dict, record: dict, *, inputs=None, validation=None,
                       predecessor=None, progress=None) -> list[str]:
    """Rebuild every bank; compare it with the record and recompute each variance."""
    inputs = lineage.bank_inputs if inputs is None else inputs
    if validation is None:
        validation = json.loads(q18.VALIDATION.read_text(encoding="utf-8"))
    if predecessor is None:
        predecessor = json.loads(q18s1.RECORD.read_text(encoding="utf-8"))
    problems = []
    record = fc.executed_record(record)
    for name, theirs in record["banks"].items():
        model, selected = inputs(name)
        internals: dict = {}
        mine = fc.evaluate_bank(config, name, model, selected,
                                config["measured_before_freezing"][name],
                                validation["banks"][name]["roots"][0],
                                predecessor["banks"][name], internals=internals)
        mine = json.loads(json.dumps(mine, allow_nan=False))
        for key in ("n_qubits", "basis_size", "universes", "deterministic_checks", "status"):
            if mine[key] != theirs[key]:
                problems.append(f"{name}.{key} does not rebuild")
        scale = theirs["estimator"]["cancellation_scale"]
        variance = theirs["estimator"]["variance"]
        drift = _rebuild_tolerance(variance, scale)
        if abs(mine["estimator"]["variance"] - variance) > RESOLUTION * (1 + scale) * 4:
            problems.append(f"{name}.estimator.variance does not rebuild")
        for key in ("ground_energy", "residual_functional_norm", "energy_functional_norm"):
            if not _close(mine["estimator"][key], theirs["estimator"][key], rel=1e-9,
                          abs_=1e-12):
                problems.append(f"{name}.estimator.{key} does not rebuild")
        for protocol, price in theirs["protocols"].items():
            got = mine["protocols"][protocol]
            label = f"{name} {protocol}"
            for key in ("energy_settings", "residual_settings", "energy_partition_sha256",
                        "residual_partition_sha256", "energy_partitioned_words",
                        "residual_partitioned_words", "energy_largest_setting",
                        "residual_largest_setting", "partition_problems",
                        "variances_nonnegative", "statevector_agreement"):
                if got[key] != price[key]:
                    problems.append(f"{label}.{key} does not rebuild")
            if not _close(got["energy_variance_one_shot"], price["energy_variance_one_shot"],
                          rel=1e-9, abs_=1e-15):
                problems.append(f"{label}.energy_variance_one_shot does not rebuild")
            for key in ("residual_variance_one_shot", "cost_ratio", "neyman_ratio"):
                if (got[key] is None) != (price[key] is None) or (
                        got[key] is not None and not _close(got[key], price[key], rel=drift,
                                                            abs_=1e-15)):
                    problems.append(f"{label}.{key} does not rebuild")
            for side, words, weights in (
                    ("energy", internals["sh_words"], internals["energy_weights"]),
                    ("residual", internals["combined"], internals["residual_weights"])):
                groups = fc.fc_partition(words, protocol, model.n)
                if fc.partition_digest(groups) != price[f"{side}_partition_sha256"]:
                    problems.append(f"{label}: the {side} partition does not rebuild")
                    continue
                own = own_variances(internals["psi"], model.n, groups, weights)
                rebuilt = got[f"{side}_setting_variances"]
                recorded = price[f"{side}_setting_variances"]
                relative = 1e-9 if side == "energy" else drift
                if len(recorded) != len(rebuilt) or any(
                        not _close(a, b, rel=relative, abs_=1e-15)
                        for a, b in zip(recorded, rebuilt)):
                    problems.append(f"{label}.{side}_setting_variances does not rebuild")
                if len(own) != len(rebuilt) or any(
                        abs(a - b) > fc.STATEVECTOR_TOLERANCE * max(1.0, abs(b))
                        for a, b in zip(own, rebuilt)):
                    problems.append(f"{label}: a {side} setting variance disagrees with "
                                    "this checker's own Pauli action")
        if progress is not None:
            progress(f"  {name}: rebuilt")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-recompute", action="store_true",
                        help="restate the record without rebuilding the banks")
    args = parser.parse_args(argv)
    if not producer.RECORD.exists():
        print(f"FAIL no record at {producer.RECORD}")
        return 1
    config_bytes = declaration.CONFIG.read_bytes()
    config = declaration.load_config()
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    notes: list[str] = []
    problems = [f"declaration: {p}" for p in declaration.static_problems(config)]
    problems += declaration.commit_order_problems(notes)
    problems += rederivation_problems(config, record, config_bytes)
    if not args.no_recompute and not problems:
        problems += recompute_problems(config, record, progress=print)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    decision = record["decision"]
    ratios = ", ".join(
        f"{name} " + "/".join(f"{price['cost_ratio']:.4g}"
                              for price in entry["protocols"].values())
        for name, entry in record["banks"].items())
    print(f"OK Q18-S4: verdict {decision['verdict']}; fully commuting R (greedy/cover): "
          f"{ratios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
