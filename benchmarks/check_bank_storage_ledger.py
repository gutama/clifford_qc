"""Recompute and gate the Phase 2M-A bank storage ledger.

Two jobs, as the project's other checkers do. The record is rebuilt from
scratch and compared field by field, and the contracts it must satisfy are
re-derived from its own contents rather than trusted:

* the packed identity ``cached_operator_bytes == 24 * coefficient_occurrences``
  holds exactly on every measured bank, and the committed back-fill divides by
  the same constant with a zero remainder -- a remainder would mean the record
  was produced under a different packed model and the recovery is invalid
  rather than approximate;
* every pair count closes: retained plus selection equals built, the retained
  block never exceeds the ``M(M+1)/2`` it would need, and the retained fraction
  is the quotient of the two rather than a separately written number;
* the declared policy is ``retain_all`` and its consequences are checked rather
  than assumed -- eviction, recomputation and spill are zero, and the peak
  resident row count equals the current one, which is what "nothing is ever
  freed" *means* and would stop being true the moment 2M-C lands without
  updating this record;
* the frontier fields are exercised: at least one measured bank rejected
  candidates, so the retained-against-frontier arithmetic is measured on a live
  frontier rather than defined into existence on banks that never reject;
* the measured bytes decompose -- containers plus boxed objects -- and the
  boxed slots account for exactly two per coefficient, deduplicated plus
  shared, so no allocation is double-counted or dropped;
* coefficient reuse divides resident occurrences by the word universe of those
  same rows; selected-subspace ``W`` is carried separately, and old records that
  did not preserve rejected-row ``W`` leave true reuse unavailable;
* the committed rows still say what the record says they say: every back-filled
  field is re-read from ``molecular/results/`` rather than trusted as
  transcribed;
* the calibrated attribution is re-derived from the minimum observed rate, its
  threshold verdict follows mechanically, and no verdict is writable by hand;
* the record carries no go/no-go outcome: 2M's gate is a property of an
  implementation that does not exist yet, and a baseline that quietly graded
  itself would be the failure this phase exists to prevent;
* no cost-record field appears anywhere -- this tier prices nothing.

Measured byte counts are interpreter-dependent, so they are compared under a
declared relative tolerance rather than exactly; see
``INTERPRETER_DEPENDENT_FIELDS``. Every count, pair and packed byte stays exact.

    python benchmarks/check_bank_storage_ledger.py
"""

from __future__ import annotations

import json
import math
import sys

from clifford_qc.reproducibility import compare_json_records, guarded_contract_problems

try:  # package import in tests versus direct script execution
    from benchmarks.run_bank_storage_ledger import (
        BANK_DOMINANCE_FRACTION,
        PACKED_BYTES_PER_COEFFICIENT,
        REFERENCE,
        ROOT,
        SCHEMA,
        build_record,
        load_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_bank_storage_ledger import (
        BANK_DOMINANCE_FRACTION,
        PACKED_BYTES_PER_COEFFICIENT,
        REFERENCE,
        ROOT,
        SCHEMA,
        build_record,
        load_config,
    )

# Fields that would make this a cost record. The tier is structural; if one of
# these ever appears the record has quietly changed what it claims. Same list,
# for the same reason, that check_priceability_screen.py carries. Word
# universes and byte counts are structural quantities and are deliberately not
# on it: this record is *about* bytes.
COST_RECORD_KEYS = {
    "C_time",
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
    "shot_to_target",
}

# Byte counts taken from live Python objects, and the quantities derived from
# them. Dict capacity, integer boxing and object headers are CPython
# implementation details rather than portable constants, so these are compared
# under their own relative tolerance while every count, pair and packed byte
# below stays exact. The environment contract pins the interpreter's minor
# version but not its patch release, which is the gap this covers.
INTERPRETER_DEPENDENT_FIELDS = frozenset({
    "attributed_row_bytes",
    "attributed_fraction_of_delta",
    "attributed_fraction_of_peak",
    "bytes_per_coefficient_used",
    "distinct_boxed_objects",
    "majority_attribution_fraction_range",
    "lower_bound_bytes_per_entry",
    "lower_bound_ceiling_bytes",
    "maximum_bytes_per_coefficient",
    "measured_boxed_bytes",
    "measured_bytes_per_coefficient",
    "measured_container_bytes",
    "measured_operator_bytes",
    "minimum_bytes_per_coefficient",
    "packing_headroom",
    "packing_headroom_available",
    "packing_headroom_range",
    "per_bank_bytes_per_coefficient",
    "pooled_bytes_per_coefficient",
    "pooled_measured_bytes",
    "pooled_packing_headroom",
    "shared_boxed_slots",
})

# What a CPython patch release is allowed to move these by before the gate
# calls it a regression. Set at 2%: a change in the retained representation --
# the thing this record exists to detect -- moves them by tens of percent,
# while the object-layout drift a patch release could introduce is far below
# it. Declared here with its first use rather than fitted to an observed
# failure; no drift has been observed, and the alternative of an exact
# comparison is what makes a gate resolve on which runner it drew.
INTERPRETER_RTOL = 0.02


def _cost_key_paths(node, path: str = "$") -> list[str]:
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            if key in COST_RECORD_KEYS:
                found.append(f"{path}.{key}")
            found += _cost_key_paths(value, f"{path}.{key}")
        return found
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _cost_key_paths(v, f"{path}[{i}]")]
    return []


def _numbers(node, path: str = "$") -> dict[str, float]:
    """Every numeric leaf under an interpreter-dependent key, by path."""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key in INTERPRETER_DEPENDENT_FIELDS:
                out.update(_leaves(value, here))
            else:
                out.update(_numbers(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(_numbers(value, f"{path}[{index}]"))
    return out


def _leaves(node, path: str) -> dict[str, float]:
    if isinstance(node, bool) or node is None:
        return {}
    if isinstance(node, (int, float)):
        return {path: float(node)}
    if isinstance(node, dict):
        return {k: v for key, value in node.items()
                for k, v in _leaves(value, f"{path}.{key}").items()}
    if isinstance(node, list):
        return {k: v for i, value in enumerate(node)
                for k, v in _leaves(value, f"{path}[{i}]").items()}
    return {}


def interpreter_field_problems(expected: dict, actual: dict) -> list[str]:
    """Compare the measured byte fields under their own declared tolerance."""
    left = _numbers(expected)
    right = _numbers(actual)
    problems = []
    for path in sorted(set(left) - set(right)):
        problems.append(f"{path}: present in the committed record, absent on rebuild")
    for path in sorted(set(right) - set(left)):
        problems.append(f"{path}: produced by the rebuild, absent from the committed record")
    for path in sorted(set(left) & set(right)):
        if not math.isclose(left[path], right[path], rel_tol=INTERPRETER_RTOL,
                            abs_tol=0.0):
            problems.append(
                f"{path}: committed {left[path]:.17g}, rebuilt {right[path]:.17g}, "
                f"outside the declared {INTERPRETER_RTOL:.0%} interpreter "
                "tolerance. This field is a live object's size, so a difference "
                "this large is a change in the retained representation rather "
                "than a patch-release layout drift")
    return problems


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


@guarded_contract_problems
def measured_bank_problems(record: dict) -> list[str]:
    """Each priced bank's own arithmetic, and the policy's consequences."""
    problems: list[str] = []
    policy = record["storage_policy"]
    for row in record["measured_banks"]:
        where = f"{row['system']}/{row['kind']}"
        packed = row["packed_operator_bytes"]
        occurrences = row["coefficient_occurrences"]
        if packed != PACKED_BYTES_PER_COEFFICIENT * occurrences:
            problems.append(
                f"{where}: packed_operator_bytes {packed} is not "
                f"{PACKED_BYTES_PER_COEFFICIENT} * {occurrences}. The committed "
                "back-fill recovers T_coeff through exactly this identity, so a "
                "bank where it fails invalidates those rows as well as this one")
        retained = row["retained_block_pairs"]
        selection = row["selection_pairs"]
        built = row["pairs_built"]
        if retained + selection != built:
            problems.append(
                f"{where}: {retained} retained + {selection} selection pairs is "
                f"not the {built} built")
        basis = row["basis_size"]
        if row["complete_block_pairs"] != basis * (basis + 1) // 2:
            problems.append(
                f"{where}: complete_block_pairs {row['complete_block_pairs']} is "
                f"not M(M+1)/2 for M={basis}")
        if retained > row["complete_block_pairs"]:
            problems.append(
                f"{where}: {retained} retained pairs exceeds the "
                f"{row['complete_block_pairs']} the block can hold")
        if built and not _close(row["retained_pair_fraction"], retained / built):
            problems.append(
                f"{where}: retained_pair_fraction {row['retained_pair_fraction']} "
                f"is not {retained}/{built}")
        resident_universe = row["resident_word_universe"]
        if resident_universe and not _close(
                row["coefficient_reuse"], occurrences / resident_universe):
            problems.append(
                f"{where}: coefficient_reuse is not T_coeff/resident_W")
        selected_universe = row["word_universe"]
        if selected_universe and not _close(
                row["coefficient_occurrences_per_selected_element_word"],
                occurrences / selected_universe):
            problems.append(
                f"{where}: resident coefficients per selected word is not "
                "T_coeff/W_selected")
        if row["storage_policy"] != policy:
            problems.append(
                f"{where}: row policy {row['storage_policy']!r} disagrees with the "
                f"record's {policy!r}")
        for field in ("evicted_rows", "recomputed_rows", "spill_bytes"):
            if row[field] != 0:
                problems.append(
                    f"{where}: {field} is {row[field]} under {policy!r}, which has "
                    "no eviction path. Either the policy label is wrong or the bank "
                    "has grown one without this record being updated")
        if row["peak_resident_operator_rows"] != row["resident_operator_rows"]:
            problems.append(
                f"{where}: peak resident rows {row['peak_resident_operator_rows']} "
                f"differs from the current {row['resident_operator_rows']}. Under "
                f"{policy!r} the resident count never falls, so the two are equal "
                "by construction; a difference means rows are being freed and the "
                "policy label no longer describes the bank")
        # Both paths price the same resident rows; only selected-subspace W is
        # subset-scoped. A mismatch here means the byte quotient mixes populations.
        measured_occurrences = row["measured_coefficient_occurrences"]
        if measured_occurrences != occurrences:
            problems.append(
                f"{where}: the measured walk saw {measured_occurrences} "
                f"coefficients against {occurrences} in resources(); both must "
                "cover the same resident cache")
        if (row["measured_container_bytes"] + row["measured_boxed_bytes"]
                != row["measured_operator_bytes"]):
            problems.append(
                f"{where}: container plus boxed bytes is not the measured total")
        slots = row["distinct_boxed_objects"] + row["shared_boxed_slots"]
        if slots != 2 * measured_occurrences:
            problems.append(
                f"{where}: {slots} boxed slots accounted for against "
                f"{2 * measured_occurrences} expected (one key and one "
                "coefficient per occurrence); an allocation is double-counted "
                "or dropped")
        if measured_occurrences and not _close(
                row["measured_bytes_per_coefficient"],
                row["measured_operator_bytes"] / measured_occurrences):
            problems.append(f"{where}: measured bytes per coefficient is not the quotient")
        if measured_occurrences and not _close(
                row["packing_headroom"],
                row["measured_operator_bytes"]
                / (PACKED_BYTES_PER_COEFFICIENT * measured_occurrences)):
            problems.append(f"{where}: packing headroom is not the quotient")
    if not any(row["selection_pairs"] > 0 for row in record["measured_banks"]):
        problems.append(
            "no measured bank rejected a candidate, so every retained-pair "
            "fraction here is 1 by construction and the frontier arithmetic this "
            "record exists to baseline was never exercised")
    return problems


@guarded_contract_problems
def committed_row_problems(record: dict) -> list[str]:
    """The back-filled rows, re-read from the records they claim to quote."""
    problems: list[str] = []
    config = load_config()
    directory = ROOT / config["committed_records"]["directory"]
    declared = list(config["committed_records"]["records"])
    quoted = [row["record"] for row in record["committed_records"]]
    if quoted != declared:
        problems.append(
            f"the record prices {quoted}, but the config declares {declared}")
    for row in record["committed_records"]:
        name = row["record"]
        try:
            payload = json.loads((directory / name).read_text(encoding="utf-8"))
        except OSError as exc:
            problems.append(f"{name}: cannot be re-read ({exc})")
            continue
        packed = payload["adaptive_cached_operator_bytes"]
        occurrences, remainder = divmod(packed, PACKED_BYTES_PER_COEFFICIENT)
        if remainder:
            problems.append(
                f"{name}: adaptive_cached_operator_bytes {packed} is not a "
                f"multiple of {PACKED_BYTES_PER_COEFFICIENT}; T_coeff cannot be "
                "recovered from a record produced under a different packed model")
            continue
        for field, expected in (
                ("packed_operator_bytes", packed),
                ("coefficient_occurrences", occurrences),
                ("pairs_built", payload["adaptive_pairs_built"]),
                ("subspace_size_m", payload["subspace_size_m"]),
                ("word_universe", payload["element_word_universe"]),
                ("peak_rss_bytes", payload["adaptive_peak_rss_bytes"]),
                ("peak_rss_delta_bytes", payload["adaptive_peak_rss_delta_bytes"])):
            if row[field] != expected:
                problems.append(
                    f"{name}: {field} is {row[field]} here but {expected} in the "
                    "record it quotes; this row is transcribed, not read")
        basis = payload["subspace_size_m"]
        built = payload["adaptive_pairs_built"]
        retained = basis * (basis + 1) // 2
        if row["retained_block_pairs"] != retained:
            problems.append(
                f"{name}: retained_block_pairs {row['retained_block_pairs']} is "
                f"not M(M+1)/2 = {retained} for M={basis}")
        if row["selection_pairs"] != built - retained:
            problems.append(f"{name}: selection pairs do not close against built")
        if not _close(row["retained_pair_fraction"], retained / built):
            problems.append(f"{name}: retained_pair_fraction is not the quotient")
        if row["resident_word_universe"] is not None or row["coefficient_reuse"] is not None:
            problems.append(
                f"{name}: claims resident-W reuse although rejected-row W was not recorded")
        if not _close(row["coefficient_occurrences_per_selected_element_word"],
                      occurrences / payload["element_word_universe"]):
            problems.append(
                f"{name}: resident coefficients per selected word is not T_coeff/W_selected")
        if not row.get("reuse_recovery_boundary"):
            problems.append(f"{name}: does not state why true reuse cannot be recovered")
    return problems


@guarded_contract_problems
def rate_problems(record: dict) -> list[str]:
    """The pooled rate, re-derived from the rows it pools."""
    problems: list[str] = []
    rate = record["measured_rate"]
    rows = record["measured_banks"]
    measured = sum(row["measured_operator_bytes"] for row in rows)
    occurrences = sum(row["measured_coefficient_occurrences"] for row in rows)
    if rate["pooled_measured_bytes"] != measured:
        problems.append("pooled measured bytes is not the sum over the priced banks")
    if rate["pooled_coefficient_occurrences"] != occurrences:
        problems.append("pooled occurrences is not the sum over the priced banks")
    if not _close(rate["pooled_bytes_per_coefficient"], measured / occurrences):
        problems.append(
            "the pooled rate is not the quotient of the pooled totals; a mean "
            "over banks would weight the smallest as heavily as the largest")
    per_bank = rate["per_bank_bytes_per_coefficient"]
    if len(per_bank) != len(rows):
        problems.append("the per-bank rate table does not cover every priced bank")
    for row in rows:
        key = f"{row['system']}/{row['kind']}"
        if key not in per_bank:
            problems.append(f"{key}: absent from the per-bank rate table")
        elif not _close(per_bank[key], row["measured_bytes_per_coefficient"]):
            problems.append(f"{key}: per-bank rate disagrees with its own row")
    if per_bank:
        if not _close(rate["minimum_bytes_per_coefficient"], min(per_bank.values())):
            problems.append("the recorded minimum rate is not the minimum")
        if not _close(rate["maximum_bytes_per_coefficient"], max(per_bank.values())):
            problems.append("the recorded maximum rate is not the maximum")
    return problems


@guarded_contract_problems
def attribution_problems(record: dict) -> list[str]:
    """The attribution and its dominance verdict, re-derived from the rate."""
    problems: list[str] = []
    attribution = record["peak_rss_attribution"]
    rate = record["measured_rate"]
    used = attribution["bytes_per_coefficient_used"]
    if not _close(used, rate["minimum_bytes_per_coefficient"]):
        problems.append(
            f"the attribution uses {used} bytes per coefficient, not the minimum "
            f"{rate['minimum_bytes_per_coefficient']} it declares as its "
            "low-rate calibration point")
    if attribution["attributed_majority_fraction"] != BANK_DOMINANCE_FRACTION:
        problems.append(
            "the attributed-majority threshold is "
            f"{attribution['attributed_majority_fraction']}, not "
            f"the declared {BANK_DOMINANCE_FRACTION}")
    committed = {row["record"]: row for row in record["committed_records"]}
    dominated = []
    for row in attribution["rows"]:
        name = row["record"]
        source = committed.get(name)
        if source is None:
            problems.append(f"{name}: attributed but not among the committed rows")
            continue
        if row["coefficient_occurrences"] != source["coefficient_occurrences"]:
            problems.append(f"{name}: attribution and ledger disagree on T_coeff")
            continue
        attributed = used * row["coefficient_occurrences"]
        if not _close(row["attributed_row_bytes"], attributed):
            problems.append(f"{name}: attributed bytes is not rate * T_coeff")
        peak = source["peak_rss_bytes"]
        if not _close(row["attributed_fraction_of_peak"], attributed / peak):
            problems.append(f"{name}: the peak fraction is not the quotient")
        delta = source["peak_rss_delta_bytes"]
        expected_delta = (attributed / delta) if delta else None
        if expected_delta is None:
            if row["attributed_fraction_of_delta"] is not None:
                problems.append(
                    f"{name}: records a delta fraction against a zero delta")
        elif not _close(row["attributed_fraction_of_delta"], expected_delta):
            problems.append(f"{name}: the delta fraction is not the quotient")
        dominates = (attributed / peak) >= BANK_DOMINANCE_FRACTION
        if row["attributed_majority_of_peak"] != dominates:
            problems.append(
                f"{name}: records attributed_majority_of_peak="
                f"{row['attributed_majority_of_peak']}, but its calibrated fraction gives "
                f"{dominates}. The verdict follows from the threshold and is not "
                "writable by hand")
        if dominates:
            dominated.append(name)
    if attribution["records_with_attributed_majority_of_peak"] != dominated:
        problems.append(
            "the majority-attribution record list disagrees with the row verdicts")
    if not attribution.get("what_this_does_not_establish"):
        problems.append(
            "the attribution records no boundary; a fraction of peak RSS is the "
            "quantity here most likely to be re-read as a promised reduction")
    return problems


@guarded_contract_problems
def baseline_problems(record: dict) -> list[str]:
    """The baseline may carry headroom and no verdict at all."""
    problems: list[str] = []
    baseline = record["baseline"]
    if baseline["verdict"] != "baseline_only_no_go_no_go_evaluated":
        problems.append(
            f"the baseline records verdict {baseline['verdict']!r}. 2M's go/no-go "
            "asks for a measured reduction from an implementation that does not "
            "exist yet, so this record establishes what that gate is read against "
            "and may not grade it")
    if not baseline.get("why_no_verdict"):
        problems.append("the baseline withholds a verdict without saying why")
    if not _close(baseline["packing_headroom_available"],
                  record["measured_rate"]["pooled_packing_headroom"]):
        problems.append("the baseline's headroom is not the pooled measurement")
    fractions = [row["retained_pair_fraction"] for row in record["committed_records"]]
    low, high = baseline["committed_retained_pair_fraction_range"]
    if not (_close(low, min(fractions)) and _close(high, max(fractions))):
        problems.append(
            "the retained-pair range is not the range over the committed rows")
    exercised = sorted({row["system"] for row in record["measured_banks"]
                        if row["selection_pairs"] > 0})
    if sorted(baseline["frontier_exercised_on"]) != exercised:
        problems.append(
            "the frontier-exercised list disagrees with the priced banks")
    if not baseline.get("eviction_headroom_statement"):
        problems.append(
            "the baseline reports a retained-pair range with no statement that it "
            "is eviction headroom rather than an achieved reduction")
    return problems


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "structural":
        problems.append(
            f"evidence tier is {record.get('evidence_tier')!r}; this record samples "
            "nothing and must not claim a stronger tier")
    if record.get("packed_bytes_per_coefficient") != PACKED_BYTES_PER_COEFFICIENT:
        problems.append(
            f"the record declares {record.get('packed_bytes_per_coefficient')!r} "
            f"packed bytes per coefficient against this checker's "
            f"{PACKED_BYTES_PER_COEFFICIENT}")
    if not record.get("claim_boundary"):
        problems.append("no claim boundary recorded")
    if not record.get("interpreter_dependence"):
        problems.append(
            "the record publishes measured byte counts without saying they are a "
            "property of the interpreter that measured them")
    for path in _cost_key_paths(record):
        problems.append(
            f"cost-record field at {path}: this tier prices nothing, so its "
            "presence means the record has changed what it claims")
    problems += measured_bank_problems(record)
    problems += committed_row_problems(record)
    problems += rate_problems(record)
    problems += attribution_problems(record)
    problems += baseline_problems(record)
    return problems


def main() -> int:
    load_config()  # the config's own declaration gates
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(
        expected, actual, atol=1e-12, rtol=1e-12,
        ignored_keys=frozenset({"provenance"}) | INTERPRETER_DEPENDENT_FIELDS)
    problems += interpreter_field_problems(expected, actual)
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {p}" for p in contract_problems(actual)]
    for problem in problems:
        print(f"  {problem}")
    print("bank storage ledger:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
