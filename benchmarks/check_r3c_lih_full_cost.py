#!/usr/bin/env python3
"""Regenerate and validate the R3c LiH margin-stop full-cost record.

Two jobs, as this repository's other checkers have: rebuild the record from
scratch and compare it field by field, and re-derive every contract it claims
from its own contents rather than trusting them.

Most of the pricing contract is not restated here. R3c's cells come out of the
same ``run_protocol_cost`` machinery that priced BeH2, so
``check_protocol_cost``'s per-cell rules are imported and applied: the smallest
confirmed pass with a confirmed failure below it, monotone confirmation, stable
retained rank at both deciding endpoints, margins that agree with the marginal
flag, an interval widened exactly on the sides R1 flagged and nowhere else, and
a ``k*`` region that accounts for every rung once. A second copy of those rules
here would be a second thing to keep in step with the sampler.

What is specific to this phase:

* the record inherits the preregistration's claim boundary. This is the one
  checker in the tree that *requires* inheritance rather than refusing it --
  R3b's config bounded a commit that had sampled nothing, so its record had to
  state its own, and R3c's was written tenselessly so that it would not have to;
* the config it names is the file that landed, by the digest the migration
  manifest froze, and the whole result-free checker still passes on it;
* the pricing table and the QR3 re-derivation follow from the drawn cells,
  re-derived through the producer's own functions rather than read back;
* the run was drawn by the frozen instrument and under the frozen stack, both
  read back out of the record rather than out of the config that asked for them;
* R3b's rejection and ``protocol_cost.json``'s one-instance abstention are both
  untouched. This record adds a second price; it does not reach back and edit
  the record that said there was one.

    python benchmarks/check_r3c_lih_full_cost.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
    guarded_contract_problems,
    sampling_stream_mismatch,
)

try:  # package import in tests versus direct script execution
    from benchmarks.check_protocol_cost import (
        _bracket_problems,
        _crossing_problems,
        _k_star_problems,
        _verdict_problems,
    )
    from benchmarks.check_r3c_preregistration import (
        TEMPORAL_BOUNDARY_PHRASES,
        load_config,
    )
    from benchmarks.run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from benchmarks.run_mapping_axis import _canonical_sha256, load_device_cards
    from benchmarks.run_r3c_lih_full_cost import (
        CONFIG,
        FIRST_PRICED_INSTANCE,
        PROTOCOL_COST_RECORD,
        REFERENCE,
        SCHEMA,
        _file_sha256,
        _frozen_cards,
        build_record,
        declared_environment,
        instrument_problems,
        pricing_decision,
        preregistration_problems,
        qr3_second_instance,
        r3b_untouched_problems,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_protocol_cost import (
        _bracket_problems,
        _crossing_problems,
        _k_star_problems,
        _verdict_problems,
    )
    from check_r3c_preregistration import (
        TEMPORAL_BOUNDARY_PHRASES,
        load_config,
    )
    from run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from run_mapping_axis import _canonical_sha256, load_device_cards
    from run_r3c_lih_full_cost import (
        CONFIG,
        FIRST_PRICED_INSTANCE,
        PROTOCOL_COST_RECORD,
        REFERENCE,
        SCHEMA,
        _file_sha256,
        _frozen_cards,
        build_record,
        declared_environment,
        instrument_problems,
        pricing_decision,
        preregistration_problems,
        qr3_second_instance,
        r3b_untouched_problems,
    )

#: Fields that belong to a scope-decision probe and would misdescribe this one.
#:
#: R3b and QR3b were probes: they could accept or reject a bank and were
#: forbidden to price it, so they carried ``full_run_authorized`` and
#: ``screen_prediction`` and sanitized every cost derivative out. R3c is the
#: run those probes were deciding about. Carrying one of their fields here
#: would either re-decide a question this record does not evaluate or offer a
#: second authorization the preregistration already gave once.
PROBE_ONLY_KEYS = {
    "scoping_probe",
    "screen_prediction",
    "full_run_authorized",
    "eligible_for_full_run",
    "resolution_gate_passes",
}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _preregistration_problems(record: dict, config: dict) -> list[str]:
    """The record's own account of what authorized it, checked against the file."""
    problems: list[str] = []
    prereg = record.get("preregistration", {})
    if prereg.get("config") != "benchmarks/configs/r3c_lih_full_cost.json":
        problems.append("the record names the wrong preregistration config")
    if prereg.get("landed_before_any_sampling") is not True:
        problems.append("the record does not claim a preregistration")
    if prereg.get("checker") != "benchmarks/check_r3c_preregistration.py":
        problems.append("the record names the wrong result-free checker")
    if prereg.get("authorized_by_this_config") is not True:
        problems.append(
            "the record does not carry the authorization its config gives; R3c's "
            "run is licensed by that field and by nothing else"
        )
    if record.get("config_sha256") != _canonical_sha256(config):
        problems.append("the record does not bind the canonical config payload")
    live = _file_sha256(CONFIG)
    if record.get("config_file_sha256") != live:
        problems.append(
            "the record's config file digest is not the committed file's; either "
            "the preregistration moved after the run or the record is not of it"
        )
    problems += [f"preregistration: {problem}" for problem in preregistration_problems(config)]
    return problems


def _boundary_problems(record: dict, config: dict) -> list[str]:
    """Inheritance is required here, and refused by the R3b checker next door.

    A claim boundary bounds the artifact it sits in. R3b's config opened "No
    sampling has been performed under this config" -- true of the commit that
    landed it, false of the record that drew forty cells -- so that record had
    to state its own and quote the config's beside it. R3c's config was written
    tenselessly for exactly this reason: it says what the *config* carries and
    what a record may report, and both remain true once the run exists. So the
    record inherits it, and a record that quietly substituted its own wording
    would have escaped the sentence the preregistration bound it to.
    """
    problems: list[str] = []
    boundary = record.get("claim_boundary", "")
    if boundary != config["claim_boundary"]:
        problems.append(
            "the record's claim boundary is not the preregistration's; R3c's is "
            "tenseless and is meant to be inherited verbatim"
        )
    lowered = boundary.lower()
    for phrase in TEMPORAL_BOUNDARY_PHRASES:
        if phrase in lowered:
            problems.append(
                f"an executed run carries a temporal claim boundary: {phrase!r}"
            )
    prereg = record.get("preregistration", {})
    if prereg.get("claim_boundary_inherited_from_config") is not True:
        problems.append("the record does not declare that it inherited the boundary")
    if prereg.get("claim_boundary_is_tenseless") is not True:
        problems.append("the record does not declare the boundary tenseless")
    return problems


def _instrument_and_environment_problems(record: dict, config: dict) -> list[str]:
    """What actually drew the cells, read back out of the record.

    The producer checks the running stack before it samples; this checks the
    stack the committed record says it was drawn under, which is the only one a
    later reader can verify. Both matter and neither implies the other.
    """
    problems: list[str] = []
    run = record.get("full_cost_run", {})
    protocol = config["protocol"]
    if run.get("executed") is not True:
        problems.append("the committed record did not execute its authorized run")
    if run.get("is_a_cost_record") is not True:
        problems.append("the run does not identify itself as a cost record")
    for field in ("exploratory_replicas", "confirmatory_replicas"):
        if run.get(field) != protocol[field]:
            problems.append(f"full_cost_run.{field} is not the preregistered count")
    if run.get("search_endpoints_effective_shots_per_setting") != list(SEARCH_ENDPOINTS):
        problems.append("the run reports an endpoint grid other than the frozen one")
    if run.get("seed_roots") != protocol["seed_roots"]:
        problems.append("the run reports seed roots other than the preregistered ones")
    if run.get("bootstrap_replicates") != protocol["bootstrap_replicates"]:
        problems.append("the run reports another bootstrap replicate count")

    system = run.get("sampling_evidence")
    if not isinstance(system, dict):
        problems.append("the run carries no sampling evidence")
        return problems
    problems += instrument_problems(system, config)

    declared = declared_environment(config)
    provenance = record.get("provenance", {})
    dependencies = provenance.get("dependencies", {})
    stamped = {
        "python": ".".join(str(provenance.get("python", "")).split(".")[:2]),
        "numpy": dependencies.get("numpy"),
        "scipy": dependencies.get("scipy"),
        "stim": dependencies.get("stim"),
    }
    if stamped != declared:
        problems.append(
            f"the record was stamped under {stamped} but R3c froze {declared}; "
            "these seed roots name a different stream there"
        )
    if record.get("execution_environment", {}).get("matches_preregistration") is not True:
        problems.append("the record does not claim the frozen execution environment")
    return problems


def _cell_problems(record: dict) -> list[str]:
    """R1's pricing rules on every drawn cell, and PLAN 6.7's k* on every arm.

    Imported from ``check_protocol_cost`` rather than restated: the cells come
    out of the same sampler, so one implementation of the rule is what keeps
    the two records gated the same way.
    """
    problems: list[str] = []
    system = record.get("full_cost_run", {}).get("sampling_evidence", {})
    card_hashes = {
        card.get("name"): card.get("sha256") for card in record.get("device_cards", [])
    }
    key = system.get("system")
    if system.get("status") != "searched":
        problems.append(
            f"{key}: a run that cleared the bias floor before sampling is not "
            "recorded as searched"
        )
        return problems
    if record.get("accuracy_target_millihartree") != ACCURACY_TARGET_MILLIHARTREE:
        problems.append(
            "the accuracy target drifted from the one R1-R3 is matched at, so "
            "these costs are not comparable to the record they extend"
        )
    if system.get("max_exact_subspace_bias_millihartree", float("inf")) >= (
        ACCURACY_TARGET_MILLIHARTREE
    ):
        problems.append(f"{key}: a bank above the bias floor was priced")
    if system.get("r1_cross_check", {}).get("status") != "not_requested_for_extension":
        problems.append(
            f"{key}: an R1 cross-check was reported, but R1 priced BeH2 and this "
            "is a different instance with no column of R1's to agree with"
        )

    for arm in system.get("arms", []):
        mapping = arm["mapping"]
        sizes = [rung["block_size"] for rung in arm["rungs"]]
        if sizes != sorted(set(sizes)):
            problems.append(f"{key}/{mapping}: rungs are unordered or duplicated")
        if any(size > arm["measured_qubits"] for size in sizes):
            problems.append(f"{key}/{mapping}: a rung exceeds the measured register")
        for rung in arm["rungs"]:
            if set(rung.get("estimators", {})) != set(ESTIMATORS):
                problems.append(
                    f"{key}/{mapping} k={rung['block_size']}: estimator pair is "
                    "incomplete"
                )
                continue
            for estimator, cell in rung["estimators"].items():
                prefix = f"{key}/{mapping} k={rung['block_size']} {estimator}"
                problems += _crossing_problems(prefix, cell)
                problems += _bracket_problems(prefix, cell, card_hashes)

    by_mapping = {arm["mapping"]: arm for arm in system.get("arms", [])}
    summary = system.get("k_star", {})
    if set(summary) != set(card_hashes):
        problems.append(f"{key}: k* is not reported on every declared card")
    for card, payload in summary.items():
        for estimator in ESTIMATORS:
            if estimator not in payload:
                problems.append(f"{key}/{card}: {estimator} k* missing")
                continue
            for mapping, entry in payload[estimator]["by_arm"].items():
                prefix = f"{key}/{card}/{estimator}/{mapping}"
                problems += _k_star_problems(
                    prefix, by_mapping[mapping], entry, card, estimator
                )
                problems += _verdict_problems(prefix, payload[estimator], mapping)
    return problems


def _readout_problems(record: dict, config: dict, cards) -> list[str]:
    """The pricing table and the QR3 verdict must follow from the drawn cells."""
    problems: list[str] = []
    system = record.get("full_cost_run", {}).get("sampling_evidence", {})
    pricing = record.get("pricing", {})
    expected = pricing_decision(system, config)
    if expected != pricing:
        problems.append("the pricing decision does not re-derive from the drawn cells")
    if pricing.get("status") not in {
        "priced_second_instance",
        "right_censored_at_frozen_grid",
    }:
        problems.append(
            "the pricing status is not one of the preregistered readouts"
        )
    if pricing.get("wider_grid_licensed_by_this_record") is not False:
        problems.append("the record licenses a wider endpoint grid")
    if pricing.get("grid_ceiling_effective_shots_per_setting") != SEARCH_ENDPOINTS[-1]:
        problems.append("the censoring ceiling is not the frozen last endpoint")
    for cell in pricing.get("cells", []):
        for field in (
            "confirmed_failing_effective_shots_per_setting",
            "confirmed_passing_effective_shots_per_setting",
        ):
            value = cell.get(field)
            if value is not None and value not in SEARCH_ENDPOINTS:
                problems.append(
                    f"{cell['mapping']} k={cell['block_size']} {cell['estimator']}: "
                    f"{field} {value} is outside the frozen grid"
                )

    qr3 = record.get("qr3_second_instance", {})
    if qr3 != qr3_second_instance(system, cards, config, expected):
        problems.append("the QR3 statement does not re-derive from the drawn cells")
    gate_open = pricing.get("status") == "priced_second_instance"
    if gate_open != (qr3.get("status") == "re_derived"):
        problems.append(
            "QR3 was re-derived without a second priced instance, or withheld "
            "with one; the preregistration ties the two together"
        )
    if qr3.get("first_instance", {}).get("record_sha256") != _file_sha256(
        PROTOCOL_COST_RECORD
    ):
        problems.append(
            "the QR3 comparison does not bind the committed BeH2 cost record"
        )
    return problems


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    config = load_config()
    cards = load_device_cards()
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_role") != "accuracy_matched_cost_record":
        problems.append("the run is not labelled an accuracy-matched cost record")
    if record.get("evidence_tier") != "exact":
        problems.append("the oracle comparator is not labelled exact")
    if record.get("search_uncertainty_evidence") != "heuristic":
        problems.append("Monte Carlo search uncertainty is not labelled heuristic")

    leaked = sorted(set(_walk_keys(record)) & PROBE_ONLY_KEYS)
    if leaked:
        problems.append(f"the cost record carries scope-probe fields: {leaked}")

    if {card.get("name"): card.get("sha256") for card in record.get("device_cards", [])} != (
        _frozen_cards(config)
    ):
        problems.append("the record's device cards are not the three R3c pinned")

    for field in ("protocol", "acceptance_rule", "structural_gates",
                  "reporting_contract", "estimand", "parent_lineage"):
        if record.get(field) != config[field]:
            problems.append(f"{field} drifted from the preregistration")

    selection = record.get("selection", {})
    candidate = config["candidate"]
    if list(selection.get("labels", [])) != list(candidate["selected_labels"]):
        problems.append("selected labels drifted from the preregistration")
    if selection.get("basis_size") != candidate["expected_basis_size"]:
        problems.append("selected basis size drifted from the preregistration")
    if selection.get("source_word_universe") != candidate["expected_binding_word_universe"]:
        problems.append("the bank's word universe drifted from the preregistration")
    if selection.get("stopping_reason") != "smallest prefix clearing the accuracy margin":
        problems.append("selection did not stop on the declared margin rule")
    if selection.get("basis_size", 0) >= selection.get("intrinsic_stop_basis_size", 0):
        problems.append(
            "the priced bank is not shorter than the intrinsic-stop bank QR3b "
            "probed, so this record is not pricing what it claims to"
        )
    if abs(
        selection.get("bias_millihartree", float("inf"))
        - float(candidate["expected_bias_millihartree"])
    ) > float(candidate["bias_tolerance_millihartree"]):
        problems.append("the re-derived bias drifted from the preregistration")

    problems += _preregistration_problems(record, config)
    problems += _boundary_problems(record, config)
    problems += _instrument_and_environment_problems(record, config)
    problems += _cell_problems(record)
    problems += _readout_problems(record, config, cards)
    return problems


def frozen_records_untouched_problems() -> list[str]:
    """The two records this one extends, neither of which it may edit.

    R3b rejected the bank for a full run under its own ``2+2`` instrument, and
    ``protocol_cost.json`` abstained on QR3 with one priced instance. Both stay
    true of the records that carry them: R3c's authorization came from its own
    config, and its second price lives in its own record. A run that had reached
    back to relabel either would have made the earlier finding unfalsifiable
    after the fact.
    """
    problems = r3b_untouched_problems()
    frozen = json.loads(PROTOCOL_COST_RECORD.read_text(encoding="utf-8"))
    qr3 = frozen.get("qr3_accuracy_matched", {})
    if qr3.get("status") != "abstains":
        problems.append(
            "the frozen cost record no longer abstains on QR3; R3c reports its "
            "own comparison and does not rewrite that one"
        )
    if qr3.get("priced_instances") != [FIRST_PRICED_INSTANCE]:
        problems.append(
            "the frozen cost record's priced-instance list changed; R3c adds a "
            "second price in its own record, not in that one"
        )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # Every value here descends from sampled shots fed through an
    # ill-conditioned projected eigensolve, so a differing environment answers a
    # different question rather than verifying this one -- and a value diff
    # would invite a widened tolerance as the repair.
    stream = sampling_stream_mismatch(expected)
    if stream:
        print("R3c LiH full cost: FAIL (build environment differs)")
        for problem in stream:
            print(f"  {problem}")
        print("  skipped the rebuild: under a different NumPy the sampled "
              "projected eigensolves do not verify this record")
        for problem in contract_problems(expected):
            print(f"  committed record: {problem}")
        return 1

    actual = build_record(workers=args.workers)
    problems = compare_json_records(
        expected, actual, atol=CROSS_MACHINE_ATOL, rtol=CROSS_MACHINE_RTOL
    )
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {item}" for item in contract_problems(actual)]
    problems += frozen_records_untouched_problems()
    for problem in problems[:30]:
        print(f"  {problem}")
    if len(problems) > 30:
        print(f"  ... and {len(problems) - 30} more")
    print("R3c LiH full cost:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
