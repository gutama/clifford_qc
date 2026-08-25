#!/usr/bin/env python3
"""Recompute and gate the R3 accuracy-matched ``C(epsilon)`` and ``k*`` record.

Two independent jobs, as the project's other checkers do: the record is rebuilt
from scratch and compared field by field, and every contract it claims is
re-derived from its own contents rather than trusted.

The contracts, in the order a reader should care about them:

* the setting count of every ``(arm, k)`` cell reproduces the structural R3
  grid -- the condition that makes this the cost layer *of that grid* rather
  than a second, differently-partitioned experiment;
* every system's priced/unpriced status agrees with its own bias floor -- the
  budget-8 H4 bank exceeds the target on every arm and must stay unpriced, the
  converged one clears it and must not;
* each crossing satisfies R1's own pricing rules -- smallest confirmed pass,
  a confirmed failure below it, monotone confirmation, one retained rank at
  both deciding endpoints, and margins that agree with the marginal flag;
* each cost interval is the bracket its crossing licenses, widened exactly on
  the sides R1 flagged and nowhere else;
* each ``k*`` region is re-derived from the intervals, contains its own point
  argmin, and accounts for every rung exactly once;
* the QR1 and QR4 verdicts are re-derived from the costs they summarize;
* the ``jw`` arm's crossings are compared against R1's frozen BeH2 column.

    python benchmarks/check_protocol_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

from clifford_qc.reproducibility import (
    CROSS_MACHINE_ATOL,
    CROSS_MACHINE_RTOL,
    compare_json_records,
    guarded_contract_problems,
    sampling_stream_mismatch,
)

try:  # package import in tests versus direct script execution
    from benchmarks.check_exact_shot_search import _rank_stability_problems
    from benchmarks.run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from benchmarks.run_exact_shot_search import (
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        MARGINAL_TARGET_FRACTION,
        SEARCH_ENDPOINTS,
    )
    from benchmarks.run_protocol_cost import (
        CONFIG,
        REFERENCE,
        SCHEMA,
        _instance_cost_spread,
        build_record,
        load_config,
        r1_crossings,
        structural_settings,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_exact_shot_search import _rank_stability_problems
    from run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from run_exact_shot_search import (
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        MARGINAL_TARGET_FRACTION,
        SEARCH_ENDPOINTS,
    )
    from run_protocol_cost import (
        CONFIG,
        REFERENCE,
        SCHEMA,
        _instance_cost_spread,
        build_record,
        load_config,
        r1_crossings,
        structural_settings,
    )

# C_time is a sum of per-setting durations times an integer shot vector, so two
# routes to the same schedule agree to rounding, not to the last bit.
COST_RELATIVE_TOLERANCE = 1e-9

# Every millihartree field in this record is an energy *difference* against the
# exact sector reference, so cancellation would normally argue for a loosened
# per-field tolerance -- and that argument is still declined. The producer takes
# its reference from the dense eigensolve rather than ARPACK precisely so the
# difference reproduces bit for bit, and it does: the same fields drift by a
# constant ~1e-10 mHa under the default 'auto' path and by nothing at all under
# 'dense'. No per-key widening is granted here for that reason.
#
# The record-versus-rebuild comparison below uses CROSS_MACHINE_*, which is a
# different quantity and not a retreat from the above. Within one machine these
# fields drift by exactly zero; across machines OpenBLAS picks a different
# kernel and they drift up to 1e-11, which a 1e-12 gate cannot survive. See the
# constant's own comment for the measurement.


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= COST_RELATIVE_TOLERANCE * max(abs(left), abs(right), 1.0)


def _crossing_problems(prefix: str, arm: dict) -> list[str]:
    """R1's pricing rules, re-derived on one cell rather than inherited."""
    problems: list[str] = []
    search = arm.get("shot_to_target", {})
    passing = search.get("confirmed_passing_effective_shots_per_setting")
    failing = search.get("confirmed_failing_effective_shots_per_setting")
    confirmation = {
        item.get("effective_shots_per_setting"): item
        for item in arm.get("confirmation", [])
    }
    exploration = [
        item.get("effective_shots_per_setting") for item in arm.get("exploration", [])
    ]
    if exploration != list(SEARCH_ENDPOINTS):
        problems.append(f"{prefix}: exploration grid drift")
    if passing is None:
        if arm.get("device_costs"):
            problems.append(f"{prefix}: unconfirmed target was priced")
        if arm.get("cost_bracket", {}).get("priced"):
            problems.append(f"{prefix}: unconfirmed target carries a cost interval")
        return problems
    if not confirmation.get(passing, {}).get("passes_target"):
        problems.append(f"{prefix}: priced endpoint did not pass")
    smallest = sorted(
        endpoint for endpoint, summary in confirmation.items()
        if summary.get("passes_target")
    )
    if smallest and passing != smallest[0]:
        problems.append(f"{prefix}: priced endpoint is not the smallest confirmed pass")
    if any(
        endpoint > passing and not summary.get("passes_target")
        for endpoint, summary in confirmation.items()
    ):
        problems.append(f"{prefix}: priced crossing is nonmonotone")
    if failing is not None and confirmation.get(failing, {}).get("passes_target") is not False:
        problems.append(f"{prefix}: reported failing endpoint did not fail")
    problems.extend(_rank_stability_problems(prefix, confirmation, passing, failing))

    marginal = search.get("environment_marginal_endpoints")
    if marginal is None:
        problems.append(f"{prefix}: crossing margin was not recorded")
    elif not isinstance(marginal, list) or not all(
        item in ("passing", "failing") for item in marginal
    ):
        problems.append(
            f"{prefix}: marginal endpoints are not a list of 'passing'/'failing' labels"
        )
    elif bool(marginal) != bool(search.get("crossing_is_environment_marginal")):
        problems.append(f"{prefix}: marginal endpoints disagree with the flag")
    else:
        for label in ("passing", "failing"):
            fraction = search.get(f"{label}_target_margin_fraction")
            if fraction is None:
                continue
            if (abs(fraction) <= MARGINAL_TARGET_FRACTION) != (label in marginal):
                problems.append(
                    f"{prefix}: {label} margin {fraction:+.3f} disagrees with "
                    "its marginal listing"
                )
    return problems


def _bracket_problems(prefix: str, arm: dict, card_hashes: dict) -> list[str]:
    """The interval must be the one its crossing licenses -- no wider, no narrower."""
    problems: list[str] = []
    search = arm.get("shot_to_target", {})
    bracket = arm.get("cost_bracket", {})
    passing = search.get("confirmed_passing_effective_shots_per_setting")
    failing = search.get("confirmed_failing_effective_shots_per_setting")
    if passing is None:
        return problems
    if not bracket.get("priced"):
        problems.append(f"{prefix}: a confirmed crossing carries no cost interval")
        return problems
    grid = list(SEARCH_ENDPOINTS)
    marginal = search.get("environment_marginal_endpoints") or []

    if bracket.get("point_effective_shots_per_setting") != passing:
        problems.append(f"{prefix}: interval point is not the reported count")
    expected_widened = sorted(
        side for side in ("passing", "failing")
        if side in marginal and not (side == "failing" and failing is None)
    )
    if sorted(bracket.get("widened_sides", [])) != expected_widened:
        problems.append(
            f"{prefix}: widened sides {bracket.get('widened_sides')} do not match "
            f"the marginal flags {expected_widened}"
        )
    upper = bracket.get("upper_effective_shots_per_setting")
    if "passing" in marginal:
        index = grid.index(passing)
        expected_upper = grid[index + 1] if index + 1 < len(grid) else None
    else:
        expected_upper = passing
    if upper != expected_upper:
        problems.append(
            f"{prefix}: upper endpoint {upper} should be {expected_upper}"
        )
    lower = bracket.get("lower_effective_shots_per_setting")
    if failing is None:
        expected_lower = None
    elif "failing" in marginal:
        index = grid.index(failing)
        expected_lower = grid[index - 1] if index > 0 else None
    else:
        expected_lower = failing
    if lower != expected_lower:
        problems.append(
            f"{prefix}: lower endpoint {lower} should be {expected_lower}"
        )
    if bool(bracket.get("unbounded_below")) != (expected_lower is None):
        problems.append(f"{prefix}: unbounded-below flag disagrees with the interval")
    if bool(bracket.get("unbounded_above")) != (expected_upper is None):
        problems.append(f"{prefix}: unbounded-above flag disagrees with the interval")

    cards = bracket.get("cards", {})
    if set(cards) != set(card_hashes):
        problems.append(f"{prefix}: cost interval is missing a declared device card")
    for name, entry in cards.items():
        if entry.get("device_card_sha256") != card_hashes.get(name):
            problems.append(f"{prefix}: {name} device-card hash drift")
        point = entry.get("C_time_epsilon_us")
        if not entry.get("admissible"):
            if point is not None:
                problems.append(f"{prefix}: {name} is inadmissible but carries a runtime")
            continue
        low = entry.get("C_time_lower_us")
        high = entry.get("C_time_upper_us")
        if point is None:
            problems.append(f"{prefix}: {name} is admissible with no runtime")
            continue
        if low is None or low > point + COST_RELATIVE_TOLERANCE * max(point, 1.0):
            problems.append(
                f"{prefix}: {name} lower bound {low} exceeds its point cost {point}"
            )
        if high is not None and high < point - COST_RELATIVE_TOLERANCE * max(point, 1.0):
            problems.append(
                f"{prefix}: {name} upper bound {high} is below its point cost {point}"
            )
        prices = arm.get("device_costs", {}).get(name, {})
        priced = prices.get("accuracy", {}).get("C_time_epsilon_us")
        if not _close(priced, point):
            problems.append(
                f"{prefix}: {name} interval point {point} disagrees with the "
                f"priced schedule {priced}"
            )
        if prices.get("accuracy", {}).get("evidence_tier") != "exact":
            problems.append(f"{prefix}: {name} priced schedule is not exact-tier")
    return problems


def _k_star_problems(prefix: str, arm: dict, entry: dict, card: str,
                     estimator: str) -> list[str]:
    """Re-derive the region, its membership, and its bookkeeping."""
    problems: list[str] = []
    priced, unresolved, inadmissible = [], [], []
    for rung in arm["rungs"]:
        bracket = rung["estimators"][estimator]["cost_bracket"]
        cell = bracket.get("cards", {}).get(card)
        if not bracket.get("priced") or cell is None:
            unresolved.append(rung["block_size"])
        elif not cell["admissible"]:
            inadmissible.append(rung["block_size"])
        else:
            priced.append((rung["block_size"], cell))

    for label, expected in (
        ("unresolved_block_sizes", sorted(unresolved)),
        ("inadmissible_block_sizes", sorted(inadmissible)),
        ("priced_block_sizes", sorted(size for size, _ in priced)),
    ):
        if entry.get(label) != expected:
            problems.append(f"{prefix}: {label} is {entry.get(label)}, expected {expected}")
    total = (
        len(entry.get("unresolved_block_sizes", []))
        + len(entry.get("inadmissible_block_sizes", []))
        + len(entry.get("priced_block_sizes", []))
    )
    if total != len(arm["rungs"]):
        problems.append(
            f"{prefix}: {total} rungs accounted for out of {len(arm['rungs'])}"
        )
    if not priced:
        if entry.get("k_star_region") or entry.get("k_star_point") is not None:
            problems.append(f"{prefix}: a k* was reported with no priced rung")
        return problems

    best = min(priced, key=lambda item: (item[1]["C_time_epsilon_us"], item[0]))
    if entry.get("k_star_point") != best[0]:
        problems.append(
            f"{prefix}: k* point {entry.get('k_star_point')} is not the cheapest "
            f"reported count ({best[0]})"
        )
    uppers = [
        cell["C_time_upper_us"] for _, cell in priced
        if cell["C_time_upper_us"] is not None
    ]
    if uppers:
        ceiling = min(uppers)
        expected_region = sorted(
            size for size, cell in priced if cell["C_time_lower_us"] <= ceiling
        )
        if not _close(entry.get("region_ceiling_us"), ceiling):
            problems.append(f"{prefix}: region ceiling drifted")
    else:
        expected_region = sorted(size for size, _ in priced)
    if entry.get("k_star_region") != expected_region:
        problems.append(
            f"{prefix}: k* region {entry.get('k_star_region')} is not the set of "
            f"rungs reaching the smallest upper bound ({expected_region})"
        )
    if entry.get("k_star_point") not in (entry.get("k_star_region") or []):
        problems.append(
            f"{prefix}: k* point {entry.get('k_star_point')} lies outside its own "
            "region, which its construction forbids"
        )
    if entry.get("resolved") != (len(expected_region) == 1):
        problems.append(f"{prefix}: resolved flag disagrees with the region width")
    expected_status = "resolved" if len(expected_region) == 1 else "region"
    if uppers and entry.get("status") != expected_status:
        problems.append(
            f"{prefix}: status {entry.get('status')!r} disagrees with a region of "
            f"width {len(expected_region)}"
        )
    return problems


def _verdict_problems(prefix: str, payload: dict, mapping: str) -> list[str]:
    """QR1 and QR4 are summaries; they must summarize what is in the record."""
    problems: list[str] = []
    entry = payload["by_arm"][mapping]
    qr1 = payload["qr1_settings_proxy"][mapping]
    costs = entry.get("costs", [])
    if len(costs) < 2:
        if qr1.get("comparable"):
            problems.append(f"{prefix}: QR1 claims comparability with < 2 priced rungs")
        return problems
    by_cost = [
        item["block_size"] for item in
        sorted(costs, key=lambda item: (item["C_time_epsilon_us"], item["block_size"]))
    ]
    by_settings = [
        item["block_size"] for item in
        sorted(costs, key=lambda item: (item["settings"], item["block_size"]))
    ]
    if qr1.get("by_accuracy_matched_cost") != by_cost:
        problems.append(f"{prefix}: QR1 cost ordering drifted")
    if qr1.get("by_setting_count") != by_settings:
        problems.append(f"{prefix}: QR1 setting-count ordering drifted")
    if qr1.get("reorders_settings_ordering") != (by_cost != by_settings):
        problems.append(f"{prefix}: QR1 verdict disagrees with its own orderings")
    return problems



def _scope_problems(record: dict) -> list[str]:
    """Every structural system is evaluated or deferred, never silently absent.

    ``cost_layer_scope.systems`` is the evaluated set. It includes a bank whose
    bias floor makes pricing impossible, because that failure is itself the
    result. A deferred bank separately records the evidence status and the
    decision not to extend the frozen protocol.
    """
    problems: list[str] = []
    scope = record.get("cost_layer_scope")
    if not isinstance(scope, dict):
        return ["the cost layer declares no scope"]
    structural = record.get("structural_reference", {}).get(
        "systems_in_structural_grid"
    )
    if not isinstance(structural, list) or not structural:
        return ["the record does not name the structural grid it narrows"]

    evaluated = list(scope.get("systems", []))
    deferred = list(scope.get("deferred", []))
    named = [
        item
        for item in deferred
        if isinstance(item, dict) and isinstance(item.get("system"), str)
    ]
    for index, item in enumerate(deferred):
        if item not in named:
            problems.append(f"deferral {index} names no system")
    deferred_keys = [item["system"] for item in named]
    if not all(isinstance(key, str) for key in evaluated):
        problems.append(
            "cost-layer scope evaluates something that is not a system name"
        )
        evaluated = [key for key in evaluated if isinstance(key, str)]
    if sorted(evaluated + deferred_keys) != sorted(
        key for key in structural if isinstance(key, str)
    ):
        problems.append(
            f"cost-layer scope {sorted(evaluated + deferred_keys)} does not "
            f"partition the structural grid {sorted(structural, key=repr)}"
        )
    if set(evaluated) & set(deferred_keys):
        problems.append("a system is both evaluated and deferred")
    if sorted(record.get("systems", {})) != sorted(evaluated):
        problems.append(
            "the systems the record carries are not the ones its scope evaluates"
        )

    for item in named:
        key = item["system"]
        if not item.get("reason"):
            problems.append(f"{key}: deferred with no reason")
        status = item.get("status")
        if not isinstance(status, str) or not status:
            problems.append(f"{key}: deferred with no evidence status")
        if status == "right_censored":
            if (
                item.get("search_ceiling_effective_shots_per_setting")
                != SEARCH_ENDPOINTS[-1]
            ):
                problems.append(
                    f"{key}: right-censored search does not name the frozen "
                    f"ceiling {SEARCH_ENDPOINTS[-1]}"
                )
            if item.get("further_search") != "deferred":
                problems.append(
                    f"{key}: right-censored evidence does not record "
                    "further_search=deferred"
                )

        probe = item.get("scoping_probe")
        if isinstance(probe, dict) and probe.get("is_a_record") is not False:
            problems.append(f"{key}: a scoping probe does not disclaim record status")
        if isinstance(probe, dict):
            for field in ("exploratory_replicas", "confirmatory_replicas"):
                value = probe.get(field)
                headline = (
                    EXPLORATORY_REPLICAS if field.startswith("expl")
                    else CONFIRMATORY_REPLICAS
                )
                if isinstance(value, int) and value >= headline:
                    problems.append(
                        f"{key}: scoping probe claims {field}={value}, at or above "
                        f"the headline {headline}; that is a record, not a probe"
                    )
            if status == "right_censored":
                unresolved = probe.get("single_assignment_cells_unresolved")
                total = probe.get("single_assignment_cells_total")
                if (
                    not isinstance(unresolved, int)
                    or not isinstance(total, int)
                    or unresolved <= 0
                    or unresolved > total
                ):
                    problems.append(
                        f"{key}: right-censoring probe carries no valid unresolved "
                        "cell count"
                    )
        elif status == "right_censored":
            problems.append(f"{key}: right-censored deferral carries no scoping probe")
    return problems

def _qr3_problems(record: dict) -> list[str]:
    """QR3's verdict is re-derived from the record's own costs, not trusted.

    This is the contract that most needs re-deriving rather than reading: the
    verdict is the one field a reader will quote, it is a *conclusion* rather
    than a measurement, and the producer that wrote it is the code under test.
    So the check recomputes the whole payload from the priced cells the record
    itself carries and requires the stored one to match.
    """
    problems: list[str] = []
    stored = record.get("qr3_accuracy_matched")
    if not isinstance(stored, dict):
        return ["the accuracy-matched mapping-versus-instance verdict is missing"]
    systems = record.get("systems", {})
    priced = [
        key for key, value in systems.items() if value.get("status") == "searched"
    ]
    cards = [
        SimpleNamespace(name=card.get("name"))
        for card in record.get("device_cards", [])
    ]
    if (stored.get("status") == "abstains") != (len(priced) < 2):
        problems.append(
            f"QR3 status {stored.get('status')!r} disagrees with {len(priced)} "
            "priced instance(s): it abstains if and only if fewer than two are priced"
        )
    derived = _instance_cost_spread(systems, cards, priced)
    problems.extend(
        f"QR3: {problem}"
        for problem in compare_json_records(stored, derived, atol=0.0, rtol=1e-12)
    )
    return problems


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "exact":
        problems.append("oracle comparator is not labelled exact")
    if record.get("search_uncertainty_evidence") != "heuristic":
        problems.append("Monte Carlo search uncertainty is not labelled heuristic")
    protocol = record.get("protocol", {})
    if protocol.get("exploratory_replicas", 0) < EXPLORATORY_REPLICAS:
        problems.append("headline record has fewer than 30 exploratory replicas")
    if protocol.get("confirmatory_replicas", 0) < CONFIRMATORY_REPLICAS:
        problems.append("headline record has fewer than 100 confirmatory replicas")
    if protocol.get("search_endpoints_effective_shots_per_setting") != list(SEARCH_ENDPOINTS):
        problems.append("shot-search endpoint grid drifted")
    if list(protocol.get("estimators", [])) != list(ESTIMATORS):
        problems.append("estimator pair drifted")
    if record.get("accuracy_target_millihartree") != ACCURACY_TARGET_MILLIHARTREE:
        problems.append(
            "the accuracy target drifted from the one the rest of R1-R3 is "
            "matched at, so these costs are not comparable to their records"
        )
    problems.extend(_qr3_problems(record))
    problems.extend(_scope_problems(record))
    card_hashes = {card.get("name"): card.get("sha256") for card in record.get("device_cards", [])}
    if not card_hashes:
        problems.append("no device cards declared")

    frozen = structural_settings()
    target = record.get("accuracy_target_millihartree", float("inf"))
    arms_declared = list(protocol.get("mapping_arms", []))

    for key, system in record.get("systems", {}).items():
        if [arm["mapping"] for arm in system.get("arms", [])] != arms_declared:
            problems.append(f"{key}: arm list does not match the declared arms")
        bias = system.get("max_exact_subspace_bias_millihartree", float("inf"))
        expected_status = "searched" if bias < target else "bias_floor_exceeds_target"
        if system.get("status") != expected_status:
            problems.append(f"{key}: status disagrees with its exact subspace bias")

        for arm in system.get("arms", []):
            mapping = arm["mapping"]
            width = arm["measured_qubits"]
            sizes = [rung["block_size"] for rung in arm["rungs"]]
            if sizes != sorted(set(sizes)):
                problems.append(f"{key}/{mapping}: rungs are unordered or duplicated")
            if any(size > width for size in sizes):
                problems.append(
                    f"{key}/{mapping}: a rung exceeds the arm's {width} measured qubits"
                )
            for rung in arm["rungs"]:
                expected = frozen.get((key, mapping, rung["block_size"]))
                if expected is not None and rung.get("settings") != expected:
                    problems.append(
                        f"{key}/{mapping} k={rung['block_size']}: "
                        f"{rung.get('settings')} settings, the structural R3 grid "
                        f"froze {expected}"
                    )
                if set(rung.get("estimators", {})) != set(ESTIMATORS):
                    problems.append(
                        f"{key}/{mapping} k={rung['block_size']}: estimator pair "
                        "is incomplete"
                    )
                    continue
                for estimator, cell in rung["estimators"].items():
                    prefix = f"{key}/{mapping} k={rung['block_size']} {estimator}"
                    if expected_status != "searched":
                        if cell.get("device_costs") or cell.get(
                            "cost_bracket", {}
                        ).get("priced"):
                            problems.append(
                                f"{prefix}: priced above its system's bias floor"
                            )
                        continue
                    problems.extend(_crossing_problems(prefix, cell))
                    problems.extend(_bracket_problems(prefix, cell, card_hashes))

        if expected_status != "searched":
            if "k_star" in system:
                problems.append(f"{key}: an unpriced system reported a k*")
            continue

        by_mapping = {arm["mapping"]: arm for arm in system["arms"]}
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
                    problems.extend(
                        _k_star_problems(prefix, by_mapping[mapping], entry, card, estimator)
                    )
                    problems.extend(
                        _verdict_problems(prefix, payload[estimator], mapping)
                    )
            for mapping, verdict in payload.get("qr4_pooling", {}).items():
                left = payload["single_assignment"]["by_arm"][mapping]["k_star_region"]
                right = payload["pooled"]["by_arm"][mapping]["k_star_region"]
                prefix = f"{key}/{card}/{mapping} QR4"
                if not left or not right:
                    if verdict.get("comparable"):
                        problems.append(f"{prefix}: comparable with an empty region")
                    continue
                if verdict.get("single_assignment_region") != left or (
                    verdict.get("pooled_region") != right
                ):
                    problems.append(f"{prefix}: quoted regions drifted")
                if verdict.get("pooling_moves_k_star") != (not (set(left) & set(right))):
                    problems.append(
                        f"{prefix}: verdict disagrees with the region overlap it quotes"
                    )

        cross = system.get("r1_cross_check", {})
        if cross.get("status") != "compared":
            problems.append(f"{key}: the R1 cross-check did not run")
        else:
            expected_rows = r1_crossings()
            for row in cross.get("rows", []):
                pinned = expected_rows.get((row["block_size"], row["estimator"]), {})
                if row.get("r1_confirmed_passing") != pinned.get("confirmed_passing"):
                    problems.append(
                        f"{key}: R1 cross-check quotes {row.get('r1_confirmed_passing')} "
                        f"for k={row['block_size']} {row['estimator']}, the frozen "
                        f"record says {pinned.get('confirmed_passing')}"
                    )
                if row.get("r1_environment_marginal") != pinned.get(
                    "environment_marginal"
                ):
                    problems.append(
                        f"{key}: R1 cross-check misquotes the marginal flag for "
                        f"k={row['block_size']} {row['estimator']}"
                    )
                if row.get("agrees") != (
                    row.get("r1_confirmed_passing") == row.get("r3_confirmed_passing")
                ):
                    problems.append(f"{key}: R1 cross-check agreement flag drifted")
            disagreeing = [row for row in cross.get("rows", []) if not row.get("agrees")]
            if cross.get("disagreements_all_environment_marginal") != all(
                row.get("r1_environment_marginal") or row.get("r3_environment_marginal")
                for row in disagreeing
            ):
                problems.append(f"{key}: R1 cross-check marginal summary drifted")
            # A disagreement on a crossing *neither* record calls marginal means
            # two independent streams resolved the same bank differently, which
            # is a finding about the search rather than about R3.
            for row in disagreeing:
                if not (
                    row.get("r1_environment_marginal")
                    or row.get("r3_environment_marginal")
                ):
                    problems.append(
                        f"{key}: k={row['block_size']} {row['estimator']} reprices "
                        f"R1's {row.get('r1_confirmed_passing')} as "
                        f"{row.get('r3_confirmed_passing')} on a crossing neither "
                        "record calls environment-marginal"
                    )
    return problems




def _subset_config(keys: tuple[str, ...]) -> dict:
    """Narrow the config to a rebuildable subset of the systems this layer evaluates.

    Both lists move together. ``_cost_layer`` requires the cost layer's systems
    to partition the structural ones, so narrowing ``systems`` alone hands the
    producer a config it rejects. Only evaluated systems may be named: a deferred
    system has nothing to rebuild, so asking for one is a mistake worth
    reporting rather than an empty comparison worth running.
    """
    config = load_config(CONFIG)
    layer = config.get("cost_layer") or {"systems": config["systems"], "deferred": []}
    evaluated = list(layer["systems"])
    unknown = sorted(set(keys) - set(evaluated))
    if unknown:
        deferred = {
            item["system"] for item in layer.get("deferred", []) if item.get("system")
        }
        detail = sorted(set(unknown) & deferred)
        hint = f" ({', '.join(detail)} is deferred, not evaluated)" if detail else ""
        raise ValueError(f"unknown systems: {unknown}{hint}")
    subset = [key for key in evaluated if key in keys]
    return {
        **config,
        "systems": subset,
        # The subset is its own complete partition: everything it declares is
        # evaluated, nothing is deferred. The committed record's scope is not what
        # a subset run compares -- it compares system subtrees -- so narrowing
        # the declaration here costs nothing and keeps the producer's own
        # partition check meaningful rather than bypassed.
        "cost_layer": {**layer, "systems": subset, "deferred": []},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default="",
        help=(
            "comma-separated subset to rebuild and compare; omitted means the "
            "whole record. The record-level contracts are re-derived from the "
            "committed record either way, so a subset run still gates them -- "
            "what a subset cannot do is prove the systems it skipped rebuild."
        ),
    )
    args = parser.parse_args()
    keys = tuple(value for value in args.systems.split(",") if value)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    # Same reason check_exact_shot_search states: every value here descends from
    # sampled shots fed through an ill-conditioned projected eigensolve, so a
    # differing environment answers a different question rather than verifying
    # this one, and a value diff would invite a widened tolerance as the repair.
    stream = sampling_stream_mismatch(expected)
    if stream:
        print("protocol cost: FAIL (build environment differs)")
        for problem in stream:
            print(f"  {problem}")
        print("  skipped the rebuild: under a different environment it answers a "
              "different question rather than verifying this record")
        problems = contract_problems(expected)
        if problems:
            print("  the committed record also fails its own contracts:")
            for problem in problems[:30]:
                print(f"    {problem}")
        else:
            print("  the committed record still passes every contract check "
                  "that does not require a rebuild")
        return 1
    if keys:
        # A subset rebuild is compared subtree by subtree rather than whole:
        # the record-level summaries are functions of *every* system, so a
        # partial rebuild would legitimately differ on them and comparing them
        # would report that as drift. They are re-derived from the committed
        # record below, which needs no rebuild at all.
        actual = build_record(workers=4, config=_subset_config(keys))
        problems = []
        for key in keys:
            problems.extend(
                f"{key}: {problem}"
                for problem in compare_json_records(
                    expected["systems"][key],
                    actual["systems"][key],
                    atol=CROSS_MACHINE_ATOL,
                    rtol=CROSS_MACHINE_RTOL,
                )
            )
    else:
        actual = build_record(workers=4)
        problems = compare_json_records(expected, actual, atol=CROSS_MACHINE_ATOL, rtol=CROSS_MACHINE_RTOL)
    problems.extend(contract_problems(expected))
    problems.extend(f"rebuilt record: {problem}" for problem in contract_problems(actual))
    if problems:
        print("protocol cost: FAIL")
        for problem in problems[:30]:
            print(f"  {problem}")
        return 1
    print("protocol cost: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
