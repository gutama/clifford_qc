#!/usr/bin/env python3
"""R3 -- the accuracy-matched ``C(epsilon)`` layer and the ``k*`` regions.

``run_protocol_axis.py`` shipped R3's structural half: group counts, coverage,
and synthesis resources over the ``mapping x k`` grid, priced at uniform shots
and labelled ``structural`` precisely because a setting count is not a cost.
This producer closes the other half. It runs R1's exact-oracle nonlinear shot
search -- actual joint bitstrings, the full measured ``(S, H)`` reconstruction,
the ``E + 2 sigma`` selected-rank sweep, replica RMSE against the exact sector
ground energy -- once per ``(mapping arm, k)`` cell, and turns the confirmed
shot-to-target crossings into ``C_time(epsilon)`` and the ``k*`` of PLAN §6.7.

**BeH2 only, and the record says why.** H4's frozen bank carries a 3.019 mHa
exact subspace bias against a 1.6 mHa target, so no shot count reaches the
target on any arm and no arm may be priced. H4 is recorded with that status
rather than dropped, because "unattainable at this target" is the measurement.

**A crossing is a bracket, so a cost is an interval.** The search resolves a
shot count only to the geometric grid: the true count lies in
``(confirmed_fail, confirmed_pass]``, so the rung's honest cost lies in
``(C(confirmed_fail), C(confirmed_pass)]``. R1 additionally flags a crossing as
*environment-marginal* when either deciding endpoint sits within +-10% of the
target -- a band calibrated against a NumPy change that actually moved one
reported count by a factor of four. PLAN §6.7 names this layer as that flag's
downstream consumer, so a flagged side widens the interval by one grid step on
the side that could move. ``k*`` is then reported as the set of rungs whose
interval reaches the smallest upper bound: a single ``k`` where the intervals
separate, a region where they do not. The point argmin of the reported counts
is recorded beside it, and it is always inside its own region.

    python benchmarks/run_protocol_cost.py --workers 4
    python benchmarks/check_protocol_cost.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Sequence

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import Generator
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.elements import MatrixElementBank

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_clifford_hierarchy import (
        ACCURACY_TARGET_MILLIHARTREE,
        _compiled_settings,
        _synthesize,
    )
    from benchmarks.run_exact_shot_search import (
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        SEARCH_ENDPOINTS,
        _confirm_result,
        _confirmation_endpoints,
        _device_prices,
        _run_endpoints,
        _summaries,
    )
    from benchmarks.run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from benchmarks.run_protocol_axis import CONFIG, _mapped_bank, load_config
except ImportError:  # pragma: no cover - direct script execution
    from run_clifford_hierarchy import (
        ACCURACY_TARGET_MILLIHARTREE,
        _compiled_settings,
        _synthesize,
    )
    from run_exact_shot_search import (
        CONFIRMATORY_REPLICAS,
        ESTIMATORS,
        EXPLORATORY_REPLICAS,
        SEARCH_ENDPOINTS,
        _confirm_result,
        _confirmation_endpoints,
        _device_prices,
        _run_endpoints,
        _summaries,
    )
    from run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from run_protocol_axis import CONFIG, _mapped_bank, load_config

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference_results" / "protocol_cost.json"
STRUCTURAL_REFERENCE = HERE / "reference_results" / "protocol_axis.json"
R1_REFERENCE = HERE / "reference_results" / "exact_shot_search.json"
SCHEMA = "clifford_qc.protocol_cost.v1"

# Disjoint from R1's 260_8/260_9/261_0 roots by construction, and deliberately
# not a reuse of them: the jw arm reprices the same BeH2 bank R1 already
# searched, so drawing an independent stream makes the agreement of the two
# records evidence about the search rather than an identity.
EXPLORATORY_SEED = 270_813_000
CONFIRMATORY_SEED = 270_913_000
BOOTSTRAP_SEED = 271_013_000


def _system_specs() -> dict[str, dict]:
    return {spec["key"]: spec for spec in load_mapping_config()["systems"]}


def structural_settings() -> dict[tuple[str, str, int], int]:
    """The setting count the structural R3 grid froze, per ``(system, arm, k)``."""
    record = json.loads(STRUCTURAL_REFERENCE.read_text(encoding="utf-8"))
    return {
        (system["system"], arm["mapping"], rung["block_size"]): int(rung["settings"])
        for system in record["systems"]
        for arm in system["arms"]
        for rung in arm["rungs"]
    }


def r1_crossings() -> dict[tuple[int, str], dict]:
    """R1's confirmed BeH2 crossings and their flags, keyed ``(k, estimator)``.

    R1 searched BeH2 through the DA-CASE producer; this record reaches the same
    bank through the mapping-axis selection transported by the identity-like
    ``jw`` encoding. The two paths agree on the exact energy to 1e-14, so the
    ``jw`` column is a like-for-like repricing under an independent stream.

    R1's own environment-marginal flag comes along, because it is what says
    which of its counts were never resolved in the first place: three of R1's
    eight BeH2 crossings are flagged, and those are exactly the ones an
    independent stream is entitled to land on the other side of.
    """
    record = json.loads(R1_REFERENCE.read_text(encoding="utf-8"))
    return {
        (int(row["block_size"]), estimator): {
            "confirmed_passing": arm["shot_to_target"].get(
                "confirmed_passing_effective_shots_per_setting"
            ),
            "environment_marginal": bool(
                arm["shot_to_target"].get("crossing_is_environment_marginal")
            ),
        }
        for row in record["systems"]["beh2"]["rows"]
        for estimator, arm in row["estimators"].items()
    }


_ARM_CACHE: dict[tuple[str, str], dict] = {}


def arm_problem(system: str, mapping: str) -> dict:
    """Build one ``(system, mapping)`` bank, exactly as the structural grid does.

    Cached per process so a worker sweeping four rungs of one arm pays the
    transport once. The word universe drops the identity code for the same
    reason ``run_protocol_axis`` does -- it is what makes this grid's setting
    counts the structural record's setting counts -- and the grouped word
    sessions drop it again downstream, as R1's search already does.
    """
    key = (system, mapping)
    if key in _ARM_CACHE:
        return _ARM_CACHE[key]
    spec = _system_specs()[system]
    model, _ = _build_model(spec)
    _, selected = _selected_generators(model, spec)
    reference = ExactMVBackend().state(model.reference, ())
    sector = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact_energy = float(sector.ground_state(model.hamiltonian, k=1)[0][0])

    _, restriction = _mapped_bank(mapping, model, reference)
    transported = restriction.transport(
        hamiltonian=as_multivector(model.hamiltonian),
        reference=reference,
        generators=(generator.mv for generator in selected),
    )
    mapped = [
        Generator(generator.label, image)
        for generator, image in zip(selected, transported.generators)
    ]
    bank = MatrixElementBank(transported.reference, transported.hamiltonian, mapped)
    result = bank.solve()
    problem = {
        "system": system,
        "mapping": mapping,
        "n_qubits": transported.n,
        "codes": sorted(code for code in bank.word_set() if code != 0),
        "bank": bank,
        "result": result,
        "exact_ground_energy": exact_energy,
        "ground_energy": float(result.ground_energy),
        "bias_millihartree": abs(float(result.ground_energy) - exact_energy) * 1e3,
        "retained_rank": int(result.effective_rank),
        "basis_size": len(mapped),
    }
    _ARM_CACHE[key] = problem
    return problem


def _cost_bracket(
    cards,
    resources,
    n_qubits: int,
    search: dict,
) -> dict:
    """Turn one confirmed crossing into a per-card cost interval.

    Three endpoints are priced, not one. The *point* endpoint is the smallest
    confirmed pass, which is the count R1 reports. The *upper* endpoint is that
    same count unless the passing side is environment-marginal, in which case
    the deciding endpoint could fail elsewhere and push the count one grid step
    up. The *lower* endpoint is the largest confirmed failure -- the count is
    provably not smaller -- stepped one grid point down when the failing side is
    marginal and so could pass elsewhere. A crossing that lies below the grid
    has no confirmed failure and therefore no lower bound at all; it is recorded
    as unbounded below rather than pinned to the smallest endpoint tested.
    """
    passing = search.get("confirmed_passing_effective_shots_per_setting")
    failing = search.get("confirmed_failing_effective_shots_per_setting")
    if passing is None:
        return {
            "status": search.get("status"),
            "priced": False,
            "point_effective_shots_per_setting": None,
            "lower_effective_shots_per_setting": None,
            "upper_effective_shots_per_setting": None,
            "unbounded_below": True,
            "unbounded_above": True,
            "widened_sides": [],
            "cards": {},
        }
    grid = list(SEARCH_ENDPOINTS)
    marginal = search.get("environment_marginal_endpoints") or []
    widened = []

    upper = passing
    if "passing" in marginal:
        index = grid.index(passing)
        if index + 1 < len(grid):
            upper = grid[index + 1]
            widened.append("passing")
        else:  # the top of the grid cannot be stepped past
            upper = None
            widened.append("passing")

    lower = failing
    if failing is not None and "failing" in marginal:
        index = grid.index(failing)
        lower = grid[index - 1] if index > 0 else None
        widened.append("failing")

    point_prices = _device_prices(cards, resources, passing, n_qubits)
    lower_prices = _device_prices(cards, resources, lower, n_qubits)
    upper_prices = _device_prices(cards, resources, upper, n_qubits)

    def scalar(prices: dict, name: str):
        cost = prices.get(name)
        return None if cost is None else cost["accuracy"]["C_time_epsilon_us"]

    per_card = {}
    for card in cards:
        point = scalar(point_prices, card.name)
        per_card[card.name] = {
            "device_card_sha256": card.sha256,
            # An inadmissible rung has no runtime on this card at any shot
            # count, so it carries no interval either -- reported as such
            # rather than folded into a minimum it cannot attain.
            "admissible": point is not None,
            "C_time_epsilon_us": point,
            "C_time_lower_us": 0.0 if lower is None else scalar(lower_prices, card.name),
            "C_time_upper_us": None if upper is None else scalar(upper_prices, card.name),
        }
    return {
        "status": search.get("status"),
        "priced": True,
        "point_effective_shots_per_setting": passing,
        "lower_effective_shots_per_setting": lower,
        "upper_effective_shots_per_setting": upper,
        "unbounded_below": lower is None,
        "unbounded_above": upper is None,
        "widened_sides": widened,
        "cards": per_card,
    }


def _search_cell(arguments) -> dict:
    """One ``(arm, k)`` cell: R1's search, then its cost interval."""
    (system, mapping, arm_index, block_size,
     exploratory_replicas, confirmatory_replicas, cards) = arguments
    problem = arm_problem(system, mapping)
    codes = problem["codes"]
    n = problem["n_qubits"]
    bank = problem["bank"]
    result = problem["result"]

    groups = block_commuting_partition(n, codes, block_size)
    row, resources, compatibility, _ = _synthesize(n, codes, groups, block_size)
    settings = _compiled_settings(n, codes, groups, block_size)
    if any(len(setting.readouts) != int(compatibility[index].sum())
           for index, setting in enumerate(settings)):
        raise AssertionError("compiled readouts disagree with hierarchy compatibility")

    frozen = structural_settings().get((system, mapping, block_size))
    if frozen is not None and row["settings"] != frozen:
        raise ValueError(
            f"{system}/{mapping} k={block_size}: {row['settings']} settings but "
            f"the structural R3 grid froze {frozen}; this layer would be pricing "
            f"a different partition than the one it claims to price"
        )

    namespace = (arm_index,)
    requested = {estimator: SEARCH_ENDPOINTS for estimator in ESTIMATORS}
    exploratory_raw = _run_endpoints(
        reference=bank.reference,
        bank=bank,
        result=result,
        settings=settings,
        requested=requested,
        replicas=exploratory_replicas,
        seed=EXPLORATORY_SEED,
        block_size=block_size,
        namespace=namespace,
    )
    exploration = _summaries(
        exploratory_raw, problem["exact_ground_energy"],
        block_size=block_size, seed_namespace=0, namespace=namespace,
    )
    confirmation_requested = {
        estimator: _confirmation_endpoints(exploration[estimator])
        for estimator in ESTIMATORS
    }
    confirmation_raw = _run_endpoints(
        reference=bank.reference,
        bank=bank,
        result=result,
        settings=settings,
        requested=confirmation_requested,
        replicas=confirmatory_replicas,
        seed=CONFIRMATORY_SEED,
        block_size=block_size,
        namespace=namespace,
    )
    confirmation = _summaries(
        confirmation_raw, problem["exact_ground_energy"],
        block_size=block_size, seed_namespace=1, namespace=namespace,
    )

    estimators = {}
    for estimator in ESTIMATORS:
        search = _confirm_result(exploration[estimator], confirmation[estimator])
        estimators[estimator] = {
            "exploration": exploration[estimator],
            "confirmation": confirmation[estimator],
            "shot_to_target": search,
            "device_costs": _device_prices(
                cards, resources,
                search["confirmed_passing_effective_shots_per_setting"], n,
            ),
            "cost_bracket": _cost_bracket(cards, resources, n, search),
        }
    return {
        "mapping": mapping,
        "block_size": block_size,
        "settings": row["settings"],
        "compatible_readouts": int(compatibility.sum()),
        "mean_reader_count": float(compatibility.sum(axis=0).mean()),
        "estimators": estimators,
    }


def _unpriced_cell(mapping: str, block_size: int, settings: int | None) -> dict:
    """A rung of a system whose bias floor already exceeds the target."""
    return {
        "mapping": mapping,
        "block_size": block_size,
        "settings": settings,
        "estimators": {
            estimator: {
                "shot_to_target": {
                    "status": "bias_floor_exceeds_target",
                    "confirmed_failing_effective_shots_per_setting": None,
                    "confirmed_passing_effective_shots_per_setting": None,
                },
                "device_costs": {},
                "cost_bracket": {
                    "status": "bias_floor_exceeds_target",
                    "priced": False,
                    "cards": {},
                },
            }
            for estimator in ESTIMATORS
        },
    }


def _k_star(arm: dict, card_name: str, estimator: str) -> dict:
    """PLAN §6.7's ``k*`` for one arm, card, and estimator -- as a region.

    ``point`` is the argmin of the reported counts, which is what a record that
    read each crossing as an integer would publish. ``region`` is every rung
    whose cost interval reaches the smallest upper bound, and it is the answer
    this record stands behind: two rungs whose intervals overlap are not
    separated by this experiment, however their point estimates happen to sort.
    The point is always a member of its region -- ``C_point <= C_upper``
    elementwise makes the smallest upper bound at least the smallest point --
    and the checker re-derives that rather than trusting it.
    """
    priced, unresolved, inadmissible = [], [], []
    for rung in arm["rungs"]:
        bracket = rung["estimators"][estimator]["cost_bracket"]
        entry = bracket.get("cards", {}).get(card_name)
        if not bracket.get("priced") or entry is None:
            unresolved.append(rung["block_size"])
            continue
        if not entry["admissible"]:
            inadmissible.append(rung["block_size"])
            continue
        priced.append({
            "block_size": rung["block_size"],
            "settings": rung["settings"],
            "C_time_epsilon_us": entry["C_time_epsilon_us"],
            "C_time_lower_us": entry["C_time_lower_us"],
            "C_time_upper_us": entry["C_time_upper_us"],
        })
    base = {
        "unresolved_block_sizes": sorted(unresolved),
        "inadmissible_block_sizes": sorted(inadmissible),
        "priced_block_sizes": sorted(item["block_size"] for item in priced),
        "costs": sorted(priced, key=lambda item: item["block_size"]),
    }
    if not priced:
        return {
            **base,
            "status": "no_priced_rung",
            "k_star_point": None,
            "k_star_region": [],
            "resolved": None,
            "point_margin_over_runner_up": None,
        }
    best = min(priced, key=lambda item: (item["C_time_epsilon_us"], item["block_size"]))
    ordered = sorted(item["C_time_epsilon_us"] for item in priced)
    margin = (
        None if len(ordered) < 2 or ordered[0] <= 0.0
        else ordered[1] / ordered[0]
    )
    # A rung with no upper bound (its passing side is marginal at the top of
    # the grid) bounds nothing, so it cannot shrink another rung's region.
    uppers = [
        item["C_time_upper_us"] for item in priced
        if item["C_time_upper_us"] is not None
    ]
    if not uppers:
        return {
            **base,
            "status": "no_bounded_rung",
            "k_star_point": best["block_size"],
            "k_star_region": sorted(item["block_size"] for item in priced),
            "resolved": False,
            "point_margin_over_runner_up": margin,
        }
    ceiling = min(uppers)
    region = sorted(
        item["block_size"] for item in priced
        if item["C_time_lower_us"] <= ceiling
    )
    return {
        **base,
        "status": "resolved" if len(region) == 1 else "region",
        "k_star_point": best["block_size"],
        "k_star_region": region,
        "resolved": len(region) == 1,
        "region_ceiling_us": ceiling,
        "point_margin_over_runner_up": margin,
    }


def _settings_ordering_verdict(k_star: dict) -> dict:
    """QR1, per arm/card/estimator: does ``C(epsilon)`` reorder the rungs?

    The falsifier QR1 names is an ordering that always agrees with the setting
    count, in which case the count was an adequate proxy and this layer is
    overhead. Ties in either key are not reorderings, so both sequences are
    compared as the sort a tie-tolerant reader would produce.
    """
    costs = k_star["costs"]
    if len(costs) < 2:
        return {"comparable": False, "reorders_settings_ordering": None}
    by_cost = [
        item["block_size"] for item in
        sorted(costs, key=lambda item: (item["C_time_epsilon_us"], item["block_size"]))
    ]
    by_settings = [
        item["block_size"] for item in
        sorted(costs, key=lambda item: (item["settings"], item["block_size"]))
    ]
    return {
        "comparable": True,
        "by_accuracy_matched_cost": by_cost,
        "by_setting_count": by_settings,
        "reorders_settings_ordering": by_cost != by_settings,
    }


def _pooling_verdict(assigned: dict, pooled: dict) -> dict:
    """QR4: does coverage move ``k*`` between the two estimators?

    Answered on the regions, not on the point argmins. Two regions that
    overlap have not been shown to differ, so a moved point inside a shared
    region is not evidence that pooling relocated ``k*``.
    """
    left, right = assigned["k_star_region"], pooled["k_star_region"]
    if not left or not right:
        return {
            "comparable": False,
            "pooling_moves_k_star": None,
            "pooling_moves_k_star_point": None,
        }
    return {
        "comparable": True,
        "single_assignment_region": left,
        "pooled_region": right,
        "regions_disjoint": not (set(left) & set(right)),
        "pooling_moves_k_star": not (set(left) & set(right)),
        "pooling_moves_k_star_point": (
            assigned["k_star_point"] != pooled["k_star_point"]
        ),
    }


def _mapping_cost_spread(system: dict, card_name: str, estimator: str) -> dict:
    """The mapping's effect on ``C(epsilon)`` at fixed ``k``, by measured width.

    Grouped by measured qubit count for the reason the structural record gives:
    a ``+2q`` arm at ``k = n`` is a narrower full-commuting problem, so folding
    it into one spread would report a register-size effect as a mapping effect.
    An all-arm spread is reported beside it and labelled, because on *cost* --
    unlike on group count -- the two removed qubits are a real saving that a
    reader is entitled to see priced.
    """
    rows: dict[int, dict[int, list[tuple[str, float]]]] = {}
    everything: dict[int, list[tuple[str, float]]] = {}
    for arm in system["arms"]:
        width = arm["measured_qubits"]
        for rung in arm["rungs"]:
            entry = (
                rung["estimators"][estimator]["cost_bracket"]
                .get("cards", {})
                .get(card_name)
            )
            if entry is None or not entry.get("admissible"):
                continue
            value = entry["C_time_epsilon_us"]
            rows.setdefault(width, {}).setdefault(
                rung["block_size"], []
            ).append((arm["mapping"], value))
            everything.setdefault(rung["block_size"], []).append(
                (arm["mapping"], value)
            )

    def summarize(cells: dict[int, list[tuple[str, float]]], expected: int) -> dict:
        out = {}
        for block_size, values in sorted(cells.items()):
            if len(values) < 2 or len(values) != expected:
                continue
            costs = [value for _, value in values]
            cheapest = min(values, key=lambda item: (item[1], item[0]))
            dearest = max(values, key=lambda item: (item[1], item[0]))
            out[str(block_size)] = {
                "arms": [name for name, _ in values],
                "C_time_epsilon_us": costs,
                "spread": max(costs) / min(costs),
                "cheapest_mapping": cheapest[0],
                "dearest_mapping": dearest[0],
            }
        return out

    by_width = {}
    for width, cells in sorted(rows.items()):
        arms = sorted({
            arm["mapping"] for arm in system["arms"]
            if arm["measured_qubits"] == width
        })
        if len(arms) < 2:
            continue
        by_width[str(width)] = {
            "arms": arms,
            "by_block_size": summarize(cells, len(arms)),
        }
    return {
        "equal_measured_width": by_width,
        "all_arms": {
            "arms": [arm["mapping"] for arm in system["arms"]],
            "note": (
                "mixes 6- and 8-qubit arms; legitimate on runtime, where the "
                "two removed qubits are a real saving, and not comparable on "
                "group count, where it would be a register-size artefact"
            ),
            "by_block_size": summarize(everything, len(system["arms"])),
        },
    }


def _system_summary(system: dict, cards) -> dict:
    output = {}
    for card in cards:
        per_estimator = {}
        for estimator in ESTIMATORS:
            per_arm = {
                arm["mapping"]: _k_star(arm, card.name, estimator)
                for arm in system["arms"]
            }
            per_estimator[estimator] = {
                "by_arm": per_arm,
                "qr1_settings_proxy": {
                    mapping: _settings_ordering_verdict(payload)
                    for mapping, payload in per_arm.items()
                },
            }
        output[card.name] = {
            **per_estimator,
            "qr4_pooling": {
                mapping: _pooling_verdict(
                    per_estimator["single_assignment"]["by_arm"][mapping],
                    per_estimator["pooled"]["by_arm"][mapping],
                )
                for mapping in per_estimator["single_assignment"]["by_arm"]
            },
            "mapping_cost_spread": {
                estimator: _mapping_cost_spread(system, card.name, estimator)
                for estimator in ESTIMATORS
            },
        }
    return output


def _r1_cross_check(system: dict) -> dict:
    """Does the ``jw`` arm reprice R1's BeH2 crossings under a fresh stream?

    The two records reach the same bank by different routes and draw disjoint
    random streams, so an agreeing endpoint is corroboration rather than an
    identity. Disagreement is tolerated exactly where *either* record calls the
    crossing environment-marginal: that flag is the statement that the endpoint
    was never a resolved count, and it is symmetric -- a crossing R1 decided on
    the target is one this record may decide the other way even when its own
    replicas land clear of the band. A disagreement neither record flags would
    mean two independent streams resolved the same bank differently, which is a
    finding about the search rather than a detail of this layer, so the rows
    record both flags instead of a bare mismatch count.
    """
    jw = next((arm for arm in system["arms"] if arm["mapping"] == "jw"), None)
    if jw is None:
        return {"status": "no_jw_arm"}
    r1 = r1_crossings()
    rows = []
    for rung in jw["rungs"]:
        for estimator in ESTIMATORS:
            search = rung["estimators"][estimator]["shot_to_target"]
            here = search.get("confirmed_passing_effective_shots_per_setting")
            there = r1.get((rung["block_size"], estimator), {})
            rows.append({
                "block_size": rung["block_size"],
                "estimator": estimator,
                "r1_confirmed_passing": there.get("confirmed_passing"),
                "r3_confirmed_passing": here,
                "agrees": here == there.get("confirmed_passing"),
                "r1_environment_marginal": bool(there.get("environment_marginal")),
                "r3_environment_marginal": bool(
                    search.get("crossing_is_environment_marginal")
                ),
            })
    disagreements = [row for row in rows if not row["agrees"]]
    return {
        "status": "compared",
        "compared_crossings": len(rows),
        "agreeing_crossings": len(rows) - len(disagreements),
        "disagreements_all_environment_marginal": all(
            row["r1_environment_marginal"] or row["r3_environment_marginal"]
            for row in disagreements
        ),
        "rows": rows,
    }


def build_record(
    *,
    exploratory_replicas: int = EXPLORATORY_REPLICAS,
    confirmatory_replicas: int = CONFIRMATORY_REPLICAS,
    workers: int = 1,
    config: dict | None = None,
    cards=None,
) -> dict:
    if exploratory_replicas <= 0 or confirmatory_replicas <= 0:
        raise ValueError("replica counts must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config(CONFIG) if config is None else config
    cards = load_device_cards() if cards is None else list(cards)
    if not cards:
        raise ValueError("at least one device card is required")
    arms: Sequence[str] = config["mapping_arms"]
    block_sizes = config["protocol"]["block_sizes"]
    frozen = structural_settings()

    systems = {}
    for key in config["systems"]:
        arm_records = []
        cells: list[tuple] = []
        for arm_index, mapping in enumerate(arms):
            problem = arm_problem(key, mapping)
            n = problem["n_qubits"]
            # k is a block width, so it is clamped to the arm's own register --
            # the same clamp the structural grid applies, and the reason a
            # +2q arm records k=6 where a full-width arm records k=8.
            rungs = sorted({min(int(size), n) for size in block_sizes})
            arm_records.append({
                "mapping": mapping,
                "arm_index": arm_index,
                "measured_qubits": n,
                "word_universe": len(problem["codes"]),
                "basis_size": problem["basis_size"],
                "retained_rank": problem["retained_rank"],
                "ground_energy": problem["ground_energy"],
                "exact_subspace_bias_millihartree": problem["bias_millihartree"],
                "_rungs": rungs,
            })
            cells.extend(
                (key, mapping, arm_index, size,
                 exploratory_replicas, confirmatory_replicas, cards)
                for size in rungs
            )

        bias = max(arm["exact_subspace_bias_millihartree"] for arm in arm_records)
        searched = bias < ACCURACY_TARGET_MILLIHARTREE
        if searched:
            if workers == 1:
                results = [_search_cell(cell) for cell in cells]
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(_search_cell, cells))
        else:
            results = [
                _unpriced_cell(mapping, size, frozen.get((key, mapping, size)))
                for _, mapping, _, size, *_ in cells
            ]
        by_arm: dict[str, list[dict]] = {}
        for row in results:
            by_arm.setdefault(row["mapping"], []).append(row)
        for arm in arm_records:
            arm["rungs"] = sorted(
                by_arm[arm["mapping"]], key=lambda row: row["block_size"]
            )
            arm.pop("_rungs")

        system = {
            "system": key,
            "exact_ground_energy": arm_problem(key, arms[0])["exact_ground_energy"],
            "max_exact_subspace_bias_millihartree": bias,
            "status": "searched" if searched else "bias_floor_exceeds_target",
            "arms": arm_records,
        }
        if searched:
            system["k_star"] = _system_summary(system, cards)
            system["r1_cross_check"] = _r1_cross_check(system)
        systems[key] = system

    return {
        "schema": SCHEMA,
        "phase": "R3_accuracy_matched_cost",
        "discharges": [
            "accuracy-matched C(epsilon) over the mapping x k grid",
            "k* regions under three device cards, per §6.7",
        ],
        "evidence_tier": "exact",
        "search_uncertainty_evidence": "heuristic",
        "accuracy_target_millihartree": ACCURACY_TARGET_MILLIHARTREE,
        "primary_metric": (
            "replica RMSE of the nonlinear selected-rank Ritz energy against "
            "the full exact ground energy, per (mapping, k) cell"
        ),
        "pass_rule": (
            "zero solver failures and the one-sided 95% nonparametric-bootstrap "
            "upper bound on RMSE <= 1.6 mHa"
        ),
        "k_star_rule": (
            "k* is the set of rungs whose cost interval reaches the smallest "
            "upper bound; the interval is (C(confirmed_fail), C(confirmed_pass)], "
            "widened one grid step on any side R1 flagged environment-marginal"
        ),
        "claim_boundary": (
            "accuracy-matched logical runtime under declared illustrative device "
            "cards, on one instance at one target, with heuristic Monte Carlo "
            "search uncertainty; not a finite-sample energy certificate, a "
            "device-noise simulation, a hardware result, or an instance-independent "
            "preference for any mapping or block size"
        ),
        "protocol": {
            "mapping_arms": list(arms),
            "block_sizes": list(block_sizes),
            "estimators": list(ESTIMATORS),
            "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
            "nested_endpoints": True,
            "exploratory_replicas": exploratory_replicas,
            "confirmatory_replicas": confirmatory_replicas,
            "exploratory_seed": EXPLORATORY_SEED,
            "confirmatory_seed": CONFIRMATORY_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seed_derivation": (
                "numpy.SeedSequence(root, spawn_key=(arm_index, ...)) -- one "
                "disjoint namespace per mapping arm, and disjoint from R1's roots"
            ),
            "shot_search_producer": "benchmarks/run_exact_shot_search.py",
        },
        "structural_reference": {
            "path": "benchmarks/reference_results/protocol_axis.json",
            "settings_reproduced": True,
        },
        "r1_reference": {
            "path": "benchmarks/reference_results/exact_shot_search.json",
        },
        "qr3_accuracy_matched": {
            "status": "abstains",
            "reason": (
                "an accuracy-matched mapping-versus-instance comparison needs "
                "two priced instances, and only BeH2 clears its bias floor; the "
                "mapping spread in C(epsilon) is recorded, the cross-instance "
                "verdict is not"
            ),
        },
        "device_cards": [
            {"sha256": card.sha256, **card.to_dict()} for card in cards
        ],
        "systems": systems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploratory-replicas", type=int, default=EXPLORATORY_REPLICAS)
    parser.add_argument("--confirmatory-replicas", type=int, default=CONFIRMATORY_REPLICAS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = stamp_record(build_record(
        exploratory_replicas=args.exploratory_replicas,
        confirmatory_replicas=args.confirmatory_replicas,
        workers=args.workers,
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for key, system in record["systems"].items():
        print(f"{key}: {system['status']}")
        if system["status"] != "searched":
            continue
        for card, payload in system["k_star"].items():
            for estimator in ESTIMATORS:
                regions = " ".join(
                    f"{mapping}:{entry['k_star_region']}"
                    for mapping, entry in payload[estimator]["by_arm"].items()
                )
                print(f"  {card:22s} {estimator:17s} k*={regions}")
    print(args.out)


if __name__ == "__main__":
    main()
