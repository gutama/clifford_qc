#!/usr/bin/env python3
"""R3c -- the LiH ``margin_stop`` headline exact-tier cost run.

The exact tier has priced exactly one bank. BeH2 resolves inside the frozen
``64..65536`` grid; ``h4`` fails the accuracy floor and ``h4_converged`` is
right-censored above the grid, so ``protocol_cost.json`` carries
``qr3_accuracy_matched: abstains`` -- a mapping-versus-instance comparison needs
two priced instances and has one. R3S then admitted a second candidate: the
same LiH CAS(4e,4o) instance QR3b froze, stopped by the margin rule at ``M = 2``
and ``W = 1439`` instead of at the greedy's own threshold's ``M = 13`` and
``W = 7740``.

**Why the headline protocol and not another probe.** R3b drew that bank's forty
cells at ``2+2`` replicas and rejected it. The rejection stands, and this
producer does not reopen it -- but its labelled post-hoc diagnosis was that the
failure mode had changed completely: no cell failed to bracket a crossing inside
the grid, against fifteen such cells on the intrinsic-stop bank, and what
remained was exploratory crossings two confirmatory replicas did not reproduce.
That is a statement about a ``2+2`` instrument, not about ``W``. Re-sizing a
probe after seeing which cells failed is the move preregistration exists to
prevent; testing the *target* instrument is not, and that is what R3c declared.

**The authorization is R3c's own.** ``configs/r3c_lih_full_cost.json`` landed
result-free, before any sampling, carrying
``full_30_plus_100_run_authorized_by_this_config: true``. R3b's record still
carries ``full_run_authorized: false`` and this producer refuses to run if it
ever stops doing so: the two statements are about different declarations and
neither promotes the other.

**One execution, under a frozen stack.** Seed roots do not name the same stream
across NumPy releases, so the preregistration pins Python 3.12, NumPy 2.5.2,
SciPy 1.18.0 and Stim 1.16.0 -- the environment ``protocol_cost.json`` was built
under, which is also what makes the two instances comparable. Every gate that
can fail is checked before a shot is drawn: this run costs hours, and an
environment or lineage failure discovered at the stamp costs all of them.

**What this record may claim.** Accuracy-matched logical runtime on the frozen
LiH margin-stop bank at one target under three illustrative device cards. Each
cell either supplies a confirmed finite interval or is right-censored at
``65536``; censoring is not infinite cost and does not license a wider grid.
QR3 is re-derived against the frozen BeH2 price only if this record supplies a
second priced instance, and the earlier abstention stays true of the record that
carries it. The claim boundary is inherited from the config, which -- unlike
R3b's -- was written tenselessly so that a record reporting sampled cells does
not open by denying them.

    python benchmarks/run_r3c_lih_full_cost.py --workers 4
    python benchmarks/run_r3c_lih_full_cost.py --skip-run   # structural half only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from clifford_qc import record_environment
from clifford_qc.reproducibility import stamp_record

try:  # package import in tests versus direct benchmark execution
    from benchmarks.check_r3c_preregistration import CONFIG, load_config, static_problems
    from benchmarks.r3_environment_migration import FROZEN_CONFIG_SHA256
    from benchmarks.run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from benchmarks.run_mapping_axis import (
        _canonical_sha256,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from benchmarks.run_protocol_axis import build_system_record as build_protocol_system
    from benchmarks.run_protocol_cost import (
        _instance_cost_spread,
        arm_problem,
        build_record as build_cost_record,
    )
    from benchmarks.run_qr3b_instance_preflight import (
        _probe_config,
        _without_cost_derivatives,
    )
    from benchmarks.run_r3b_margin_stop_probe import (
        _downstream_spec,
        derive_selection as derive_margin_bank,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_r3c_preregistration import CONFIG, load_config, static_problems
    from r3_environment_migration import FROZEN_CONFIG_SHA256
    from run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
    from run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from run_mapping_axis import (
        _canonical_sha256,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from run_protocol_axis import build_system_record as build_protocol_system
    from run_protocol_cost import (
        _instance_cost_spread,
        arm_problem,
        build_record as build_cost_record,
    )
    from run_qr3b_instance_preflight import _probe_config, _without_cost_derivatives
    from run_r3b_margin_stop_probe import (
        _downstream_spec,
        derive_selection as derive_margin_bank,
    )

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference_results" / "r3c_lih_full_cost.json"
SCHEMA = "clifford_qc.r3c_lih_full_cost.v1"
R3B_RECORD = HERE / "reference_results" / "r3b_margin_stop_probe.json"
PROTOCOL_COST_RECORD = HERE / "reference_results" / "protocol_cost.json"
FIRST_PRICED_INSTANCE = "beh2"

#: How a cell's shot interval ends up, in the preregistration's own vocabulary.
#:
#: The acceptance rule reads "a finite interval requires a confirmed failing
#: predecessor and confirmed passing endpoint", and everything short of that at
#: ``65536`` is right-censored. Three distinct things fall short and they are
#: not interchangeable, so they are named rather than pooled: no confirmed
#: crossing anywhere in the grid; a crossing whose passing endpoint R1 flagged
#: environment-marginal at the last grid point, which has no finite upper bound;
#: and a crossing at the first endpoint tested, which has no confirmed failing
#: predecessor and so no lower bound.
#:
#: The first two are right-censored: neither has a finite upper bound inside the
#: grid, which is what the rule's "without a confirmed finite interval at 65536"
#: names. They are still distinguished, because only the first says the grid
#: never located a crossing at all, while the second located one and lost its
#: upper bound to R1's marginal flag at the last endpoint. The third is censored
#: on the *other* side and is counted separately for that reason.
FINITE_INTERVAL = "finite_interval"
NO_CONFIRMED_CROSSING = "right_censored_no_confirmed_crossing_in_grid"
OPEN_ABOVE = "open_above_marginal_at_grid_ceiling"
OPEN_BELOW = "open_below_no_confirmed_failing_predecessor"
OPEN_BOTH = "open_on_both_sides"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_cards(config: dict) -> dict[str, str]:
    return {
        row["name"]: row["sha256"] for row in config["protocol"]["device_cards"]
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def declared_environment(config: dict) -> dict[str, str]:
    """The preregistration's stack, keyed the way the shared contract keys it."""
    declared = config["protocol"]["execution_environment"]
    return {
        "python": declared["python_minor"],
        "numpy": declared["numpy"],
        "scipy": declared["scipy"],
        "stim": declared["stim"],
    }


def _installed_versions() -> dict[str, str | None]:
    """The running stack, keyed and truncated as the shared contract keys it."""
    versions: dict[str, str | None] = {
        "python": ".".join(
            platform.python_version().split(".")[: record_environment.PYTHON_PARTS]
        )
    }
    for name in ("numpy", "scipy", "stim"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_problems(config: dict) -> list[str]:
    """Refuse a stack the preregistration did not freeze, before sampling.

    ``stamp_record`` already refuses an environment no committed record
    declares, and that guard stays in place. It is not this one. It asks
    whether the run belongs to the repository's record set; this asks whether
    it is the *specific* stack R3c named, which is stricter and is the reason
    the config names it: ``numpy.SeedSequence`` roots do not define the same
    stream across NumPy releases, so a run under another version draws
    different shots from the same declared seeds and prices a different
    experiment under this record's name. It is also the stack
    ``protocol_cost.json`` was built under, which is what lets the BeH2 price
    and this one be compared at all.

    An absent library is a failure here rather than something to skip past.
    The cross-record gate passes over one because the chemistry and bridge
    extras are genuinely optional; these three are what draws and prices the
    replicas.

    Checked here rather than at the stamp because the difference is hours: the
    guard that fires after the last replica costs the whole run.
    """
    installed = _installed_versions()
    problems: list[str] = []
    for name, expected in sorted(declared_environment(config).items()):
        actual = installed.get(name)
        if actual is None:
            problems.append(
                f"{name} is not installed; R3c froze it at {expected} before "
                "any sampling"
            )
        elif actual != expected:
            problems.append(
                f"{name} {actual} is not the {expected} R3c froze before any "
                "sampling; the same seed roots name a different stream here"
            )
    return problems


def preregistration_problems(config: dict) -> list[str]:
    """The config must still be the file that landed, byte for byte.

    ``static_problems`` re-runs the whole result-free checker, which is what
    binds the bank, the protocol, the seed streams, the cards and the frozen
    R3b and QR3b findings. The digest check is the one thing it cannot do for
    itself: it validates whatever config it is given, and a preregistration
    edited after landing would still validate. The migration manifest froze
    this file's bytes in an earlier commit, so comparing against it is the
    statement that nothing was rewritten between the declaration and the run.
    """
    problems = [f"preregistration: {problem}" for problem in static_problems(config)]
    frozen = FROZEN_CONFIG_SHA256["r3c_lih_full_cost"]
    actual = _file_sha256(CONFIG)
    if actual != frozen:
        problems.append(
            f"the R3c config file digest is {actual}, not the {frozen} frozen "
            "when it landed; a preregistration edited after the fact is not one"
        )
    return problems


def r3b_untouched_problems() -> list[str]:
    """R3b rejected this bank for a full run, and still does.

    R3c's authorization is its own declaration, not an upgrade of that
    rejection, and the two only stay distinguishable while R3b's record keeps
    saying what it said. A run that found ``full_run_authorized: true`` there
    would mean the pilot had been rewritten to license its own successor, which
    is the one reading of this phase the preregistration forbids.
    """
    decision = _read(R3B_RECORD).get("decision", {})
    problems: list[str] = []
    if decision.get("status") != "rejected_unresolved_at_frozen_grid":
        problems.append("R3b's recorded rejection changed; R3c may not reopen it")
    if decision.get("full_run_authorized") is not False:
        problems.append(
            "R3b's record now authorizes a full run; R3c's authorization is its "
            "own config's and may not be sourced from the pilot it succeeds"
        )
    if decision.get("eligible_for_full_run") is not False:
        problems.append("R3b's record now makes the bank eligible")
    return problems


def qr3_reference_problems(config: dict, cards) -> list[str]:
    """Can the frozen BeH2 price be the other half of a QR3 comparison?

    Answered before sampling, because a mismatch here would make the comparison
    invalid after the shots are spent rather than before they are drawn. Two
    prices are comparable only if they were measured with the same instrument:
    the same replica counts, the same endpoint grid, the same estimators, the
    same cards, and the same stack -- the last because these are sampled
    quantities and the earlier record's own provenance is what R3c's frozen
    environment was set from.
    """
    problems: list[str] = []
    record = _read(PROTOCOL_COST_RECORD)
    protocol = record.get("protocol", {})
    declared = config["protocol"]

    if record.get("schema") != "clifford_qc.protocol_cost.v1":
        problems.append("the frozen cost record has an unexpected schema")
    # Each field is compared against whatever will really decide this run. The
    # replica counts are passed to the sampler out of the config; the endpoint
    # grid and the estimator pair are module constants the sampler reads for
    # itself, and the preregistration checker is what binds the config to them.
    for field in ("exploratory_replicas", "confirmatory_replicas"):
        if protocol.get(field) != declared[field]:
            problems.append(
                f"the frozen BeH2 price used {protocol.get(field)} {field} "
                f"against R3c's {declared[field]}; the two are not the same "
                "instrument and their costs are not comparable"
            )
    if protocol.get("search_endpoints_effective_shots_per_setting") != list(
        SEARCH_ENDPOINTS
    ):
        problems.append("the frozen BeH2 price used a different endpoint grid")
    if protocol.get("estimators") != list(ESTIMATORS):
        problems.append("the frozen BeH2 price used different estimators")

    roots = set(declared["seed_roots"].values())
    frozen_roots = {
        protocol.get("exploratory_seed"),
        protocol.get("confirmatory_seed"),
        protocol.get("bootstrap_seed"),
    }
    if roots & frozen_roots:
        problems.append(
            "an R3c seed root aliases the frozen cost record's stream, so the "
            "two instances would not be independently drawn"
        )

    frozen_cards = {row["name"]: row["sha256"] for row in record.get("device_cards", [])}
    if frozen_cards != {card.name: card.sha256 for card in cards}:
        problems.append(
            "the frozen BeH2 price was taken under different device cards, so "
            "its logical times are not on this record's scale"
        )

    provenance = record.get("provenance", {})
    dependencies = provenance.get("dependencies", {})
    built_under = {
        "python": ".".join(str(provenance.get("python", "")).split(".")[:2]),
        "numpy": dependencies.get("numpy"),
        "scipy": dependencies.get("scipy"),
        "stim": dependencies.get("stim"),
    }
    if built_under != declared_environment(config):
        problems.append(
            "the frozen BeH2 price was built under a different stack than the "
            "one R3c freezes; the two sampled records are not comparable"
        )

    beh2 = record.get("systems", {}).get(FIRST_PRICED_INSTANCE, {})
    if beh2.get("status") != "searched":
        problems.append("the frozen cost record no longer prices BeH2")
    return problems


def bank_problems(config: dict, mapping: dict, spec: dict) -> list[str]:
    """Every arm is the arm the preregistration priced its admission on.

    The margin rule is re-derived on the source side, where the linear encoding
    family leaves the bias invariant. What the encoding does change is the
    measured problem, and the preregistration froze that too, arm by arm: the
    word universe that decides reconstruction variance and the qwc setting count
    that the shot budget is spent per. Both are cheap to recompute and neither
    is checked anywhere else before the search starts, so a bank that had
    drifted would otherwise be discovered forty priced cells later.
    """
    problems: list[str] = []
    candidate = config["candidate"]
    target = float(config["acceptance_rule"]["accuracy_target_millihartree"])
    settings = {
        arm["mapping"]: arm["measurement"]["settings"] for arm in mapping["arms"]
    }
    for name in config["mapping_arms"]:
        problem = arm_problem(spec["key"], name, spec)
        words = len(problem["codes"])
        if words != candidate["expected_arm_word_universe"][name]:
            problems.append(
                f"{name}: re-derived W={words} against the declared "
                f"{candidate['expected_arm_word_universe'][name]}"
            )
        if settings.get(name) != candidate["expected_arm_settings"][name]:
            problems.append(
                f"{name}: re-derived {settings.get(name)} settings against the "
                f"declared {candidate['expected_arm_settings'][name]}"
            )
        if config["structural_gates"]["all_mapping_arms_must_clear_bias_floor"] and (
            problem["bias_millihartree"] >= target
        ):
            problems.append(
                f"{name}: bias {problem['bias_millihartree']} mHa does not clear "
                f"the {target} mHa floor, so no shot count reaches the target"
            )
        if problem["n_qubits"] > config["structural_gates"]["source_qubits"]:
            problems.append(f"{name}: measured register exceeds the source instance")
    binding = max(
        len(arm_problem(spec["key"], name, spec)["codes"])
        for name in config["mapping_arms"]
    )
    if binding != candidate["expected_binding_word_universe"]:
        problems.append("the binding word universe drifted from the preregistration")
    if binding > config["structural_gates"]["word_universe_ceiling"]:
        problems.append(
            f"binding W={binding} exceeds the screen's ceiling; R3S would not "
            "have admitted this bank"
        )
    return problems


def instrument_problems(system: dict, config: dict) -> list[str]:
    """Did the declared instrument draw these cells?

    Reads the replica counts back out of the sampled summaries rather than out
    of the arguments they were passed. The two agree unless something between
    the config and the sampler dropped a replica, and a cost record drawn at a
    count other than the frozen ``30+100`` is not the preregistered one however
    it came about.
    """
    problems: list[str] = []
    declared = config["protocol"]
    for arm in system["arms"]:
        for rung in arm["rungs"]:
            for estimator in ESTIMATORS:
                payload = rung["estimators"][estimator]
                for phase, expected in (
                    ("exploration", declared["exploratory_replicas"]),
                    ("confirmation", declared["confirmatory_replicas"]),
                ):
                    counts = {row["replicas"] for row in payload[phase]}
                    if counts - {expected}:
                        problems.append(
                            f"{arm['mapping']} k={rung['block_size']} {estimator}: "
                            f"{phase} drew {sorted(counts)} replicas, not "
                            f"the declared {expected}"
                        )
    return problems


def _classify(bracket: dict) -> str:
    if not bracket.get("priced"):
        return NO_CONFIRMED_CROSSING
    above = bool(bracket.get("unbounded_above"))
    below = bool(bracket.get("unbounded_below"))
    if above and below:
        return OPEN_BOTH
    if above:
        return OPEN_ABOVE
    if below:
        return OPEN_BELOW
    return FINITE_INTERVAL


def pricing_decision(system: dict, config: dict) -> dict:
    """Apply the frozen crossing and censoring rules to the drawn cells.

    The rule has one shape and two readouts. A cell with a confirmed failing
    predecessor and a confirmed passing endpoint has a finite interval; a cell
    without one at ``65536`` is right-censored, which is not infinite cost and
    does not license a wider grid. The record's status is the config's own
    ``possible_readouts``: finite confirmed brackets make LiH a second priced
    instance, and their absence leaves it censored at the frozen grid.

    ``right_censored_by_search_status`` is a tabulation and deliberately not a
    verdict. R3b's lesson was that the grid-fit and confirmation mechanisms have
    different remedies and that splitting them *after* the draw is a post-hoc
    diagnostic however true it is; the counts are reported because they are the
    data, and no gate here reads them.
    """
    rule = config["acceptance_rule"]
    ceiling = int(SEARCH_ENDPOINTS[-1])
    card_names = sorted(_frozen_cards(config))
    cells: list[dict] = []
    solves = {"attempted": 0, "failed": 0}
    for arm in system["arms"]:
        for rung in arm["rungs"]:
            for estimator in ESTIMATORS:
                payload = rung["estimators"][estimator]
                search = payload["shot_to_target"]
                bracket = payload["cost_bracket"]
                for row in payload["confirmation"]:
                    solves["attempted"] += int(row["replicas"])
                    solves["failed"] += int(row["replicas"]) - int(
                        row["successful_solves"]
                    )
                cells.append({
                    "mapping": arm["mapping"],
                    "measured_qubits": arm["measured_qubits"],
                    "block_size": rung["block_size"],
                    "estimator": estimator,
                    "search_status": search.get("status"),
                    "interval": _classify(bracket),
                    "confirmed_failing_effective_shots_per_setting": search.get(
                        "confirmed_failing_effective_shots_per_setting"
                    ),
                    "confirmed_passing_effective_shots_per_setting": search.get(
                        "confirmed_passing_effective_shots_per_setting"
                    ),
                    "lower_effective_shots_per_setting": bracket.get(
                        "lower_effective_shots_per_setting"
                    ),
                    "upper_effective_shots_per_setting": bracket.get(
                        "upper_effective_shots_per_setting"
                    ),
                    "environment_marginal_widened_sides": bracket.get(
                        "widened_sides", []
                    ),
                    "admissible_cards": sorted(
                        name
                        for name, entry in bracket.get("cards", {}).items()
                        if entry.get("admissible")
                    ),
                })

    finite = [cell for cell in cells if cell["interval"] == FINITE_INTERVAL]
    # Right-censored is "no finite upper bound inside the grid", which is what
    # the rule's "without a confirmed finite interval at 65536" names: a cell
    # that never bracketed, and a cell whose passing endpoint at the ceiling
    # lost its upper bound to R1's marginal flag. A cell open only *below* is
    # censored on the other side and is counted separately.
    censored = [
        cell for cell in cells
        if cell["interval"] in (NO_CONFIRMED_CROSSING, OPEN_ABOVE, OPEN_BOTH)
    ]
    by_status: dict[str, int] = {}
    for cell in censored:
        key = str(cell["search_status"])
        by_status[key] = by_status.get(key, 0) + 1
    priceable = [cell for cell in finite if cell["admissible_cards"]]
    return {
        "status": (
            "priced_second_instance" if priceable else "right_censored_at_frozen_grid"
        ),
        "crossing_policy": rule["crossing_policy"],
        "censoring_policy": rule["censoring_policy"],
        "grid_ceiling_effective_shots_per_setting": ceiling,
        "wider_grid_licensed_by_this_record": False,
        "evaluated_cells": len(cells),
        "cells_with_a_finite_interval": len(finite),
        "cells_with_a_finite_interval_admissible_on_some_card": len(priceable),
        "cells_without_a_finite_interval": len(cells) - len(finite),
        "right_censored_cells": len(censored),
        "cells_open_above_at_the_grid_ceiling": sum(
            cell["interval"] in (OPEN_ABOVE, OPEN_BOTH) for cell in cells
        ),
        "cells_open_below_without_a_confirmed_failure": sum(
            cell["interval"] in (OPEN_BELOW, OPEN_BOTH) for cell in cells
        ),
        "every_cell_priced": len(finite) == len(cells),
        "right_censored_by_search_status": dict(sorted(by_status.items())),
        "right_censored_by_search_status_note": (
            "a tabulation of the drawn cells, not a verdict. R3b's split of "
            "grid fit from confirmation power was labelled "
            "post_hoc_diagnostic_not_preregistered and this record does not "
            "promote it: no gate here reads these counts, and neither the grid "
            "nor the probe is re-sized by them"
        ),
        "confirmatory_solves_attempted": solves["attempted"],
        "confirmatory_solve_failures": solves["failed"],
        "priced_cells_by_card": {
            name: sum(name in cell["admissible_cards"] for cell in finite)
            for name in card_names
        },
        "cells": cells,
    }


def qr3_second_instance(system: dict, cards, config: dict, pricing: dict) -> dict:
    """QR3 at the exact tier, re-derived only if this record prices a second bank.

    The gate is the preregistration's: "QR3 is re-derived only if the record
    supplies a second priced instance." So the comparison is attempted when the
    LiH bank has a finite interval to contribute and is otherwise reported as
    undetermined, with the reason -- an abstention is a result about resolution
    and is recorded as one rather than left out.

    ``protocol_cost.json`` is read, not rewritten. Its own
    ``qr3_accuracy_matched`` abstains on one priced instance and stays true of
    the record that carries it; what is new lives here, where the two prices
    are put side by side under one instrument.
    """
    frozen = _read(PROTOCOL_COST_RECORD)
    first = frozen["systems"][FIRST_PRICED_INSTANCE]
    base = {
        "gate": (
            "QR3 is re-derived only if this record supplies a second priced "
            "instance; the endpoint grid does not move either way"
        ),
        "first_instance": {
            "system": FIRST_PRICED_INSTANCE,
            "record": "benchmarks/reference_results/protocol_cost.json",
            "record_sha256": _file_sha256(PROTOCOL_COST_RECORD),
            "verdict_in_that_record": frozen["qr3_accuracy_matched"]["status"],
            "left_unchanged": True,
        },
        "second_instance": {
            "system": system["system"],
            "status": pricing["status"],
            "cells_with_a_finite_interval": pricing["cells_with_a_finite_interval"],
        },
    }
    if pricing["status"] != "priced_second_instance":
        return {
            **base,
            "status": "not_re_derived",
            "reason": (
                "no cell supplies a confirmed finite interval that is admissible "
                "on a declared card, so this record adds no second price and QR3 "
                "remains undetermined at the exact tier"
            ),
            "priced_instances": [FIRST_PRICED_INSTANCE],
        }
    comparison = _instance_cost_spread(
        {FIRST_PRICED_INSTANCE: first, system["system"]: system},
        cards,
        [FIRST_PRICED_INSTANCE, system["system"]],
    )
    return {
        **base,
        "status": "re_derived",
        "comparison": comparison,
        "instrument": (
            "both instances priced at 30+100 replicas over the same 64..65536 "
            "grid, both estimators, and the same three cards, under one frozen "
            "stack, from disjoint seed roots"
        ),
    }


def build_record(*, workers: int = 1, run: bool = True) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config()
    problems = preregistration_problems(config)
    if problems:
        raise ValueError(
            "R3c will not run against a preregistration that does not validate:\n  "
            + "\n  ".join(problems)
        )

    environment = environment_problems(config)
    if run and environment:
        raise ValueError(
            "R3c freezes the stack its seed roots name a stream under:\n  "
            + "\n  ".join(environment)
            + "\n  Install the pinned versions -- `python "
            "benchmarks/check_record_environment.py --constraints` writes them "
            "-- and re-run; drawing these replicas elsewhere prices a different "
            "experiment under this record's name."
        )

    target = float(config["acceptance_rule"]["accuracy_target_millihartree"])
    if target != ACCURACY_TARGET_MILLIHARTREE:
        raise ValueError(
            f"the preregistration targets {target} mHa and the shot search "
            f"tests against {ACCURACY_TARGET_MILLIHARTREE}; the checker compares "
            "the config against a literal, so only this catches a moved constant"
        )

    cards = load_device_cards()
    if {card.name: card.sha256 for card in cards} != _frozen_cards(config):
        raise ValueError(
            "the installed device cards are not the three the preregistration "
            "pinned by name and SHA-256"
        )

    problems = r3b_untouched_problems() + qr3_reference_problems(config, cards)
    if problems:
        raise ValueError(
            "R3c will not run against moved lineage:\n  " + "\n  ".join(problems)
        )

    selection = derive_margin_bank({
        "candidate": config["candidate"],
        "acceptance_gates": {
            "accuracy_target_millihartree": config["acceptance_rule"][
                "accuracy_target_millihartree"
            ],
        },
    })
    spec = _downstream_spec(config["candidate"], selection)
    mapping_base = load_mapping_config()
    measurement = {
        **mapping_base["measurement"],
        "grouping_protocol_by_system": {spec["key"]: "qwc_groups"},
    }
    mapping = build_mapping_system(
        spec, arms=config["mapping_arms"], cards=cards, measurement=measurement,
    )
    structural = build_protocol_system(
        spec,
        arms=config["mapping_arms"],
        block_sizes=config["protocol"]["block_sizes"],
        cards=cards,
        shots=int(mapping_base["measurement"]["uniform_raw_shots_per_setting"]),
        frozen={},
    )
    problems = bank_problems(config, mapping, spec)
    if problems:
        raise ValueError(
            "R3c will not price a bank that is not the declared one:\n  "
            + "\n  ".join(problems)
        )

    base = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_role": "accuracy_matched_cost_record",
        "evidence_tier": config["reporting_contract"]["evidence_tier"],
        "evidence_tier_basis": config["reporting_contract"]["evidence_tier_basis"],
        "search_uncertainty_evidence": "heuristic",
        "config_sha256": _canonical_sha256(config),
        "config_file_sha256": _file_sha256(CONFIG),
        "preregistration": {
            "config": "benchmarks/configs/r3c_lih_full_cost.json",
            "landed_before_any_sampling": True,
            "checker": "benchmarks/check_r3c_preregistration.py",
            "authorized_by_this_config": config["structural_gates"][
                "full_30_plus_100_run_authorized_by_this_config"
            ],
            # Inherited, where R3b's had to be restated. R3b's config bounded a
            # commit that had sampled nothing -- "no sampling has been
            # performed" -- so its record could not carry that sentence without
            # denying its own forty cells. R3c's was written tenselessly for
            # exactly this: it says what the *config* carries and what a record
            # may report, both of which stay true once the run exists.
            "claim_boundary_inherited_from_config": True,
            "claim_boundary_is_tenseless": True,
        },
        "parent_lineage": config["parent_lineage"],
        "estimand": config["estimand"],
        "claim_boundary": config["claim_boundary"],
        "accuracy_target_millihartree": ACCURACY_TARGET_MILLIHARTREE,
        "primary_metric": (
            "replica RMSE of the nonlinear selected-rank Ritz energy against the "
            "full exact sector ground energy, per (mapping, k) cell"
        ),
        "pass_rule": config["acceptance_rule"]["passes_when"],
        "k_star_rule": (
            "k* is the set of rungs whose cost interval reaches the smallest "
            "upper bound; the interval is (C(confirmed_fail), C(confirmed_pass)], "
            "widened one grid step on any side R1 flagged environment-marginal"
        ),
        "selection": selection,
        # The preflights are structural context, and the asymptotic pricing the
        # mapping layer attaches to them is removed for the same reason the
        # probes removed it: this record's C_time values are exact-tier
        # measurements, and two different cost notions in one file is one too
        # many. The labelled asymptotic bias/variance summary survives.
        "mapping_preflight": _without_cost_derivatives(mapping),
        "structural_protocol_preflight": _without_cost_derivatives(structural),
        "protocol": config["protocol"],
        "acceptance_rule": config["acceptance_rule"],
        "reporting_contract": config["reporting_contract"],
        "structural_gates": config["structural_gates"],
        "execution_environment": {
            **config["protocol"]["execution_environment"],
            "matches_preregistration": not environment,
            "problems": environment,
        },
        "device_cards": [{"sha256": card.sha256, **card.to_dict()} for card in cards],
    }
    if not run:
        return stamp_record({
            **base,
            "full_cost_run": {
                "executed": False,
                "is_a_cost_record": False,
                "reason": "explicitly skipped by caller",
            },
            "pricing": {"status": "run_not_executed"},
            "qr3_second_instance": {
                "status": "not_re_derived",
                "reason": "no cells were drawn",
            },
        })

    exact = build_cost_record(
        exploratory_replicas=int(config["protocol"]["exploratory_replicas"]),
        confirmatory_replicas=int(config["protocol"]["confirmatory_replicas"]),
        workers=workers,
        config=_probe_config(config),
        cards=cards,
        system_specs={spec["key"]: spec},
        frozen_settings={},
        seed_roots=config["protocol"]["seed_roots"],
        # R1's cross-check reprices *BeH2*'s crossings through the jw arm. This
        # record prices a different instance, so there is nothing of R1's to
        # agree or disagree with and the field would be a comparison against an
        # unrelated bank.
        r1_cross_check_enabled=False,
    )
    system = exact["systems"][spec["key"]]
    problems = instrument_problems(system, config)
    if problems:
        raise ValueError(
            "R3c drew cells with an instrument other than the frozen one:\n  "
            + "\n  ".join(problems)
        )
    pricing = pricing_decision(system, config)
    return stamp_record({
        **base,
        "full_cost_run": {
            "executed": True,
            "is_a_cost_record": True,
            "exploratory_replicas": config["protocol"]["exploratory_replicas"],
            "confirmatory_replicas": config["protocol"]["confirmatory_replicas"],
            "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
            "bootstrap_replicates": config["protocol"]["bootstrap_replicates"],
            "seed_roots": config["protocol"]["seed_roots"],
            "sampling_evidence": system,
        },
        "pricing": pricing,
        "qr3_second_instance": qr3_second_instance(system, cards, config, pricing),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-run", action="store_true",
        help="build the structural half only, drawing no replicas; the record "
             "it writes is a preview and is not the committed R3c evidence")
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record(workers=args.workers, run=not args.skip_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pricing = record["pricing"]
    print("pricing:", pricing["status"])
    if pricing["status"] != "run_not_executed":
        print(
            f"  {pricing['cells_with_a_finite_interval']}/"
            f"{pricing['evaluated_cells']} cells with a finite interval, "
            f"{pricing['right_censored_cells']} right-censored"
        )
        print("qr3:    ", record["qr3_second_instance"]["status"])
        comparison = record["qr3_second_instance"].get("comparison")
        if comparison and comparison.get("status") == "compared":
            print("  verdict:", comparison["verdict"])
    print(args.out)


if __name__ == "__main__":
    main()
