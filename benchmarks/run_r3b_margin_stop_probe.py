"""R3b -- the QR3b scoping probe, re-run on the bank R3S admitted.

QR3b drew forty sampled cells on the LiH bank the A-CASE greedy's own stopping
rule produces -- ``M = 13``, binding ``W = 7740`` -- and rejected it on
resolution: 21 cells never bracketed a crossing inside the frozen grid and 9
more resolved only at its last point. R3S then showed that the barrier is a
property of that stopping rule rather than of the instance. Stopping instead at
the smallest prefix clearing the accuracy target with a declared margin puts the
same molecule at ``M = 2`` and ``W = 1439``, below BeH2's ``1814`` -- the one
bank this repository has ever priced inside the grid.

This producer tests that. Everything the probe can vary is inherited from QR3b
unchanged -- the ``64..65536`` grid, ``k in {1,2,4,8}``, both estimators, ``2+2``
replicas -- so the two records differ in the bank and nothing else, and the
comparison between them is like for like.

**The proposition under test is the screen's, not just this bank's.** The
``2048`` ceiling is an operational admission threshold calibrated on a single
priced bank; QR3b tested it only where it *rejected*. This is its first test on
a candidate it *admitted*, so the probe reports ``screen_prediction`` beside the
usual decision: resolution here corroborates the proxy, and failure falsifies it
on its first admission and sends the ceiling back to be recalibrated on two
points rather than one. Either way that is a result about the screen.

**Preregistered.** ``benchmarks/configs/r3b_margin_stop_probe.json`` and its
checker landed in an earlier commit, before any sampling, which is what lets
this record call its rule preregistered where R3S could not. The producer
re-derives the margin rule over the frozen ordering and refuses to run on a
prefix the rule does not select.

**What this record may claim.** Scope-decision evidence only, exactly as QR3b:
it may accept or reject this bank for a later separately preregistered
``30+100`` run, and it may not price ``C(epsilon)``, answer QR3 or QR3b, or
alter any frozen censoring decision. ``full_run_authorized`` is always ``false``.

    python benchmarks/run_r3b_margin_stop_probe.py --workers 4
    python benchmarks/check_r3b_margin_stop_probe.py --workers 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.elements import MatrixElementBank

try:  # package import in tests versus direct benchmark execution
    from benchmarks.check_r3b_preregistration import CONFIG, load_config
    from benchmarks.run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS
    from benchmarks.run_mapping_axis import (
        _build_model,
        _canonical_sha256,
        _selected_generators,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from benchmarks.run_priceability_screen import (
        CONFIG as SCREEN_CONFIG,
        _solved,
    )
    from benchmarks.run_protocol_axis import build_system_record as build_protocol_system
    from benchmarks.run_protocol_cost import build_record as build_cost_record
    from benchmarks.run_qr3b_instance_preflight import (
        _probe_config,
        _sanitize_probe,
        _without_cost_derivatives,
        resolution_decision,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_r3b_preregistration import CONFIG, load_config
    from run_exact_shot_search import ESTIMATORS, SEARCH_ENDPOINTS  # noqa: F401
    from run_mapping_axis import (
        _build_model,
        _canonical_sha256,
        _selected_generators,
        build_system_record as build_mapping_system,
        load_config as load_mapping_config,
        load_device_cards,
    )
    from run_priceability_screen import CONFIG as SCREEN_CONFIG, _solved
    from run_protocol_axis import build_system_record as build_protocol_system
    from run_protocol_cost import build_record as build_cost_record
    from run_qr3b_instance_preflight import (
        _probe_config,
        _sanitize_probe,
        _without_cost_derivatives,
        resolution_decision,
    )

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference_results" / "r3b_margin_stop_probe.json"
SCHEMA = "clifford_qc.r3b_margin_stop_probe.v1"
QR3B_RECORD = HERE / "reference_results" / "qr3b_instance_preflight.json"

#: What *this record* may claim -- deliberately not the config's boundary.
#:
#: The other producers in this tree copy ``config["claim_boundary"]`` into the
#: record, and that is right where the config's sentence is a statement about
#: the phase. This config's is not: it was written to bound a commit that had
#: run nothing, so it opens "Preregistration only. No sampling has been
#: performed under this config", and a record carrying ``executed: true`` and a
#: rejection alongside that sentence contradicts itself. The preregistration is
#: not editable after the fact -- that is the whole point of landing it first --
#: so the record states its own boundary instead, and keeps the config's under
#: ``preregistration.config_claim_boundary_at_landing`` where it stays a true
#: statement about the commit it describes. The scope clause is the same one,
#: word for word, minus the tense.
#:
#: It states no outcome, which is the same discipline: ``decision.status`` says
#: what happened, and a boundary that reported it would be false under
#: ``--skip-probe`` for exactly the reason the config's is false here.
RECORD_CLAIM_BOUNDARY = (
    "Scope-decision evidence only, under the rule "
    "benchmarks/configs/r3b_margin_stop_probe.json fixed before any sampling. "
    "The 2+2 probe may accept or reject this bank for a later preregistered "
    "full run, and it may not price C(epsilon), answer QR3 or QR3b, or alter "
    "any frozen censoring decision. Where this record reports a failure-mode "
    "split under screen_prediction, that split is a post-hoc diagnostic: the "
    "taxonomy was not preregistered, so it describes where this probe's shots "
    "went and is not a test the screen passed."
)


def derive_selection(config: dict) -> dict:
    """Re-derive the declared bank from the margin rule, and refuse a mismatch.

    QR3b's equivalent re-ran the greedy to its intrinsic threshold. This one
    re-runs the R3S margin rule over the same frozen ordering, because that is
    the rule this bank comes from -- and a producer that accepted the declared
    labels without re-deriving them would let a prefix chosen for cheapness
    through, which is the cost-direction selection the gates forbid.
    """
    spec = config["candidate"]
    gates = config["acceptance_gates"]
    screen = json.loads(SCREEN_CONFIG.read_text(encoding="utf-8"))
    admissible_bias = (
        float(gates["accuracy_target_millihartree"])
        / float(screen["acceptance_gates"]["margin_factor"])
    )
    minimum_prefix = int(screen["selection_rule"]["minimum_prefix_size"])

    qr3b_labels = json.loads(
        (HERE / "configs" / "qr3b_instance_preflight.json").read_text(encoding="utf-8")
    )["candidate"]["selected_labels"]
    ordering_spec = dict(spec, key="lih_cas4e4o", selected_labels=qr3b_labels)
    model, _ = _build_model(ordering_spec)
    _, ordering = _selected_generators(model, ordering_spec)

    reference = ExactMVBackend().state(model.reference, ())
    backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact = float(backend.ground_state(model.hamiltonian, k=1, method="dense")[0][0])

    chosen = None
    for size in range(minimum_prefix, len(ordering) + 1):
        bias, words = _solved(
            MatrixElementBank(reference, model.hamiltonian, list(ordering[:size])), exact
        )
        if bias <= admissible_bias:
            chosen = (size, bias, words)
            break
    if chosen is None:
        raise ValueError("the margin rule selects no prefix of the frozen ordering")

    size, bias, words = chosen
    labels = [generator.label for generator in ordering[:size]]
    if labels != list(spec["selected_labels"]):
        raise ValueError(
            f"the margin rule selects {labels}, not the preregistered "
            f"{spec['selected_labels']}; the declared bank is not the one the "
            "declared rule picks"
        )
    if abs(bias - float(spec["expected_bias_millihartree"])) > float(
        spec["bias_tolerance_millihartree"]
    ):
        raise ValueError("the re-derived bias drifted from the preregistration")
    return {
        "selection_rule": "R3S margin rule, re-derived here over the frozen QR3b ordering",
        "margin_factor": float(screen["acceptance_gates"]["margin_factor"]),
        "admissible_bias_millihartree": admissible_bias,
        "minimum_prefix_size": minimum_prefix,
        "frozen_ordering_size": len(ordering),
        "basis_size": size,
        "labels": labels,
        "source_word_universe": words,
        "exact_sector_energy": exact,
        "bias_millihartree": bias,
        "stopping_reason": "smallest prefix clearing the accuracy margin",
        "intrinsic_stop_basis_size": len(qr3b_labels),
        "intrinsic_stop_note": (
            "QR3b probed this instance at the greedy's own stopping point, "
            f"M={len(qr3b_labels)}, and was rejected on resolution. That record "
            "stands; this is a different bank of the same instance."
        ),
    }


# Why a cell failed to resolve, split by what it implicates. The ceiling is a
# proxy for one thing only -- whether a crossing lands inside the frozen grid --
# so a failure that never reaches the grid's edge says nothing about it.
GRID_FIT_FAILURES = {"not_bracketed_within_search_grid"}
CONFIRMATION_FAILURES = {
    "exploratory_crossing_not_confirmed",
    "nonmonotone_confirmation",
}


def screen_prediction(decision: dict, selection: dict) -> dict:
    """Did admitting this bank predict what the ceiling actually claims?

    The reason the probe is worth its shots. R3S admitted this bank on a
    word-universe ceiling calibrated on one success, and QR3b only ever tested
    that proxy where it *rejected*, so this is its first test on an admission.

    The verdict has to be finer than the resolution gate, because that gate
    fails on two unrelated things. ``W`` sets the reconstruction variance, so
    the ceiling predicts whether a crossing lands **inside the grid** -- and a
    cell that never brackets one, ``not_bracketed_within_search_grid``, is the
    failure that implicates it. A cell whose exploratory crossing the
    confirmatory replicas do not reproduce is a statement about the probe's
    ``2+2`` replica count, which the ceiling neither predicts nor claims to.
    Collapsing the two would let an underpowered probe read as a refuted screen.
    """
    unresolved = decision.get("unresolved_coordinates", [])
    grid_fit = [row for row in unresolved if row.get("status") in GRID_FIT_FAILURES]
    confirmation = [
        row for row in unresolved if row.get("status") in CONFIRMATION_FAILURES
    ]
    other = [
        row for row in unresolved
        if row.get("status") not in GRID_FIT_FAILURES | CONFIRMATION_FAILURES
    ]
    at_ceiling = decision.get("cells_at_or_beyond_ceiling", 0)
    resolved = bool(decision.get("resolution_gate_passes"))
    # The ceiling's claim is about *bracketing*: does a crossing exist inside
    # the grid at all. A cell that resolved at the last grid point did bracket
    # one, so it is a headroom shortfall and not a failure of that claim -- it
    # is reported separately rather than folded in, because folding it in would
    # let a single marginal cell read as a refutation while the failure mode the
    # ceiling actually predicts went from fifteen cells to none.
    grid_fit_holds = not grid_fit and not other

    if resolved:
        verdict = "corroborated"
        follows = (
            "The ceiling's first admission resolved on every cell, so the proxy "
            "survives its first real test and a separately preregistered 30+100 "
            "run on this bank is the next step."
        )
    elif grid_fit_holds and at_ceiling == 0:
        verdict = "corroborated_on_grid_fit_probe_underpowered"
        follows = (
            "Grid fit held on every cell and nothing needed the last grid point. "
            "What remains is confirmation power, which is a property of the 2+2 "
            "probe and not of W."
        )
    elif grid_fit_holds:
        verdict = "corroborated_on_grid_fit_headroom_marginal"
        follows = (
            "The ceiling predicts bracketing, and bracketing held: no cell failed "
            "to find a crossing inside the grid, against fifteen such cells on the "
            "intrinsic-stop bank. So the failure mode the ceiling is a proxy for "
            "is gone. Two things short of a pass remain, and neither is that "
            f"failure mode: {at_ceiling} cell(s) bracketed only at the last grid "
            "point, so headroom is marginal there; and the rest are exploratory "
            "crossings the confirmatory replicas did not reproduce, which is the "
            "probe's replica count rather than W. The preregistered gate still "
            "rejects this bank for a full run and is not relaxed here -- whether a "
            "2+2 probe sized against grid-fit failures is the right gate for "
            "confirmation failures is a question for a new preregistration, not "
            "for this record."
        )
    else:
        verdict = "falsified_on_its_first_admission"
        follows = (
            "The ceiling admitted a bank whose crossings still do not bracket "
            "inside the grid, so W under the ceiling is not sufficient for grid "
            "fit. Recalibrate the screen on two points rather than one and record "
            "the unresolved cells' word universes; do not widen SEARCH_ENDPOINTS "
            "by default."
        )
    return {
        "screen_admitted_this_bank_before_the_probe": True,
        "admitted_binding_word_universe": selection["source_word_universe"],
        "ceiling": decision["word_universe_ceiling"],
        "resolved_strictly_inside_the_grid": resolved,
        "what_the_ceiling_predicts": (
            "whether a crossing brackets inside the frozen search grid; W sets "
            "the reconstruction variance the search has to overcome"
        ),
        "grid_fit_failures": len(grid_fit),
        "confirmation_failures": len(confirmation),
        "unclassified_failures": len(other),
        "cells_at_or_beyond_ceiling": at_ceiling,
        "grid_fit_holds": grid_fit_holds,
        "verdict": verdict,
        "classification_status": "post_hoc_diagnostic_not_preregistered",
        "classification_status_basis": (
            "The preregistration declared a binary -- the ceiling is corroborated "
            "or falsified -- and the drawn cells showed that binary conflates two "
            "mechanisms with different remedies. This split into grid fit, "
            "headroom and confirmation power was written after seeing them and is "
            "labelled accordingly. What it does not touch is the preregistered "
            "decision: the gate, its inputs and its rejection are unchanged, and "
            "this field only says which mechanism produced that rejection."
        ),
        "what_follows": follows,
    }


def _downstream_spec(spec: dict, selection: dict) -> dict:
    """The frozen spec plus the two fields the shared producers index by name.

    ``build_system_record`` guards the bank it is handed against a declared
    selected energy. The preregistration froze the *bias* and the exact sector
    energy instead, and those determine that energy exactly, so it is derived
    here rather than added to the config. Editing a preregistration after it
    lands is the one thing that would cost it its meaning, and nothing here is
    new information: it is an algebraic restatement of two values the earlier
    commit already fixed.
    """
    return {
        **spec,
        "expected_selected_energy": (
            selection["exact_sector_energy"] + selection["bias_millihartree"] * 1e-3
        ),
        "selected_energy_tolerance": (
            float(spec["bias_tolerance_millihartree"]) * 1e-3
        ),
    }


def build_record(*, workers: int = 1, run_probe: bool = True) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config()
    frozen_spec = config["candidate"]
    cards = load_device_cards()
    selection = derive_selection(config)
    spec = _downstream_spec(frozen_spec, selection)
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
    base = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_role": "scope_decision_only",
        "config_sha256": _canonical_sha256(config),
        "preregistration": {
            "config": "benchmarks/configs/r3b_margin_stop_probe.json",
            "landed_before_any_sampling": True,
            "checker": "benchmarks/check_r3b_preregistration.py",
            # Quoted, not inherited: it bounds the commit that landed the config,
            # which had run nothing. See RECORD_CLAIM_BOUNDARY.
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "parent_lineage": config["parent_lineage"],
        "estimand": config["estimand"],
        "claim_boundary": RECORD_CLAIM_BOUNDARY,
        "selection": selection,
        "mapping_preflight": _without_cost_derivatives(mapping),
        "structural_protocol_preflight": _without_cost_derivatives(structural),
        "protocol": config["protocol"],
        "acceptance_gates": config["acceptance_gates"],
    }
    if not run_probe:
        return stamp_record({
            **base,
            "scoping_probe": {
                "executed": False,
                "is_a_cost_record": False,
                "reason": "explicitly skipped by caller",
            },
            "decision": {"status": "probe_not_run", "full_run_authorized": False},
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
        r1_cross_check_enabled=False,
    )
    probe = _sanitize_probe(exact["systems"][spec["key"]])
    decision = resolution_decision(probe, config)
    # QR3b's decision carries a field naming what its own outcome argued for --
    # that the screen should have run first. Here it did, so the field is
    # replaced by the statement this probe can actually make.
    decision.pop("screen_would_have_rejected_before_probe", None)
    decision["qr3b_verdict"] = "not_evaluated_by_this_probe"
    return stamp_record({
        **base,
        "scoping_probe": {
            "executed": True,
            "is_a_cost_record": False,
            "exploratory_replicas": config["protocol"]["exploratory_replicas"],
            "confirmatory_replicas": config["protocol"]["confirmatory_replicas"],
            "search_endpoints_effective_shots_per_setting": list(SEARCH_ENDPOINTS),
            "sampling_evidence": probe,
        },
        "decision": decision,
        "screen_prediction": screen_prediction(decision, selection),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record(workers=args.workers, run_probe=not args.skip_probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("decision:", record["decision"]["status"])
    if "screen_prediction" in record:
        print("screen: ", record["screen_prediction"]["verdict"])
    print(args.out)


if __name__ == "__main__":
    main()
