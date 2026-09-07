"""Recompute and gate the G1 structural-preconditioner record.

Two jobs, as the project's other checkers do. The record is rebuilt from
scratch and compared field by field, and the contracts it must satisfy are
re-derived from the record's own contents rather than trusted:

* the chain really ran in the declared order A, B -> D -> C -> E, and each
  stage's arithmetic closes -- ``entered == survived + removed``, and every
  stage entered with what the previous one left;
* G1's first gate holds on every pool and instance, with an *empty* symmetric
  difference in both directions rather than only equal counts;
* the second, independent form of that gate holds too: on the excitation pool,
  filters A-D land on exactly the set ``determinant_excitations(conserve_sz=
  True)`` keeps, which is the pool every mapping and cost record is built on;
* filter E's marginal is reported per pool and never aggregated, and wherever
  it removed anything, every candidate entering it was a distinct Pauli word --
  so no removal can be explained as deduplication;
* D's containment clause has a zero marginal *in the order applied* and a
  nonzero standalone one equal to B's, which is the overlap measured rather
  than asserted;
* the global centralizer is a strict subset of the reference-conditioned
  survivors, which is what makes section 3.5B's choice load-bearing rather
  than stylistic;
* every stabilizer-bilinear square is exactly ``-1``, so the record carries the
  computed form of section 3.5's real-versus-complex argument;
* the degree sweep matches the excitation reach at the matched cap and nowhere
  else, which bounds what the pool correspondence says;
* the QG1 verdict follows mechanically from the measured fields, so no verdict
  can be written into the record by hand;
* no cost-record field appears anywhere: this tier prices nothing.

    python benchmarks/check_g1_structural_preconditioner.py
"""

from __future__ import annotations

import json
import sys

from clifford_qc.reproducibility import compare_json_records

try:  # package import in tests versus direct script execution
    from benchmarks.run_g1_structural_preconditioner import (
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_g1_structural_preconditioner import (
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
    )

FILTER_ORDER = ("A", "B", "D", "C", "E")

# Fields that would make this a cost record. The tier is structural; if one of
# these ever appears the record has quietly changed what it claims. Same list,
# for the same reason, that check_priceability_screen.py carries.
COST_RECORD_KEYS = {
    "C_time",
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
    "settings",
    "shot_to_target",
    "shots",
    "word_universe",
}


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


def _pools(record: dict):
    for system in record.get("systems", []):
        for pool in system.get("pools", []):
            yield system, pool


def chain_problems(record: dict) -> list[str]:
    """The filter ledger's own arithmetic, per pool."""
    problems: list[str] = []
    for system, pool in _pools(record):
        where = f"{system['system']}/{pool['pool']}"
        stages = pool.get("stages", [])
        if [stage["filter"] for stage in stages] != list(FILTER_ORDER):
            problems.append(
                f"{where}: filters ran as "
                f"{[stage['filter'] for stage in stages]}, not the declared "
                f"{list(FILTER_ORDER)}; a marginal is only interpretable at the "
                "position in the chain where its filter ran"
            )
            continue
        entering = pool["pool_size"]
        for stage in stages:
            if stage["entered"] != entering:
                problems.append(
                    f"{where}/{stage['filter']}: entered {stage['entered']} but "
                    f"the previous stage left {entering}"
                )
            if stage["entered"] != stage["survived"] + stage["removed"]:
                problems.append(
                    f"{where}/{stage['filter']}: {stage['entered']} entered, "
                    f"{stage['survived']} survived, {stage['removed']} removed"
                )
            if stage["removed"] != len(stage["removed_labels"]):
                problems.append(
                    f"{where}/{stage['filter']}: removed count disagrees with "
                    "its own label list"
                )
            entering = stage["survived"]
        if entering != pool["surviving"]:
            problems.append(
                f"{where}: the chain left {entering} but the record reports "
                f"{pool['surviving']} surviving"
            )
        if pool["surviving"] != len(pool["surviving_labels"]):
            problems.append(f"{where}: surviving count disagrees with its label list")
    return problems


def gate_problems(record: dict) -> list[str]:
    """G1's two declared gates, re-derived from what the record measured."""
    problems: list[str] = []
    for system, pool in _pools(record):
        where = f"{system['system']}/{pool['pool']}"
        gate = pool.get("gate_abcd_versus_existing_pauli_filters")
        if gate is None:
            problems.append(f"{where}: no A-D agreement gate recorded")
            continue
        if gate["ga_only"] or gate["pauli_only"]:
            problems.append(
                f"{where}: filters A-D disagree with the existing post-encoding "
                f"filters ({len(gate['ga_only'])} GA-only, "
                f"{len(gate['pauli_only'])} Pauli-only). Section 5's G1 gate makes "
                "this a bug in one of the two paths, not a finding"
            )
        if gate["agree"] != (not gate["ga_only"] and not gate["pauli_only"]):
            problems.append(f"{where}: the agree flag contradicts its own difference lists")
        if gate["agree"] and gate["ga_survivors"] != gate["pauli_side_accepts"]:
            problems.append(f"{where}: agreement claimed with unequal survivor counts")

        builder = pool.get("gate_abcd_versus_the_pool_builder")
        if builder is not None and not builder["agree"]:
            problems.append(
                f"{where}: filters A-D do not reproduce the set the shipped pool "
                f"builder keeps ({builder['abcd_survivors']} against "
                f"{builder['builder_keeps']})"
            )

        marginal = pool.get("e_marginal")
        if marginal is None:
            problems.append(f"{where}: filter E's marginal is not reported separately")
            continue
        if marginal["entered"] != marginal["survived"] + marginal["removed"]:
            problems.append(f"{where}: E's marginal does not close")
        if marginal["removed"] and not marginal["removal_is_not_word_deduplication"]:
            problems.append(
                f"{where}: E removed {marginal['removed']} candidates while its "
                f"{marginal['entered']} entrants were not all distinct Pauli "
                "words, so the removals are not separated from deduplication and "
                "the marginal cannot carry section 3.5E"
            )
        if marginal["removal_is_not_word_deduplication"] != (
            marginal["distinct_pauli_words_entering"] == marginal["entered"]
        ):
            problems.append(f"{where}: the deduplication flag contradicts its own counts")

        contrast = pool.get("global_centralizer_contrast")
        if contrast is not None and not contrast["global_is_a_subset"]:
            problems.append(
                f"{where}: the global commutant is not a subset of the "
                "reference-conditioned survivors. Global commutation is "
                "*sufficient* for sector preservation, so this containment is an "
                "invariant and its failure means one of the two tests is wrong"
            )

    # Section 3.5B's distinction has to bite somewhere in the record, and the
    # wide pool is where it can. On the excitation family it does not: those
    # candidates conserve N and S_z globally, so the global test and the
    # reference-conditioned one coincide there. That is a property of the pool,
    # not a defect, and it is also the reason the chain agrees with the pool
    # builder -- so the requirement is "exercised on the Majorana pool", not
    # "exercised on every pool".
    exercised = [
        pool["global_centralizer_contrast"][
            "candidates_a_global_test_would_wrongly_reject"]
        for _, pool in _pools(record)
        if pool["pool"] == "majorana_monomials"
        and "global_centralizer_contrast" in pool
    ]
    if not exercised:
        problems.append(
            "no Majorana-pool global-centralizer contrast recorded, so nothing "
            "measures whether section 3.5B's reference-conditioned form was needed"
        )
    elif not all(count > 0 for count in exercised):
        problems.append(
            "the global centralizer rejects nothing the reference-conditioned "
            "filter keeps, even on the wide Majorana pool. Section 3.5B's choice "
            "is then unexercised, and the record must not present it as "
            "load-bearing"
        )
    return problems


def overlap_problems(record: dict) -> list[str]:
    """D's two clauses, and the overlap with B the ordering creates."""
    problems: list[str] = []
    for system, pool in _pools(record):
        where = f"{system['system']}/{pool['pool']}"
        stages = {stage["filter"]: stage for stage in pool.get("stages", [])}
        d_stage = stages.get("D")
        if d_stage is None:
            continue
        detail = d_stage.get("detail", {})
        if detail.get("leaves_target_sector") != 0:
            problems.append(
                f"{where}: D's containment clause removed "
                f"{detail.get('leaves_target_sector')} in the order applied, but B "
                "precedes it and tests the same character on the same action; a "
                "nonzero value here means the two filters disagree"
            )
        standalone = pool.get("standalone_removals", {})
        if standalone.get("D_leaves_target_sector") != standalone.get("B"):
            problems.append(
                f"{where}: D's standalone containment removal "
                f"{standalone.get('D_leaves_target_sector')} differs from B's "
                f"{standalone.get('B')}; the record claims they test the same "
                "condition, and the standalone pair is what measures that"
            )
        if standalone.get("D") != standalone.get(
                "D_annihilates_reference", 0) + standalone.get("D_leaves_target_sector", 0):
            problems.append(f"{where}: D's standalone clauses do not sum to its total")
    return problems


def witness_problems(record: dict) -> list[str]:
    """Section 3.5's real-versus-complex argument, as computed numbers."""
    problems: list[str] = []
    witnesses = record.get("algebra_witnesses", {}).get("stabilizer_bilinear_squares", {})
    if not witnesses:
        problems.append("no stabilizer-bilinear witness recorded")
    for register, squares in sorted(witnesses.items()):
        for p, value in enumerate(squares):
            if complex(value) != complex(-1.0, 0.0):
                problems.append(
                    f"n={register}, mode {p}: (gamma_2p gamma_2p+1)^2 = {value}, not -1. "
                    "Section 3.5's claim that the real Cl(2n,0) branch is vacuous for "
                    "these stabilizers rests on this square"
                )
    return problems


def correspondence_problems(record: dict) -> list[str]:
    """The matched-cap comparison, and the sweep that bounds it."""
    problems: list[str] = []
    for system in record.get("systems", []):
        correspondence = system.get("pool_correspondence_at_the_matched_cap", {})
        same = correspondence.get("same_set")
        empty = not correspondence.get("ga_only") and not correspondence.get("excitation_only")
        if same != empty:
            problems.append(
                f"{system['system']}: the same_set flag contradicts its own "
                "difference lists"
            )

    sweep = record.get("degree_sweep", {})
    rows = sweep.get("rows", [])
    matched = int(record["candidate_pools"]["majorana_monomials"]["max_degree"])
    for row in rows:
        expected = row["max_degree"] == matched
        if row["matches_the_rank_2_excitation_reach"] != expected:
            problems.append(
                f"degree sweep at max_degree {row['max_degree']}: reach match is "
                f"{row['matches_the_rank_2_excitation_reach']}, expected {expected}. "
                "The pools are compared at the matched cap only; a match above it "
                "would mean the cap is not what makes the comparison valid, and a "
                "mismatch at it would invalidate the comparison itself"
            )
        if row["action_equivalence_classes"] > sweep.get("sector_dimension", 0):
            problems.append(
                f"degree sweep at max_degree {row['max_degree']}: "
                f"{row['action_equivalence_classes']} classes exceed the "
                f"{sweep.get('sector_dimension')}-dimensional sector, which is an "
                "upper bound on distinct actions on one reference"
            )
    degrees = [row["max_degree"] for row in rows]
    classes = [row["action_equivalence_classes"] for row in rows]
    if degrees != sorted(degrees):
        problems.append("degree sweep rows are not in increasing degree order")
    if any(b < a for a, b in zip(classes, classes[1:])):
        problems.append(
            "action-equivalence classes fall as the degree cap rises; a wider pool "
            "cannot reach fewer directions, so this is an error rather than a finding"
        )
    return problems


def probe_problems(record: dict) -> list[str]:
    """Section 3.5B's parameter has to be exercised, not merely declared."""
    problems: list[str] = []
    probe = record.get("character_probe", {})
    if not probe:
        return ["no target-character probe recorded"]
    if probe.get("declared_character") == probe.get("reference_character"):
        problems.append(
            "the character probe declares the reference's own character, so it "
            "exercises nothing"
        )
    if not probe.get("b_stage_responds_to_the_declared_character"):
        problems.append(
            "filter B admits the same population under both characters, so the "
            "target character is not a live parameter and section 3.5B's "
            "requirement is unmet"
        )
    if probe.get("survivors_after_B_declared_character", 0) <= 0:
        problems.append(
            "filter B admits nothing under the declared character, so the probe "
            "cannot show B responding rather than rejecting everything"
        )
    if not probe.get("finding"):
        problems.append("the character probe records no finding")
    return problems


def verdict_problems(record: dict) -> list[str]:
    """QG1's verdict has to follow from the measured fields, not be asserted."""
    problems: list[str] = []
    outcomes = record.get("gate_outcomes", {})
    qg1 = record.get("qg1", {})

    gate_one = all(
        pool["gate_abcd_versus_existing_pauli_filters"]["agree"]
        for _, pool in _pools(record)
    )
    e_on_majorana = all(
        pool["e_marginal"]["removed"] > 0
        for _, pool in _pools(record) if pool["pool"] == "majorana_monomials"
    )
    e_on_excitations = any(
        pool["e_marginal"]["removed"] > 0
        for _, pool in _pools(record) if pool["pool"] == "determinant_excitations"
    )
    if outcomes.get("abcd_reproduce_the_existing_pauli_filters") != gate_one:
        problems.append("the recorded gate-one outcome disagrees with the pool gates")
    if outcomes.get("e_removes_candidates_on_the_majorana_pool") != e_on_majorana:
        problems.append("the recorded Majorana E outcome disagrees with the pool marginals")
    if outcomes.get("e_removes_nothing_on_the_excitation_pool") != (not e_on_excitations):
        problems.append("the recorded excitation E outcome disagrees with the pool marginals")

    fires = gate_one and not e_on_majorana and not e_on_excitations
    if qg1.get("falsifier_fires") != fires:
        problems.append(
            f"QG1 records falsifier_fires={qg1.get('falsifier_fires')}, but the "
            f"measured fields give {fires}. The falsifier is the conjunction "
            "'A-D agree AND E removes nothing', and it is not writable by hand"
        )
    expected = ("falsified_track_G_stops" if fires
                else "not_falsified_but_content_is_pool_dependent")
    if qg1.get("verdict") != expected:
        problems.append(
            f"QG1 verdict is {qg1.get('verdict')!r}, but the measured fields give "
            f"{expected!r}"
        )
    if not qg1.get("what_this_does_not_license"):
        problems.append(
            "QG1 records no boundary on what its verdict licenses; a positive "
            "verdict here is the one most likely to be over-read as a resource claim"
        )
    return problems


def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "structural":
        problems.append(
            f"evidence tier is {record.get('evidence_tier')!r}; this record samples "
            "nothing and must not claim a stronger tier"
        )
    if not record.get("claim_boundary"):
        problems.append("no claim boundary recorded")
    for path in _cost_key_paths(record):
        problems.append(
            f"cost-record field at {path}: this tier prices nothing, so its "
            "presence means the record has changed what it claims"
        )
    problems += chain_problems(record)
    problems += gate_problems(record)
    problems += overlap_problems(record)
    problems += witness_problems(record)
    problems += correspondence_problems(record)
    problems += probe_problems(record)
    problems += verdict_problems(record)
    return problems


def main() -> int:
    load_config()  # the config's own declaration gates
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(expected, actual, atol=1e-12, rtol=1e-12)
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {p}" for p in contract_problems(actual)]
    for problem in problems:
        print(f"  {problem}")
    print("G1 structural preconditioner:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
