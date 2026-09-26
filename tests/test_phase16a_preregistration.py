"""Contracts for the result-free Phase 16A time-evolved QSCI declaration.

The committed declaration must pass every static clause, and each clause must
reject a deliberate mutation of the config aimed at it: a result smuggled in,
a clause the rest of the declaration cannot satisfy (the Phase 16B v3
failure), an oracle arm allowed to promote, a drifted input. The two 8-qubit
instances are recomputed here in full; H2O's recomputation is the CI gate's.
"""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_phase16a_preregistration as gate


@pytest.fixture(scope="module")
def config() -> dict:
    return gate.load_config()


def _problems(config: dict, mutate) -> list[str]:
    broken = copy.deepcopy(config)
    mutate(broken)
    return gate.static_problems(broken)


def test_the_committed_declaration_passes_every_static_clause(config):
    assert gate.static_problems(config) == []


def test_the_ladder_and_combination_rule_are_total():
    assert {gate.status_of(r, b) for r in (False, True) for b in (False, True)} == set(
        gate.STATUSES)
    assert gate.status_of(False, True) == "UNDETERMINED"
    assert gate.verdict_of(["PASS"] * 3) == "GO"
    assert gate.verdict_of(["FAIL"] * 3) == "NO_GO"
    assert gate.verdict_of(["PASS", "PASS", "UNDETERMINED"]) == "CONDITIONAL"
    assert gate.verdict_of(["PASS", "INVALID", "PASS"]) == "INVALID"


def test_a_result_field_is_refused_at_load(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["h4_r09"]["median_error"] = 1e-3
    path = tmp_path / "config.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="result field"):
        gate.load_config(path)
    broken = copy.deepcopy(config)
    broken["contains_results"] = True
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="contains_results"):
        gate.load_config(path)


MUTATIONS = {
    "budget that does not split over the times":
        (lambda c: c["shot_grid"]["total_shots"].append(65538), "split evenly"),
    "admission read below the top budget":
        (lambda c: c["admission_criterion"].__setitem__("top_budget", 4096),
         "top of the shot grid"),
    "tie tolerance comparable to the target":
        (lambda c: c["decision_rule"].__setitem__("tie_tolerance", 1e-3),
         "tie tolerance"),
    "an oracle arm allowed to promote":
        (lambda c: c["decision_rule"]["diagnostics_cannot_promote"].remove(
            "exact_ground_oracle"), "oracle arms must never promote"),
    "an oracle candidate":
        (lambda c: c["arms"]["te_trotter_pooled"].__setitem__("category", "oracle"),
         "implementable input"),
    "a sampled control":
        (lambda c: c["arms"]["matched_selected_ci"].__setitem__(
            "category", "implementable"), "control must be classical"),
    "two arms on one seed stream":
        (lambda c: c["seeds"]["arm_indices"].__setitem__("te_exact_pooled", 0),
         "share a seed stream"),
    "a sampled arm with no seed stream":
        (lambda c: c["seeds"]["arm_indices"].pop("exact_ground_oracle"),
         "cover exactly the sampled arms"),
    "a required instance in another unit":
        (lambda c: c["instances"]["h4_r09"].__setitem__("energy_unit", "eV"),
         "target's units"),
    "a moved status ladder":
        (lambda c: c["decision_rule"]["instance_status_order"].reverse(),
         "ladder order moved"),
    "a reworded combination rule":
        (lambda c: c["decision_rule"].__setitem__(
            "combination_rule", "any PASS gives GO"), "combination_rule text"),
    "a permitted follow-up":
        (lambda c: c["prespecified_followup"].__setitem__("permitted", "one re-run"),
         "permits no follow-up"),
    "a tensed claim boundary":
        (lambda c: c.__setitem__(
            "claim_boundary", c["claim_boundary"] + " No sampling has been performed."),
         "tensed"),
    "a drifted input":
        (lambda c: c["implementation_lineage"]["files"][0].__setitem__(
            "sha256", "0" * 64), "sha256 drifted"),
    "an unbound provenance file":
        (lambda c: c["implementation_lineage"]["files"].pop(1),
         "not bound in implementation_lineage"),
    "a non-doubling Trotter grid":
        (lambda c: c["trotter"].__setitem__("step_grid", [1, 3, 9]), "1, 2, 4"),
    "an arm with no role in the decision":
        (lambda c: c["arms"].__setitem__("adapt_vqe", {"category": "implementable"}),
         "no declared role"),
    "a required instance also screened out":
        (lambda c: c["screened_and_not_admitted"].__setitem__("beh2_r13264", {}),
         "both required and screened"),
    "seed order that is not the required order":
        (lambda c: c["seeds"]["instance_order"].reverse(), "seed instance order"),
    "an advantage claim":
        (lambda c: c["record_requirements"]["must_carry"].__setitem__(
            "quantum_advantage_claim", True), "quantum_advantage_claim"),
    "a missing required key":
        (lambda c: c.pop("robustness"), "missing required keys"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_clause_rejects_its_mutation(config, name):
    mutate, expected = MUTATIONS[name]
    problems = _problems(config, mutate)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("name", ["h4_r09", "beh2_r13264"])
def test_the_small_instances_recompute_their_frozen_numbers(config, name):
    """Admission and every declared number, from the committed inputs."""
    got = gate.structural_quantities(config, config["instances"][name])
    declared = config["measured_before_freezing"][name]
    for key in gate._INT_FIELDS + gate._LIST_INT_FIELDS:
        assert got[key] == declared[key], key
    for key in gate._FLOAT_FIELDS:
        assert gate._close(got[key], declared[key]), key
    assert got["reference_error"] > config["target"]["value"]
    assert got["support_zero_noise_error"] <= config["target"]["value"]
    assert got["expected_support_at_top_budget"] < got["sector_dimension"]
    assert got["minimum_fidelity_margin"] >= config["robustness"]["min_fidelity_margin"]
