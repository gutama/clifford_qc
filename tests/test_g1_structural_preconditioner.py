"""Contracts for the G1 structural-preconditioner record.

G1's failure mode is a marginal that looks like a finding and is a property of
the pool it was taken on, or a QG1 verdict written rather than derived. These
tests pin the declaration, the two gates, the pool-dependence the record's whole
reading turns on, and the checker's ability to catch each of those going wrong.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.check_g1_structural_preconditioner import (
    chain_problems,
    contract_problems,
    correspondence_problems,
    gate_problems,
    overlap_problems,
    probe_problems,
    verdict_problems,
    witness_problems,
)
from benchmarks.run_g1_structural_preconditioner import (
    CONFIG,
    REFERENCE,
    load_config,
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _pool(record: dict, system: str, pool: str) -> dict:
    entry = next(s for s in record["systems"] if s["system"] == system)
    return next(p for p in entry["pools"] if p["pool"] == pool)


def _write(tmp_path, config: dict):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------

def test_config_declares_every_gate_with_its_statement():
    config = load_config()
    gates = config["gates"]
    assert gates["abcd_reproduce_the_existing_pauli_filters"] is True
    assert gates["e_marginal_reported_separately"] is True
    assert gates["restriction_primitive_is_reused"] is True
    assert gates["no_cost_fields"] is True
    for statement in ("abcd_gate_statement", "e_marginal_gate_statement",
                      "restriction_gate_statement", "no_cost_fields_statement"):
        assert gates[statement]
    assert config["claim_boundary"]
    assert config["target_character"]["parameter_basis"]


@pytest.mark.parametrize("mutation", [
    {"gates": {"abcd_reproduce_the_existing_pauli_filters": False}},
    {"gates": {"e_marginal_reported_separately": False}},
    {"gates": {"restriction_primitive_is_reused": False}},
    {"gates": {"no_cost_fields": False}},
])
def test_a_relaxed_gate_is_refused_at_load(tmp_path, mutation):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in mutation.items():
        config[key].update(value)
    with pytest.raises(ValueError, match="may not relax"):
        load_config(_write(tmp_path, config))


def test_a_pre_filtered_excitation_pool_is_refused_at_load(tmp_path):
    """The error this config exists to prevent, refused at the boundary.

    Building the excitation pool with ``conserve_sz`` already applied would make
    filter B report a zero marginal, and a reader could not distinguish a
    vacuous filter from a pool that had it applied first.
    """
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["candidate_pools"]["determinant_excitations"]["conserve_sz"] = True
    with pytest.raises(ValueError, match="conserve_sz false"):
        load_config(_write(tmp_path, config))


def test_a_degree_sweep_missing_the_matched_cap_is_refused(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["degree_sweep"] = [2, 6, 8]
    with pytest.raises(ValueError, match="the pool's own cap"):
        load_config(_write(tmp_path, config))


def test_a_missing_character_probe_is_refused(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    del config["excited_character_probe"]["sz"]
    with pytest.raises(ValueError, match="parameter from"):
        load_config(_write(tmp_path, config))


# --------------------------------------------------------------------------
# The committed record
# --------------------------------------------------------------------------

def test_committed_record_satisfies_its_contracts(record):
    assert contract_problems(record) == []


def test_the_record_prices_nothing(record):
    assert record["evidence_tier"] == "structural"
    assert "provenance" in record


def test_gate_one_holds_with_an_empty_symmetric_difference(record):
    """Equal counts are not agreement; the difference lists have to be empty."""
    for system in record["systems"]:
        for pool in system["pools"]:
            gate = pool["gate_abcd_versus_existing_pauli_filters"]
            assert gate["agree"] is True
            assert gate["ga_only"] == []
            assert gate["pauli_only"] == []
            assert gate["ga_survivors"] == gate["pauli_side_accepts"]


def test_the_chain_reproduces_the_shipped_pool_builder(record):
    """The second, independent form of gate one, on the pool that matters."""
    for system in record["systems"]:
        pool = next(p for p in system["pools"]
                    if p["pool"] == "determinant_excitations")
        builder = pool["gate_abcd_versus_the_pool_builder"]
        assert builder["agree"] is True
        assert builder["abcd_survivors"] == builder["builder_keeps"]


def test_e_removes_candidates_on_the_wide_pool_and_none_on_the_narrow_one(record):
    """The split that is the whole reading of this record.

    A marginal is a property of its pool. E has content on the Majorana family
    section 3.5 specifies and none on the excitation family the mapping and cost
    records are built on, and conflating the two is the error available here.
    """
    for system in record["systems"]:
        wide = _pool(record, system["system"], "majorana_monomials")["e_marginal"]
        narrow = _pool(record, system["system"],
                       "determinant_excitations")["e_marginal"]
        assert wide["removed"] > 0
        assert narrow["removed"] == 0


def test_e_removals_are_separated_from_word_deduplication(record):
    """Section 3.5E's claim needs the entrants to be distinct words."""
    for system in record["systems"]:
        pool = _pool(record, system["system"], "majorana_monomials")
        marginal = pool["e_marginal"]
        assert marginal["removal_is_not_word_deduplication"] is True
        assert marginal["distinct_pauli_words_entering"] == marginal["entered"]


def test_the_pools_coincide_at_the_matched_cap(record):
    """What decides G2: the admissible pool is the pool R2b already builds."""
    for system in record["systems"]:
        correspondence = system["pool_correspondence_at_the_matched_cap"]
        assert correspondence["same_set"] is True
        assert correspondence["ga_only"] == []
        assert correspondence["excitation_only"] == []


def test_the_sweep_matches_only_at_the_matched_cap(record):
    """Above the cap the Majorana pool reaches more because it is wider."""
    matched = record["candidate_pools"]["majorana_monomials"]["max_degree"]
    rows = record["degree_sweep"]["rows"]
    matching = [r["max_degree"] for r in rows
                if r["matches_the_rank_2_excitation_reach"]]
    assert matching == [matched]
    assert rows[-1]["action_equivalence_classes"] <= record["degree_sweep"][
        "sector_dimension"]


def test_a_non_prefix_reference_is_among_the_declared_instances(record):
    """The case ``occupied_spin_orbitals`` exists for is actually exercised."""
    assert any(not system["reference_is_prefix_filled"]
               for system in record["systems"])


def test_the_reference_determinant_reaches_the_filters(record):
    """Equal counts across instances must not mean the reference was ignored.

    The three instances share a sector, so their marginals agree by symmetry.
    What would be a bug is agreeing on the surviving *labels* across different
    occupied sets, which is what this pins.
    """
    by_occupancy: dict[tuple, set] = {}
    for system in record["systems"]:
        pool = _pool(record, system["system"], "majorana_monomials")
        key = tuple(system["reference_occupied"])
        labels = frozenset(pool["surviving_labels"])
        if key in by_occupancy:
            assert by_occupancy[key] == labels
        by_occupancy[key] = labels
    assert len(by_occupancy) > 1
    assert len(set(by_occupancy.values())) == len(by_occupancy)


def test_the_character_probe_records_the_restriction_interaction(record):
    """Section 3.5B's parameter is live, and C is what closes on it."""
    probe = record["character_probe"]
    assert probe["b_stage_responds_to_the_declared_character"] is True
    assert probe["survivors_after_B_declared_character"] > 0
    assert probe["restriction_arm_annihilates_the_declared_character"] is True
    assert probe["final_survivors_declared_character_without_a_restriction"] > 0
    assert "not independent" in probe["finding"]


def test_the_falsifier_is_evaluated_not_asserted(record):
    qg1 = record["qg1"]
    assert qg1["falsifier_fires"] is False
    assert qg1["verdict"] == "not_falsified_but_content_is_pool_dependent"
    assert qg1["what_this_does_not_license"]


def test_the_claim_boundary_does_not_overstate_the_pre_encoding_claim(record):
    """The one sentence most likely to be over-read on review."""
    boundary = record["claim_boundary"]
    assert "none of them consults the fermion-to-qubit encoding" in boundary
    assert "NOT computed in some encoding-free representation" in boundary
    assert "Jordan-Wigner" in boundary
    assert "smaller Hilbert space" in boundary


# --------------------------------------------------------------------------
# The checker has to catch each of those going wrong
# --------------------------------------------------------------------------

def test_a_hand_written_verdict_is_caught(record):
    tampered = copy.deepcopy(record)
    tampered["qg1"]["falsifier_fires"] = True
    tampered["qg1"]["verdict"] = "falsified_track_G_stops"
    assert any("not writable by hand" in p for p in verdict_problems(tampered))


def test_an_upgraded_e_marginal_is_caught(record):
    """Claiming E content on the excitation pool is the over-reading to block."""
    tampered = copy.deepcopy(record)
    pool = _pool(tampered, "beh2", "determinant_excitations")
    pool["e_marginal"]["removed"] = 5
    pool["e_marginal"]["survived"] -= 5
    assert verdict_problems(tampered)


def test_a_disagreeing_gate_is_caught(record):
    tampered = copy.deepcopy(record)
    _pool(tampered, "beh2", "majorana_monomials")[
        "gate_abcd_versus_existing_pauli_filters"]["ga_only"] = ["G(0,1)"]
    problems = gate_problems(tampered)
    assert any("not a finding" in p for p in problems)


def test_a_deduplication_explained_removal_is_caught(record):
    tampered = copy.deepcopy(record)
    marginal = _pool(tampered, "beh2", "majorana_monomials")["e_marginal"]
    marginal["distinct_pauli_words_entering"] = marginal["entered"] - 1
    marginal["removal_is_not_word_deduplication"] = False
    assert any("not separated from deduplication" in p
               for p in gate_problems(tampered))


def test_a_broken_chain_is_caught(record):
    tampered = copy.deepcopy(record)
    _pool(tampered, "beh2", "majorana_monomials")["stages"][1]["survived"] += 1
    assert chain_problems(tampered)


def test_a_reordered_chain_is_caught(record):
    tampered = copy.deepcopy(record)
    stages = _pool(tampered, "beh2", "majorana_monomials")["stages"]
    stages[2], stages[3] = stages[3], stages[2]
    assert any("not the declared" in p for p in chain_problems(tampered))


def test_a_lost_b_d_overlap_is_caught(record):
    tampered = copy.deepcopy(record)
    pool = _pool(tampered, "beh2", "majorana_monomials")
    pool["standalone_removals"]["D_leaves_target_sector"] = 0
    assert overlap_problems(tampered)


def test_a_wrong_stabilizer_square_is_caught(record):
    tampered = copy.deepcopy(record)
    squares = tampered["algebra_witnesses"]["stabilizer_bilinear_squares"]
    key = sorted(squares)[0]
    squares[key][0] = "(1+0j)"
    assert any("not -1" in p for p in witness_problems(tampered))


def test_a_sweep_matching_above_the_cap_is_caught(record):
    tampered = copy.deepcopy(record)
    for row in tampered["degree_sweep"]["rows"]:
        row["matches_the_rank_2_excitation_reach"] = True
    assert correspondence_problems(tampered)


def test_a_falling_class_count_is_caught(record):
    tampered = copy.deepcopy(record)
    tampered["degree_sweep"]["rows"][-1]["action_equivalence_classes"] = 1
    assert any("cannot reach fewer" in p for p in correspondence_problems(tampered))


def test_a_dead_character_probe_is_caught(record):
    tampered = copy.deepcopy(record)
    tampered["character_probe"]["b_stage_responds_to_the_declared_character"] = False
    assert any("live parameter" in p for p in probe_problems(tampered))


def test_a_leaked_cost_field_is_caught(record):
    tampered = copy.deepcopy(record)
    _pool(tampered, "beh2", "majorana_monomials")["shots"] = 8000
    assert any("prices nothing" in p for p in contract_problems(tampered))
