"""The Phase 15 preflight declaration, held to its gate.

Nothing here forms H² or a second-moment row on a declared bank. The sector
predicate the ceiling rests on is checked against an independent Pauli product,
and its closure under products on the Hubbard dimer, which is not declared.
Each clause of the gate must reject a deliberate mutation of the config.
"""

from __future__ import annotations

import copy
import json

import pytest

import benchmarks.check_phase15_preregistration as gate
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.pauli_kernel import label_to_code, word_mul_reference
from clifford_qc.subspace.contracts import as_multivector


@pytest.fixture(scope="module")
def config():
    return gate.load_config()


@pytest.fixture(scope="module")
def recomputed(config):
    """Every bank's frozen numbers, rebuilt once for the module."""
    computed: dict = {}
    notes: list[str] = []
    problems = gate.structural_problems(config, notes, computed)
    return problems, computed


def test_the_committed_declaration_passes_every_static_clause(config):
    assert gate.static_problems(config) == []


def test_the_committed_numbers_recompute_and_match_the_ledger(recomputed):
    problems, computed = recomputed
    assert problems == []
    assert sorted(computed) == sorted(gate.ledger_baseline())


def test_only_h2o_can_bind_a_clause(config, recomputed):
    _, computed = recomputed
    decisive = {name for name, row in computed.items()
                if row["word_clause_can_bind"] or row["storage_clause_can_bind"]}
    assert decisive == {"h2o_cas8e6o"} == set(config["decisive_banks"])
    assert gate.reachable_verdicts(computed) == {"FULL", "RESTRICTED"}


def test_a_result_field_is_refused_at_load(tmp_path, config):
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["h4"]["k_word_universe"] = 12345
    path = tmp_path / "config.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="result field"):
        gate.load_config(path)
    broken = copy.deepcopy(config)
    broken["contains_results"] = True
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="contains_results"):
        gate.load_config(path)


# ------------------------------------------------------------------ the rule

def test_the_ladder_is_total_and_the_verdict_reads_eligibility():
    assert gate.status_of((True, True), (True, True)) == "ELIGIBLE"
    assert gate.status_of((False, True), (False, True)) == "WORD_PROHIBITIVE"
    assert gate.status_of((True, False), (True, False)) == "STORAGE_PROHIBITIVE"
    assert gate.status_of((False, False), (False, False)) == "BOTH_PROHIBITIVE"
    assert gate.status_of((False, True), (True, True)) == "CONVENTION_SENSITIVE"
    assert gate.verdict_of(["ELIGIBLE"] * 5) == "FULL"
    assert gate.verdict_of(["ELIGIBLE"] * 4 + ["WORD_PROHIBITIVE"]) == "RESTRICTED"
    assert gate.verdict_of(["ELIGIBLE"] * 4 + ["CONVENTION_SENSITIVE"]) == "RESTRICTED"
    assert gate.verdict_of(["STORAGE_PROHIBITIVE"] * 5) == "NONE"
    assert gate.verdict_of(["ELIGIBLE"] * 4 + ["INVALID"]) == "INVALID"


def test_clauses_compare_integers_at_the_declared_thresholds(config):
    ratio = config["clauses"]["word"]["max_ratio"]
    anchor = config["clauses"]["storage"]["max_coefficient_occurrences"]
    at_edge = gate.clause_passes(config, sh_words=100, combined_words=100 * ratio,
                                 total_coefficients=anchor)
    past_edge = gate.clause_passes(config, sh_words=100, combined_words=100 * ratio + 1,
                                   total_coefficients=anchor + 1)
    assert at_edge == (True, True)
    assert past_edge == (False, False)


# ------------------------------------------------------------------ the sector

def _commutes(n: int, word: int, string: int) -> bool:
    left = word_mul_reference(n, word, string)
    right = word_mul_reference(n, string, word)
    return left == right


@pytest.mark.parametrize("n", [2, 4])
def test_the_sector_predicate_is_commutation_with_both_spin_parities(n):
    """Checked against an independent O(n) product, over every word."""
    up = label_to_code("".join("Z" if j % 2 == 0 else "I" for j in range(n)))
    down = label_to_code("".join("Z" if j % 2 else "I" for j in range(n)))
    inside = 0
    for code in range(4 ** n):
        expected = _commutes(n, code, up) and _commutes(n, code, down)
        assert gate.in_spin_parity_sector(n, code) == expected, code
        inside += expected
    assert inside == gate.sector_ceiling(n)


def test_products_of_spin_conserving_operators_stay_in_the_sector():
    """The ceiling's premise, exercised on the undeclared Hubbard dimer."""
    model = hubbard(2)
    square = as_multivector(model.hamiltonian) * as_multivector(model.hamiltonian)
    assert all(gate.in_spin_parity_sector(model.n, code) for code in square.terms)
    flip = MV(model.n, {label_to_code("XXII"): 0.5, label_to_code("YYII"): 0.5})
    assert not all(gate.in_spin_parity_sector(model.n, code)
                   for code in (flip * square).terms)


def test_the_sector_needs_an_even_register():
    with pytest.raises(ValueError):
        gate.sector_ceiling(3)


# ------------------------------------------------------------------ mutations

def _problems(config, mutate):
    broken = copy.deepcopy(config)
    mutate(broken)
    return gate.static_problems(broken)


def _set(path, value):
    def mutate(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


MUTATIONS = {
    "a word ratio of one": (_set(["clauses", "word", "max_ratio"], 1), "above one"),
    "a softer anchor": (
        _set(["clauses", "storage", "max_coefficient_occurrences"], 200_000_000),
        "is not beh2_stretched_results.json"),
    "a smaller committed anchor": (
        lambda c: (c["clauses"]["storage"].__setitem__("anchor_record", "h2o_results.json"),
                   c["clauses"]["storage"].__setitem__("max_coefficient_occurrences",
                                                       70902895)),
        "largest committed count"),
    "an unknown anchor record": (
        _set(["clauses", "storage", "anchor_record"], "nowhere.json"), "not in the ledger"),
    "a strict tolerance below the declared": (
        _set(["pruning", "strict_tolerance"], 1e-13), "above the declared"),
    "a strict tolerance that prunes H": (
        _set(["pruning", "strict_tolerance"], 1e-3), "would prune a Hamiltonian word"),
    "a declared tolerance not the product's": (
        _set(["pruning", "declared_tolerance"], 1e-10), "multivector product's own"),
    "a wrong decisive set": (
        _set(["decisive_banks"], ["h2o_cas8e6o", "beh2"]), "decisive_banks must be"),
    "an unreachable verdict claimed": (
        _set(["reachable_verdicts"], ["FULL", "RESTRICTED", "NONE"]),
        "reachable_verdicts must be"),
    "a dropped bank": (
        lambda c: c["banks"]["systems"].remove("hubbard_2x2"),
        "exactly the ledger's fixed-label banks"),
    "another basis source": (
        _set(["banks", "source_config"], "benchmarks/configs/other.json"), "must come from"),
    "another baseline": (
        _set(["baseline", "record"], "benchmarks/reference_results/other.json"),
        "committed Phase 2M-A ledger"),
    "a tensed claim boundary": (
        lambda c: c.__setitem__("claim_boundary", c["claim_boundary"] + " It has been run."),
        "tensed"),
    "a permitted follow-up": (
        _set(["prespecified_followup", "permitted"], "one refinement"), "no follow-up"),
    "an advantage claim": (
        _set(["record_requirements", "must_carry", "quantum_advantage_claim"], True),
        "quantum_advantage_claim"),
    "a non-structural label": (_set(["evidence", "label"], "finite_sample"), "structural"),
    "a drifted input": (
        lambda c: c["implementation_lineage"]["inputs"][0].__setitem__("sha256", "0" * 64),
        "drifted"),
    "an unbound product": (
        lambda c: c["implementation_lineage"].__setitem__(
            "implementation", c["implementation_lineage"]["implementation"][1:]),
        "must be bound"),
    "an unbound mapping config": (
        lambda c: c["implementation_lineage"].__setitem__(
            "inputs", c["implementation_lineage"]["inputs"][1:]),
        "must be bound"),
    "an unbound FCIDUMP": (
        lambda c: c["implementation_lineage"].__setitem__(
            "inputs", [row for row in c["implementation_lineage"]["inputs"]
                       if "h2o" not in row["path"]]),
        "h2o_cas8e6o: source is not bound"),
    "a revision with no statement": (
        lambda c: c["revisions"].append({"revision": 1, "changes": ["x"]}), "does not state"),
    "misnumbered revisions": (
        lambda c: c["revisions"][0].__setitem__("revision", 1), "numbered"),
    "a moved combination rule": (
        _set(["decision_rule", "combination_rule"], "all ELIGIBLE gives FULL"),
        "combination_rule"),
    "a moved ladder": (
        lambda c: c["decision_rule"]["bank_status_order"].reverse(), "order moved"),
    "a missing consequence": (
        lambda c: c["consequences"].pop("INVALID"), "declare its consequence"),
    "a wrong schema": (_set(["schema"], "other"), "schema"),
    "a missing frozen bank": (
        lambda c: c["measured_before_freezing"].pop("beh2"), "cover exactly"),
    "a baseline above the anchor": (
        _set(["measured_before_freezing", "h4", "sh_coefficient_occurrences"], 10 ** 9),
        "baseline alone exceeds"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_clause_rejects_its_mutation(config, name):
    mutate, expected = MUTATIONS[name]
    problems = _problems(config, mutate)
    assert any(expected in problem for problem in problems), problems


def test_a_moved_frozen_number_fails_the_recomputation(config, recomputed, monkeypatch):
    _, computed = recomputed
    monkeypatch.setattr(gate, "structural_quantities",
                        lambda config, name: dict(computed[name]))
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["beh2"]["sh_word_universe"] += 1
    problems = gate.structural_problems(broken)
    assert any("beh2: sh_word_universe declared" in p for p in problems), problems
    broken = copy.deepcopy(config)
    broken["measured_before_freezing"]["h4"]["word_clause_can_bind"] = True
    assert any("h4: word_clause_can_bind" in p for p in gate.structural_problems(broken))


def test_a_sector_violation_voids_the_ceiling(config, recomputed, monkeypatch):
    _, computed = recomputed
    leaking = {name: dict(row) for name, row in computed.items()}
    leaking["hubbard_2x2"]["sh_sector_violations"] = 1
    monkeypatch.setattr(gate, "structural_quantities",
                        lambda config, name: leaking[name])
    problems = gate.structural_problems(config)
    assert any("leaves the spin-parity sector" in p for p in problems), problems


# ------------------------------------------------------------------ lineage and order

def test_the_implementation_binding_lapses_once_a_record_exists(config):
    drifted = copy.deepcopy(config)
    drifted["implementation_lineage"]["implementation"][0]["sha256"] = "0" * 64
    assert any("drifted" in p for p in gate.lineage_problems(drifted, record_exists=False))
    assert gate.lineage_problems(drifted, record_exists=True) == []
    drifted["implementation_lineage"]["inputs"][0]["sha256"] = "0" * 64
    assert any("drifted" in p for p in gate.lineage_problems(drifted, record_exists=True))


def _one_commit(monkeypatch, tmp_path, boundary):
    record = tmp_path / "record.json"
    record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gate, "RECORD", record)
    monkeypatch.setattr(gate, "_last_commit", lambda path: "a" * 40)
    monkeypatch.setattr(gate, "_first_commit", lambda path: "a" * 40)
    monkeypatch.setattr(gate, "_shallow_boundary", lambda: boundary)


def test_commit_order_skips_a_commit_on_the_shallow_boundary(tmp_path, monkeypatch):
    _one_commit(monkeypatch, tmp_path, {"a" * 40})
    notes: list[str] = []
    assert gate.commit_order_problems(notes) == []
    assert any("too shallow" in note for note in notes)


def test_commit_order_refuses_a_record_sharing_the_config_commit(tmp_path, monkeypatch):
    _one_commit(monkeypatch, tmp_path, set())
    assert any("share a commit" in p for p in gate.commit_order_problems([]))
