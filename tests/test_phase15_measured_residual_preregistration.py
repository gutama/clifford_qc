"""The result-free Phase 15 measured-residual preregistration and its gate.

Nothing here forms a second-moment row, a residual functional, a grouping of a
combined universe or a group variance on a declared bank. The declared
estimator is exercised on the undeclared Hubbard dimer: its linearization
against finite differences, the Ritz-vector term it needs, its zero mean and
its invariance under a constant shift of H. Each clause of the gate must
reject a deliberate mutation of the config.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

import benchmarks.check_phase15_measured_residual_preregistration as gate
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.backends import ExactMVBackend
from clifford_qc.measurement.functionals import ritz_functional
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.subspace import MatrixElementBank, SecondMomentBank
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.second_moment import RESOLUTION

ORDER = [0, 1, 2]


@pytest.fixture(scope="module")
def config():
    return gate.load_config()


@pytest.fixture(scope="module")
def recomputed(config):
    computed: dict = {}
    problems = gate.structural_problems(config, computed=computed)
    return problems, computed


def test_the_committed_declaration_passes_every_static_clause(config):
    assert gate.static_problems(config) == []


def test_the_committed_numbers_recompute_and_every_premise_holds(recomputed):
    problems, computed = recomputed
    assert problems == []
    assert set(computed) == {"h4", "h4_converged", "beh2", "h2o_cas8e6o", "hubbard_2x2"}
    for row in computed.values():
        assert row["full_rank"] and row["ground_root_nondegenerate"]
        assert row["energy_settings_match_mapping_axis"]


def test_a_result_field_is_refused_at_load(tmp_path, config):
    bad = copy.deepcopy(config)
    bad["measured_before_freezing"]["beh2"]["cost_ratio"] = 3.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="result field"):
        gate.load_config(path)
    honest = copy.deepcopy(config)
    honest["contains_results"] = True
    path.write_text(json.dumps(honest), encoding="utf-8")
    with pytest.raises(ValueError, match="contains_results"):
        gate.load_config(path)


# ------------------------------------------------------------------ the rule

def test_the_ratio_is_shots_at_one_matched_standard_error():
    """R from its definition equals the shot counts it abbreviates, at any s."""
    settings_u, v_sigma2, settings_sh, v_e, variance = 40, 0.3, 10, 0.02, 0.04
    ratio = gate.cost_ratio(residual_settings=settings_u, residual_variance=v_sigma2,
                            energy_settings=settings_sh, energy_variance=v_e,
                            variance=variance)
    for s in (1e-3, 1e-2):
        sigma_shots = settings_u * (v_sigma2 / (4.0 * variance)) / s**2
        energy_shots = settings_sh * v_e / s**2
        assert ratio == pytest.approx(sigma_shots / energy_shots, rel=1e-12)


@pytest.mark.parametrize("field", ["variance", "energy_settings", "energy_variance"])
def test_the_ratio_refuses_a_degenerate_denominator(field):
    values = dict(residual_settings=4, residual_variance=0.1, energy_settings=2,
                  energy_variance=0.1, variance=0.1)
    values[field] = 0
    with pytest.raises(ValueError):
        gate.cost_ratio(**values)


def test_the_ladder_is_total_and_reads_both_protocols():
    assert gate.status_of(10.0, None, 10.0) == "AFFORDABLE"
    assert gate.status_of(10.000001, None, 10.0) == "PROHIBITIVE"
    assert gate.status_of(3.0, 4.0, 10.0) == "AFFORDABLE"
    assert gate.status_of(30.0, 40.0, 10.0) == "PROHIBITIVE"
    assert gate.status_of(3.0, 40.0, 10.0) == "GROUPING_SENSITIVE"
    assert gate.status_of(30.0, 4.0, 10.0) == "GROUPING_SENSITIVE"
    assert gate.verdict_of(["AFFORDABLE"] * 3) == "FULL"
    assert gate.verdict_of(["PROHIBITIVE", "GROUPING_SENSITIVE"]) == "NONE"
    assert gate.verdict_of(["AFFORDABLE", "GROUPING_SENSITIVE"]) == "RESTRICTED"
    assert gate.verdict_of(["AFFORDABLE", "INVALID"]) == "INVALID"


def test_every_verdict_is_reachable_on_this_population(config):
    alternatives = [config["grouping"]["alternative_protocol_by_system"][name] is not None
                    for name in config["banks"]["systems"]]
    assert alternatives.count(False) == 1  # H2O alone has no alternative
    assert gate.reachable_verdicts(alternatives) == {"FULL", "RESTRICTED", "NONE"}
    assert gate.reachable_verdicts([False]) == {"FULL", "NONE"}


# ------------------------------------------------------------------ the estimator

def _dimer(shift: float = 0.0):
    model = hubbard(2)
    generators = [identity_generator(model.n), *_raw_pool(model)]
    reference = ExactMVBackend().state(model.reference, ())
    hamiltonian = as_multivector(model.hamiltonian) + MV.scalar(model.n, shift)
    bank = MatrixElementBank(reference, hamiltonian, generators)
    second = SecondMomentBank(bank)
    rows = gate.block_rows(bank, second, ORDER)
    means = gate.reference_means(reference, gate.row_words(rows))
    weights, variance = gate.residual_functional(rows, len(ORDER), means)
    residual = second.residual(bank.solve(ORDER), 0)
    return {"bank": bank, "rows": rows, "means": means, "weights": weights,
            "variance": variance, "residual": residual, "reference": reference}


@pytest.fixture(scope="module")
def dimer():
    return _dimer()


def _frozen_check(dimer, config, weights, direction):
    rule = config["linearization"]["finite_difference"]
    return gate.finite_difference_check(
        dimer["rows"], len(ORDER), dimer["means"], weights, direction,
        step=rule["step"], relative_tolerance=rule["relative_tolerance"],
        cancellation_scale=dimer["residual"].cancellation_scale, resolution=RESOLUTION)


def test_the_pipeline_reproduces_the_second_moment_bank(dimer):
    assert dimer["residual"].resolved and dimer["variance"] > 0.1
    assert dimer["variance"] == pytest.approx(dimer["residual"].variance, abs=1e-12)
    assert gate.residual_variance(dimer["rows"], len(ORDER), dimer["means"]) == (
        pytest.approx(dimer["variance"], abs=1e-12))


def test_the_functional_has_zero_reference_mean(dimer):
    mean = sum(value * dimer["means"][word] for word, value in dimer["weights"].items())
    assert abs(mean) <= 1e-12 * (1.0 + dimer["residual"].cancellation_scale)


@pytest.mark.parametrize("shift", [0.0, -75.0])
def test_both_frozen_directions_pass_the_finite_difference_rule(config, shift):
    case = _dimer(shift)
    sh_words = set(case["bank"].word_set(ORDER))
    directions = gate.check_directions(case["weights"], sh_words,
                                       seed=config["linearization"]["finite_difference"]["seed"])
    assert set(directions) == set(config["linearization"]["finite_difference"]["directions"])
    for direction in directions.values():
        assert _frozen_check(case, config, case["weights"], direction)["passes"]


def _weights_without_the_ritz_vector_term(dimer):
    """The linearization a naive estimator would write: c held fixed."""
    S, H, K = gate.projected_matrices(dimer["rows"], len(ORDER), dimer["means"])
    from clifford_qc.subspace.linalg import solve_projected

    result = solve_projected(S, H)
    c, energy = result.coefficients[:, 0], float(result.energies[0])
    variance = float((c.conj() @ K @ c).real) - energy * energy
    out: dict[int, float] = {}
    for (a, b), (s, h, q) in dimer["rows"].items():
        for word in set(s) | set(h) | set(q):
            value = np.conj(c[a]) * c[b] * (
                complex(q.get(word, 0)) - 2 * energy * complex(h.get(word, 0))
                + (energy * energy - variance) * complex(s.get(word, 0)))
            out[word] = out.get(word, 0.0) + (value.real if a == b else 2 * value.real)
    return out


def test_dropping_the_ritz_vector_term_fails_the_rule(config, dimer):
    naive = _weights_without_the_ritz_vector_term(dimer)
    words = set(naive) | set(dimer["weights"])
    gap = math.sqrt(sum((naive.get(w, 0.0) - dimer["weights"].get(w, 0.0)) ** 2
                        for w in words if w))
    norm = math.sqrt(sum(v * v for w, v in dimer["weights"].items() if w))
    assert 0.01 < gap / norm < 0.1  # the 3.7% the declaration quotes
    directions = gate.check_directions(naive, set(dimer["bank"].word_set(ORDER)),
                                       seed=config["linearization"]["finite_difference"]["seed"])
    assert not _frozen_check(dimer, config, naive, directions["sh_gaussian"])["passes"]


def test_a_constant_shift_of_h_changes_no_weight(dimer):
    shifted = _dimer(3.7)
    assert shifted["variance"] == pytest.approx(dimer["variance"], abs=1e-12)
    words = (set(shifted["weights"]) | set(dimer["weights"])) - {0}
    assert max(abs(shifted["weights"].get(w, 0.0) - dimer["weights"].get(w, 0.0))
               for w in words) <= 1e-12


def test_the_linearization_refuses_a_truncated_block(dimer):
    rows = dict(dimer["rows"])
    duplicate = {(0, 0): rows[(0, 0)], (0, 1): rows[(0, 0)], (1, 1): rows[(0, 0)]}
    with pytest.raises(ValueError, match="retained"):
        gate.residual_functional(duplicate, 2, dimer["means"])


def test_grouping_partitions_every_non_identity_word(dimer):
    words = gate.row_words(dimer["rows"])
    for protocol in gate.PROTOCOLS:
        groups = gate.group_words(words, protocol, 4)
        assigned = [word.code for group in groups for word in group]
        assert sorted(assigned) == sorted(words - {0})
        for group in groups:
            for a in group:
                for b in group:
                    assert all(a.letter(q) in ("I", b.letter(q)) or b.letter(q) == "I"
                               for q in range(4))
    with pytest.raises(ValueError, match="unsupported"):
        gate.group_words(words, "fully_commuting", 4)


def test_group_variances_match_their_definition(dimer):
    weights = {w: v for w, v in dimer["weights"].items() if w}
    groups = gate.group_words(set(weights), "qwc_groups", 4)
    variances = gate.group_variances(dimer["reference"], groups, weights)
    assert len(variances) == len(groups) and min(variances) >= -1e-12
    for group, variance in zip(groups, variances):
        operator = MV(4, {word.code: weights[word.code] for word in group})
        mean = 16 * operator.trace_pairing(dimer["reference"])
        second = 16 * (operator * operator).trace_pairing(dimer["reference"])
        assert variance == pytest.approx(max(0.0, (second - mean**2).real), abs=1e-12)


def test_the_energy_side_reuses_the_committed_ritz_functional(dimer):
    """V_E's functional is the existing one, so its committed values transfer."""
    result = dimer["bank"].solve(ORDER)
    functional = ritz_functional(dimer["bank"], result.indices,
                                 result.ritz_vector(), result.ground_energy)
    assert functional.coefficients and 0 not in functional.coefficients


# ------------------------------------------------------------------ mutations

def _set(path, value):
    def mutate(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


MUTATIONS = {
    "ladder_order": (_set(("decision_rule", "bank_status_order"), [
        {"status": "PROHIBITIVE"}, {"status": "AFFORDABLE"},
        {"status": "GROUPING_SENSITIVE"}]), "ladder order"),
    "combination": (_set(("decision_rule", "combination_rule"), "majority"),
                    "combination_rule"),
    "consequence": (lambda c: c["consequences"].pop("NONE"), "consequence"),
    "threshold": (_set(("statistic", "max_ratio"), 1), "max_ratio"),
    "protocol": (_set(("grouping", "declared_protocol_by_system", "h4"),
                      "qwc_basis_cover"), "mapping-axis"),
    "alternative": (_set(("grouping", "alternative_protocol_by_system", "beh2"), None),
                    "alternative"),
    "h2o_alternative": (_set(("grouping", "alternative_protocol_by_system",
                              "h2o_cas8e6o"), "qwc_groups"), "quadratic"),
    "estimator": (_set(("grouping", "estimator"), "pooled"), "single_assignment"),
    "allocation": (_set(("grouping", "allocation"), "neyman"), "uniform"),
    "step": (_set(("linearization", "finite_difference", "step"), 2.0), "step"),
    "tolerance": (_set(("linearization", "finite_difference", "relative_tolerance"),
                       0.05), "tolerance"),
    "directions": (_set(("linearization", "finite_difference", "directions"),
                        ["functional"]), "directions"),
    "reachable": (_set(("reachable_verdicts",), ["FULL", "NONE"]), "reachable"),
    "premise": (_set(("measured_before_freezing", "beh2", "full_rank"), False),
                "whole block"),
    "evidence": (_set(("evidence", "label"), "finite_sample"), "asymptotic"),
    "tensed": (_set(("claim_boundary",), "No ratio has been computed."), "tensed"),
    "advantage": (_set(("record_requirements", "must_carry", "quantum_advantage_claim"),
                       True), "quantum_advantage"),
    "followup": (_set(("prespecified_followup", "permitted"), "one rerun"), "follow-up"),
    "population": (_set(("banks", "systems"), ["h4", "beh2"]), "validated"),
    "lineage": (lambda c: c["implementation_lineage"]["inputs"].pop(0), "must be bound"),
    "drift": (_set(("implementation_lineage", "implementation", 0, "sha256"), "0" * 64),
              "drifted"),
    "revision": (lambda c: c["revisions"].append({"revision": 1, "changes": ["x"]}),
                 "no ratio existed"),
    "schema": (_set(("schema",), "v0"), "schema"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_clause_rejects_its_mutation(config, name):
    mutate, needle = MUTATIONS[name]
    bad = copy.deepcopy(config)
    mutate(bad)
    problems = gate.static_problems(bad)
    assert any(needle in problem for problem in problems), problems


def test_a_moved_frozen_number_fails_the_comparison(config, recomputed):
    _, computed = recomputed
    declared = copy.deepcopy(config["measured_before_freezing"]["h4"])
    assert gate.compare(declared, computed["h4"]) == []
    declared["energy_settings"] += 1
    declared["residual_norm"] *= 1.0 + 1e-6
    declared["mapping_axis_linearized_energy_variance"] = 0.5
    problems = gate.compare(declared, computed["h4"])
    assert len(problems) == 3


def test_the_implementation_binding_lapses_once_a_record_exists(config):
    bad = copy.deepcopy(config)
    bad["implementation_lineage"]["implementation"][0]["sha256"] = "0" * 64
    assert any("drifted" in p for p in gate.lineage_problems(bad, record_exists=False))
    assert not any("drifted" in p for p in gate.lineage_problems(bad, record_exists=True))
    bad["implementation_lineage"]["inputs"][0]["sha256"] = "0" * 64
    assert any("drifted" in p for p in gate.lineage_problems(bad, record_exists=True))


def test_commit_order_waits_for_a_record():
    notes: list[str] = []
    if gate.RECORD.exists():  # pragma: no cover - after the producer lands
        pytest.skip("a record exists; its order is checked by the gate itself")
    assert gate.commit_order_problems(notes) == []
    assert any("no record yet" in note for note in notes)
